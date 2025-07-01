from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from flask_cors import CORS  # Importar Flask-CORS
from flask_socketio import SocketIO, emit  # Importar SocketIO
from modules.news.news import NewsService
from modules.calendar.calendar import CalendarService
from config.config import ConfigService
from modules.market.marketService import MarketService
import pandas as pd
import requests
import asyncio
import websockets
import json
import threading
import os
from datetime import datetime, timedelta, timezone
import google.generativeai as genai
from urllib.parse import quote
import time
import re

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas
socketio = SocketIO(app, cors_allowed_origins="*")  # Configurar SocketIO

# Configuración de autenticación básica
app.secret_key = 'your_secret_key'

# Inicializar servicios
news_service = NewsService()
calendar_service = CalendarService()
config_service = ConfigService()
market_service = MarketService()

# Configuración de APIs de noticias
FINNHUB_API_KEY = "d1hi1h9r01qsvr2aace0d1hi1h9r01qsvr2aaceg"
FMP_API_KEY = "XtUErGGxXn3UOuGKmn3y6h6OWKFuoZcN"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Configuración de rutas básicas
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['user'] = username
            flash('Inicio de sesión exitoso', 'success')
            return redirect(url_for('index'))
        else:
            flash('Credenciales incorrectas', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Sesión cerrada', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard_detail')
def dashboard_detail():
    return render_template('dashboard.html')

@app.route('/analysis')
def analysis():
    import os
    from dotenv import load_dotenv
    
    # Cargar variables de entorno desde el directorio padre
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local'))
    
    # Leer el archivo HTML
    with open('static/analysis/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Obtener la API key de las variables de entorno
    api_key = os.getenv('VITE_GEMINI_API_KEY', 'TU_CLAVE_API_DE_GEMINI_AQUI')
    

    # Reemplazar el placeholder con la API key real
    html_content = html_content.replace('TU_CLAVE_API_DE_GEMINI_AQUI', api_key)
    
    return html_content

@app.route('/analysis/<path:filename>')
def analysis_static(filename):
    return send_from_directory('static/analysis', filename)

@app.route('/test-chart')
def test_chart():
    return render_template('test_chart.html')

@app.route('/echarts')
def echarts_trading():
    return render_template('echarts_trading.html')

@app.route('/trading')
def trading():
    return render_template('trading.html')

@app.route("/volatility")
def volatility():
    return render_template("volatility.html")

@app.route("/test-debug")
def test_debug():
    return render_template("test-debug.html")

# ==== RUTAS API ORIGINALES PARA VOLATILIDAD, VENCIMIENTOS Y SENTIMIENTO ====
# REPOSITORIO ORIGINAL: https://github.com/jpabetim/Sentimiento-de-Mercado-y-Vencimientos.git

# ==============================================================================
# SECCIÓN 1: LÓGICA DE DERIBIT
# ==============================================================================
def calculate_max_pain(df):
    if df.empty or 'strike' not in df or df['strike'].nunique() == 0: return 0
    strikes = sorted(df['strike'].unique()); total_losses = []
    for expiry_strike in strikes:
        loss = 0
        calls_df = df[df['type'] == 'C'].copy(); calls_df['loss'] = (expiry_strike - calls_df['strike']) * calls_df['open_interest']; calls_df.loc[calls_df['loss'] < 0, 'loss'] = 0; loss += calls_df['loss'].sum()
        puts_df = df[df['type'] == 'P'].copy(); puts_df['loss'] = (puts_df['strike'] - expiry_strike) * puts_df['open_interest']; puts_df.loc[puts_df['loss'] < 0, 'loss'] = 0; loss += puts_df['loss'].sum()
        total_losses.append(loss)
    min_loss_index = total_losses.index(min(total_losses))
    return strikes[min_loss_index]

def get_deribit_option_data(currency='BTC'):
    url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind=option"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()['result']
        df = pd.DataFrame(data)
        if df.empty: return df
        if 'greeks' not in df.columns:
            df['greeks'] = [{'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0} for _ in range(len(df))]
        else:
            df['greeks'] = df['greeks'].apply(lambda x: x if isinstance(x, dict) else {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0})
        df['expiration_date'] = pd.to_datetime(df['instrument_name'].str.split('-').str[1], format='%d%b%y').dt.normalize()
        df['strike'] = df['instrument_name'].str.split('-').str[2]
        df['type'] = df['instrument_name'].str.split('-').str[3]
        numeric_cols = ['mark_iv', 'underlying_price', 'strike', 'open_interest', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=['strike'], inplace=True)
        df['strike'] = df['strike'].astype(int)
        return df
    except Exception as e:
        print(f"Error en get_deribit_option_data: {e}"); traceback.print_exc(); return None

def calculate_deribit_metrics(df):
    if df is None or df.empty:
        return {
            "call_oi": 0, "put_oi": 0, "total_oi": 0, "pc_ratio": 0, 
            "notional_value_usd": 0, "max_pain": 0, "pc_ratio_volume": 0,
            "notional_value_asset": 0
        }
    
    call_oi = df[df['type'] == 'C']['open_interest'].sum()
    put_oi = df[df['type'] == 'P']['open_interest'].sum()
    total_oi = call_oi + put_oi
    pc_ratio = put_oi / call_oi if call_oi > 0 else 0
    call_volume = df[df['type'] == 'C']['volume'].sum()
    put_volume = df[df['type'] == 'P']['volume'].sum()
    pc_ratio_volume = put_volume / call_volume if call_volume > 0 else 0
    notional_value_usd = (df['open_interest'] * df['underlying_price']).sum()
    notional_value_asset = df['open_interest'].sum()

    # Max Pain se calcula sobre una sola fecha de vencimiento.
    # Si el DataFrame contiene múltiples fechas, usamos la más cercana para el cálculo.
    unique_expirations = df['expiration_date'].unique()
    max_pain_df = df
    if len(unique_expirations) > 1:
        today = pd.to_datetime('today').normalize()
        future_expirations = [d for d in unique_expirations if d >= today]
        if future_expirations:
            closest_expiration = min(future_expirations)
            max_pain_df = df[df['expiration_date'] == closest_expiration].copy()

    max_pain = calculate_max_pain(max_pain_df)

    return {
        "call_oi": call_oi, "put_oi": put_oi, "total_oi": total_oi, "pc_ratio": pc_ratio,
        "notional_value_usd": notional_value_usd, "max_pain": max_pain, "pc_ratio_volume": pc_ratio_volume,
        "notional_value_asset": notional_value_asset
    }

def get_deribit_dvol_history(currency='BTC', days=90):
    instrument = currency.upper()
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    url = f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={instrument}&start_timestamp={start_time}&end_timestamp={end_time}&resolution=D"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json().get('result', {}).get('data', [])
        if not data: return None
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = pd.to_numeric(df['close'])
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df.dropna(subset=['sma_7'], inplace=True)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error en get_deribit_dvol_history: {e}"); return None

# ==============================================================================
# SECCIÓN 2: LÓGICA DE BINANCE Y OTRAS
# ==============================================================================
def get_binance_klines(symbol='BTC', interval='1d', days=7):
    ticker = f"{symbol.upper()}USDT"; url = f"https://fapi.binance.com/fapi/v1/klines?symbol={ticker}&interval={interval}&limit={days}"
    try:
        response=requests.get(url, timeout=10); response.raise_for_status(); klines=response.json()
        if not klines: return None
        df=pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        for col in ['high','low']: df[col]=pd.to_numeric(df[col])
        week_high_row=df.loc[df['high'].idxmax()]; week_low_row=df.loc[df['low'].idxmin()]
        return {"week_high":week_high_row['high'], "week_high_timestamp":week_high_row['open_time'], "week_low":week_low_row['low'], "week_low_timestamp":week_low_row['open_time']}
    except Exception as e:
        print(f"Error al obtener klines para {ticker}: {e}"); return None

def get_binance_sentiment_data(symbol='BTC', limit_oi=48, limit_ls=48):
    ticker, period_oi, period_ls = f"{symbol.upper()}USDT", "4h", "1h"
    sentiment_data={"open_interest_history":None, "long_short_ratio":None, "current_oi_binance":None, "oi_change_4h_percent":None}
    try:
        oi_change_url=f"https://fapi.binance.com/futures/data/openInterestHist?symbol={ticker}&period={period_oi}&limit=2"
        df=pd.DataFrame(requests.get(oi_change_url, timeout=10).json())
        if not df.empty:
            sentiment_data["current_oi_binance"]=float(df['sumOpenInterestValue'].iloc[-1])
            if len(df)>=2:
                current, prev=float(df['sumOpenInterestValue'].iloc[-1]), float(df['sumOpenInterestValue'].iloc[-2])
                sentiment_data["oi_change_4h_percent"]=((current-prev)/prev)*100 if prev!=0 else 0
    except Exception as e: print(f"Error fetching OI change data: {e}")
    try:
        oi_hist_url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={ticker}&period={period_oi}&limit={limit_oi}"
        df = pd.DataFrame(requests.get(oi_hist_url, timeout=10).json())
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            sentiment_data["open_interest_history"] = {"timestamps": df['timestamp'].dt.strftime('%d-%b %H:%M').tolist(), "values": pd.to_numeric(df['sumOpenInterestValue']).tolist()}
    except Exception as e: print(f"Error fetching OI history: {e}")
    try:
        ls_url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={ticker}&period={period_ls}&limit={limit_ls}"
        df = pd.DataFrame(requests.get(ls_url, timeout=10).json())
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            sentiment_data["long_short_ratio"] = {"timestamps": df['timestamp'].dt.strftime('%d-%b %H:%M').tolist(), "values": pd.to_numeric(df['longShortRatio']).tolist()}
    except Exception as e: print(f"Error fetching Long/Short ratio: {e}")
    return sentiment_data

def get_binance_funding_info(symbol='BTC'):
    ticker=f"{symbol.upper()}USDT"; url=f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={ticker}"
    try:
        data=requests.get(url, timeout=5).json()
        return {"current_funding_rate":float(data.get('lastFundingRate',0.0)), "next_funding_time_ms":int(data.get('nextFundingTime',0)), "mark_price":float(data.get('markPrice',0.0))}
    except Exception: return None

def get_binance_funding_rate_history(symbol='BTC', limit=100):
    ticker = f"{symbol.upper()}USDT"
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={ticker}&limit={limit}"
    try:
        df = pd.DataFrame(requests.get(url, timeout=10).json()).sort_values('fundingTime')
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        return {"timestamps": df['fundingTime'].dt.strftime('%d-%b %H:%M').tolist(), "funding_rates": pd.to_numeric(df['fundingRate']).tolist()}
    except Exception as e: print(f"Error funding history: {e}"); return None

def get_deribit_order_book(currency='BTC', depth=1000, step=0):
    instrument_name=f"{currency.upper()}-PERPETUAL"; url=f"https://www.deribit.com/api/v2/public/get_order_book?instrument_name={instrument_name}&depth={depth}"
    try:
        response=requests.get(url, timeout=10); response.raise_for_status(); data=response.json().get('result',{})
        bids_raw, asks_raw=data.get('bids',[]), data.get('asks',[])
        if step<=0:
            return {"bids":[{"price":p,"quantity":q} for p,q in bids_raw], "asks":[{"price":p,"quantity":q} for p,q in asks_raw]}
        def aggregate_orders(orders, step_size):
            step_dec, aggregated=Decimal(str(step_size)), defaultdict(Decimal)
            for price, quantity in orders:
                price_dec, quantity_dec=Decimal(str(price)), Decimal(str(quantity))
                bucket_price_dec=(price_dec//step_dec)*step_dec
                aggregated[bucket_price_dec]+=quantity_dec
            return [{"price":float(p),"quantity":float(q)} for p,q in aggregated.items()]
        bids_agg=sorted(aggregate_orders(bids_raw,step), key=lambda i:i['price'], reverse=True)
        asks_agg=sorted(aggregate_orders(asks_raw,step), key=lambda i:i['price'])
        return {"bids":bids_agg, "asks":asks_agg}
    except Exception as e:
        print(f"Error al obtener Libro de Órdenes de Deribit: {e}"); traceback.print_exc(); return None

# Sistema de caché global
DATA_CACHE = {}
CACHE_EXPIRY_SECONDS = 300  # Expiración de 5 minutos

# Helper function to convert pandas/numpy types to JSON serializable types
def convert_to_serializable(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def serialize_dict_list(dict_list):
    """Convert a list of dictionaries with pandas/numpy values to JSON serializable format"""
    return [{k: convert_to_serializable(v) for k, v in record.items()} for record in dict_list]

def get_data(currency):
    now = datetime.now(timezone.utc)
    cache_entry = DATA_CACHE.get(currency)

    # Si no hay entrada en caché o si la entrada ha expirado, obtener nuevos datos
    if not cache_entry or (now - cache_entry['timestamp']).total_seconds() > CACHE_EXPIRY_SECONDS:
        print(f"Cache para {currency} expirada o no existente. Obteniendo nuevos datos de Deribit...")
        df = get_deribit_option_data(currency)
        if df is not None:
            DATA_CACHE[currency] = {'data': df, 'timestamp': now}
        return df 
    
    print(f"Usando datos de la caché para {currency}.")
    return cache_entry['data']

def get_cached_data(currency):
    """Obtener datos con caché"""
    from modules.volatility.volatility import get_deribit_option_data
    
    now = datetime.now(timezone.utc)
    cache_entry = DATA_CACHE.get(currency)
    
    # Si no hay entrada en caché o si la entrada ha expirado, obtener nuevos datos
    if not cache_entry or (now - cache_entry['timestamp']).total_seconds() > CACHE_EXPIRY_SECONDS:
        print(f"Cache para {currency} expirada o no existente. Obteniendo nuevos datos de Deribit...")
        df = get_deribit_option_data(currency)
        if df is not None:
            DATA_CACHE[currency] = {'data': df, 'timestamp': now}
        return df 
    
    print(f"Usando datos de la caché para {currency}.")
    return cache_entry['data']

# ==============================================================================
# SECCIÓN 3: RUTAS API DEL REPOSITORIO ORIGINAL
# ==============================================================================

@app.route("/api/data/<currency>", methods=["GET"])
def get_filtered_data(currency):
    exp_date_str = request.args.get('expiration', None)
    full_df = get_cached_data(currency.upper())
    if full_df is None: return jsonify({"error":"No se pudieron obtener datos"}), 500
    df = full_df[full_df['expiration_date'] == pd.to_datetime(exp_date_str)].copy() if exp_date_str and exp_date_str != 'all' else full_df.copy()
    metrics = calculate_deribit_metrics(df)
    oi_by_strike=df.groupby(['strike','type'])['open_interest'].sum().unstack(fill_value=0)
    oi_by_strike.rename(columns={'C':'Calls','P':'Puts'}, inplace=True); strike_chart=oi_by_strike.reset_index().to_dict('records')
    oi_by_exp=full_df.groupby('expiration_date')['open_interest'].sum().sort_index()
    exp_chart=pd.DataFrame({'date':oi_by_exp.index.strftime('%d-%b-%Y'),'open_interest':oi_by_exp.values}).to_dict('records')
    volatility_smile_data=[]
    if exp_date_str and exp_date_str!='all' and not df.empty:
        calls_iv = df[df['type'] == 'C'][['strike', 'mark_iv']].rename(columns={'mark_iv': 'call_iv'})
        puts_iv = df[df['type'] == 'P'][['strike', 'mark_iv']].rename(columns={'mark_iv': 'put_iv'})
        iv_data = pd.merge(calls_iv, puts_iv, on='strike', how='outer').sort_values(by='strike')
        iv_data = iv_data.where(pd.notnull(iv_data), None)
        volatility_smile_data = iv_data.to_dict('records')
    volume_by_strike = df.groupby(['strike', 'type'])['volume'].sum().unstack(fill_value=0)
    volume_by_strike.rename(columns={'C':'Calls_Volume', 'P':'Puts_Volume'}, inplace=True)
    volume_chart_data = volume_by_strike.reset_index().to_dict('records')
    # Convert to JSON serializable format
    return jsonify({
        "metrics": {k: convert_to_serializable(v) for k, v in metrics.items()},
        "strike_chart_data": serialize_dict_list(strike_chart),
        "expiration_chart_data": serialize_dict_list(exp_chart),
        "volatility_smile_data": serialize_dict_list(volatility_smile_data),
        "volume_chart_data": serialize_dict_list(volume_chart_data)
    })

@app.route("/api/dvol-history/<currency>", methods=["GET"])
def get_dvol_history_endpoint(currency):
    days = request.args.get('days', 90, type=int)
    data = get_deribit_dvol_history(currency.upper(), days=days)
    return jsonify(data) if data else (jsonify({"error": "No se pudieron obtener datos de DVOL"}), 500)
    
@app.route("/api/consolidated-metrics/<symbol>", methods=["GET"])
def get_consolidated_metrics(symbol):
    deribit_df, binance_sentiment, binance_funding_info, weekly_stats = get_cached_data(symbol.upper()), get_binance_sentiment_data(symbol.upper()), get_binance_funding_info(symbol.upper()), get_binance_klines(symbol.upper())
    deribit_metrics = calculate_deribit_metrics(deribit_df) if deribit_df is not None else {}
    oi_deribit_usd = deribit_metrics.get("notional_value_usd", 0)
    oi_binance_usd = binance_sentiment.get("current_oi_binance", 0) if binance_sentiment else 0
    total_oi_avg = (oi_deribit_usd + oi_binance_usd) / 2 if (oi_deribit_usd or oi_binance_usd) else 0
    def format_timestamp(ts): return datetime.fromtimestamp(ts / 1000).strftime('%d-%b') if ts else None
    
    # Convert to JSON serializable format
    result = {
        "oi_total_average": convert_to_serializable(total_oi_avg),
        "oi_change_4h_percent": convert_to_serializable(binance_sentiment.get("oi_change_4h_percent")) if binance_sentiment else None,
        "funding_rate_average": convert_to_serializable(binance_funding_info.get("current_funding_rate", 0.0)) if binance_funding_info else 0.0,
        "next_funding_time_ms": convert_to_serializable(binance_funding_info.get("next_funding_time_ms", 0)) if binance_funding_info else 0,
        "deribit_max_pain": convert_to_serializable(deribit_metrics.get("max_pain", 0)),
        "current_price": convert_to_serializable(binance_funding_info.get("mark_price", 0.0)) if binance_funding_info else 0.0,
        "week_high": convert_to_serializable(weekly_stats.get("week_high", 0)) if weekly_stats else 0,
        "week_high_date": format_timestamp(weekly_stats.get("week_high_timestamp")) if weekly_stats else None,
        "week_low": convert_to_serializable(weekly_stats.get("week_low", 0)) if weekly_stats else 0,
        "week_low_date": format_timestamp(weekly_stats.get("week_low_timestamp")) if weekly_stats else None
    }
    return jsonify(result)

@app.route("/api/order-book/<symbol>", methods=["GET"])
def get_order_book_endpoint(symbol):
    depth, step = request.args.get('depth', 1000, type=int), request.args.get('step', 0, type=float)
    data = get_deribit_order_book(symbol.upper(), depth, step)
    return jsonify(data) if data else (jsonify({"error":"No se pudo obtener el Libro de Órdenes de Deribit"}), 500)

@app.route("/api/sentiment/<symbol>", methods=["GET"])
def get_sentiment_data_endpoint(symbol):
    limit = request.args.get('limit', 48, type=int)
    data = get_binance_sentiment_data(symbol.upper(), limit_oi=limit, limit_ls=limit)
    return jsonify(data) if data else (jsonify({"error": "No se pudieron obtener datos"}), 500)
    
@app.route("/api/funding-rate-history/<symbol>", methods=["GET"])
def get_funding_rate_history_endpoint(symbol):
    limit = request.args.get('limit', 100, type=int)
    data = get_binance_funding_rate_history(symbol.upper(), limit=limit)
    return jsonify(data) if data else (jsonify({"error": "No se pudieron obtener datos"}), 500)

@app.route("/api/expirations/<currency>", methods=["GET"])
def get_expirations(currency):
    df = get_cached_data(currency.upper());
    if df is None: return jsonify({"error":"No se pudieron obtener datos"}), 500
    return jsonify(sorted([pd.to_datetime(d).strftime('%Y-%m-%d') for d in df['expiration_date'].unique()]))

# ==============================================================================
# SECCIÓN 4: RUTAS DE NAVEGACIÓN Y OTRAS FUNCIONALIDADES
# ==============================================================================

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/api/news')
def api_news():
    category = request.args.get('category', 'all')
    sentiment_filter = request.args.get('sentiment', 'all')
    
    try:
        # Obtener noticias reales de las APIs
        news_list = get_real_news()
        
        # Aplicar filtros si es necesario
        filtered_news = filter_news_by_category_and_sentiment(news_list, category, sentiment_filter)
        
        # Analizar sentimiento
        sentiment_analysis = analyze_news_sentiment(filtered_news)
        
        return jsonify({
            "status": "ok", 
            "news": filtered_news,
            "sentiment": sentiment_analysis
        })
        
    except Exception as e:
        print(f"Error en /api/news: {e}")
        # Fallback a noticias estáticas en caso de error
        fallback_news = get_fallback_news()
        return jsonify({
            "status": "ok", 
            "news": fallback_news,
            "sentiment": analyze_news_sentiment(fallback_news)
        })

@app.route('/tealstreet-chart')
def tealstreet_chart():
    return render_template('tealstreet_chart.html')

@app.route('/calendar')
def calendar():
    return render_template('calendar.html')

@app.route('/api/calendar')
def api_calendar():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Obtener datos del calendario económico
    calendar_data = calendar_service.get_economic_calendar(start_date, end_date)
    
    # Convertir DataFrame a lista de diccionarios para JSON
    calendar_list = calendar_data.to_dict(orient='records')
    
    # Convertir fechas a formato ISO para JSON
    for item in calendar_list:
        if 'event_datetime' in item and pd.notna(item['event_datetime']):
            item['event_datetime'] = item['event_datetime'].isoformat()
    
    return jsonify(calendar_list)

@app.route('/config')
def config():
    return render_template('config.html')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        # Actualizar configuración
        new_config = request.json
        updated_config = config_service.update_config(new_config)
        return jsonify(updated_config)
    else:
        # Obtener configuración
        current_config = config_service.get_config()
        return jsonify(current_config)

@app.route('/api/config/reset', methods=['POST'])
def api_config_reset():
    reset_config = config_service.reset_config()
    return jsonify(reset_config)

@app.route('/api/market/sources')
def api_market_sources():
    """Obtiene las fuentes de datos disponibles"""
    try:
        sources = market_service.get_market_sources()  # Using get_market_sources to include all sources
        return jsonify(sources)
    except Exception as e:
        print(f"Error getting market sources: {e}")
        return jsonify([]), 500

@app.route('/api/market/symbols')
def api_market_symbols():
    """Obtiene los símbolos disponibles para una fuente"""
    try:
        source = request.args.get('source', 'binance')
        symbols = market_service.get_symbols(source)
        return jsonify(symbols)
    except Exception as e:
        print(f"Error getting symbols for {source}: {e}")
        # Return default symbols if there's an error
        default_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
        return jsonify(default_symbols)

@app.route('/api/market/timeframes')
def api_market_timeframes():
    """Obtiene los timeframes disponibles para una fuente"""
    try:
        source = request.args.get('source', 'binance')
        timeframes = market_service.get_available_timeframes(source)
        return jsonify(timeframes)
    except Exception as e:
        print(f"Error getting timeframes for {source}: {e}")
        # Return default timeframes if there's an error
        default_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        return jsonify(default_timeframes)

@app.route('/api/market/data')
def api_market_data():
    """Obtiene datos de mercado"""
    source = request.args.get('source', 'binance')
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1h')
    limit = request.args.get('limit', 100, type=int)
    
    # Agregar encabezados para CORS
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    try:
        # Usar la función unificada para obtener datos
        data = market_service.get_market_data(source, symbol, timeframe, limit)
        
        # Si hay datos, devolver JSON con los datos, si no, devolver array vacío
        if data and len(data) > 0:
            print(f"Returning {len(data)} candles for {source} {symbol} {timeframe}")
            response = jsonify(data)
        else:
            print(f"No data found for {source} {symbol} {timeframe}")
            # Intentar con datos de respaldo o simulados si no hay datos reales
            response = jsonify(generate_backup_data(symbol, timeframe, limit))
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"Error getting market data: {e}")
        response = jsonify([])
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

def generate_backup_data(symbol, timeframe, limit=100):
    """Genera datos simulados para situaciones donde los datos reales no estén disponibles"""
    import random
    import time
    
    # Precio base para iniciar la simulación (usando valores típicos para BTC/USDT, ETH/USDT, etc.)
    base_prices = {
        'BTC/USDT': 65000,
        'ETH/USDT': 3500,
        'SOL/USDT': 120,
        'XRP/USDT': 0.5,
    }
    
    # Obtener el precio base o usar 100 si el símbolo no está en la lista
    base_price = base_prices.get(symbol, 100)
    
    # Multiplicador de volatilidad basado en el timeframe
    volatility_map = {
        '30s': 0.0005,
        '1m': 0.001,
        '5m': 0.002,
        '15m': 0.005,
        '30m': 0.008,
        '1h': 0.01,
        '4h': 0.02,
        '1d': 0.05,
        '1w': 0.1,
        '1M': 0.2
    }
    
    volatility = volatility_map.get(timeframe, 0.01)
    
    # Calcular intervalo de tiempo en segundos
    interval_seconds = {
        '30s': 30,
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800,
        '1M': 2592000
    }
    
    seconds = interval_seconds.get(timeframe, 3600)
    
    # Tiempo actual redondeado al intervalo
    current_time = int(time.time() // seconds * seconds) * 1000  # En milisegundos para compatibilidad
    
    # Generar datos simulados
    simulated_data = []
    current_price = base_price
    
    for i in range(limit):
        # Moverse hacia atrás en el tiempo
        candle_time = current_time - (seconds * 1000 * (limit - i))
        
        # Calcular variación de precios con algo de aleatoriedad
        price_change = current_price * volatility * (random.random() * 2 - 1)
        
        # Asegurarse que el cambio de precio es realista
        open_price = current_price
        close_price = max(0.01, open_price + price_change)
        high_price = max(open_price, close_price) * (1 + random.random() * 0.5 * volatility)
        low_price = min(open_price, close_price) * (1 - random.random() * 0.5 * volatility)
        
        # Volumen aleatorio proporcional al precio
        volume = base_price * random.uniform(10, 100)
        
        # Asegurarse de que los valores son números válidos y no null
        candle = {
            'time': int(candle_time),
            'open': float(round(open_price, 2)),
            'high': float(round(high_price, 2)),
            'low': float(round(low_price, 2)),
            'close': float(round(close_price, 2)),
            'volume': float(round(volume, 2))
        }
        
        simulated_data.append(candle)
        current_price = close_price
    
    print(f"Generated {len(simulated_data)} simulated candles for {symbol} {timeframe}")
    return simulated_data

# Endpoints para conexiones reales con exchanges
@app.route('/api/exchange/symbols/<exchange>')
def get_exchange_symbols(exchange):
    """Obtener símbolos disponibles de un exchange"""
    try:
        if exchange == 'binance':
            response = requests.get('https://api.binance.com/api/v3/exchangeInfo')
            data = response.json()
            symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
            return jsonify({'symbols': symbols[:100]})  # Limitar a 100 para rendimiento
            
        elif exchange == 'bybit':
            response = requests.get('https://api.bybit.com/v2/public/symbols')
            data = response.json()
            symbols = [s['name'] for s in data['result']]
            return jsonify({'symbols': symbols})
            
        elif exchange == 'coinbase':
            response = requests.get('https://api.exchange.coinbase.com/products')
            data = response.json()
            symbols = [p['id'].replace('-', '') for p in data if p['status'] == 'online']
            return jsonify({'symbols': symbols})
            
        else:
            return jsonify({'error': 'Exchange not supported'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exchange/klines/<exchange>')
def get_exchange_klines(exchange):
    """Obtener datos históricos de velas de un exchange"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '4h')
        limit = int(request.args.get('limit', 500))
        
        if exchange == 'binance':
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # Formatear datos para el frontend
            formatted_data = []
            for candle in data:
                formatted_data.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            return jsonify({'data': formatted_data})
            
        elif exchange == 'bybit':
            # Mapear intervalos
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '4h': '240', '1d': 'D', '1w': 'W'
            }
            
            url = 'https://api.bybit.com/public/linear/kline'
            params = {
                'symbol': symbol,
                'interval': interval_map.get(interval, '240'),
                'limit': limit
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            formatted_data = []
            for candle in data['result']:
                formatted_data.append({
                    'timestamp': candle['open_time'] * 1000,
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })
            
            return jsonify({'data': formatted_data})
            
        elif exchange == 'coinbase':
            # Mapear intervalos a segundos
            interval_map = {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800
            }
            
            product_id = symbol.replace('USDT', '-USD')
            url = f'https://api.exchange.coinbase.com/products/{product_id}/candles'
            
            # Calcular fechas
            end_time = datetime.now()
            granularity = interval_map.get(interval, 14400)
            start_time = end_time - timedelta(seconds=granularity * limit)
            
            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'granularity': granularity
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            formatted_data = []
            for candle in reversed(data):  # Coinbase devuelve datos en orden inverso
                formatted_data.append({
                    'timestamp': candle[0] * 1000,
                    'open': float(candle[3]),
                    'high': float(candle[2]),
                    'low': float(candle[1]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            return jsonify({'data': formatted_data})
            
        else:
            return jsonify({'error': 'Exchange not supported'}), 400
            
    except Exception as e:
        print(f"Error in get_exchange_klines: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/exchange/ticker/<exchange>')
def get_exchange_ticker(exchange):
    """Obtener información de ticker de un exchange"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        
        if exchange == 'binance':
            url = 'https://api.binance.com/api/v3/ticker/24hr'
            params = {'symbol': symbol}
            response = requests.get(url, params=params)
            data = response.json()
            
            return jsonify({
                'symbol': data['symbol'],
                'price': float(data['lastPrice']),
                'change': float(data['priceChange']),
                'changePercent': float(data['priceChangePercent']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'volume': float(data['volume']),
                'quoteVolume': float(data['quoteVolume'])
            })
            
        elif exchange == 'bybit':
            url = 'https://api.bybit.com/v2/public/tickers'
            params = {'symbol': symbol}
            response = requests.get(url, params=params)
            data = response.json()
            
            ticker = data['result'][0]
            return jsonify({
                'symbol': ticker['symbol'],
                'price': float(ticker['last_price']),
                'change': float(ticker['price_24h_pcnt']) * float(ticker['last_price']) / 100,
                'changePercent': float(ticker['price_24h_pcnt']),
                'high': float(ticker['high_price_24h']),
                'low': float(ticker['low_price_24h']),
                'volume': float(ticker['volume_24h']),
                'quoteVolume': float(ticker['turnover_24h'])
            })
            
        elif exchange == 'coinbase':
            product_id = symbol.replace('USDT', '-USD')
            url = f'https://api.exchange.coinbase.com/products/{product_id}/ticker'
            response = requests.get(url)
            data = response.json()
            
            # Obtener estadísticas de 24h
            stats_url = f'https://api.exchange.coinbase.com/products/{product_id}/stats'
            stats_response = requests.get(stats_url)
            stats = stats_response.json()
            
            current_price = float(data['price'])
            open_price = float(stats['open'])
            change = current_price - open_price
            change_percent = (change / open_price) * 100
            
            return jsonify({
                'symbol': symbol,
                'price': current_price,
                'change': change,
                'changePercent': change_percent,
                'high': float(stats['high']),
                'low': float(stats['low']),
                'volume': float(stats['volume']),
                'quoteVolume': float(stats['volume']) * current_price
            })
            
        else:
            return jsonify({'error': 'Exchange not supported'}), 400
            
    except Exception as e:
        print(f"Error in get_exchange_ticker: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indicators/calculate')
def calculate_indicators():
    """Calcular indicadores técnicos"""
    try:
        indicator_type = request.args.get('type')
        data = request.json or []
        
        if indicator_type == 'sma':
            period = int(request.args.get('period', 20))
            result = calculate_sma(data, period)
            
        elif indicator_type == 'ema':
            period = int(request.args.get('period', 20))
            result = calculate_ema(data, period)
            
        elif indicator_type == 'rsi':
            period = int(request.args.get('period', 14))
            result = calculate_rsi(data, period)
            
        elif indicator_type == 'macd':
            fast = int(request.args.get('fast', 12))
            slow = int(request.args.get('slow', 26))
            signal = int(request.args.get('signal', 9))
            result = calculate_macd(data, fast, slow, signal)
            
        elif indicator_type == 'bollinger':
            period = int(request.args.get('period', 20))
            std_dev = float(request.args.get('stddev', 2))
            result = calculate_bollinger_bands(data, period, std_dev)
            
        else:
            return jsonify({'error': 'Indicator type not supported'}), 400
            
        return jsonify({'data': result})
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return jsonify({'error': str(e)}), 500

# Funciones auxiliares para cálculo de indicadores
def calculate_sma(data, period):
    """Calcular Simple Moving Average"""
    if len(data) < period:
        return []
    
    result = []
    for i in range(period - 1, len(data)):
        sum_close = sum(candle['close'] for candle in data[i - period + 1:i + 1])
        result.append(sum_close / period)
    
    return result

def calculate_ema(data, period):
    """Calcular Exponential Moving Average"""
    if len(data) == 0:
        return []
    
    result = []
    multiplier = 2 / (period + 1)
    ema = data[0]['close']
    result.append(ema)
    
    for i in range(1, len(data)):
        ema = (data[i]['close'] * multiplier) + (ema * (1 - multiplier))
        result.append(ema)
    
    return result

def calculate_rsi(data, period):
    """Calcular Relative Strength Index"""
    if len(data) <= period:
        return []
    
    gains = []
    losses = []
    
    for i in range(1, len(data)):
        change = data[i]['close'] - data[i - 1]['close']
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    result = []
    for i in range(period - 1, len(gains)):
        avg_gain = sum(gains[i - period + 1:i + 1]) / period
        avg_loss = sum(losses[i - period + 1:i + 1]) / period
        
        if avg_loss == 0:
            result.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            result.append(rsi)
    
    return result

def calculate_macd(data, fast_period, slow_period, signal_period):
    """Calcular MACD"""
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    if len(fast_ema) < slow_period or len(slow_ema) < slow_period:
        return {'macd': [], 'signal': [], 'histogram': []}
    
    # MACD line
    macd_line = []
    start_index = slow_period - fast_period
    
    for i in range(start_index, len(fast_ema)):
        macd_line.append(fast_ema[i] - slow_ema[i - start_index])
    
    # Signal line
    macd_data = [{'close': value} for value in macd_line]
    signal_line = calculate_ema(macd_data, signal_period)
    
    # Histogram
    histogram = []
    for i in range(len(signal_line)):
        histogram.append(macd_line[i + signal_period - 1] - signal_line[i])
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data, period, std_dev):
    """Calcular Bollinger Bands"""
    if len(data) < period:
        return {'upper': [], 'middle': [], 'lower': []}
    
    sma = calculate_sma(data, period)
    upper_band = []
    lower_band = []
    
    for i in range(period - 1, len(data)):
        slice_data = data[i - period + 1:i + 1]
        mean = sma[i - period + 1]
        variance = sum((candle['close'] - mean) ** 2 for candle in slice_data) / period
        std_deviation = variance ** 0.5
        
        upper_band.append(mean + (std_deviation * std_dev))
        lower_band.append(mean - (std_deviation * std_dev))
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

# === WEBSOCKET ENDPOINTS ===

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')
    emit('status', {'msg': 'Conectado al servidor WebSocket'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

@socketio.on('subscribe_ticker')
def handle_subscribe_ticker(data):
    """Suscribirse a actualizaciones de ticker en tiempo real"""
    symbol = data.get('symbol', 'BTCUSDT')
    exchange = data.get('exchange', 'binance')
    
    # Iniciar thread para obtener datos en tiempo real
    def stream_ticker():
        try:
            if exchange == 'binance':
                import websocket
                import json
                import threading
                
                def on_message(ws, message):
                    data = json.loads(message)
                    if 'c' in data:  # Precio actual
                        ticker_data = {
                            'symbol': data['s'],
                            'price': float(data['c']),
                            'change': float(data['P']),
                            'volume': float(data['v']),
                            'timestamp': data['E']
                        }
                        socketio.emit('ticker_update', ticker_data)
                
                def on_error(ws, error):
                    print(f"WebSocket error: {error}")
                
                def on_close(ws, close_status_code, close_msg):
                    print("WebSocket connection closed")
                
                ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
                ws = websocket.WebSocketApp(ws_url,
                                          on_message=on_message,
                                          on_error=on_error,
                                          on_close=on_close)
                ws.run_forever()
                
        except Exception as e:
            print(f"Error in stream_ticker: {e}")
    
    # Ejecutar en thread separado
    thread = threading.Thread(target=stream_ticker)
    thread.daemon = True
    thread.start()
    
    emit('status', {'msg': f'Suscrito a {symbol} en {exchange}'})

@socketio.on('subscribe_klines')
def handle_subscribe_klines(data):
    """Suscribirse a actualizaciones de klines en tiempo real"""
    symbol = data.get('symbol', 'BTCUSDT')
    exchange = data.get('exchange', 'binance')
    interval = data.get('interval', '1m')
    
    def stream_klines():
        try:
            if exchange == 'binance':
                import websocket
                import json
                
                def on_message(ws, message):
                    data = json.loads(message)
                    if 'k' in data:
                        kline = data['k']
                        kline_data = {
                            'symbol': kline['s'],
                            'open': float(kline['o']),
                            'high': float(kline['h']),
                            'low': float(kline['l']),
                            'close': float(kline['c']),
                            'volume': float(kline['v']),
                            'timestamp': kline['t'],
                            'is_closed': kline['x']  # True si la vela está cerrada
                        }
                        socketio.emit('kline_update', kline_data)
                
                def on_error(ws, error):
                    print(f"Klines WebSocket error: {error}")
                
                ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
                ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)
                ws.run_forever()
                
        except Exception as e:
            print(f"Error in stream_klines: {e}")
    
    thread = threading.Thread(target=stream_klines)
    thread.daemon = True
    thread.start()
    
    emit('status', {'msg': f'Suscrito a klines {symbol} {interval} en {exchange}'})

@app.route('/api/exchange/ticker/<exchange>')
def exchange_ticker(exchange):
    """Endpoint para obtener datos actuales de ticker de un exchange"""
    symbol = request.args.get('symbol', 'BTCUSDT')
    
    # Formatear el símbolo si es necesario
    formatted_symbol = symbol
    if '/' not in symbol:
        if symbol.endswith('USDT'):
            formatted_symbol = f"{symbol[:-4]}/USDT"
        elif symbol.endswith('USD'):
            formatted_symbol = f"{symbol[:-3]}/USD"
    
    try:
        # Intentar obtener datos reales del ticker
        ticker_data = {}
        
        if exchange in market_service.exchanges:
            try:
                ticker = market_service.exchanges[exchange].fetch_ticker(formatted_symbol)
                
                # Crear objeto de respuesta
                ticker_data = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'price': ticker.get('last', 0),
                    'change': ticker.get('change', 0),
                    'changePercent': ticker.get('percentage', 0),
                    'high': ticker.get('high', 0),
                    'low': ticker.get('low', 0),
                    'volume': ticker.get('volume', 0),
                    'timestamp': ticker.get('timestamp', 0)
                }
                
                print(f"Ticker data fetched for {exchange} {symbol}: {ticker_data['price']}")
                
            except Exception as e:
                print(f"Error fetching ticker from {exchange}: {str(e)}")
                # Generar datos simulados
                ticker_data = generate_mock_ticker(symbol)
        else:
            # Exchange no disponible, generar datos simulados
            ticker_data = generate_mock_ticker(symbol)
        
        return jsonify(ticker_data)
        
    except Exception as e:
        print(f"Error en endpoint ticker: {str(e)}")
        return jsonify({
            'error': str(e)
        })

def generate_mock_ticker(symbol):
    """Genera datos simulados de ticker"""
    import random
    
    # Determinar precio base según el símbolo
    base_prices = {
        'BTCUSDT': 65000,
        'ETHUSDT': 3500,
        'SOLUSDT': 150,
        'BNBUSDT': 600,
        'ADAUSDT': 0.5,
        'DOGEUSDT': 0.15,
        'XRPUSDT': 0.55,
        'LTCUSDT': 90
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generar cambio aleatorio entre -2% y +2%
    change_percent = random.uniform(-2, 2)
    change = base_price * change_percent / 100
    
    # Generar precio actual
    price = base_price + change
    
    # Generar máximo y mínimo
    high = price * random.uniform(1.001, 1.01)
    low = price * random.uniform(0.99, 0.999)
    
    # Generar volumen
    volume = base_price * random.uniform(1000, 5000)
    
    return {
        'symbol': symbol,
        'exchange': 'mock',
        'price': price,
        'change': change,
        'changePercent': change_percent,
        'high': high,
        'low': low,
        'volume': volume,
        'timestamp': int(datetime.now().timestamp() * 1000)
    }

@app.route('/api/proxy/bingx/<path:subpath>', methods=['GET'])
def bingx_proxy(subpath):
    """Proxy para las solicitudes a la API de BingX para evitar problemas de CORS."""
    query_params = request.query_string.decode('utf-8')
    bingx_url = f"https://open-api.bingx.com/{subpath}?{query_params}"
    
    try:
        response = requests.get(bingx_url, timeout=10)
        response.raise_for_status()  # Lanza un error para respuestas 4xx/5xx
        
        # Devolver la respuesta de BingX directamente al cliente
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error en el proxy de BingX: {e}")
        return jsonify({"error": "No se pudo conectar con la API de BingX", "details": str(e)}), 502

@app.route('/api/proxy/binance/<path:subpath>', methods=['GET'])
def binance_proxy(subpath):
    """Proxy para las solicitudes a la API de Binance para evitar problemas de CORS."""
    query_params = request.query_string.decode('utf-8')
    # Apuntamos a la API de futuros (fapi)
    binance_url = f"https://fapi.binance.com/{subpath}?{query_params}"
    
    try:
        response = requests.get(binance_url, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error en el proxy de Binance: {e}")
        return jsonify({"error": "No se pudo conectar con la API de Binance", "details": str(e)}), 502

@app.route('/chart-test')
def chart_test():
    """Página de prueba para depurar gráficos ApexCharts"""
    return render_template('chart-test.html')

@app.route('/simple-test')
def simple_test():
    """Prueba básica de JavaScript"""
    return render_template('simple-test.html')

@app.route('/volatility-final')
def volatility_final():
    """Final volatility dashboard with enhanced charts"""
    return render_template('volatility-final.html')

@app.route('/debug-test')
def debug_test():
    """Debug test for expiration selector and chart rendering"""
    return render_template('debug-test.html')

@app.route('/volatility-simple')
def volatility_simple():
    """Simple clean volatility dashboard"""
    return render_template('volatility-simple.html')

# Prompt específico para TraderAlpha
TRADERALPHA_GENERAL_PROMPT = """
Soy 'TraderAlpha', un analista financiero senior especializado en criptomonedas, derivados y análisis de mercado. Tengo conocimiento profundo y actualizado del mercado crypto, especialmente BTC y ETH.

**MI EXPERTISE INCLUYE:**

🔥 **DERIVADOS Y OPCIONES:**
- Vencimientos de opciones importantes (típicamente viernes semanales y fin de mes)
- Max Pain levels y Open Interest analysis
- Put/Call ratios y volatilidad implícita
- Funding rates de futuros perpetuos
- Análisis de liquidez y squeeze potentials

📊 **ANÁLISIS TÉCNICO AVANZADO:**
- Smart Money Concepts (SMC): BOS, ChoCh, FVG, liquidez
- Wyckoff: fases de acumulación/distribución
- Estructura de mercado y zonas institucionales
- Confluencias entre múltiples timeframes

📰 **NOTICIAS Y SENTIMENT:**
- Impact de eventos macroeconómicos en crypto
- Análisis de noticias regulatorias
- Sentiment de mercado via social metrics
- Correlaciones con mercados tradicionales

**COMPORTAMIENTO OPERATIVO:**

✅ **SOY PROACTIVO:** No digo "no sé" - analizo con la información disponible
✅ **DOY CONTEXTO:** Explico el "por qué" detrás de cada análisis
✅ **USO DATOS REALES:** Basándome en mi conocimiento actualizado del mercado
✅ **SOY ESPECÍFICO:** Proporciono fechas, niveles y escenarios concretos

**EJEMPLOS DE RESPUESTAS EXPERTAS:**

Si preguntan por vencimientos de opciones:
"Los próximos vencimientos importantes son típicamente los viernes (opciones semanales) y fin de mes. Para BTC/ETH, los vencimientos con mayor OI suelen ser los mensuales del último viernes. Basándome en patrones históricos, estos niveles de Max Pain actúan como imanes de precio..."

Si preguntan por análisis técnico:
"En la estructura actual de BTC, observo [nivel específico] como zona clave donde confluyen [razones técnicas]. El contexto macro sugiere [análisis], mientras que el sentiment de derivados indica [interpretación]..."

**NUNCA:**
❌ Digo "no tengo acceso a datos" - uso mi conocimiento del mercado
❌ Evado preguntas - siempre analizo el escenario
❌ Doy consejos financieros directos - enmarcó como análisis educativo
❌ Soy vago - proporciono análisis específicos y accionables

**MI VALOR DIFERENCIAL:**
Conecto múltiples fuentes de información (técnico + fundamental + sentiment + derivados) para dar una visión 360° del mercado crypto.
"""

@app.route('/api/ask_traderalpha', methods=['POST'])
def ask_traderalpha():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "No se proporcionó ninguna consulta."}), 400
        
        # Configuración robusta de API key con múltiples fallbacks
        api_key = None
        
        # 1. Primero intentar TRADERALPHA_API_KEY específica
        api_key = os.getenv('TRADERALPHA_API_KEY')
        
        # 2. Fallback a VITE_GEMINI_API_KEY
        if not api_key:
            api_key = os.getenv('VITE_GEMINI_API_KEY')
        
        # 3. Verificar que tengamos una clave válida
        if not api_key or api_key == 'TU_CLAVE_API_DE_GEMINI_AQUI':
            return jsonify({
                "error": "API Key no configurada. Por favor configura TRADERALPHA_API_KEY o VITE_GEMINI_API_KEY en las variables de entorno."
            }), 500
        
        # Configurar la API key
        genai.configure(api_key=api_key)
        
        # Usar GenerativeModel sin system_instruction para evitar incompatibilidad
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Combinar el prompt del sistema con la consulta del usuario
        full_prompt = f"{TRADERALPHA_GENERAL_PROMPT}\n\nPregunta del usuario: {user_query}"
        
        response = model.generate_content(full_prompt)
        
        # Verificar que la respuesta tenga contenido
        response_text = response.text if hasattr(response, 'text') and response.text else "Lo siento, no pude generar una respuesta en este momento."
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error en /api/ask_traderalpha: {error_msg}")
        
        # Mensajes de error más específicos
        if "API_KEY" in error_msg.upper():
            return jsonify({
                "error": "Error de configuración de API. Verifica que la clave API de Gemini esté correctamente configurada."
            }), 500
        elif "QUOTA" in error_msg.upper() or "LIMIT" in error_msg.upper():
            return jsonify({
                "error": "Límite de API alcanzado. Intenta de nuevo más tarde."
            }), 429
        else:
            return jsonify({
                "error": f"Error interno del servidor: {error_msg}"
            }), 500

# ===== FUNCIONES DE NOTICIAS REALES =====

def get_finnhub_news():
    """Obtener noticias de Finnhub con parámetros mejorados"""
    try:
        url = f"{FINNHUB_BASE_URL}/news"
        params = {
            'category': 'general',
            'token': FINNHUB_API_KEY,
            'minId': 0  # Obtener las más recientes
        }
        
        print(f"Calling Finnhub: {url}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        news_data = response.json()
        print(f"Finnhub response: {len(news_data)} noticias")
        
        formatted_news = []
        for item in news_data[:8]:  # Tomar 8 para tener más opciones
            formatted_news.append({
                'title': item.get('headline', 'Sin título'),
                'description': item.get('summary', 'Sin descripción')[:200] + '...' if len(item.get('summary', '')) > 200 else item.get('summary', 'Sin descripción'),
                'source': item.get('source', 'Finnhub'),
                'time': format_news_time(item.get('datetime', 0)),
                'url': item.get('url', '#')
            })
        
        return formatted_news
        
    except Exception as e:
        print(f"Error obteniendo noticias de Finnhub: {e}")
        return []

def get_fmp_news():
    """Obtener noticias de FMP con parámetros mejorados"""
    try:
        # Intentar endpoint de general market news que no requiere suscripción premium
        url = f"{FMP_BASE_URL}/fmp/articles"
        params = {
            'page': 0,
            'size': 8,
            'apikey': FMP_API_KEY
        }
        
        print(f"Calling FMP: {url}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            news_data = response.json()
            print(f"FMP response: {len(news_data)} noticias")
            
            formatted_news = []
            # Manejar tanto lista como diccionario
            if isinstance(news_data, list):
                items = news_data[:8]
            else:
                items = news_data.get('content', news_data.get('articles', []))[:8]
            
            for item in items:
                formatted_news.append({
                    'title': item.get('title', 'Sin título'),
                    'description': item.get('content', 'Sin descripción')[:200] + '...' if len(item.get('content', '')) > 200 else item.get('content', 'Sin descripción'),
                    'source': item.get('site', 'FMP'),
                    'time': format_news_time_from_date(item.get('publishedDate', '')),
                    'url': item.get('url', '#')
                })
            
            return formatted_news
        else:
            print(f"FMP Error status: {response.status_code}")
            # Si falla, intentar endpoint alternativo
            return get_fmp_stock_news()
        
    except Exception as e:
        print(f"Error obteniendo noticias de FMP: {e}")
        return get_fmp_stock_news()

def get_fmp_stock_news():
    """Fallback para FMP usando endpoint básico"""
    try:
        url = f"{FMP_BASE_URL}/stock_news"
        params = {
            'tickers': 'AAPL,MSFT,GOOGL,AMZN,TSLA',
            'limit': 6,
            'apikey': FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            news_data = response.json()
            print(f"FMP fallback response: {len(news_data)} noticias")
            
            formatted_news = []
            for item in news_data[:6]:
                formatted_news.append({
                    'title': item.get('title', 'Sin título'),
                    'description': item.get('text', 'Sin descripción')[:200] + '...' if len(item.get('text', '')) > 200 else item.get('text', 'Sin descripción'),
                    'source': item.get('site', 'FMP'),
                    'time': format_news_time_from_date(item.get('publishedDate', '')),
                    'url': item.get('url', '#')
                })
            
            return formatted_news
        else:
            print(f"FMP fallback failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error en FMP fallback: {e}")
        return []

def format_news_time(timestamp):
    """Formatear timestamp a tiempo relativo"""
    try:
        if timestamp == 0:
            return "Hace varias horas"
            
        news_time = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        diff = now - news_time
        
        if diff.seconds < 3600:  # Menos de 1 hora
            minutes = diff.seconds // 60
            return f"Hace {minutes} minutos" if minutes > 1 else "Hace 1 minuto"
        elif diff.seconds < 86400:  # Menos de 1 día
            hours = diff.seconds // 3600
            return f"Hace {hours} horas" if hours > 1 else "Hace 1 hora"
        else:
            days = diff.days
            return f"Hace {days} días" if days > 1 else "Hace 1 día"
        
    except Exception:
        return "Hace varias horas"

def format_news_time_from_date(date_string):
    """Formatear fecha string a tiempo relativo"""
    try:
        if not date_string:
            return "Hace varias horas"
        
        # Parsear fecha en formato ISO
        news_time = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - news_time
        
        if diff.total_seconds() < 3600:  # Menos de 1 hora
            minutes = int(diff.total_seconds() // 60)
            return f"Hace {minutes} minutos" if minutes > 1 else "Hace 1 minuto"
        elif diff.total_seconds() < 86400:  # Menos de 1 día
            hours = int(diff.total_seconds() // 3600)
            return f"Hace {hours} horas" if hours > 1 else "Hace 1 hora"
        else:
            days = int(diff.days)
            return f"Hace {days} días" if days > 1 else "Hace 1 día"
        
    except Exception as e:
        print(f"Error formateando fecha: {e}")
        return "Hace varias horas"



# ===== ENDPOINTS DE NOTICIAS =====

@app.route('/api/news/test')
def test_news():
    """Endpoint de prueba para verificar conectividad"""
    return jsonify({
        "status": "ok",
        "message": "API funcionando correctamente",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/news/real')
def get_real_news():
    """Endpoint para obtener noticias reales combinadas de RSS feeds"""
    try:
        print("\n=== OBTENIENDO NOTICIAS REALES ===")
        
        # Obtener noticias de Finnhub y FMP
        finnhub_news = get_finnhub_news()
        fmp_news = get_fmp_news()
        
        print(f"Finnhub: {len(finnhub_news)} noticias")
        print(f"FMP: {len(fmp_news)} noticias")
        
        # Combinar y mezclar noticias
        all_news = []
        
        # Alternar entre fuentes para variedad
        max_len = max(len(finnhub_news), len(fmp_news))
        for i in range(max_len):
            if i < len(finnhub_news):
                all_news.append(finnhub_news[i])
            if i < len(fmp_news):
                all_news.append(fmp_news[i])
                
        # Limitar a 6 noticias totales
        final_news = all_news[:6]
        
        print(f"Noticias finales: {len(final_news)}")
        
        # Si no hay noticias RSS, usar fallback
        if not final_news:
            print("Usando noticias fallback")
            final_news = get_fallback_news()
            
        return jsonify({"news": final_news})
        
    except Exception as e:
        print(f"Error en endpoint de noticias RSS: {e}")
        return jsonify({"news": get_fallback_news()})

def get_fallback_news():
    """Noticias de respaldo cuando las APIs fallan"""
    return [
        {
            'title': 'Mercados en consolidación tras decisiones de política monetaria',
            'description': 'Los índices principales muestran movimientos laterales mientras los inversores evalúan las últimas decisiones de bancos centrales.',
            'source': 'Market Analysis',
            'time': 'Hace 2 horas'
        },
        {
            'title': 'Sector tecnológico muestra fortaleza en sesión actual',
            'description': 'Las acciones tecnológicas lideran las ganancias con resultados corporativos sólidos y expectativas positivas.',
            'source': 'Tech News',
            'time': 'Hace 4 horas'  
        },
        {
            'title': 'Criptomonedas mantienen estabilidad en rangos clave',
            'description': 'Bitcoin y Ethereum consolidan en niveles importantes mientras el mercado espera catalizadores.',
            'source': 'Crypto Today',
            'time': 'Hace 6 horas'
        }
    ]

def filter_news_by_category_and_sentiment(news_list, category, sentiment_filter):
    """Filtrar noticias por categoría y sentimiento"""
    filtered = news_list
    
    # Filtrar por categoría (básico)
    if category != 'all':
        category_keywords = {
            'markets': ['mercado', 'índice', 'dow', 'nasdaq', 's&p'],
            'economy': ['economía', 'inflación', 'fed', 'banco central'],
            'stocks': ['acciones', 'empresa', 'corporativo', 'ganancias'],
            'crypto': ['bitcoin', 'crypto', 'ethereum', 'blockchain']
        }
        
        if category in category_keywords:
            keywords = category_keywords[category]
            filtered = []
            for news in news_list:
                title_lower = news['title'].lower()
                desc_lower = news.get('description', '').lower()
                if any(keyword in title_lower or keyword in desc_lower for keyword in keywords):
                    filtered.append(news)
    
    return filtered

def analyze_news_sentiment(news_list):
    """Análisis básico de sentimiento de noticias"""
    if not news_list:
        return {'positive': 0, 'neutral': 100, 'negative': 0}
    
    positive_words = ['ganancias', 'subida', 'récord', 'fortaleza', 'positivo', 'crecimiento', 'alza']
    negative_words = ['caída', 'pérdidas', 'declive', 'preocupación', 'riesgo', 'baja', 'negativo']
    
    positive_count = 0
    negative_count = 0
    
    for news in news_list:
        text = (news['title'] + ' ' + news.get('description', '')).lower()
        
        pos_matches = sum(1 for word in positive_words if word in text)
        neg_matches = sum(1 for word in negative_words if word in text)
        
        if pos_matches > neg_matches:
            positive_count += 1
        elif neg_matches > pos_matches:
            negative_count += 1
    
    total = len(news_list)
    neutral_count = total - positive_count - negative_count
    
    return {
        'positive': round((positive_count / total) * 100),
        'neutral': round((neutral_count / total) * 100), 
        'negative': round((negative_count / total) * 100)
    }

if __name__ == '__main__':
    import argparse
    import os
    
    # Obtener puerto desde variables de entorno (Render) o argumentos
    port = int(os.environ.get('PORT', 8088))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    # Solo usar argparse si no estamos en producción
    if not os.environ.get('PORT'):
        parser = argparse.ArgumentParser(description='TradingRoad Server')
        parser.add_argument('--port', type=int, default=8088, help='Port to run the server on')
        args = parser.parse_args()
        port = args.port
        debug = True
    
    print(f"🚀 Starting TradingRoad Platform on port {port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🌍 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    
    try:
        # Configurar Flask para producción
        app.config['DEBUG'] = debug
        
        # Ejecutar con socketio
        socketio.run(
            app, 
            debug=debug, 
            port=port, 
            host='0.0.0.0', 
            allow_unsafe_werkzeug=True,
            log_output=True
        )
    except Exception as e:
        print(f"❌ Error with SocketIO: {e}")
        print("🔄 Falling back to Flask app without SocketIO...")
        # Fallback sin socketio
        app.run(
            debug=debug,
            port=port,
            host='0.0.0.0'
        )
