from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from flask_cors import CORS  # Importar Flask-CORS
# from flask_socketio import SocketIO, emit  # Importar SocketIO - Comentado temporalmente
import os
import json
import time
import re
import traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from decimal import Decimal
from collections import defaultdict
from dotenv import load_dotenv

# Importaciones de servicios (necesarias al inicio)
from modules.news.news import NewsService
from modules.calendar.calendar import CalendarService
from config.config import ConfigService
from modules.market.marketService import MarketService

# Importaciones pesadas bajo demanda para ahorrar memoria inicial
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai

# Cargar variables de entorno al inicio
load_dotenv('.env')
load_dotenv('.env.local')

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas
# socketio = SocketIO(app, cors_allowed_origins="*")  # Configurar SocketIO - Comentado temporalmente

# Configuración de autenticación básica
app.secret_key = 'your_secret_key'

# Inicializar servicios
news_service = NewsService()
calendar_service = CalendarService()
config_service = ConfigService()
market_service = MarketService()

# Configuración de APIs de noticias usando variables de entorno
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY') or os.getenv('VITE_FINNHUB_API_KEY') or "d1hi1h9r01qsvr2aace0d1hi1h9r01qsvr2aaceg"
FMP_API_KEY = os.getenv('FMP_API_KEY') or os.getenv('VITE_FMP_API_KEY') or "XtUErGGxXn3UOuGKmn3y6h6OWKFuoZcN"
TRADERALPHA_API_KEY = os.getenv('TRADERALPHA_API_KEY') or os.getenv('VITE_TRADERALPHA_API_KEY')
TRANSLATE_API_KEY = os.getenv('TRANSLATE_API_KEY') or os.getenv('VITE_TRANSLATE_API_KEY')
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

@app.route('/trading')
def trading():
    return render_template('trading.html')

@app.route("/volatility")
def volatility():
    return render_template("volatility.html")

@app.route("/analysis")
def analysis():
    return render_template("tradingroad_analysis.html")

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
    """Obtener datos con caché usando la función local get_deribit_option_data"""
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
            if hasattr(item['event_datetime'], 'isoformat'):
                item['event_datetime'] = item['event_datetime'].isoformat()
            else:
                item['event_datetime'] = str(item['event_datetime'])
    
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
        new_price = max(current_price + price_change, current_price * 0.8)  # No más del 20% de caída
        new_price = min(new_price, current_price * 1.2)  # No más del 20% de subida
        
        # Crear OHLC realista
        open_price = current_price
        close_price = new_price
        
        # High y Low con variación adicional
        high_variation = volatility * 0.3 * random.random()
        low_variation = volatility * 0.3 * random.random()
        
        high = max(open_price, close_price) * (1 + high_variation)
        low = min(open_price, close_price) * (1 - low_variation)
        
        # Volumen aleatorio
        volume = random.randint(1000, 100000)
        
        simulated_data.append({
            'timestamp': int(candle_time),
            'open': round(open_price, 6),
            'high': round(high, 6),
            'low': round(low, 6),
            'close': round(close_price, 6),
            'volume': volume
        })
        
        current_price = close_price
    
    print(f"Generated {len(simulated_data)} simulated candles for {symbol} {timeframe}")
    return simulated_data

# Endpoints para conexiones reales con exchanges
@app.route('/api/exchange/symbols/<exchange>')
def get_exchange_symbols(exchange):
    """Obtiene símbolos disponibles de un exchange específico"""
    try:
        if exchange.lower() == 'binance':
            url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
                return jsonify(symbols[:100])  # Limitar a 100 símbolos
        
        # Fallback para otros exchanges o errores
        return jsonify(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'])
        
    except Exception as e:
        print(f"Error obteniendo símbolos de {exchange}: {e}")
        return jsonify(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'])

@app.route('/api/exchange/klines/<exchange>')
def get_exchange_klines(exchange):
    """Obtiene datos de velas de un exchange específico"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        limit = request.args.get('limit', 100, type=int)
        
        if exchange.lower() == 'binance':
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance máximo 1000
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                klines = response.json()
                # Convertir formato Binance a formato estándar
                formatted_data = []
                for kline in klines:
                    formatted_data.append({
                        'timestamp': int(kline[0]),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                return jsonify(formatted_data)
        
        # Fallback a datos simulados
        return jsonify(generate_backup_data(symbol, interval, limit))
        
    except Exception as e:
        print(f"Error obteniendo klines de {exchange}: {e}")
        return jsonify(generate_backup_data(
            request.args.get('symbol', 'BTCUSDT'), 
            request.args.get('interval', '1h'), 
            request.args.get('limit', 100, type=int)
        ))

# Funciones para manejo de noticias
def get_rss_news():
    """Obtener noticias de RSS feeds financieros gratuitos"""
    try:
        import feedparser
        print("Obteniendo noticias de RSS feeds...")
        
        # RSS feeds gratuitos de noticias financieras
        rss_feeds = [
            {
                'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'source': 'Yahoo Finance'
            },
            {
                'url': 'https://www.marketwatch.com/rss/topstories',
                'source': 'MarketWatch'
            },
            {
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'source': 'Bloomberg'
            }
        ]
        
        all_news = []
        
        for feed_info in rss_feeds:
            try:
                print(f"Obteniendo noticias de {feed_info['source']}...")
                feed = feedparser.parse(feed_info['url'])
                
                for entry in feed.entries[:2]:  # Limitar a 2 por feed
                    # Calcular tiempo relativo
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_time = time.mktime(entry.published_parsed)
                        hours_ago = max(1, int((time.time() - published_time) / 3600))
                        time_str = f'Hace {hours_ago} horas'
                    else:
                        time_str = 'Reciente'
                    
                    news_item = {
                        'title': entry.title[:120] if hasattr(entry, 'title') else 'Sin título',
                        'description': entry.summary[:200] + '...' if hasattr(entry, 'summary') and len(entry.summary) > 200 else entry.summary[:200] if hasattr(entry, 'summary') else 'Sin descripción',
                        'source': feed_info['source'],
                        'time': time_str,
                        'url': entry.link if hasattr(entry, 'link') else '#'
                    }
                    all_news.append(news_item)
                    
            except Exception as e:
                print(f"Error procesando RSS de {feed_info['source']}: {e}")
                continue
        
        print(f"Noticias RSS obtenidas: {len(all_news)}")
        return all_news[:6]  # Limitar a 6 noticias totales
        
    except ImportError:
        print("feedparser no está instalado. Instalando...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'feedparser'])
        return get_rss_news()  # Intentar de nuevo después de instalar
    except Exception as e:
        print(f"Error obteniendo noticias RSS: {e}")
        return []

def get_fmp_news():
    """Obtener noticias de Financial Modeling Prep"""
    try:
        print(f"Obteniendo noticias de FMP con API key: {FMP_API_KEY[:10]}...")
        # Intentar diferentes endpoints de FMP
        endpoints = [
            f"{FMP_BASE_URL}/stock_news?tickers=AAPL,MSFT,GOOGL,TSLA&limit=5&apikey={FMP_API_KEY}",
            f"{FMP_BASE_URL}/general_news?page=0&apikey={FMP_API_KEY}",
            f"https://financialmodelingprep.com/api/v4/general_news?page=0&size=5&apikey={FMP_API_KEY}"
        ]
        
        for i, url in enumerate(endpoints):
            try:
                print(f"Probando endpoint FMP {i+1}/3...")
                response = requests.get(url, timeout=10)
                print(f"Respuesta FMP endpoint {i+1}: {response.status_code}")
                
                if response.status_code == 200:
                    news_data = response.json()
                    if news_data and len(news_data) > 0:
                        print(f"Noticias obtenidas de FMP: {len(news_data)}")
                        formatted_news = []
                        for item in news_data[:3]:
                            if item.get('title') or item.get('headline'):
                                title = item.get('title', item.get('headline', 'Sin título'))
                                description = item.get('text', item.get('summary', 'Sin descripción'))
                                
                                formatted_news.append({
                                    'title': title[:100],
                                    'description': description[:200] + '...' if len(description) > 200 else description,
                                    'source': item.get('site', item.get('source', 'FMP')),
                                    'time': 'Hace ' + str(max(1, int((time.time() - pd.to_datetime(item.get('publishedDate', item.get('datetime', datetime.now()))).timestamp()) / 3600))) + ' horas' if item.get('publishedDate') or item.get('datetime') else 'Reciente',
                                    'url': item.get('url', '#')
                                })
                        
                        if formatted_news:
                            print(f"Noticias formateadas de FMP: {len(formatted_news)}")
                            return formatted_news
                else:
                    print(f"Error en endpoint FMP {i+1}: {response.text[:100]}")
            except Exception as e:
                print(f"Error en endpoint FMP {i+1}: {e}")
                continue
                
    except Exception as e:
        print(f"Error general obteniendo noticias de FMP: {e}")
    return []

def filter_news_by_category_and_sentiment(news_list, category, sentiment_filter):
    """Filtrar noticias por categoría y sentimiento"""
    # Por ahora, devolver todas las noticias ya que no tenemos filtros implementados
    return news_list

def analyze_news_sentiment(news_list):
    """Analizar sentimiento de las noticias"""
    if not news_list:
        return {'positive': 0, 'negative': 0, 'neutral': 0}
    
    # Análisis básico de sentimiento por palabras clave
    positive_words = ['sube', 'gana', 'aumenta', 'positivo', 'crecimiento', 'beneficios']
    negative_words = ['baja', 'pierde', 'cae', 'negativo', 'crisis', 'pérdidas']
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for news in news_list:
        text = (news.get('title', '') + ' ' + news.get('description', '')).lower()
        positive_score = sum(1 for word in positive_words if word in text)
        negative_score = sum(1 for word in negative_words if word in text)
        
        if positive_score > negative_score:
            sentiment_counts['positive'] += 1
        elif negative_score > positive_score:
            sentiment_counts['negative'] += 1
        else:
            sentiment_counts['neutral'] += 1
    
    return sentiment_counts

def get_real_news():
    """Obtener noticias reales usando el NewsService actualizado"""
    try:
        print("\n=== OBTENIENDO NOTICIAS REALES CON RSS ===")
        
        # Usar el NewsService para obtener noticias de RSS feeds
        news_list = news_service.get_yahoo_finance_news(category="all", limit=10)
        
        print(f"Noticias obtenidas: {len(news_list)}")
        
        return news_list
        
    except Exception as e:
        print(f"Error en get_real_news: {e}")
        return get_fallback_news()

def get_fallback_news():
    """Noticias de respaldo realistas cuando las APIs fallan"""
    import random
    from datetime import datetime, timedelta
    
    # Plantillas de noticias realistas basadas en eventos típicos del mercado
    news_templates = [
        {
            'title': 'Bitcoin supera los $68,000 tras aprobación de nuevos ETFs institucionales',
            'description': 'La criptomoneda líder alcanza nuevos máximos después de que varios fondos institucionales anunciaran su entrada al mercado de activos digitales.',
            'source': 'CryptoNews',
            'category': 'crypto'
        },
        {
            'title': 'Fed mantiene tasas de interés estables en decisión unánime',
            'description': 'La Reserva Federal decidió mantener las tasas entre 5.25% y 5.50%, citando datos mixtos de inflación y empleo en su última reunión.',
            'source': 'Financial Times',
            'category': 'monetary'
        },
        {
            'title': 'Tesla reporta ganancias récord en Q4 impulsadas por ventas en China',
            'description': 'Las acciones de Tesla suben 8% en premarket tras reportar ingresos de $29.3 mil millones, superando estimaciones de analistas.',
            'source': 'MarketWatch',
            'category': 'earnings'
        },
        {
            'title': 'Nvidia anuncia nueva arquitectura de chips para IA generativa',
            'description': 'La compañía presenta sus procesadores de próxima generación, diseñados específicamente para aplicaciones de inteligencia artificial.',
            'source': 'Tech Finance',
            'category': 'tech'
        },
        {
            'title': 'Mercados asiáticos cierran mixtos ante incertidumbre geopolítica',
            'description': 'El Nikkei avanza 0.8% mientras que el Hang Seng retrocede 1.2% en medio de tensiones comerciales renovadas.',
            'source': 'Asia Markets',
            'category': 'markets'
        },
        {
            'title': 'Petróleo WTI alcanza $82 por barril tras recortes de producción OPEP+',
            'description': 'Los precios del crudo suben 3.5% después de que la OPEP+ anunciara extensión de recortes de producción hasta junio.',
            'source': 'Energy News',
            'category': 'commodities'
        },
        {
            'title': 'JPMorgan eleva precio objetivo del S&P 500 para 2025',
            'description': 'El banco de inversión proyecta que el índice alcance 5,400 puntos, respaldado por sólidas ganancias corporativas.',
            'source': 'Investment News',
            'category': 'analysis'
        },
        {
            'title': 'Ethereum actualización mejora escalabilidad y reduce comisiones',
            'description': 'La red implementa mejoras que reducen costos de transacción en 40% y aumenta la velocidad de procesamiento.',
            'source': 'Blockchain Today',
            'category': 'crypto'
        }
    ]
    
    # Seleccionar 3 noticias aleatorias
    selected_news = random.sample(news_templates, 3)
    
    # Generar horarios realistas (últimas 24 horas)
    now = datetime.now()
    formatted_news = []
    
    for i, news in enumerate(selected_news):
        # Generar tiempo aleatorio en las últimas 24 horas
        hours_ago = random.randint(1, 24)
        minutes_ago = random.randint(0, 59)
        
        # Crear fecha realista
        news_date = now - timedelta(hours=hours_ago, minutes=minutes_ago)
        time_str = f"Hace {hours_ago} horas" if hours_ago > 1 else "Hace 1 hora"
        
        formatted_news.append({
            'title': news['title'],
            'description': news['description'],
            'source': news['source'],
            'time': time_str,
            'date': news_date.strftime("%Y-%m-%d %H:%M:%S"),  # Añadir fecha formatada
            'timestamp': news_date.timestamp(),  # Añadir timestamp
            'url': '#',
            'category': news['category'],
            'sentiment': 'neutral'  # Añadir sentimiento por defecto
        })
    
    return formatted_news

# Endpoint para el asistente IA que puede recibir datos del gráfico
@app.route('/api/ai/analyze-chart', methods=['POST'])
def analyze_chart_with_ai():
    """
    Endpoint para analizar el gráfico con IA
    Recibe datos del gráfico (OHLC, indicadores, imagen base64) y los analiza
    """
    try:
        data = request.json
        
        # Extraer datos del request
        chart_data = data.get('chartData', [])  # Array de velas OHLC
        indicators = data.get('indicators', {})  # Indicadores técnicos
        symbol = data.get('symbol', 'BTC/USDT')
        timeframe = data.get('timeframe', '1h')
        user_question = data.get('question', 'Analiza este gráfico')
        chart_image_base64 = data.get('chartImage', '')  # Imagen del gráfico en base64
        
        # Configurar Gemini - usar las variables correctas para Flask
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('VITE_GEMINI_API_KEY')
        if not api_key or api_key == 'TU_CLAVE_API_DE_GEMINI_AQUI':
            return jsonify({
                'error': 'API key de Gemini no configurada correctamente',
                'message': 'Por favor configura tu API key en las variables de entorno'
            }), 400
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Construir prompt con datos del gráfico
        prompt = f"""
        Eres un analista de trading experto especializado en análisis técnico Wyckoff y Smart Money Concepts (SMC).
        
        Analiza el siguiente gráfico de {symbol} en timeframe {timeframe}:
        
        DATOS DEL GRÁFICO:
        - Número de velas: {len(chart_data)}
        - Precio actual: {chart_data[-1]['close'] if chart_data else 'N/A'}
        - Máximo del período: {max([candle['high'] for candle in chart_data]) if chart_data else 'N/A'}
        - Mínimo del período: {min([candle['low'] for candle in chart_data]) if chart_data else 'N/A'}
        
        INDICADORES TÉCNICOS:
        {json.dumps(indicators, indent=2)}
        
        DATOS OHLC (últimas 10 velas):
        {json.dumps(chart_data[-10:] if len(chart_data) >= 10 else chart_data, indent=2)}
        
        PREGUNTA DEL USUARIO:
        {user_question}
        
        Por favor proporciona un análisis detallado que incluya:
        1. Análisis de la estructura del precio (Wyckoff)
        2. Identificación de zonas de liquidity y order blocks (SMC)
        3. Niveles de soporte y resistencia clave
        4. Señales de entrada y salida potenciales
        5. Gestión de riesgo recomendada
        6. Outlook general del mercado
        
        Responde en español de manera clara y estructurada.
        """
        
        # Si hay imagen del gráfico, incluirla en el análisis
        if chart_image_base64:
            try:
                import base64
                # Decodificar la imagen base64
                image_data = base64.b64decode(chart_image_base64.split(',')[1] if ',' in chart_image_base64 else chart_image_base64)
                
                # Crear objeto de imagen para Gemini
                import io
                from PIL import Image
                image = Image.open(io.BytesIO(image_data))
                
                # Generar respuesta con imagen y texto
                response = model.generate_content([prompt, image])
            except Exception as img_error:
                print(f"Error procesando imagen: {img_error}")
                # Fallback: análisis solo con texto
                response = model.generate_content(prompt)
        else:
            # Análisis solo con texto
            response = model.generate_content(prompt)
        
        return jsonify({
            'analysis': response.text,
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en análisis IA: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

# Endpoint para asistente IA global (funciona en todas las secciones)

def get_live_market_data():
    """
    Obtiene datos de mercado en tiempo real para el asistente IA
    """
    try:
        market_data = {}
        
        # Obtener datos de criptomonedas (Binance)
        crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
        for symbol in crypto_symbols:
            try:
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    clean_symbol = symbol.replace('USDT', '')
                    market_data[clean_symbol] = {
                        'price': float(data['lastPrice']),
                        'change_24h': float(data['priceChangePercent']),
                        'volume_24h': float(data['volume'])
                    }
            except Exception as e:
                print(f"Error obteniendo {symbol}: {e}")
        
        # Obtener datos de índices tradicionales si hay tiempo
        try:
            spy_url = f"https://financialmodelingprep.com/api/v3/quote/SPY?apikey={FMP_API_KEY}"
            spy_response = requests.get(spy_url, timeout=3)
            if spy_response.status_code == 200:
                spy_data = spy_response.json()
                if spy_data:
                    spy_info = spy_data[0]
                    market_data['SPY'] = {
                        'price': spy_info['price'],
                        'change_24h': spy_info['changesPercentage'],
                        'volume_24h': spy_info.get('volume', 0)
                    }
        except Exception as e:
            print(f"Error obteniendo SPY: {e}")
            
        return market_data
        
    except Exception as e:
        print(f"Error en get_live_market_data: {e}")
        return {}

def format_market_data_for_ai(market_data):
    """
    Formatea los datos de mercado para que la IA los pueda entender fácilmente
    """
    if not market_data:
        return "Datos de mercado no disponibles actualmente."
    
    formatted_text = "📊 RESUMEN DE MERCADO ACTUAL:\n"
    
    # Separar criptomonedas y tradicionales
    crypto_data = {}
    traditional_data = {}
    
    for symbol, data in market_data.items():
        if symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
            crypto_data[symbol] = data
        else:
            traditional_data[symbol] = data
    
    # Formato para criptomonedas
    if crypto_data:
        formatted_text += "\n🪙 CRIPTOMONEDAS:\n"
        for symbol, data in crypto_data.items():
            change_emoji = "🟢" if data['change_24h'] > 0 else "🔴" if data['change_24h'] < 0 else "⚪"
            formatted_text += f"• {symbol}: ${data['price']:,.2f} ({change_emoji} {data['change_24h']:+.2f}%)\n"
    
    # Formato para instrumentos tradicionales
    if traditional_data:
        formatted_text += "\n📈 MERCADOS TRADICIONALES:\n"
        for symbol, data in traditional_data.items():
            change_emoji = "🟢" if data['change_24h'] > 0 else "🔴" if data['change_24h'] < 0 else "⚪"
            formatted_text += f"• {symbol}: ${data['price']:,.2f} ({change_emoji} {data['change_24h']:+.2f}%)\n"
    
    formatted_text += f"\n⏰ Última actualización: {datetime.now().strftime('%H:%M:%S')}"
    
    return formatted_text

@app.route('/api/ai/assistant', methods=['POST'])
def ai_assistant():
    """
    Endpoint para el asistente IA que funciona en todas las secciones
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        section = data.get('section', 'general')  # dashboard, analysis, news, etc.
        context = data.get('context', {})  # Contexto adicional
        
        if not user_message:
            return jsonify({
                'error': 'Mensaje no proporcionado',
                'response': 'Por favor, escribe una pregunta o solicitud.'
            }), 400
        
        # Configurar Gemini
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('VITE_GEMINI_API_KEY')
        if not api_key or api_key == 'TU_CLAVE_API_DE_GEMINI_AQUI':
            return jsonify({
                'error': 'API key de Gemini no configurada',
                'response': 'Lo siento, el asistente IA no está disponible. Por favor configura la API key de Gemini.'
            }), 400
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Construir prompt específico según la sección
        section_prompts = {
            'dashboard': """Eres un asistente experto en análisis de mercados financieros y trading. 
                         Estás ayudando en la sección del dashboard donde se muestran datos de mercado en tiempo real.
                         
                         IMPORTANTE: Tienes acceso a datos de mercado actuales que aparecerán en el contexto.
                         Úsa SIEMPRE estos datos reales en tus respuestas para:
                         - Analizar tendencias actuales de precios
                         - Identificar oportunidades de trading
                         - Comentar sobre el sentimiento del mercado
                         - Comparar el rendimiento entre diferentes activos
                         
                         Proporciona análisis claros, específicos y CONCISOS (máximo 150 palabras) basados en los datos reales proporcionados.""",
            
            'analysis': """Eres un analista técnico experto especializado en análisis Wyckoff y Smart Money Concepts (SMC).
                        Estás en la sección de análisis técnico avanzado con acceso a datos OHLC en tiempo real.
                        
                        IMPORTANTE: 
                        - SIEMPRE revisa el CONTEXTO ADICIONAL que contiene datos OHLC reales del gráfico que el usuario está viendo
                        - Analiza los datos de velas (OHLC) proporcionados para identificar patrones, tendencias, soportes/resistencias
                        - Usa los datos reales para dar análisis específicos y precisos
                        - Si hay datos OHLC disponibles, NO pidas imágenes - analiza directamente los datos
                        - Sé CONCISO pero específico (máximo 200 palabras)
                        - Incluye niveles de precios específicos basados en los datos reales
                        
                        Ayuda con interpretación de gráficos, patrones, indicadores y estrategias de trading basándote en los datos OHLC reales.""",
            
            'news': """Eres un analista de noticias financieras experto. Estás en la sección de noticias.
                     Ayuda a interpretar el impacto de las noticias en los mercados, análisis de sentimiento
                     y correlaciones entre eventos y movimientos de precios. Respuestas CONCISAS (máximo 150 palabras).""",
            
            'volatility': """Eres un especialista en análisis de volatilidad y derivados financieros.
                           Estás en la sección de volatilidad. Ayuda con interpretación de datos de opciones,
                           volatilidad implícita, y estrategias de derivados. Respuestas CONCISAS (máximo 150 palabras).""",
            
            'general': """Eres un asistente experto en trading y análisis de mercados financieros.
                        Ayuda con cualquier consulta relacionada con trading, análisis técnico, 
                        mercados financieros y estrategias de inversión. Respuestas CONCISAS (máximo 150 palabras)."""
        }
        
        base_prompt = section_prompts.get(section, section_prompts['general'])
        
        # Agregar datos de mercado en tiempo real si es dashboard
        market_context = ""
        if section == 'dashboard':
            try:
                # Obtener datos de mercado actuales
                market_data = get_live_market_data()
                if market_data:
                    market_context = f"""
                    
DATOS DE MERCADO EN TIEMPO REAL (incluye en tu análisis):
{format_market_data_for_ai(market_data)}
"""
            except Exception as e:
                print(f"Error obteniendo datos de mercado para IA: {e}")
        
        # Agregar contexto si está disponible
        context_text = ""
        if context:
            print(f"📊 Contexto recibido para IA (sección: {section}):")
            print(f"Keys: {list(context.keys()) if isinstance(context, dict) else 'No es dict'}")
            if isinstance(context, dict) and len(str(context)) < 1000:
                print(f"Contenido: {context}")
            else:
                print(f"Contenido muy largo, primeros 500 chars: {str(context)[:500]}...")
            context_text = f"\n\nCONTEXTO ADICIONAL:\n{json.dumps(context, indent=2)}"
        
        full_prompt = f"""
        {base_prompt}
        {market_context}
        
        MENSAJE DEL USUARIO:
        {user_message}
        {context_text}
        
        INSTRUCCIONES IMPORTANTES:
        - Responde de manera CLARA, CONCISA y ÚTIL en español
        - Máximo 200 palabras para análisis técnico, 150 para otras secciones
        - Si es una pregunta técnica, sé específico pero breve
        - Si es sobre estrategias, incluye consideraciones de riesgo básicas
        - Si preguntan sobre un gráfico específico y no puedes verlo, sugiere que compartan una imagen o describan lo que ven
        """
        
        # Generar respuesta
        response = model.generate_content(full_prompt)
        
        return jsonify({
            'response': response.text,
            'section': section,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en asistente IA: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Error interno',
            'response': f'Lo siento, ocurrió un error: {str(e)}'
        }), 500

# Rutas adicionales para la página de análisis avanzado
@app.route('/api/exchange/ticker/<exchange>')
def get_exchange_ticker(exchange):
    """Obtiene el ticker actual de un exchange específico"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        
        if exchange.lower() == 'binance':
            url = f"https://api.binance.com/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return jsonify({
                    'symbol': data['symbol'],
                    'price': float(data['lastPrice']),
                    'change': float(data['priceChange']),
                    'changePercent': float(data['priceChangePercent']),
                    'volume': float(data['volume']),
                    'high': float(data['highPrice']),
                    'low': float(data['lowPrice'])
                })
        
        # Fallback data
        return jsonify({
            'symbol': symbol,
            'price': 67500.00,
            'change': 1200.50,
            'changePercent': 1.81,
            'volume': 28500000000,
            'high': 68200.00,
            'low': 66800.00
        })
        
    except Exception as e:
        print(f"Error obteniendo ticker de {exchange}: {e}")
        return jsonify({
            'symbol': symbol,
            'price': 67500.00,
            'change': 1200.50,
            'changePercent': 1.81,
            'volume': 28500000000,
            'high': 68200.00,
            'low': 66800.00
        })

# Ruta simple para manejar las solicitudes de Socket.IO (devolvemos error 404 limpio)
@app.route('/socket.io/')
def socket_io_fallback():
    """Fallback para solicitudes de Socket.IO no implementadas"""
    return jsonify({'error': 'Socket.IO not implemented'}), 404

if __name__ == '__main__':
    # Solo ejecutar Flask directamente en desarrollo local
    # En producción se usa Gunicorn, por lo que esto no se ejecutará
    print("=== INICIANDO TRADEROAD FLASK APP (DESARROLLO) ===")
    
    # Obtener puerto de la variable de entorno (Render lo proporciona)  
    port = int(os.getenv('PORT', 5007))
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"Backend API: Running on port {port}")
    if os.getenv('RENDER'):
        print("WARNING: Ejecutándose en modo desarrollo en producción!")
        print("Debería usar Gunicorn en su lugar.")
    else:
        print(f"Local URL: http://localhost:{port}")
    print(f"Debug mode: {debug_mode}")
    print("==========================================")
    
    # Ejecutar Flask con configuración para desarrollo
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
