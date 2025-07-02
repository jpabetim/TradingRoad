# backend/app.py - VERSIÓN FINAL CORREGIDA

import os
import json
import requests
import pandas as pd
import numpy
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask.json.provider import JSONProvider
from datetime import datetime, timedelta, timezone
import traceback
from collections import defaultdict
from decimal import Decimal

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
        return {"timestamps": df['timestamp'].dt.strftime('%Y-%m-%d').tolist(), "values": df['sma_7'].tolist()}
    except Exception as e:
        print(f"Error al obtener historial de DVOL para {instrument}: {e}"); return None

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

# ==============================================================================
# SECCIÓN 4: CONFIGURACIÓN DE FLASK Y RUTAS API
# ==============================================================================
class NumpyJSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("default",self.default); return json.dumps(obj, **kwargs)
    def default(self, obj):
        if isinstance(obj, (numpy.integer, numpy.int64)): return int(obj)
        if isinstance(obj, (numpy.floating, numpy.float64)): return float(obj)
        if isinstance(obj, numpy.ndarray): return obj.tolist()
        if isinstance(obj, Decimal): return float(obj)
        return super(NumpyJSONProvider, self).default(obj)

app=Flask(__name__); app.json=NumpyJSONProvider(app); CORS(app);

DATA_CACHE = {}
CACHE_EXPIRY_SECONDS = 300 # Expiración de 5 minutos

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

@app.route("/api/data/<currency>", methods=["GET"])
def get_filtered_data(currency):
    exp_date_str = request.args.get('expiration', None)
    full_df = get_data(currency.upper())
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
    return jsonify({"metrics":metrics, "strike_chart_data":strike_chart, "expiration_chart_data":exp_chart, "volatility_smile_data":volatility_smile_data, "volume_chart_data": volume_chart_data})

@app.route("/api/dvol-history/<currency>", methods=["GET"])
def get_dvol_history_endpoint(currency):
    days = request.args.get('days', 90, type=int)
    data = get_deribit_dvol_history(currency.upper(), days=days)
    return jsonify(data) if data else (jsonify({"error": "No se pudieron obtener datos de DVOL"}), 500)
    
@app.route("/api/consolidated-metrics/<symbol>", methods=["GET"])
def get_consolidated_metrics(symbol):
    deribit_df, binance_sentiment, binance_funding_info, weekly_stats = get_data(symbol.upper()), get_binance_sentiment_data(symbol.upper()), get_binance_funding_info(symbol.upper()), get_binance_klines(symbol.upper())
    deribit_metrics = calculate_deribit_metrics(deribit_df) if deribit_df is not None else {}
    oi_deribit_usd = deribit_metrics.get("notional_value_usd", 0)
    oi_binance_usd = binance_sentiment.get("current_oi_binance", 0) if binance_sentiment else 0
    total_oi_avg = (oi_deribit_usd + oi_binance_usd) / 2 if (oi_deribit_usd or oi_binance_usd) else 0
    def format_timestamp(ts): return datetime.fromtimestamp(ts / 1000).strftime('%d-%b') if ts else None
    return jsonify({"oi_total_average":total_oi_avg, "oi_change_4h_percent":binance_sentiment.get("oi_change_4h_percent") if binance_sentiment else None, "funding_rate_average":binance_funding_info.get("current_funding_rate", 0.0) if binance_funding_info else 0.0, "next_funding_time_ms":binance_funding_info.get("next_funding_time_ms", 0) if binance_funding_info else 0, "deribit_max_pain":deribit_metrics.get("max_pain", 0), "current_price":binance_funding_info.get("mark_price", 0.0) if binance_funding_info else 0.0, "week_high":weekly_stats.get("week_high", 0) if weekly_stats else 0, "week_high_date":format_timestamp(weekly_stats.get("week_high_timestamp")) if weekly_stats else None, "week_low":weekly_stats.get("week_low", 0) if weekly_stats else 0, "week_low_date":format_timestamp(weekly_stats.get("week_low_timestamp")) if weekly_stats else None})

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
    df = get_data(currency.upper());
    if df is None: return jsonify({"error":"No se pudieron obtener datos"}), 500
    return jsonify(sorted([pd.to_datetime(d).strftime('%Y-%m-%d') for d in df['expiration_date'].unique()]))

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)