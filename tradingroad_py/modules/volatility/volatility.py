# Funciones integradas de Volatilidad, Vencimientos y Sentimiento de Mercado

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from decimal import Decimal
import traceback

# ==============================================================================
# SECCIÓN 1: LÓGICA DE DERIBIT
# ==============================================================================
def calculate_max_pain(df):
    """Calcula el punto de máximo dolor para opciones"""
    if df.empty or 'strike' not in df or df['strike'].nunique() == 0: 
        return 0
    
    strikes = sorted(df['strike'].unique())
    total_losses = []
    
    for expiry_strike in strikes:
        loss = 0
        
        # Pérdidas para calls
        calls_df = df[df['type'] == 'C'].copy()
        calls_df['loss'] = (expiry_strike - calls_df['strike']) * calls_df['open_interest']
        calls_df.loc[calls_df['loss'] < 0, 'loss'] = 0
        loss += calls_df['loss'].sum()
        
        # Pérdidas para puts
        puts_df = df[df['type'] == 'P'].copy()
        puts_df['loss'] = (puts_df['strike'] - expiry_strike) * puts_df['open_interest']
        puts_df.loc[puts_df['loss'] < 0, 'loss'] = 0
        loss += puts_df['loss'].sum()
        
        total_losses.append(loss)
    
    min_loss_index = total_losses.index(min(total_losses))
    return strikes[min_loss_index]

def get_deribit_option_data(currency='BTC'):
    """Obtiene datos de opciones de Deribit"""
    url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind=option"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()['result']
        
        df = pd.DataFrame(data)
        if df.empty: 
            return df
        
        # Procesar datos griegos
        if 'greeks' not in df.columns:
            df['greeks'] = [{'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0} for _ in range(len(df))]
        else:
            df['greeks'] = df['greeks'].apply(lambda x: x if isinstance(x, dict) else {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0})
        
        # Procesar fechas y strikes
        df['expiration_date'] = pd.to_datetime(df['instrument_name'].str.split('-').str[1], format='%d%b%y').dt.normalize()
        df['strike'] = df['instrument_name'].str.split('-').str[2]
        df['type'] = df['instrument_name'].str.split('-').str[3]
        
        # Convertir columnas numéricas
        numeric_cols = ['mark_iv', 'underlying_price', 'strike', 'open_interest', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        df.dropna(subset=['strike'], inplace=True)
        df['strike'] = df['strike'].astype(int)
        
        # Convertir tipos numpy a tipos Python nativos para JSON
        for col in df.select_dtypes(include=[np.integer]).columns:
            df[col] = df[col].astype(int)
        for col in df.select_dtypes(include=[np.floating]).columns:
            df[col] = df[col].astype(float)
        
        return df
        
    except Exception as e:
        print(f"Error en get_deribit_option_data: {e}")
        traceback.print_exc()
        return None

def calculate_deribit_metrics(df):
    """Calcula métricas de Deribit"""
    if df is None or df.empty:
        return {}
    
    try:
        calls_df = df[df['type'] == 'C']
        puts_df = df[df['type'] == 'P']
        
        call_oi = calls_df['open_interest'].sum()
        put_oi = puts_df['open_interest'].sum()
        total_oi = call_oi + put_oi
        
        call_volume = calls_df['volume'].sum()
        put_volume = puts_df['volume'].sum()
        
        put_call_ratio_oi = put_oi / call_oi if call_oi > 0 else 0
        put_call_ratio_volume = put_volume / call_volume if call_volume > 0 else 0
        
        max_pain = calculate_max_pain(df)
        
        # Calcular valor nocional
        underlying_price = df['underlying_price'].iloc[0] if not df.empty else 0
        notional_value_usd = total_oi * underlying_price
        
        # Convertir a tipos Python nativos para serialización JSON
        return {
            'call_oi': int(call_oi) if not np.isnan(call_oi) else 0,
            'put_oi': int(put_oi) if not np.isnan(put_oi) else 0,
            'total_oi': int(total_oi) if not np.isnan(total_oi) else 0,
            'put_call_ratio_oi': float(put_call_ratio_oi) if not np.isnan(put_call_ratio_oi) else 0.0,
            'put_call_ratio_volume': float(put_call_ratio_volume) if not np.isnan(put_call_ratio_volume) else 0.0,
            'max_pain': int(max_pain) if not np.isnan(max_pain) else 0,
            'notional_value_usd': float(notional_value_usd) if not np.isnan(notional_value_usd) else 0.0,
            'underlying_price': float(underlying_price) if not np.isnan(underlying_price) else 0.0
        }
        
    except Exception as e:
        print(f"Error en calculate_deribit_metrics: {e}")
        return {}

def get_deribit_dvol_history(currency='BTC', days=90):
    """Obtiene historial de volatilidad de Deribit"""
    instrument = currency.upper()
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={instrument}&start_timestamp={start_time}&end_timestamp={end_time}&resolution=D"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('result', {}).get('data', [])
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = pd.to_numeric(df['close'])
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df.dropna(subset=['sma_7'], inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error en get_deribit_dvol_history: {e}")
        return None

# ==============================================================================
# SECCIÓN 2: LÓGICA DE BINANCE Y OTRAS
# ==============================================================================
def get_binance_klines(symbol='BTC', interval='1d', days=7):
    """Obtiene datos de velas de Binance"""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}USDT&interval={interval}&limit={days}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        week_high_idx = df['high'].idxmax()
        week_low_idx = df['low'].idxmin()
        
        return {
            'week_high': df.loc[week_high_idx, 'high'],
            'week_high_timestamp': df.loc[week_high_idx, 'timestamp'],
            'week_low': df.loc[week_low_idx, 'low'],
            'week_low_timestamp': df.loc[week_low_idx, 'timestamp']
        }
        
    except Exception as e:
        print(f"Error en get_binance_klines: {e}")
        return None

def get_binance_sentiment_data(symbol='BTC', limit_oi=48, limit_ls=48):
    """Obtiene datos de sentimiento de Binance"""
    try:
        # Obtener OI actual
        oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}USDT"
        oi_response = requests.get(oi_url, timeout=10)
        oi_response.raise_for_status()
        current_oi = float(oi_response.json()['openInterest'])
        
        # Obtener historial de OI
        oi_hist_url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}USDT&period=5m&limit={limit_oi}"
        oi_hist_response = requests.get(oi_hist_url, timeout=10)
        oi_hist_response.raise_for_status()
        oi_hist_data = oi_hist_response.json()
        
        # Obtener ratio Long/Short
        ls_url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}USDT&period=5m&limit={limit_ls}"
        ls_response = requests.get(ls_url, timeout=10)
        ls_response.raise_for_status()
        ls_data = ls_response.json()
        
        # Calcular cambio de OI en 4h
        oi_change_4h = 0
        if len(oi_hist_data) >= 48:  # 48 * 5min = 4h
            oi_4h_ago = float(oi_hist_data[-48]['sumOpenInterest'])
            oi_change_4h = ((current_oi - oi_4h_ago) / oi_4h_ago) * 100 if oi_4h_ago > 0 else 0
        
        return {
            'current_oi_binance': current_oi,
            'oi_change_4h_percent': oi_change_4h,
            'oi_history': oi_hist_data,
            'long_short_ratio': ls_data
        }
        
    except Exception as e:
        print(f"Error en get_binance_sentiment_data: {e}")
        return None

def get_binance_funding_info(symbol='BTC'):
    """Obtiene información de funding de Binance"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}USDT"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'current_funding_rate': float(data['lastFundingRate']),
            'next_funding_time_ms': int(data['nextFundingTime']),
            'mark_price': float(data['markPrice'])
        }
        
    except Exception as e:
        print(f"Error en get_binance_funding_info: {e}")
        return None

def get_binance_funding_rate_history(symbol='BTC', limit=100):
    """Obtiene historial de funding rate de Binance"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}USDT&limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return [{
            'timestamp': item['fundingTime'],
            'funding_rate': float(item['fundingRate'])
        } for item in data]
        
    except Exception as e:
        print(f"Error en get_binance_funding_rate_history: {e}")
        return None

def get_deribit_order_book(currency='BTC', depth=1000, step=0):
    """Obtiene libro de órdenes de Deribit"""
    instrument = f"{currency}-PERPETUAL"
    url = f"https://www.deribit.com/api/v2/public/get_order_book?instrument_name={instrument}&depth={depth}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()['result']
        
        # Procesar bids y asks
        bids = [[float(price), float(amount)] for price, amount in data['bids']]
        asks = [[float(price), float(amount)] for price, amount in data['asks']]
        
        # Aplicar agregación si se especifica step
        if step > 0:
            bids = aggregate_order_book_levels(bids, step, 'bid')
            asks = aggregate_order_book_levels(asks, step, 'ask')
        
        return {
            'bids': bids[:50],  # Limitar a 50 niveles
            'asks': asks[:50],
            'timestamp': data['timestamp']
        }
        
    except Exception as e:
        print(f"Error en get_deribit_order_book: {e}")
        return None

def aggregate_order_book_levels(levels, step, side):
    """Agrega niveles del libro de órdenes"""
    if not levels or step <= 0:
        return levels
    
    aggregated = {}
    
    for price, amount in levels:
        # Redondear precio al step más cercano
        if side == 'bid':
            rounded_price = int(price / step) * step
        else:  # ask
            rounded_price = (int(price / step) + 1) * step
        
        if rounded_price in aggregated:
            aggregated[rounded_price] += amount
        else:
            aggregated[rounded_price] = amount
    
    # Convertir de vuelta a lista ordenada
    result = [[price, amount] for price, amount in aggregated.items()]
    result.sort(key=lambda x: x[0], reverse=(side == 'bid'))
    
    return result
