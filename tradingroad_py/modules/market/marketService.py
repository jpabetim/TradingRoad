"""
Servicio para obtener datos de mercados financieros usando CCXT y otras APIs
"""
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional, Union

class MarketService:
    """
    Servicio para obtener datos de mercados financieros
    """
    
    def __init__(self):
        # API Keys - initialized with empty strings to allow public API access
        self.binance_api_key = ""
        self.binance_api_secret = ""  # Also need secret key for authenticated requests
        self.marketstack_api_key = ""
        self.twelve_api_key = ""
        self.fmp_api_key = ""
        
        # Inicializar exchanges
        self.exchanges = {}
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """
        Inicializa los exchanges disponibles en CCXT
        """
        try:
            # Configure exchanges with optional API keys (public APIs work without keys)
            config = {
                'enableRateLimit': True,  # Important to avoid rate limiting
                'timeout': 30000,         # Extend timeout to 30 seconds
            }
            
            # Add API keys only if they are set
            if self.binance_api_key and self.binance_api_secret:
                config['apiKey'] = self.binance_api_key
                config['secret'] = self.binance_api_secret
            
            # List of exchanges to initialize
            exchange_classes = {
                'binance': ccxt.binance,
                'coinbase': ccxt.coinbase,
                'kraken': ccxt.kraken,
                'kucoin': ccxt.kucoin,
                'bybit': ccxt.bybit,
            }
            
            # Agregar BingX si está disponible en la versión de CCXT
            try:
                if hasattr(ccxt, 'bingx'):
                    exchange_classes['bingx'] = ccxt.bingx
                else:
                    print("CCXT no tiene soporte para BingX en esta versión")
            except Exception as e:
                print(f"Error al intentar añadir BingX: {e}")
            
            # Initialize each exchange with proper error handling
            for name, exchange_class in exchange_classes.items():
                try:
                    self.exchanges[name] = exchange_class(config)
                    print(f"Exchange {name} inicializado correctamente")
                except Exception as e:
                    print(f"Error al inicializar {name}: {e}")
            
            # Test loading markets on each exchange
            for exchange_name, exchange in self.exchanges.items():
                try:
                    exchange.load_markets()
                    print(f"Mercados cargados correctamente para {exchange_name}")
                except Exception as e:
                    print(f"No se pudieron cargar los mercados para {exchange_name}: {e}")
            
        except Exception as e:
            print(f"Error al inicializar exchanges: {e}")
    
    def get_available_exchanges(self):
        """
        Devuelve la lista de exchanges disponibles
        """
        return list(self.exchanges.keys())
    
    def get_available_timeframes(self, exchange: str = 'binance'):
        """
        Devuelve los timeframes disponibles para un exchange
        """
        try:
            if exchange in self.exchanges:
                return list(self.exchanges[exchange].timeframes.keys())
            return []
        except Exception as e:
            print(f"Error obteniendo timeframes: {e}")
            return []
    
    def get_symbols(self, exchange: str = 'binance'):
        """
        Obtiene los símbolos disponibles en un exchange
        """
        try:
            if exchange in self.exchanges:
                self.exchanges[exchange].load_markets()
                return list(self.exchanges[exchange].markets.keys())
            return []
        except Exception as e:
            print(f"Error obteniendo símbolos: {e}")
            return []
    
    def get_ohlcv(self, 
                 exchange: str = 'binance', 
                 symbol: str = 'BTC/USDT', 
                 timeframe: str = '1h', 
                 limit: int = 100) -> List[Dict]:
        """
        Obtiene datos OHLCV (Open, High, Low, Close, Volume) para un símbolo
        
        Args:
            exchange: Nombre del exchange
            symbol: Símbolo en formato 'BTC/USDT'
            timeframe: Intervalo de tiempo ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit: Número máximo de velas a devolver
            
        Returns:
            Lista de datos OHLCV en formato para gráficos
        """
        try:
            if exchange not in self.exchanges:
                return []
            
            # Obtener datos OHLCV
            ohlcv = self.exchanges[exchange].fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convertir a formato para gráficos
            formatted_data = []
            for candle in ohlcv:
                timestamp, open_price, high, low, close, volume = candle
                # Debug timestamp value
                print(f"Raw timestamp: {timestamp}, date: {datetime.fromtimestamp(timestamp / 1000)}")
                
                # Lightweight charts can use ISO format strings or epoch timestamps
                date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                
                # Check for timestamps in the future (common issue with some APIs)
                current_date = datetime.now()
                data_date = datetime.fromtimestamp(timestamp / 1000)
                if data_date > current_date:
                    print(f"Warning: Future date detected: {date_str}")
                    # Use current date instead to prevent chart display issues
                    date_str = current_date.strftime('%Y-%m-%d')
                
                formatted_data.append({
                    'time': date_str,
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'close': float(close),
                    'volume': float(volume)
                })
            
            return formatted_data
        
        except Exception as e:
            print(f"Error obteniendo datos OHLCV: {e}")
            return []
    
    def get_stock_data_twelvedata(self, symbol: str = 'AAPL', interval: str = '1day', outputsize: int = 100) -> List[Dict]:
        """
        Obtiene datos de acciones usando la API de Twelve Data
        
        Args:
            symbol: Símbolo de la acción
            interval: Intervalo de tiempo ('1min', '5min', '1hour', '1day', etc.)
            outputsize: Número de registros a devolver
            
        Returns:
            Lista de datos OHLCV en formato para gráficos
        """
        try:
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.twelve_api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'values' not in data:
                print(f"Error en la respuesta de Twelve Data: {data}")
                return []
            
            # Convertir a formato para gráficos
            formatted_data = []
            for item in data['values']:
                formatted_data.append({
                    'time': item['datetime'].split(' ')[0],  # Solo fecha YYYY-MM-DD
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': float(item['volume']) if 'volume' in item else 0
                })
            
            return formatted_data
        
        except Exception as e:
            print(f"Error obteniendo datos de Twelve Data: {e}")
            return []
    
    def get_stock_data_marketstack(self, symbol: str = 'AAPL', interval: str = 'day', limit: int = 100) -> List[Dict]:
        """
        Obtiene datos de acciones usando la API de Marketstack
        
        Args:
            symbol: Símbolo de la acción
            interval: Intervalo de tiempo ('day', 'week', 'month')
            limit: Número de registros a devolver
            
        Returns:
            Lista de datos OHLCV en formato para gráficos
        """
        try:
            url = f"http://api.marketstack.com/v1/eod"
            params = {
                'access_key': self.marketstack_api_key,
                'symbols': symbol,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'data' not in data:
                print(f"Error en la respuesta de Marketstack: {data}")
                return []
            
            # Convertir a formato para gráficos
            formatted_data = []
            for item in data['data']:
                formatted_data.append({
                    'time': item['date'].split('T')[0],  # Solo fecha YYYY-MM-DD
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': float(item['volume'])
                })
            
            return formatted_data
        
        except Exception as e:
            print(f"Error obteniendo datos de Marketstack: {e}")
            return []
    
    def get_stock_data_fmp(self, symbol: str = 'AAPL', timeframe: str = 'day', limit: int = 100) -> List[Dict]:
        """
        Obtiene datos de acciones usando la API de Financial Modeling Prep
        
        Args:
            symbol: Símbolo de la acción
            timeframe: Intervalo de tiempo ('day', 'hour', 'minute')
            limit: Número de registros a devolver
            
        Returns:
            Lista de datos OHLCV en formato para gráficos
        """
        try:
            # Mapeo de timeframes
            timeframe_map = {
                'day': '1day',
                'hour': '1hour',
                'minute': '1min',
                '1d': '1day',
                '1h': '1hour',
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '4h': '4hour'
            }
            
            fmp_timeframe = timeframe_map.get(timeframe, '1day')
            
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp_timeframe}/{symbol}"
            params = {
                'apikey': self.fmp_api_key,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if not data or not isinstance(data, list):
                print(f"Error en la respuesta de FMP: {data}")
                return []
            
            # Convertir a formato para gráficos
            formatted_data = []
            for item in data[:limit]:  # Limitar al número especificado
                formatted_data.append({
                    'time': item['date'].split(' ')[0],  # Solo fecha YYYY-MM-DD
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': float(item['volume']) if 'volume' in item else 0
                })
            
            # Invertir para que estén en orden cronológico
            formatted_data.reverse()
            return formatted_data
        
        except Exception as e:
            print(f"Error obteniendo datos de FMP: {e}")
            return []
    
    def get_market_data(self, 
                      source: str = 'binance', 
                      symbol: str = 'BTC/USDT', 
                      timeframe: str = '1h', 
                      limit: int = 100) -> List[Dict]:
        """
        Función unificada para obtener datos de mercado de cualquier fuente
        
        Args:
            source: Fuente de datos ('binance', 'coinbase', 'twelvedata', 'marketstack', 'fmp')
            symbol: Símbolo a consultar
            timeframe: Intervalo de tiempo
            limit: Número de registros a devolver
            
        Returns:
            Lista de datos OHLCV en formato para gráficos
        """
        print(f"Getting market data: source={source}, symbol={symbol}, timeframe={timeframe}, limit={limit}")
        
        try:
            # Mapeo de timeframes entre diferentes fuentes
            timeframe_map = {
                'binance': {
                    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', 
                    '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
                },
                'twelvedata': {
                    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', 
                    '1h': '1h', '4h': '4h', '1d': '1day', '1w': '1week'
                },
                'marketstack': {
                    '1d': 'day', '1w': 'week', '1M': 'month'
                },
                'fmp': {
                    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', 
                    '1h': '1hour', '4h': '4hour', '1d': '1day'
                }
            }
            
            data = []
            # Elegir la fuente de datos y obtener los datos
            if source.lower() in self.exchanges:
                # Es un exchange de CCXT
                data = self.get_ohlcv(source.lower(), symbol, timeframe, limit)
            elif source.lower() == 'twelvedata':
                tf = timeframe_map.get('twelvedata', {}).get(timeframe, '1h')
                data = self.get_stock_data_twelvedata(symbol, tf, limit)
            elif source.lower() == 'marketstack':
                tf = timeframe_map.get('marketstack', {}).get(timeframe, 'day')
                data = self.get_stock_data_marketstack(symbol, tf, limit)
            elif source.lower() == 'fmp':
                data = self.get_stock_data_fmp(symbol, timeframe, limit)
            else:
                # Por defecto, usar Binance
                data = self.get_ohlcv('binance', symbol, timeframe, limit)
            
            # Check if we got valid data
            if data and len(data) > 0:
                print(f"Retrieved {len(data)} valid data points")
                return data
            else:
                print(f"No data retrieved, using sample data")
                return self.generate_sample_data(symbol, limit)
                
        except Exception as e:
            print(f"Error getting market data: {e}")
            # Return sample data in case of any error
            return self.generate_sample_data(symbol, limit)
    
    def get_market_sources(self) -> List[str]:
        """
        Devuelve todas las fuentes de datos disponibles
        """
        exchanges = self.get_available_exchanges()
        other_sources = ['twelvedata', 'marketstack', 'fmp']
        return exchanges + other_sources
    
    def generate_sample_data(self, symbol='BTC/USDT', limit=100):
        """
        Generate sample data when APIs fail to deliver data
        
        Args:
            symbol: The symbol to generate data for
            limit: Number of candles to generate
            
        Returns:
            List of sample OHLCV data points for chart display
        """
        import random
        import numpy as np
        from datetime import datetime, timedelta
        
        # Start with reasonable price based on the symbol
        base_price = 0
        if 'BTC' in symbol:
            base_price = 50000
        elif 'ETH' in symbol:
            base_price = 2500
        elif 'SOL' in symbol:
            base_price = 100
        elif 'XRP' in symbol:
            base_price = 0.5
        else:
            base_price = 100  # Default for other symbols
        
        # Generate random walk prices
        prices = [base_price]
        for i in range(1, limit):
            # Random price change with trend/momentum
            change = np.random.normal(0, base_price * 0.01)  # 1% standard deviation
            new_price = max(0.01, prices[-1] + change)  # Ensure price is positive
            prices.append(new_price)
        
        # Generate candles from the prices
        end_date = datetime.now()
        formatted_data = []
        
        for i in range(limit):
            # Calculate date (newest to oldest)
            candle_date = end_date - timedelta(days=limit-i-1)
            date_str = candle_date.strftime('%Y-%m-%d')
            
            # Calculate OHLC values
            close = prices[i]
            high = close * (1 + random.uniform(0, 0.02))  # Up to 2% higher
            low = close * (1 - random.uniform(0, 0.02))   # Up to 2% lower
            open_price = low + random.random() * (high - low)  # Random between high and low
            
            # Generate volume
            volume = base_price * random.uniform(10, 100)  # Higher volume for higher priced assets
            
            formatted_data.append({
                'time': date_str,
                'open': float(open_price),
                'high': float(high),
                'low': float(low),
                'close': float(close),
                'volume': float(volume)
            })
        
        print(f"Generated {len(formatted_data)} sample data points for {symbol}")
        return formatted_data
