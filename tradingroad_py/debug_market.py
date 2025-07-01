"""
Script for debugging market service connectivity
"""
from modules.market.marketService import MarketService
import json

def main():
    # Create the market service
    print("Initializing Market Service...")
    market_service = MarketService()
    
    # Get available exchanges
    print("\nAvailable Exchanges:")
    exchanges = market_service.get_available_exchanges()
    print(exchanges)
    
    # Test with Binance
    exchange = 'binance'
    print(f"\nTesting with exchange: {exchange}")
    
    # Get symbols for exchange
    print(f"Symbols for {exchange}:")
    symbols = market_service.get_symbols(exchange)
    if symbols:
        print(f"Found {len(symbols)} symbols")
        print(f"First 5 symbols: {symbols[:5]}")
    else:
        print("No symbols found!")
    
    # Get timeframes for exchange
    print(f"\nTimeframes for {exchange}:")
    timeframes = market_service.get_available_timeframes(exchange)
    print(timeframes)
    
    # Test data retrieval
    print("\nTesting data retrieval...")
    symbol = 'BTC/USDT' if 'BTC/USDT' in symbols else (symbols[0] if symbols else None)
    
    if symbol:
        print(f"Getting data for {symbol} on {exchange} with 1h timeframe")
        try:
            data = market_service.get_ohlcv(exchange, symbol, '1h', 10)
            if data:
                print(f"Retrieved {len(data)} candles")
                print("First candle:")
                print(json.dumps(data[0], indent=2))
            else:
                print("No data retrieved!")
        except Exception as e:
            print(f"Error retrieving data: {e}")
    else:
        print("No symbol available to test with")
    
    # Test alternate sources
    alt_source = 'twelvedata'
    print(f"\nTesting with alternative source: {alt_source}")
    try:
        alt_data = market_service.get_stock_data_twelvedata('AAPL', '1hour', 10)
        if alt_data:
            print(f"Retrieved {len(alt_data)} candles from {alt_source}")
            print("First candle:")
            print(json.dumps(alt_data[0], indent=2))
        else:
            print(f"No data retrieved from {alt_source}!")
    except Exception as e:
        print(f"Error retrieving data from {alt_source}: {e}")

if __name__ == "__main__":
    main()
