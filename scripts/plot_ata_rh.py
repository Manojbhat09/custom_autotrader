# import datetime
# import robin_stocks.robinhood as r
# import pandas as pd
# from tqdm import tqdm
# import logging

# logging.basicConfig(level=logging.INFO)

# def load_historic_data(symbol, end_date_str, interval='5minute'): #  Interval must be "15second","5minute","10minute","hour","day",or "week"
#     # Set up the end date and start date (5 months before the end date)
#     end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
#     start_date = end_date - datetime.timedelta(days=5*30)  # Approximate 5 months
#     current_end_date = end_date
#     all_data = pd.DataFrame()

#     # Login to Robinhood (replace with your credentials)
#     r.login(username='manojbhat09@gmail.com', password='MENkeys796@09@')
    
#     with tqdm(total=(end_date - start_date).days) as pbar:
#         while current_end_date > start_date:
#             # Calculate the start date for the current 7-day chunk
#             current_start_date = current_end_date - datetime.timedelta(days=7)
#             # Ensure the current start date does not go beyond the desired start date
#             current_start_date = max(current_start_date, start_date)
            
#             # Convert dates to string format for robin_stocks
#             current_start_str = current_start_date.strftime("%Y-%m-%d")
#             current_end_str = current_end_date.strftime("%Y-%m-%d")
            
#             try:
#                 # Fetch data for the current 7-day chunk
#                 historical_data = r.crypto.get_crypto_historicals(symbol, interval=interval, bounds='24_7', span='week',  info=None)
                
#                 # If data is returned, convert to DataFrame and append to all_data
#                 if historical_data:
#                     df = pd.DataFrame(historical_data)
#                     df['begins_at'] = pd.to_datetime(df['begins_at'])
#                     df.set_index('begins_at', inplace=True)
#                     all_data = pd.concat([all_data, df])
                
#             except Exception as e:
#                 logging.error(f"Error loading data for {symbol} from {current_start_str} to {current_end_str}: {e}")
            
#             # Update the progress bar
#             pbar.update((current_end_date - current_start_date).days)
            
#             # Update current_end_date for the next iteration
#             current_end_date = current_start_date
    
#     # Log out of Robinhood
#     r.logout()
    
#     # Sort the data by timestamp
#     all_data = all_data.sort_index()
    
#     return all_data

# # Example usage:
# end_date_str = '2022-10-01'
# symbols = ['BTC', 'ETH']  # Note: Use the ticker symbols that are recognized by Robinhood
# for symbol in symbols:
#     data = load_historic_data(symbol, end_date_str)
#     import pdb; pdb.set_trace()
#     print(data.shape)

#     if data is not None and not data.empty:
#         print(f'Data for {symbol} from {data.index.min()} to {data.index.max()} retrieved successfully.')



import ccxt
import pandas as pd
from tqdm import tqdm
import logging
import datetime

logging.basicConfig(level=logging.INFO)

def load_historic_data(symbol, end_date_str, interval='5m'):
    # Configure the exchange (e.g., Binance)
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    
    # Set up the end date and start date (5 months before the end date)
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    start_date = end_date - datetime.timedelta(days=5*30)  # Approximate 5 months
    current_start_date = end_date - datetime.timedelta(days=7)  # Start with a 7-day chunk before the end date
    all_data = pd.DataFrame()
    
    with tqdm(total=(end_date - start_date).days) as pbar:
        while current_start_date > start_date:
            # Convert dates to timestamp format for ccxt
            current_start_timestamp = int(current_start_date.timestamp()) * 1000
            
            try:
                # Fetch data for the current 7-day chunk
                ohlcv = exchange.fetch_ohlcv(symbol, interval, since=current_start_timestamp)
                # If data is returned, convert to DataFrame and append to all_data
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    all_data = pd.concat([all_data, df])
                
            except Exception as e:
                logging.error(f"Error loading data for {symbol} from {current_start_date} to {end_date}: {e}")
                
                # Update the progress bar
                pbar.update(7)  # Each chunk covers 7 days
                
                # Update current_start_date for the next iteration
                current_start_date = current_start_date - datetime.timedelta(days=7)
    # Sort the data by timestamp
    all_data = all_data.sort_index()
    
    return all_data

# Example usage:
end_date_str = '2023-08-01'
symbols = ['BTC/USDT', 'ETH/USDT']  # Use the ticker symbols that are recognized by the exchange
for symbol in symbols:
    data = load_historic_data(symbol, end_date_str)
    import pdb; pdb.set_trace()
    if data is not None and not data.empty:
        print(f'Data for {symbol} from {data.index.min()} to {data.index.max()} retrieved successfully.')