import matplotlib.pyplot as plt
from data_process import seqTradeDataset, collate_fn, collate_fn_angled
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import argparse
Ticker = "ETH-USD"
def fetch_and_preprocess_data(ticker="ETH-USD", start_date="2022-01-01", end_date="2022-10-01", time_interval="30m", period='2mo'):
    # Fetching the data
    # processed_data = seqTradeDataset.fetch_and_preprocess_data(
    #     ticker=ticker, 
    #     start_date=start_date, 
    #     end_date=end_date, 
    #     time_interval=time_interval)
    # if not len(processed_data):
    #     logging.error(f"No data fetched for ticker: {ticker}")
    #     return None, None
    
    # logging.info(f'Data fetched and preprocessed for ticker: {ticker}')
    # ticker=Ticker
    # period=period
    # start_date=None 
    # end_date=None
    # time_interval=time_interval 
    # run_dir=None
    processed_data = seqTradeDataset.fetch_data(ticker=ticker, period=period, start_date=start_date, end_date=end_date, time_interval=time_interval)
    return processed_data

# # Assuming processed_data is the output from fetch_and_preprocess_data
# processed_data = fetch_and_preprocess_data(ticker=Ticker, 
#                                            start_date = "2022-01-01", 
#                                            end_date = "2022-10-01", 
#                                            time_interval = "15m", 
#                                            period = "10mo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and preprocess data.')
    parser.add_argument('--ticker', default='ETH-USD', help='The ticker symbol.')
    parser.add_argument('--start_date', default='2022-01-01', help='The start date.')
    parser.add_argument('--end_date', default='2022-10-01', help='The end date.')
    parser.add_argument('--time_interval', default='30m', help='The time interval.')
    parser.add_argument('--period', default='2mo', help='The period.')
    
    args = parser.parse_args()
    
    processed_data = fetch_and_preprocess_data(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        time_interval=args.time_interval,
        period=args.period
    )

    # import pdb; pdb.set_trace()
    
    if not len(processed_data):
        print("its empty")
        exit() 
    processed_data = processed_data[:,  3].reshape(-1, 1)
    # Check if the data is not None (indicating an error)
    if processed_data is not None:
        # If the data is a Pandas DataFrame
        if isinstance(processed_data, pd.DataFrame):
            # Plot all columns
            ax = processed_data.plot()
            # Save the plot to a file
            ax.figure.savefig('plot.png')
        # If the data is a numpy array
        elif isinstance(processed_data, np.ndarray):
            # Assuming the data is 2D with time along the first dimension
            fig, ax = plt.subplots()
            for i in range(processed_data.shape[1]):
                ax.plot(processed_data[:, i])
            # Save the plot to a file
            fig.savefig('plot.png')
        else:
            print(f"Unknown data type: {type(processed_data)}")
    else:
        print("No data to plot.")



# import ccxt
# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import datetime
# import robin_stocks.robinhood as rh

# rh.login(username='manojbhat09@gmail.com', password='MENkeys796@09@')

# # def
# #  fetch_and_preprocess_data(ticker="ETH/USD", start_date="2022-01-01", end_date="2022-10-01", time_interval="30m"):
# #     exchange = ccxt.coinbasepro()  # or any other exchange
# #     time_interval_ccxt = {'15m': '15m', '30m': '30m', '1h': '1h', '1d': '1D'}[time_interval]
# #     # import pdb; pdb.set_trace()
# #     start_timestamp = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
# #     end_timestamp = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp()) * 1000
# #     duration = (end_timestamp - start_timestamp) / (1000 * 60)  # convert ms to minutes
# #     limit = int(duration / int(time_interval[:-1]))  # calculate limit based on interval and date range
# #     ohlcv = exchange.fetch_ohlcv(ticker, timeframe=time_interval_ccxt, since=start_timestamp, limit=limit)
# #     df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# #     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
# #     df.set_index('timestamp', inplace=True)
# #     return df

# def fetch_and_preprocess_data_robinhood(ticker="ETH", start_date="2022-01-01", end_date="2022-10-01", time_interval="30m"):
#     # Convert dates to timestamps
#     start_timestamp = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
#     end_timestamp = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp())

#     # Fetch historical data (this is a simplification and may not work as expected, 
#     # you would need to explore robin_stocks' methods and possibly fetch and process the data in a different way)
#     historical_data = rh.crypto.get_crypto_historicals(ticker, interval=time_interval, span='year')
#     import pdb; pdb.set_trace()
#     # ... rest of data preprocessing and plotting code ...

#     # Log out of Robinhood
#     rh.logout()
#     return historical_data


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Fetch and preprocess data.')
#     parser.add_argument('--ticker', default='ETH/USD', help='The ticker symbol.')
#     parser.add_argument('--start_date', default='2022-01-01', help='The start date.')
#     parser.add_argument('--end_date', default='2022-10-01', help='The end date.')
#     parser.add_argument('--time_interval', default='1d', help='The time interval.')
    
#     args = parser.parse_args()
    
#     processed_data = fetch_and_preprocess_data_robinhood(
#         ticker=args.ticker,
#         start_date=args.start_date,
#         end_date=args.end_date,
#         time_interval=args.time_interval
#     )
    
#     import pdb; pdb.set_trace()
#     if processed_data is not None and not processed_data.empty:
#         ax = processed_data[['open', 'high', 'low', 'close']].plot()
#         ax.figure.savefig('plot.png')
#     else:
#         print("No data to plot.")