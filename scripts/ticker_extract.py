import robin_stocks.robinhood as rh
import pandas as pd
import numpy as np
import os

DATA_ROOT = "/home/mbhat/tradebot/custom_autotrader/data/raw"
STOCK_TICKER = "TSLA"

def extract_stock_historicals(rh, ticker):
    # Get historical data for stock+
    stock_history = rh.stocks.get_stock_historicals(f'{ticker}', interval='5minute', span='week')

    # Convert the historical data to a pandas dataframe
    stock_df = pd.DataFrame(stock_history)

    try:
        # Preprocess the data (e.g., convert dates to datetime objects, drop unnecessary columns, etc.)
        stock_df['date'] = pd.to_datetime(stock_df['begins_at'])
        stock_df.set_index('date', inplace=True)

        # Save the DataFrame to a CSV file
        stock_csv_path = os.path.join(DATA_ROOT, f"{ticker}_data")
        stock_df.to_csv(f'{stock_csv_path}.csv')
    except Exception as e:
        import pdb
        pdb.set_trace()

if __name__ == "__main__":
    # Set up Robinhood API authentication
    rh.login(username='manojbhat09@gmail.com', password='MENkeys796@09@')

    extract_stock_historicals(rh, STOCK_TICKER)

#username='manojbhat09@gmail.com', password='MENkeys796@09@'