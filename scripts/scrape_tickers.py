import robin_stocks.robinhood as rh
from ticker_extract import extract_stock_historicals
import pandas as pd
import os

DATA_ROOT = "/home/mbhat/tradebot/custom_autotrader/data/raw"
WATCHLIST = "Tech"

if __name__ == "__main__":

    # Set up Robinhood API authentication
    rh.login(username='manojbhat09@gmail.com', password='MENkeys796@09@')
    watchlist = rh.get_watchlist_by_name(f'{WATCHLIST}')
    symbols = [i['symbol'] for i in watchlist['results']] 
    '''
    ['CVNA', 'DE', 'CCL', 'SIVBQ', 'UPST', 'MBLY', 'NFLX', 'VTI', 'DASH', 'NKE', 'T', 'TMUS', 'TSLA', 'AAPL', 'UBER', 'TM', 'QCOM', 'ADBE', 'NVDA', 'ORCL', 'META', 'AMZN', 'GOOG', 'MSFT']
    '''

    for symbol in symbols:
        extract_stock_historicals(rh, symbol)