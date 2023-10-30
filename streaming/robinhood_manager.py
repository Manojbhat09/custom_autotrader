
import robin_stocks.robinhood as rh
import streamlit as st
import logging 

class RobinhoodManager():
    def __init__(self, username, password):
        rh.login(username=username, password=password)
        self.rh = rh
    @st.cache(allow_output_mutation=True)
    def get_watchlist_by_name(self, list_name):
        try:
            watchlist = rh.get_watchlist_by_name(list_name)
            symbols = [instrument['symbol'] for instrument in watchlist['results']] if watchlist and 'results' in watchlist else []
        except Exception as e:
            logging.error(f"Error getting all unique symbols: {e}")
            print(e)
            symbols = []
        return symbols
    
    @st.cache(allow_output_mutation=True)
    def get_all_watchlists(self):
        all_watchlists = rh.get_all_watchlists()
        return all_watchlists
    
    @st.cache(allow_output_mutation=True)
    def get_all_unique_symbols(self):
        all_unique_symbols = set()
        try:
            all_watchlists = rh.get_all_watchlists()
            for watchlist in all_watchlists['results']:
                watchlist_name = watchlist['display_name']
                instruments = self.get_watchlist_by_name(watchlist_name)
                all_unique_symbols.update(instruments)
        except Exception as e:
            logging.error(f"Error getting all unique symbols: {e}")
            print(e)
        return all_unique_symbols
    
    def get_portfolio_value(self):
        try:
            portfolio_info = self.rh.profiles.load_portfolio_profile()
            market_value = portfolio_info['market_value']
            return market_value
        except Exception as e:
            logging.error(f"Error getting portfolio value: {e}")
            print(e)
            return None

    def get_holdings_info(self):
        try:
            holdings_info = self.rh.account.build_holdings()
            return holdings_info
        except Exception as e:
            logging.error(f"Error getting holdings info: {e}")
            print(e)
            return None
        
    def get_option_calls(self):
        try:
            url = 'https://api.robinhood.com/options/instruments/'
            payload = {'type': 'call'}
            option_calls_data = self.rh.helper.request_get(url, 'regular', payload)
            return option_calls_data
        except Exception as e:
            logging.error(f"Error getting option calls: {e}")
            print(e)
            return None
        
    def get_my_option_positions(self):
        try:
            url = 'https://api.robinhood.com/options/positions/'
            option_positions_data = self.rh.helper.request_get(url, 'results')
            return option_positions_data
        except Exception as e:
            logging.error(f"Error getting option positions: {e}")
            print(e)
            return None
        
    def get_option_instrument_info(self, option_url):
        try:
            option_instrument_data = self.rh.helper.request_get(option_url, 'regular')
            return option_instrument_data
        except Exception as e:
            logging.error(f"Error getting option instrument info: {e}")
            print(e)
            return None

    
if __name__ == "__main__":
    manager = RobinhoodManager(username='manojbhat09@gmail.com', password='MENkeys796@09@')
    watchlist = manager.get_watchlist_by_name("Tech")
    print(watchlist)  # Should print the symbols in the Tech watchlist