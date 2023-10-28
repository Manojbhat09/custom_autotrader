
import robin_stocks.robinhood as rh
import streamlit as st
import logging 

class RobinhoodManager():
    def __init__(self, username, password):
        rh.login(username=username, password=password)

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
    
if __name__ == "__main__":
    manager = RobinhoodManager(username='manojbhat09@gmail.com', password='MENkeys796@09@')
    watchlist = manager.get_watchlist_by_name("Tech")
    print(watchlist)  # Should print the symbols in the Tech watchlist