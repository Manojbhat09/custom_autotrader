import robin_stocks as r
import os
from robinhood_manager import RobinhoodManager
# Log in to Robinhood (replace with your username and password)
username, password = os.environ['RH_USERNAME'], os.environ['RH_PASSWORD']
rh = RobinhoodManager(username, password)

# Get the bid price, ask price, and the spread for a cryptocurrency (e.g., Bitcoin)
crypto_symbol = "BTC"
crypto_info = rh.rh.crypto.get_crypto_quote(crypto_symbol)

# Extracting bid price, ask price, and calculating the spread
bid_price = float(crypto_info['bid_price'])
ask_price = float(crypto_info['ask_price'])
spread = ask_price - bid_price
import pdb;pdb.set_trace()
print(bid_price, ask_price, spread)

