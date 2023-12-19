from robinhood_manager import RobinhoodManager
import robin_stocks.robinhood as rh
import os 
username, password = os.environ['RH_USERNAME'], os.environ['RH_PASSWORD']
manager = RobinhoodManager(username, password)

# Fetch the list of option positions
option_positions_data = manager.get_my_option_positions()
import pdb; pdb.set_trace()
# Assuming there's at least one option position in the list
if option_positions_data:
    # Get the URL of the first option instrument
    option_url = option_positions_data[0]['option']

    # Fetch detailed information about the first option instrument
    option_instrument_info = manager.get_option_instrument_info(option_url)
    print(f'Option Instrument Info: {option_instrument_info}')
else:
    print('No option positions found.')