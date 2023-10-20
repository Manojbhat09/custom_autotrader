from finrl.config import ERL_PARAMS, INDICATORS
from finrl.meta.data_processor import DataProcessor
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.elegantrl.models import DRLAgent
# from finrl.agents.elegantrl.train.config import Arguments

# Initialize Alpaca API keys
API_KEY = "PKCADLJ1RCHS2XRN735C"
API_SECRET = "mifKUEOKW2tfu7l3YAf7uHASRFivTi7D73seDyLR"
API_BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize stock tickers and other parameters
ticker_list = ['AAPL', 'MSFT']
state_dim = 1 + 2 + 3 * len(ticker_list) + len(INDICATORS) * len(ticker_list)
action_dim = len(ticker_list)

# Fetch data
DP = DataProcessor(data_source='alpaca', API_KEY=API_KEY, API_SECRET=API_SECRET, API_BASE_URL=API_BASE_URL)
data = DP.download_data(start_date='2021-10-04', end_date='2021-10-08', ticker_list=ticker_list, time_interval='1Min')

# Clean and preprocess data
data = DP.clean_data(data)
data = DP.add_technical_indicator(data, INDICATORS)

data['turbulence'] = 0.0 
price_array, tech_array, turbulence_array= DP.df_to_array(data, if_vix=True) # turbulence_array 

# Initialize environment
env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "if_train": True,
    "turbulence_array": turbulence_array,
}

env_instance = StockTradingEnv(config=env_config)
# env_instance = StockTradingEnv(**env_config)

# Initialize DRL Agent and Model
agent = DRLAgent(env=env_instance, price_array=env_config['price_array'], tech_array=env_config['tech_array'], turbulence_array=env_config['turbulence_array'])
model = agent.get_model("ppo", model_kwargs=ERL_PARAMS)
import pdb; pdb.set_trace()
# Train the model
trained_model = agent.train_model(model=model, total_timesteps=1e5)

# Test the model (You can implement your own testing logic)
# For demonstration, we'll just print "Testing Complete"
print("Testing Complete")

# Your paper trading logic can go here
# ...

