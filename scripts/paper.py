import pandas as pd
from finrl.config import config
from finrl.marketdata.alpacaapi import AlpacaDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading_alpaca import AlpacaStockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline
import alpaca_trade_api as tradeapi

# Define your Alpaca API credentials
api = tradeapi.REST('<Alpaca-API-key>', '<Alpaca-Secret-key>', base_url='https://paper-api.alpaca.markets') 

# Use AlpacaDownloader to get data
data_df = AlpacaDownloader(
    start_date='2008-01-01',
    end_date='2021-01-01',
    ticker_list=['AAPL']
).fetch_data()

# Feature Engineering
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=False,
    user_defined_feature=False
)

processed = fe.preprocess_data(data_df)

# Splitting data
train = data_split(processed, '2009-01-01','2019-01-01')
trade = data_split(processed, '2019-01-01','2021-01-01')

# Create the environment
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "out_of_cash_penalty": 0,
    "reward_scaling": 1e-4,
    "state_space": 'auto',
    "stock_dim": 1,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": 3,
    "api": api  # Pass the Alpaca API object
}

e_train_gym = AlpacaStockTradingEnv(df=train, **env_kwargs)

# Training the DRL agent
agent = DRLAgent(env=e_train_gym)

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "verbose": 0
}

model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=50000)
