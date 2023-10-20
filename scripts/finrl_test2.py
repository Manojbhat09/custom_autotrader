from __future__ import annotations

# Import necessary libraries and modules
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.elegantrl.models import DRLAgent
from finrl.config import INDICATORS
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# Define main function to encapsulate the workflow
def main():

    # Instantiate YahooDownloader and fetch data
    df = YahooDownloader(
        start_date='2009-01-01',
        end_date='2021-01-01',
        ticker_list=['AAPL']
    ).fetch_data()

    # Feature Engineering
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False
    )

    # Preprocess the data
    processed = fe.preprocess_data(df)

    # Split the data into training and trading datasets
    train = data_split(processed, '2009-01-01','2019-01-01')
    trade = data_split(processed, '2019-01-01','2021-01-01')

    # Define environment parameters
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    # Create the trading environment
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    # Custom Environment Class
    class MyStockTradingEnv(StockTradingEnv):
        def _reward(self):
            return self.state[0]*self.reward_scaling

    # Instantiate the agent
    agent = DRLAgent(env=e_train_gym)

    # Define PPO parameters and instantiate the model
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "verbose": 0
    }

    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    # Train the model
    trained_ppo = agent.train_model(
        model=model_ppo,
        tb_log_name='ppo',
        total_timesteps=50000
    )

# Entry point
if __name__ == "__main__":
    main()
