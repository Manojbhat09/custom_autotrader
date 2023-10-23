from __future__ import annotations
import os
from datetime import datetime
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from scipy import optimize
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt
from finrl import config
from finrl import config_tickers

def create_run_directory(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    TRAIN_START_DATE = "2009-01-01"
    TRAIN_END_DATE = "2020-07-01"
    TRADE_START_DATE = "2020-07-01"
    TRADE_END_DATE = "2021-10-31"

    run_dir = create_run_directory('runs_td3')
    log_dir = os.path.join(run_dir, 'logs_finrl')
    os.makedirs(log_dir, exist_ok=True)
    results_dir = os.path.join(run_dir, 'results_finrl')
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, 'plots_finrl')
    os.makedirs(plot_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, 'models_finrl')
    os.makedirs(model_dir, exist_ok=True)

    df = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=config_tickers.DOW_30_TICKER,
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    train = data_split(processed, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed, TRADE_START_DATE, TRADE_END_DATE)

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(config.INDICATORS) * stock_dimension
    env_kwargs = {
        # Other environment arguments
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)
    TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
    
    tmp_path = config.RESULTS_DIR + "/td3"
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_td3.set_logger(new_logger_td3)
    trained_td3 = agent.train_model(model=model_td3, tb_log_name="td3", total_timesteps=500000)
    trained_td3.save(os.path.join(model_dir, 'model_td3'))

    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
        model=trained_td3, environment=e_trade_gym
    )
    df_account_value_td3.to_csv(os.path.join(results_dir, 'df_account_value_td3.csv'), index=False)
    df_actions_td3.to_csv(os.path.join(results_dir, 'df_actions_td3.csv'), index=False)

    # MVO and Stock Prices saving
    mvo_df = processed.sort_values(["date", "tic"], ignore_index=True)[["date", "tic", "close"]]
    mvo_df.to_csv(os.path.join(results_dir, 'mvo.csv'), index=False)
    stock_prices_df = trade[["date", "tic", "close"]]
    stock_prices_df.to_csv(os.path.join(results_dir, 'stock_prices.csv'), index=False)

    # Merge the DataFrames on the date index
    merged_df = df_account_value_td3.merge(stock_prices_df, on='date').merge(mvo_df, on='date')

    # Set the date as the index for plotting
    merged_df.set_index('date', inplace=True)

    # Plot the data
    plot_path = os.path.join(plot_dir, 'td3_stock_MVO_plot.png')  # Adjusted to save plot in plot_dir
    plt.figure(figsize=(15, 5))
    ax = merged_df.plot()
    ax.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(plot_path)
    
if __name__ == "__main__":
    main()
