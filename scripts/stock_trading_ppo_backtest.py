import os
import logging
from datetime import datetime
import pandas as pd
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

def create_run_directory(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp

def get_latest_run_directory(base_dir):
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    latest_run_dir = max(run_dirs, key=lambda d: datetime.strptime(d, "%Y%m%d_%H%M%S"))
    return os.path.join(base_dir, latest_run_dir)

def backtest_script(account_value_path, actions_path, stock_prices_path, backtest_dir):
    # Set up logging
    logging.basicConfig(filename=os.path.join(backtest_dir, 'backtest.log'), level=logging.INFO)
    
    # Load the necessary data
    df_account_value_td3 = pd.read_csv(account_value_path)
    df_actions_td3 = pd.read_csv(actions_path)
    stock_prices = pd.read_csv(stock_prices_path)

    # Calculate and display backtest statistics
    backtest_result = backtest_stats(df_account_value_td3)
    logging.info(backtest_result)

    # Plot the portfolio value over time
    plot_path = os.path.join(backtest_dir, 'backtest_plot.png')
    backtest_plot(df_account_value_td3, df_actions_td3, filename=plot_path)

    # Calculate daily returns
    daily_returns = get_daily_return(df_account_value_td3)
    logging.info(daily_returns.head())

    # Calculate and display baseline performance for comparison
    baseline_result = get_baseline(stock_prices)
    logging.info(baseline_result)

if __name__ == "__main__":
    base_dir = 'runs'
    latest_run_dir = get_latest_run_directory(base_dir)
    backtest_dir = os.path.join(latest_run_dir, 'backtest')
    os.makedirs(backtest_dir, exist_ok=True)
    
    account_value_path = os.path.join(latest_run_dir, 'results_finrl', 'df_account_value_td3.csv')
    actions_path = os.path.join(latest_run_dir, 'results_finrl', 'df_actions_td3.csv')
    stock_prices_path = os.path.join(latest_run_dir, 'results_finrl', 'stock_prices.csv')
    
    backtest_script(account_value_path, actions_path, stock_prices_path, backtest_dir)
