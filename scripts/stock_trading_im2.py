import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
from scipy import optimize
from scipy.optimize import linprog
from pypfopt.efficient_frontier import EfficientFrontier
from finrl import config
from finrl import config_tickers
from finrl.main import check_and_make_directories
from stable_baselines3.common.callbacks import BaseCallback
import logging
import itertools

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = False
if_using_td3 = True
if_using_sac = True

def create_run_directory(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_results(df_account_value, df_actions, results_dir):
    df_account_value.to_csv(os.path.join(results_dir, 'df_account_value.csv'), index=False)
    df_actions.to_csv(os.path.join(results_dir, 'df_actions.csv'), index=False)

def plot_actions(df_actions, plot_dir):
    sns.countplot(data=df_actions, x='actions')
    plt.savefig(os.path.join(plot_dir, 'actions_plot.png'))
    plt.close()

def train_save_plot_model(agent, model_name, model_dir, plot_dir, log_dir, results_dir, env_train, total_timesteps=1000):
    model = agent.get_model(model_name)
    
    # Set up logger
    tmp_path = results_dir + f"/{model_name}"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Train model with custom callback
    trained_model = agent.train_model(model=model, tb_log_name=model_name, total_timesteps=total_timesteps) # callback=callback)
    
    # Save model
    trained_model.save(os.path.join(model_dir, f'model_{model_name}'))
    
    # Plot actions (if the model has been trained successfully)
    try:
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=env_train)
        plot_actions(df_actions, plot_dir)
    except Exception as e:
        print(f"Error plotting actions for {model_name}: {e}")
    
    return trained_model

def setup_directories():
    run_dir = create_run_directory('runs')
    log_dir = os.path.join(run_dir, 'logs_finrl')
    os.makedirs(log_dir, exist_ok=True)
    results_dir = os.path.join(run_dir, 'results_finrl')
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, 'plots_finrl')
    os.makedirs(plot_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, 'models_finrl')
    os.makedirs(model_dir, exist_ok=True)
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, results_dir]
    )
    return run_dir, log_dir, results_dir, plot_dir, model_dir

def fetch_and_preprocess_data(TRAIN_START_DATE, TRADE_END_DATE):
    df = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=config_tickers.DOW_30_TICKER,
    ).fetch_data()
    processed = preprocess_data(df)
    return df, processed

# Define the setup_logging function
def setup_logging(log_dir):
    logging.basicConfig(filename=os.path.join(log_dir, 'script.log'), level=logging.INFO)

def preprocess_data(df):
    df.sort_values(["date", "tic"], ignore_index=True).head()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])

    processed_full = processed_full.fillna(0)

    processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

    return processed_full

def data_split_wrap(processed_full, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE):
    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    return train, trade

def setup_environment(initial_amount, reward_scaling, train, processed_full, technical_indicators_list=None ):
    
    # Configurations for the trading environment
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(technical_indicators_list or INDICATORS) * stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": technical_indicators_list or INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling,
    }

    # e_train_gym = StockTradingEnv(df=processed_full, **env_kwargs)
    # env_train, _ = e_train_gym.get_sb_env()
    # import pdb; pdb.set_trace()
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    # Advanced setup (e.g., custom reward function, data normalization, etc.)
    # ... (Any additional advanced setup can go here)

    return env_kwargs, env_train


# function obtains maximal return portfolio using linear programming
def MaximizeReturns(MeanReturns, PortfolioSize):
    # dependencies

    c = np.multiply(-1, MeanReturns)
    A = np.ones([PortfolioSize, 1]).T
    b = [1]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method="simplex")

    return res

def MinimizeRisk(CovarReturns, PortfolioSize):
    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T) - b
        return constraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = {"type": "eq", "fun": constraintEq}
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(
        f,
        x0=xinit,
        args=(CovarReturns),
        bounds=bnds,
        constraints=cons,
        tol=10**-3,
    )

    return opt

def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        AEq = np.ones(x.shape)
        bEq = 1
        EqconstraintVal = np.matmul(AEq, x.T) - bEq
        return EqconstraintVal

    def constraintIneq(x, MeanReturns, R):
        AIneq = np.array(MeanReturns)
        bIneq = R
        IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
        return IneqconstraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = (
        {"type": "eq", "fun": constraintEq},
        {"type": "ineq", "fun": constraintIneq, "args": (MeanReturns, R)},
    )
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(
        f,
        args=(CovarReturns),
        method="trust-constr",
        x0=xinit,
        bounds=bnds,
        constraints=cons,
        tol=10**-3,
    )

    return opt

def StockReturnsComputing(StockPrice, Rows, Columns):
    import numpy as np

    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = (
                (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
            ) * 100

    return StockReturn

def mean_variance_optimization(processed_full, train, trade):
    # Obtain optimal portfolio sets that maximize return and minimize risk
    mvo_df = processed_full.sort_values(["date", "tic"], ignore_index=True)[
        ["date", "tic", "close"]
    ]

    fst = mvo_df
    fst = fst.iloc[0 * 29 : 0 * 29 + 29, :]
    tic = fst["tic"].tolist()

    mvo = pd.DataFrame()
    
    for k in range(len(tic)):
        mvo[tic[k]] = 0

    for i in range(mvo_df.shape[0] // 29):
        n = mvo_df
        n = n.iloc[i * 29 : i * 29 + 29, :]
        date = n["date"][i * 29]
        mvo.loc[date] = n["close"].tolist()

    # input k-portfolio 1 dataset comprising 15 stocks
    # StockFileName = './DJIA_Apr112014_Apr112019_kpf1.csv'

    Rows = 1259  # excluding header
    Columns = 15  # excluding date
    portfolioSize = 29  # set portfolio size

    # read stock prices in a dataframe
    # df = pd.read_csv(StockFileName,  nrows= Rows)

    # extract asset labels
    # assetLabels = df.columns[1:Columns+1].tolist()
    # print(assetLabels)

    # extract asset prices
    # StockData = df.iloc[0:, 1:]
    StockData = mvo.head(mvo.shape[0] - 336)
    TradeData = mvo.tail(336)
    # df.head()
    TradeData.to_numpy()

    # compute asset returns
    arStockPrices = np.asarray(StockData)
    [Rows, Cols] = arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    # compute mean returns and variance covariance matrix of returns
    meanReturns = np.mean(arReturns, axis=0)
    covReturns = np.cov(arReturns, rowvar=False)

    # set precision for printing results
    np.set_printoptions(precision=3, suppress=True)

    # display mean returns and variance-covariance matrix of returns
    print("Mean returns of assets in k-portfolio 1\n", meanReturns)
    print("Variance-Covariance matrix of returns\n", covReturns)

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(29)])
    mvo_weights

    LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
    Initial_Portfolio

    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
    MVO_result

    return mvo_weights, MVO_result, Portfolio_Assets, Initial_Portfolio, Portfolio_Assets

# Define the three metric plotting functions
def plot_rolling_sharpe(df_account_value, plot_dir):
    rolling_window = 252  # Define the rolling window size (e.g., 252 trading days in a year)
    df_account_value['rolling_sharpe'] = (
        (df_account_value['daily_return'].rolling(rolling_window).mean()) / 
        (df_account_value['daily_return'].rolling(rolling_window).std())
    )
    df_account_value['rolling_sharpe'].plot(title='Rolling Sharpe Ratio')
    plt.savefig('rolling_sharpe_plot.png')  # Save the plot
    plt.close()

def plot_max_drawdown(df_account_value, plot_dir):
    cumulative_returns = (1 + df_account_value['daily_return']).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max

    plt.figure()
    plt.plot(cumulative_returns, label='Cumulative Returns')
    plt.plot(rolling_max, label='Rolling Max')
    plt.plot(drawdown, label='Drawdown')
    plt.legend()
    plt.title('Maximum Drawdown Over Time')
    plt.savefig('max_drawdown_plot.png')  # Save the plot
    plt.close()

def plot_rolling_volatility(df_account_value, plot_dir):
    rolling_window = 252  # Define the rolling window size (e.g., 252 trading days in a year)
    df_account_value['rolling_volatility'] = (
        df_account_value['daily_return'].rolling(rolling_window).std() * (252 ** 0.5)
    )
    df_account_value['rolling_volatility'].plot(title='Rolling Volatility')
    plt.savefig('rolling_volatility_plot.png')  # Save the plot
    plt.close()

# Define the function to call the above three functions
def plot_metrics_over_time(df_account_value, plot_dir):
    plot_rolling_sharpe(df_account_value, plot_dir)
    plot_max_drawdown(df_account_value, plot_dir)
    plot_rolling_volatility(df_account_value, plot_dir)

def calculate_metrics(df_account_value):
    """
    Calculate performance metrics for a trading strategy.

    Args:
        df_account_value (pd.DataFrame): DataFrame containing account values over time.

    Returns:
        dict: Dictionary containing calculated performance metrics.
    """
    # Calculate daily returns
    df_account_value['daily_return'] = df_account_value['total_assets'].pct_change()

    # Calculate annualized Sharpe ratio (assuming 252 trading days in a year)
    sharpe_ratio = (252 ** 0.5) * df_account_value['daily_return'].mean() / df_account_value['daily_return'].std()

    # Calculate Compound Annual Growth Rate (CAGR)
    start_value = df_account_value['total_assets'].iloc[0]
    end_value = df_account_value['total_assets'].iloc[-1]
    num_years = len(df_account_value) / 252  # Assuming 252 trading days in a year
    cagr = ((end_value / start_value) ** (1 / num_years)) - 1

    # Calculate Sortino Ratio
    negative_returns = df_account_value['daily_return'][df_account_value['daily_return'] < 0]
    sortino_ratio = (252 ** 0.5) * df_account_value['daily_return'].mean() / negative_returns.std()

    # Calculate Maximum Drawdown
    cumulative_returns = (1 + df_account_value['daily_return']).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calculate annualized Volatility
    volatility = df_account_value['daily_return'].std() * np.sqrt(252)

    # Create a dictionary to store the calculated metrics
    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'CAGR': cagr,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility,
    }

    return metrics

def evaluate_models(trained_models, e_trade_gym, env_kwargs, MVO_result, plot_dir):
    models_results = {}
    df_account_values = {}
    for model_name, trained_model in trained_models.items():
        try:
            # df_account_value, df_actions = DRLAgent.DRL_prediction(
            #     model=trained_model, environment=trade, **env_kwargs
            # )
            df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained_model, environment=e_trade_gym, 
            )
            
            # Save results
            save_results(df_account_value, df_actions, os.path.join(plot_dir, model_name))
            
            # Plot actions
            plot_actions(df_actions, os.path.join(plot_dir, model_name))
            plot_account_values_comparison(model_name, df_account_value, MVO_result, plot_dir)
            # Store df_account_value in the dictionary
            df_account_values[model_name] = df_account_value

            # Calculate performance metrics here (e.g., Sharpe ratio, CAGR, etc.)
            metrics = calculate_metrics(df_account_value)

            plot_metrics_over_time(df_account_value, plot_dir)

            # Add the calculated metrics to models_results dictionary
            models_results[model_name] = metrics
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            models_results[model_name] = None
    
    # Print the performance metrics for each model
    for model_name, metrics in models_results.items():
        print(f"Performance Metrics for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")
            
    # Plot all results
    plot_all_results(plot_dir, df_account_values, MVO_result)

    return models_results



def plot_account_values_comparison(model_name, df_account_value, MVO_result, plot_dir):
    # Create a new DataFrame with MVO result and the specified model's account value
    df_comparison = pd.concat([df_account_value, MVO_result], axis=1)
    df_comparison.columns = [model_name, 'MVO']

    # Plot the account values for the specified model and MVO
    plt.figure(figsize=(15, 5))
    plt.plot(df_comparison.index, df_comparison[model_name], label=model_name)
    plt.plot(df_comparison.index, df_comparison['MVO'], linestyle='--', label='MVO')

    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.title(f'Account Value Comparison with MVO for {model_name}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot to a unique file in the specified directory
    comparison_plot_path = os.path.join(plot_dir, f'{model_name}_account_value_comparison.png')
    plt.savefig(comparison_plot_path)
    plt.close()

def plot_all_results(plot_dir, df_account_values, MVO_result):
    # Combine df_account_value dataframes into one DataFrame
    result = pd.concat(df_account_values, axis=1)
    result.columns = df_account_values.keys()
    
    # Add MVO_result to the DataFrame
    result = pd.concat([result, MVO_result], axis=1)
    
    plot_path = os.path.join(plot_dir, 'all_models_plot.png')
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()
    ax = result.plot()
    ax.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig = ax.get_figure()
    fig.savefig(plot_path)

def save_mvo_stats(mvo_weights, Portfolio_Assets, MVO_result, initial_asset_prices, final_asset_prices):
    import pdb; pdb.set_trace()
    # # Example asset prices (replace with your actual data)
    # initial_asset_prices = [100.0, 150.0, 80.0, 120.0]  # Initial prices of assets
    # final_asset_prices = [110.0, 160.0, 85.0, 130.0]    # Final prices of assets

    # Calculate the initial portfolio value based on MVO weights and initial asset prices
    initial_portfolio_value = sum(mvo_weights * initial_asset_prices)

    # Calculate the final portfolio value based on MVO weights and final asset prices
    final_portfolio_value = sum(mvo_weights * final_asset_prices)

    # Calculate the portfolio returns over the trading period
    portfolio_returns = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value

    # Print key MVO statistics
    print("Mean Variance Optimization Results:")
    print(f"Expected Return: {MVO_result['expected_return']}")
    print(f"Portfolio Risk (Variance): {MVO_result['portfolio_risk']}")
    print(f"Sharpe Ratio: {MVO_result['sharpe_ratio']}")

    # Plot the portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(Portfolio_Assets.index, Portfolio_Assets.values, label="Portfolio Value")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.title("Portfolio Value Over Time")
    plt.show()

def main():
    # Constants
    TRAIN_START_DATE = "2009-01-01"
    TRAIN_END_DATE = "2020-07-01"
    TRADE_START_DATE = "2020-07-01"
    TRADE_END_DATE = "2021-10-31"

    # Set up directories
    run_dir, log_dir, results_dir, plot_dir, model_dir = setup_directories()
    
    # For logging
    setup_logging(log_dir)  # Call the setup_logging function here

    # Fetch and preprocess data
    df, processed_full = fetch_and_preprocess_data(TRAIN_START_DATE, TRADE_END_DATE)

    # Data split
    train, trade = data_split_wrap(processed_full, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE)

    # Example call with custom technical indicators
    custom_indicators = ["MACD", "RSI", "CCI"]
    hmax= 100
    initial_amount= 1000000
    reward_scaling= 1e-4
    env_kwargs, env_train = setup_environment(initial_amount, reward_scaling, train, processed_full, technical_indicators_list=INDICATORS)
    
    # Initialize DRL Agent
    agent = DRLAgent(env=env_train)

    # Mean Variance Optimization
    mvo_weights, MVO_result, Portfolio_Assets, inital_assets, final_assets=  mean_variance_optimization(processed_full, train, trade)
    save_mvo_stats(mvo_weights, Portfolio_Assets, MVO_result, inital_assets, final_assets)

    # Model training
    trained_models = {}
    if if_using_a2c:
        model_a2c = agent.get_model("a2c")
        trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=1000) 
        trained_a2c.save(os.path.join(model_dir, 'model_a2c'))  # save the final model
        trained_models.update({'a2c': trained_a2c})

    if if_using_ddpg:
        model_ddpg = agent.get_model("ddpg")
        trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=1000) 
        trained_ddpg.save(os.path.join(model_dir, 'model_ddpg'))  # save the final model
        trained_models.update({'ddpg': trained_ddpg})

    if if_using_ppo:
        PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

        # Set up logger for PPO
        tmp_path_ppo = os.path.join(results_dir, "ppo")
        new_logger_ppo = configure(tmp_path_ppo, ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger_ppo)

        # Train PPO model
        trained_ppo = agent.train_model(
            model=model_ppo,
            tb_log_name="ppo",
            total_timesteps=1000,  # You can adjust the total timesteps
        )

        # Save the trained PPO model
        trained_ppo.save(os.path.join(model_dir, 'model_ppo'))
        trained_models.update({'ppo': trained_ppo})

    # TD3 Model
    if if_using_td3:
        TD3_PARAMS = {
            "batch_size": 100,
            "buffer_size": 1000000,
            "learning_rate": 0.001,
        }
        model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

        # Set up logger for TD3
        tmp_path_td3 = os.path.join(RESULTS_DIR, "td3")
        new_logger_td3 = configure(tmp_path_td3, ["stdout", "csv", "tensorboard"])
        model_td3.set_logger(new_logger_td3)

        # Train TD3 model
        trained_td3 = agent.train_model(
            model=model_td3,
            tb_log_name="td3",
            total_timesteps=1000,  # You can adjust the total timesteps
        )

        # Save the trained TD3 model
        trained_td3.save(os.path.join(model_dir, 'model_td3'))
        trained_models.update({'td3': trained_td3})

    # SAC Model
    if if_using_sac:
        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }
        model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

        # Set up logger for SAC
        tmp_path_sac = os.path.join(RESULTS_DIR, "sac")
        new_logger_sac = configure(tmp_path_sac, ["stdout", "csv", "tensorboard"])
        model_sac.set_logger(new_logger_sac)

        # Train SAC model
        trained_sac = agent.train_model(
            model=model_sac,
            tb_log_name="sac",
            total_timesteps=10000,  # You can adjust the total timesteps
        )

        # Save the trained SAC model
        trained_sac.save(os.path.join(model_dir, 'model_sac'))
        trained_models.update({'sac': trained_sac})

    # DRL Prediction & Portfolio Evaluation
    e_trade_gym = StockTradingEnv(
        df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
    )
    import pdb; pdb.set_trace()
    models_results = evaluate_models(trained_models, trade, e_trade_gym, MVO_result, plot_dir)


if __name__ == "__main__":
    main()
