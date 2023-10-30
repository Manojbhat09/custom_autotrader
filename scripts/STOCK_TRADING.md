In the provided code, the initial and final portfolio values are computed within the context of Mean-Variance Optimization (MVO) as well as using different reinforcement learning models. Here's how they are computed:

Initial Portfolio Value:
The initial portfolio value for MVO is computed using the optimized weights obtained from the EfficientFrontier module of the pypfopt library. The initial portfolio value for each asset is obtained by multiplying these weights with the inverse of the last known price of each asset:
ini
Copy code
LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
Final Portfolio Value:
The final portfolio value is computed using the trading data (TradeData) and the initial portfolio values (Initial_Portfolio). The dot product of TradeData and Initial_Portfolio gives the portfolio value over time, which is then saved in a DataFrame MVO_result:
ini
Copy code
Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
For reinforcement learning models, the final portfolio values are computed using the DRL_prediction method from the DRLAgent class. This method runs the trained models on the trading environment to obtain the account values and actions over time. For each model (A2C, DDPG, PPO, TD3, SAC), the final portfolio values are stored in DataFrames df_account_value_a2c, df_account_value_ddpg, df_account_value_ppo, df_account_value_td3, df_account_value_sac respectively. Here is an example with PPO:
markdown
Copy code
df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
            model=trained_moedl, environment=e_trade_gym
        )
Graphical Representation:
The code then plots the final portfolio values of PPO and MVO over time in a graph, and saves the plot as plot4.png:
sql
Copy code
result = pd.merge(df_result_ppo, MVO_result, left_index=True, right_index=True)
result.columns = ["ppo", "mean var"]
plt.figure()
ax = result.plot()
fig = ax.get_figure()
fig.savefig('plot4.png')
Note: There are commented-out sections of code that seem to be intended for plotting the final portfolio values of all models alongside the MVO result, but they are not executed in the provided code.
This way, the code is set up to compute, compare, and visualize the initial and final portfolio values using both MVO and reinforcement learning approaches.

2404 chars / 322 words
Copy


User
Where to get these 


print(f"Expected Return: {MVO_result['expected_return']}")
    print(f"Portfolio Risk (Variance): {MVO_result['portfolio_risk']}")
    print(f"Sharpe Ratio: {MVO_result['sharpe_ratio']}")

MVO_result just has mean var

The code snippet you've provided attempts to print out the expected return, portfolio risk (variance), and Sharpe ratio from a dictionary or DataFrame named MVO_result. However, based on the code you've shared, MVO_result only contains the portfolio values over time under the column name "Mean Var". The expected return, portfolio risk, and Sharpe ratio are not computed or stored in MVO_result within the provided code.

To obtain these metrics, you would need to compute them based on the portfolio values or returns. Here's how you might do it:

Expected Return: The expected return of the portfolio can be computed as the mean of the daily returns over the trading period.
ini
Copy code
daily_returns = MVO_result['Mean Var'].pct_change().dropna()
expected_return = daily_returns.mean()
Portfolio Risk (Variance): The portfolio risk can be computed as the variance of the daily returns over the trading period.
ini
Copy code
portfolio_risk = daily_returns.var()
Sharpe Ratio: The Sharpe ratio can be computed as the ratio of the expected return to the standard deviation of the daily returns, often annualized. If rf is the risk-free rate,
ini
Copy code
annualized_return = (1 + expected_return)**252 - 1
annualized_std = daily_returns.std() * np.sqrt(252)
sharpe_ratio = (annualized_return - rf) / annualized_std
Now you can print these computed metrics:

python
Copy code
print(f"Expected Return: {expected_return}")
print(f"Portfolio Risk (Variance): {portfolio_risk}")
print(f"Sharpe Ratio: {sharpe_ratio}")
Please adjust these computations to match any specific assumptions or methodologies you are adhering to in your analysis.





The code you provided is implementing a form of portfolio optimization using both traditional Mean-Variance Optimization (MVO) and Deep Reinforcement Learning (DRL) strategies. Let's break down the objectives and the outputs of both approaches to better understand what you might want to plot and analyze:

Mean-Variance Optimization (MVO):

Objective: To find the optimal allocation of assets in a portfolio such that for a given level of risk, the portfolio has the maximum expected return, or vice versa.
Output: The optimal weights of assets in the portfolio, and potentially, the portfolio's expected return, risk (variance), and Sharpe ratio.
Deep Reinforcement Learning (DRL):

Objective: To learn a policy for trading that maximizes some measure of cumulative return over time, based on the evolving state of the market and the portfolio.
Output: The actions taken by the agent (buy, hold, sell), and the resulting portfolio value over time.
What to Plot and Analyze:
Plotting Portfolio Value Over Time:

This is a common way to visualize and compare the performance of different trading strategies.
You can plot the portfolio value over time for each DRL model (A2C, DDPG, PPO, TD3, SAC) and compare it with the portfolio value over time of the MVO strategy (MVO_result['Mean Var']).
Plotting Asset Weights Over Time:

If you're interested in how the allocation of assets in the portfolio changes over time, you could plot the asset weights over time.
For the MVO strategy, the asset weights are fixed based on the optimization at the beginning (initial_portfolio), so you'd see a flat line for each asset.
For the DRL strategies, you'd need to extract the asset weights from the actions taken by the agent at each time step.
Comparing Expected Return, Risk, and Sharpe Ratio:

These metrics allow you to quantify and compare the performance of different strategies on a risk-adjusted basis.
You can compute these metrics for both the MVO and DRL strategies using the formulas in the previous message.
Conclusion:
By comparing the portfolio value over time and the risk-adjusted performance metrics, you can analyze how the DRL strategies compare to the traditional MVO strategy.
Whether you focus on MVO_result or initial_portfolio depends on whether you're more interested in the evolving portfolio value or the initial asset allocation.
Overall, you're analyzing both portfolio and trade performance - the portfolio value reflects the cumulative result of all trades, while the asset weights reflect the decisions made by the strategies at each time step.



In financial terms, an asset refers to a resource with economic value that an individual, corporation, or country owns or controls with the expectation that it will provide future benefit. In the context of your code and the stock market, assets typically refer to securities such as stocks or bonds. The term "asset allocation" refers to the strategy of dividing a portfolio's investments among different asset categories, such as stocks, bonds, and cash. The goal is to provide a desired risk and return profile based on an investor's objectives and risk tolerance.

In your code, the asset allocation is initially determined in two distinct ways:

Mean-Variance Optimization (MVO):

The initial allocation of assets under MVO is determined by optimizing the portfolio's expected return for a given level of risk, or vice versa. This optimization is performed using historical data to estimate the mean returns and covariance matrix of the assets' returns.
The code snippet provided uses a library function from pypfopt to find the optimal asset weights that maximize the Sharpe ratio, which is a measure of risk-adjusted return.
java
Copy code
from pypfopt.efficient_frontier import EfficientFrontier
ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(29)])
Here, mvo_weights represents the initial allocation of assets in the MVO portfolio. Each element of mvo_weights represents the amount of money allocated to a particular asset.

Deep Reinforcement Learning (DRL):

In DRL, the initial allocation of assets is not explicitly determined by an optimization procedure like in MVO. Instead, the DRL agent learns to make trading decisions (buy, hold, sell) over time through interaction with the environment.
The DRL models start with a specified initial amount of money, as set in the env_kwargs dictionary:
makefile
Copy code
env_kwargs = {
    "initial_amount": 1000000,
    ...
}
In summary, under MVO the initial asset allocation is determined by an optimization procedure based on historical data, while in DRL the asset allocation evolves over time as the agent learns to make trading decisions.




The objective of both Mean-Variance Optimization (MVO) and Deep Reinforcement Learning (DRL) in this context is to manage a portfolio in a way that maximizes returns while managing risk. However, they approach this goal from different perspectives and methodologies:

Mean-Variance Optimization (MVO):

MVO seeks to find the optimal static allocation of assets in a portfolio based on historical data to maximize the expected return for a given level of risk or to minimize the risk for a given level of expected return.
The result of MVO is a fixed set of portfolio weights for each asset, which does not change over time unless re-optimized with new data.
Deep Reinforcement Learning (DRL):

DRL, on the other hand, learns a policy for making trading decisions (buy, sell, hold) dynamically over time based on the evolving state of the market and the portfolio.
The DRL models aim to learn a strategy that adapts to new information and market conditions to achieve the goal of maximizing the total return over the trading period.
When you refer to evolving the portfolio value or initial asset allocation, it essentially means that the allocation of assets in the portfolio can change over time based on the decisions made by the DRL models.

Now, regarding the comparison of DRL and MVO on the plot:

It's not necessary that the DRL value matches the MVO curve. They may or may not align depending on the effectiveness of the learned policy by DRL models and the accuracy of the assumptions made by the MVO based on historical data.
What you might be interested in is comparing the performance of the DRL strategies with the MVO strategy to see which method is more effective in achieving higher returns or better risk-adjusted returns over the trading period.
You can compare the final portfolio values, the trajectory of portfolio values over time, or other performance metrics such as the Sharpe Ratio to evaluate and compare the performance of DRL and MVO.
In essence, the comparison can provide insights into whether a static optimization approach like MVO or a dynamic learning approach like DRL is more effective for portfolio management in the tested market conditions.