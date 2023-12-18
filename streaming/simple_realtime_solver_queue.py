import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, STOCH
import pickle
import math

# Define the Solver class here

class Solver:
    def __init__(self, window_size):
        self.window_size = window_size
        self.prices = []
        self.fullprices = []

    def update_prices(self, new_price, index, idx):
        self.fullprices.append((new_price, index, idx))
        self.prices.append(new_price)
        if len(self.prices) > self.window_size:
            self.prices.pop(0)
            self.fullprices.pop(0)

    def gen_series(self, prices_lists, k):
        result_list = []
        for prices in prices_lists:
            result_tuple = self.gen_transactions(k, prices)
            result_list.append(result_tuple)
        return result_tuple

    def gen_transactions(self, k) -> int:
        if len(self.prices) < self.window_size:
            return [0, []]
        assert isinstance(self.prices, (list, np.ndarray)), f'Expected self.prices of type list or numpy.ndarray, got {type(self.prices)}'
        assert isinstance(k, int) and k >= 0, f'Expected k to be a non-negative integer, got {k}'

        n = len(self.prices)
        
        # solve special cases
        if not n or k == 0:
            return 0, []

        # find all consecutively increasing subsequence
        transactions = []
        start = 0
        end = 0
        for i in range(1, n):
            if self.prices[i] >= self.prices[i-1]:
                end = i
            else:
                if end > start:
                    transactions.append([start, end])
                start = i
        if end > start:
            transactions.append([start, end])

        while len(transactions) > k:
            # check delete loss
            delete_index = 0
            min_delete_loss = math.inf
            for i in range(len(transactions)):
                t = transactions[i]
                profit_loss = self.prices[t[1]] - self.prices[t[0]]
                if profit_loss < min_delete_loss:
                    min_delete_loss = profit_loss
                    delete_index = i

            # check merge loss
            merge_index = 0
            min_merge_loss = math.inf
            for i in range(1, len(transactions)):
                t1 = transactions[i-1]
                t2 = transactions[i]
                profit_loss = self.prices[t1[1]] - self.prices[t2[0]]
                if profit_loss < min_merge_loss:
                    min_merge_loss = profit_loss
                    merge_index = i

            # delete or merge
            if min_delete_loss <= min_merge_loss:
                transactions.pop(delete_index)
            else:
                transactions[merge_index - 1][1] = transactions[merge_index][1]
                transactions.pop(merge_index)

        return sum(self.prices[j]-self.prices[i] for i, j in transactions), transactions
    
# Load data
realtime_data = pickle.load(open("realtimedata2", "rb"))

# Convert timestamp to datetime and set as index
realtime_data['Timestamp'] = pd.to_datetime(realtime_data['Timestamp'])
realtime_data.set_index('Timestamp', inplace=True)

# Initialize balance and positions
initial_balance = 10000
balance = initial_balance
positions = 0

# Initialize a DataFrame to store buy/sell signals and balance
realtime_data['buy_signal'] = 0
realtime_data['sell_signal'] = 0
realtime_data['balance'] = initial_balance

# Initialize an empty DataFrame for processed data
processed_data = pd.DataFrame(columns=realtime_data.columns)

# Instantiate a Solver
# solver = Solver()

# Number of transactions we want to allow
k = 2
solver = Solver(window_size=29)  # Initialize the solver with desired window size
for idx, (index, new_data_row) in enumerate(realtime_data.iterrows()):
    processed_data = processed_data.append(new_data_row)
    
    solver.update_prices(new_data_row['Close'], index, idx)
    total_profit, transactions = solver.gen_transactions(k)
    print(f"[tick] current indx {idx} {index}" )
    print(f"total profit found: {total_profit}")
    # we are getting k transactions for the given window of time based price data. 
    # scan through the transactions list to find the the buy and sell signal, but these are past timestamp signals
    for transaction in transactions:
        # Check the type of transaction: buy or sell
        if transaction[1] > transaction[0]:  # This means a buy then sell
            buy_price = processed_data.iloc[transaction[0]]['Close']
            sell_price = processed_data.iloc[transaction[1]]['Close']

            buy_price, buy_time, buy_idx = solver.fullprices[transaction[0]]
            sell_price, sell_time, sell_idx = solver.fullprices[transaction[1]]

            if balance > 0:  # Execute buy
                processed_data.at[buy_time, 'buy_signal'] += 1
                positions = balance / buy_price
                balance = 0
                print(f"Bought at {buy_price} {buy_time} {buy_idx}")
            if positions > 0:  # Execute sell
                processed_data.at[sell_time, 'sell_signal'] += 1
                balance = positions * sell_price
                positions = 0
                print(f"Sold at {sell_price} {sell_time} {sell_idx}")
    print(f"[end] balance is {balance} {positions}")

    # given the past signals and profit and transactions buy and sell prices, 
    # derive a method to excecute a decision to buy, sell or 
    # hold at the current timestamp
    # TODO

    # Update balance history for each step
    processed_data.at[new_data_row.name, 'balance'] = balance if positions == 0 else positions * processed_data.iloc[-1]['Close']

# Final balance and profit calculations
final_balance = processed_data['balance'].iloc[-1]
profit_loss = final_balance - initial_balance

print(f'Initial Balance: ${initial_balance}')
print(f'Final Balance: ${final_balance}')
print(f'Profit/Loss: ${profit_loss}')

df = processed_data.copy()

def plot_signals(df, num):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.scatter(df[df['buy_signal'] > 0].index, df[df['buy_signal'] > 0]['Close'], marker='^', color='green', label='Buy Signal')
    plt.scatter(df[df['sell_signal'] > 0].index, df[df['sell_signal'] > 0]['Close'], marker='v', color='red', label='Sell Signal')
    plt.title('Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_buysell_{num}.png')

def plot_balance(df, num):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['balance'], label='Balance', color='orange')
    plt.title('Balance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_balanceovertime_{num}.png')

def plot_signals_size(df, num):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')

    # Filter the DataFrame to only include rows with buy or sell signals
    buy_df = df[df['buy_signal'] > 0]
    sell_df = df[df['sell_signal'] > 0]

    # Adjust marker size based on the number of signals
    buy_marker_size = buy_df['buy_signal'] * 20  # Adjust 20 to scale the size as needed
    sell_marker_size = sell_df['sell_signal'] * 20  # Adjust 20 to scale the size as needed

    plt.scatter(buy_df.index, buy_df['Close'], s=buy_marker_size, color='green', label='Buy Signal', alpha=0.6)
    plt.scatter(sell_df.index, sell_df['Close'], s=sell_marker_size, color='red', label='Sell Signal', alpha=0.6)

    plt.title('Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_buysellsize_{num}.png')
    
def plot_transactions(df, num):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.scatter(df[df['buy_signal'] > 0].index, df[df['buy_signal'] > 0]['Close'], marker='^', color='green', label='Buy Signal')
    plt.scatter(df[df['sell_signal'] > 0].index, df[df['sell_signal'] > 0]['Close'], marker='v', color='red', label='Sell Signal')
    plt.title('Transaction Points on Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_transactions_{num}.png')

def plot_drawdown(df, num):
    rolling_max = df['balance'].cummax()
    drawdown = df['balance'] - rolling_max
    drawdown_percent = drawdown / rolling_max * 100

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, drawdown_percent, label='Drawdown', color='brown')
    plt.title('Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_drawdown_{num}.png')

def plot_histogram_of_returns(df, num):
    df['return'] = df['balance'].pct_change()
    plt.figure(figsize=(12, 6))
    df['return'].hist(bins=50, color='gray')
    plt.title('Histogram of Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig(f'simple_realtime_solver_returns_histogram_{num}.png')

def plot_cumulative_profit_loss(df, num):
    df['cumulative_profit_loss'] = df['balance'] - initial_balance
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_profit_loss'], label='Cumulative Profit/Loss', color='purple')
    plt.title('Cumulative Profit/Loss Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit/Loss')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_cumulative_profit_loss_{num}.png')


plot_signals(df, 0)
plot_balance(df, 0)
# plot_signals_size(df, 0)
plot_cumulative_profit_loss(df, 1)
plot_histogram_of_returns(df, 1)
plot_drawdown(df, 1)
plot_transactions(df, 1)