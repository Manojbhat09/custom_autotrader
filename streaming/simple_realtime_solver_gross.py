
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, STOCH
import pickle
import math
import tqdm
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

# Initialize a DataFrame to store buy/sell signals and balance
initial_balance = 10000
realtime_data['buy_signal'] = 0
realtime_data['sell_signal'] = 0
realtime_data['balance'] = initial_balance

# Initialize an empty DataFrame for processed data
processed_data = pd.DataFrame(columns=realtime_data.columns)

# Instantiate a Solver
# solver = Solver()



def trade_realtime(processed_data, window_size):
    global realtime_data
    # Initialize balance and positions
    initial_balance = 10000
    balance = initial_balance
    positions = 0
        
    # Number of transactions we want to allow
    k = 2
    solver = Solver(window_size=window_size)  # Initialize the solver with desired window size
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
    return final_balance, profit_loss

# final_balance, profit_loss = trade_realtime(processed_data, window_size=20)
gross_balance, gross_profit_loss = [], []
range_start, range_end = 5, 55
for i in tqdm.tqdm(range(range_start, range_end)):
    processed_data = pd.DataFrame(columns=realtime_data.columns)
    final_balance, profit_loss = trade_realtime(processed_data, window_size=i)
    gross_balance.append(final_balance)
    gross_profit_loss.append(profit_loss)



# Find the window size that yielded the maximum final balance and profit/loss
max_balance_idx = gross_balance.index(max(gross_balance))
max_profit_loss_idx = gross_profit_loss.index(max(gross_profit_loss))

best_window_size_balance = range_start + max_balance_idx
best_window_size_profit_loss = range_start + max_profit_loss_idx

print(f"Best window size for max balance: {best_window_size_balance}")
print(f"Best window size for max profit/loss: {best_window_size_profit_loss}")

df = processed_data.copy()


def plot_gross_balance(gross_balance, xaxis, num):
    plt.figure(figsize=(12, 6))
    plt.plot(xaxis, gross_balance, label='Balance at window size', color='purple')
    plt.title('Balance at window size')
    plt.xlabel('window size')
    plt.ylabel('Balance')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_gross_balance_curve_{num}.png')

def plot_gross_pl(gross_balance, xaxis, num):
    plt.figure(figsize=(12, 6))
    plt.plot(xaxis, gross_balance, label='profit/loss at window size', color='purple')
    plt.title('profit_loss at window size')
    plt.xlabel('window size')
    plt.ylabel('profit_loss')
    plt.legend()
    plt.savefig(f'simple_realtime_solver_gross_profit_loss_curve_{num}.png')

plot_gross_balance(gross_balance, range(range_start, range_end), 0)
plot_gross_pl(gross_profit_loss, range(range_start, range_end), 0)
exit()
