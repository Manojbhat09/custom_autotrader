import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, STOCH
import pickle

# Load historical data
realtime_data = pickle.load(open("realtimedata2", "rb"))
data = pickle.load(open("realtimedata", "rb"))
df = realtime_data.copy()
# Convert timestamp to datetime and set as index
realtime_data['Timestamp'] = pd.to_datetime(realtime_data['Timestamp'])
realtime_data.set_index('Timestamp', inplace=True)
realtime_data.rename(columns={'Value': 'Close'}, inplace=True)

# Initialize balance and positions
initial_balance = 10000
balance = initial_balance
positions = 0

# Initialize a DataFrame to store buy/sell signals and balance
realtime_data['buy_signal'] = 0
realtime_data['sell_signal'] = 0
realtime_data['balance'] = initial_balance

def plot_stochastic(df, num):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['stochastic_k'], label='%K', color='orange')
    plt.plot(df.index, df['stochastic_d'], label='%D', color='black')
    plt.axhline(80, color='red', linestyle='--')
    plt.axhline(20, color='green', linestyle='--')
    plt.title('Stochastic Oscillator')
    plt.xlabel('Date')
    plt.ylabel('Stochastic')
    plt.legend()
    plt.savefig(f'simple_decision_stochastic_oscillator_{num}.png')

def plot_rsi(df, num):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['rsi'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig(f'simple_decision_rsi_{num}.png')

def process_new_data(new_data_row, df, index):
    # Append new data row to the DataFrame
    df = df.append(new_data_row)
    
    # Calculate indicators with the new data
    df['upper_band'], df['middle_band'], df['lower_band'] = BBANDS(df['Close'], timeperiod=20)
    df['rsi'] = RSI(df['Close'], timeperiod=14)
    df['stochastic_k'], df['stochastic_d'] = STOCH(df['Close'], df['Close'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
    # plot_stochastic(df, index)
    # plot_rsi(df, index)
    
    # Get the latest data point
    last_row = df.iloc[-1]

    # Define trading signals based on the latest indicators
    buy_signal = (last_row['rsi'] < 70) & (last_row['stochastic_k'] < 20)
    sell_signal = (last_row['rsi'] > 30) | (last_row['stochastic_k'] > 60)
    df.at[last_row.name, 'buy_signal'] =  int(buy_signal)
    df.at[last_row.name, 'sell_signal'] = int(sell_signal)
    global balance, positions

    # Execute Trades based on signals
    if buy_signal and balance > 0:
        positions = balance / last_row['Close']
        balance = 0
        print(f"Bought at {last_row['Close']}")
    elif sell_signal and positions > 0:
        balance = positions * last_row['Close']
        positions = 0
        print(f"Sold at {last_row['Close']}")

    # Update balance
    df.at[last_row.name, 'balance'] = balance if positions == 0 else positions * last_row['Close']

    return df, balance, positions

# Simulate real-time data streaming
# This loop simulates the real-time aspect of the data streaming.
# In a real application, this would be replaced with a callback or a WebSocket connection
# that triggers when new data is received.
balance_history = []
# Initialize an empty DataFrame for processed data
processed_data = pd.DataFrame()
for index, new_data_row in realtime_data.iterrows():
    processed_data, balance, positions = process_new_data(new_data_row, processed_data, index )
    balance_history.append(balance)

# Print final balance and profit
# Calculate final balance and profit
final_balance = processed_data['balance'].iloc[-1]
profit_loss = final_balance - initial_balance
print(f'Initial Balance: ${initial_balance}')
print(f'Final Balance: ${final_balance}')
print(f'Profit/Loss: ${final_balance - initial_balance}')

df = processed_data.copy()

# Plotting Buy/Sell Signals and Balance
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.scatter(df[df['buy_signal'] == 1].index, df[df['buy_signal'] == 1]['Close'], marker='^', color='green', label='Buy Signal')
plt.scatter(df[df['sell_signal'] == -1].index, df[df['sell_signal'] == -1]['Close'], marker='v', color='red', label='Sell Signal')
plt.plot(df.index, df['balance'], label='Balance', color='orange')
plt.xlabel('Date')
plt.ylabel('Price / Balance')
plt.title('Trading Strategy with Buy/Sell Signals')
plt.legend()

# Calculate final balance
final_balance = df['balance'].iloc[-1]

print(f'Initial Balance: ${initial_balance:.2f}')
print(f'Final Balance: ${final_balance:.2f}')
print(f'Profit: ${final_balance - initial_balance:.2f}')


plt.savefig('simple_decision_initial.png')


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.plot(df.index, df['upper_band'], label='Upper Band', color='red')
plt.plot(df.index, df['middle_band'], label='Middle Band', color='green')
plt.plot(df.index, df['lower_band'], label='Lower Band', color='red')
plt.fill_between(df.index, df['upper_band'], df['lower_band'], color='red', alpha=0.1)
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('simple_decision_bollinger_bands.png')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['rsi'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.savefig('simple_decision_rsi.png')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['stochastic_k'], label='%K', color='orange')
plt.plot(df.index, df['stochastic_d'], label='%D', color='black')
plt.axhline(80, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.title('Stochastic Oscillator')
plt.xlabel('Date')
plt.ylabel('Stochastic')
plt.legend()
plt.savefig('simple_decision_stochastic_oscillator.png')


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.scatter(df[df['buy_signal'] == 1].index, df[df['buy_signal'] == 1]['Close'], marker='^', color='green', label='Buy Signal')
plt.scatter(df[df['sell_signal'] == -1].index, df[df['sell_signal'] == -1]['Close'], marker='v', color='red', label='Sell Signal')
plt.title('Buy/Sell Signals with Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('simple_decision_buy_sell_signals.png')





