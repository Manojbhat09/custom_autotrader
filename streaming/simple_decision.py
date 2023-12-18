import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from talib import RSI, BBANDS, STOCH

# Load data
data = pickle.load(open("realtimedata", "rb"))
predictions = pickle.load(open("predictions2", "rb"))
realtime_data = pickle.load(open("realtimedata2", "rb"))
past_predictions = pickle.load(open("predictions", "rb"))


# Assuming 'Timestamp' and 'Value' are the columns in your data
df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df.rename(columns={'Value': 'close_price'}, inplace=True)

# Calculate Bollinger Bands, RSI, and Stochastic Oscillator
df['upper_band'], df['middle_band'], df['lower_band'] = BBANDS(df['close_price'], timeperiod=20)
df['rsi'] = RSI(df['close_price'], timeperiod=14)
df['stochastic_k'], df['stochastic_d'] = STOCH(df['close_price'], df['close_price'], df['close_price'], fastk_period=14, slowk_period=3, slowd_period=3)

# Define trading signals
df['buy_signal'] = np.where((df['rsi'] < 40) & (df['stochastic_k'] < 30), 1, 0)
df['sell_signal'] = np.where((df['rsi'] > 70) | (df['stochastic_k'] > 80), -1, 0)

# Execute Trades
initial_balance = 10000  # Adjust as needed
balance = initial_balance
positions = 0

balance_history = []

for index, row in df.iterrows():
    if row['buy_signal'] == 1 and balance > 0:
        positions = balance / row['close_price']
        balance = 0
    elif row['sell_signal'] == -1 and positions > 0:
        balance = positions * row['close_price']
        positions = 0

    current_balance = balance if positions == 0 else positions * row['close_price']
    balance_history.append(current_balance)

df['balance'] = balance_history

# Plotting Buy/Sell Signals and Balance
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close_price'], label='Close Price', color='blue')
plt.scatter(df[df['buy_signal'] == 1].index, df[df['buy_signal'] == 1]['close_price'], marker='^', color='green', label='Buy Signal')
plt.scatter(df[df['sell_signal'] == -1].index, df[df['sell_signal'] == -1]['close_price'], marker='v', color='red', label='Sell Signal')
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
plt.plot(df.index, df['close_price'], label='Close Price', color='blue')
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
plt.plot(df.index, df['close_price'], label='Close Price', color='blue')
plt.scatter(df[df['buy_signal'] == 1].index, df[df['buy_signal'] == 1]['close_price'], marker='^', color='green', label='Buy Signal')
plt.scatter(df[df['sell_signal'] == -1].index, df[df['sell_signal'] == -1]['close_price'], marker='v', color='red', label='Sell Signal')
plt.title('Buy/Sell Signals with Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('simple_decision_buy_sell_signals.png')





