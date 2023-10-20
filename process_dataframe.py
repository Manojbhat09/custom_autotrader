import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_mse_for_ma(window_size, df):
    # import pdb; pdb.set_trace()
    df['MA'] = df['close_price'].rolling(window=window_size).mean()
    df['next_day_close'] = df['close_price'].shift(-1)
    mse = mean_squared_error(df['next_day_close'][:len(df['MA'].dropna())].dropna(), df['MA'].dropna())
    return mse

def compute_mse_for_macd(short_ema, long_ema, df):
    df['EMA_short'] = df['close_price'].ewm(span=short_ema, adjust=False).mean()
    df['EMA_long'] = df['close_price'].ewm(span=long_ema, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['next_day_macd'] = df['MACD'].shift(-1)
    mse = mean_squared_error(df['next_day_macd'].dropna(), df['MACD'][:len(df['next_day_macd'].dropna())].dropna())
    return mse


# Sample data
file_path = os.path.join("data", "raw", "TSLA_data_new.csv")
df = pd.read_csv(file_path)
df = pd.DataFrame(df)

# Grid search for Moving Average
ma_window_sizes = range(2, 21)  # e.g., check window sizes from 2 to 20
best_ma_window = min(ma_window_sizes, key=lambda x: compute_mse_for_ma(x, df.copy()))

# Grid search for MACD EMAs
ema_ranges = range(5, 30)
best_ema_pair = min(((i, j) for i in ema_ranges for j in ema_ranges if i < j), key=lambda x: compute_mse_for_macd(x[0], x[1], df.copy()))

print(f"Best window size for Moving Average: {best_ma_window}")
print(f"Best EMA pair for MACD: {best_ema_pair}")   

df['date'] = pd.to_datetime(df['date'])
# Moving Average
# df['MA_5'] = df['close_price'].rolling(window=5).mean()
df[f'MA_{best_ma_window}'] = df['close_price'].rolling(window=best_ma_window).mean()

# RSI
delta = df['close_price'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
df['EMA_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
df['MA_20'] = df['close_price'].rolling(window=20).mean()
df['BB_std'] = df['close_price'].rolling(window=20).std()
df['BB_upper'] = df['MA_20'] + (df['BB_std'] * 2)
df['BB_lower'] = df['MA_20'] - (df['BB_std'] * 2)

# Normalize data
df['normalized_close'] = (df['close_price'] - df['close_price'].min()) / (df['close_price'].max() - df['close_price'].min())

# Standardize data
df['standardized_close'] = (df['close_price'] - df['close_price'].mean()) / df['close_price'].std()

# Lag features
df['lag_1'] = df['close_price'].shift(1)
df['lag_2'] = df['close_price'].shift(2)

print(df)
# Save the dataframe to CSV
file_path = os.path.join("data", "processed", "TSLA.csv")
df.to_csv(file_path, index=False)


# -------------------------- Plots ---------------------------------

# Create a folder to save the plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Drop rows with NaN values
df_cleaned = df.dropna()

df_cleaned = df_cleaned.sort_values(by='begins_at')

# Ensure 'close_price' is in numeric format (convert if necessary)
# df_cleaned['close_price'] = pd.to_numeric(df_cleaned['close_price'], errors='coerce')
df_cleaned['index'] = np.arange(len(df_cleaned))
# Create a list of columns to plot

# Define the columns to plot in each group
columns_group1 = ['BB_upper', 'BB_lower', 'lag_1', 'lag_2']
columns_group2 = ['normalized_close', 'standardized_close', 'Signal_Line', 'MACD']

# Create two separate plots for the specified columns in each group
plt.figure(figsize=(12, 6))

# Plot columns in Group 1
for column in columns_group1:
    plt.plot(df.index, df[column], label=column)

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Set the x-axis tick format as integers
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

# Save and show the plot for Group 1
plt.savefig('plots/group1_values.png')  # Save the plot for Group 1
plt.show()

# Create a new figure for Group 2
plt.figure(figsize=(12, 6))

# Plot columns in Group 2
for column in columns_group2:
    plt.plot(df.index, df[column], label=column)

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Set the x-axis tick format as integers
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

# Save and show the plot for Group 2
plt.savefig('plots/group2_values.png')  # Save the plot for Group 2
plt.show()


'''
columns_to_plot = [ 'BB_upper', 'BB_lower', 'lag_1', 'lag_2'] #'normalized_close', 'standardized_close', 'Signal_Line','MACD',

plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

# Loop through the selected columns and plot each one
for column in columns_to_plot:
    plt.plot(df_cleaned['index'], df_cleaned[column], label=column)

# Add labels and a legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Set the x-axis tick format as integers
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

# Limit the x-axis range (adjust as needed)
plt.xlim(0, len(df_cleaned))

# Show the plot
plt.savefig('plots/selected_columns.png')  # Save the plot
plt.show()

'''


# # Plot all time series columns
# axes = df_cleaned[['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'MA_3', 'RSI', 'MA_20', 'BB_std', 'BB_upper', 'BB_lower', 'normalized_close', 'standardized_close', 'lag_1', 'lag_2']].plot(subplots=True, figsize=(12, 18))

# # Save each subplot to the 'plots' folder
# for i, ax in enumerate(axes):
#     ax.get_figure().savefig(f'plots/plot_{i}.png')

# # Show the plots
# plt.show()
