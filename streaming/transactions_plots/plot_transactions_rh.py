import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns

csv_path = "transactions_2023_robinhood.csv"
# Read the data into a pandas DataFrame
df = pd.read_csv(csv_path, error_bad_lines=False)
# import pdb; pdb.set_trace()

# Convert 'Activity Date' to datetime
df['Activity Date'] = pd.to_datetime(df['Activity Date'])

# Convert 'Price' and 'Quantity' to numeric, coercing errors
df['Price'] = pd.to_numeric(df['Price'].str.replace('[$,]', ''), errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Amount'] = pd.to_numeric(df['Amount'].str.replace('[$,()]', ''), errors='coerce')

# Replace NaN values with 0 for plotting purposes
df['Price'].fillna(0, inplace=True)
df['Quantity'].fillna(0, inplace=True)
df['Amount'].fillna(0, inplace=True)

def plot_transaction_analysis():
    # Calculate cumulative capital
    df['Cumulative Capital'] = df['Amount'].cumsum()

    # Separate buys and sells
    buys = df[df['Trans Code'] == 'BTO']
    sells = df[df['Trans Code'] == 'STC']

    # Profit/Loss per Instrument
    pnl_per_instrument = df.groupby('Instrument')['Amount'].sum()

    # Performance Over Time
    df['Realized P/L'] = df.apply(lambda x: x['Amount'] if x['Trans Code'] in ['STC', 'SLD'] else 0, axis=1)
    df['Cumulative Realized P/L'] = df['Realized P/L'].cumsum()

    # Transaction Volume
    transaction_volume = df['Instrument'].value_counts()

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(14, 20))

    # Plot buys and sells by date
    axs[0].scatter(buys['Activity Date'], buys['Price'] * buys['Quantity'], color='green', label='Buys')
    axs[0].scatter(sells['Activity Date'], sells['Price'] * sells['Quantity'], color='red', label='Sells')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Value of Transaction')
    axs[0].set_title('Options Bought and Sold Over Time')
    axs[0].legend()

    # Plot cumulative capital over time
    axs[1].plot(df['Activity Date'], df['Cumulative Capital'], color='blue', label='Cumulative Capital')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Cumulative Capital')
    axs[1].set_title('Total Capital Over Time')
    axs[1].legend()

    # Plot Profit/Loss per Instrument
    pnl_per_instrument.plot(kind='bar', ax=axs[2])
    axs[2].set_xlabel('Instrument')
    axs[2].set_ylabel('Profit/Loss')
    axs[2].set_title('Profit/Loss per Instrument')

    # Plot Transaction Volume
    transaction_volume.plot(kind='bar', ax=axs[3])
    axs[3].set_xlabel('Instrument')
    axs[3].set_ylabel('Volume')
    axs[3].set_title('Transaction Volume per Instrument')

    plt.tight_layout()
    plt.savefig('transactions_analysis_delete.png')
    plt.cla() 
    plt.clf()

def categorize_strategy(description):
    if isinstance(description, str):
        if 'Call' in description:
            return 'Call Option'
        elif 'Put' in description:
            return 'Put Option'
        else:
            return 'Equity'
    else:
        return 'Unknown'  # Or another appropriate category for non-string descriptions

def plot_strategy():
    df['Strategy'] = df['Description'].apply(categorize_strategy)

    # Continue with the groupby operation
    strategy_groups = df.groupby('Strategy')
    strategy_performance = strategy_groups['Amount'].sum().sort_values(ascending=False)

    # Plot strategy performance
    strategy_performance.plot(kind='bar')
    plt.title('Strategy Performance')
    plt.ylabel('Net Amount ($)')
    plt.xlabel('Strategy')
    plt.tight_layout()
    plt.savefig('strategy_performance_delete.png')

    plt.cla() 
    plt.clf()

def plot_daily_profit_loss():
    # Filter out only option transactions
    options_df = df[df['Description'].str.contains("Call|Put", na=False)]

    # Group by Activity Date and Trans Code to find net profit/loss per day for each transaction type
    daily_options_profit = options_df.groupby(['Activity Date', 'Trans Code'])['Amount'].sum().unstack().fillna(0)

    # Calculating the net profit/loss per day
    daily_options_profit['Net'] = daily_options_profit['STC'] + daily_options_profit['BTO']

    # Plotting the net profit/loss bar chart
    daily_options_profit['Net'].plot(kind='bar', figsize=(14, 7), color=np.where(daily_options_profit['Net']>=0, 'green', 'red'))
    plt.title('Daily Options Buy/Sell Profit/Loss')
    plt.xlabel('Activity Date')
    plt.ylabel('Net Profit/Loss ($)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('daily_options_profit_delete.png')

    plt.cla() 
    plt.clf()

def plot_pl_instrument():
    options_df = pd.DataFrame(df)
    options_df['Activity Date'] = pd.to_datetime(options_df['Activity Date'])

    # Assuming 'BTO' represents buying options and 'STC' represents selling options
    # Make 'BTO' values negative to reflect the cost of buying
    options_df['Amount'] = np.where(options_df['Trans Code'] == 'BTO', -options_df['Amount'], options_df['Amount'])

    # Group by Activity Date and Instrument to find net profit/loss per day for each Instrument
    daily_options_profit = options_df.groupby(['Activity Date', 'Instrument'])['Amount'].sum().unstack().fillna(0)

    # Assuming 'options_df' is your DataFrame containing options transactions
    # Here we create a unique color for each instrument using seaborn's color palette
    unique_instruments = options_df['Instrument'].unique()
    palette = sns.color_palette("hls", len(unique_instruments))
    colors = dict(zip(unique_instruments, palette))

    # Now we use this 'colors' dictionary to plot the bar chart
    ax = daily_options_profit.plot(
        kind='bar',
        stacked=True,
        color=[colors.get(name, 'grey') for name in daily_options_profit.columns],
        figsize=(14, 7)
    )

    # Customize plot
    ax.set_title('Daily Options Buy/Sell Profit/Loss by Instrument')
    ax.set_xlabel('Activity Date')
    ax.set_ylabel('Net Profit/Loss ($)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig('daily_options_profit_by_instrument_bar_delete.png')


    plt.cla() 
    plt.clf()

def plot_pl_instrument_2():
    # Create a pivot table with the sum of Amount for each Activity Date and Instrument
    pivot_df = df.pivot_table(index='Activity Date', columns='Instrument', values='Amount', aggfunc='sum')

    # Plotting
    plt.figure(figsize=(15, 8))
    for instrument in pivot_df.columns:
        # Get the data for this instrument
        instrument_data = pivot_df[instrument].dropna()
        # Plot the bars for each date
        bars = plt.bar(instrument_data.index, instrument_data,
                    label=instrument, alpha=0.5)

        # Adding labels to bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, 
                    f"{instrument}: {round(yval, 2)}", 
                    va='bottom' if yval > 0 else 'top', 
                    ha='center', rotation=90, fontsize=8)

    plt.title('Daily Options Buy/Sell Profit/Loss by Instrument')
    plt.xlabel('Activity Date')
    plt.ylabel('Net Profit/Loss ($)')
    plt.xticks(rotation=90)
    plt.legend(title='Instrument', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


    # Save the plot as a PNG file
    plt.savefig('daily_options_profit_loss_instrument_delete.png')

    plt.cla() 
    plt.clf()

def plot_daily_profit_loss2():
    
    # Filter out only option transactions
    options_df = df[df['Description'].str.contains("Call|Put", na=False)]

    # Group by Activity Date and Trans Code to find net profit/loss per day for each transaction type
    daily_options_profit = options_df.groupby(['Activity Date', 'Trans Code'])['Amount'].sum().unstack().fillna(0)

    # Adjusting 'BTO' values to be negative for net calculation
    daily_options_profit['BTO'] = -daily_options_profit['BTO']

    # Calculating the net profit/loss per day
    daily_options_profit['Net'] = daily_options_profit['STC'] + daily_options_profit['BTO']

    # Plotting the net profit/loss bar chart
    daily_options_profit['Net'].plot(kind='bar', figsize=(14, 7), color=np.where(daily_options_profit['Net']>=0, 'green', 'red'))
    plt.title('Daily Options Buy/Sell Profit/Loss')
    plt.xlabel('Activity Date')
    plt.ylabel('Net Profit/Loss ($)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('daily_options_profit2_delete.png')
    plt.cla() 
    plt.clf()
    # Calculate daily returns
    df['Daily Return'] = df['Cumulative Capital'].pct_change()

    # Assuming a 95% confidence interval for VaR
    var_95 = df['Daily Return'].quantile(0.05)
    print(f"Value at Risk (95%): {var_95}")

    # Plotting the daily returns histogram
    df['Daily Return'].hist(bins=50)
    plt.title('Daily Returns Histogram')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('daily_returns_histogram_delete.png')

def plot_coorelation():
    # Calculate correlations of the instruments' returns
    instrument_returns = df.pivot_table(index='Activity Date', columns='Instrument', values='Amount').pct_change()
    correlations = instrument_returns.corr()

    # Plotting the correlation matrix
    plt.matshow(correlations)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix_delete.png')

plot_transaction_analysis()
plot_daily_profit_loss2()
plot_strategy()
plot_daily_profit_loss()
plot_pl_instrument()
plot_pl_instrument_2()
plot_coorelation()

'''

# Convert the 'Activity Date' column to datetime
df['Activity Date'] = pd.to_datetime(df['Activity Date'])

# For each unique date, calculate the total capital
# This is done by summing the 'Amount' for each 'Activity Date'
df['Total Capital'] = df.groupby('Activity Date')['Amount'].transform('sum').cumsum()

# Now filter the DataFrame for option transactions only (ignoring transfers and other non-option transactions)
options_df = df[df['Description'].str.contains('Call|Put', na=False)]

# Separate buys and sells for options
buys = options_df[options_df['Trans Code'] == 'BTO']
sells = options_df[options_df['Trans Code'] == 'STC']

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot the options transactions: Buys in one color, Sells in another
ax1.scatter(buys['Activity Date'], buys['Amount'], color='red', label='Buys (BTO)')
ax1.scatter(sells['Activity Date'], sells['Amount'], color='green', label='Sells (STC)')

# Plot total capital over time on a secondary y-axis
ax2 = ax1.twinx()
# import pdb; pdb.set_trace()
ax2.plot(df['Activity Date'], df['Total Capital'], color='blue', label='Total Capital', marker='o')

# Formatting the plot
ax1.set_title('Options Transactions and Total Capital Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Transaction Amount ($)')
ax2.set_ylabel('Total Capital ($)')
ax1.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
ax2.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

# Legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.show()
'''