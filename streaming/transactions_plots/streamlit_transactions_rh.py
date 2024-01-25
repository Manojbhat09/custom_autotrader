import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np 
import plotly.graph_objects as go
# Set the page to wide mode and the layout of your charts
st.set_page_config(layout="wide")

def full_width_plot(func):
    """Decorator to update Plotly figure layout for full width in Streamlit."""
    def wrapper(*args, **kwargs):
        # Call the function to get the figure
        fig = func(*args, **kwargs)
        # Update the layout
        fig.update_layout(
            autosize=True,
            width=None,  # This sets the width to be responsive to the container width
            height=800  # You can adjust the height as needed
        )
        # Use the full width of the page in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        return fig
    return wrapper

# Read the data into a pandas DataFrame
@st.cache
def load_data(csv_path):
    df = pd.read_csv(csv_path, error_bad_lines=False)
    df['Activity Date'] = pd.to_datetime(df['Activity Date'])
    df['Price'] = pd.to_numeric(df['Price'].str.replace('[$,]', '', regex=True), errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'].str.replace('[$,()]', '', regex=True), errors='coerce')
    # Drop any rows where 'Activity Date' has become NaT due to coercion
    df = df.dropna(subset=['Activity Date'])
    df.fillna(0, inplace=True)
    return df

@full_width_plot
def plot_profit_loss_bar_chart(df):
    # Assuming 'BTO' represents buying options and 'STC' represents selling options
    df['Amount'] = np.where(df['Trans Code'] == 'BTO', -df['Amount'], df['Amount'])

    # Group by Activity Date and Instrument to find net profit/loss per day for each Instrument
    daily_options_profit = df.groupby(['Activity Date', 'Instrument'])['Amount'].sum().reset_index()

    # Create a color dictionary for each instrument using Plotly's default cycle
    unique_instruments = daily_options_profit['Instrument'].unique()
    color_sequence = px.colors.qualitative.Plotly

    # Now, let's plot the data
    fig = go.Figure()

    # Loop through each unique instrument and add a bar to the figure
    for instrument in unique_instruments:
        instrument_data = daily_options_profit[daily_options_profit['Instrument'] == instrument]
        fig.add_trace(go.Bar(
            x=instrument_data['Activity Date'],
            y=instrument_data['Amount'],
            name=instrument,
            # Use the Plotly default qualitative color cycle, or define your own color sequence
            marker_color=color_sequence[unique_instruments.tolist().index(instrument) % len(color_sequence)]
        ))

    # Update the layout of the figure to include titles and axis labels
    fig.update_layout(
        barmode='stack',
        title='Daily Options Buy/Sell Profit/Loss by Instrument',
        xaxis_title='Activity Date',
        yaxis_title='Net Profit/Loss ($)',
        legend_title='Instrument',
        hovermode='x'
    )

    return fig

# Function to plot transactions
@full_width_plot
def plot_transactions(df):
    buys = df[df['Trans Code'] == 'BTO']
    sells = df[df['Trans Code'] == 'STC']
    buys['Transaction Value'] = buys['Price'] * buys['Quantity']
    sells['Transaction Value'] = sells['Price'] * sells['Quantity']
    fig = px.scatter(buys, x='Activity Date', y='Transaction Value', color='Instrument')
    fig.update_traces(mode='markers', marker=dict(size=5))
    fig.add_trace(px.scatter(sells, x='Activity Date', y='Transaction Value', color='Instrument').data[0])
    fig.update_layout(title='Transactions Over Time', xaxis_title='Date', yaxis_title='Transaction Value')
    return fig

# Function to plot cumulative capital over time
@full_width_plot
def plot_cumulative_capital(df):
    df['Cumulative Capital'] = df['Amount'].cumsum()
    fig = px.line(df, x='Activity Date', y='Cumulative Capital')
    fig.update_layout(title='Cumulative Capital Over Time', xaxis_title='Date', yaxis_title='Cumulative Capital')
    return fig

# Function to plot profit/loss per instrument
@full_width_plot
def plot_pnl_per_instrument(df):

    # # Calculate cumulative capital
    # df['Cumulative Capital'] = df['Amount'].cumsum()

    # # Separate buys and sells
    # buys = df[df['Trans Code'] == 'BTO']
    # sells = df[df['Trans Code'] == 'STC']

    # # Profit/Loss per Instrument
    # pnl_per_instrument = df.groupby('Instrument')['Amount'].sum()

    # # Performance Over Time
    # df['Realized P/L'] = df.apply(lambda x: x['Amount'] if x['Trans Code'] in ['STC', 'SLD'] else 0, axis=1)
    # df['Cumulative Realized P/L'] = df['Realized P/L'].cumsum()

    # # Transaction Volume
    # transaction_volume = df['Instrument'].value_counts()
    # pnl_per_instrument = df[df['Instrument'] != 0]
    pnl_per_instrument = df.groupby('Instrument')['Amount'].sum().reset_index()
    import pdb; pdb.set_trace()
    # Filter out the row where 'Instrument' is '0' or any non-meaningful value
    

    # import pdb; pdb.set_trace()
    fig = px.bar(pnl_per_instrument, x='Instrument', y='Amount', color='Instrument')
    fig.update_layout(title='Profit/Loss per Instrument', xaxis_title='Instrument', yaxis_title='Profit/Loss')
    return fig

# Function to plot transaction volume per instrument
@full_width_plot
def plot_transaction_volume(df):
    transaction_volume = df['Instrument'].value_counts().reset_index()
    fig = px.bar(transaction_volume, x='index', y='Instrument', color='index')
    fig.update_layout(title='Transaction Volume per Instrument', xaxis_title='Instrument', yaxis_title='Volume')
    return fig

# Layout for Streamlit
st.title("Trading Data Analysis with Streamlit and Plotly")

# Load the data
df = load_data("transactions_2023_robinhood.csv")

# Plotting
plot_profit_loss_bar_chart(pd.DataFrame(df))
# Assuming 'BTO' represents buying options and 'STC' represents selling options
df['Amount'] = np.where(df['Trans Code'] == 'BTO', -df['Amount'], df['Amount'])
plot_transactions(df)
plot_cumulative_capital(df)
plot_pnl_per_instrument(df)
plot_transaction_volume(df)



