import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.append("..")
from robinhood_manager import RobinhoodManager
from datetime import datetime, timedelta
import pickle
from robinhood_options_data_manager import OptionsDataManager

username, password = os.environ['RH_USERNAME'], os.environ['RH_PASSWORD']
manager = RobinhoodManager(username, password)
st.set_page_config(layout="wide")

def fetch_and_cache_options_data(tickers, cache_file='options_data_cache.pkl', cache_duration=timedelta(hours=1)):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as file:
            cached_data = pickle.load(file)
            cache_time, consolidated_df = cached_data['time'], cached_data['data']
            if datetime.now() - cache_time < cache_duration:
                return consolidated_df

    option_data = manager.get_options_data(tickers)
    dataframes = [pd.DataFrame(data).assign(Ticker=ticker) for ticker, data in option_data.items() if isinstance(data, list)]
    consolidated_df = pd.concat(dataframes, ignore_index=True)

    with open(cache_file, 'wb') as file:
        pickle.dump({'time': datetime.now(), 'data': consolidated_df}, file)

    return consolidated_df

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

@full_width_plot
def plot_options_cost(df):
    # Debugging: Print the DataFrame to check its content
    # print("DataFrame Content:")
    # print(df)

    # Check if the DataFrame is empty or if the 'Ticker' and 'mark_price' columns are missing
    if df.empty or 'Ticker' not in df.columns or 'mark_price' not in df.columns:
        st.error("No data available or required columns are missing in the data.")
        return

    # Using 'ask_price' as the y-axis value
    fig = px.bar(df, x='Ticker', y='ask_price', color='Ticker', title='Current Options Ask Price per Ticker')
    return fig

@full_width_plot
def plot_options_volume(df):
    # Using 'ask_price' as the y-axis value
    fig = px.bar(df, x='strike_price', y='volume', color='Ticker', 
                 title='Options Volume Near Current Stock Price')
    
    
    return fig

@full_width_plot
def plot_options_volume_gradient_calls_puts(filtered_data):
    # Determine the number of rows needed for subplots (calls and puts for each ticker)
    rows = len(filtered_data) * 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Plot for each ticker, separate plots for calls and puts
    row = 1
    
    for ticker, data_info in filtered_data.items():
        current_price = data_info['current_price']
        calls_df = data_info['calls']
        puts_df = data_info['puts']

        # Plot calls
        fig.add_trace(
            go.Bar(
                x=calls_df['strike_price'],
                y=calls_df['mark_price'],
                marker=dict(
                    color=calls_df['volume'],
                    colorscale='Viridis',
                ),
                name=f"{ticker} Calls"
            ),
            row=row,
            col=1
        )
        # Add vertical line for current price on calls subplot
        fig.add_shape(
            type='line',
            x0=current_price, y0=0,
            x1=current_price, y1=max(calls_df['mark_price']),
            line=dict(color="RoyalBlue", width=3),
            row=row, col=1
        )
        row += 1

        # Plot puts
        fig.add_trace(
            go.Bar(
                x=puts_df['strike_price'],
                y=puts_df['mark_price'],
                marker=dict(
                    color=puts_df['volume'],
                    colorscale='Viridis',
                ),
                name=f"{ticker} Puts"
            ),
            row=row,
            col=1
        )
        # Add vertical line for current price on puts subplot
        fig.add_shape(
            type='line',
            x0=current_price, y0=0,
            x1=current_price, y1=max(puts_df['mark_price']),
            line=dict(color="RoyalBlue", width=3),
            row=row, col=1
        )
        row += 1

    # Update layout
    # fig.update_layout(height=300*rows//2, title_text='Options Volume and Mark Price Near Current Stock Price by Ticker')

    return fig

@full_width_plot
def plot_options_volume_gradient(filtered_data):
    # Create a figure with subplots for each ticker
    rows = len(filtered_data)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Plot for each ticker
    for i, (ticker, data_info) in enumerate(filtered_data.items(), start=1):
        current_price = data_info['current_price']
        ticker_df = data_info['data']

        # Plot the volume gradient bar chart
        fig.add_trace(
            go.Bar(
                x=ticker_df['strike_price'],
                y=ticker_df['mark_price'],
                marker=dict(
                    color=ticker_df['volume'],
                    colorscale='Viridis',
                    showscale=True if i == rows else False,  # Show color scale only on the last plot
                ),
                name=ticker
            ),
            row=i,
            col=1
        )

    # Update layout
    fig.update_layout(height=300*rows, title_text='Options Volume and Mark Price Near Current Stock Price by Ticker')

    return fig

@full_width_plot
def plot_strike_options_volume(df):
    # Get unique tickers from the DataFrame
    tickers = df['Ticker'].unique()

    # Create a figure with subplots for each ticker
    fig = make_subplots(rows=len(tickers), cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Iterate over each ticker and plot its data
    for i, ticker in enumerate(tickers, start=1):
        # Filter options for the current ticker
        ticker_df = df[df['Ticker'] == ticker]

        # Group by strike price and sum the volumes
        grouped = ticker_df.groupby('strike_price')['volume'].sum().reset_index()

        # Sort the dataframe to get the nearest strike prices
        grouped_sorted = grouped.sort_values(by='strike_price')

        # Plot for the current ticker
        fig.add_trace(
            go.Bar(x=grouped_sorted['strike_price'], y=grouped_sorted['volume'], name=ticker),
            row=i,
            col=1
        )

    # Update layout
    fig.update_layout(height=300*len(tickers), title_text='Options Volume Near Current Stock Price by Ticker')

    return fig


# Streamlit UI
def main():
    st.title("Options Data Analysis")
    cache_file = "options_data_cache.csv"
    tickers = ['AAPL', 'TSLA']
    options_data_manager= OptionsDataManager(manager, cache_file=cache_file)

    # Fetch and plot options data
    if 'options_data' not in st.session_state:
        st.session_state.options_data = []

    if st.button('Bar Chart of strike, mark and volume buys and puts'):
        options_data = options_data_manager.get_current_options_by_expiry(tickers)
        filtered_data = options_data_manager.get_current_prices_and_filter_options_type(options_data)
        plot_fig = plot_options_volume_gradient_calls_puts(filtered_data)


    if st.button('Bar Chart of strike, mark and volume'):
        options_data = options_data_manager.get_current_options_by_expiry(tickers)
        filtered_data = options_data_manager.get_current_prices_and_filter_options(options_data)
        plot_fig = plot_options_volume_gradient(filtered_data)


    if st.button('show bar chart of options with volume'):
        options_data_df= options_data_manager.get_near_strike_options_volume(tickers) # dataframe
        st.write(options_data_df.head())
        plot_options_volume(options_data_df)

    # Current Options Data Section
    if st.button('Show Current Options Data'):
        
        
        current_options_data = options_data_manager.get_current_options_by_expiry(tickers)
        st.write(current_options_data)
        plot_options_cost(current_options_data)
        

    if st.button('Fetch and Plot Options Data'):
          # Replace with the tickers you are interested in
        # Assuming manager is an instance of RobinhoodManager
        option_data = fetch_and_cache_options_data(tickers)
        dataframes = [pd.DataFrame(data).assign(Ticker=ticker) for ticker, data in option_data.items() if isinstance(data, list)]
        consolidated_df = pd.concat(dataframes, ignore_index=True)
        st.session_state.options_data = consolidated_df
        
        # Example: Filter AAPL call options and calculate average ask price
        # aapl_calls = consolidated_df[(consolidated_df['Ticker'] == 'AAPL') & (consolidated_df['type'] == 'call')]
        # average_ask_price = aapl_calls['ask_price'].mean()

        # # Plotting
        plot_options_cost(consolidated_df)
        # st.plotly_chart(plot_fig, use_container_width=True

if __name__ == "__main__":
    main()