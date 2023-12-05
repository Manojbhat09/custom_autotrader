import streamlit as st
import pandas as pd
import threading
from threading import Lock

from robinhood_manager import RobinhoodManager
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(page_title="Login", layout='wide')
from data_manager import DataManager, ModelInference

import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np
import random
import time
import sys
sys.path.append("..")
from scripts.models import make_model
from pages import display_login_page, display_main_plotting_page, display_option_details_page, \
    display, display_streaming_dashboard
import torch
torch.manual_seed(42)  # Replace 42 with your desired seed value
if torch.cuda.is_available():  # If you're using a GPU
    torch.cuda.manual_seed_all(42)
from auth_manager import verify_token

USE_ML = True
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NFLX', 'SPY', 'BA']  # Your list of tickers
time_frame = st.sidebar.selectbox("Select Time Frame", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'])
interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '15m', '30m', '60m', '1d'])
indicators = st.sidebar.write("Addtional Indicators")
# additional_indicator = st.sidebar.selectbox("Select Additional Indicator", ['None', 'RSI', 'MACD', 'Other'])
rsi_selected = st.sidebar.checkbox('RSI')
macd_selected = st.sidebar.checkbox('MACD')
bollinger_bands_selected = st.sidebar.checkbox('Bollinger Bands')  # Added as an example
fibonacci_retracements_selected = st.sidebar.checkbox('Fibonacci Retracements')  # Added as an example
ichimoku_cloud_selected = st.sidebar.checkbox('Ichimoku Cloud')  # Added as an example

config = {
    'scrollZoom': True, 
    'displayModeBar': True, 
    'toImageButtonOptions': {
            'format': 'svg', # one of png, svg, jpeg, webp
            'filename': 'custom_image',
            'height': 500,
            'width': 700,
            'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
            }, 
    'modeBarButtonsToAdd': ['drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
            ]
    }

# Streamlit expects to run the script from top to bottom on each interaction.
# Therefore, use session state to maintain state across interactions.
# Initialize session state variables
if 'trading_on' not in st.session_state:
    st.session_state.trading_on = False
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=['Time', 'Type', 'Price', 'Amount', 'Profit'])
if 'streamed_data' not in st.session_state:
    st.session_state.streamed_data = pd.DataFrame(columns=['time', 'price'])
if 'data_lock' not in st.session_state:
    st.session_state.data_lock = Lock()
if 'update_event' not in st.session_state:
    st.session_state.update_event = threading.Event()

# Placeholder for your predictive model function
def predictive_model():
    # Implement your predictive model here
    # This function should return a prediction based on your model
    return np.random.random()

# Placeholder for your trading decision function
def trading_decision(prediction):
    # Implement your decision-making logic here
    # This function should return a decision based on the prediction
    return "Buy" if prediction > 0.5 else "Sell"

# Placeholder for your trade execution function
def execute_trade(decision):
    # Implement your trade execution logic here
    # This function should return transaction details such as price and amount
    return {'price': np.random.random(), 'amount': np.random.randint(1, 10)}

# Placeholder for your profit calculation function
def calculate_profit(transaction_details):
    # Implement your profit calculation here
    # This function should return the profit for the given transaction
    return transaction_details['price'] * transaction_details['amount']


# Data streaming function
def stream_data():
    # Simulate streaming by appending new data to the existing DataFrame
    # In a real application, this would be replaced by reading from a CSV or data source
    
    with st.session_state.data_lock:
        new_data = {'time': pd.Timestamp.now(), 'price': np.random.random()}
        st.session_state.streamed_data = st.session_state.streamed_data.append(new_data, ignore_index=True)
        st.experimental_rerun()

# Report generation function
def generate_report():
    with st.session_state.data_lock:
        # In a real application, more complex analysis and formatting would be done here
        report = f"Report Generated at {pd.Timestamp.now()}\n"
        report += f"Total Transactions: {len(st.session_state.transactions)}\n"
        report += f"Recent Transactions:\n{st.session_state.transactions.tail()}\n"
    return report

# Functions to start and stop trading
def start_trading():
    with st.session_state.data_lock:
        st.session_state.trading_on = True
        st.session_state.update_event.clear()
        st.session_state.trading_thread = threading.Thread(target=trading_algorithm, daemon=True)
        st.script_run_ctx._add_script_run_ctx(st.session_state.trading_thread)
        st.session_state.trading_thread.start()  # This will now complete because of the break in trading_algorithm

# Modify the stop_trading function to ensure the thread stops properly
def stop_trading():
    with st.session_state.data_lock:
        st.session_state.trading_on = False
        st.session_state.update_event.set() # Signal the threads to stop
        if st.session_state.trading_thread.is_alive():
            st.session_state.trading_thread.join()

# Trading function to be run in a separate thread
def trading_algorithm():

    while st.session_state.trading_on:
        # Simulate data streaming
        stream_data()

        # Simulate a streaming data update
        new_data = {'time': pd.Timestamp.now(), 'price': np.random.random()}
        with st.session_state.data_lock:
            st.session_state.streamed_data = st.session_state.streamed_data.append(new_data, ignore_index=True)

        if not st.session_state.trading_on:
            break

        # Get the prediction from the model
        prediction = predictive_model()
        # Make a trading decision
        decision = trading_decision(prediction)
        # Execute the trade
        transaction_details = execute_trade(decision)
        # Calculate profit from the transaction
        profit = calculate_profit(transaction_details)
        # Update the transactions DataFrame
        new_transaction = pd.DataFrame([{
            'Time': pd.Timestamp.now(),
            'Type': decision,
            'Price': transaction_details['price'],
            'Amount': transaction_details['amount'],
            'Profit': profit
        }])
        with st.session_state.data_lock:
            st.session_state.transactions = pd.concat([new_transaction, st.session_state.transactions], ignore_index=True)
            # st.session_state.transactions = st.session_state.transactions.append(new_transaction, ignore_index=True)

        # Signal that an update is available
        st.session_state.update_event.set()

        # Wait for the next interval
        time.sleep(60)  # Assuming a 1-minute interval between trades

# Function to generate a sample time series data
def generate_time_series():
    # Generate some sample data
    time_index = pd.date_range('2020-01-01', periods=100, freq='D')
    series = pd.Series(np.random.rand(100), index=time_index)
    return series

# Function to generate a sample transactions table
def generate_transactions_table():
    # Generate some sample transaction data
    transactions = pd.DataFrame({
        'Buy Amount': np.random.rand(10),
        'Sell Amount': np.random.rand(10),
        'Charge Amount': np.random.rand(10),
    })
    return transactions

# Function to update the line chart and transaction table
def update_dashboard():
    while not st.session_state.update_event.is_set():
        # Wait for an update signal
        st.session_state.update_event.wait()

        # Update the line chart
        if not st.session_state.streamed_data.empty:
            live_data = st.session_state.streamed_data.set_index('time')['price']
            st.line_chart(live_data)

        # Update the transactions table
        if not st.session_state.transactions.empty:
            transactions_table = st.session_state.transactions.tail(10)
            st.table(transactions_table)

        # Clear the update event
        st.session_state.update_event.clear()

def get_robinhood_ticker_lists():

    all_watchlists = robinhood_manager.get_all_watchlists()
    ticker_lists_from_robinhood = {}
    for watchlist in all_watchlists['results']:
        watchlist_name = watchlist['display_name']
        # Assuming get_watchlist_by_name returns a list of symbols directly
        symbols = robinhood_manager.get_watchlist_by_name(watchlist_name)
        if symbols:  # Only add the list if it has symbols
            ticker_lists_from_robinhood[watchlist_name] = symbols[:20]
            
    return ticker_lists_from_robinhood

# Function to display the dashboard
def display_dashboard():
    st.title("Advanced Trading Dashboard")

    # Create a navigation menu to switch between pages
    pages = ["Main Plotting Page", "Option Details Page", "Options List Page", "Streaming Trading Dashboard"]
    selected_page = st.sidebar.selectbox("Select a Page", pages)

    if selected_page == "Main Plotting Page":
        # Display the main plotting page
        display_main_plotting_page()
    elif selected_page == "Option Details Page":
        # Display the option details page
        display_option_details_page()
    elif selected_page == "Option List Page":
        # Display the option details page
        display_option_details_page()
    elif selected_page == "Streaming Trading Dashboard":
        display_streaming_dashboard()

def main():
    query_params = st.experimental_get_query_params()
    token = query_params.get("token", None)
    if token and verify_token(token):
        display_dashboard()
    else:
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            display_dashboard()
        else:
            st.write("Please login to access the dashboard.")
            display_login_page()


if __name__ == '__main__':
    main()
