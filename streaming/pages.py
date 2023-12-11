

import streamlit as st
import pandas as pd
import threading
from threading import Lock

from robinhood_manager import RobinhoodManager
import matplotlib.pyplot as plt
import streamlit as st
# st.set_page_config(page_title="Login", layout='wide')
from data_manager import DataManager, ModelInference
from auth_manager import AuthManager, verify_token
import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np
import random
import time
import sys
# from app import display_dashboard
from auth_manager import AuthManager, verify_token

robinhood_manager = RobinhoodManager(username='manojbhat09@gmail.com', password='MONkeys796@09')
auth = AuthManager()
time_frame = st.sidebar.selectbox("Select Time Frame", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'])
interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '15m', '30m', '60m', '1d'])
indicators = st.sidebar.write("Addtional Indicators")
rsi_selected = st.sidebar.checkbox('RSI')
macd_selected = st.sidebar.checkbox('MACD')
bollinger_bands_selected = st.sidebar.checkbox('Bollinger Bands')  # Added as an example
fibonacci_retracements_selected = st.sidebar.checkbox('Fibonacci Retracements')  # Added as an example
ichimoku_cloud_selected = st.sidebar.checkbox('Ichimoku Cloud')  # Added as an example

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


def display_login_page():
    st.sidebar.title("User Authentication")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")
    register_button = st.sidebar.button("Register")
    google_login_button = st.sidebar.button("Login with Google")  # Add this line
    

        # Read HTML content
    with open('login_app/templates/login.html', 'r') as file:
        login_html = file.read()

    # Use st.markdown to render HTML
    st.markdown(login_html, unsafe_allow_html=True)

    # if login_button:
    #     if auth.login(username, password):
    #         st.session_state['logged_in'] = True
    #         st.session_state['robinhood_manager'] = robinhood_manager
    #         display_dashboard()
    #     else:
    #         st.sidebar.error("Invalid username or password.")
    # if register_button:
    #     auth.register(username, password)
    #     st.sidebar.success("Registered successfully.")
    # if google_login_button:  # Add this block
    #     st.write("Redirecting to Google login...")
    #     # Assuming your Flask app is running on http://localhost:5000
    #     flask_login_url = "http://localhost:5000/login/google"
    #     st.markdown(f'<meta http-equiv="refresh" content="0;url={flask_login_url}">', unsafe_allow_html=True)
    #     st.write("Redirecting...")
    #     st.redirect(flask_login_url)
    #     st.session_state['robinhood_manager'] = robinhood_manager

def display_main_plotting_page():

    ticker_lists = {
        'List 1': ['AAPL', 'GOOGL', 'MSFT'],
        'List 2': ['AMZN', 'FB', 'TSLA'],
        # ... add up to 20 lists ...
    }
    if 'robinhood_manager' in st.session_state:
        robinhood_manager = st.session_state['robinhood_manager']
        unique_symbols = robinhood_manager.get_all_unique_symbols()
        ticker_lists = get_robinhood_ticker_lists() # overriding
        ticker_lists.update({"none" : []})
        # Create a dropdown menu with the unique symbols
    
    # adding crypto list 
    ticker_lists.update({"Crypto" : ['BTC', "ETH", "SOL"]})

    selected_list = st.sidebar.selectbox("Select Ticker List", list(ticker_lists.keys()))
    tickers = ticker_lists[selected_list]

    # hardcoding the string for yfinance dataset 
    
    crypto_ticker = [ticker_name for ticker_name in ticker_lists.keys() if 'Crypto' in ticker_name]
    if len(crypto_ticker):
        for crypto_list_name in crypto_ticker:
            for idx, ticker in enumerate(ticker_lists[crypto_list_name]):
                if 'BTC' in ticker:
                    ticker_lists[crypto_list_name][idx] = 'BTC-USD'
                if 'ETH' in ticker:
                    ticker_lists[crypto_list_name][idx] = 'ETH-USD'

    # Pagination logic
    num_plots_perpage = 4
    page = st.sidebar.slider('Page:', 1, (len(tickers) + 2) // num_plots_perpage)
    start_idx = (page - 1) * num_plots_perpage
    end_idx = min(start_idx + num_plots_perpage, len(tickers))
    num_plots_perpage = end_idx - start_idx

    subplot_titles = []
    for ticker in tickers[start_idx:end_idx]:
        subplot_titles.extend([f"{ticker}_Price", f"{ticker}_aIndicator"])

    fig = sp.make_subplots(
        rows=num_plots_perpage, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles
    )

    for i, ticker in enumerate(tickers[start_idx:end_idx]):
        data_manager = DataManager(ticker, time_frame, interval)
        price_data = data_manager.get_price_data()
        row_idx = i + 1  # Adjusting for 1-based index

        # Price and Moving Average
        data_manager.plot_price_ma(ticker, price_data, fig, row_idx)

        # button_key = f'Predict Future Prices_{ticker}_{i}'  # Create a unique key for each button
        # if st.button('Predict Future Prices', key=button_key):
        #     zoom_range = st.session_state.get('zoom_range', (len(price_data) - 100, len(price_data)))  # Default to last 100 data points
        #     # The code above for running inference and plotting predicted prices
        #     # Instantiate ModelInference
        #     model_inference = ModelInference(model_name='SegmentBayesianHeadingModel', checkpoint_path='/home/mbhat/tradebot/custom_autotrader/run/experiment_20231103_162231/models/BTC-USD_checkpoint_experiment_20231103_162231.pth')
            
        #     update_predictions(price_data=price_data, window_size=60, model_inference=model_inference, cache=cache)
        #     cached_price_data = cache.cached_price_data()[-60:]  # Assuming prediction for last 60 data points
        #     cached_segment_data = cache.cached_segment_data()
        #     # Run inference
        #     # if not model_inference.check_model_compatability(ticker, time_frame, interval):
        #     #     skip with error logs and print out compatiablity criteria
            
        #     update_predictions(price_data=price_data, window_size=60, model_inference=model_inference, cache=cache)
        #     # outputs, start_idx = model_inference.run_inference_bayesian_heading(price_data, ticker)
        #     # # Get predicted prices
        #     # y_pred_segments, y_pred_angles, y_pred_profit, y_pred_prices_mean, y_pred_prices_low, y_pred_prices_high = outputs

        #     # Plot and display predicted prices
        #     fig = data_manager.plot_predicted_prices(*cached_price_data(), fig, row_idx, start_idx)
        #     fig = data_manager.plot_predicted_segments(*cached_segment_data(), fig, row_idx, start_idx, zoom_range)
        
        #     # Setting the x-axis range for zoom and space for horizon predictions
        #     xaxis_range = [zoom_range[0], zoom_range[1] + 15]  # Assuming horizon predictions for next 15 data points
        #     fig.update_xaxes(range=xaxis_range, row=row_idx, col=2)

        #     f'''
        #     Max Profit predicted: {y_pred_profit[0]}
        #     '''
        
        # Additional Indicators
        if rsi_selected: # additional_indicator
            data_manager.plot_rsi(ticker, fig, row_idx)
        if macd_selected:
            data_manager.plot_macd(ticker, fig, row_idx)
        if bollinger_bands_selected:
            data_manager.plot_bollinger_bands(ticker, price_data, fig, row_idx)
        if fibonacci_retracements_selected:
            data_manager.plot_fibonacci_retracements(ticker, price_data, fig, row_idx)
        if ichimoku_cloud_selected:
            data_manager.plot_ichimoku_cloud(ticker, price_data, fig, row_idx)

    fig.update_layout(
        height=400*num_plots_perpage,  # Adjust height to accommodate the number of tickers
        title_text="Multi-Ticker Analysis",
        #showlegend=True
    )

    container = st.container()
    with container:
        st.write("""
            <style>
                .stContainer {
                    overflow-y: auto;
                    max-height: 600px;
                }
            </style>
        """, unsafe_allow_html=True)
        # st.plotly_chart(fig, use_container_width=True)
        # Display the plot
        st.plotly_chart(fig, use_container_width=True, height=1100)

# Define a function to get option data from Robinhood
@st.cache(ttl=120)  # Cache the data for 5 minutes
def get_option_data():
    option_info = robinhood_manager.get_option_instrument_info('YOUR_OPTION_URL_HERE')
    return option_info

def display_option_details_page():
    st.title("Option Details Page")
    option_placeholder = st.empty()  # Create a placeholder for option details
    update_button = st.button("Stop Updates")
    option_info_list = []
    while not update_button:
        start_time = time.time()    
        option_positions_data = robinhood_manager.get_my_option_positions()
        num_options = len(option_positions_data)
        num_columns = 3  # Number of columns for displaying options
        columns = st.columns(num_columns)
        
        # with st.progress(0) as progress_bar:
        with st.empty() as progress_bar_placeholder,st.empty() as table_placeholder:
            for i, option_data in enumerate(option_positions_data[:5]):
                option_url = option_data['option']
                option_instrument_info = robinhood_manager.get_option_instrument_info(option_url)
                
                # # Calculate the column index and row index for displaying options
                # col_idx = i % num_columns
                # row_idx = i // num_columns
                option_info = {
                    "Option": f"Option {i + 1}",
                    "Chain Symbol": option_instrument_info['chain_symbol'],
                    "Expiration Date": option_instrument_info['expiration_date'],
                    "Strike Price": f"${option_instrument_info['strike_price']}",
                    "Option Type": option_instrument_info['type'].capitalize()
                }
                
                option_info_list.append(option_info)
                st.write(option_info)
                # Update progress bar
                progress = (i + 1) / num_options
                st.write(f"Progress: {int(progress * 100)}%")

                # progress_bar.progress(progress)
                 # Update the table
                 # Update the table
                 # Display option information in a tabular format
                st.table(option_info_list)

        st.subheader("Option Information:")
        end_time = time.time()  # Record the end time
        latency = end_time - start_time  # Calculate the time taken
        st.write(f"Time taken to fetch and display option data: {latency:.2f} seconds")
        
        # Update option details every 2 minutes
        # time.sleep(120)
        update_button = st.button("Stop Updates")  # Check if the stop button is clicked

def display():

    # Sidebar for controls
    with st.sidebar:
        st.header('Controls')
        if st.button('Toggle Autotrade'):
            if st.session_state.trading_on:
                stop_trading()
            else:
                start_trading()
            st.session_state.trading_on = not st.session_state.trading_on

        technical_indicators = st.multiselect('Add to plot:', ['SMA', 'EMA', 'RSI'])

        # Input for target profit
        target_profit = st.text_input('Target Profit')

        # Dropdown for selecting coins
        selected_coin = st.selectbox('Select Coin', ['BTC', 'ETH', 'SHIB'])

        if st.button('Generate Report'):
            report = generate_report()
            st.text_area('Report', report, height=300)

    

    # Main dashboard area
    with st.container():
        st.header('Live Data and Transactions')
        # Placeholders for live data and transactions
        live_data_placeholder = st.empty()
        transactions_placeholder = st.empty()

    # Start update threads if not already running
    if 'update_threads' not in st.session_state:
        st.session_state.update_threads = {
            'live_data': threading.Thread(target=update_dashboard, args=(live_data_placeholder,), daemon=True),
            'transactions': threading.Thread(target=update_dashboard, args=(transactions_placeholder,), daemon=True)
        }
        for thread in st.session_state.update_threads.values():
            thread.start()

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



def display_streaming_dashboard():
    st.title('Streamlit Trading Dashboard')

    # Sidebar
    with st.sidebar:
        st.header('Controls')
        technical_indicators = st.multiselect('Add to plot:', ['SMA', 'EMA', 'RSI'])

        # Input for target profit
        target_profit = st.text_input('Target Profit')

        # Dropdown for selecting coins
        selected_coin = st.selectbox('Select Coin', ['BTC', 'ETH', 'SHIB'])

        # Toggle for autotrade
        if st.button('Toggle Autotrade'):
            if st.session_state.trading_on:
                stop_trading()
            else:
                start_trading()
            st.session_state.trading_on = not st.session_state.trading_on

    # Analysis Dashboard
    with st.container():
        st.header('Analysis Dashboard')
        # Input for target profit
        target_profit = st.text_input('Target Profit')

        # # Dropdown for selecting coins
        # selected_coin = st.selectbox('Select Coin', ['BTC', 'ETH', 'SHIB'])

        # Starting and stopping the trading thread
        if autotrade and not hasattr(st.session_state, 'trading_thread'):
            st.session_state.trading_thread = threading.Thread(target=start_trading, daemon=True)
            st.script_run_ctx._add_script_run_ctx(st.session_state.trading_thread)
            st.session_state.trading_thread.start()
        elif not autotrade and hasattr(st.session_state, 'trading_thread'):
            st.session_state.trading_thread = None

    with st.container():
        # Dynamic plot for streamed data
        live_data_placeholder = st.empty()
        # Dynamic table for transactions
        transactions_placeholder = st.empty()

        # # Continuously update the live plot and transactions table
        # while True:
        #     # Make sure to lock the data when accessing it
        #     with st.session_state.data_lock:
        #         if not st.session_state.streamed_data.empty:
        #             live_data_placeholder.line_chart(st.session_state.streamed_data.set_index('time')['price'])
        #         if not st.session_state.transactions.empty:
        #             transactions_placeholder.table(st.session_state.transactions.head(10))
        #     # Sleep briefly to allow other threads to run and to prevent constant updates
        #     time.sleep(0.1)

    with st.container():
        # Time series plot for past data
        # st.line_chart(np.random.randn(100, 2))  # Placeholder for actual data

        # Time interval discretization
        discretization_interval = st.slider('Discretization Interval:', 5, 60, 5)

        # Profit updates
        if st.session_state.transactions.empty:
            st.write("No transactions made yet.")
        else:
            st.write(f"Latest Profit: {st.session_state.transactions.iloc[0]['Profit']}")

    # Table of transaction details
    with st.container():
        st.header('Transaction Details')
        if not st.session_state.transactions.empty:
            st.table(st.session_state.transactions.head(10))

    # Generate report button and area
    with st.container():
        if st.button('Generate Report'):
            st.session_state.report_generated = generate_report()
        st.text_area('Report', st.session_state.report_generated if 'report_generated' in st.session_state else '')

