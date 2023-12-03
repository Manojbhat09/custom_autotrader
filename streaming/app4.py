import streamlit as st
import pandas as pd
import threading
from threading import Lock
import time
import numpy as np

# Streamlit expects to run the script from top to bottom on each interaction.
# Therefore, use session state to maintain state across interactions.
if 'trading_on' not in st.session_state:
    st.session_state.trading_on = False
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame()
if 'data_lock' not in st.session_state:
    st.session_state.data_lock = Lock()
if 'streamed_data' not in st.session_state:
    st.session_state.streamed_data = pd.DataFrame()


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
        st.session_state.trading_thread = threading.Thread(target=trading_algorithm, daemon=True)
        st.session_state.trading_thread.start()

def stop_trading():
    with st.session_state.data_lock:
        st.session_state.trading_on = False
        if st.session_state.trading_thread.is_alive():
            st.session_state.trading_thread.join()

# Modify the stop_trading function to ensure the thread stops properly
def stop_trading():
    if st.session_state.trading_on:
        st.session_state.trading_on = False
        if st.session_state.trading_thread.is_alive():
            st.session_state.trading_thread.join()  # This will now complete because of the break in trading_algorithm


# Trading function to be run in a separate thread
def trading_algorithm():
    while st.session_state.trading_on:
        # Simulate data streaming
        stream_data()

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

def main():
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

        # Continuously update the live plot and transactions table
        while True:
            # Make sure to lock the data when accessing it
            with st.session_state.data_lock:
                if not st.session_state.streamed_data.empty:
                    live_data_placeholder.line_chart(st.session_state.streamed_data.set_index('time')['price'])
                if not st.session_state.transactions.empty:
                    transactions_placeholder.table(st.session_state.transactions.head(10))
            # Sleep briefly to allow other threads to run and to prevent constant updates
            time.sleep(0.1)

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

    

if __name__ == '__main__':
    main()
