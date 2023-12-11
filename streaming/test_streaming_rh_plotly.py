import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from datetime import datetime
import random
from robinhood_manager import RobinhoodManager
import plotly.express as px
from infer_toy import generate_predictions_rnn

st.set_page_config(layout="wide")
# Streamlit layout
st.title('Streamlit Trading Dashboard')
robinhood_manager = RobinhoodManager(username='manojbhat09@gmail.com', password='MONkeys796@09')

def get_seconds_from_interval(interval):
    interval_mapping = {
        '1sec': 1,
        '3sec': 3,
        '5sec': 5,
        '15sec': 15,
        '30sec': 30,
        '1min': 60,
        '5min': 5 * 60,
        '10min': 10 * 60,
        '15min': 15 * 60,
        '30min': 30 * 60,
        '1hr': 60 * 60
    }
    return interval_mapping.get(interval, 1)  # Default to 1 second if interval not found

# Function to generate dummy news articles
def generate_news_articles():
    headlines = [
        "Breaking News: Market Hits Record High",
        "Economic Outlook Appears Bright, Says Experts",
        "Technology Stocks Soar Amid Positive Forecasts",
        "New Renewable Energy Sources Discovered",
        "Global Markets React to Latest Trade Talks",
        "Cryptocurrency Prices Surge Overnight",
        "Innovative Tech Startups Attract Major Investments",
        "Healthcare Industry Sees Unprecedented Growth",
        "Automotive Sector Revolutionized by AI",
        "Real Estate Market Trends Upward This Quarter"
    ]
    # Simulate a list of random news articles
    articles = [{"headline": random.choice(headlines), "link": f"https://news.com/article_{random.randint(1, 100)}"} for _ in range(10)]
    return articles

# Function to simulate autotrading (replace with your logic)
def autotrade(data):
    # Dummy logic for autotrading
    if not data.empty and len(data) > 1:
        last_value = data['Value'].iloc[-1]
        if last_value > 0:  # Dummy condition for buying/selling
            return 'buy', last_value
        else:
            return 'sell', last_value
    return 'hold', 0

# Function to generate new predictions
def generate_predictions(data, last_timestamp, num_predictions=5):
    future_times = pd.date_range(start=last_timestamp, periods=num_predictions+1, freq='S')[1:]
    # predicted_values = np.random.randn(num_predictions)
    last_point = data["Value"]
    predicted_values =np.random.normal(last_point, 20, size=num_predictions)
    # predicted_values =np.random.randint(2400, 2500, size=num_predictions)
    predict_data_point= pd.DataFrame({'Timestamp': future_times, 'Prediction': predicted_values})
    print(predict_data_point)
    return predict_data_point

def get_real_time_rh_data():
    real_time_data = robinhood_manager.get_current_crypto_price(st.session_state.selected_ticker)
    new_data_point = {
        'Timestamp': pd.Timestamp.now(),
        'Value': real_time_data
    }
    print("rh data: ", new_data_point)
    return new_data_point

def get_dummpy_data():
    new_data_point = {
        'Timestamp': pd.Timestamp.now(), 
        'Value':  np.random.randn()
    }
    return new_data_point

# Define a function to update the data, predictions, and past predictions
def update_data_and_predictions():

    new_data_point = get_real_time_rh_data()
    # new_data_point = get_dummpy_data()

    st.session_state.data = st.session_state.data.append(new_data_point, ignore_index=True)
    
    # Archive the current predictions as past predictions
    st.session_state.past_predictions = st.session_state.past_predictions.append(
        st.session_state.predictions.iloc[0], ignore_index=True
    )
    
    # Generate new predictions
    new_predictions = generate_predictions_rnn(new_data_point, st.session_state.data.iloc[-1]['Timestamp'])
    st.session_state.predictions = new_predictions

def gen_plot_figs():
    # Create the plot using Plotly
    fig = go.Figure()

    # Add real data to the plot
    fig.add_trace(go.Scatter(
        x=st.session_state.data['Timestamp'],
        y=st.session_state.data['Value'],
        mode='lines+markers',
        name='Real Data'
    ))

    # Add past predictions to the plot
    fig.add_trace(go.Scatter(
        x=st.session_state.past_predictions['Timestamp'],
        y=st.session_state.past_predictions['Prediction'],
        mode='lines+markers',
        name='Past Predictions',
        line=dict(color='orange')
    ))

    # Add new predictions to the plot
    fig.add_trace(go.Scatter(
        x=st.session_state.predictions['Timestamp'],
        y=st.session_state.predictions['Prediction'],
        mode='lines+markers',
        name='Predictions',
        line=dict(color='green')
    ))

    # Update the layout to show the last real data point in the center
    last_real_timestamp = st.session_state.data['Timestamp'].iloc[-1]
    fig.update_layout(
        xaxis_range=[last_real_timestamp - pd.Timedelta(seconds=20), last_real_timestamp + pd.Timedelta(seconds=20)],
        title='Real-Time Data and Predictions'
    )
    return fig

# Function to add a log entry and update the log display
def add_log_entry(entry):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['logs'] += f"{timestamp}: {entry}\n"
    # Update the log display
    # log_placeholder.text_area("Logs", st.session_state['logs'], height=300)

# Function to save transactions to CSV
def save_transactions_to_csv(ticker, transactions, filename="transactions.csv"):
    transactions.to_csv(f"{ticker}_{filename}", index=False)

def create_pie_chart(data):
    df = pd.DataFrame(data)
    fig = px.pie(df, values='Equity', names='Ticker', title='Portfolio Distribution')
    return fig

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state['logs'] = "Application started.\n"
    # Placeholder for logs
    
# if 'robinhood_manager' not in st.session_state:
#     robinhood_manager = st.session_state['robinhood_manager']
# Initialize session state for news articles
if 'news_articles' not in st.session_state:
    st.session_state['news_articles'] = generate_news_articles()
    st.session_state['last_news_update'] = datetime.now()
# Initialize session state variables for user inputs
if 'min_transaction_profit' not in st.session_state:
    st.session_state['min_transaction_profit'] = 0
if 'target_profit' not in st.session_state:
    st.session_state['target_profit'] = 0
if 'plot_update_interval' not in st.session_state:
    st.session_state['plot_update_interval'] = '1sec'
if 'trading_interval' not in st.session_state:
    st.session_state['trading_interval'] = '1min'
# Initialize session state variables
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = 'BTC'
# Initialize session state for transactions and last save timestamp
if 'transactions' not in st.session_state:
    st.session_state['transactions'] = pd.DataFrame(columns=['Timestamp', 'Action', 'Price'])
if 'last_save_time' not in st.session_state:
    st.session_state['last_save_time'] = datetime.now()
# Initialize other session state variables
if 'autotrading' not in st.session_state:
    st.session_state['autotrading'] = False
if 'transactions' not in st.session_state:
    st.session_state['transactions'] = pd.DataFrame(columns=['Timestamp', 'Action', 'Price'])

if 'past_predictions' not in st.session_state:
    st.session_state['past_predictions'] = pd.DataFrame(columns=['Timestamp', 'Prediction'])

# Initialize the data, predictions, and past predictions in the session state
if 'data' not in st.session_state:
    new_data_point = get_real_time_rh_data()
    st.session_state['data'] = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=1, freq='S'),
        'Value': new_data_point['Value'] # np.random.randn(1)
    })

if 'predictions' not in st.session_state: # dummy predictions
    st.session_state['predictions'] = generate_predictions(st.session_state['data'], pd.Timestamp.now())

# Update the data and predictions for the plot
update_data_and_predictions()

# Sidebar for input controls
with st.sidebar:
    st.header('Settings')
    # Dropdown for coin selection
    ticker = st.selectbox('Select Coin:', ['BTC', 'ETH', 'SHIB'])

    # Detect if the coin selection has changed
    if ticker != st.session_state.selected_ticker:
        # Update the selected coin in session state
        st.session_state.selected_ticker = ticker

        # Reset data and transactions for the new coin
        try:
            filename = f"{ticker}_transactions.csv"
            st.session_state.transactions = pd.read_csv(filename)
        except Exception as e:
            print(f"could not read the csv file for {ticker}", e)
            st.session_state.transactions = pd.DataFrame(columns=['Timestamp', 'Action', 'Price'])
        st.session_state.transactions['Timestamp'] = pd.to_datetime(st.session_state.transactions['Timestamp'], errors='coerce')
        # Reset trading status (if applicable)
        st.session_state.autotrading = False

    # Minimum transaction profit input
    st.session_state.min_transaction_profit = st.text_input(
        'Minimum Transaction Profit:', st.session_state.min_transaction_profit)

    # Target profit input
    st.session_state.target_profit = st.text_input(
        'Target Profit:', st.session_state.target_profit)

    # Button to toggle autotrade
    if st.button('Toggle Autotrade'):
        st.session_state.autotrading = not st.session_state.autotrading
        add_log_entry(f"Trading {st.session_state.autotrading}")
        add_log_entry(f"{ticker}")

    st.header("Top News Articles")
    for article in st.session_state.news_articles:
        st.markdown(f"[{article['headline']}]({article['link']})")
    
    log_placeholder = st.empty()
    log_placeholder.text_area("Logs", st.session_state['logs'], height=300)



fig = gen_plot_figs()
# We use a container to ensure the plot doesn't get removed on reruns
plot_container = st.container()
plot_container.plotly_chart(fig, use_container_width=True)

# Time interval selection for plot updates
st.session_state.plot_update_interval = st.select_slider(
    'Plot Update Interval:',
    options=['1sec', '3sec', '5sec', '15sec','30sec', '1min', '5min', '10min', '15min', '30min', '1hr'],
    value=st.session_state.plot_update_interval
)

# Time interval selection for trading frequency
st.session_state.trading_interval = st.select_slider(
    'Trading Interval:',
    options=['1min', '5min', '10min', '15min', '30min', '1hr', '1d', '1w', '1mo'],
    value=st.session_state.trading_interval
)

# Update news articles at random intervals
current_time = datetime.now()
if (current_time - st.session_state.last_news_update).seconds >= random.randint(30, 300):  # Random interval between 30 seconds to 5 minutes
    st.session_state.news_articles = generate_news_articles()
    st.session_state.last_news_update = current_time
    st.experimental_rerun()

# Container for transactions and profit display
with st.container():
    if st.session_state.autotrading:
        # Simulate a trade
        action, price = autotrade(st.session_state.data)
        if action != 'hold':
            new_transaction = {'Timestamp': pd.Timestamp.now(), 'Action': action, 'Price': price}
            st.session_state.transactions = st.session_state.transactions.append(new_transaction, ignore_index=True)

    # Calculate and display profit
    # Replace this with your actual profit calculation logic
    profit = np.sum(st.session_state.transactions['Price'])  # Dummy profit calculation
    st.subheader('Profit')
    st.write(f'Total Profit: {profit}')

    # Display transactions
    st.subheader('Transactions')
    
    top_transactions = st.session_state.transactions.tail(5)
    
    sorted_transactions = top_transactions.sort_values(by='Timestamp', ascending=False)
    st.table(sorted_transactions)

    # Check if an hour has passed since the last save
    current_time = datetime.now()
    if (current_time - st.session_state.last_save_time).seconds >= 60:
        save_transactions_to_csv(ticker, st.session_state.transactions)
        st.session_state.last_save_time = current_time
    
# Generate report button
if st.button('Generate Report'):
    # Replace this with your report generation logic
    report = f'Report generated on {pd.Timestamp.now()}\nTotal Profit: {profit}'
    st.text(report)


# Pause for 1 second before rerunning to update the plot
sleep_duration = get_seconds_from_interval(st.session_state.plot_update_interval)
time.sleep(sleep_duration)
st.experimental_rerun()
