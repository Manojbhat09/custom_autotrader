import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from datetime import datetime
import random
from robinhood_manager import RobinhoodManager
import plotly.express as px
import os
st.set_page_config(layout="wide")
# Streamlit layout
st.title('Piechart portfolio')
username, password = os.environ['RH_USERNAME'], os.environ['RH_PASSWORD']
robinhood_manager = RobinhoodManager(username, password)

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
def generate_predictions(last_timestamp, num_predictions=5):
    future_times = pd.date_range(start=last_timestamp, periods=num_predictions+1, freq='S')[1:]
    predicted_values = np.random.randn(num_predictions)
    return pd.DataFrame({'Timestamp': future_times, 'Prediction': predicted_values})

# Define a function to update the data, predictions, and past predictions
def update_data_and_predictions():
    # Add new data point
    new_data_point = {
        'Timestamp': pd.Timestamp.now(),
        'Value': np.random.randn()
    }
    st.session_state.data = st.session_state.data.append(new_data_point, ignore_index=True)
    
    # Archive the current predictions as past predictions
    st.session_state.past_predictions = st.session_state.past_predictions.append(
        st.session_state.predictions.iloc[0], ignore_index=True
    )
    
    # Generate new predictions
    new_predictions = generate_predictions(st.session_state.data.iloc[-1]['Timestamp'])
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
    fig = px.pie(df, values='Equity', names='Ticker', title='Portfolio Distribution',
                 color_discrete_sequence=px.colors.sequential.RdBu,
                 hover_data=['Percentage'])

    # Explode the first slice (you can customize this as needed)
    fig.update_traces(textposition='inside', textinfo='percent+label',
                      pull=[0.1 if i == 0 else 0 for i in range(len(df))])

    # Customize layout and legend
    fig.update_layout(legend=dict(
        title="Stocks",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # Add annotations or titles as needed
    fig.add_annotation(x=0.5, y=0.5, text="Portfolio", showarrow=False)

    return fig

def create_general_pie_chart(data, values_column, names_column, title, legend_title):
    df = pd.DataFrame(data)
    fig = px.pie(df, values=values_column, names=names_column, title=title,
                 color_discrete_sequence=px.colors.sequential.RdBu,
                 hover_data=[values_column])

    # Explode the first slice (customize as needed)
    fig.update_traces(textposition='inside', textinfo='percent+label',
                      pull=[0.1 if i == 0 else 0 for i in range(len(df))])

    # Customize layout and legend
    fig.update_layout(legend=dict(
        title=legend_title,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # Add annotations or titles as needed
    fig.add_annotation(x=0.5, y=0.5, text=legend_title, showarrow=False)

    return fig

def create_crypto_pie_chart(crypto_data):
    df = pd.DataFrame(crypto_data)
    fig = px.pie(df, values='Quantity', names='Currency', title='Crypto Portfolio Distribution')
    return fig

def create_crypto_pie_chart(crypto_data):
    df = pd.DataFrame(crypto_data)
    fig = px.pie(df, values='Value', names='Currency', title='Crypto Portfolio Distribution')
    return fig

def create_options_pie_chart(option_data):
    option_fig = create_general_pie_chart(option_data, 'Value', 'Option', 'Options Portfolio Distribution', 'Options')
    return option_fig

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state['logs'] = "Application started.\n"
    # Placeholder for logs
    
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
    st.session_state['selected_ticker'] = 'ETH'
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
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = generate_predictions(pd.Timestamp.now())
if 'past_predictions' not in st.session_state:
    st.session_state['past_predictions'] = pd.DataFrame(columns=['Timestamp', 'Prediction'])
if 'robinhood_manager' not in st.session_state:
    st.session_state['robinhood_manager'] = robinhood_manager

# Initialize the data, predictions, and past predictions in the session state
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=1, freq='S'),
        'Value': np.random.randn(1)
    })

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

# generate the plot figures
# if 'robinhood_manager' in st.session_state:
robinhood_manager = st.session_state['robinhood_manager']
data = robinhood_manager.get_portfolio_holdings()
fig = create_pie_chart(data)

plot_container = st.container()
plot_container.plotly_chart(fig, use_container_width=True)

cryto_data =robinhood_manager.get_crypto_value_holdings()
crypto_fig = create_crypto_pie_chart(cryto_data)

plot_container = st.container()
plot_container.plotly_chart(crypto_fig, use_container_width=True)

option_holdings = robinhood_manager.get_option_holdings()
option_fig = create_options_pie_chart(option_holdings)
plot_container.plotly_chart(option_fig, use_container_width=True)