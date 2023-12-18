import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from datetime import datetime
import random
from robinhood_manager import RobinhoodManager
import plotly.express as px
from infer_toy import generate_predictions_rnn, import_latest_experiment_config, convert_to_dataframe
from data_manager import ModelInference
import os
import tqdm
import pytz
from talib import RSI, BBANDS, STOCH
st.set_page_config(layout="wide")
# Streamlit layout
st.title('Streamlit Trading Dashboard')


def autotrade(current_data):
    """
    Make a trading decision based on the current market data.

    :param current_data: DataFrame with the latest market data
    :return: (action, price) - The trading action ('buy', 'sell', or 'hold') and the price at which to execute
    """
    # Initialize the DataFrame to store the current data
    df = pd.DataFrame(current_data)

    # Calculate indicators
    df['upper_band'], df['middle_band'], df['lower_band'] = BBANDS(df['Close'], timeperiod=20)
    df['rsi'] = RSI(df['Close'], timeperiod=14)
    df['stochastic_k'], df['stochastic_d'] = STOCH(df['Close'], df['Close'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)

    # Get the latest data point
    last_row = df.iloc[-1]

    # Define trading signals
    buy_signal = (last_row['rsi'] < 70) & (last_row['stochastic_k'] < 20)
    sell_signal = (last_row['rsi'] > 30) | (last_row['stochastic_k'] > 60)

    # Decision-making
    if buy_signal:
        action = 'buy'
        price = last_row['Close']
    elif sell_signal:
        action = 'sell'
        price = last_row['Close']
    else:
        action = 'hold'
        price = None

    return action, price

def get_current_times():
    utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)
    current_timezone_time = utc_time.astimezone()
    pst_time = utc_time.astimezone(pytz.timezone('US/Pacific'))
    est_time = utc_time.astimezone(pytz.timezone('US/Eastern'))
    return utc_time, current_timezone_time, pst_time, est_time

def update_time_display():
    global time_placeholder
    utc_time, current_timezone_time, pst_time, est_time = get_current_times()
    time_text = (
        f"UTC Time: {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Local Timezone: {current_timezone_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"PST Time: {pst_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"EST Time: {est_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    time_placeholder.text(time_text)

def setup_model():
    run_directory = '../run/'
    Config, exp_folder = import_latest_experiment_config(run_directory)
    Config.WINDOW_SIZE = 60

    # getting the checkpoints
    Config.CHECKPOINT_FOLDER = os.path.join(exp_folder, "models")
    checkpoints = [d for d in os.listdir(Config.CHECKPOINT_FOLDER)]
    Config.CHECKPOINT_PATH = os.path.join(Config.CHECKPOINT_FOLDER, checkpoints[0])

    ticker = Config.TICKER.split("-")[0]
    num_data_points = Config.WINDOW_SIZE +5   # Number of points to fetch for initial data
    last_timestamp = pd.Timestamp('2023-01-01 00:00:00')  # Example last timestamp

    model_inference = ModelInference(Config.MODEL_NAME, Config.CHECKPOINT_PATH, config=Config)
    print("checkpoint loaded succesfully")
    return model_inference

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
    last_point = data.get("Value", data.get("Close", 0))
    predicted_values =np.random.normal(last_point, 20, size=num_predictions)
    # predicted_values =np.random.randint(2400, 2500, size=num_predictions)
    predict_data_point= pd.DataFrame({'Timestamp': future_times, 'Prediction': predicted_values})
    print(predict_data_point)
    return predict_data_point

def get_real_time_rh_data():
    real_time_data = st.session_state.robinhood_manager.get_current_crypto_price(st.session_state.selected_ticker)
    new_data_point = {
        'Timestamp': pd.Timestamp.now(),
        'Value': real_time_data
    }
    print("rh data: ", new_data_point)
    return new_data_point

def get_real_time_crypto_data(ticker):
    real_time_data = st.session_state.robinhood_manager.get_real_time_crypto_data(ticker)
    new_data_point = {
        'Timestamp': pd.Timestamp.now(),
        'Value': real_time_data["Close"]
    }
    print("rh data: ", new_data_point)

    return new_data_point, real_time_data 

def get_dummpy_data():
    new_data_point = {
        'Timestamp': pd.Timestamp.now(), 
        'Value':  np.random.randn()
    }
    return new_data_point

def interval_to_pandas_resampling_rule(desired_interval):
    # Mapping of intervals to pandas resampling rule format
    rule_mapping = {
        '1sec': '1S',
        '3sec': '3S',
        '5sec': '5S',
        '15sec': '15S',
        '30sec': '30S',
        '1min': '1T',
        '5min': '5T',
        '10min': '10T',
        '15min': '15T',
        '30min': '30T',
        '1hr': '1H', 
        '1d': '1D',
        '1w': '1W',
        '1mo': '1M'  # Note: '1M' stands for calendar month end frequency
    }
    return rule_mapping.get(desired_interval, '1T')  # Default to 1 minute if interval not found


def calculate_interval_span(desired_interval, points_required):
    # Map the desired interval to robin_stocks interval and calculate the span
    interval_mapping = {
        '1sec': ('15second', 'hour'),
        '3sec': ('15second', 'hour'),
        '5sec': ('15second', 'hour'),
        '15sec': ('15second', 'hour'),
        '30sec': ('15second', 'hour'),
        '1min': ('15second', 'hour'),
        '5min': ('5minute', 'day'),
        '10min': ('5minute', 'day'),
        '15min': ('5minute', 'day'),
        '30min': ('10minute', 'week'),
        '1hr': ('hour', 'week'), 
        '1d': ('day', 'year'),
        '1w': ('day', '5year'),
        '1mo': ('week', '5year')
    }

    robin_stocks_interval, robin_stocks_span = interval_mapping.get(desired_interval, ('5minute', 'day'))

    return robin_stocks_interval, robin_stocks_span


def fetch_and_resample_crypto_data( symbol, desired_interval, points_required):
    '''
    If the interval is 15sec, then there is 15sec for the hour so 4 points every minute for 60 minutes, 240 points we have
    But if its 3 sec interval then 15sec is interpolated into 3sec so 5*4*60 points for the hour, 1200 points so points are more 
    Now to compare, we have to get more points in the 3sec one to satisfy the time range of 1 hour
    
    '''
    interval, span = calculate_interval_span(desired_interval, points_required)
    historical_data = st.session_state.robinhood_manager.get_crypto_historicals(symbol, interval=interval, span=span)

    # Convert to DataFrame and resample
    df = pd.DataFrame(historical_data)
    df['begins_at'] = pd.to_datetime(df['begins_at'])
    df.set_index('begins_at', inplace=True)

    resampling_rule = interval_to_pandas_resampling_rule(desired_interval)
    resampled_df = df.resample(resampling_rule).agg({'open_price': 'first','close_price': 'last','high_price': 'max','low_price': 'min','volume': 'sum'})

    # Convert None to NaN
    resampled_df = resampled_df.applymap(lambda x: np.nan if x is None else x)

    # Apply linear interpolation
    resampled_df = resampled_df.astype(float).interpolate()
    return resampled_df

def ready_for_inference(data):
    cols = data.columns
    print(cols)
    data = convert_to_dataframe(data) 
    return data

# Define a function to update the data, predictions, and past predictions
def update_data_and_predictions(load_data = True, debug=True):
    # new_data_point = get_real_time_rh_data()
    new_data_point, real_time_data = get_real_time_crypto_data(st.session_state.selected_ticker)
    # new_data_point = get_dummpy_data()

    st.session_state.data = st.session_state.data.append(new_data_point, ignore_index=True)
    st.session_state.realtime_data = st.session_state.realtime_data.append(real_time_data, ignore_index=True)
    
    # Archive the current predictions as past predictions
    if len(st.session_state.predictions):
        st.session_state.past_predictions = st.session_state.past_predictions.append(
            st.session_state.predictions.iloc[0], ignore_index=True
        )

    if load_data:

        if debug:
            import pickle
            st.session_state.data = pickle.load(open("realtimedata", "rb"))
            st.session_state.predictions = pickle.load(open("predictions2", "rb"))
            st.session_state.realtime_data = pickle.load(open("realtimedata2", "rb"))
            st.session_state.past_predictions = pickle.load(open("predictions", "rb"))
        
        else:
            # generate random prediction data to add that to the historic data from the api on just one call
            crypto_df = fetch_and_resample_crypto_data(symbol=st.session_state.selected_ticker, desired_interval=st.session_state.plot_update_interval, points_required=61)
            crypto_data = st.session_state.robinhood_manager.rename_pd(crypto_df)
            crypto_data = pd.DataFrame(crypto_data)[["Timestamp", "Close"]]
            crypto_data.columns = ['Timestamp', "Value"]
            st.session_state.realtime_data = crypto_data
            st.session_state.data = crypto_data

            # Generate and update predictions for 60 items
            dist = len(st.session_state['data']) - 60
            for idx in tqdm.tqdm(range(60)):
                last_timestamp = st.session_state['data']['Timestamp'].iloc[dist + idx]
                new_predictions = generate_predictions(st.session_state['data'].iloc[dist + idx], last_timestamp)
                
                # Update past predictions and current predictions
                if len(st.session_state.predictions):
                    st.session_state.past_predictions = st.session_state.past_predictions.append(
                        st.session_state.predictions.iloc[0], ignore_index=True
                    )
                else:
                    st.session_state.past_predictions = new_predictions
                st.session_state.predictions = new_predictions
            return

    
    # Generate new predictions
    if len(st.session_state.data) < 60:
        # new_predictions = generate_predictions(st.session_state['data'].iloc[-1], pd.Timestamp.now())
        crypto_df = fetch_and_resample_crypto_data(symbol=st.session_state.selected_ticker, desired_interval=st.session_state.plot_update_interval, points_required=61)
        crypto_data = st.session_state.robinhood_manager.rename_pd(crypto_df)
        st.session_state.realtime_data = pd.DataFrame(crypto_data)
        crypto_data = pd.DataFrame(crypto_data)[["Timestamp", "Close"]]
        crypto_data.columns = ['Timestamp', "Value"]
        st.session_state.data = crypto_data
        # Generate and update predictions for 60 items
        dist = len(st.session_state['data']) - 60
        for idx in tqdm.tqdm(range(60)):
            last_timestamp = st.session_state['data']['Timestamp'].iloc[dist + idx]
            new_predictions = generate_predictions(st.session_state['data'].iloc[dist + idx], last_timestamp)
            
            # Update past predictions and current predictions
            if len(st.session_state.predictions):
                st.session_state.past_predictions = st.session_state.past_predictions.append(
                    st.session_state.predictions.iloc[0], ignore_index=True
                )
            else:
                st.session_state.past_predictions = new_predictions
            st.session_state.predictions = new_predictions
    else:
        print("running new inference")
        data = ready_for_inference(st.session_state.realtime_data)
        last_timestamp = st.session_state.data.iloc[-1]['Timestamp']
        last_timestamp = pd.to_datetime(last_timestamp).tz_localize('US/Pacific')
        last_timestamp = last_timestamp.tz_convert('UTC')
        new_predictions = generate_predictions_rnn( st.session_state.inference_model , data, last_timestamp)
        new_predictions = np.array(new_predictions)
        print(st.session_state.realtime_data.iloc[-1])
        print(new_predictions[:, 1])
        new_predictions = pd.DataFrame({'Timestamp': new_predictions[:, 0], 'Prediction': new_predictions[:, 1]})
    st.session_state.predictions = new_predictions

def gen_plot_figs():
    # Create the plot using Plotly
    fig = go.Figure()

    # Add real data to the plot
    fig.add_trace(go.Scatter(
        x=st.session_state.data['Timestamp'],
        y=st.session_state.data.get('Value', st.session_state.data.get("Close", 0)),
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
    
if 'robinhood_manager' not in st.session_state:
    st.session_state['robinhood_manager'] = RobinhoodManager(username='manojbhat09@gmail.com', password='MONkeys796@09')
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
    st.session_state['plot_update_interval'] = '5sec'
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
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=["Timestamp", "Value"])
if 'realtime_data' not in st.session_state:
    st.session_state['realtime_data'] = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
if 'inference_model' not in st.session_state:
    st.session_state['inference_model'] = setup_model()

# Initialize the data, predictions, and past predictions in the session state
if 'data' not in st.session_state:
    new_data_point = get_real_time_rh_data()
    st.session_state['data'] = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=1, freq='S'),
        'Value': new_data_point['Value'] # np.random.randn(1)
    })

if 'predictions' not in st.session_state: # dummy predictions
    st.session_state['predictions'] = pd.DataFrame(columns=["Timestamp", "Prediction"])
    # st.session_state['predictions'] = generate_predictions(st.session_state['data'], pd.Timestamp.now())

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

# Place this inside your main function
time_placeholder = st.empty()
update_time_display()

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
