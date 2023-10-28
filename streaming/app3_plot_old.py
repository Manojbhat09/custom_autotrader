from robinhood_manager import RobinhoodManager
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(page_title="Login", layout='wide')
from data_manager import DataManager
from auth_manager import AuthManager, verify_token
import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np
import random


auth = AuthManager()
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NFLX', 'SPY', 'BA']  # Your list of tickers
time_frame = st.sidebar.selectbox("Select Time Frame", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'])
interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '15m', '30m', '60m', '1d'])
additional_indicator = st.sidebar.selectbox("Select Additional Indicator", ['None', 'RSI', 'MACD', 'Other'])
robinhood_manager = RobinhoodManager(username='manojbhat09@gmail.com', password='MENkeys796@09')

# Function to plot Bollinger Bands
def plot_bollinger_bands(df):
    window = 20
    df['Middle Band'] = df['Close'].rolling(window).mean()
    df['Upper Band'] = df['Middle Band'] + 1.96*df['Close'].rolling(window).std()
    df['Lower Band'] = df['Middle Band'] - 1.96*df['Close'].rolling(window).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Middle Band'], name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], name='Lower Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))

    st.plotly_chart(fig)

# Function to plot Fibonacci Retracements
def plot_fibonacci_retracements(df):
    peak = df['Close'].max()
    trough = df['Close'].min()
    levels = [0, 0.236, 0.382, 0.5, 0.618, 1]
    fibs = [(peak - trough) * level + trough for level in levels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
    for level, fib in zip(levels, fibs):
        fig.add_hline(y=fib, line_dash="dash", annotation_text=f'Fib {level}')

    st.plotly_chart(fig)

# Function to plot Ichimoku Cloud
def plot_ichimoku_cloud(df):
    highs = df['High']
    lows = df['Low']
    tenkan_sen = (highs.rolling(window=9).max() + lows.rolling(window=9).min()) / 2
    kijun_sen = (highs.rolling(window=26).max() + lows.rolling(window=26).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (highs.rolling(window=52).max() + lows.rolling(window=52).min()) / 2
    chikou_span = df['Close'].shift(-26)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=tenkan_sen, name='Tenkan-sen'))
    fig.add_trace(go.Scatter(x=df.index, y=kijun_sen, name='Kijun-sen'))
    fig.add_trace(go.Scatter(x=df.index, y=senkou_span_a, name='Senkou Span A'))
    fig.add_trace(go.Scatter(x=df.index, y=senkou_span_b, name='Senkou Span B'))
    fig.add_trace(go.Scatter(x=df.index, y=chikou_span, name='Chikou Span'))

    st.plotly_chart(fig)

def random_color():
    return f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'

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
    
    if login_button:
        if auth.login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['robinhood_manager'] = robinhood_manager
            display_dashboard()
            
        else:
            st.sidebar.error("Invalid username or password.")
    if register_button:
        auth.register(username, password)
        st.sidebar.success("Registered successfully.")
    if google_login_button:  # Add this block
        st.write("Redirecting to Google login...")
        # Assuming your Flask app is running on http://localhost:5000
        flask_login_url = "http://localhost:5000/login/google"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={flask_login_url}">', unsafe_allow_html=True)
        st.write("Redirecting...")
        # st.redirect(flask_login_url)
        st.session_state['robinhood_manager'] = robinhood_manager

# Function to display the dashboard
def display_dashboard():
    st.title("Advanced Trading Dashboard")
    ticker_lists = {
        'List 1': ['AAPL', 'GOOGL', 'MSFT'],
        'List 2': ['AMZN', 'FB', 'TSLA'],
        # ... add up to 20 lists ...
    }
    if 'robinhood_manager' in st.session_state:
        robinhood_manager = st.session_state['robinhood_manager']
        unique_symbols = robinhood_manager.get_all_unique_symbols()
        ticker_lists = get_robinhood_ticker_lists() # overriding
        # Create a dropdown menu with the unique symbols

    selected_list = st.sidebar.selectbox("Select Ticker List", list(ticker_lists.keys()))
    tickers = ticker_lists[selected_list]

    subplot_titles = []
    for ticker in tickers:
        subplot_titles.extend([f"{ticker}_Price", f"{ticker}_aIndicator"])

    fig = sp.make_subplots(
        rows=len(tickers), cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles
    )

    for i, ticker in enumerate(tickers):
        data = DataManager(ticker, time_frame, interval)
        price_data = data.get_price_data()
        ma_data = data.get_moving_average(20)
        row_idx = i + 1  # Adjusting for 1-based index

        # Price and Moving Average
        fig.add_trace(
                go.Scatter(
                    x=list(range(len(price_data))),
                    y=price_data['Close'],
                    customdata=price_data.index,
                    hovertemplate='%{customdata}: %{y}<extra></extra>',
                    line=dict(color=random_color()),
                    mode='lines',
                    name=f'{ticker} Price'
                ),
                row=row_idx,
                col=1
            )
        fig.add_trace(
                go.Scatter(
                    x=list(range(len(ma_data))),
                    y=ma_data,
                    customdata=ma_data.index,
                    hovertemplate='%{customdata}: %{y}<extra></extra>',
                    line=dict(color=random_color()),
                    mode='lines',
                    name=f'{ticker} 20-day MA'
                ),
                row=row_idx,
                col=1
            )
        # fig.add_trace(go.Scatter(x=list(range(len(price_data))), y=price_data['Close'], mode='lines', name=f'{ticker} Price'), row=row_idx, col=1)
        # fig.add_trace(go.Scatter(x=list(range(len(price_data))) , y=ma_data, mode='lines', name=f'{ticker} 20-day MA'), row=row_idx, col=1)

        # Additional Indicators
        if additional_indicator == 'RSI':
            rsi_data = data.get_RSI(14)
            # fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data, mode='lines', name=f'{ticker} RSI'), row=row_idx, col=2)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(rsi_data))),
                    y=rsi_data,
                    customdata=rsi_data.index,
                    hovertemplate='%{customdata}: %{y}<extra></extra>',
                    line=dict(color=random_color()),
                    mode='lines',
                    name=f'{ticker} RSI'
                ),
                row=row_idx,
                col=2
            )
        elif additional_indicator == 'MACD':
            macd_data, signal_data = data.get_MACD()
            # fig.add_trace(go.Scatter(x=macd_data.index, y=macd_data, mode='lines', name=f'{ticker} MACD'), row=row_idx, col=2)
            # fig.add_trace(go.Scatter(x=signal_data.index, y=signal_data, mode='lines', name=f'{ticker} Signal Line'), row=row_idx, col=2)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(macd_data))),
                    y=macd_data,
                    customdata=macd_data.index,
                    hovertemplate='%{customdata}: %{y}<extra></extra>',
                    line=dict(color=random_color()),
                    mode='lines',
                    name=f'{ticker} MACD'
                ),
                row=row_idx,
                col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(signal_data))),
                    y=signal_data,
                    customdata=signal_data.index,
                    hovertemplate='%{customdata}: %{y}<extra></extra>',
                    line=dict(color=random_color()),
                    mode='lines',
                    name=f'{ticker} Signal Line'
                ),
                row=row_idx,
                col=2
            )
        elif additional_indicator == "Bollinger Bands":
            plot_bollinger_bands(price_data)
        elif additional_indicator == "Fibonacci Retracements":
            plot_fibonacci_retracements(price_data)
        elif additional_indicator == "Ichimoku Cloud":
            plot_ichimoku_cloud(price_data)

    fig.update_layout(
        height=600*len(tickers),  # Adjust height to accommodate the number of tickers
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

# Main App
# def main():
#     
 
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


if __name__ == "__main__":
    main()


'''

    # Create a figure with subplots
    fig, axs = plt.subplots(len(tickers), 2, figsize=(10, 5 * len(tickers)))

    for i, ticker in enumerate(tickers):
        # Assume price_data and ma_data are obtained for the current ticker
        data = DataManager(ticker, time_frame, interval)
        price_data = data.get_price_data()  # Example function to get data
        ma_data = data.get_moving_average(20)  # Example function to get data

        # Price and Moving Average
        axs[i, 0].plot(price_data.index, price_data['Close'], label=f'{ticker} Price')
        axs[i, 0].plot(ma_data.index, ma_data, label=f'{ticker} 20-day MA')
        axs[i, 0].legend()
        axs[i, 0].set_title(f'{ticker} Price and Moving Average')

        # Additional Indicator (e.g., RSI)
        rsi_data = data.get_RSI(14)  # Example function to get data
        axs[i, 1].plot(rsi_data.index, rsi_data, label=f'{ticker} RSI')
        axs[i, 1].legend()
        axs[i, 1].set_title(f'{ticker} RSI')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the figure in Streamlit
    st.pyplot(fig)


'''