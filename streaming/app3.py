from robinhood_manager import RobinhoodManager
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
        rsi_data = data.get_RSI(14)
        macd_data, signal_data = data.get_MACD()

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
        st.plotly_chart(fig, use_container_width=True)

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

