import streamlit as st
st.set_page_config(page_title="Login")
from data_manager import DataManager
from auth_manager import AuthManager
import plotly.subplots as sp
import plotly.graph_objects as go

auth = AuthManager()
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NFLX', 'SPY', 'BA']  # Your list of tickers
time_frame = st.sidebar.selectbox("Select Time Frame", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'])
interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '15m', '30m', '60m', '1d'])
additional_indicator = st.sidebar.selectbox("Select Additional Indicator", ['None', 'RSI', 'MACD', 'Other'])
ticker_lists = {
        'List 1': ['AAPL', 'GOOGL', 'MSFT'],
        'List 2': ['AMZN', 'FB', 'TSLA'],
        # ... add up to 20 lists ...
    }


def display_login_page():
    st.sidebar.title("User Authentication")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")
    register_button = st.sidebar.button("Register")
    if login_button:
        if auth.login(username, password):
            st.session_state['logged_in'] = True
            display_dashboard()
        else:
            st.sidebar.error("Invalid username or password.")
    if register_button:
        auth.register(username, password)
        st.sidebar.success("Registered successfully.")

# Function to display the dashboard
def display_dashboard():
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
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name=f'{ticker} Price'), row=row_idx, col=1)
        fig.add_trace(go.Scatter(x=ma_data.index, y=ma_data, mode='lines', name=f'{ticker} 20-day MA'), row=row_idx, col=1)

        # Additional Indicators
        if additional_indicator == 'RSI':
            fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data, mode='lines', name=f'{ticker} RSI'), row=row_idx, col=2)
        elif additional_indicator == 'MACD':
            macd_data, signal_data = data.get_MACD()
            fig.add_trace(go.Scatter(x=macd_data.index, y=macd_data, mode='lines', name=f'{ticker} MACD'), row=row_idx, col=2)
            fig.add_trace(go.Scatter(x=signal_data.index, y=signal_data, mode='lines', name=f'{ticker} Signal Line'), row=row_idx, col=2)

    fig.update_layout(
        height=200*len(tickers),  # Adjust height to accommodate the number of tickers
        title_text="Multi-Ticker Analysis",
        showlegend=False
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

# Main App
def main():
    st.title("Advanced Trading Dashboard")
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        display_dashboard()
    else:
        display_login_page()

if __name__ == "__main__":
    main()

