import streamlit as st
from data_manager import DataManager
from auth_manager import AuthManager

auth = AuthManager()

st.title("Advanced Trading Dashboard")

# User Authentication
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    if auth.login(username, password):
        print("Logged in successfully.")
    else:
        print("Invalid username or password.")
        
if st.sidebar.button("Register"):
    auth.register(username, password)
    print("Registered successfully.")

# Main App
ticker = st.sidebar.text_input("Enter Ticker:", value='AAPL').upper()
time_frame = st.sidebar.selectbox("Select Time Frame", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'])

data = DataManager(ticker, time_frame)

st.write("### Real-time Data")
chart = st.plotly_chart(data.plot_price_data(), use_container_width=True)

st.write("### Moving Average")
ma_window = st.slider("MA Window", 1, 50, 20)
st.plotly_chart(data.plot_moving_average(ma_window), use_container_width=True)

st.write("### RSI")
rsi_window = st.slider("RSI Window", 1, 50, 14, key='ma_window_key')
st.line_chart(data.get_RSI(rsi_window))

st.write("### MACD")
st.line_chart(data.get_MACD())

st.write("### Price Prediction")
st.write(data.predict_prices())


# Display data and indicators
# st.write("### Price Data")
# st.line_chart(data.data['Close'])

# st.write("### Moving Average")
# ma_window = st.slider("MA Window", 1, 50, 20)
# st.line_chart(data.get_moving_average(ma_window))
