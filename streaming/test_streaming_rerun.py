import streamlit as st
import pandas as pd
import numpy as np
import time
from itertools import count

# Define the generator function for data stream
def data_stream():
    for _ in count():
        new_value = np.random.randn()
        new_data = {'Timestamp': pd.Timestamp.now(), 'Value': new_value, 'Prediction': new_value + np.random.randn() * 0.1}  # Adding noise for prediction
        yield new_data
        time.sleep(1)

# Initialize session state to store the generator
if 'data_gen' not in st.session_state:
    st.session_state['data_gen'] = data_stream()

# Initialize an empty DataFrame to store the streamed data
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['Timestamp', 'Value', 'Prediction'])

# Function to fetch the next data point
def next_data_point():
    new_data = next(st.session_state['data_gen'])
    st.session_state['data'] = st.session_state['data'].append(new_data, ignore_index=True)

# Streamlit layout
st.title('Streamlit Trading Dashboard')

# Analysis Dashboard
with st.container():
    st.header('Analysis Dashboard')

    # Dropdown for coin selection (not functional in toy example)
    coin = st.selectbox('Coin:', ['BTC', 'ETH', 'SHIB'])

    # Placeholder for the real-time chart
    chart_placeholder = st.empty()

    # Generate report button
    if st.button('Generate Report'):
        st.write('Report would be generated here.')

# Fetch the next data point and update chart
next_data_point()
chart_data = st.session_state['data'].set_index('Timestamp')[['Value', 'Prediction']]
chart_placeholder.line_chart(chart_data)

# Use Streamlit's caching to save the state across reruns
@st.cache(allow_output_mutation=True)
def get_state():
    return {'counter': 0}

state = get_state()

# Increment the counter
state['counter'] += 1

# Check if we should rerun the script
if state['counter'] < 100:  # Set the number of updates you want
    time.sleep(2)
    st.experimental_rerun()

# At the end of the script, the counter is reset
state['counter'] = 0
