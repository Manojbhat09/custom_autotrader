import streamlit as st
import pandas as pd
import numpy as np
import time

def data_stream():
    """
    Generator function to simulate streaming data.
    Yields a new data point with the current timestamp and a random value.
    """
    while True:
        new_data = {'Timestamp': pd.Timestamp.now(), 'Value': np.random.randn()}
        yield new_data
        time.sleep(1)  # Sleep for 1 second; adjust the frequency as needed

# Initialize session state to store the data stream
if 'data_stream' not in st.session_state:
    st.session_state.data_stream = data_stream()

# Initialize an empty DataFrame to store the streamed data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Timestamp', 'Value'])

# Creating a placeholder for the line chart
chart_placeholder = st.empty()

# Loop to update the chart with streamed data
for _ in range(100):  # Number of updates; adjust as needed
    new_data = next(st.session_state.data_stream)
    st.session_state.data = st.session_state.data.append(new_data, ignore_index=True)
    chart_placeholder.line_chart(st.session_state.data.set_index('Timestamp')['Value'])
