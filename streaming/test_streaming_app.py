import streamlit as st
import pandas as pd
import numpy as np
import time

# Initialize an empty DataFrame to store the streamed data
data = pd.DataFrame(columns=['Timestamp', 'Value'])

# Creating a placeholder for the line chart
chart_placeholder = st.empty()

# Loop to simulate real-time data streaming
for _ in range(100):  # Simulating 100 data points; adjust as needed
    # Generate a new data point with the current timestamp and a random value
    new_data = {'Timestamp': pd.Timestamp.now(), 'Value': np.random.randn()}
    data = data.append(new_data, ignore_index=True)

    # Update the line chart with the new data
    chart_placeholder.line_chart(data.set_index('Timestamp')['Value'])

    # Sleep for a short duration to simulate real-time streaming
    time.sleep(1)  # Sleep for 1 second; adjust the frequency as needed
