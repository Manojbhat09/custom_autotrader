import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from itertools import count
import time

# Define the generator function for data stream
def data_stream():
    for _ in count():
        new_data = {'Timestamp': pd.Timestamp.now(), 'Value': np.random.randn()}
        yield new_data
        time.sleep(1)  # Simulate data stream delay

# Initialize session state to store the generator
if 'data_gen' not in st.session_state:
    st.session_state['data_gen'] = data_stream()

# Initialize an empty DataFrame to store the streamed data
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['Timestamp', 'Value'])

# Initialize an empty DataFrame to store predictions
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = pd.DataFrame(columns=['Timestamp', 'Prediction'])

# Function to generate predictions
def generate_predictions(current_time, num_predictions=5):
    predictions = []
    for i in range(num_predictions):
        # Assuming predictions are 1 timestep apart and start from the next timestep
        prediction_time = current_time + pd.Timedelta(seconds=i+1)
        predicted_value = np.random.randn()  # Replace with actual prediction logic
        predictions.append({'Timestamp': prediction_time, 'Prediction': predicted_value})
    return predictions

# Function to update the data and predictions
def update_data_and_predictions():
    new_data = next(st.session_state['data_gen'])
    st.session_state['data'] = st.session_state['data'].append(new_data, ignore_index=True)
    
    # Generate predictions based on the latest timestamp
    latest_timestamp = new_data['Timestamp']
    new_predictions = generate_predictions(latest_timestamp)
    st.session_state['predictions'] = pd.DataFrame(new_predictions)

# Streamlit layout
st.title('Streamlit Trading Dashboard')

# Analysis Dashboard
with st.container():
    st.header('Analysis Dashboard')

    # Dropdown for coin selection (not functional in this example)
    coin = st.selectbox('Coin:', ['BTC', 'ETH', 'SHIB'])

    # Placeholder for the real-time chart
    chart_placeholder = st.empty()

# Update data and predictions
update_data_and_predictions()

# Combine data and predictions for plotting
combined_df = pd.concat([st.session_state['data'], st.session_state['predictions']])

# Plot the real-time chart and predictions using Plotly
fig = go.Figure()

# Add the real data to the plot
fig.add_trace(go.Scatter(
    x=st.session_state['data']['Timestamp'], 
    y=st.session_state['data']['Value'], 
    mode='lines+markers', 
    name='Real Data'
))

# Add the predictions to the plot
fig.add_trace(go.Scatter(
    x=st.session_state['predictions']['Timestamp'], 
    y=st.session_state['predictions']['Prediction'], 
    mode='lines+markers', 
    name='Predictions'
))

# Update the layout to center the last real data point
last_real_timestamp = st.session_state['data']['Timestamp'].iloc[-1]
fig.update_layout(
    xaxis_range=[last_real_timestamp - pd.Timedelta(seconds=5),  # 5 seconds before
                 last_real_timestamp + pd.Timedelta(seconds=5)],  # 5 seconds after
    title='Real-Time Data and Predictions'
)

chart_placeholder.plotly_chart(fig, use_container_width=True)

# Schedule the script to rerun periodically
time.sleep(1)
st.experimental_rerun()
