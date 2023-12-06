import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time

# Check if the 'data' and 'predictions' keys are already in session state (which persists across reruns)
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=1, freq='S'),
        'Value': np.random.randn(1)
    })

if 'predictions' not in st.session_state:
    st.session_state['predictions'] = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=5, freq='S')[1:],
        'Prediction': np.random.randn(5)[1:]
    })

# Define a function to update the data and predictions
def update_data_and_predictions():
    # Add new data point
    new_data_point = {
        'Timestamp': pd.Timestamp.now(),
        'Value': np.random.randn()
    }
    st.session_state.data = st.session_state.data.append(new_data_point, ignore_index=True)
    
    # Generate new predictions
    last_timestamp = st.session_state.data['Timestamp'].iloc[-1]
    new_predictions = pd.DataFrame({
        'Timestamp': pd.date_range(start=last_timestamp, periods=6, freq='S')[1:],  # 5 new predictions
        'Prediction': np.random.randn(5)
    })
    st.session_state.predictions = new_predictions

# Update the data and predictions for the plot
update_data_and_predictions()

# Streamlit layout
st.title('Streamlit Trading Dashboard')

# Create the plot using Plotly
fig = go.Figure()

# Add real data to the plot
fig.add_trace(go.Scatter(
    x=st.session_state.data['Timestamp'],
    y=st.session_state.data['Value'],
    mode='lines+markers',
    name='Real Data'
))

# Add predicted data to the plot
fig.add_trace(go.Scatter(
    x=st.session_state.predictions['Timestamp'],
    y=st.session_state.predictions['Prediction'],
    mode='lines+markers',
    name='Predictions'
))

# Update the layout to show the last real data point in the center
last_real_timestamp = st.session_state.data['Timestamp'].iloc[-1]
fig.update_layout(
    xaxis_range=[last_real_timestamp - pd.Timedelta(seconds=10), last_real_timestamp + pd.Timedelta(seconds=10)],
    title='Real-Time Data and Predictions'
)

# We use a container to ensure the plot doesn't get removed on reruns
plot_container = st.container()
plot_container.plotly_chart(fig, use_container_width=True)

time.sleep(1)
st.experimental_rerun()
