import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time

# Function to generate new predictions
def generate_predictions(last_timestamp, num_predictions=5):
    future_times = pd.date_range(start=last_timestamp, periods=num_predictions+1, freq='S')[1:]
    predicted_values = np.random.randn(num_predictions)
    return pd.DataFrame({'Timestamp': future_times, 'Prediction': predicted_values})

# Initialize the data, predictions, and past predictions in the session state
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=1, freq='S'),
        'Value': np.random.randn(1)
    })

if 'predictions' not in st.session_state:
    st.session_state['predictions'] = generate_predictions(pd.Timestamp.now())

if 'past_predictions' not in st.session_state:
    st.session_state['past_predictions'] = pd.DataFrame(columns=['Timestamp', 'Prediction'])

# Define a function to update the data, predictions, and past predictions
def update_data_and_predictions():
    # Add new data point
    new_data_point = {
        'Timestamp': pd.Timestamp.now(),
        'Value': np.random.randn()
    }
    st.session_state.data = st.session_state.data.append(new_data_point, ignore_index=True)
    
    # Archive the current predictions as past predictions
    st.session_state.past_predictions = st.session_state.past_predictions.append(
        st.session_state.predictions.iloc[0], ignore_index=True
    )
    
    # Generate new predictions
    new_predictions = generate_predictions(st.session_state.data.iloc[-1]['Timestamp'])
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
    xaxis_range=[last_real_timestamp - pd.Timedelta(seconds=10), last_real_timestamp + pd.Timedelta(seconds=10)],
    title='Real-Time Data and Predictions'
)

# We use a container to ensure the plot doesn't get removed on reruns
plot_container = st.container()
plot_container.plotly_chart(fig, use_container_width=True)

# Pause for 1 second before rerunning to update the plot
time.sleep(1)
st.experimental_rerun()
