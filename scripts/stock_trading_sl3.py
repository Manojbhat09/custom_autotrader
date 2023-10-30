import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator

# Function to load and plot data from a specified directory
# @st.cache
def load_and_plot(directory):
    # Load data
    df_account_value = pd.read_csv(os.path.join(directory, 'df_account_value.csv'))
    df_actions = pd.read_csv(os.path.join(directory, 'df_actions.csv'))

    # Create plots
    fig1, ax1 = plt.subplots()
    df_account_value.plot(ax=ax1)
    ax1.set_title('Account Value Over Time')

    fig2, ax2 = plt.subplots()
    df_actions.plot(ax=ax2)
    ax2.set_title('Actions Over Time')

    return fig1, fig2

# Function to load and display logs from a specified directory
@st.cache
def display_logs(directory):
    log_file = os.path.join(directory, 'script.log')
    with open(log_file, 'r') as file:
        logs = file.read()
    st.text_area("Logs", logs, height=200)

# Function to load and display TensorBoard logs from a specified directory
@st.cache
def display_tensorboard_logs(directory):
    ea = event_accumulator.EventAccumulator(directory)
    ea.Reload()  # loads events from file

    st.write('TensorBoard Scalars:')
    for tag in ea.Tags()['scalars']:
        st.write(f'{tag}')
        st.line_chart(pd.DataFrame(ea.Scalars(tag)))

st.title('Experiment Analysis')

# Allow user to select an experiment run
experiment_run = st.sidebar.selectbox(
    'Select an experiment run:',
    os.listdir('runs')
)

# Display experiment data for the selected run
if experiment_run:
    st.subheader(f'Analysis for run {experiment_run}')

    # Allow user to select a model
    model = st.sidebar.selectbox(
        'Select a model:',
        os.listdir(os.path.join('runs', experiment_run, 'plots_finrl'))
    )

    # Display model data for the selected model
    if model:
        st.subheader(f'Model: {model}')

        # Get the directory containing data for the selected model
        model_dir = os.path.join('runs', experiment_run, 'plots_finrl', model)

        # Load and plot data
        fig1, fig2 = load_and_plot(model_dir)
        st.pyplot(fig1)
        st.pyplot(fig2)

        # Display images
        st.subheader('Images:')
        image_files = [f for f in os.listdir(model_dir) if f.endswith('.png')]
        for image_file in image_files:
            image_path = os.path.join(model_dir, image_file)
            image = Image.open(image_path)
            st.image(image, caption=image_file)

        # Display logs
        st.subheader('Logs:')
        logs_dir = os.path.join('runs', experiment_run, 'logs_finrl')
        display_logs(logs_dir)

        # Display TensorBoard logs
        st.subheader('TensorBoard Logs:')
        tensorboard_dir = os.path.join('runs', experiment_run, 'results_finrl', model)
        display_tensorboard_logs(tensorboard_dir)
