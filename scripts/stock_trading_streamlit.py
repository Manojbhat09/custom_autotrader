import streamlit as st
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Base directory
BASE_DIR = "runs"

def list_runs():
    return os.listdir(BASE_DIR)

def display_run(run):
    st.write(f"## Run: {run}")
    run_dir = os.path.join(BASE_DIR, run)
    
    # Displaying logs
    logs_dir = os.path.join(run_dir, "logs_finrl")
    if os.path.exists(logs_dir):
        with st.expander("Logs"):
            for log_file in os.listdir(logs_dir):
                st.text(open(os.path.join(logs_dir, log_file)).read())
    
    # Displaying models
    models_dir = os.path.join(run_dir, "models_finrl")
    if os.path.exists(models_dir):
        with st.expander("Models"):
            st.write(os.listdir(models_dir))
    
    # Displaying plots
    plots_dir = os.path.join(run_dir, "plots_finrl")
    if os.path.exists(plots_dir):
        with st.expander("Plots"):
            for model in os.listdir(plots_dir):
                model_dir = os.path.join(plots_dir, model)
                for file in os.listdir(model_dir):
                    if file.endswith(".png"):
                        st.image(os.path.join(model_dir, file))
                    elif file.endswith(".csv"):
                        df = pd.read_csv(os.path.join(model_dir, file))
                        st.write(df)
    
    # Displaying results
    results_dir = os.path.join(run_dir, "results_finrl")
    if os.path.exists(results_dir):
        with st.expander("Results"):
            for file in os.listdir(results_dir):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(results_dir, file))
                    st.write(df)
                else:
                    st.write(file)

st.sidebar.title("FinRL Experiment Runs")
selected_run = st.sidebar.selectbox("Select Run", list_runs())
display_run(selected_run)
