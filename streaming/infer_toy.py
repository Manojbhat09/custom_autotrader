
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append("..")
from scripts.data_process import seqTradeDataset, collate_fn, collate_fn_angled, InputPreprocessor
from scripts.models import make_model
from robinhood_manager import RobinhoodManager
from train_crypto import Config
from data_manager import ModelInference
from tqdm import tqdm
import importlib
import os
import random

Config=None 
robinhood_manager = RobinhoodManager(username='manojbhat09@gmail.com', password='MONkeys796@09')

# Function to dynamically import Config from the latest experiment
def import_latest_experiment_config(run_directory, experiment_name=None):
    # Filter directories that start with 'experiment_'
    experiment_dirs = [d for d in os.listdir(run_directory) if os.path.isdir(os.path.join(run_directory, d)) and d.startswith('experiment_')]

    # Select the specific experiment if given, else find the latest
    if experiment_name and experiment_name in experiment_dirs:
        latest_experiment = experiment_name
    else:
        latest_experiment = sorted(experiment_dirs, reverse=True)[0]

    # Construct the path to the train_crypto.py in the latest experiment
    config_path = os.path.join(run_directory, latest_experiment, 'results', 'script_copies', 'train_crypto.py')
    # Import the Config class from the specified file
    spec = importlib.util.spec_from_file_location("trainer", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)  # Load the module if you want to run the training 
    return config_module.Config, os.path.join(run_directory, latest_experiment)  # Return the Config class

def fetch_real_time_crypto_data(ticker, num_points):
    data= []
    for _ in tqdm(range(num_points)):
        data_point, real_time_data = robinhood_manager.get_real_time_crypto_data(ticker)
        data.append(real_time_data)
    return data 

# # Function to generate predictions
# def generate_predictions(model_inference, data, last_timestamp, num_predictions=5):
#     # Ensure there is enough data
#     if len(data) < Config.WINDOW_SIZE:
#         raise ValueError("Not enough data for the input window.")

#     future_times = pd.date_range(start=last_timestamp, periods=Config.HORIZON, freq='T')
#     predictions = []
#     all_data_points = []

#     # Generate predictions using ModelInference
#     # We are generating x predictions for the same input and seeing the variance in the outputs

#     for i in tqdm(range(num_predictions)):
#         torch.manual_seed(42*i)
#         random.seed(42*i)
#         model_output, _ = model_inference.run_inference_bayesian_heading(data, ticker)
#         mean_predictions = model_output[3]  # Assuming mean predictions are at index 3

#         # Update predictions list
#         predictions.extend(mean_predictions)
#         # Update data for the next prediction
#         # This part may need adjustment based on how your model expects the input data to be updated
#         # data = update_data_with_predictions(data, mean_predictions)

#         # Inverse transform predictions
#         # inverse_predictions = model_inference.inverse_scale(np.array(mean_predictions), column_idx=3, scaler_list=seqTradeDataset.input_scaler_list)
#         predict_data_point = pd.DataFrame({'Timestamp': future_times, 'Prediction': mean_predictions[:len(future_times)].squeeze() })
#         all_data_points.append(predict_data_point)
#     return all_data_points

# Function to generate predictions
def generate_predictions_rnn(model_inference, data, last_timestamp, ticker='BTC', window_size=60, horizon=15):
    # Ensure there is enough data
    if len(data) < window_size:
        raise ValueError("Not enough data for the input window.")

    future_times = pd.date_range(start=last_timestamp, periods=horizon, freq='T')
    predictions = []

    model_output, _ = model_inference.run_inference_bayesian_heading(data, ticker)
    mean_predictions = model_output[3]  # Assuming mean predictions are at index 3

    # Update predictions list
    predictions.extend(mean_predictions)

    predict_data_point = pd.DataFrame({'Timestamp': future_times, 'Prediction': mean_predictions[:len(future_times)].squeeze() })
    return predict_data_point

def convert_to_dataframe(data):
    # Convert each dictionary in the list to a DataFrame row
    df = pd.DataFrame(data)

    # Convert the 'Timestamp' column to a datetime index
    df['Datetime'] = pd.to_datetime(df['Timestamp'], utc=True)
    df = df.set_index('Datetime')

    # Drop the original 'Timestamp' column
    df = df.drop(columns=['Timestamp'])

    # Add Dividends and Stock Splits columns, initialized to zero
    df['Dividends'] = 0
    df['Stock Splits'] = 0

    return df

# Main execution
if __name__ == "__main__":
    # Fetch latest Config
    run_directory = '../run/'
    Config, exp_folder = import_latest_experiment_config(run_directory)
    Config.WINDOW_SIZE = 60

    # getting the checkpoints
    Config.CHECKPOINT_FOLDER = os.path.join(exp_folder, "models")
    checkpoints = [d for d in os.listdir(Config.CHECKPOINT_FOLDER)]
    Config.CHECKPOINT_PATH = os.path.join(Config.CHECKPOINT_FOLDER, checkpoints[0])

    ticker = Config.TICKER.split("-")[0]
    num_data_points = Config.WINDOW_SIZE +5   # Number of points to fetch for initial data
    num_predictions = 5  # Number of future predictions
    last_timestamp = pd.Timestamp('2023-01-01 00:00:00')  # Example last timestamp
    # import pdb; pdb.set_trace()
    # Fetch initial real-time data

    # if there is a pickle dump
    import pickle 
    file = open("65_btc_samples", "rb")
    data_raw = pickle.load(file)
    file.close()

    # data = fetch_real_time_crypto_data(ticker, num_data_points)
    import pdb; pdb.set_trace()
    data = convert_to_dataframe(data_raw)
    # Initialize ModelInference with the model and checkpoint path
    model_inference = ModelInference(Config.MODEL_NAME, Config.CHECKPOINT_PATH, config=Config)
    # Generate predictions
    output  = generate_predictions_rnn(model_inference, data, last_timestamp, num_predictions=num_predictions)# Assuming 'data' is your DataFrame with the last known data points
    file = open("65_btc_predictions", "wb")
    pickle.dump(output, file)
    file.close()
    
