'''
Why Input and Output Dimension Might Be 1:

In many stock price forecasting tasks, the goal is to predict the next day's closing price based on past closing prices. In such cases, you might only use the closing price as a feature, leading to an input dimension of 1. Similarly, since you're predicting a single value (next day's closing price), the output dimension would also be 1.
However, if you're using multiple technical indicators (like Moving Averages, RSI, MACD, etc.) as features, then the input dimension should match the number of features (e.g., 18).

'''
from tqdm import tqdm
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import glob
import os
import re

current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape x to include a batch dimension which is 1

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out
    

# Define a function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train or evaluate a model")

    # Add a command-line argument for training
    parser.add_argument("--train", action="store_true", help="Train the model")

    # Parse the command-line arguments
    args = parser.parse_args()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    if not os.path.exists('plots/training'):
        os.makedirs('plots/training')

    # Assuming df is your dataframe with the features
    os.makedirs('data/processed', exist_ok=True)
    file_path = os.path.join("data", "processed", "TSLA.csv")
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)

    # Drop non-numeric columns for simplicity
    df = df.drop(columns=['date', 'begins_at', 'session', 'interpolated', 'symbol'])

    # Interpolate the missing values
    df.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Splitting data into train and test sets using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(scaled_data):
        train, test = scaled_data[train_index], scaled_data[test_index]

    seq_length = 20

    X_train_close, y_train_close = create_sequences(train, seq_length) # multi-feature
    X_test_close, y_test_close = create_sequences(test, seq_length) # multi-feature

    X_train_close, y_train_close = shuffle(X_train_close, y_train_close, random_state=42)

    # Convert data to PyTorch tensors
    X_train_close = torch.tensor(X_train_close, dtype=torch.float32)
    y_train_close = torch.tensor(y_train_close, dtype=torch.float32)
    X_test_close = torch.tensor(X_test_close, dtype=torch.float32)
    y_test_close = torch.tensor(y_test_close, dtype=torch.float32)

    input_dim = 19
    hidden_dim = 32
    num_layers = 2
    output_dim = 19

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

    # Ensure directories exist
    checkpoint_folder = 'models/checkpoints/'
    os.makedirs('models/checkpoints', exist_ok=True)
    # Assuming 'model' is your trained PyTorch model
    checkpoint_path = f'models/checkpoints/TSLA_multi_feature_checkpoint_{current_datetime}.pth'

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if(args.train):
        
        print("training the model")

        train_losses = []
        test_losses = []
        num_epochs = 1000
        epoch_range = tqdm(range(num_epochs), desc="Training Progress", ncols=100, ascii=True)

        for epoch in epoch_range:
            epoch_range.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            outputs = model(X_train_close)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train_close)
            loss.backward()
            optimizer.step()
            
            # Calculate and store the training loss
            train_loss = loss.item()
            train_losses.append(train_loss)
            
            # Calculate the test loss
            with torch.no_grad():
                model.eval()
                test_outputs = model(X_test_close)
                test_loss = criterion(test_outputs, y_test_close).item()
                test_losses.append(test_loss)
                model.train()
            
            # Print and save the losses plot every 10 epochs
            if (epoch + 1) % 10 == 0:
                epoch_range.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
            epoch_range.set_postfix(train_loss=train_loss, test_loss=test_loss)
                
        # Plot the training and test losses
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()

        # Save the plot in the specified folder
        plot_path = f'plots/training/loss_plot_multi_feature_{current_datetime}.png'  # Change this path as needed
        plt.savefig(plot_path)
        plt.close()


        torch.save(model.state_dict(), checkpoint_path)
        print("Model checkpoint saved to checkpoint_path")
        tqdm.write(f"Saved checkpoint: {checkpoint_path}")

    print("evaluating the model")

    # Define the pattern to match checkpoint file names
    pattern = r'TSLA_multi_feature_checkpoint_\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.pth'

    # List all files in the folder
    files = os.listdir(checkpoint_folder)

    # Filter the files based on the pattern
    matching_files = [file for file in files if re.match(pattern, file)]

    # Sort the matching files by modification time (latest first)
    matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_folder, x)), reverse=True)

    if matching_files:
        latest_checkpoint = matching_files[0]
        latest_checkpoint = os.path.join(checkpoint_folder, latest_checkpoint)
        print("Latest checkpoint:", latest_checkpoint)
    else:
        print("No matching checkpoint files found in the folder.")

    # Save the plot in the specified folder
    os.makedirs('plots/testing', exist_ok=True)

    # Use re.search to find the pattern in the string
    match = re.search(pattern, latest_checkpoint)

    if match:
        checkpoint_name = match.group()
        print("Checkpoint name:", checkpoint_name)
    else:
        print("No checkpoint name found in the string.")

    # Load the model for evaluation
    model.load_state_dict(torch.load(latest_checkpoint))
    model.eval()

    # Test the model
    with torch.no_grad():
        predictions = model(X_test_close)
        mse_loss = criterion(predictions, y_test_close)
        print(f"Test MSE Loss: {mse_loss.item()}")

    predictions = np.array(predictions)
    # Plotting each feature
    
    features = [
    'open_price', 'close_price', 'high_price', 'low_price',
    'volume', 'MA_3', 'RSI', 'EMA_12', 'EMA_26',
    'MACD', 'Signal_Line', 'MA_20', 'BB_std', 'BB_upper', 'BB_lower',
    'normalized_close', 'standardized_close', 'lag_1', 'lag_2'
    ]

    for i, feature in enumerate(features):
        plt.figure(figsize=(15, 6))
        plt.plot(range(0, len(predictions)), predictions[:, i], 'ro', label=f'Predicted {feature}')
        plt.plot(range(len(y_test_close)), y_test_close[:, i], 'g-', label=f'Actual {feature}')
        plt.legend()
        plt.title(f'Predicted vs Actual {feature}')
        plt.xlabel('Days')
        plt.ylabel(feature)
        plt.grid(True)

        # Calculate Mean Squared Error for each feature
        mae = mean_absolute_error(y_test_close[:,  i], predictions[:, i])
        mse = mean_squared_error(y_test_close[:,  i], predictions[:, i])
        rmse = np.sqrt(mse)

        print(f"Mean Absolute Error (MAE) {feature}: {mae:.4f}")
        print(f"Mean Squared Error (MSE) {feature}: {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE) {feature}: {rmse:.4f}")

        plot_path = f'plots/testing/test_plot_multi_feature_{current_datetime}_{checkpoint_name}_{feature}.png'  # Change this path as needed
        plt.savefig(plot_path)
        plt.close()


    
    
   



'''
evaluating the model
Latest checkpoint file: models/checkpoints/TSLA_multi_feature_checkpoint_2023-09-28 14:48:54.pth
Test MSE Loss: 0.013643053360283375
Checkpoint name: TSLA_multi_feature_checkpoint_2023-09-28 14:48:54
Mean Absolute Error (MAE): 0.0800
Mean Squared Error (MSE): 0.0136
Root Mean Squared Error (RMSE): 0.1168

evaluating the model
Latest checkpoint: models/checkpoints/TSLA_multi_feature_checkpoint_2023-09-28 14:48:54.pth
Test MSE Loss: 0.013643053360283375
Mean Absolute Error (MAE) open_price: 0.0537
Mean Squared Error (MSE) open_price: 0.0054
Root Mean Squared Error (RMSE) open_price: 0.0734

Mean Absolute Error (MAE) close_price: 0.1269
Mean Squared Error (MSE) close_price: 0.0279
Root Mean Squared Error (RMSE) close_price: 0.1671

Mean Absolute Error (MAE) high_price: 0.1126
Mean Squared Error (MSE) high_price: 0.0185
Root Mean Squared Error (RMSE) high_price: 0.1360
Mean Absolute Error (MAE) low_price: 0.0827
Mean Squared Error (MSE) low_price: 0.0159
Root Mean Squared Error (RMSE) low_price: 0.1261
Mean Absolute Error (MAE) volume: 0.1172
Mean Squared Error (MSE) volume: 0.0199
Root Mean Squared Error (RMSE) volume: 0.1410
Mean Absolute Error (MAE) MA_3: 0.0646
Mean Squared Error (MSE) MA_3: 0.0064
Root Mean Squared Error (RMSE) MA_3: 0.0800
Mean Absolute Error (MAE) RSI: 0.1547
Mean Squared Error (MSE) RSI: 0.0350
Root Mean Squared Error (RMSE) RSI: 0.1870
Mean Absolute Error (MAE) EMA_12: 0.0532
Mean Squared Error (MSE) EMA_12: 0.0040
Root Mean Squared Error (RMSE) EMA_12: 0.0633
Mean Absolute Error (MAE) EMA_26: 0.0306
Mean Squared Error (MSE) EMA_26: 0.0014
Root Mean Squared Error (RMSE) EMA_26: 0.0375
Mean Absolute Error (MAE) Ellipsis: 0.0802
Mean Squared Error (MSE) Ellipsis: 0.0110
Root Mean Squared Error (RMSE) Ellipsis: 0.1050
Checkpoint name: TSLA_multi_feature_checkpoint_2023-09-28 14:48:54

'''
