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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.autograd import Variable
from scripts.data_process import seqTradeDataset, collate_fn
import matplotlib.pyplot as plt
import argparse
import glob
import os
import re

current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class MultiHeadModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_segments, future_timestamps=15):
        super(MultiHeadModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_segments)])  # Buy-Sell pairs
        self.profit_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, future_timestamps)  # Predict next 15 timestamps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :]
        action_segments = [head(out) for head in self.action_heads] # 20x2 segments each 32  
        
        profit_prediction = self.profit_head(out)  # 32 of them 32x1 
        price_prediction = self.price_head(out) # 15x1 closing price
        return action_segments, profit_prediction, price_prediction
    
def custom_loss_vanila(y_true_segments, y_pred_segments, y_true_profit, y_pred_profit, y_true_prices, y_pred_prices, sigma=1.0):
    
    segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
    profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
    price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)

    total_loss = segment_loss + profit_loss + price_loss 
    
    return total_loss

def custom_loss_weighted_price(y_true_segments, y_pred_segments, y_true_profit, y_pred_profit, y_true_prices, y_pred_prices, sigma=1.0):
    
    segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
    profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
    price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
    
    time_decay = torch.exp(-torch.arange(15).float() / sigma).cuda()  # Exponential decay
    weighted_price_loss = time_decay * nn.MSELoss(reduction='none')(y_pred_prices, y_true_prices).mean(dim=0)

    total_loss = segment_loss + profit_loss + price_loss + weighted_price_loss.sum() # (1 + delta)*price_loss
    
    return total_loss

def custom_loss_weighted_price_penalty(y_true_segments, y_pred_segments, y_true_profit, y_pred_profit, y_true_prices, y_pred_prices, sigma=1.0):
    
    segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
    profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
    price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
    
    time_decay = torch.exp(-torch.arange(15).float() / sigma).cuda()  # Exponential decay
    weighted_price_loss = time_decay * nn.MSELoss(reduction='none')(y_pred_prices, y_true_prices).mean(dim=0)

    total_loss = segment_loss + profit_loss + price_loss - weighted_price_loss.sum() # (1 - delta)*price_loss
    
    return total_loss

def custom_loss_gaussian_penalty(y_true_segments, y_pred_segments, y_true_profit, y_pred_profit, y_true_prices, y_pred_prices, sigma=1.0):
    
    segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
    profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
    price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
    gaussian_penalty_seg = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_pred_segments, y_true_segments)).sum()
    gaussian_penalty_price = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_true_prices, y_true_prices)).sum()
    gauss_penalty = gaussian_penalty_price + gaussian_penalty_seg
    max_fiter_gauss = max(gaussian_penalty_seg, gaussian_penalty_price)
    min_filter_gauss = min(gaussian_penalty_seg, gaussian_penalty_price)

    total_loss = segment_loss + profit_loss + price_loss
    max_emperical = max(segment_loss, profit_loss, price_loss)
    min_emperical = min(segment_loss, profit_loss, price_loss)

    scaled_gauss = ( (gauss_penalty - min_filter_gauss) * ( max_emperical - min_emperical ) / (max_fiter_gauss - min_filter_gauss ) ) + min_emperical
    total_loss -= (1/3) * scaled_gauss
    
    return total_loss

def custom_loss_gaussian_penalty_decay(y_true_segments, y_pred_segments, y_true_profit, y_pred_profit, y_true_prices, y_pred_prices, sigma=1.0):
    
    segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
    profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
    price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
    gaussian_penalty_seg = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_pred_segments, y_true_segments)).sum()
    gaussian_penalty_price = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_true_prices, y_true_prices)).sum()
    gauss_penalty = gaussian_penalty_price + gaussian_penalty_seg
    max_fiter_gauss = max(gaussian_penalty_seg, gaussian_penalty_price)
    min_filter_gauss = min(gaussian_penalty_seg, gaussian_penalty_price)

    time_decay = torch.exp(-torch.arange(15).float() / sigma).cuda()  # Exponential decay
    weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices, y_true_prices).mean(dim=0)).sum()

    total_loss = segment_loss + profit_loss + price_loss + weighted_price_loss
    max_emperical = max(segment_loss, profit_loss, price_loss, weighted_price_loss)
    min_emperical = min(segment_loss, profit_loss, price_loss, weighted_price_loss)

    scaled_gauss = ( (gauss_penalty - min_filter_gauss) * ( max_emperical - min_emperical ) / (max_fiter_gauss - min_filter_gauss ) ) + min_emperical
    total_loss -= (1/3) * scaled_gauss
    
    return total_loss


def postprocess(action_tensor):
    action_tensor = torch.stack(action_tensor) #stack list
    action_tensor = action_tensor.transpose(0,1)
    return action_tensor

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
    # os.makedirs('data/processed', exist_ok=True)
    # file_path = os.path.join("data", "processed", "TSLA.csv")
    # df = pd.read_csv(file_path)
    # df = pd.DataFrame(df)

    ticker = "ETH-USD"
    period = "2mo"
    start_date = "2022-01-01"
    end_date = "2022-10-01"
    time_interval = "30m"  # Daily data
    raw_data = seqTradeDataset.fetch_data(
        ticker=ticker, 
        start_date=start_date, 
        end_date=end_date, 
        time_interval=time_interval)
    engineered_data = seqTradeDataset.feature_engineering(raw_data)
    processed_data, scaler, date_data = seqTradeDataset.preprocess_data(engineered_data, drop_columns=True)   

    # Creating the Dataset and DataLoader
    eth_dataset = seqTradeDataset(processed_data, 
                            window_size=20, 
                            horizon=15)
    eth_dataloader = DataLoader(eth_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    eth_dataloader_val = DataLoader(eth_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    eth_dataloader_test = DataLoader(eth_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 9 is the features size
    # 20 segments 
    model = MultiHeadModel(input_dim=9, hidden_dim=12, num_layers=2, num_segments=20).to(device)

    # Ensure directories exist
    checkpoint_folder = 'models/checkpoints/'
    os.makedirs('models/checkpoints', exist_ok=True)
    # Assuming 'model' is your trained PyTorch model
    checkpoint_path = f'models/checkpoints/{ticker}_checkpoint_{current_datetime}.pth'

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if(args.train):
        
        print("training the model")

        train_losses = []
        test_losses = []
        num_epochs = 1000
        epoch_range = tqdm(range(num_epochs), desc="Training Progress", ncols=100, ascii=True)
        best_test_loss = float('inf')

        for epoch in epoch_range:
            epoch_range.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            
            total_train_loss = 0
            total_test_loss = 0
            
            for batch_idx, (x_batch, *y_batch) in enumerate(eth_dataloader):
                x_batch = x_batch.to(device)
                x_batch = x_batch.transpose(0, 1) # batch size in the middle [60, 32, 9]
                y_segments, y_profit, y_prices = y_batch
                y_segments, y_profit, y_prices = y_segments.to(device), y_profit.to(device), y_prices.to(device)

                # Reset the gradients
                optimizer.zero_grad()
                
                # Forward pass
                pred_segments, pred_profit, pred_prices = model(x_batch)
                pred_segments = postprocess(pred_segments)
                # Compute the loss
                loss = custom_loss(y_segments, pred_segments, y_profit, pred_profit, y_prices, pred_prices)
                
                # Backward pass
                loss.backward()
                
                # Update the weights
                optimizer.step()
                
                # Accumulate the training loss
                total_train_loss += loss.item()
            
            # Average training loss
            avg_train_loss = total_train_loss / len(eth_dataloader)
            train_losses.append(avg_train_loss)
            
            # # Print and save the losses plot every 10 epochs
            # if (epoch + 1) % 10 == 0:
            #     epoch_range.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')
            
            # Evaluation (assuming you have a separate dataloader for validation/testing)
            model.eval()
            with torch.no_grad():
                for batch_idx, (x_batch, *y_batch) in enumerate(eth_dataloader_test):
                    x_batch = x_batch.to(device)
                    x_batch = x_batch.transpose(0, 1) # batch size in the middle [60, 32, 9]
                    y_segments, y_profit, y_prices = y_batch
                    y_segments, y_profit, y_prices = y_segments.to(device), y_profit.to(device), y_prices.to(device)
                    pred_segments, pred_profit, pred_prices = model(x_batch)
                    pred_segments = postprocess(pred_segments)
                    loss = custom_loss(y_segments, pred_segments, y_profit, pred_profit, y_prices, pred_prices)
                    total_test_loss += loss.item()
            
            # Average test loss
            avg_test_loss = total_test_loss / len(eth_dataloader_test)
            test_losses.append(avg_test_loss)
            
            # Switch back to training mode
            model.train()
            
            epoch_range.set_postfix(train_loss=avg_train_loss, test_loss=avg_test_loss)
            
            if (epoch + 1) % 10 == 0:
                epoch_range.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path} at epoch {epoch + 1}")

        # Plot the training and test losses
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()

        # Save the plot in the specified folder
        plot_path = f'plots/training/loss_plot_{current_datetime}.png'  # Change this path as needed
        plt.savefig(plot_path)
        plt.close()


        torch.save(model.state_dict(), checkpoint_path)
        print("Model checkpoint saved to checkpoint_path")
        tqdm.write(f"Saved checkpoint: {checkpoint_path}")

    print("evaluating the model")

    # Define the pattern to match checkpoint file names
    pattern = fr'{ticker}_checkpoint_\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.pth'

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
        latest_checkpoint = os.path.join(checkpoint_path)
        print("No matching checkpoint files found in the folder.")

    # Load the model for evaluation
    try:
        model.load_state_dict(torch.load(latest_checkpoint))
    except Exception as e:
        print(e)
        print("Moving on")
    model.eval()

    # Test the model
    # Assuming test_dataloader is your dataloader for testing
    total_test_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(eth_dataloader_val):
            x_batch, (y_segments, y_profit, y_prices) = x_batch.to(device), (y_batch[0].to(device), y_batch[1].to(device), y_batch[2].to(device))
            pred_segments, pred_profit, pred_prices = model(x_batch)
            loss = custom_loss(y_segments, pred_segments, y_profit, pred_profit, y_prices, pred_prices)
            total_test_loss += loss.item()
            all_predictions.append(pred_prices.cpu().numpy())
            all_targets.append(y_prices.cpu().numpy())

    # Flatten the list of predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(range(0, len(all_predictions)), all_predictions, 'ro', label='Predicted Close Price')
    plt.plot(range(len(all_targets)), all_targets, 'g-', label='Actual Close Price')
    plt.legend()
    plt.title('Predicted vs Actual Close Prices')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.grid(True)

    # Save the plot in the specified folder
    os.makedirs('plots/testing', exist_ok=True)

    # Use re.search to find the pattern in the string
    match = re.search(pattern, latest_checkpoint)

    if match:
        checkpoint_name = match.group()
        print("Checkpoint name:", checkpoint_name)
    else:
        print("No checkpoint name found in the string.")

    plot_path = f'plots/testing/test_plot_{current_datetime}_{checkpoint_name}.png'  # Change this path as needed
    plt.savefig(plot_path)
    plt.close()

'''
evaluating the model
Latest checkpoint: models/checkpoints/TSLA_checkpoint_2023-09-28 14:09:54.pth
Test MSE Loss: 0.011901536956429482
Checkpoint name: TSLA_checkpoint_2023-09-28 14:09:54.pth
Mean Absolute Error (MAE): 0.0672
Mean Squared Error (MSE): 0.0119
Root Mean Squared Error (RMSE): 0.1091
'''