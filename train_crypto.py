'''
Why Input and Output Dimension Might Be 1:

In many stock price forecasting tasks, the goal is to predict the next day's closing price based on past closing prices. In such cases, you might only use the closing price as a feature, leading to an input dimension of 1. Similarly, since you're predicting a single value (next day's closing price), the output dimension would also be 1.
However, if you're using multiple technical indicators (like Moving Averages, RSI, MACD, etc.) as features, then the input dimension should match the number of features (e.g., 18).

usage:
 python train_crypto.py --debug --train --expname height_angle_loss --model SegmentHeadingModel
 python train_crypto.py --eval --debug --expname incr_hidden_bayesian --model SegmentBayesianHeadingModel --checkpoint_path run/experiment_20231030_095433/models/ETH-USD_checkpoint_experiment_20231030_095433.pth
 python train_crypto.py --debug --train --expname="bayes_head_15m_60_15_64_geom_mse_profit_penalty_loss" --model SegmentBayesianHeadingModel
 '''
from scripts.loss_fn import LossFunctions
from scripts.models import MultiHeadModel, make_model
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
from scripts.data_process import seqTradeDataset, collate_fn, collate_fn_angled
import matplotlib.pyplot as plt
import argparse
from logging.handlers import RotatingFileHandler
from torch.utils.tensorboard import SummaryWriter
import logging
import glob
import shutil
import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
import torchvision
import warnings 
warnings.filterwarnings("ignore") 

# Define the root directory for this run
def generate_experiment_name():
    import datetime
    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'experiment_{current_datetime}'
    return experiment_name
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_datetime = generate_experiment_name()
run_dir = f'run/{current_datetime}'

# Define sub-directories
ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FOLDER = os.path.join(run_dir, 'models')
PLOT_FOLDER = os.path.join(run_dir, 'plots')
LOG_DIR = os.path.join(run_dir, 'logs')
TB_LOG_DIR = os.path.join(run_dir, 'tb_logs')
TRAINING_PLOT_FOLDER = os.path.join(PLOT_FOLDER, 'training')
TESTING_PLOT_FOLDER = os.path.join(PLOT_FOLDER, 'testing')
RESULTS_FOLDER = os.path.join(run_dir, 'results')
SCRIPTS_FOLDER = os.path.join(run_dir, 'results', 'scripts')

def assign_variables(test):
    global ROOT, CHECKPOINT_FOLDER, PLOT_FOLDER, LOG_DIR, TB_LOG_DIR, TRAINING_PLOT_FOLDER, TESTING_PLOT_FOLDER, RESULTS_FOLDER, SCRIPTS_FOLDER, current_datetime, run_dir

    if not test:
        # Create sub-directories
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
        os.makedirs(PLOT_FOLDER, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(TRAINING_PLOT_FOLDER, exist_ok=True)
        os.makedirs(TESTING_PLOT_FOLDER, exist_ok=True)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        os.makedirs(SCRIPTS_FOLDER, exist_ok=True)

def copy_files(SCRIPTS_FOLDER):
    # Copy scripts to SCRIPT_FOLDER
    script_name = os.path.basename(__file__)
    shutil.copy(script_name, f'{SCRIPTS_FOLDER}/{script_name}')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(current_dir, "scripts", "loss_fn.py"), os.path.join(SCRIPTS_FOLDER, "loss_fn.py"))
    shutil.copy(os.path.join(current_dir, "scripts", "models.py"), os.path.join(SCRIPTS_FOLDER, "models.py"))
    shutil.copy(os.path.join(current_dir, "scripts", "data_process.py"), os.path.join(SCRIPTS_FOLDER, "data_process.py"))

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def close(self):
        self.writer.close()

    def log_images(self, tag, images, step):
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_grid, step)

    def log_plot(self, tag, fig, step):
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = Image.frombytes("RGBA", canvas.get_width_height(), buf.tobytes(), "raw", "RGBA")
        image_tensor = torchvision.transforms.ToTensor()(image)
        self.writer.add_image(tag, image_tensor, step)

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device, tb_logger):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tb_logger = tb_logger

    def train(self, epoch):

        total_train_loss = 0
        for batch_idx, data in enumerate(self.dataloader):
            x_batch = data[0].to(self.device)
            x_batch = x_batch.transpose(0, 1)  # batch size in the middle [60, 32, 9]
            y_batch = [item.to(self.device) for item in data[1:]]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x_batch)

            # Compute the loss
            loss = self.criterion(*y_batch, *outputs, epoch=epoch)
            loss.backward()
            self.optimizer.step()
            total_train_loss += loss.item()

        return total_train_loss

    @staticmethod
    def postprocess(outputs):
        # criterion: *y_batch, *outputs
        # y_batch: segments, angles, profits, prices. outputs = segments, angles, profits, prices
        action_tensor = outputs[0]
        action_tensor = torch.stack(action_tensor) #stack list
        action_tensor = action_tensor.transpose(0,1)
        outputs[0] = action_tensor

class Evaler:
    def __init__(self, model, dataloader, criterion, device, tb_logger, price_column=3):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.tb_logger = tb_logger
        self.column_idx = price_column

    def eval(self, epoch):
        total_test_loss = 0
        for batch_idx, data in enumerate(self.dataloader):
            x_batch = data[0].to(self.device)
            x_batch = x_batch.transpose(0, 1)  # batch size in the middle [60, 32, 9]
            y_batch = [item.to(self.device) for item in data[1:]]

            outputs = self.model(x_batch)
            loss = self.criterion(*y_batch, *outputs, epoch=batch_idx)
            total_test_loss += loss.item()

            x_batch = x_batch.transpose(0, 1)
            ex_inp, ex_tgt, ex_pred_mean, ex_pred_std = x_batch[-1][:, 3], y_batch[4][-1, :], outputs[3][0][-1, :], outputs[3][1][-1, :] # closing price selecting the last one
            ex_inp, ex_tgt, ex_pred_mean, ex_pred_std = self.extract_numpy(ex_inp, ex_tgt, ex_pred_mean, ex_pred_std)
            ex_pred_low = ex_pred_mean - ex_pred_std
            ex_pred_high = ex_pred_mean + ex_pred_std
            ex_inp, ex_tgt, ex_pred_low, ex_pred_high, ex_pred_mean = self.inverse_scale(ex_inp, ex_tgt, ex_pred_low, ex_pred_high, ex_pred_mean, column_idx=self.column_idx)
            plot_tensorboard_bayes(ex_inp, ex_tgt, [ex_pred_low, ex_pred_high, ex_pred_mean], self.tb_logger, epoch * len(self.dataloader) + batch_idx)
        return total_test_loss

    def extract_numpy(self, *args):
        return [arg.cpu().numpy() for arg in args]
    
    def inverse_scale(self, *args, column_idx=3):
        out = []
        for arg in args:
            out.append( self.dataloader.dataset.inverse_transform_predictions(arg, column_idx=column_idx)  )
        return out
    

    def test_model(self, model, dataloader, device):
        model.eval()
        all_predictions = []
        all_targets = []
        total_test_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):

                x_batch = data[0].to(self.device)
                x_batch = x_batch.transpose(0, 1)  # batch size in the middle [60, 32, 9]
                y_batch = [item.to(self.device) for item in data[1:]]

                outputs = self.model(x_batch)
                loss = self.criterion(*y_batch, *outputs, epoch=batch_idx)
                total_test_loss += loss.item()

                x_batch = x_batch.transpose(0, 1)
                ex_inp, ex_tgt, ex_pred_mean, ex_pred_std = x_batch[-1][:, 3], y_batch[4][-1, :], outputs[3][0][-1, :], outputs[3][1][-1, :] # closing price selecting the last one
                ex_inp, ex_tgt, ex_pred_mean, ex_pred_std = self.extract_numpy(ex_inp, ex_tgt, ex_pred_mean, ex_pred_std)
                ex_pred_low = ex_pred_mean - ex_pred_std
                ex_pred_high = ex_pred_mean + ex_pred_std
                ex_inp, ex_tgt, ex_pred_low, ex_pred_high, ex_pred_mean = self.inverse_scale(ex_inp, ex_tgt, ex_pred_low, ex_pred_high, ex_pred_mean, column_idx=self.column_idx)
                plot_tensorboard_bayes_test(ex_inp, ex_tgt, [ex_pred_low, ex_pred_high, ex_pred_mean], self.tb_logger, batch_idx)

                all_predictions.append(ex_pred_mean)
                all_targets.append(ex_tgt)

            # Flatten the list of predictions and targets
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # Calculate metrics
            mae = mean_absolute_error(all_targets, all_predictions)
            mse = mean_squared_error(all_targets, all_predictions)
            rmse = np.sqrt(mse)
            save_results(mae, mse, rmse, 'evaluation_results.txt')
            return mae, mse, rmse

def setup_logging(log_dir, log_level=logging.DEBUG):
    log_file_path = os.path.join(log_dir, 'run.log')
    logger = logging.getLogger('')
    logger.setLevel(log_level)
    
    # File Handler with rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)  # 10 MB per log file, keep 5 old log files
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info('Logging setup complete.')

def log_dict(data, logger, log_level=logging.INFO):
    for key, value in data.items():
        logger.log(log_level, f"{key}: {value}")

def plot_results(train_losses, test_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plot_path = os.path.join(TRAINING_PLOT_FOLDER, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

def postprocess(action_tensor):
    action_tensor = torch.stack(action_tensor) #stack list
    action_tensor = action_tensor.transpose(0,1)
    return action_tensor

def save_results(mae, mse, rmse, filename):
    result_file_path = os.path.join(RESULTS_FOLDER, filename)
    with open(result_file_path, 'w') as file:
        file.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        file.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        file.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logging.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


def fetch_and_preprocess_data(ticker="ETH-USD", start_date="2022-01-01", end_date="2022-10-01", time_interval="30m", period='2mo'):
    # Fetching the data
    processed_data = seqTradeDataset.fetch_and_preprocess_data(
        ticker=ticker, 
        period=period, 
        start_date=start_date, 
        end_date=end_date, 
        time_interval=time_interval, 
        run_dir=run_dir)
    if not len(processed_data):
        logging.error(f"No data fetched for ticker: {ticker}")
        return None, None
    
    logging.info(f'Data fetched and preprocessed for ticker: {ticker}')
    return processed_data

def plot_time_series_with_predictions(x, y_true, y_pred):
    # Assuming x, y_true, and y_pred are 1D arrays or list-like objects
    total_length = len(x) + len(y_true)  # Total length of the x-axis
    x_axis = np.arange(total_length)  # Generating x-axis values

    fig, ax = plt.subplots()
    ax.plot(x_axis[:len(x)], x, label='Historical')
    ax.plot(x_axis[len(x):], y_true, label='True Future')
    ax.plot(x_axis[len(x):], y_pred, label='Predicted Future', linestyle='dashed')

    ax.legend()
    plt.show()
    return fig

def plot_time_series_with_predictions_bayes(x, y_true, y_pred):
    y_low, y_high, y_pred_mean = y_pred
    total_length = len(x) + len(y_true)  # Total length of the x-axis
    x_axis = np.arange(total_length)  # Generating x-axis values

    fig, ax = plt.subplots()
    ax.plot(x_axis[:len(x)], x, label='Historical')
    ax.plot(x_axis[len(x):], y_true, label='True Future')
    ax.plot(x_axis[len(x):], y_pred_mean, label='Predicted Mean', linestyle='dashed')

    # Plotting uncertainty intervals (1 standard deviation)
    ax.fill_between(
        x_axis[len(x):],
        y_low,
        y_high,
        color='gray',
        alpha=0.5,
        label='1 Std Dev'
    )

    ax.legend()
    plt.show()
    return fig


def plot_tensorboard(x_batch, y_prices, pred_prices, tb_logger, timestamp):
    fig = plot_time_series_with_predictions(x_batch, y_prices, pred_prices)
    tb_logger.log_plot('Validation/TimeSeriesWithPredictions', fig, timestamp)
    plt.close(fig)  # Close the figure to free memory


def plot_tensorboard_bayes(x_batch, y_prices, pred_prices, tb_logger, timestamp):
    fig = plot_time_series_with_predictions_bayes(x_batch, y_prices, pred_prices)
    tb_logger.log_plot('Validation/TimeSeriesWithPredictions', fig, timestamp)
    plt.close(fig)  # Close 

def plot_tensorboard_bayes_test(x_batch, y_prices, pred_prices, tb_logger, timestamp):
    fig = plot_time_series_with_predictions_bayes(x_batch, y_prices, pred_prices)
    fig.savefig( os.path.join(TESTING_PLOT_FOLDER, f"pred_plot_{timestamp}.png") )
    tb_logger.log_plot('Test/TimeSeriesWithPredictions', fig, timestamp)
    plt.close(fig)  # Close 

def main(args):
    global PLOT_FOLDER, TESTING_PLOT_FOLDER, RESULTS_FOLDER
   
    log_level = logging.DEBUG if args.debug else logging.INFO  # Assuming you have a debug flag
    assign_variables(args.eval)
    
    if not args.eval:
        setup_logging(LOG_DIR)
        added_experiment_name = args.expname+"_"+args.model
        tb_log_dir= os.path.join(TB_LOG_DIR, current_datetime+"_"+added_experiment_name)
        tb_logger = TensorBoardLogger(tb_log_dir)

    if 'head' in args.model.lower() or 'angle' in args.model.lower():
        args.add_angle = True
        collate = collate_fn_angled
    else:
        args.add_angle = False
        collate = collate_fn


    # Assuming df is your dataframe with the features
    # Fetch and preprocess data
    Ticker = "BTC-USD"
    horizon = 15
    window_size = 60
    batch_size = 32
    start_date = "2022-01-01" # ignored
    end_date = "2022-10-01" # ignored
    time_interval = "15m"
    period = "1mo"  
    num_segments = 20
    hidden_dim = 64
    num_layers = 5
    input_dims = 18
    processed_data = fetch_and_preprocess_data(ticker=Ticker, 
                                                   start_date = start_date, 
                                                   end_date = end_date, 
                                                   time_interval = time_interval, 
                                                   period = period)
    # Creating the Dataset and DataLoader
    eth_dataset = seqTradeDataset(processed_data, 
                            window_size=window_size, 
                            horizon=horizon, 
                            add_angles=args.add_angle)
    eth_dataloader = DataLoader(eth_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    eth_dataloader_val = DataLoader(eth_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    eth_dataloader_test = DataLoader(eth_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # 9 is the features size
    # 20 segments 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(CHECKPOINT_FOLDER,  f'{Ticker}_checkpoint_{current_datetime}.pth')

    # Model setups
    model =  make_model(name=args.model, input_dim=input_dims, hidden_dim=hidden_dim, num_layers=num_layers, num_segments=num_segments, future_timestamps=horizon, device=device) # increase hidden to 32 for bayes

    model.train()
    if args.train:
        logging.info("training the model")
        copy_files(SCRIPTS_FOLDER)
        
        train_losses = []
        test_losses = []
        num_epochs = 2000
        epoch_range = tqdm(range(num_epochs), desc="Training Progress", ncols=100, ascii=True)
        best_test_loss = float('inf')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = LossFunctions('geometric_bayesian_weighted', tb_logger=tb_logger, horizon=horizon)
        criterion = loss_fn.get_loss_function()
        trainer = Trainer(model, eth_dataloader, optimizer, criterion, device, tb_logger)
        evaler = Evaler(model, eth_dataloader_val, criterion, device, tb_logger)

        for epoch in epoch_range:
            epoch_range.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            total_test_loss = 0
            
            total_train_loss = trainer.train(epoch)
            # Average training loss
            avg_train_loss = total_train_loss / len(eth_dataloader)
            train_losses.append(avg_train_loss)

            # Log training loss to TensorBoard
            tb_logger.log_scalar('Loss/train', avg_train_loss, epoch)
            loss_fn.log_losses(epoch, 'train')
            
            if (epoch + 1) % 200 == 0:
                # Log histograms of activations
                for name, module in model.named_modules():
                    if isinstance(module, nn.ReLU):  # Change this to whatever activation your model uses
                        input_tensor = torch.randn(1, 9)  # Assuming input size of 9, adjust as necessary
                        input_tensor = input_tensor.to(device)
                        activation = module(input_tensor)
                        tb_logger.log_histogram(f'Activations/{name}', activation, epoch)

                # In your training loop, after loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        tb_logger.log_histogram(f'Gradients/{name}', param.grad, epoch)
            
            # Evaluation (assuming you have a separate dataloader for validation/testing)
            model.eval()
            with torch.no_grad():
                total_test_loss = evaler.eval(epoch)    
            # Average test loss
            avg_test_loss = total_test_loss / len(eth_dataloader_val)
            test_losses.append(avg_test_loss)
            tb_logger.log_scalar('Loss/test', avg_test_loss, epoch)
            loss_fn.log_losses(epoch, 'val')
            
            # Switch back to training mode
            model.train()
            epoch_range.set_postfix(train_loss=avg_train_loss, test_loss=avg_test_loss)
            
            if (epoch + 1) % 10 == 0:
                epoch_range.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Saved checkpoint: {checkpoint_path} at epoch {epoch + 1}")

        # After training/evaluation, save loss trajectories and metrics to CSV
        loss_df = pd.DataFrame({
            'epoch': list(range(1, num_epochs + 1)),
            'train_loss': train_losses,
            'test_loss': test_losses
        })
        loss_df.to_csv(f'{RESULTS_FOLDER}/loss_trajectories.csv', index=False)

        tb_logger.close()  # Close TensorBoard logger
            
        plot_results(train_losses, test_losses)

        # Assuming val_dataloader is your dataloader for validation
        mae, mse, rmse = evaler.test_model(model, eth_dataloader_val, device)
        print("done")

    elif args.eval:
        logging.info("Testing started")
        experiment_location = os.path.join(os.path.dirname(args.checkpoint_path), "..")
        TESTING_FOLDER = os.path.join(experiment_location, 'testing')
        TESTING_PLOT_FOLDER = os.path.join(TESTING_FOLDER, "plots")
        RESULTS_FOLDER = TESTING_FOLDER
        os.makedirs(TESTING_FOLDER, exist_ok=True)
        copy_files(TESTING_FOLDER)
        tb_log_dir= os.path.join(experiment_location, "tb_logs")
        log_folder_names = os.listdir(tb_log_dir)

        tblog_experiment_folder = log_folder_names[0]
        tb_dir = os.path.join(tb_log_dir, tblog_experiment_folder)
        tb_logger = TensorBoardLogger(tb_dir)
        loss_fn = LossFunctions('geometric_bayesian', tb_logger=tb_logger, horizon=horizon)
        criterion = loss_fn.get_loss_function()
        
        # Ensure these directories exist
        os.makedirs(PLOT_FOLDER, exist_ok=True)
        os.makedirs(TESTING_PLOT_FOLDER, exist_ok=True)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        # Load and evaluate the model
        try:
            model.load_state_dict(torch.load(args.checkpoint_path))
            evaler = Evaler(model, eth_dataloader_val, criterion, device, tb_logger)
        except Exception as e:
            logging.exception("An error occurred while loading the model checkpoint.")
            logging.info("Moving on")
        mae, mse, rmse = evaler.test_model(model, eth_dataloader_val, device)
        print("done")
        tb_logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a model")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint for evaluation")
    parser.add_argument("--expname", type=str, default="default_name", help="Experiment name")
    parser.add_argument("--model", type=str, default="SegmentHeadingModel", help="model name to match")
    args = parser.parse_args()
    main(args)


'''
evaluating the model
Latest checkpoint: models/checkpoints/TSLA_checkpoint_2023-09-28 14:09:54.pth
Test MSE Loss: 0.011901536956429482
Checkpoint name: TSLA_checkpoint_2023-09-28 14:09:54.pth
Mean Absolute Error (MAE): 0.0672
Mean Squared Error (MSE): 0.0119
Root Mean Squared Error (RMSE): 0.1091


write an advanced algorithm, be very very creative in loss function and model, browse if needed for more context and knowledge 

'''