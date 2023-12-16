# Import necessary libraries
import yfinance as yf
import streamlit as st
from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import sqlite3 
import random
import sys
import torch
import joblib
import os

# Adding the parent directory to the system path to access scripts
sys.path.append("..")

# Importing custom modules from the scripts directory
from scripts.data_process import seqTradeDataset, InputPreprocessor, Scaling1d, Solver
from scripts.models import make_model

# Function to hash SQLite3 connections for caching in Streamlit
def hash_sqlite3_connection(conn):
    return id(conn)

class Config:
    # Configuration for model inference
    MODEL_NAME = "SegmentBayesianHeadingModel" 
    HORIZON = 15
    WINDOW_SIZE = 30
    DEVICE = 'cuda'
    INPUT_DIMS = 9
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    NUM_SEGMENTS = 20
    BATCH_SIZE = 32
    NUM_EPOCHS = 2000
    LEARNING_RATE = 0.01
    TIME_INTERVAL = "60m"
    PERIOD = "1mo"
    START_DATE = "2022-01-01"
    END_DATE = "2022-10-01"
    TICKER = "BTC-USD"

    # Configuration for data manager
    DEFAULT_TICKER = 'AAPL'
    DEFAULT_PERIOD = '1d'
    DEFAULT_INTERVAL = '1m'
    SQLITE_DB_PATH = 'stock_data.db'
    PRICE_PREDICTION_FUTURE_DAYS = 5

    # Plotting configuration
    PLOTLY_TEMPLATE = 'plotly_dark'
    PLOT_MARGIN = dict(l=10, r=10, t=30, b=10)

    # Configuration for yfinance
    YFINANCE_TIMEOUT = 10

    # Configuration for linear regression
    LINEAR_REGRESSION_WINDOW = 5

    # Configuration for technical indicators
    MOVING_AVERAGE_WINDOW = 20
    RSI_WINDOW = 14
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9

    # Constants for scalers and paths
    SCALER_DIR = ".."
    CHECKPOINT_PATH = 'path/to/your/saved/model.pth'

    # Miscellaneous
    RANDOM_SEED = 42
    STREAMLIT_CACHE_HASH_FUNCS = {sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection}

# Setting a seed for random number generation for consistency
random.seed(Config.RANDOM_SEED)

# ModelInference class to handle model loading and inference
class ModelInference:
    def __init__(self, model_name, checkpoint_path, config=None):
        # Initialize the model with specified parameters
        if config:
            Config = config
        device = Config.DEVICE
        if not model_name:
            model_name = Config.MODEL_NAME
        self.model = make_model(name=model_name, input_dim=Config.INPUT_DIMS, hidden_dim=Config.HIDDEN_DIM, num_layers=Config.NUM_LAYERS, num_segments=Config.NUM_SEGMENTS, future_timestamps=Config.HORIZON, device=device)

        self.checkpoint_path = checkpoint_path
        self.load_model(checkpoint_path)
        self.run_inference = self.get_inference_fn(model_name)
        self.segment_scaler = Scaling1d(min_data=0, max_data=15, min_range=0, max_range=1)
        self.input_scaler = None
        self.dataset = seqTradeDataset([], just_init=True)
        # self.dataset =     eth_dataset = seqTradeDataset(processed_data, 
        #                     window_size=60, 
        #                     horizon=15)

    # Method to get the inference function based on the model name
    def get_inference_fn(self, name):
        model_classes = {'SegmentBayesianHeadingModel': self.run_inference_bayesian_heading}
        if name in model_classes:
            return model_classes[name]
        else:
            raise ValueError(f"Model name {name} not recognized!")

    # Method to load the model from the checkpoint
    def load_model(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            return self.model
        except Exception as e:
            raise ValueError(f"Failed to load the model: {e}")

    # Method for post-processing model output
    def postprocess_data(self, model_output):
        y_pred_segments, y_pred_angles, y_pred_profit, y_pred_prices = model_output
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        y_pred_angles = torch.arctan2(sin_angles, cos_angles)

        # Convert tensors to NumPy arrays
        y_pred_segments = y_pred_segments.cpu().detach().numpy()
        y_pred_angles = y_pred_angles.cpu().detach().numpy()
        y_pred_profit = y_pred_profit.cpu().detach().numpy()

        # Process prices
        mean, std = y_pred_prices
        y_pred_low = mean-std 
        y_pred_high = mean+std 
        y_pred_mean = mean.cpu().detach().numpy()
        y_pred_low = y_pred_low.cpu().detach().numpy()
        y_pred_high = y_pred_high.cpu().detach().numpy()

        # Load scalers for inverse scaling
        dir = os.path.join(os.path.split(self.checkpoint_path)[0], "..")
        input_scaler_list = joblib.load(os.path.join(dir, 'input_scaler_list.pkl'))
        y_pred_mean, y_pred_low, y_pred_high = self.inverse_scale(y_pred_mean, y_pred_low, y_pred_high, scaler_list=input_scaler_list)
        
        return y_pred_segments, y_pred_angles, y_pred_profit, y_pred_mean, y_pred_low, y_pred_high

    # Method to preprocess dataset
    # without fetching data from yfinance
    def preprocess_dataset(self, data, ticker):
        if not len(data):
            print(f"No data fetched for ticker: {ticker}")
            return None
        input_processor = InputPreprocessor()
        processed_features = input_processor(data)
        
        data_features = self.dataset.feature_engineering(data)
        dir = os.path.join(os.path.split(self.checkpoint_path)[0], "..")
        total_features = np.concatenate([np.array(data_features), processed_features], axis=-1)
        data = pd.DataFrame(data=total_features, index=data.index)
        scaled_data, scaler, date = self.dataset.preprocess_data(data, load_scaler=dir)
        self.input_scaler = scaler
        return scaled_data, scaler, date

    # Method to prune data for a specific horizon
    def prune_for_horizon(self, data, window_size, horizon, train=False):
        data = data.sort_index(ascending=True)
        if train:
            start_idx = len(data) - window_size - horizon 
        else:
            start_idx = len(data) - window_size
        start_idx = max(0, start_idx)
        pruned_data = data.iloc[start_idx:start_idx + window_size]
        return pruned_data, start_idx

    # Method to preprocess data
    def preprocess_data(self, data, ticker, train=False):
        horizon = Config.HORIZON
        window_size = Config.WINDOW_SIZE
        data_features, scaler, date = self.preprocess_dataset(data, ticker)
        data_to_prune = pd.DataFrame(data=data_features, index=data.index)
        data_features, start_idx = self.prune_for_horizon(data_to_prune, window_size, horizon, train)
        data_features = data_features.to_numpy()
        input_tensor = torch.tensor(data_features, dtype=torch.float32)
        segments_tensor = torch.tensor(data_features, dtype=torch.float32)
        return input_tensor, segments_tensor, start_idx

    # Method to run Bayesian heading inference
    def run_inference_bayesian_heading(self, data_inp, ticker, only_futures=True):
        # make copy for use
        if type(data_inp) == np.array:
            data = np.array(data_inp)
        elif type(data_inp) == pd.DataFrame:
            data = data_inp.copy()
        preprocessed_data, segments_tensor, start_idx = self.preprocess_data(data, ticker)
        preprocessed_data = preprocessed_data[:, None, ...].cuda()
        with torch.no_grad():
            model_output = self.model(preprocessed_data)

        outputs = self.postprocess_data(model_output)
        return outputs, start_idx

    # Static method for inverse scaling
    @staticmethod
    def inverse_scale(*args, column_idx=3, scaler_list=None):
        out = []
        dataset = seqTradeDataset([], just_init=True)
        for arg in args:
            out.append(dataset.inverse_transform_predictions(arg, column_idx=column_idx, scaler_list=scaler_list))
        return out

# DataManager class to manage stock data
class DataManager:
    def __init__(self, ticker=Config.DEFAULT_TICKER, period=Config.DEFAULT_PERIOD, interval=Config.DEFAULT_INTERVAL):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.conn = sqlite3.connect(Config.SQLITE_DB_PATH)
        self.c = self.conn.cursor()
        self.create_table()
        self.data = self.get_price_data()

    # Destructor to close database connection
    def __del__(self):
        self.conn.close()

    # Method to create database table
    def create_table(self):
        self.c.execute('''CREATE TABLE IF NOT EXISTS stock_data (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          ticker TEXT,
                          timestamp TEXT,
                          open REAL,
                          high REAL,
                          low REAL,
                          close REAL,
                          volume INTEGER)''')
        self.conn.commit()

    # Method to generate a random color
    def random_color(self):
        return f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'

    # Cached method to get price data from API
    @st.cache(hash_funcs=Config.STREAMLIT_CACHE_HASH_FUNCS)
    def get_price_data_from_api(self):
        try:
            data = yf.Ticker(self.ticker).history(period=self.period, interval=self.interval)
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    # @memory.cache
    # def get_price_data_from_api(self):
    #     try:
    #         data = yf.Ticker(self.ticker).history(period=self.period, interval=self.interval)
    #         return data
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return None

    # Method to validate data
    def validate_data(self, data):
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_cols)

    # Method to save data to database
    def save_to_db(self, data):
        for index, row in data.iterrows():
            self.c.execute('''INSERT INTO stock_data (ticker, timestamp, open, high, low, close, volume)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (self.ticker, index, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        self.conn.commit()

    # Method to filter trading hours
    def filter_trading_hours(self, data):
        try:    
            data.index = data.index.tz_convert('America/New_York')
        except Exception as e:
            print(e)
            print("moving on")
        return data

    # Cached method to get price data
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_price_data(self):
        self.c.execute('''SELECT * FROM stock_data WHERE ticker = ?''', (self.ticker,))
        data = self.c.fetchall()
        if data:
            columns = ['id', 'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(data, columns=columns).set_index('timestamp')
            return df.drop

            # Continue the get_price_data method
            return df.drop(columns=['id', 'ticker'])
        else:
            # Fetch data from yfinance if not in database
            data = self.get_price_data_from_api()
            if data is not None and self.validate_data(data):
                data = self.filter_trading_hours(data)
                self.save_to_db(data)
                return data
            else:
                print("Failed to retrieve or validate data.")
                return None

    # Cached method to update data
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def update_data(self):
        new_data = self.get_price_data_from_api()
        if new_data is not None and self.validate_data(new_data):
            self.save_to_db(new_data)
            self.data = new_data

    # Cached method to calculate moving average
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_moving_average(self, window):
        return self.data['Close'].rolling(window=window).mean()

    # Cached method to calculate RSI
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_RSI(self, window):
        delta = self.data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Cached method to calculate MACD
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_MACD(self):
        exp12 = self.data['Close'].ewm(span=Config.MACD_FAST_PERIOD, adjust=False).mean()
        exp26 = self.data['Close'].ewm(span=Config.MACD_SLOW_PERIOD, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=Config.MACD_SIGNAL_PERIOD, adjust=False).mean()
        return macd, signal

    # Method to predict future prices using linear regression
    def predict_prices(self):
        model = LinearRegression()
        X = np.array(range(len(self.data))).reshape(-1, 1)
        y = self.data['Close'].values
        model.fit(X, y)
        future_X = np.array(range(len(self.data), len(self.data) + Config.PRICE_PREDICTION_FUTURE_DAYS)).reshape(-1, 1)
        return model.predict(future_X)

    # Method to plot price data
    def plot_price_data(self):
        fig = go.Figure(data=[go.Candlestick(x=self.data.index,
                                             open=self.data['Open'],
                                             high=self.data['High'],
                                             low=self.data['Low'],
                                             close=self.data['Close'])])
        fig.update_layout(
            title='Price Data',
            xaxis_title='Date',
            yaxis_title='Price',
            template=Config.PLOTLY_TEMPLATE, 
            margin=Config.PLOT_MARGIN
        )
        fig.update_xaxes(rangeslider_visible=True)
        return fig

    # Method to plot moving average
    def plot_moving_average(self, window):
        ma = self.get_moving_average(window)
        fig = go.Figure(data=[go.Scatter(x=self.data.index, y=ma, mode='lines', name='MA')])
        fig.update_layout(
            title=f'Moving Average (Window: {window})',
            xaxis_title='Date',
            yaxis_title='Price',
            template=Config.PLOTLY_TEMPLATE, 
            margin=Config.PLOT_MARGIN
        )
        return fig

    def plot_macd(self, ticker, fig, row_idx):
        macd_data, signal_data = self.data.get_MACD()
        # fig.add_trace(go.Scatter(x=macd_data.index, y=macd_data, mode='lines', name=f'{ticker} MACD'), row=row_idx, col=2)
        # fig.add_trace(go.Scatter(x=signal_data.index, y=signal_data, mode='lines', name=f'{ticker} Signal Line'), row=row_idx, col=2)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(macd_data))),
                y=macd_data,
                customdata=macd_data.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                line=dict(color=self.random_color()),
                mode='lines',
                name=f'{ticker} MACD'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(signal_data))),
                y=signal_data,
                customdata=signal_data.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                line=dict(color=self.random_color()),
                mode='lines',
                name=f'{ticker} Signal Line'
            ),
            row=row_idx,
            col=2
        )
    
    def plot_rsi(self, ticker, fig, row_idx):
        rsi_data = self.data.get_RSI(14)
        # fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data, mode='lines', name=f'{ticker} RSI'), row=row_idx, col=2)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rsi_data))),
                y=rsi_data,
                customdata=rsi_data.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                line=dict(color=self.random_color()),
                mode='lines',
                name=f'{ticker} RSI'
            ),
            row=row_idx,
            col=2
        )

    def plot_price_ma(self, ticker, price_data, fig, row_idx): 
        ma_data = self.get_moving_average(20)
        fig.add_trace(
                go.Scatter(
                    x=list(range(len(price_data))),
                    y=price_data['Close'],
                    customdata=price_data.index,
                    hovertemplate='%{customdata}: %{y}<extra></extra>',
                    line=dict(color=self.random_color()),
                    mode='lines',
                    name=f'{ticker} Price'
                ),
                row=row_idx,
                col=1
            )
        fig.add_trace(
                go.Scatter(
                    x=list(range(len(ma_data))),
                    y=ma_data,
                    customdata=ma_data.index,
                    hovertemplate='%{customdata}: %{y}<extra></extra>',
                    line=dict(color=self.random_color()),
                    mode='lines',
                    name=f'{ticker} 20-day MA'
                ),
                row=row_idx,
                col=1
            )

    # Function to plot Bollinger Bands
    def plot_bollinger_bands(self, ticker, df, fig, row_idx = 0):
        window = 20
        df['Middle Band'] = df['Close'].rolling(window).mean()
        df['Upper Band'] = df['Middle Band'] + 1.96*df['Close'].rolling(window).std()
        df['Lower Band'] = df['Middle Band'] - 1.96*df['Close'].rolling(window).std()
        x_indices = list(range(len(df)))
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df['Upper Band'],
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                line=dict(color=self.random_color()),
                name='Upper Band'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df['Middle Band'],
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Middle Band'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df['Lower Band'],
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Lower Band'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df['Close'],
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Close Price'
            ),
            row=row_idx,
            col=2
        )


    # Function to plot Fibonacci Retracements
    def plot_fibonacci_retracements(self, ticker, df, fig, row_idx=0):
        peak = df['Close'].max()
        trough = df['Close'].min()
        levels = [0, 0.236, 0.382, 0.5, 0.618, 1]
        fibs = [(peak - trough) * level + trough for level in levels]
        x_indices = list(range(len(df)))
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df['Close'],
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Close Price'
            ),
            row=row_idx,
            col=2
        )
        for level, fib in zip(levels, fibs):
            fig.add_hline(
                y=fib,
                line_dash="dash",
                annotation_text=f'Fib {level}',
                row=row_idx,
                col=2
            )



    # Function to plot Ichimoku Cloud
    def plot_ichimoku_cloud(self,ticker, df, fig, row_idx=0):
        highs = df['High']
        lows = df['Low']
        tenkan_sen = (highs.rolling(window=9).max() + lows.rolling(window=9).min()) / 2
        kijun_sen = (highs.rolling(window=26).max() + lows.rolling(window=26).min()) / 2
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = (highs.rolling(window=52).max() + lows.rolling(window=52).min()) / 2
        chikou_span = df['Close'].shift(-26)

        x_indices = list(range(len(df)))
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=tenkan_sen,
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Tenkan-sen'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=kijun_sen,
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Kijun-sen'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=senkou_span_a,
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Senkou Span A'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=senkou_span_b,
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Senkou Span B'
            ),
            row=row_idx,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=chikou_span,
                customdata=df.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                name='Chikou Span'
            ),
            row=row_idx,
            col=2
        )

    def plot_predicted_prices(self, price_data, y_pred_prices_mean, y_pred_prices_low, y_pred_prices_high, fig, row_idx, start_idx):
        # Create x values (indices) for the predicted prices
        # import pdb; pdb.set_trace()
        x_indices = list(range(start_idx, start_idx+len(y_pred_prices_mean)))
        # import pdb; pdb.set_trace()
        # Plot mean predicted prices
        if len(y_pred_prices_mean.shape) == 2:
            y_pred_prices_mean = y_pred_prices_mean.reshape(-1)
            y_pred_prices_low = y_pred_prices_low.reshape(-1)
            y_pred_prices_high = y_pred_prices_high.reshape(-1)
            
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=y_pred_prices_mean,
                line=dict(color=self.random_color()),
                customdata=price_data.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                mode='lines',
                name='Mean Predicted Prices'
            ),
            row=row_idx,
            col=1  # Assuming you want to plot in the second column
        )

        # Plot high predicted prices
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=y_pred_prices_high,
                line=dict(color=self.random_color()),
                customdata=price_data.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                mode='lines',
                name='High Predicted Prices',
                fill=None  # No fill for the high prices trace
            ),
            row=row_idx,
            col=1  # Assuming you want to plot in the second column
        )

        # Plot low predicted prices
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=y_pred_prices_low,
                line=dict(color=self.random_color()),
                customdata=price_data.index,
                hovertemplate='%{customdata}: %{y}<extra></extra>',
                mode='lines',
                name='Low Predicted Prices',
                fill='tonexty'  # Fill the area between this trace and the previous trace
            ),
            row=row_idx,
            col=1  # Assuming you want to plot in the second column
        )
        return fig  # Return the updated fig object

    def plot_predicted_segments(self, price_data, y_pred_segments, y_pred_angles, fig, row_idx, start_idx, zoom_range):
        
        y_pred_segments = prune_predictions(y_pred_segments)
        # Plotting segments
        for i, segment in enumerate(y_pred_segments):
            start_idx, end_idx = segment  # Assuming segments are tuples/lists of start and end indices
            segment_slope = y_pred_angles[i]
            x_segment = [start_idx, end_idx]
            y_segment = [price_data[start_idx], price_data[start_idx] + (end_idx - start_idx) * segment_slope]
            fig.add_trace(
                go.Scatter(
                    x=x_segment,
                    y=y_segment,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=f'Segment {i+1}'
                ),
                row=row_idx,
                col=2
            )
        return fig

# Function to setup machine learning environment
def setup_ml(model_path, model_name='SegmentBayesianHeadingModel'):
    input_dim, hidden_dim, num_layers, num_segments, device = 10, 128, 2, 20, 'cpu'
    model = make_model(model_name, input_dim, hidden_dim, num_layers, num_segments, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


def prune_predictions(arr):
    # Clamp values to be within 0 and 14
    arr_clamped = np.clip(arr, 0, 14)
    
    # Split the array into fractional and integer parts
    integer_parts, fractional_parts = np.divmod(arr_clamped, 1)
    
    # Round down or up based on the fractional part
    rounded_down = np.floor(arr_clamped)
    rounded_up = np.ceil(arr_clamped)
    
    # If fractional part is <= 0.5, use rounded_down, else use rounded_up
    pruned_indices = np.where(fractional_parts <= 0.5, rounded_down, rounded_up)
    
    return pruned_indices

if __name__ == "__main__":
    # Usage example
    dm = DataManager('AAPL')
    dm.update_data()  # Call this method to update data
