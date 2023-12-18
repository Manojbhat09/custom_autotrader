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
random.seed(42)  # Replace 42 with your desired seed value
import sys
sys.path.append("..")
from scripts.data_process import seqTradeDataset, Scaling1d, Solver
from scripts.models import make_model
import torch
import joblib
import os
def hash_sqlite3_connection(conn):
    return id(conn)  # or some other way of uniquely identifying the connection objec


class ModelInference:
    def __init__(self, model_name, checkpoint_path):
        device = 'cuda'
        self.model = make_model(name=model_name, input_dim=9, hidden_dim=64, num_layers=2, num_segments=20, future_timestamps=20, device=device)
        # self.model.critera 
        self.checkpoint_path = checkpoint_path
        self.load_model(checkpoint_path)
        self.run_inference = self.get_inferenece_fn(model_name)
        self.segment_scaler = Scaling1d(min_data=0, max_data=15, min_range=0, max_range=1)
        self.input_scaler = None

    def get_inferenece_fn(self, name):
        """Fetch the model by matching the name and creating the object and returning"""
        model_classes = {
            'SegmentBayesianHeadingModel': self.run_inference_bayesian_heading
        }
        
        if name in model_classes:
            return model_classes[name]
        else:
            raise ValueError(f"Model name {name} not recognized!")

    def load_model(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            return self.model
        except Exception as e:
            raise ValueError(f"Failed to load the model: {e}")
    
    
    # def postprocess_data(self, output):
    #     # Convert tensor to NumPy array
    #     sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
    #     pred_angles = torch.arctan2(sin_angles, cos_angles)
    #     ex_pred_mean, ex_pred_std = outputs[3][0][-1, :], outputs[3][1][-1, :] # closing price selecting the last one
    #     ex_inp, ex_tgt, ex_pred_mean, ex_pred_std = self.extract_numpy(ex_inp, ex_tgt, ex_pred_mean, ex_pred_std)
    #     ex_pred_low = ex_pred_mean - ex_pred_std
    #     ex_pred_high = ex_pred_mean + ex_pred_std
    #     ex_inp, ex_tgt, ex_pred_low, ex_pred_high, ex_pred_mean = self.inverse_scale(ex_inp, ex_tgt, ex_pred_low, ex_pred_high, ex_pred_mean, column_idx=self.column_idx)
    #     return output_array
    
    def postprocess_data(self, model_output):
        # Assume model_output is a tuple of tensors
        y_pred_segments, y_pred_angles, y_pred_profit, y_pred_prices = model_output
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        y_pred_angles = torch.arctan2(sin_angles, cos_angles)

        # Convert tensors to NumPy arrays
        y_pred_segments = y_pred_segments.cpu().detach().numpy()
        y_pred_angles = y_pred_angles.cpu().detach().numpy()
        y_pred_profit = y_pred_profit.cpu().detach().numpy()
        # y_pred_prices = y_pred_prices.cpu().detach().numpy()  # Assume this contains mean, low, and high prices

        # For simplicity, let's assume y_pred_prices has shape (3, N)
        mean, std = y_pred_prices

        y_pred_low = mean-std 
        y_pred_high = mean+std 
        y_pred_mean = mean.cpu().detach().numpy()
        y_pred_low = y_pred_low.cpu().detach().numpy()
        y_pred_high = y_pred_high.cpu().detach().numpy()
        

        dir = os.path.join(os.path.split(self.checkpoint_path)[0], "..")
        input_scaler_list = joblib.load(os.path.join(dir, 'input_scaler_list.pkl'))
        input_scaler = joblib.load(os.path.join(dir, 'input_scaler.pkl'))
        # inverse scale
        y_pred_mean, y_pred_low, y_pred_high = self.inverse_scale(y_pred_mean, y_pred_low, y_pred_high, scaler_list=input_scaler_list)
        # Now you have the data in a usable format, you can return it as needed
        return y_pred_segments, y_pred_angles, y_pred_profit, y_pred_mean, y_pred_low, y_pred_high

    def preprocess_dataset(self, data, ticker):
        if not len(data):
            print(f"No data fetched for ticker: {ticker}")
            return None
        data = seqTradeDataset.feature_engineering(data)
        dir = os.path.join(os.path.split(self.checkpoint_path)[0], "..")
        scaled_data, scaler, date = seqTradeDataset.preprocess_data(data, load_scaler=dir)
        self.input_scaler = scaler
        return scaled_data, scaler, date
    
    def prune_for_horizon(self, data, window_size, horizon):
        # Ensure the data is sorted by datetime index in ascending order
        # attach index to the df
        data = data.sort_index(ascending=True)
        # import pdb; pdb.set_trace()s
        # Calculate the starting index for pruning
        start_idx = len(data) - window_size - horizon -400
        
        # Ensure the start index is not negative
        start_idx = max(0, start_idx)
        
        # Prune the data
        pruned_data = data.iloc[start_idx:start_idx + window_size]
        
        return pruned_data, start_idx
    # def check_model_compatability():
    #     # if length of the data in rows and timetstamp is not compatible with the model window, future timesteps then fit it otherwise skipZ
    #     # write checks for criteria 
        
    def preprocess_data(self, data, ticker):
        # check if the window is suffcient i.e model window input is 
        horizon = 15
        window_size = 30
        
        data_features, scaler, date = self.preprocess_dataset(data, ticker)
        data_to_prune = pd.DataFrame(data=data_features, index=data.index) # add the index of date time to sort
        data_features, start_idx=self.prune_for_horizon(data_to_prune, window_size, horizon)
        data_features= data_features.to_numpy() # removes the index of datetimeand and makes numpy
        # max_profit, segments = Solver.gen_transactions(Solver(), data_features[3], k=20)
        # # Check if seg represents a single segment or multiple segments
        # segments = seqTradeDataset.ensure_dimension(segments)
        # segments.sort(key=lambda i:i[0])

        # scaled_segments = self.segment_scale.scale(segments)
        input_tensor = torch.tensor(data_features, dtype = torch.float32)
        segments_tensor = torch.tensor(data_features, dtype = torch.float32)
        # check if the input tensor has batch size 1

        return input_tensor, segments_tensor, start_idx
    
    # def run_inference_bayesian_heading(self, data ,ticker, only_futures = True ):
    #     preprocessed_data = self.preprocess_data(data, ticker)
    #     with torch.no_grad():
    #         model_output = self.model(preprocessed_data) # prices have mean and std vectors in the tensor

    #     y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices_low, y_pred_prices_high, y_pred_prices_mean  = self.postprocess_data(model_output)
    #     outputs = [y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices_low, y_pred_prices_high, y_pred_prices_mean ]
    #     # next plot the pred prices
    #     return outputs 
    
    def run_inference_bayesian_heading(self, data, ticker, only_futures=True):
        preprocessed_data, segments_tensor, start_idx = self.preprocess_data(data, ticker)
        # add the batch dimension 

        preprocessed_data = preprocessed_data[:, None, ...].cuda()
        with torch.no_grad():
            model_output = self.model(preprocessed_data)  # Assume model takes two inputs, adjust as necessary

        outputs = self.postprocess_data(model_output)

        return outputs, start_idx
    
    @staticmethod
    def inverse_scale(*args, column_idx=3, scaler_list=None):
        out = []
        dataset = seqTradeDataset([], just_init=True) # mock dataset

        for arg in args:
            out.append( dataset.inverse_transform_predictions(arg, column_idx=column_idx, scaler_list=scaler_list)  )
        return out
    
def setup_ml():
    # Load the trained model
    model_name = 'SegmentBayesianModel'  # Replace with the name of the model you want to use
    input_dim, hidden_dim, num_layers, num_segments, device = 10, 128, 2, 20, 'cpu'  # Replace with your actual parameters
    model = make_model(model_name, input_dim, hidden_dim, num_layers, num_segments, device)
    model.load_state_dict(torch.load('path/to/your/saved/model.pth'))
    model.eval()


class DataManager:
    def __init__(self, ticker, period='1d', interval='1m'):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.conn = sqlite3.connect('stock_data.db')  # Connect to the database
        self.c = self.conn.cursor()
        self.create_table()
        self.data = self.get_price_data()

    def __del__(self):
        self.conn.close()  # Close the database connection when DataManager is destroyed

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

    
    def random_color(self):
        return f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'


    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_price_data_from_api(self): #ticker period interval
        try:
            data = yf.Ticker(self.ticker).history(period=self.period, interval=self.interval)
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def validate_data(self, data):
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_cols)

    def save_to_db(self, data):
        for index, row in data.iterrows():
            self.c.execute('''INSERT INTO stock_data (ticker, timestamp, open, high, low, close, volume)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (self.ticker, index, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        self.conn.commit()

    # def resample_data(self, data):
    #     # Resample to a regular interval, say 1 minute
    #     data_resampled = data.resample('1T').ohlc()  # '1T' denotes 1-minute intervals
    #     # Forward-fill missing values
    #     data_ffilled = data_resampled.ffill()
    #     return data_ffilled

    def filter_trading_hours(self, data):
        # Convert the index to the local timezone (EST)
        try:    
            data.index = data.index.tz_convert('America/New_York')
        except Exception as e:
            print(e)
            print("moving on")
        # Filter the data for the trading hours
        # data = data.between_time('09:30', '16:00')
        return data
    
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_price_data(self):
        # Check if data already exists in the database
        self.c.execute('''SELECT * FROM stock_data WHERE ticker = ?''', (self.ticker,))
        data = self.c.fetchall()
        if data:
            # Load data from database if it exists
            columns = ['id', 'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(data, columns=columns).set_index('timestamp')
            return df.drop(columns=['id', 'ticker'])
        else:
            # Fetch data from yfinance if not in database
            data = self.get_price_data_from_api()
            if data is not None and self.validate_data(data):
                # data = self.resample_data(data)
                # Save new data to database
                data = self.filter_trading_hours(data)
                # self.save_to_db(data)
                return data
            else:
                print("Failed to retrieve or validate data.")
                return None
    
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def update_data(self):
        new_data = self.get_price_data_from_api()
        if new_data is not None and self.validate_data(new_data):
            # self.save_to_db(new_data)
            self.data = new_data

    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_moving_average(self, window):
        return self.data['Close'].rolling(window=window).mean()
    
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_RSI(self, window):
        delta = self.data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @st.cache(hash_funcs={sqlite3.Connection: hash_sqlite3_connection, sqlite3.Cursor: hash_sqlite3_connection})
    def get_MACD(self):
        exp12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def predict_prices(self):
        model = LinearRegression()
        X = np.array(range(len(self.data))).reshape(-1, 1)
        y = self.data['Close'].values
        model.fit(X, y)
        future_X = np.array(range(len(self.data), len(self.data) + 5)).reshape(-1, 1)
        return model.predict(future_X)
    
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
            template='plotly_dark',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        fig.update_xaxes(rangeslider_visible=True)
        return fig

    def plot_moving_average(self, window):
        ma = self.data['Close'].rolling(window=window).mean()
        fig = go.Figure(data=[go.Scatter(x=self.data.index, y=ma, mode='lines', name='MA')])
        fig.update_layout(
            title=f'Moving Average (Window: {window})',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            margin=dict(l=10, r=10, t=30, b=10)
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
# Usage:
dm = DataManager('AAPL')
dm.update_data()  # Call this method to update data