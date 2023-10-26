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

def hash_sqlite3_connection(conn):
    return id(conn)  # or some other way of uniquely identifying the connection objec

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
    def get_price_data_from_api(self):
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


# Usage:
dm = DataManager('AAPL')
dm.update_data()  # Call this method to update data