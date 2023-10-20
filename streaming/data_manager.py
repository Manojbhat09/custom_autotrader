import yfinance as yf
from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import sqlite3 

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
                # Save new data to database
                self.save_to_db(data)
                return data
            else:
                print("Failed to retrieve or validate data.")
                return None

    def update_data(self):
        new_data = self.get_price_data_from_api()
        if new_data is not None and self.validate_data(new_data):
            self.save_to_db(new_data)
            self.data = new_data
        
    def get_moving_average(self, window):
        return self.data['Close'].rolling(window=window).mean()
        
    def get_RSI(self, window):
        delta = self.data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
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

# Usage:
dm = DataManager('AAPL')
dm.update_data()  # Call this method to update data