import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from finrl.meta.data_processor import YahooFinance
import yfinance as yf
import math
import pandas
import logging
import joblib
import os

import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pywt
import numpy as np

from sklearn.impute import SimpleImputer

MAX_TRANSACTIONS = 10

class Scaling1d:
    def __init__(self, min_data, max_data, min_range, max_range):
        self.min_data = min_data
        self.max_data = max_data
        self.min_range = min_range
        self.max_range = max_range

    def scale(self, data):
        data_type, shape = self._get_data_type_and_shape(data)
        scaled_data = self._scale_data(data)
        return self._restore_data_type_and_shape(scaled_data, data_type, shape)

    def inverse_scale(self, scaled_data):
        data_type, shape = self._get_data_type_and_shape(scaled_data)
        original_data = self._inverse_scale_data(scaled_data)
        return self._restore_data_type_and_shape(original_data, data_type, shape)

    def _scale_data(self, data):
        data = self._convert_to_tensor(data)
        scaled_data = self.min_range + ((data - self.min_data) / (self.max_data - self.min_data)) * (self.max_range - self.min_range)
        return scaled_data

    def _inverse_scale_data(self, scaled_data):
        scaled_data = self._convert_to_tensor(scaled_data)
        original_data = self.min_data + ((scaled_data - self.min_range) / (self.max_range - self.min_range)) * (self.max_data - self.min_data)
        return original_data

    def _get_data_type_and_shape(self, data):
        if isinstance(data, torch.Tensor):
            return torch.Tensor, data.shape
        elif isinstance(data, np.ndarray):
            return np.ndarray, data.shape
        elif isinstance(data, list):
            return list, np.array(data).shape
        else:
            raise ValueError('Unsupported data type')

    def _restore_data_type_and_shape(self, data, data_type, shape):
        if data_type is torch.Tensor:
            return data.reshape(shape)
        elif data_type is np.ndarray:
            return data.cpu().numpy().reshape(shape)
        elif data_type is list:
            return data.cpu().numpy().reshape(shape).tolist()

    def _convert_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            return torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError('Unsupported data type')

    def set_scaling_params(self, min_data, max_data, min_range, max_range):
        self.min_data = min_data
        self.max_data = max_data
        self.min_range = min_range
        self.max_range = max_range

    def get_scaling_params(self):
        return self.min_data, self.max_data, self.min_range, self.max_range

class InputPreprocessor(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.pca = PCA(n_components=5) 
        self.scaler = StandardScaler()
        
    def forward(self, x):
        x = np.array(x)
        # FFT for frequency components
        fft_features = np.fft.rfft(x[:, [0,4]]) 
        fft_features = fft_features.real
        
        # Wavelet encoding
        coeffs = pywt.wavedec(x[:, 3], 'db1', level=2)
        wavelet_features = []
        for coeff in coeffs:
            wavelet_features.append(coeff) 
        
        # Differential encoding
        diffs = np.diff(x[:,3])
        
        # PCA dimensionality reduction
        x_scaled = self.scaler.fit_transform(x)
        pca_features = self.pca.fit_transform(x_scaled)
        
        # Concat all features
        wavelet_array = np.concatenate(wavelet_features, axis=0) 
        max_shape = x.shape[0]

        fft_features = np.pad(fft_features, [(0, max_shape - fft_features.shape[0]), (0,0)])
        # wavelet_array = np.pad(wavelet_array, [(0, max_shape - wavelet_array.shape[0])])  
        wavelet_array = wavelet_array[:max_shape]
        diffs = np.pad(diffs, [(0, max_shape - diffs.shape[0])])
        pca_features = np.pad(pca_features, [(0, max_shape - pca_features.shape[0]), (0,0)])

        features = np.concatenate([fft_features, wavelet_array[:, None], diffs[:, None], pca_features], axis=-1)
        # total_features = np.concatenate([x, features], axis=-1)
        return features

class seqTradeDataset(Dataset):
    def __init__(self, data, 
                 window_size=50, 
                 horizon=15, 
                 transform=None, 
                 ticker=None,
                 max_transactions=5, 
                 add_angles=True, just_init=False):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.transform = transform
        self.processor = YahooFinance()
        self.preprocessor = Solver()
        self.max_transactions = max_transactions
        self.add_angles = add_angles
        MAX_TRANSACTIONS = self.max_transactions
        self.segment_scaler = Scaling1d(min_data=0, max_data=self.horizon, min_range=0, max_range=1)
        if just_init:
            return
        if not hasattr(self, "input_scaler"):
            self.input_scaler = MinMaxScaler()

        if len(self.data) <=1:
            self.data = self.fetch_data()

        if len(self.data) < self.window_size + self.horizon:
            raise ValueError(f'Insufficient data: {len(self.data)} data points found, {self.window_size + self.horizon} required.')
        
        assert isinstance(self.data, np.ndarray), f'Expected data of type np.ndarray, got {type(self.data)}'

        if np.isnan(self.data).any():
            raise ValueError('Data contains NaN values.')

        if self.data.shape[1] <= 3 or not np.issubdtype(self.data[:, 3].dtype, np.number):
            raise ValueError("Expected numeric 'Close' price data at index 3")

        assert isinstance(window_size, int) and window_size > 0, f'Expected window_size to be a positive integer, got {window_size}'
        assert isinstance(horizon, int) and horizon > 0, f'Expected horizon to be a positive integer, got {horizon}'
        assert isinstance(max_transactions, int) and max_transactions > 0, f'Expected max_transactions to be a positive integer, got {max_transactions}'

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        # print(f'Accessing index {idx}, dataset size is {len(self.data)}')
        if idx < 0 or idx >= len(self.data) - self.window_size - self.horizon + 1:
            raise IndexError(f'Index {idx} out of bounds!')

        if idx + self.window_size + self.horizon > len(self.data):
            raise IndexError(f'Insufficient future data for index {idx}.')

        # Extract the past window of stock prices
        x = torch.tensor(self.data[idx:idx+self.window_size, :], dtype=torch.float32)  # Assuming 'Close' price is at index 3
        # Generate segments for the given index
        max_profit, segments = self.generate_segments(idx)
        closing_prices =self.data[idx:idx+self.window_size, 3]
        if self.add_angles:
            heights, angles = self.generate_heights_angles(segments, closing_prices)  # Passing closing prices

        # Prepare the future price prediction target
        future_prices = self.data[idx+self.window_size:idx+self.window_size+self.horizon, 3]
        y_price_prediction = torch.tensor(future_prices, dtype=torch.float32)
        
        # Conditionally apply a transform to x, if provided
        if self.transform:
            x = self.transform(x)

        
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_segments = self.segment_scaler.scale(segments)
        
        # Package segments, max profit, and price prediction target into a single tuple
        if self.add_angles:
             y = (torch.tensor(scaled_segments, dtype=torch.float32), 
                torch.tensor(heights, dtype=torch.float32), 
                torch.tensor(angles, dtype=torch.float32), 
                torch.tensor(max_profit, dtype=torch.float32), 
                y_price_prediction)
        else:
            y = (torch.tensor(scaled_segments, dtype=torch.float32), torch.tensor(max_profit, dtype=torch.float32), y_price_prediction)
        
        return x, y
    
    @classmethod
    def fetch_data(cls, downloader=None, 
                   ticker="AAPL", 
                   ticker_list=None, 
                   period='1mo', 
                   start_date=None, 
                   end_date=None, 
                   extract_details=False,
                   time_interval=None):
        logger = logging.getLogger(__name__)
        logger.debug("Fetching data with the following parameters:")
        logger.debug(f"Downloader: {downloader}, Ticker: {ticker}, Ticker List: {ticker_list}, "
                     f"Period: {period}, Start Date: {start_date}, End Date: {end_date}, "
                     f"Extract Details: {extract_details}, Time Interval: {time_interval}")

        if downloader:
            processor = downloader()
        else:
            processor = YahooFinance()

        if ticker_list and type(ticker_list) == list:
            tickers = ticker_list
        else:
            ticker_list = [ticker]
        if period:
            logger.debug(f"Fetching data for period: {period}")
            if len(ticker_list) == 1:
                data = yf.Ticker(ticker_list[0]).history(period, interval=time_interval if time_interval else '1d')
                logger.debug(f"Data fetched for {ticker_list[0]} for period: {period}")
            else:
                data = []
                interval = time_interval if time_interval else '1d'
                for ticker in ticker_list:
                    data_piece = yf.Ticker(ticker).history(period, interval=interval)
                    data.append(data_piece)
                    logger.debug(f"Data fetched for {ticker} for period: {period}")
        else:
            if extract_details:
                data = []
                if len(ticker_list) == 1:
                    data = yf.Ticker(ticker_list[0])
                else:
                    for ticker in ticker_list:
                        data.append(yf.Ticker(ticker))
                logger.debug("Extracted details for the specified tickers.")
                return data
            logger.debug("Fetching data for custom date range.")
            data = processor.download_data(ticker_list, start_date, end_date, time_interval)

        logger.debug("Data fetching completed.")
        return data

    def generate_segments(self, idx):
        assert isinstance(idx, int) and idx >= 0, f'Expected idx to be a non-negative integer, got {idx}'
        assert idx + self.window_size + self.horizon <= len(self.data), f'Index {idx} out of bounds!'
        
        series = self.data[idx+self.window_size:idx+self.window_size+self.horizon, 3]  # TODO Considering 'Close' price for segments
        
        if series.size == 0 or self.max_transactions == 0:
            return 0, []
        
        if len(series) < 2:  # At least two data points are required to generate segments
            return 0, []
        
        def ensure_dimension(seg):
            seg_tensor = torch.tensor(seg, dtype=torch.float32)
            if len(seg_tensor.shape) == 1:  # If seg is 1D
                seg_tensor = seg_tensor.unsqueeze(0)  # Convert to 2D
            elif len(seg_tensor.shape) == 2:  # If seg is 2D
                pass
            return seg_tensor.tolist()

        try:
            assert hasattr(self.preprocessor, 'gen_transactions'), 'Preprocessor object lacks gen_transactions method'
            max_profit, segments = self.preprocessor.gen_transactions(series, self.max_transactions)
            # Check if seg represents a single segment or multiple segments
            segments = ensure_dimension(segments)
            segments.sort(key=lambda i:i[0])
        except Exception as e:
            raise RuntimeError(f'Failed to generate segments: {e}')
        
        if max_profit is None or segments is None:
            return 0, []
        
        assert isinstance(max_profit, (int, float)), f'Expected max_profit of type int or float, got {type(max_profit)}'
        assert isinstance(segments, list) and all(isinstance(segment, list) and len(segment) == 2 for segment in segments), 'Invalid segments format'
        return max_profit, segments

    @classmethod
    def generate_segments_fn(data, window_size, horizon, k):
        segments_data = []
        for i in range(len(data) - window_size - horizon + 1):
            window_data = data[i:i + window_size]
            future_data = data[i + window_size:i + window_size + horizon]
            solver = Solver()
            max_profit, transactions = solver.gen_transactions(k, window_data)
            segments_data.append({
                "window_data": window_data,
                "future_data": future_data,
                "max_profit": max_profit,
                "transactions": transactions
            })
        return segments_data

    @staticmethod
    def generate_heights(segments, closing_prices):
        heights = []
        for seg in segments:
            height = closing_prices[seg[1]] - closing_prices[seg[0]]
            heights.append(height)
        
        return np.array(heights)
    
    @staticmethod
    def generate_heights_angles(segments, closing_prices):

        heights = []
        for seg in segments:

            height = closing_prices[ int(seg[1]) ] - closing_prices[ int(seg[0]) ]
            heights.append(height)
        
        # Ensure no division by zero
        index_differences = np.maximum(np.array([seg[1] - seg[0] for seg in segments]), 1)
        price_differences = np.array([ closing_prices[ int(seg[1]) ] - closing_prices[ int(seg[0]) ] for seg in segments])
        angles = np.arctan(price_differences / index_differences)
        
        return np.array(heights), angles
    
    # Feature Engineering
    @classmethod
    def feature_engineering(self, data):
        data['Daily Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily Return'].rolling(window=10).std()
        # ... other feature engineering steps
        return data

    # Preprocessing
    @classmethod
    def preprocess_data_(self, data, drop_columns=False):
        data.dropna(inplace=True)
        data.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
        date=None
        if drop_columns:
            date = data.index.values
            data = data.drop(columns=['date', 'begins_at', 'session', 'interpolated', 'symbol'], errors='ignore')
        scaler = MinMaxScaler()

        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        self.input_scaler = scaler
        return scaled_data, self.input_scaler, date
    
    def inverse_transform_predictions_(self, y_pred_scaled):
        # Ensure the input_scaler has been fitted
        if self.input_scaler.scale_ is None or self.input_scaler.min_ is None:
            raise ValueError("The input_scaler has not been fitted yet.")
        # Reverse the scaling of the predicted prices
        y_pred = self.input_scaler.inverse_transform(y_pred_scaled)
        return y_pred
    
    @classmethod
    def preprocess_data(cls, data, drop_columns=False, run_dir=None, load_scaler=''):
        imputer = SimpleImputer(strategy='mean')  # or use 'median', 'most_frequent'
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
        data.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
        date = None
        if drop_columns:
            date = data.index.values
            data = data.drop(columns=['date', 'begins_at', 'session', 'interpolated', 'symbol'], errors='ignore')

        if load_scaler:
            input_scaler_list = joblib.load(os.path.join(load_scaler, 'input_scaler_list.pkl'))
            input_scaler = joblib.load(os.path.join(load_scaler, 'input_scaler.pkl'))
            
            cls.input_scaler_list = input_scaler_list
            cls.input_scaler = input_scaler
            data = np.array(data)
            # for i in range(data.shape[1]):
            #     scaled_column = cls.input_scaler_list[i].transform(data[..., i].reshape(-1, 1))
            #     cls.input_scaler_list.append(scaler)  # Add the fitted sc

            scaled_data = cls.input_scaler.transform(data)
            return scaled_data, cls.input_scaler_list, date 
              
        cls.input_scaler_list = []  # Initialize an empty list to hold the scalers
        # scaled_data = np.empty_like(data)  # Initialize an empty array to hold the scaled data
        data = np.array(data)
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            scaled_column = scaler.fit_transform(data[..., i].reshape(-1, 1))
            cls.input_scaler_list.append(scaler)  # Add the fitted scaler to the list
            # scaled_data[..., i] = scaled_column.flatten()  # Add the scaled column to the scaled_data array

        scaler = MinMaxScaler()
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        cls.input_scaler = scaler

        # Save the scalers to a file
        if run_dir:
            joblib.dump(cls.input_scaler_list, os.path.join(run_dir, 'input_scaler_list.pkl'))
            joblib.dump(cls.input_scaler, os.path.join(run_dir, 'input_scaler.pkl'))

        return scaled_data, cls.input_scaler_list, date

    def inverse_transform_predictions(self, y_pred_scaled, column_idx, scaler_list=None):
        # Ensure the input_scaler_list has been initialized and the column_idx is valid
        if not hasattr(self, 'input_scaler_list') or not 0 <= column_idx < len(self.input_scaler_list):
            raise ValueError("Either the input_scaler_list has not been initialized or the column_idx is out of bounds.")
        # Reverse the scaling of the predicted prices for the specified column
        if scaler_list:
            return scaler_list[column_idx].inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_pred = self.input_scaler_list[column_idx].inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred.flatten()
    
    @staticmethod
    def fetch_and_preprocess_data(ticker="AAPL", period='1mo', start_date=None, end_date=None, time_interval=None, run_dir=None):
        data = seqTradeDataset.fetch_data(ticker=ticker, period=period, start_date=start_date, end_date=end_date, time_interval=time_interval)
        # data = data.drop(['Dividends', "Stock Splits"], axis=1) # dropping mostly empty columns
        # import pdb; pdb.set_trace()
        if not len(data):
            print(f"No data fetched for ticker: {ticker}")
            return None
        # save the data 
        joblib.dump(data, os.path.join(run_dir, 'raw_data.pkl')) 
        # Input preprocessing
        input_processor = InputPreprocessor()
        processed_features = input_processor(data)
        
        data_features = seqTradeDataset.feature_engineering(data)

        # join 
        total_features = np.concatenate([np.array(data_features), processed_features], axis=-1)
        data = pandas.DataFrame(data=total_features, index=data.index)
        scaled_data, _, _ = seqTradeDataset.preprocess_data(data, run_dir=run_dir)
        return scaled_data

class Solver:
    def gen_series(self, prices_lists, k):
        result_list = []
        for prices in prices_lists:
            result_tuple = self.gen_transactions(k, prices)
            result_list.append(result_tuple)
        return result_tuple

    def gen_transactions(self, prices, k) -> int:
        assert isinstance(prices, (list, np.ndarray)), f'Expected prices of type list or numpy.ndarray, got {type(prices)}'
        assert isinstance(k, int) and k >= 0, f'Expected k to be a non-negative integer, got {k}'

        n = len(prices)
        
        # solve special cases
        if not n or k == 0:
            return 0, []

        # find all consecutively increasing subsequence
        transactions = []
        start = 0
        end = 0
        for i in range(1, n):
            if prices[i] >= prices[i-1]:
                end = i
            else:
                if end > start:
                    transactions.append([start, end])
                start = i
        if end > start:
            transactions.append([start, end])

        while len(transactions) > k:
            # check delete loss
            delete_index = 0
            min_delete_loss = math.inf
            for i in range(len(transactions)):
                t = transactions[i]
                profit_loss = prices[t[1]] - prices[t[0]]
                if profit_loss < min_delete_loss:
                    min_delete_loss = profit_loss
                    delete_index = i

            # check merge loss
            merge_index = 0
            min_merge_loss = math.inf
            for i in range(1, len(transactions)):
                t1 = transactions[i-1]
                t2 = transactions[i]
                profit_loss = prices[t1[1]] - prices[t2[0]]
                if profit_loss < min_merge_loss:
                    min_merge_loss = profit_loss
                    merge_index = i

            # delete or merge
            if min_delete_loss <= min_merge_loss:
                transactions.pop(delete_index)
            else:
                transactions[merge_index - 1][1] = transactions[merge_index][1]
                transactions.pop(merge_index)

        return sum(prices[j]-prices[i] for i, j in transactions), transactions

def collate_fn(batch):
    
    # print(f'batch: {batch}')  # Debug: Print the batch to inspect its structure
    # print(f'type of first item in batch: {type(batch[0])}')  # Debug: Inspect the type of the first item
    # print(f'first item in batch: {batch[0]}')  # Debug: Inspect the first item
    data, segments, prices, histories = [], [], [], []
    for item in batch:
        data.append(item[0])
        segments.append(item[1][0])
        prices.append(item[1][1])
        histories.append(item[1][2])

    max_segments = MAX_TRANSACTIONS * 2
    max_len = max(len(seg[0]) for seg in segments)

    try:
        # Manually pad the segments
        padded_segments = []
        for seg in segments:
            # Create a new tensor of zeros with the desired dimensions
            padded_seg = torch.zeros( max_segments, max_len, dtype=torch.float32)
            # Copy the segments data into the new tensor
            
            padded_seg[:seg.shape[0], :len(seg[0])] = torch.tensor(seg, dtype=torch.float32)
            padded_segments.append(padded_seg)
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
    # Now stack each list into a tensor
    return (
        torch.stack(data),
        torch.stack(padded_segments),
        torch.stack(prices),
        torch.stack(histories)
    )

def collate_fn_angled(batch):
    data, segments, heights, angles, profits, prices = [], [], [], [], [], []
    for item in batch:
        data.append(item[0])
        segments.append(item[1][0])
        heights.append(item[1][1])
        angles.append(item[1][2])
        profits.append(item[1][3])
        prices.append(item[1][4])

    max_segments = MAX_TRANSACTIONS * 2  # Assuming each transaction consists of 2 segments
    max_len = max(len(seg[0]) for seg in segments)

    padded_segments, padded_heights, padded_angles = [], [], []

    for seg, h, ang in zip(segments, heights, angles):
        # Create new tensors of zeros with the desired dimensions
        padded_seg = torch.zeros(max_segments, max_len, dtype=torch.float32)
        padded_height = torch.zeros(max_segments, dtype=torch.float32)
        padded_angle = torch.zeros(max_segments, dtype=torch.float32)
        
        # Copy the segments, heights, and angles data into the new tensors
        seg_len = len(seg[0])
        padded_seg[:seg.shape[0], :seg_len] = torch.tensor(seg, dtype=torch.float32)
        padded_height[:len(h)] = torch.tensor(h, dtype=torch.float32)
        padded_angle[:len(ang)] = torch.tensor(ang, dtype=torch.float32)

        padded_segments.append(padded_seg)
        padded_heights.append(padded_height)
        padded_angles.append(padded_angle)

    # Stack each list into a tensor
    return (
        torch.stack(data),
        torch.stack(padded_segments), torch.stack(padded_heights), torch.stack(padded_angles),
        torch.stack(profits),
        torch.stack(prices)
    )

if __name__ == "__main__":
    # Fetching and processing data
    # Usage:
    ticker = "ETH-USD"
    period = "2mo"
    start_date = "2022-01-01"
    end_date = "2022-10-01"
    time_interval = "1D"  # Daily data
    raw_data_period= seqTradeDataset.fetch_data(
        YahooFinance, 
        ticker=ticker, 
        period=period)
    raw_data = seqTradeDataset.fetch_data(
        YahooFinance, 
        ticker=ticker, 
        start_date=start_date, 
        end_date=end_date, 
        time_interval=time_interval)
    engineered_data = feature_engineering(raw_data)
    processed_data, scaler = preprocess_data(engineered_data)   

    # Creating the Dataset and DataLoader
    eth_dataset = seqTradeDataset(processed_data, 
                            window_size=60, 
                            horizon=15)
    eth_dataloader = DataLoader(eth_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    

    # trail download
    processor = YahooFinance()
    ticker_list = ["ETH-USD", "BTC-USD"]  # Add more tickers as needed
    start_date = "2022-01-01"
    end_date = "2022-10-01"
    time_interval = "1D"  # Daily data
    crypto_data = processor.download_data(ticker_list, start_date, end_date, time_interval)