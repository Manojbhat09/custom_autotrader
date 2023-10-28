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



class seqTradeDataset(Dataset):
    def __init__(self, data, 
                 window_size=50, 
                 horizon=15, 
                 transform=None, 
                 ticker=None,
                 max_transactions=5):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.transform = transform
        self.processor = YahooFinance()
        self.preprocessor = Solver()
        self.max_transactions = max_transactions
        MAX_TRANSACTIONS = self.max_transactions
        self.segment_scaler = Scaling1d(min_data=0, max_data=self.horizon, min_range=0, max_range=1)
        if len(self.data) <=1:
            self.data = self.fetch_data()

        if len(self.data) < self.window_size + self.horizon:
            raise ValueError(f'Insufficient data: {len(self.data)} data points found, {self.window_size + self.horizon} required.')
        
        assert isinstance(self.data, np.ndarray), f'Expected data of type numpy.ndarray, got {type(self.data)}'

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

        # Prepare the future price prediction target
        future_prices = self.data[idx+self.window_size:idx+self.window_size+self.horizon, 3]
        y_price_prediction = torch.tensor(future_prices, dtype=torch.float32)
        
        # Conditionally apply a transform to x, if provided
        if self.transform:
            x = self.transform(x)

        
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_segments = self.segment_scaler.scale(segments)
        
        # Package segments, max profit, and price prediction target into a single tuple
        y = (torch.tensor(scaled_segments, dtype=torch.float32), torch.tensor(max_profit, dtype=torch.float32), y_price_prediction)
        
        return x, y
    
    # def __getitem__(self, idx):
    #     x = self.data[idx:idx+self.window_size]
    #     profit, y_segments = self.generate_segments(idx)
    #     if self.transform:
    #         x = self.transform(x)
    #     return x, y_segments

    # def generate_segments(self, idx):
    #     series = self.data[idx:idx+self.window_size+self.horizon, 3]  # Considering 'Close' price for segments
    #     profit, y_segments = self.generate_segments(idx)  # Use the generate_segments method here
    #     if self.transform:
    #         x = self.transform(x)
    #     return x, profit, torch.tensor(y_segments, dtype=torch.float32) 
    
    @classmethod
    def fetch_data(self,downloader=None, 
                   ticker="AAPL", 
                   ticker_list=None, 
                   period='1mo', 
                   start_date=None, 
                   end_date=None, 
                   extract_details=False,
                   time_interval=None):
        if downloader:
            processor = downloader()
        else:
            processor = YahooFinance()
        if ticker_list and type(ticker_list) == list:
            tickers = ticker_list
        else:
            ticker_list = [ticker]
        if period:
            if len(ticker_list) ==1:
                data = yf.Ticker(ticker_list[0]).history(period, interval=time_interval if time_interval else '1d')
            else:
                data = []
                interval=time_interval if time_interval else '1d'
                for ticker in ticker_list:
                    data.append(yf.Ticker(ticker).history(period, interval=interval))
        else:
            if extract_details:
                data=[]
                if len(ticker_list)==1:
                    data = yf.Ticker(ticker_list[0])
                else:
                    for ticker in ticker_list:
                        data.append(yf.Ticker(ticker))
                return data
            data = processor.download_data(ticker_list, start_date, end_date, time_interval)
        return data

    def generate_segments(self, idx):
        assert isinstance(idx, int) and idx >= 0, f'Expected idx to be a non-negative integer, got {idx}'
        assert idx + self.window_size + self.horizon <= len(self.data), f'Index {idx} out of bounds!'
        
        series = self.data[idx+self.window_size:idx+self.window_size+self.horizon, 3]  # Considering 'Close' price for segments
        
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
    
    # Feature Engineering
    @classmethod
    def feature_engineering(self, data):
        data['Daily Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily Return'].rolling(window=10).std()
        # ... other feature engineering steps
        return data

    # Preprocessing
    @classmethod
    def preprocess_data(self, data, drop_columns=False):
        data.dropna(inplace=True)
        data.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
        date=None
        if drop_columns:
            date = data.index.values
            data = data.drop(columns=['date', 'begins_at', 'session', 'interpolated', 'symbol'], errors='ignore')
        scaler = MinMaxScaler()
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler, date
    
    @staticmethod
    def fetch_and_preprocess_data(ticker="AAPL", period='1mo', start_date=None, end_date=None, time_interval=None):
        data = seqTradeDataset.fetch_data(ticker=ticker, period=period, start_date=start_date, end_date=end_date, time_interval=time_interval)
        data = seqTradeDataset.feature_engineering(data)
        scaled_data, _ = seqTradeDataset.preprocess_data(data)
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