
# If you're using pandas for data management:
import pandas as pd

# If you're handling dates and times:
from datetime import datetime

# If you're using numpy for numerical operations:
import numpy as np

# If your DataManager class or other parts of your code are using additional Plotly features:
import plotly.express as px

# If you are using any statistical or machine learning models:
from sklearn.metrics import mean_squared_error, r2_score

# If you're calculating execution time:
import time

class SegmentObject:
    def __init__(self, prediction_horizon_starting_index, segment_number, buy, sell, theta):
        self.prediction_horizon_starting_index = prediction_horizon_starting_index
        self.segment_number = segment_number
        self.buy = buy
        self.sell = sell
        self.theta = theta

    def get_prediction_horizon_starting_index(self):
        return self.prediction_horizon_starting_index

    def get_segment_number(self):
        return self.segment_number

    def get_buy(self):
        return self.buy

    def get_sell(self):
        return self.sell

    def get_theta(self):
        return self.theta

    def set_theta(self, theta):
        self.theta = theta

class PredictionObject:
    def __init__(self, starting_index, segments, max_profit, predicted_price):
        self.starting_index = starting_index
        self.segments = [SegmentObject(starting_index, i, *segment) for i, segment in enumerate(segments)]
        self.max_profit = max_profit
        self.predicted_price = predicted_price

    def get_prediction_horizon_starting_index(self):
        return self.prediction_horizon_starting_index

    def get_max_profit(self):
        return self.max_profit

    def get_predicted_price(self):
        return self.predicted_price

    def set_max_profit(self, max_profit):
        self.max_profit = max_profit

    def set_predicted_price(self, predicted_price):
        self.predicted_price = predicted_price

    def get_segment(self, segment_number):
        return next((seg for seg in self.segments if seg.segment_number == segment_number), None)

class IndexingManager:
    def __init__(self):
        self.prediction_index = {}  # Maps starting index to PredictionObject
        self.segment_index = {}  # Maps index to list of SegmentObjects
        self.prediction_objects = {}

    def add_segment(self, timestamp, segment_object):
        if timestamp not in self.segment_index:
            self.segment_index[timestamp] = []
        self.segment_index[timestamp].append(segment_object)

    def get_segments(self, timestamp):
        return self.segment_index.get(timestamp, [])

    def add_prediction_object(self, prediction_horizon_starting_index, prediction_object):
        self.prediction_objects[prediction_horizon_starting_index] = prediction_object

    def get_prediction_object(self, prediction_horizon_starting_index):
        return self.prediction_objects.get(prediction_horizon_starting_index, None)

    def add_prediction(self, prediction_object):
        starting_index = prediction_object.starting_index
        self.prediction_index[starting_index] = prediction_object

        for segment in prediction_object.segments:
            sell_index = starting_index + segment.sell
            if sell_index not in self.segment_index:
                self.segment_index[sell_index] = []
            self.segment_index[sell_index].append(segment)

    def get_prediction(self, starting_index):
        return self.prediction_index.get(starting_index, None)
    
    def get_prediction_assets(self, prediction_horizon_starting_index):
        return self.prediction_index.get(prediction_horizon_starting_index, None)
    
    def get_segments_at_timestamp(self, timestamp):
        return self.segment_index.get(timestamp, [])

class DecisionManager:
    def __init__(self, cache=None, indexing_manager=None):
        self.indexing_manager = indexing_manager
        self.cache = cache

    def get_decision(self, current_index):
        # Get the segments for the current index
        current_segments = self.indexing_manager.segment_index.get(current_index, [])

        # Find the segment with the maximum profit
        max_profit = 0
        best_segment = None
        for segment in current_segments:
            prediction_start = segment.prediction_horizon_starting_index
            profit = self.calculate_profit(segment, prediction_start)
            if profit > max_profit:
                max_profit = profit
                best_segment = segment

        # Buy or sell based on the best segment found
        if best_segment is None:
            return "Hold"  # No profitable segment found
        elif best_segment.buy < 15:  # Buy signal within the prediction horizon
            return "Buy"
        else:  # Sell signal within the prediction horizon
            return "Sell"

    def calculate_profit(self, price_data, segment, prediction_start):
        # Get the price data for the segment
        buy_price = price_data[prediction_start + segment.buy]
        sell_price = price_data[prediction_start + segment.sell]
        return sell_price - buy_price
    
    
class PredictionCache:
    def __init__(self):
        self.last_index = None
        self.predicted_data = []
        self.segment_data = []
        self.price_data = []

    def update_price_data(self, new_price_data):
        self.price_data.extend(new_price_data)

    def update_segment_data(self, new_segment_data):
        self.segment_data.extend(new_segment_data)

    def cached_price_data(self):
        return self.price_data

    def cached_segment_data(self):
        return self.segment_data
    
    def update_predictions(self, price_data, window_size, model_inference):
        if self.last_index is None:  # Initial run
            self.last_index = len(price_data) - window_size  # Set the last index for the initial run
            # Run initial predictions
            for i in range(0, len(price_data) - window_size+1):
                window_data = price_data[i:i + window_size]
                outputs, start_idx = model_inference.run_inference_bayesian_heading(window_data, 'ticker')  # Assuming 'ticker' is available
                y_pred_segments, y_pred_angles, y_pred_profit, y_pred_prices_mean, y_pred_prices_low, y_pred_prices_high = outputs
                self.predicted_data.append(y_pred_prices_mean[-1])  # Append last prediction of window
                self.update_segment_data(y_pred_segments)
        else:  # On refresh
            new_data_points = len(price_data) - self.last_index  # Determine the number of new data points
            for i in range(new_data_points):
                start_idx = self.last_index + i - window_size + 1  # Adjust the start index for rolling window
                window_data = price_data[start_idx:start_idx + window_size]
                y_pred_segments, y_pred_angles, y_pred_profit, y_pred_prices_mean, y_pred_prices_low, y_pred_prices_high = model_inference.run_inference_bayesian_heading(window_data, 'ticker')  # Assuming 'ticker' is available
                self.predicted_data.append(y_pred_prices_mean[-1])  # Append last prediction of window
                self.update_segment_data(y_pred_segments)
                self.last_index += 1  # Update the last index
            self.update_price_data(price_data[self.last_index - new_data_points: self.last_index])  # Update the price data cache






'''
when the app is refreshed, it gets updated with new data points and the length of price array changes or possibly remains the same. How to save a cache and then match the with the original array so that we know what data points are we predicting fresh for. 
For example, we predicted a new trajecotry like this method at initiliazatiton, now when its refreshed there is new data coming in of say x_new number of points 
so the new price trajecotyr is x_old+x_new points. We have done prediction until all x_old points but of length x_old - window length such that the last index of x_old matches the pred_old predicted points. We do the last prediction using window size from the end for the index x_old-length window to give pred_old+horizon of trajecotyr where horizon is the new predicted points. Now horizon can be smaller or larger than x_new. suppose x_new is 5 points, then we predict for x_old + 1 -window to get the index pred_old + horizon +1  points. so we keep pred_old +1 = pred_new and again predict for x_old +2 - window for pred_old +2= pred_new points and so on for 5 new points and store it as cache. But we keep the two series pricie data and predicteed data different  and just keep storing predictions on predicted series using the price data series and .
price data series is the true series it is non-changing so can be stored as it is when there is an update on the series data . but to update the predicted series we have to predict and update

extend the current algorithm to handle all cases when prediction is not possible or data is not enough or prediction is too divergent 

as its highly likely that the new data points are in the order of 1 data point every 15 minutes. and prediction is fast, so the new plots are fairly slow. so its important to keep the first point of the prediction (for the next timestamp close to the last in the price data series ) so its good to store the  prediction output corresponding to the index as price_data keeps increading in index order. the main Idea is to compare the prediction and the new datapiont that will come in after 15mins to make a trading decision but its important to plot the futures for the next 15 datapoints to see the trajecotry. As the model outputs segments of buy and sell pairs, if its  buy decision and currently I am holding, then the mixed signal is to buy at low. and if the x'th index prediction had a sell datapoint for the current timesteamp for a certain amount of profit collection and if I have already bought then I would sell. So there has to be a clever way to store the segments as well and the max profit of that prediction window range 

the predicted data is 15 timestamps from the current last timesteamp of the price range and there will be just one point update on the price_data. But also the page can be refreshed any time so the static data can be udated with new datapoints >1  as well. But prediction happens after refresh only so ther is a timestemp for each prediction  when it was made. We can add a timesteamp index  to the cache and so the prediction data for that index can also be indexed so use appropriate datastructure. We make comparisions for every new datapoint with the precition so the decision maker will be called as such. But for a given current time all predictions in the past can be checked for the max profit found in selling at the current timestamp using the segments. Say current timestamp is i, then i- horizon_size is the last prediction cache that is needed and there might be predi ctions done at the index i - horizon_size +1, i - horizon_size + 2 ... until i-1 .  so all of these indices need to be checked for predicted segments, and the segment indice might be [x1, y1], [x2, y2 ] and its possible x2 is from the predicted horizon i-horizon_size +3 or any. So a same indice might have multiple segment points predicted. But we would be interested in the current index which is a segment point for the previous prediction horizons. So make an indexing mechanism for the segment point i : [{prediction_horizon_starting_index1: segment3: (x1, y1), prediction_horizon_starting_index2: segment5: (x2, y2)}] so each index which might have  a segment point 
Such that for that prediction horizon strating index, we can get the max profit made. and we can compute the parituclar segment profit with (price_data[prediction_horizon_starting_index1+ y1] - price_data[prediction_horizon_starting_index1 + x1] )

for this go through every segment in every prediction and store them in a logical way. Also store the indexed prediction objects and segment objects for every prediction horizon index 

we are doing buy or sell decision so write a decision manager that uses indexing manager. and each prediction object has a max_predicted_profit value and every segment has an angle value. The model also predicts the price value at the given timestamp for about a horizon of 15 timestamps given the past window price data of 30 timestamps as input. So find an efficient way to make objects for it as well. Write the complete scripts to handle all kinds of scenarios. start by writing down the steps to implement it 

Input to the model:
window of 30 points of price data 
which has starting_index = current_index- window 
horizon size of prediction= 15 points 

ouptut of the model on the current timestamp 
segmets = [(x, y, theta), (buy, sell, theta)] where buy sell are indices within the horizon i.e <15 
max_profit = value in price 
predicted_price = [x1, x2, x2, x4] --  15 points mean, mean-std1, mean+std2, all in price floating point values 

Now that it is clear wite it 

'''