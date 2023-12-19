import robin_stocks.robinhood as rh
import pandas as pd
import plotly.graph_objs as go
from robinhood_manager import RobinhoodManager
import numpy as np
import os
# Assuming you have a RobinhoodManager class
username, password = os.environ['RH_USERNAME'], os.environ['RH_PASSWORD']
robinhood_manager = RobinhoodManager(username, password)

def piecewise_linear_interpolation(df):
    new_df = pd.DataFrame(df, index=df.index)

    for column in df.columns:
        column_series = df[column].copy()

        # Identify segments of non-NaN values
        segments = column_series.notna().cumsum()
        is_valid = column_series.notna()

        for segment in segments[is_valid].unique():
            if(len(column_series[segments == segment]) <= 1):
                print(f"Cannot interpolate for segment {segment} in column {column} due to insufficient data points.")
                continue
            # Find the start and end of the current segment
            start_index = column_series[segments == segment].first_valid_index()
            end_index = column_series[segments == segment].last_valid_index()

            # Find the start of the next segment
            next_segment_start = column_series[segments > segment].first_valid_index()
            
            # If there's a next segment, interpolate between the end of the current segment and the start of the next
            if next_segment_start is not None:
                end_value = column_series.loc[end_index]
                next_start_value = column_series.loc[next_segment_start]
                # Calculate the slope for linear interpolation
                slope = (float(next_start_value) - float(end_value)) / ((next_segment_start - end_index).total_seconds() / 3)  # 3 seconds interval
                # Apply interpolation
                for i, idx in enumerate(pd.date_range(start=end_index, end=next_segment_start, freq='3S')):
                    if idx not in column_series or pd.isna(column_series[idx]):
                        column_series[idx] = str(float(end_value) + slope * i)

            # Assign the reindexed series to the new_df
            new_df[column] = column_series
        import pdb; pdb.set_trace()
    return new_df

def piecewise_linear_interpolation(df):
    new_df = pd.DataFrame(df, index=df.index)

    for column in df.columns:
        column_series = df[column].copy()

        # Identify segments of non-NaN values
        segments = column_series.notna().cumsum()
        is_valid = column_series.notna()

        for segment in segments[is_valid].unique():
            if(len(column_series[segments == segment]) <= 1):
                return df  
            # Find the start and end of the current segment
            start_index = column_series[segments == segment].first_valid_index()
            end_index = column_series[segments == segment].last_valid_index()

            # Find the start of the next segment
            next_segment_start = column_series[segments > segment].first_valid_index()
            
            # If there's a next segment, interpolate between the end of the current segment and the start of the next
            if next_segment_start is not None:
                end_value = column_series.loc[end_index]
                next_start_value = column_series.loc[next_segment_start]
                # Calculate the slope for linear interpolation
                slope = (float(next_start_value) - float(end_value)) / ((next_segment_start - end_index).total_seconds() / 3)  # 3 seconds interval
                # Apply interpolation
                for i, idx in enumerate(pd.date_range(start=end_index, end=next_segment_start, freq='3S')):
                    if idx not in column_series or pd.isna(column_series[idx]):
                        column_series[idx] = str(float(end_value) + slope * i)

            # Assign the reindexed series to the new_df
            new_df[column] = column_series

    return new_df


def fetch_and_resample_crypto_data( symbol, desired_interval, points_required):
    '''
    If the interval is 15sec, then there is 15sec for the hour so 4 points every minute for 60 minutes, 240 points we have
    But if its 3 sec interval then 15sec is interpolated into 3sec so 5*4*60 points for the hour, 1200 points so points are more 
    Now to compare, we have to get more points in the 3sec one to satisfy the time range of 1 hour
    
    '''
    interval, span = calculate_interval_span(desired_interval, points_required)
    historical_data = robinhood_manager.get_crypto_historicals(symbol, interval=interval, span=span)

    # Convert to DataFrame and resample
    df = pd.DataFrame(historical_data)
    df['begins_at'] = pd.to_datetime(df['begins_at'])
    df.set_index('begins_at', inplace=True)

    resampling_rule = interval_to_pandas_resampling_rule(desired_interval)
    resampled_df = df.resample(resampling_rule).agg({'open_price': 'first','close_price': 'last','high_price': 'max','low_price': 'min','volume': 'sum'})

    # Convert None to NaN
    resampled_df = resampled_df.applymap(lambda x: np.nan if x is None else x)

    # Apply linear interpolation
    resampled_df = piecewise_linear_interpolation(resampled_df)
    return resampled_df

def interval_to_pandas_resampling_rule(desired_interval):
    # Mapping of intervals to pandas resampling rule format
    rule_mapping = {
        '1sec': '1S',
        '3sec': '3S',
        '5sec': '5S',
        '15sec': '15S',
        '30sec': '30S',
        '1min': '1T',
        '5min': '5T',
        '10min': '10T',
        '15min': '15T',
        '30min': '30T',
        '1hr': '1H', 
        '1d': '1D',
        '1w': '1W',
        '1mo': '1M'  # Note: '1M' stands for calendar month end frequency
    }
    return rule_mapping.get(desired_interval, '1T')  # Default to 1 minute if interval not found


def calculate_interval_span(desired_interval, points_required):
    # Map the desired interval to robin_stocks interval and calculate the span
    interval_mapping = {
        '1sec': ('15second', 'hour'),
        '3sec': ('15second', 'hour'),
        '5sec': ('15second', 'hour'),
        '15sec': ('15second', 'hour'),
        '30sec': ('15second', 'hour'),
        '1min': ('15second', 'hour'),
        '5min': ('5minute', 'day'),
        '10min': ('5minute', 'day'),
        '15min': ('5minute', 'day'),
        '30min': ('10minute', 'week'),
        '1hr': ('hour', 'week'), 
        '1d': ('day', 'year'),
        '1w': ('day', '5year'),
        '1mo': ('week', '5year')
    }

    robin_stocks_interval, robin_stocks_span = interval_mapping.get(desired_interval, ('5minute', 'day'))

    return robin_stocks_interval, robin_stocks_span

# Fetching data
symbol = 'BTC'
data_15s = fetch_and_resample_crypto_data(symbol, '15sec', 240) # Assuming points_required is the number of data points
data_3s = fetch_and_resample_crypto_data(symbol, '3sec', 1200)

# Plotting
fig = go.Figure()
# fig.add_trace(go.Scatter(x=data_15s.index, y=data_15s['close_price'], mode='lines', name='15s Data Resampled to 1min'))


# fig.update_layout(title='Comparison of 15s Resampled Data and 1min Data', xaxis_title='Time', yaxis_title='Price')
# fig.write_image("compare_fig1.jpg")


# Calculate tick values

# Assuming 'data_3s' is your DataFrame and 'close_price' is the column to plot
y_min = data_3s['close_price'].astype(float).min()
y_max = data_3s['close_price'].astype(float).max()
y_range = y_max - y_min
num_ticks = 100  # Let's say we want 5 ticks on the y-axis
tick_step = y_range / (num_ticks - 1)  # Calculate appropriate step size

# Generate tick values and corresponding labels
y_ticks = np.arange(start=y_min, stop=y_max + tick_step, step=tick_step)  # +tick_step to include y_max
tick_labels = [f"{y:.2f}" for y in y_ticks]

# Update the layout with the new y-axis configuration
fig.update_layout(yaxis = dict(tickmode = 'array',tickvals = y_ticks))

fig.add_trace(go.Scatter(x=data_3s.index, y=data_3s['close_price'], mode='lines', name='3sec Data'))

fig.write_image("compare_fig2.jpg")
import pdb; pdb.set_trace()