import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from utils import PredictionCache, DecisionManager, IndexingManager, PredictionObject, SegmentObject
from data_manager import ModelInference
import random
# def update_objects(model_inference):
    
#     # Instantiate the IndexingManager
#     indexing_manager = IndexingManager()

#     # Iterate over your predictions to create and add SegmentObjects and PredictionObjects
#     for i in range(len(price_data) - window_size):
#         start_idx = i
#         # Generate predictions for the given window of price data
#         segments, max_profit, predicted_prices = model_inference.predict(price_data[i:i+window_size])
        
#         # Create a PredictionObject with the generated data
#         prediction_object = PredictionObject(starting_index=start_idx, segments=segments, 
#                                             max_profit=max_profit, predicted_price=predicted_prices[-1])

#         # Add the PredictionObject to the IndexingManager
#         indexing_manager.add_prediction_object(start_idx, prediction_object)
        
#         # Create SegmentObjects for each segment and add them to the IndexingManager
#         for segment_info in segments:
#             segment_object = SegmentObject(prediction_horizon_starting_index=start_idx,
#                                         segment_number=segment_info['segment_number'],
#                                         buy=segment_info['buy'],
#                                         sell=segment_info['sell'],
#                                         theta=segment_info['theta'])
#             indexing_manager.add_segment(start_idx, segment_object)
#     return indexing_manager


def init():
    ax.set_xlim([df.index.min(), df.index.max()])
    ax.set_ylim([df['Close'].min() * 0.9, df['Close'].max() * 1.1])

# Assume data is in a pandas DataFrame called df with a DateTimeIndex and a column 'Close' for closing prices
# np.random.randn().cumsum()
df = pd.DataFrame( np.repeat(np.array([random.randrange(30200, 34600, 1) for i in range(100) ])[:, None], 7, axis=-1), columns=['open', 'low', 'high', 'Close', 'dividents', 'stock_split', 'latter'])  # Replace with your data

window_size = 30
checkpoint_path = '/home/mbhat/tradebot/custom_autotrader/run/experiment_20231103_162231/models/BTC-USD_checkpoint_experiment_20231103_162231.pth'
model_inference = ModelInference(model_name='SegmentBayesianHeadingModel', 
                                    checkpoint_path=checkpoint_path)
cache = PredictionCache()
price_data =df
def animate(i):
    global df, cache
    window_size = 30
    horizon = 15  # Prediction horizon

    # Ensure we have enough data to update the predictions.
    if i >= window_size:
        # import pdb; pdb.set_trace()
        # Get the latest window of data to pass to the model.
        latest_data = df.iloc[i-window_size:i]
        print("input data")
        print(latest_data)
        # Run inference to get predictions.
        # import pdb; pdb.set_trace()
        outputs, start_idx = model_inference.run_inference_bayesian_heading(latest_data, ticker='DummyTicker')
        y_pred_segments, y_pred_angles, y_pred_profit, y_pred_mean, y_pred_low, y_pred_high = outputs
        print(y_pred_mean.reshape(-1))
        # Clear the current axes.
        ax.clear()

        # Plot the actual prices up to the current index.
        ax.plot(df.index[:i], df['Close'][:i], label='Actual Prices')

        # Plot the predicted prices.
        # Assuming the predictions align with the data indices, adjust if necessary.
        prediction_indices = df.index[start_idx:start_idx+horizon]
        ax.plot(prediction_indices, y_pred_mean[:horizon], label='Predicted Prices Mean')
        ax.fill_between(prediction_indices, y_pred_low[:horizon].reshape(-1), y_pred_high[:horizon].reshape(-1), color='gray', alpha=0.5, label='Prediction Interval')

        # Add legend and title to the plot.
        ax.legend()
        ax.set_title(f"Time: {df.index[i-1]}")


    # # price_data = df['Close'][:i+window_size]
    # cache.update_predictions(price_data, window_size, model_inference)

    # indexing_manager = update_objects()

    # decision_manager = DecisionManager(indexing_manager=indexing_manager)
    
    # ax.clear()
    # ax.plot(df.index[:i+window_size], df['Close'][:i+window_size], label='Actual Prices')
    
    # predicted_data = cache.predicted_data[-window_size:]
    # if predicted_data:
    #     ax.plot(df.index[i:i+window_size], predicted_data, label='Predicted Prices')
    
    # ax.legend()
    # ax.set_title(f"Time: {df.index[i+window_size-1]}")
    # decision = decision_manager.get_decision(i+window_size-1)
    # ax.annotate(f'Decision: {decision}', xy=(0.05, 0.95), xycoords='axes fraction')

fig, ax = plt.subplots()
ax.set_xlim([df.index.min(), df.index.max()])
ax.set_ylim([df['Close'].min() * 0.9, df['Close'].max() * 1.1])

ani = animation.FuncAnimation(fig, animate, frames=len(df) - window_size, repeat=False, init_func=init)
plt.show()

# To save the animation as a .gif or .mp4 file:
# ani.save('rolling_predictions.mp4', writer='ffmpeg', fps=1)
# or
ani.save('rolling_predictions.gif', writer='imagemagick', fps=1)

# # To display the animation in a Jupyter Notebook:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# # To display the animation in Streamlit:
# import streamlit as st
# st.write(HTML(ani.to_jshtml()))

# # To run the animation in a blocking manner, which is more suitable for script-based execution:
# plt.show()