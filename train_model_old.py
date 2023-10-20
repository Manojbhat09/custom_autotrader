'''
Why Input and Output Dimension Might Be 1:

In many stock price forecasting tasks, the goal is to predict the next day's closing price based on past closing prices. In such cases, you might only use the closing price as a feature, leading to an input dimension of 1. Similarly, since you're predicting a single value (next day's closing price), the output dimension would also be 1.
However, if you're using multiple technical indicators (like Moving Averages, RSI, MACD, etc.) as features, then the input dimension should match the number of features (e.g., 18).

'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

# Define a function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if not os.path.exists('plots'):
    os.makedirs('plots')

if not os.path.exists('plots/training'):
    os.makedirs('plots/training')

# Assuming df is your dataframe with the features
os.makedirs('data/processed', exist_ok=True)
file_path = os.path.join("data", "processed", "TSLA.csv")
df = pd.read_csv(file_path)
df = pd.DataFrame(df)

# Drop non-numeric columns for simplicity
df = df.drop(columns=['date', 'begins_at', 'session', 'interpolated', 'symbol'])

# Interpolate the missing values
df.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Splitting data into train and test sets using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(scaled_data):
    train, test = scaled_data[train_index], scaled_data[test_index]

# Close price as the target variable
X_train, y_train = np.delete(train, 1, axis=1), train[:, 1]
X_test, y_test = np.delete(test, 1, axis=1), test[:, 1]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape x to include a batch dimension which is 1
        x = torch.unsqueeze(x, dim=1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # import pdb; pdb.set_trace()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # out = self.linear(out[:, -1, :])
        out = self.linear(out[:, -1, :])
        out = out.squeeze(-1)
        return out
    
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
input_dim = 18  # Adjust this value according to your data

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = []
test_losses = []
num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train)
    optimizer.zero_grad()
    import pdb; pdb.set_trace()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Calculate and store the training loss
    train_loss = loss.item()
    train_losses.append(train_loss)
    
    # Calculate the test loss
    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        test_losses.append(test_loss)
        model.train()
    
    # Print and save the losses plot every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Plot the training and test losses
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()
        
        # Save the plot in the specified folder
        plot_path = 'plots/training/loss_plot.png'  # Change this path as needed
        plt.savefig(plot_path)
        plt.close()

# Ensure directories exist
os.makedirs('models/checkpoints', exist_ok=True)

# Assuming 'model' is your trained PyTorch model
torch.save(model.state_dict(), 'models/checkpoints/TSLA_checkpoint.pth')
print("Model checkpoint saved to models/checkpoints/TSLA_checkpoint.pth")

# Load the model for evaluation
model.load_state_dict(torch.load('models/checkpoints/TSLA_checkpoint.pth'))
model.eval()

# Test the model
with torch.no_grad():
    test_outputs = model(X_test)
    mse_loss = criterion(test_outputs, y_test)
    print(f"Test MSE Loss: {mse_loss.item()}")

# Convert predictions back to original scale
test_outputs = scaler.inverse_transform(test_outputs.numpy())
    



# # Train the model
# num_epochs = 100
# for epoch in range(num_epochs):
#     outputs = model(X_train)
#     optimizer.zero_grad()
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()