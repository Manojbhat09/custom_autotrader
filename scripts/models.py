import torch.nn as nn
import torch

class BayesianTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianTimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mean_layer = nn.Linear(hidden_size, output_size)
        self.std_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0, _ = self.lstm(x)
        h_last = h_0[:, -1, :]
        mean = self.mean_layer(h_last)
        std = torch.exp(self.std_layer(h_last))  # Ensure std is positive
        return mean, std

class MultiHeadModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_segments, future_timestamps=15):
        super(MultiHeadModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_segments)])  # Buy-Sell pairs
        self.profit_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, future_timestamps)  # Predict next 15 timestamps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :]
        action_segments = [head(out) for head in self.action_heads] # 20x2 segments each 32  
        
        profit_prediction = self.profit_head(out)  # 32 of them 32x1 
        price_prediction = self.price_head(out) # 15x1 closing price
        return action_segments, profit_prediction, price_prediction
    
class SegmentHeightModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_segments, future_timestamps=15):
        super(SegmentHeightModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(num_segments)])  # Buy-Sell-Height triples only changed
        self.profit_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, future_timestamps)  # Predict next 15 timestamps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :]
        action_segments = [head(out) for head in self.action_heads]  # num_segments x 3 vectors each 32
        
        profit_prediction = self.profit_head(out)  # 32 of them 32x1 
        price_prediction = self.price_head(out)  # 15x1 closing price
        return action_segments, profit_prediction, price_prediction


class SegmentHeadingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_segments, future_timestamps=15):
        super(SegmentHeadingModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(num_segments)])  # Adjusted to 4 for indxes, height, angle
        self.profit_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, future_timestamps)  # Predict next 15 timestamps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :]
        # Split into separate heads for indices and angles
        action_segments = torch.stack([head(out) for head in self.action_heads], dim=1)  # Stacking for better organization
        
        # Separate indices and angles
        indices_logits, angles = action_segments[:, :, :2], action_segments[:, :, 2:]
        
        # Apply appropriate activations
        # angles = torch.sigmoid(angles) * 180  # Scale sigmoid output to [0, 180] degrees
        angles = torch.sigmoid(angles) * (torch.pi / 2)  # Scale sigmoid output to [0, π/2] radians (0 to 90 degrees)

        profit_prediction = self.profit_head(out)  # 32 of them 32x1 
        price_prediction = self.price_head(out)  # 15x1 closing price
        
        return indices_logits, angles, profit_prediction, price_prediction  # Organized as a tuple for clarity
    
class SegmentBayesianModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_segments, future_timestamps=15):
        super(SegmentBayesianModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(num_segments)])  # Adjusted to 4 for indxes, height, angle
        self.profit_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, future_timestamps)  # Predict next 15 timestamps
        self.price_head = BayesianTimeSeriesModel(hidden_dim, hidden_dim, output_size=future_timestamps)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :]
        # Split into separate heads for indices and angles
        action_segments = torch.stack([head(out) for head in self.action_heads], dim=1)  # Stacking for better organization
        
        # Separate indices and angles
        indices_logits, angles = action_segments[:, :, :2], action_segments[:, :, 2:]
        
        # Apply appropriate activations
        # angles = torch.sigmoid(angles) * 180  # Scale sigmoid output to [0, 180] degrees
        angles = torch.sigmoid(angles) * (torch.pi / 2)  # Scale sigmoid output to [0, π/2] radians (0 to 90 degrees)

        profit_prediction = self.profit_head(out)  # 32 of them 32x1 
        price_prediction = self.price_head(out[:, None, :])  # 15x1 closing price adding the last sequence dimension
        
        return indices_logits, angles, profit_prediction, price_prediction  # Organized as a tuple for clarity

def make_model(name, input_dim, hidden_dim, num_layers, num_segments, device, future_timestamps=15):
    """Fetch the model by matching the name and creating the object and returning"""
    model_classes = {
        'MultiHeadModel': MultiHeadModel,
        'SegmentModel': SegmentHeightModel,
        'SegmentHeadingModel': SegmentHeadingModel, 
        'SegmentBayesianHeadingModel': SegmentBayesianModel
    }
    
    if name in model_classes:
        model_class = model_classes[name]
        model = model_class(input_dim, hidden_dim, num_layers, num_segments, future_timestamps).to(device)
        return model
    else:
        raise ValueError(f"Model name {name} not recognized!")