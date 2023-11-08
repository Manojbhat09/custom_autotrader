import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


class SegmentBayesianProfitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_segments, future_timestamps=15):
        super(SegmentBayesianModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, 5) for _ in range(num_segments)])  # Adjusted to 4 for indxes, height, angle
        # self.profit_head = nn.Linear(hidden_dim, 1)
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
        indices_logits, angles, profit = action_segments[:, :, :2], action_segments[:, :, 2:4], action_segments[:, :, -1]
        
        # Apply appropriate activations
        # angles = torch.sigmoid(angles) * 180  # Scale sigmoid output to [0, 180] degrees
        angles = torch.sigmoid(angles) * (torch.pi / 2)  # Scale sigmoid output to [0, π/2] radians (0 to 90 degrees)
        profit_prediction = nn.ReLU(profit)
        # profit_prediction = self.profit_head(out)  # 32 of them 32x1 
        price_prediction = self.price_head(out[:, None, :])  # 15x1 closing price adding the last sequence dimension
        
        return indices_logits, angles, profit_prediction, price_prediction  # Organized as a tuple for clarity


class TransformerTimeSeriesModel(nn.Module):
    
    def __init__(self, input_size, num_heads, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(hidden_size, num_heads), 
            num_layers
        )
        self.mean_layer = nn.Linear(hidden_size, output_size)
        self.std_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h_0, _ = self.lstm(x)
        h_seq = self.transformer_encoder(h_0)
        h_last = h_seq[:, -1, :]
        mean = self.mean_layer(h_last)
        std = torch.exp(self.std_layer(h_last))
        return mean, std

class SegmentTransformerModel(nn.Module):
    
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, num_segments):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 4) for _ in range(num_segments)
        ])
        self.price_head = TransformerTimeSeriesModel(
            hidden_dim, num_heads, hidden_dim, num_layers, 15
        )
        self.profit_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h_0, _ = self.lstm(x)
        h_seq = self.transformer_encoder(h_0)
        h_last = h_seq[:, -1, :]
        action_segments = torch.stack([
            head(h_last) for head in self.action_heads
        ], dim=1)
        price_mean, price_std = self.price_head(h_last[:, None, :])
        profit_prediction = self.profit_head(h_last) 
        return action_segments, profit_prediction, price_mean, price_std

class QuantileRegressionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.quantiles = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        quantiles = self.quantiles(x)
        return quantiles

class VAEPriceModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.map_to_hidden = nn.Linear(input_dim, hidden_dim)  # Add this line
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        import pdb; pdb.set_trace()
        x = self.map_to_hidden(x)  # And this line
        z = self.encoder(x)
        price = self.decoder(z)
        return price

class StockPredictor(nn.Module):

    def __init__(self, input_dim, num_heads, hidden_dim, 
                 num_layers, num_segments, z_dim):
        super().__init__()
        self.seq_encoder = SegmentTransformerModel(
            input_dim, num_heads, hidden_dim, num_layers, num_segments
        )
        
        
    def forward(self, x):
        action_segments, max_profit, price_mean, price_std = self.seq_encoder(x)
        # Separate indices and angles
        indices_logits, angles = action_segments[:, :, :2], action_segments[:, :, 2:]
        
        # Apply appropriate activations
        # angles = torch.sigmoid(angles) * 180  # Scale sigmoid output to [0, 180] degrees
        angles = torch.sigmoid(angles) * (torch.pi / 2)  # Scale sigmoid output to [0, π/2] radians (0 to 90 degrees)
 
        
        return indices_logits, angles, max_profit, price_mean, price_std

class StockPredictorQuant(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, 
                 num_layers, num_segments, z_dim):
        super().__init__()
        self.seq_encoder = SegmentTransformerModel(
            input_dim, num_heads, hidden_dim, num_layers, num_segments
        )
        self.price_predictor = TransformerTimeSeriesModel(
            hidden_dim, num_heads, hidden_dim, num_layers, 15
        )
        self.quantile_predictor = QuantileRegressionModel(hidden_dim, hidden_dim)
        self.vae = VAEPriceModel(hidden_dim, hidden_dim, z_dim)
        
    def forward(self, x):
        segments, price_mean, price_std = self.seq_encoder(x)
        price_samples = self.vae(price_mean)
        quantiles = self.quantile_predictor(price_mean)
        return segments, price_mean, price_std, price_samples, quantiles



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
    

# input_dim = 9
# num_heads = 4
# hidden_dim = 128
# num_layers = 2
# num_segments = 5
# z_dim = 10
# batch_size = 16



# # Dataset
# class StockDataset(Dataset):

#     def __init__(self, data):
#         self.data = data 

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # Instantiate the model
# model = StockPredictor(input_dim, num_heads, hidden_dim, num_layers, num_segments, z_dim)

# # Generate some random input data with dimensions [batch_size, 2000, 9]
# input_data = torch.randn(batch_size, 2000, input_dim)


# # Dataset and DataLoader 
# dataset = StockDataset(input_data) 
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# # Optimizer
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# # VAE Loss
# def vae_loss(recon, mean, logvar):

#     recon_loss = F.mse_loss(recon, mean)  

#     # Log variance to std    
#     std = torch.exp(0.5 * logvar)  

#     # KL divergence
#     kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

#     return recon_loss + kl_div

# # Quantile loss 
# def quantile_loss(quantiles, labels):
#     quantiles = quantiles.transpose(1, 2)
#     errors = labels.unsqueeze(1) - quantiles
#     return torch.mean(
#         torch.max(errors * (labels < quantiles), errors * (labels >= quantiles))
#     )


# # Training loop
# for epoch in range(100):

#     for x_batch in dataloader:

#         # Forward
#         outputs = model(x_batch)
#         vae_out, quantile_out = outputs[1], outputs[4]
        
#         # Losses
#         vae_l = vae_loss(vae_out)
#         quantile_l = quantile_loss(quantile_out)  
#         loss = vae_l + 0.5 * quantile_l
        
#         # Backward
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#     # Logging
#     print(f"Epoch {epoch} - VAE Loss {vae_l:.4f}, Quantile Loss {quantile_l:.4f}")

# # # Forward pass
# # output = model(input_data)
# # import pdb; pdb.set_trace()
# # # Print the output dimensions to verify
# # for out in output:
# #     print(out.size())