import torch
import torch.nn as nn
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
'''

In this function:

unprofitable_loss calculates the total loss from unprofitable segments.
budget_constraint_penalty and transaction_constraint_penalty apply heavy penalties if the budget or transaction constraints are violated.
udp sums these penalties to form the Unprofitable Decision Penalty.
The final loss function now includes udp, which is added to the other loss components.

'''

class GeometricConsistencyLoss(torch.nn.Module):
    def __init__(self, horizon=15, sigma=1.0):
        super(GeometricConsistencyLoss, self).__init__()
        self.sigma = sigma
        self.horizon = horizon

    def forward(self, y_true_max_price, y_pred_max_price, y_pred_angles, y_pred_indices):

        y_pred_indices_ints = 15 * torch.sigmoid(y_pred_indices)  # Sigmoid scales values to [0, 1], then scale to [0, 15]

        # Extract predicted buy and sell indices
        y_pred_buy_idx, y_pred_sell_idx = y_pred_indices_ints[..., 0], y_pred_indices_ints[..., 1]

        # Compute predicted price differences using geometric relationships

        # Compute predicted price differences using geometric relationships
        # Assuming y_pred_angles is structured as: [..., 2] with sine and cosine values
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]

        # Use sine and cosine to compute the price difference
        # This formula assumes that sin(theta) and cos(theta) represent the slope of the line connecting the buy and sell points
        price_difference_predicted = (sin_angles / (cos_angles + 1e-8)) * (y_pred_sell_idx - y_pred_buy_idx)  # Adding a small value to prevent division by zero


        # Sum all predicted segment price differences to get predicted max price
        sum_price_difference_predicted = price_difference_predicted.sum(dim=-1)

        # Compute Gaussian error between predicted max price and true max price
        # gaussian_error_1 = torch.exp(-((y_pred_max_price - y_true_max_price)**2) / (2 * self.sigma**2))
        
        # # Compute Gaussian error between sum of all predicted segment price differences and true max price
        # gaussian_error_2 = torch.exp(-((sum_price_difference_predicted - y_true_max_price)**2) / (2 * self.sigma**2))
        
        # # The geometric consistency loss is the absolute difference between gaussian_error_1 and gaussian_error_2
        # geometric_consistency_loss = torch.abs(gaussian_error_1 - gaussian_error_2).mean()

        error1 = nn.MSELoss()(y_pred_max_price, y_true_max_price)
        error2 = nn.MSELoss()(sum_price_difference_predicted, y_true_max_price)
        geometric_consistency_loss = torch.abs(error1 - error2).mean()
        return geometric_consistency_loss


class LossFunctions:
    def __init__(self, loss_type, horizon, tb_logger):
        self.loss_type = loss_type
        self.loss_functions = {
            'vanila': self.custom_loss_vanilla,
            'weighted_price': self.custom_loss_weighted_price,
            'gaussian_penalty': self.custom_loss_gaussian_penalty,
            'gaussian_penalty_decay': self.custom_loss_gaussian_penalty_decay,
            'vanila_bipartite': self.custom_loss_vanila_bipartite,
            'bipartite': self.custom_loss_gauss_bipartite,
            'dtw_bipartite': self.custom_loss_dtw_bipartite, 
            'upd': self.custom_loss_upd,
            'geometric_consistency': self.custom_loss_geom_consistency, 
            'geometric_bayesian': self.custom_loss_geom_bayesian, 
            'geometric_bayesian_profit_penalty': self.custom_loss_geom_bayesian_profit, 
            'geo_bayesian_profit_constrained': self.custom_loss_geom_bayesian_profit_constrained
        }
        self.tb_logger = tb_logger
        self.writer = tb_logger.writer
        self.losses = {}  # to store loss values
        self.horizon = horizon
        self.geom_lossfn = GeometricConsistencyLoss(1.0, horizon)

    def get_loss_function(self):
        return self.loss_functions.get(self.loss_type, self.custom_loss_vanilla)  # Default to vanila if not found

    def log_losses(self, step, stage='train'):
        for loss_name, loss_value in self.losses.items():
            self.writer.add_scalar(f'Loss/{loss_name}_{stage}', loss_value.item(), step)

    def bipartite_matching_loss(self, pred_segments, target_segments, dist_fn=torch.cdist):
        batch_size, num_segments, _ = pred_segments.size()
        losses = torch.zeros(batch_size, device=pred_segments.device)
        for i in range(batch_size):
            batch_pred_segment = pred_segments[i]
            batch_tgt_segment = target_segments[i]
            if dist_fn == "dtw":
                cost_matrix = torch.tensor([[dtw_distance(pred, target) for target in batch_tgt_segment] for pred in batch_pred_segment])
            elif dist_fn == "cdist":
                cost_matrix = torch.cdist(batch_pred_segment, batch_tgt_segment, p=2)  # Assuming Euclidean distance as the cost
            else:
                cost_matrix = torch.tensor([[dist_fn(pred, target) for target in batch_tgt_segment] for pred in batch_pred_segment])
            temp = cost_matrix.clone().cpu().detach().numpy()
            pred_indices, target_indices = linear_sum_assignment(temp)
            matched_preds = pred_segments[i][pred_indices]
            matched_targets = target_segments[i][target_indices]
            losses[i] = torch.sum((matched_preds - matched_targets) ** 2)  # Assuming L2 loss
        return losses.mean()  


    def custom_loss_vanilla(self, y_segments, y_profit, y_prices, pred_segments, pred_profit, pred_prices, **kwargs):
        segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_segments, pred_segments))
        profit_loss = nn.MSELoss()(y_profit, pred_profit)
        price_loss = nn.MSELoss()(y_prices, pred_prices)
        
        total_loss = segment_loss + profit_loss + price_loss 
        self.losses = {
            'segment_loss': segment_loss, 
            'profit_loss': profit_loss, 
            'price_loss': price_loss, 
            'total_loss': total_loss
        }
        
        return total_loss
    

    def custom_loss_weighted_price(self, y_segments, y_profit, y_prices, pred_segments, pred_profit, pred_price, **kwargs):
        segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_segments, pred_segments))
        profit_loss = nn.MSELoss()(y_profit, pred_profit)
        price_loss = nn.MSELoss()(y_prices, pred_price)
        sigma = kwargs.get('sigma', 1.0)
        time_decay = torch.exp(-torch.arange(15).float() / sigma).cuda()
        weighted_price_loss = time_decay * nn.MSELoss(reduction='none')(y_prices, pred_price).mean(dim=0)
        total_loss = segment_loss + profit_loss + price_loss + weighted_price_loss.sum()
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss': profit_loss, 
                       'price_loss': price_loss, 
                       'weighted_price_loss': weighted_price_loss.sum(), 
                       'total_loss': total_loss}
        return total_loss
    
    def custom_loss_gaussian_penalty(self, y_true_segments, y_true_profit, y_true_prices, y_pred_segments, y_pred_profit, y_pred_prices, sigma=1.0):
        segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
        gaussian_penalty_seg = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_pred_segments, y_true_segments)).sum()
        gaussian_penalty_price = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_true_prices, y_true_prices)).sum()
        gauss_penalty = gaussian_penalty_price + gaussian_penalty_seg
        max_fiter_gauss = max(gaussian_penalty_seg, gaussian_penalty_price)
        min_filter_gauss = min(gaussian_penalty_seg, gaussian_penalty_price)

        total_loss = segment_loss + profit_loss + price_loss
        max_emperical = max(segment_loss, profit_loss, price_loss)
        min_emperical = min(segment_loss, profit_loss, price_loss)

        scaled_gauss = ( (gauss_penalty - min_filter_gauss) * ( max_emperical - min_emperical ) / (max_fiter_gauss - min_filter_gauss ) ) + min_emperical
        total_loss -= (1/3) * scaled_gauss
        
        self.losses = {
            'segment_loss': segment_loss,
            'profit_loss': profit_loss,
            'price_loss': price_loss,
            'gaussian_penalty': -(gaussian_penalty_seg + gaussian_penalty_price),
            'total_loss': total_loss
        }
        return total_loss

    def custom_loss_gaussian_penalty_decay(self, y_true_segments,  y_true_profit, y_true_prices, y_pred_segments,y_pred_profit, y_pred_prices, sigma=1.0):
        segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
        gaussian_penalty_seg = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_pred_segments, y_true_segments)).sum()
        gaussian_penalty_price = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_true_prices, y_true_prices)).sum()
        gauss_penalty = gaussian_penalty_price + gaussian_penalty_seg
        max_fiter_gauss = max(gaussian_penalty_seg, gaussian_penalty_price)
        min_filter_gauss = min(gaussian_penalty_seg, gaussian_penalty_price)

        time_decay = torch.exp(-torch.arange(15).float() / sigma).cuda()  # Exponential decay
        weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices, y_true_prices).mean(dim=0)).sum()

        total_loss = segment_loss + profit_loss + price_loss + weighted_price_loss
        max_emperical = max(segment_loss, profit_loss, price_loss, weighted_price_loss)
        min_emperical = min(segment_loss, profit_loss, price_loss, weighted_price_loss)

        scaled_gauss = ( (gauss_penalty - min_filter_gauss) * ( max_emperical - min_emperical ) / (max_fiter_gauss - min_filter_gauss ) ) + min_emperical
        total_loss -= (1/3) * scaled_gauss
        self.losses = {
            'segment_loss': segment_loss,
            'profit_loss': profit_loss,
            'price_loss': price_loss,
            'weighted_price_loss': weighted_price_loss.sum(),
            'gaussian_penalty': -(gaussian_penalty_seg + gaussian_penalty_price),
            'total_loss': total_loss
        }

        return total_loss

    def custom_loss_vanila_bipartite(self, y_true_segments, y_true_profit, y_true_prices, y_pred_segments, y_pred_profit, y_pred_prices, sigma=1.0):
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments)
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
        total_loss = segment_loss + profit_loss + price_loss
        self.losses = {'segment_loss': segment_loss, 'profit_loss': profit_loss, 'price_loss': price_loss, 'total_loss': total_loss}
        return total_loss
    
    def custom_loss_dtw_bipartite(self, y_true_segments, y_true_profit, y_true_prices, y_pred_segments, y_pred_profit, y_pred_prices, sigma=1.0):
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, "dtw")
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
        total_loss = segment_loss + profit_loss + price_loss
        self.losses = {'segment_loss': segment_loss, 'profit_loss': profit_loss, 'price_loss': price_loss, 'total_loss': total_loss}
        return total_loss
    
    def custom_loss_gauss_bipartite(self, y_true_segments, y_true_profit, y_true_prices, y_pred_segments, y_pred_profit, y_pred_prices, sigma=1.0): #
    
        segment_loss = sum(nn.MSELoss()(y_true, y_pred) for y_true, y_pred in zip(y_true_segments, y_pred_segments))
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
        gaussian_penalty_seg = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_pred_segments, y_true_segments)).sum()
        gaussian_penalty_price = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(y_true_prices, y_true_prices)).sum()
        gauss_penalty = gaussian_penalty_price + gaussian_penalty_seg
        max_fiter_gauss = max(gaussian_penalty_seg, gaussian_penalty_price)
        min_filter_gauss = min(gaussian_penalty_seg, gaussian_penalty_price)

        total_loss = segment_loss + profit_loss + price_loss
        max_emperical = max(segment_loss, profit_loss, price_loss)
        min_emperical = min(segment_loss, profit_loss, price_loss)

        scaled_gauss = ( (gauss_penalty - min_filter_gauss) * ( max_emperical - min_emperical ) / (max_fiter_gauss - min_filter_gauss ) ) + min_emperical
        total_loss -= (1/3) * scaled_gauss
        
        return total_loss
    
    
    def custom_loss_upd(self, segment_predictions, target_segments, profit_prediction, target_profit, price_prediction, target_prices, sigma=1.0):
        segment_loss = sum(nn.MSELoss()(pred, target) for pred, target in zip(segment_predictions, target_segments))
        profit_loss = nn.MSELoss()(profit_prediction, target_profit)
        price_loss = nn.MSELoss()(price_prediction, target_prices)
        time_decay = torch.exp(-torch.arange(15).float() / sigma)  # Exponential decay
        weighted_price_loss = time_decay * nn.MSELoss(reduction='none')(price_prediction, target_prices).mean(dim=1)
        gaussian_penalty = sum(torch.exp(-((pred - target)**2) / (2 * sigma**2)) for pred, target in zip(segment_predictions, target_segments))
        
        # Unprofitable Decision Penalty
        unprofitable_loss = sum((buy - sell).clamp(min=0) for buy, sell in segment_predictions)  # sum of losses from unprofitable segments
        total_buy_cost = sum(buy for buy, sell in segment_predictions)  # total cost of buying
        total_transactions = len(segment_predictions)  # total number of transactions
        
        budget_constraint_penalty = (total_buy_cost - budget).clamp(min=0) * 1e2  # heavy penalty if budget is exceeded
        transaction_constraint_penalty = (total_transactions - transactions_allowed).clamp(min=0) * 1e2  # heavy penalty if transaction limit is exceeded
        
        udp = unprofitable_loss + budget_constraint_penalty + transaction_constraint_penalty
        total_loss = segment_loss + profit_loss + weighted_price_loss.sum() - gaussian_penalty + udp
        losses = {
            'segment_loss': segment_loss,
            'profit_loss': profit_loss,
            'weighted_price_loss': weighted_price_loss.sum(),
            'gaussian_penalty': -gaussian_penalty,
            'udp': udp,
            'total_loss': segment_loss + profit_loss + weighted_price_loss.sum() - gaussian_penalty + udp
        }
        self.losses = losses  # store loss values
        return losses['total_loss']  # return total loss for backpropagation
    
    
    def custom_loss_height(self,  y_true_segments, y_true_heights, y_true_profit, y_true_prices, y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  sigma=1.0): # TODO
        geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        angle_loss = nn.MSELoss()(y_true_angles, y_pred_angles)
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments)
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
        total_loss = segment_loss + profit_loss + price_loss + geom_loss
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss': profit_loss, 
                       'price_loss': price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'total_loss': total_loss}
        return total_loss
    
    def custom_loss_geom_consistency(self,  y_true_segments, y_true_heights, y_true_angles, y_true_profit, y_true_prices, y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  sigma=1.0):
        geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        # pred_tan_angles = sin_angles/(cos_angles + 1e-08)
        pred_angles = torch.arctan2(sin_angles, cos_angles)
        angle_loss = nn.MSELoss()(y_true_angles, pred_angles)
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, dist_fn='cdist')
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)
        total_loss = segment_loss + profit_loss + price_loss + geom_loss
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss': profit_loss, 
                       'price_loss': price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'total_loss': total_loss}
        return total_loss
    
    def custom_loss_geom_consistency(self,  y_true_segments, y_true_heights, y_true_angles, y_true_profit, y_true_prices, y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  sigma=1.0):
        geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        # pred_tan_angles = sin_angles/(cos_angles + 1e-08)
        pred_angles = torch.arctan2(sin_angles, cos_angles)
        angle_loss = nn.MSELoss()(y_true_angles, pred_angles)
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, dist_fn='cdist')
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        # price_loss = nn.MSELoss()(y_true_prices, y_pred_prices)

        time_decay = torch.exp(-torch.arange(15).float() / sigma).cuda()  # Exponential decay
        weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices, y_true_prices).mean(dim=0)).sum()

        total_loss = segment_loss + profit_loss + weighted_price_loss + geom_loss
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss': profit_loss, 
                    #    'price_loss': price_loss,
                       'weighted_price_loss': weighted_price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'total_loss': total_loss}
        return total_loss
    
    def custom_loss_geom_bayesian(self,  y_true_segments, y_true_heights, y_true_angles, y_true_profit, y_true_prices, y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  sigma=1.0):
        geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        # pred_tan_angles = sin_angles/(cos_angles + 1e-08)
        pred_angles = torch.arctan2(sin_angles, cos_angles)
        angle_loss = nn.MSELoss()(y_true_angles, pred_angles)
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, dist_fn='cdist')
        # profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        # price_loss = nn.MSELoss()(y_true_prices, y_pred_prices[1])

        time_decay = torch.exp(-torch.arange(y_pred_prices[0].shape[-1]).float() / sigma).cuda()  # Exponential decay
        weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices[0], y_true_prices).mean(dim=0)).sum()

        bayesian_price_loss = bayesian_loss_function(y_true_prices, y_pred_prices[0], y_pred_prices[1])

        total_loss = (7/8)*(segment_loss + geom_loss + weighted_price_loss + angle_loss) + (1/8)*bayesian_price_loss
        self.losses = {'segment_loss': segment_loss, 
                    #    'profit_loss': profit_loss, 
                       'price_loss': weighted_price_loss,
                       'bayesian_price_loss': bayesian_price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'total_loss': total_loss}
        return total_loss
    
    def custom_loss_geom_bayesian_profit(self,  y_true_segments, y_true_heights, y_true_angles, y_true_profit, y_true_prices, 
                                         y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  sigma=1.0):
        geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        # pred_tan_angles = sin_angles/(cos_angles + 1e-08)
        pred_angles = torch.arctan2(sin_angles, cos_angles)
        angle_loss = nn.MSELoss()(y_true_angles, pred_angles)
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, dist_fn='cdist')
        # profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        # price_loss = nn.MSELoss()(y_true_prices, y_pred_prices[1])

        time_decay = torch.exp(-torch.arange(y_pred_prices[0].shape[-1]).float() / sigma).cuda()  # Exponential decay
        weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices[0], y_true_prices).mean(dim=0)).sum()

        bayesian_price_loss = bayesian_loss_function(y_true_prices, y_pred_prices[0], y_pred_prices[1])

        penalty_strength = 10.0  # or some other value\
        negative_profit_penalty = penalty_strength * (F.relu(-y_pred_profit)**2).mean()
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)

        total_loss = (7/8)*(segment_loss + geom_loss + weighted_price_loss + angle_loss + negative_profit_penalty + profit_loss) + (1/8)*bayesian_price_loss
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss':profit_loss, 
                       'negative_profit_penalty': negative_profit_penalty, 
                       'price_loss': weighted_price_loss,
                       'bayesian_price_loss': bayesian_price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'total_loss': total_loss}
        return total_loss

    def custom_loss_geom_bayesian_profit_constrained(self,  y_true_segments, y_true_heights, y_true_angles, y_true_profit, y_true_prices, 
                                         y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  sigma=1.0):
        geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        # pred_tan_angles = sin_angles/(cos_angles + 1e-08)
        pred_angles = torch.arctan2(sin_angles, cos_angles)
        angle_loss = nn.MSELoss()(y_true_angles, pred_angles)
        segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, dist_fn='cdist')
        # profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        # price_loss = nn.MSELoss()(y_true_prices, y_pred_prices[1])

        time_decay = 10*torch.exp(-torch.arange(y_pred_prices[0].shape[-1]).float() / sigma).cuda()  # Exponential decay
        weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices[0], y_true_prices).mean(dim=0)).sum()

        bayesian_price_loss = bayesian_loss_function(y_true_prices, y_pred_prices[0], y_pred_prices[1])

        penalty_strength = 10.0  # or some other value\
        negative_profit_penalty = penalty_strength * (F.relu(-y_pred_profit)**2).mean()
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)

        # Unprofitable Decision Penalty
        buy_values = y_pred_segments[..., 0]  # Shape: (batch_size, num_segments)
        sell_values = y_pred_segments[..., 1]  # Shape: (batch_size, num_segments)
        true_buy_values =y_true_segments[..., 0]

        # Calculating unprofitable_loss and total_buy_cost
        unprofitable_loss = torch.sum((buy_values - sell_values).clamp(min=0), dim=1).mean()  # Shape: (batch_size,)
        # total_buy_cost = torch.sum(buy_values, dim=1) 
        # true_buy_cost = torch.sum(true_buy_values, dim=1)
        buy_loss = torch.abs(buy_values-true_buy_values).sum()

        udp = unprofitable_loss + buy_loss

        total_loss = (7/8)*(segment_loss + geom_loss + weighted_price_loss + angle_loss + negative_profit_penalty + profit_loss + udp) + (1/8)*bayesian_price_loss
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss':profit_loss, 
                       'negative_profit_penalty': negative_profit_penalty, 
                       'price_loss': weighted_price_loss,
                       'bayesian_price_loss': bayesian_price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'unprofitable_loss': unprofitable_loss, 
                       'buy_loss': buy_loss,
                       'total_loss': total_loss}
        return total_loss

def bayesian_loss_function(y_true, mean, std):
    normal_distribution = Normal(mean, std)
    negative_log_likelihood = normal_distribution.log_prob(y_true)
    return -negative_log_likelihood.mean()
    
def dtw_distance(seqA, seqB):
    distance, path = fastdtw(seqA, seqB, dist=euclidean)
    return torch.tensor(distance)



'''
def dtw_distance(seqA, seqB):
    distance, path = fastdtw(seqA, seqB, dist=euclidean)
    return torch.tensor(distance)

def highorder(pred_segments, target_segments):
    # Using Minkowski distance with p=3
    cost_matrix = torch.cdist(pred_segments, target_segments, p=3)
    return cost_matrix

def bipartite_matching_loss_dtw(pred_segments, target_segments):
    cost_matrix = torch.tensor([[dtw_distance(pred, target) for target in target_segments] for pred in pred_segments])
    pred_indices, target_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
    # ... rest of the code

    
segment_lengths = torch.norm(target_segments - pred_segments, dim=1, keepdim=True)
normalized_cost_matrix = cost_matrix / segment_lengths

pred_slopes = (pred_segments[:, 1] - pred_segments[:, 0]) / (1e-8 + torch.norm(pred_segments[:, 1] - pred_segments[:, 0]))
target_slopes = (target_segments[:, 1] - target_segments[:, 0]) / (1e-8 + torch.norm(target_segments[:, 1] - target_segments[:, 0]))
slope_diff = torch.abs(pred_slopes.unsqueeze(1) - target_slopes.unsqueeze(0))
slope_cost_matrix = slope_diff.sum(dim=2)
final_cost_matrix = cost_matrix + alpha * slope_cost_matrix  # alpha is a weighting factor

segment_importance = # obtain the importance of each segment
weighted_cost_matrix = cost_matrix * segment_importance.unsqueeze(0)



if scaler:
            input_min, input_max = torch.tensor(scaler.data_min_).cuda(), torch.tensor(scaler.data_max_).cuda()
            y_pred_profit = (y_pred_profit - y_pred_profit.min())/(y_pred_profit.min() - y_pred_profit.max())*(input_min - input_max) + input_min

'''

if __name__ == "__main__":
    loss_fn_class = LossFunctions('vanila')  # replace 'vanila' with the desired loss type
    loss_fn = loss_fn_class.get_loss_function()

    # In your training loop:
    total_loss = loss_fn(...)  # pass arguments as required
    loss_fn_class.log_losses(1)