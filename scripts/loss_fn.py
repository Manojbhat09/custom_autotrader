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
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
import logging
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

class LossWithExponentialPreference:

    def __init__(self, alpha=0.8):
        self.alpha = alpha

    def __call__(self, l1, l2, l3, l4, l5):
        
        w1 = 1.0
        w2 = self.alpha
        w3 = self.alpha**2 
        w4 = self.alpha**3
        w5 = self.alpha**4

        weighted_loss = w1*l1 + w2*l2 + w3*l3 + w4*l4 + w5*l5  
        weighted_loss = weighted_loss / (w1 + w2 + w3 + w4 + w5)

        # print(f"Losses - Price: {l1:.4f}, Geo: {l2:.4f}, Segment: {l3:.4f}, Angle: {l4:.4f}, Bayesian: {l5:.4f}")
        
        return weighted_loss
    
class ReshapeSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, probs, expected_shape):
        
        flat_probs = probs.reshape(probs.shape[0] * probs.shape[1], probs.shape[2])  
        
        samples = torch.multinomial(flat_probs, num_samples=1)

        samples = samples.reshape(expected_shape)
        
        ctx.save_for_backward(probs, samples)

        return samples

    @staticmethod
    def backward(ctx, grad_samples):
        
        # Inside ReshapeSampler.backward

        probs, samples = ctx.saved_tensors

        batch_size, num_segments = samples.shape
        device = probs.device

        # Initialize gradient
        grad_probs = torch.zeros_like(probs)

        # Get probability of sampled indices
        sample_probs = probs.reshape(-1).index_select(0, samples.view(-1)) 

        # Assign gradient based on sample probability
        grad_probs.reshape(-1).index_add_(0, samples.view(-1), 
                                            sample_probs.new_ones(batch_size * num_segments)) 

        # Average across batch
        grad_probs = grad_probs / batch_size  
        grad_probs = grad_samples + grad_probs
        return grad_probs, None
    
class OptimalTransportLoss:

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, pred_segments, target_segments):
        
        # Cost matrix
        cost_matrix = torch.cdist(pred_segments, target_segments)
        
        # Sinkhorn smoothing 
        sinkhorn = ot.sinkhorn(pred_segments, target_segments, cost_matrix, self.epsilon)

        # Optimal transport plan
        transport_plan = sinkhorn.detach()

        # Ot loss 
        ot_loss = ot.sinkhorn_loss(
            pred_segments, 
            target_segments, 
            cost_matrix, 
            sinkhorn_iterations=100,
            epsilon=self.epsilon
        )

        return ot_loss, transport_plan

class BipartiteLoss:

    def __init__(self, tau_init=1.0, tau_anneal=0.95):
        self.tau_init = tau_init 
        self.tau_anneal = tau_anneal
    
    def bipartite_matching_loss(self, pred_segments, target_segments, epoch=None, dist_fn='euclidean'):
        """Compute bipartite matching loss between predicted and target segments.
        
        Args:
        pred_segments: Tensor of shape (N, M, 2) with predicted segments
        target_segments: Tensor of shape (N, K, 2) with target segments
        dist_fn: Distance function to use for cost matrix (default 'euclidean')
        
        Returns:
        Scalar loss value
        """

        N = pred_segments.shape[0] # Batch size
        
        # Initialize cost matrix
        cost_matrix = torch.full_like(pred_segments, 1e6) 

        # Compute actual costs
        if dist_fn == 'euclidean':
            cost_matrix = torch.cdist(pred_segments, target_segments, p=2)
        elif dist_fn == 'dtw':
            for i, pred in enumerate(pred_segments):
                for j, target in enumerate(target_segments):
                    cost_matrix[i,j] = dtw_distance(pred, target)

        if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
            logging.INFO("Warning: cost matrix contains NaN or Inf values!")
            print("Cost matrix min:", torch.min(cost_matrix))
            print("Cost matrix max:", torch.max(cost_matrix))
            import pdb; pdb.set_trace()

        tau = max(self.tau_init * (self.tau_anneal**epoch), 0.5) 

        # Flatten and sample indices
        sampled_rows, sampled_cols, prob_matrix = self.flatten_and_sample(cost_matrix, tau)

        # Gather segments
        matched_pred_segments = torch.gather(pred_segments, 1, sampled_rows[...,None])
        matched_target_segments = torch.gather(target_segments, 1, sampled_cols[...,None])

        # Losses
        reinforce_loss = nn.NLLLoss()(torch.log(prob_matrix), sampled_rows.squeeze(-1)) 
        match_loss = torch.mean((matched_pred_segments - matched_target_segments)**2)
        
        # Total loss 
        loss = reinforce_loss + match_loss
        
        return loss
    
    def flatten_and_sample(self, cost_matrix, tau):

        batch_size, num_segments, _ = cost_matrix.shape
        
        # Make contiguous
        cost_matrix = cost_matrix.contiguous()
        
        # Softmax probabilities
        prob_matrix = F.softmax(-cost_matrix / tau, dim=2)

        # Sample indices
        sampled_rows = ReshapeSampler.apply(prob_matrix, (batch_size, num_segments)) 
        sampled_cols = ReshapeSampler.apply(prob_matrix.transpose(1,2), (batch_size, num_segments))

        return sampled_rows, sampled_cols, prob_matrix
    



'''


    def flatten_and_sample(self, cost_matrix, tau):
        
        batch_size, num_segments, _ = cost_matrix.shape
        
        # Make contiguous
        cost_matrix = cost_matrix.contiguous()
        
        # Softmax probabilities
        prob_matrix = F.softmax(-cost_matrix / tau, dim=2)

        # Flatten 
        flat_probs = prob_matrix.view(batch_size * num_segments, cost_matrix.shape[2])
        
        # Sample indices
        sampled_rows = torch.multinomial(flat_probs, num_samples=1)
        sampled_cols = torch.multinomial(flat_probs.t(), num_samples=1)
        
        # Reshape indices
        sampled_rows = sampled_rows.view(batch_size, num_segments)
        sampled_cols = sampled_cols.view(batch_size, num_segments)
        
        return sampled_rows, sampled_cols, prob_matrix


        # import pdb; pdb.set_trace()

        # # Apply softmax to get assignment probabilities
        # prob_matrix = F.softmax(-cost_matrix / tau, dim=1)

        # # Flatten batch and segment dims
        # flat_probs = prob_matrix.view(prob_matrix.shape[0] * prob_matrix.shape[1], prob_matrix.shape[2])

        # # Sample indices
        # assigned_rows = torch.multinomial(flat_probs, num_samples=1) 

        # # Reshape back to add batch dim
        # assigned_rows = assigned_rows.view(prob_matrix.shape[0], -1)

        # col_prob_matrix = prob_matrix.transpose(1,2)

        # # Flatten batch and segment dims
        # flat_probs = col_prob_matrix.view(prob_matrix.shape[0] * prob_matrix.shape[1], prob_matrix.shape[2])

        # # Sample indices
        # assigned_cols = torch.multinomial(flat_probs, num_samples=1) 

        # # Reshape back to add batch dim
        # assigned_cols = assigned_rows.view(col_prob_matrix.shape[0], -1)
        
        # # # Sample indices from distribution
        # # assigned_rows = torch.multinomial(prob_matrix, num_samples=1)    
        # # assigned_cols = torch.multinomial(prob_matrix.transpose(1,0), num_samples=1)
        
        # # Gather segments using sampled indices
        # matched_pred_segments = torch.gather(pred_segments, 1, assigned_rows[:, :, None])
        # matched_target_segments = torch.gather(target_segments, 1, assigned_cols[:, :, None]) 

'''

def chamfer_distance(pred_segments, target_segments):
    
    # Flatten batches into single sets
    pred_segments = pred_segments.view(-1, 2)  
    target_segments = target_segments.view(-1, 2)
    
    # Compute nearest neighbor distances    
    pred_dists, _ = torch.cdist(pred_segments, target_segments).min(dim=1)  
    target_dists, _ = torch.cdist(target_segments, pred_segments).min(dim=1)

    # Mean losses
    pred_loss = pred_dists.mean() 
    target_loss = target_dists.mean()
     
    # Chamfer distance
    loss = pred_loss + target_loss

    return loss

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
            'geometric_bayesian_weighted': self.custom_loss_geom_bayesian_weighted,
            'geometric_bayesian_profit_penalty': self.custom_loss_geom_bayesian_profit, 
            'geo_bayesian_profit_constrained': self.custom_loss_geom_bayesian_profit_constrained
        }
        self.tb_logger = tb_logger
        self.writer = tb_logger.writer
        self.losses = {}  # to store loss values
        self.horizon = horizon
        self.geom_lossfn = GeometricConsistencyLoss(1.0, horizon)
        self.bipartite = BipartiteLoss()

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


    # def bipartite_matching_loss(self, pred_segments, target_segments, dist_fn='euclidean'):
    #     """Compute bipartite matching loss between predicted and target segments.
        
    #     Args:
    #     pred_segments: Tensor of shape (N, M, 2) with predicted segments
    #     target_segments: Tensor of shape (N, K, 2) with target segments
    #     dist_fn: Distance function to use for cost matrix (default 'euclidean')
        
    #     Returns:
    #     Scalar loss value
    #     """

    #     N = pred_segments.shape[0] # Batch size
        
    #     # Initialize cost matrix
    #     cost_matrix = torch.full_like(pred_segments, 1e6) 

    #     # Compute actual costs
    #     if dist_fn == 'euclidean':
    #         cost_matrix = torch.cdist(pred_segments, target_segments, p=2)
    #     elif dist_fn == 'dtw':
    #         for i, pred in enumerate(pred_segments):
    #             for j, target in enumerate(target_segments):
    #                 cost_matrix[i,j] = dtw_distance(pred, target)

    #     if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
    #         logging.INFO("Warning: cost matrix contains NaN or Inf values!")
    #         print("Cost matrix min:", torch.min(cost_matrix))
    #         print("Cost matrix max:", torch.max(cost_matrix))
    #         import pdb; pdb.set_trace()

    #     # Find optimal assignment 
    #     batch_size, n, m = cost_matrix.shape

    #     # Reshape to 2D matrix (BATCH_SIZE * N, M) 
    #     cost_matrix = cost_matrix.reshape(batch_size * n, m) 

    #     # Compute assignment
    #     assignment = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

    #     # Reshape assignment indices to add batch dim back
    #     assignment = np.array(assignment).reshape(batch_size, 2)

    #         # Reshape assigned row/col indices to add batch dim back
    #     assigned_rows = torch.LongTensor(assignment[0]).view(batch_size, n)
    #     assigned_cols = torch.LongTensor(assignment[1]).view(batch_size, n)
        
    #     # Extract assigned segments using torch ops
    #     matched_pred_segments = torch.gather(pred_segments, 1, assigned_rows.unsqueeze(-1).expand(-1, -1, 2))
    #     matched_target_segments = torch.gather(target_segments, 1, assigned_cols.unsqueeze(-1).expand(-1, -1, 2))

    #     # Compute loss between assigned segments
    #     loss = torch.mean(torch.sum((matched_pred_segments - matched_target_segments)**2, dim=2))
            
    #     # # Gather matched segments using assignment
    #     # matched_pred_segments = pred_segments[range(N), assignment[0]]
    #     # matched_target_segments = target_segments[range(N), assignment[1]]

    #     # # Compute average L2 distance between matched segments
    #     # loss = torch.mean(torch.sum((matched_pred_segments - matched_target_segments) ** 2, dim=1))
        
    #     return loss
    


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
    
    def custom_loss_geom_bayesian_weighted(self,  y_true_segments, y_true_heights, y_true_angles, y_true_profit, y_true_prices, 
                                           y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  
                                           epoch=None, sigma=1.0):
        geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        # pred_tan_angles = sin_angles/(cos_angles + 1e-08)
        pred_angles = torch.arctan2(sin_angles, cos_angles)
        angle_loss = nn.MSELoss()(y_true_angles, pred_angles)
        # segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, dist_fn='cdist')
        segment_loss = chamfer_distance(y_pred_segments, y_true_segments)
        # segment_loss = OptimalTransportLoss()(y_true_segments, y_pred_segments)
        # segment_loss = self.bipartite.bipartite_matching_loss(y_true_segments, y_pred_segments, epoch=epoch)
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit)
        # price_loss = nn.MSELoss()(y_true_prices, y_pred_prices[1])

        time_decay = torch.exp(-torch.arange(y_pred_prices[0].shape[-1]).float() / sigma).cuda()  # Exponential decay
        weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices[0], y_true_prices).mean(dim=0)).sum()

        bayesian_price_loss = bayesian_loss_function(y_true_prices, y_pred_prices[0], y_pred_prices[1])
        total_loss = (7/8)*(segment_loss + geom_loss + weighted_price_loss + angle_loss + profit_loss) + (1/8)*bayesian_price_loss
        # total_loss = LossWithExponentialPreference(0.8)(weighted_price_loss, geom_loss, segment_loss, angle_loss, bayesian_price_loss)
        # total_loss = weighted_price_loss + geom_loss + segment_loss +angle_loss + bayesian_price_loss + profit_loss
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss': profit_loss, 
                       'price_loss': weighted_price_loss,
                       'bayesian_price_loss': bayesian_price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'total_loss': total_loss}
        return total_loss
    
    def custom_loss_geom_bayesian_profit_weighted(self,  y_true_segments, y_true_heights, y_true_angles, y_true_profit, y_true_prices, 
                                           y_pred_segments, y_pred_angles,  y_pred_profit,  y_pred_prices,  
                                           epoch=None, sigma=1.0):
        import pdb; pdb.set_trace()
        # geom_loss = self.geom_lossfn(y_true_profit, y_pred_profit, y_pred_angles, y_pred_segments)
        sin_angles, cos_angles = y_pred_angles[..., 0], y_pred_angles[..., 1]
        # pred_tan_angles = sin_angles/(cos_angles + 1e-08)
        pred_angles = torch.arctan2(sin_angles, cos_angles)
        angle_loss = nn.MSELoss()(y_true_angles, pred_angles)
        # segment_loss = self.bipartite_matching_loss(y_true_segments, y_pred_segments, dist_fn='cdist')
        segment_loss = chamfer_distance(y_pred_segments, y_true_segments)
        # segment_loss = OptimalTransportLoss()(y_true_segments, y_pred_segments)
        # segment_loss = self.bipartite.bipartite_matching_loss(y_true_segments, y_pred_segments, epoch=epoch)
        profit_loss = nn.MSELoss()(y_true_profit, y_pred_profit.sum(axis=-1))
        # price_loss = nn.MSELoss()(y_true_prices, y_pred_prices[1])

        time_decay = torch.exp(-torch.arange(y_pred_prices[0].shape[-1]).float() / sigma).cuda()  # Exponential decay
        weighted_price_loss = (time_decay * nn.MSELoss(reduction='none')(y_pred_prices[0], y_true_prices).mean(dim=0)).sum()

        bayesian_price_loss = bayesian_loss_function(y_true_prices, y_pred_prices[0], y_pred_prices[1])
        total_loss = (7/8)*(segment_loss + geom_loss + weighted_price_loss + angle_loss + profit_loss) + (1/8)*bayesian_price_loss
        # total_loss = LossWithExponentialPreference(0.8)(weighted_price_loss, geom_loss, segment_loss, angle_loss, bayesian_price_loss)
        # total_loss = weighted_price_loss + geom_loss + segment_loss +angle_loss + bayesian_price_loss + profit_loss
        self.losses = {'segment_loss': segment_loss, 
                       'profit_loss': profit_loss, 
                       'price_loss': weighted_price_loss,
                       'bayesian_price_loss': bayesian_price_loss,
                       'angle_loss': angle_loss, 
                       'geom_loss': geom_loss, 
                       'total_loss': total_loss}
        return total_loss

def bayesian_loss_function(y_true, mean, std):
    normal_distribution = Normal(mean, std)
    log_likelihood = normal_distribution.log_prob(y_true) 
    return -log_likelihood.mean()
    
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