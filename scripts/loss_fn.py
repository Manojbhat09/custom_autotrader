

'''

In this function:

unprofitable_loss calculates the total loss from unprofitable segments.
budget_constraint_penalty and transaction_constraint_penalty apply heavy penalties if the budget or transaction constraints are violated.
udp sums these penalties to form the Unprofitable Decision Penalty.
The final loss function now includes udp, which is added to the other loss components.



'''
def custom_loss(segment_predictions, profit_prediction, price_prediction, target_segments, target_profit, target_prices, budget, transactions_allowed, sigma=1.0):
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
    
    return segment_loss + profit_loss + weighted_price_loss.sum() - gaussian_penalty + udp


'''

Defining Unprofitable Transactions:

Identify segments where selling price is lower than buying price: 
�
�
�
�
�
<
�
�
�
�
P 
sell
​
 <P 
buy
​
 
Calculate the loss for each unprofitable segment: 
�
�
�
�
�
=
�
�
�
�
,
�
−
�
�
�
�
�
,
�
Loss 
i
​
 =P 
buy,i
​
 −P 
sell,i
​
 
Budget Constraint:

Ensure the total cost of buying in all segments does not exceed the budget: 
∑
�
�
�
�
,
�
≤
�
�
�
�
�
�
∑P 
buy,i
​
 ≤Budget
Transaction Constraint:

Ensure the total number of transactions does not exceed the allowed transactions: 
�
≤
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
N≤Transactions 
allowed
​
 
Formulating the Unprofitable Decision Penalty:

Sum the losses from all unprofitable segments: 
�
�
�
�
�
�
�
=
∑
�
�
�
�
�
UDP 
loss
​
 =∑Loss 
i
​
 
If the budget or transaction constraints are violated, apply an additional heavy penalty: 
�
�
�
�
�
�
�
�
�
�
�
�
�
UDP 
constraint
​
 
Final UDP:

�
�
�
=
�
�
�
�
�
�
�
+
�
�
�
�
�
�
�
�
�
�
�
�
�
UDP=UDP 
loss
​
 +UDP 
constraint
​
 
Now, integrating this UDP into the custom loss function:

'''