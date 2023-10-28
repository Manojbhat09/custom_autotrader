
'''
given a time series of i stock tickers timeseires and K the maximum number of transactions, and a budget
return the
1. max profit that can be made 
2. max profit in k transactions 
3. min cost spent in transactions


'''

import numpy as np

class SolutionNpy:
    def maxProfit(self, timeseries, k, budget) -> int:
        timeseries = np.array(timeseries)
        num_stocks, num_timestamps = timeseries.shape
        present = timeseries[:, 0]
        future = timeseries[:, -1]

        if k == 0: return 0

        # Initialize dp to match the original code's structure
        dp = np.array([[[np.inf, 0] for _ in range(k + 1)] for _ in range(num_stocks)])
        
        for stock_idx in range(num_stocks):
            for timestamp_idx in range(num_timestamps):
                price = timeseries[stock_idx, timestamp_idx]
                for i in range(1, k + 1):
                    dp[stock_idx, i, 0] = min(dp[stock_idx, i, 0], price - dp[stock_idx, i - 1, 1])
                    dp[stock_idx, i, 1] = max(dp[stock_idx, i, 1], price - dp[stock_idx, i, 0])

        dp_max = np.zeros(budget + 1, dtype=int)
        for p, f in zip(present, future):
            for j in range(budget, p - 1, -1):
                dp_max[j] = max(dp_max[j], dp_max[j - p] + f - p)

        result_profit  = dp[:, k, 1]
        result_cost = dp[:, k, 0]

        return dp_max[-1], result_profit.tolist(), result_cost.tolist()



class SolutionNpy2:
    def maxProfit(self, timeseries, k, budget) -> int:
        timeseries = np.array(timeseries)
        num_stocks, num_timestamps = timeseries.shape
        present = timeseries[:, 0]
        future = timeseries[:, -1]

        if k == 0:
            return 0

        # Initialize dp to match the original code's structure
        dp = np.array([[[1000, 0] for _ in range(k + 1)] for _ in range(num_stocks)], dtype=int)
        
        # Initialize transaction history
        transaction_history = []

        for stock_idx in range(num_stocks):
            for timestamp_idx in range(num_timestamps):
                price = timeseries[stock_idx, timestamp_idx]
                for i in range(1, k + 1):
                    if price - dp[stock_idx, i - 1, 1] > dp[stock_idx, i, 0]:
                        dp[stock_idx, i, 0] = price - dp[stock_idx, i - 1, 1]
                        # Record the buy action
                        transaction_history.append(('buy', stock_idx, timestamp_idx, price))
                    if price - dp[stock_idx, i, 0] > dp[stock_idx, i, 1]:
                        dp[stock_idx, i, 1] = price - dp[stock_idx, i, 0]
                        # Record the sell action
                        transaction_history.append(('sell', stock_idx, timestamp_idx, price))

        dp_max = np.zeros(budget + 1, dtype=int)
        for p, f in zip(present, future):
            for j in range(budget, p - 1, -1):
                dp_max[j] = max(dp_max[j], dp_max[j - p] + f - p)

        result_profit  = dp[:, k, 1]
        result_cost = dp[:, k, 0]

        return dp_max[-1], result_profit.tolist(), result_cost.tolist(), transaction_history

class SolutionMax:
    def maxProfit(self, k: int, prices) -> int:
        # no transaction, no profit
        if k == 0: return 0
        # dp[k][0] = min cost you need to spend at most k transactions
        # dp[k][1] = max profit you can achieve at most k transactions
        dp = [[1000, 0] for _ in range(k + 1)]
        for price in prices:
            for i in range(1, k + 1):
                # price - dp[i - 1][1] is how much you need to spend
                # i.e use the profit you earned from previous transaction to buy the stock
                # we want to minimize it
                dp[i][0] = min(dp[i][0], price - dp[i - 1][1])
                # price - dp[i][0] is how much you can achieve from previous min cost
                # we want to maximize it
                dp[i][1] = max(dp[i][1], price - dp[i][0])
        # return max profit at most k transactions
		# or you can write `return dp[-1][1]`
        return dp[k][1]



class Solution:
    def maxProfit(self, timeseries, k, budget) -> int:
        num_stocks = len(timeseries)
        num_timestamps = len(timeseries[0])
        present = [stock[0] for stock in timeseries]
        future = [stock[-1] for stock in timeseries]

        if k == 0: return 0

        dp = [[[1000, 0] for _ in range(k + 1)] for _ in range(num_stocks)]
        
        for stock_idx, stock in enumerate(timeseries):
            for timestamp_idx, price in enumerate(stock):
                for i in range(1, k + 1):
                    dp[stock_idx][i][0] = min(dp[stock_idx][i][0], price - dp[stock_idx][i - 1][1])
                    dp[stock_idx][i][1] = max(dp[stock_idx][i][1], price - dp[stock_idx][i][0])

        dp_max = [0] * (budget + 1)
        for p, f in zip(present, future):
            for j in range(budget, p - 1, -1):
                dp_max[j] = max(dp_max[j], dp_max[j - p] + f - p)

        result_profit  = [dp[stock_idx][k][1] for stock_idx in range(num_stocks)]
        result_cost = [dp[stock_idx][k][0] for stock_idx in range(num_stocks)]

        return dp_max[-1], result_profit, result_cost


class Decisions:
    def maxProfit(self, timeseries, k, EOTDbudget) -> int:
        timeseries = np.array(timeseries)
        num_stocks, num_timestamps = timeseries.shape
        present = timeseries[:, 0]
        future = timeseries[:, -1]

        if k == 0: return 0

        dp = np.array([[[1000, 0] for _ in range(k + 1)] for _ in range(num_stocks)], dtype=int)
        decisions = np.empty((num_stocks, k + 1, 2), dtype=object)
        
        for stock_idx in range(num_stocks):
            for timestamp_idx in range(num_timestamps):
                price = timeseries[stock_idx, timestamp_idx]
                for i in range(1, k + 1):
                    new_cost = price - dp[stock_idx, i - 1, 1]
                    if new_cost < dp[stock_idx, i, 0]:
                        dp[stock_idx, i, 0] = new_cost
                        decisions[stock_idx, i, 0] = (i - 1, timestamp_idx)  # Buy decision
                    new_profit = price - dp[stock_idx, i, 0]
                    if new_profit > dp[stock_idx, i, 1]:
                        dp[stock_idx, i, 1] = new_profit
                        decisions[stock_idx, i, 1] = (i, timestamp_idx)  # Sell decision

        dp_max = np.zeros(EOTDbudget + 1, dtype=int)
        for p, f in zip(present, future):
            for j in range(EOTDbudget, p - 1, -1):
                dp_max[j] = max(dp_max[j], dp_max[j - p] + f - p)

        result_profit  = dp[:, k, 1]
        result_cost = dp[:, k, 0]

        # Extract buy and sell decisions
        buy_sell_decisions = []
        for stock_idx in range(num_stocks):
            buy_indices = []
            sell_indices = []
            for i in range(1, k + 1):
                buy_indices.append(decisions[stock_idx, i, 0][1])
                sell_indices.append(decisions[stock_idx, i, 1][1])
            buy_sell_decisions.append((buy_indices, sell_indices))
        

        actions = np.array([[ "hold" for i in range(num_timestamps)] for i in range(num_stocks)])
        for stock, tple in enumerate(buy_sell_decisions):
            for action in tple[0]:
                actions[stock][action] = "buy"
            for action in tple[1]:
                actions[stock][action] = "sell"

        import pdb; pdb.set_trace()
        
        # Extract buy and sell decisions
        buy_sell_costs = []

        for stock_idx in range(num_stocks):
            buy_costs = []
            sell_costs = []
            for i in range(1, k + 1):
                buy_idx = decisions[stock_idx, i, 0][1]
                sell_idx = decisions[stock_idx, i, 1][1]
                buy_costs.append(timeseries[stock_idx, buy_idx])
                sell_costs.append(timeseries[stock_idx, sell_idx])
            buy_sell_costs.append((buy_costs, sell_costs))

        return dp_max[-1], result_profit.tolist(), result_cost.tolist(), buy_sell_decisions, actions

    
class SolutionEffInit:
    def maxProfit(self, k: int, prices: list) -> int:
        if not prices:
            return 0

        n = len(prices)
        # if k >= n // 2:
        #     # If k is large enough, can make as many transactions as needed
        #     return sum(max(prices[i + 1] - prices[i], 0) for i in range(n - 1))

        # Initialize DP table
        dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n)]

        for i in range(n):
            for j in range(1, k + 1):
                # Base cases for day 0
                if i == 0:
                    dp[i][j][0] = 0
                    dp[i][j][1] = -prices[i]
                    continue

                # Transition equations
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])  # sell
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])  # buy

        # The maximum profit is in the state of not holding a stock at the last day, with k transactions
        return dp[n - 1][k][0]


class SolutionEff():
    def maxProfit(self, k: int, prices: list) -> tuple:
        if not prices:
            return 0, [], []

        n = len(prices)
        dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n)]
        actions = [[["hold", "hold"] for _ in range(k + 1)] for _ in range(n)]

        for i in range(n):
            for j in range(1, k + 1):
                if i == 0:
                    dp[i][j][0] = 0
                    dp[i][j][1] = -prices[i]
                    actions[i][j][1] = "buy"
                    continue

                sell_profit = dp[i - 1][j][1] + prices[i]
                if dp[i - 1][j][0] < sell_profit:
                    dp[i][j][0] = sell_profit
                    actions[i][j][0] = "sell"
                else:
                    dp[i][j][0] = dp[i - 1][j][0]

                buy_cost = dp[i - 1][j - 1][0] - prices[i]
                if dp[i - 1][j][1] < buy_cost:
                    dp[i][j][1] = buy_cost
                    actions[i][j][1] = "buy"
                else:
                    dp[i][j][1] = dp[i - 1][j][1]

        action_series = []
        transactions = []
        state = 0
        transactions_remaining = k
        buy_day, buy_price = None, None
        for i in range(n - 1, -1, -1):
            action = actions[i][transactions_remaining][state]
            action_series.append((i, action, prices[i]))
            if action == "buy":
                state = 1
                transactions_remaining -= 1
                buy_day, buy_price = i, prices[i]
            elif action == "sell" and buy_day is not None:
                state = 0
                transactions.append((buy_day, buy_price, i, prices[i]))
                buy_day, buy_price = None, None

        return dp[n - 1][k][0], list(reversed(action_series)), transactions

    
# class ConstrainedDecisions:
#     def maxProfit(self, timeseries, k, EOTDbudget) -> int:
#         timeseries = np.array(timeseries)
#         num_stocks, num_timestamps = timeseries.shape
#         present = timeseries[:, 0]
#         future = timeseries[:, -1]

#         if k == 0: return 0

#         # Initialize dp array
#         dp = np.full((num_stocks, num_timestamps, k + 1, EOTDbudget + 1, 2), float('-inf'), dtype=float)
#         decisions = np.empty((num_stocks, num_timestamps, k + 1, EOTDbudget + 1, 2), dtype=object)
        
#         # Set the initial state with zero transactions and full budget
#         dp[:, 0, 0, :, :] = 0
        
#         for stock_idx in range(num_stocks):
#             for timestamp_idx in range(1, num_timestamps):  # Starting from 1 as 0 is the initial state
#                 price = timeseries[stock_idx, timestamp_idx]
#                 for i in range(1, k + 1):
#                     for budget in range(price, EOTDbudget + 1):
#                         # Update for buying decision
#                         new_budget = budget - price
#                         new_cost = dp[stock_idx, timestamp_idx - 1, i - 1, new_budget, 1] - price
#                         if new_cost > dp[stock_idx, timestamp_idx, i, budget, 0]:
#                             dp[stock_idx, timestamp_idx, i, budget, 0] = new_cost
#                             decisions[stock_idx, timestamp_idx, i, budget, 0] = 'buy'
#                         # Update for selling decision
#                         new_profit = dp[stock_idx, timestamp_idx - 1, i, budget, 0] + price
#                         if new_profit > dp[stock_idx, timestamp_idx, i, budget, 1]:
#                             dp[stock_idx, timestamp_idx, i, budget, 1] = new_profit
#                             decisions[stock_idx, timestamp_idx, i, budget, 1] = 'sell'
#                         # Update for holding decision
#                         hold_profit = dp[stock_idx, timestamp_idx - 1, i, budget, 1]
#                         if hold_profit > dp[stock_idx, timestamp_idx, i, budget, 1]:
#                             dp[stock_idx, timestamp_idx, i, budget, 1] = hold_profit
#                             decisions[stock_idx, timestamp_idx, i, budget, 1] = 'hold'

#         dp_max = np.zeros(EOTDbudget + 1, dtype=int)
#         for p, f in zip(present, future):
#             for j in range(EOTDbudget, p - 1, -1):
#                 dp_max[j] = max(dp_max[j], dp_max[j - p] + f - p)

#         # Extracting the results and decisions for each stock
#         result_profit, result_cost = [], []
#         buy_sell_decisions, action_decisions = [], []
#         for stock_idx in range(num_stocks):
#             max_profit = float('-inf')
#             max_profit_timestamp, max_profit_budget = None, None
#             for timestamp_idx in range(num_timestamps):
#                 for budget in range(EOTDbudget + 1):
#                     if dp[stock_idx, timestamp_idx, k, budget, 1] > max_profit:
#                         max_profit = dp[stock_idx, timestamp_idx, k, budget, 1]
#                         max_profit_timestamp, max_profit_budget = timestamp_idx, budget
#             result_profit.append(max_profit)
#             result_cost.append(dp[stock_idx, max_profit_timestamp, k, max_profit_budget, 0])
#             actions = [decisions[stock_idx, timestamp_idx, k, max_profit_budget, 1] for timestamp_idx in range(num_timestamps)]
#             action_decisions.append(actions)

#         # Pad action_decisions with "hold" to match the length of timeseries
#         for stock_idx in range(num_stocks):
#             action_decisions[stock_idx] += ['hold'] * (num_timestamps - len(action_decisions[stock_idx]))


#         return dp_max[-1], result_profit, result_cost, buy_sell_decisions, action_decisions
    


class ListDecisions:
    def maxProfit(self, k: int, prices) -> int:
        n = len(prices)

        # solve special cases
        if not prices or k == 0:
            return 0

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

# list = [3,2,6,5,0,12]
# k  = 2
# ans = Solution().maxProfit(k, list)
# print(ans)

timeseries = [[] for i in range(2)]
timeseries[0] = [3,2,6,5,0,3]
timeseries[1] = [10,2,6,0,15,24]
budget = 20
transactions = 2

# ans = Solution()
# sol=ans.maxProfit(timeseries, transactions, budget)
# print("initial budget ", budget)
# print("final budget ", sol[0]+budget)
# print("profits ", sol[1])
# print("costs ", sol[2])

# solution = SolutionNpy()
# sol = solution.maxProfit(timeseries, transactions, budget)
# print("initial budget ", budget)
# print("final budget ", sol[0]+budget)
# print("profits ", sol[1])
# print("costs ", sol[2])

# solution = SolutionEff()
# print(solution.maxProfit(transactions, timeseries[1]))

# solution = Decisions()
# sol = solution.maxProfit(timeseries, transactions, budget)
# print(timeseries)
# print("initial budget ", budget)
# print("profits ", sol[1])
# print("profits with transactions ", sum(sol[1]))
# print("costs ", sol[2])
# print("costs in transactions ", sum(sol[2]))
# print("decisions ", sol[3])
# print("timeseries ", sol[4])
# print("-------------------")
# print("max EOTD profit ", sol[0])
# print('gross profit in transactions ', abs(sum(sol[1])) - abs(sum(sol[2])))
# print(' --- actions ---')
# print(sol[5])

# solution = ConstrainedDecisions2()
# sol = solution.maxProfit(timeseries, transactions, budget)
# print(timeseries)
# print("initial budget ", budget)
# print("profits ", sol[1])
# print("profits with transactions ", sum(sol[1]))
# print("costs ", sol[2])
# print("costs in transactions ", sum(sol[2]))
# print("decisions ", sol[3])
# print("timeseries ", sol[4])
# print("-------------------")
# print("max EOTD profit ", sol[0])
# print('gross profit in transactions ', abs(sum(sol[1])) - abs(sum(sol[2])))
# print(' --- actions ---')
# print(sol[5])

def create_vector(array, indices):
    buy_ = indices[0]
    sell_ = indices[1]
    arr = np.array(["hold" for i in range(len(array))], dtype=str)

    for i, idx in enumerate(buy_):
        import pdb;pdb.set_trace()
        arr[idx] = "buy_{}".format(idx)

    for i, idx in enumerate(sell_):
        arr[idx] = "sell_{}".format(i)
    return arr

solver = ListDecisions()
results = []
vectors = []
for series in timeseries:
    max_profit, transaction_list = solver.maxProfit(transactions, series)
    vector = create_vector(series, transaction_list)
    vectors.append(vector)
    results.append((max_profit, transaction_list))
    print(vector)
print(vectors)
print(results)