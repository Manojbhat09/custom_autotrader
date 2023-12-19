
import robin_stocks.robinhood as r
import time 
from datetime import datetime
import numpy as np
import os

class RobinhoodCryptoTransaction:
    def __init__(self, username, password, crypto_symbol, log_file="transaction_log.txt"):
        self.rh = r.login(username, password)
        self.crypto_symbol = crypto_symbol
        self.log_file = log_file
        self.load_profile()
        self.log("Logged in successfully.")

    def load_profile(self):
        self.profile = r.load_account_profile()

    def get_current_cash(self):
        self.load_profile()
        return float(self.profile['cash'])

    def get_crypto_quote(self):
        crypto_info = r.crypto.get_crypto_quote(self.crypto_symbol)
        return float(crypto_info['bid_price']), float(crypto_info['ask_price']), float(crypto_info['mark_price'])

    def calculate_spread(self, bid_price, ask_price):
        return ask_price - bid_price

    def calculate_transaction_cost(self, amount, spread_percentage):
        return amount * spread_percentage / 100
    
    def calculate_slope(self, intervals=6, sleep_duration=1):
        prices = []
        for _ in range(intervals):
            _, current_price, _ = self.get_crypto_quote()
            prices.append(current_price)
            time.sleep(sleep_duration)

        slopes = np.diff(prices)
        return np.mean(slopes)

    def calculate_realized_profit(self, buy_expected_value, buy_actual_value, sell_expected_value, sell_actual_value, transaction_cost_buy, transaction_cost_sell):
        # Cost of buying (difference between expected and actual buy values)
        cost_buy = buy_expected_value - buy_actual_value
        # Gain from selling (difference between actual and expected sell values)
        gain_sell = sell_actual_value - sell_expected_value
        # Net profit: Gain minus Cost minus Transaction Costs
        return gain_sell - cost_buy - transaction_cost_buy - transaction_cost_sell

    def is_profitable(self, buy_expected_value, buy_actual_value, sell_expected_value, sell_actual_value, transaction_cost_buy, transaction_cost_sell):
        total_cost = transaction_cost_buy + transaction_cost_sell
        total_profit = self.calculate_realized_profit(buy_expected_value, buy_actual_value, sell_expected_value, sell_actual_value, transaction_cost_buy, transaction_cost_sell)
        return total_profit > total_cost

    def calculate_realized_profit_direct(self, buy_actual_value, sell_actual_value, transaction_cost_buy, transaction_cost_sell):
        # Net profit: Gain from selling minus Cost of buying minus Transaction Costs
        return (sell_actual_value - buy_actual_value) - (transaction_cost_buy + transaction_cost_sell)

    def calculate_profit_loss(self, buy_price, sell_price, quantity, transaction_cost_buy, transaction_cost_sell):
        total_cost = quantity * buy_price + transaction_cost_buy
        total_revenue = quantity * sell_price - transaction_cost_sell
        return total_revenue - total_cost

    def execute_favbuy(self, amount):
        # Get current quotes
        buy_price, _, _ = self.get_crypto_quote()
        quantity = amount / buy_price
        transaction_cost = self.calculate_transaction_cost(buy_price, 0.36)

        # Set a more favorable buy price, e.g., a bit lower than the current market price
        favorable_buy_price = buy_price * 0.99  # Adjust this factor based on strategy
        r.order_buy_limit(self.crypto_symbol, quantity, favorable_buy_price)

        return favorable_buy_price, quantity, transaction_cost
    
    def execute_favsell(self, quantity):
        # Get current quotes
        _, sell_price, _ = self.get_crypto_quote()
        transaction_cost = self.calculate_transaction_cost(sell_price, 0.36)

        # Set a more favorable sell price, e.g., a bit higher than the current market price
        favorable_sell_price = sell_price * 1.01  # Adjust this factor based on strategy
        r.order_sell_limit(self.crypto_symbol, quantity, favorable_sell_price)

        return favorable_sell_price, transaction_cost
    
    def quick_buy(self, amount, timeout=300, price_adjustment_factor=1.01):
        start_time = time.time()
        while True:
            current_time = time.time()
            current_buy_price, _, _ = self.get_crypto_quote()
            if current_time - start_time >= timeout:
                # Adjust price to ensure execution
                quick_buy_price = current_buy_price * price_adjustment_factor
                # Execute quick buy order here
                r.order_buy_limit(self.crypto_symbol, amount / quick_buy_price, quick_buy_price)
                break
            time.sleep(60)

    def quick_sell(self, quantity, timeout=300, price_adjustment_factor=0.99):
        start_time = time.time()
        while True:
            current_time = time.time()
            _, current_sell_price, _ = self.get_crypto_quote()
            if current_time - start_time >= timeout:
                # Adjust price to ensure execution
                quick_sell_price = current_sell_price * price_adjustment_factor
                # Execute quick sell order here
                r.order_sell_limit(self.crypto_symbol, quantity, quick_sell_price)
                break
            time.sleep(60)
    
    def execute_buy(self, amount):
        # Fetches the current buy price and calculates the quantity based on the amount
        buy_price, _, _ = self.get_crypto_quote()
        quantity = amount / buy_price
        transaction_cost = self.calculate_transaction_cost(buy_price, 0.36)
        
        # Executes the buy order
        r.order_buy_crypto_by_price(self.crypto_symbol, amount)

        return buy_price, quantity, transaction_cost

    def execute_sell(self, quantity):
        # Fetches the current sell price
        _, sell_price, _ = self.get_crypto_quote()
        transaction_cost = self.calculate_transaction_cost(sell_price, 0.36)
        
        # Executes the sell order
        # Note: Replace `amount` with the desired sell amount or use a different method for limit orders
        r.order_sell_crypto_by_price(self.crypto_symbol, quantity * sell_price)

        return sell_price, transaction_cost
    
    def decide_and_execute_trade(self, action_type, amount):
        some_positive_threshold = 0.02  # 2% upward movement
        some_negative_threshold = -0.02  # 2% downward movement
        avg_slope = self.calculate_slope()
        self.log(f"the avg_slope is {avg_slope}")
        if action_type == 'buy':
            if avg_slope > some_positive_threshold:
                self.log(f"the quick_buy {amount}")
                self.quick_buy(amount)
            else:
                self.log(f"the execute_buy {amount}")
                self.execute_buy(amount)
        elif action_type == 'sell':
            if avg_slope < some_negative_threshold:
                self.log(f"the quick_sell {amount}")
                self.quick_sell(amount)
            else:
                self.log(f"the execute_sell {amount}")
                self.execute_sell(amount)


    def check_crypto_positions(self):
        return r.get_crypto_positions()
    
    def cancel_all_orders(self):
        r.cancel_all_crypto_orders()

    def execute_buy_sell_invest_profit(self, invest_percent, target_profit_percentage):
        cash = self.get_current_cash()
        self.log(f"the cash is {cash}")
        investment_amount = cash * invest_percent /100  # Investing 20% of current cash
        self.log(f"the investment_amount is {investment_amount}")
        # Execute buy
        buy_price, quantity, _ = self.execute_buy(investment_amount)

        target_sell_price = buy_price * (1 + target_profit_percentage / 100)
        self.log(f"the target_sell_price is {target_sell_price}")
        self.decide_and_execute_trade("sell", quantity)
        self.log(f"Done sell")
        pass

    def execute_buy_and_sell_when_profitable(self, buy_amount, target_profit_percentage):
        # Buy Ethereum without seeing 
        buy_price, quantity_bought, transaction_cost_buy = self.execute_buy(buy_amount)

        # Calculate the target sell price based on the desired profit percentage
        target_sell_price = buy_price * (1 + target_profit_percentage / 100)
        self.log(f"target sell price: {target_sell_price}")
        while True:
            # Continuously check the current Ethereum price
            bid, current_sell_price, mark = self.get_crypto_quote()
            self.log(f"current sell price 1  {bid}, {current_sell_price}, {mark}")
            # Check if current price meets the target sell price
            if current_sell_price >= target_sell_price:
                # Calculate the expected revenue from selling
                revenue = quantity_bought * current_sell_price
                transaction_cost_sell = self.calculate_transaction_cost(current_sell_price, 0.36)
                self.log(f"executing sell for {transaction_cost_sell}")
                # Execute sell
                self.execute_sell(quantity_bought)

                # Calculate and return net profit
                net_profit = self.calculate_profit_loss(buy_price, current_sell_price, quantity_bought, transaction_cost_buy, transaction_cost_sell)
                return net_profit

            # Wait for a short period before checking the price again
            time.sleep(60)  # Check every 60 seconds

    def execute_advanced_buy_sell(self, amount, target_profit_percentage):
        self.log(f"Initiating buy for {amount} of {self.crypto_symbol}.")
        buy_price, quantity, transaction_cost_buy = self.execute_buy(amount)
        self.log(f"Bought {quantity} of {self.crypto_symbol} at price {buy_price}.")

        target_sell_price = buy_price * (1 + target_profit_percentage / 100)

        while True:
            _, current_sell_price, _ = self.get_crypto_quote()

            potential_revenue = quantity * current_sell_price
            transaction_cost_sell = self.calculate_transaction_cost(current_sell_price, 0.36)
            self.log(f"transaction_cost_sell: {transaction_cost_sell}")

            if self.is_profitable(buy_price * quantity, potential_revenue, buy_price * quantity, potential_revenue, transaction_cost_buy, transaction_cost_sell):
                self.execute_sell(quantity)
                actual_profit = self.calculate_profit_loss(buy_price, current_sell_price, quantity, transaction_cost_buy, transaction_cost_sell)
                self.log(f"Sold {quantity} of {self.crypto_symbol} at price {current_sell_price}. Profit: {actual_profit}")
                return actual_profit

            time.sleep(10)  # Check every 10 seconds
            self.log(f"Checked price. Current: {current_sell_price}, Target: {target_sell_price}")

    def log(self, message):
        with open(self.log_file, "a") as file:
            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

if __name__ == "__main__":  
    '''
    username, password = os.environ['RH_USERNAME'], os.environ['RH_PASSWORD']
    from robinhood_crypto_transactions import RobinhoodCryptoTransaction; rh_transaction = RobinhoodCryptoTransaction(username, password, 'BTC')
    rh_transaction.execute_buy_sell_invest_profit(5, 20)
    '''
    # Example Usage
    username, password = os.environ['RH_USERNAME'], os.environ['RH_PASSWORD']
    rh_transaction = RobinhoodCryptoTransaction(username, password, 'BTC')
    bid_price, ask_price, mark_price = rh_transaction.get_crypto_quote()
    spread = rh_transaction.calculate_spread(bid_price, ask_price)
    transaction_cost_buy = rh_transaction.calculate_transaction_cost(bid_price, 0.36)
    transaction_cost_sell = rh_transaction.calculate_transaction_cost(ask_price, 0.36)

    # Simulate a buy and sell transaction
    buy_expected_value = bid_price
    buy_actual_value = bid_price - rh_transaction.calculate_transaction_cost(bid_price, 0.36)
    sell_expected_value = ask_price
    sell_actual_value = ask_price - rh_transaction.calculate_transaction_cost(ask_price, 0.36)

    # Check if the transaction is profitable
    is_profitable = rh_transaction.is_profitable(buy_expected_value, buy_actual_value, sell_expected_value, sell_actual_value, transaction_cost_buy, transaction_cost_sell)
    print("Is the transaction profitable?", is_profitable)
