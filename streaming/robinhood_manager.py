
import robin_stocks.robinhood as rh
import streamlit as st
import logging 
import pandas as pd

class RobinhoodManager():
    def __init__(self, username, password):
        rh.login(username=username, password=password)
        self.rh = rh
    @st.cache(allow_output_mutation=True)
    def get_watchlist_by_name(self, list_name):
        try:
            watchlist = rh.get_watchlist_by_name(list_name)
            symbols = [instrument['symbol'] for instrument in watchlist['results']] if watchlist and 'results' in watchlist else []
        except Exception as e:
            logging.error(f"Error getting all unique symbols: {e}")
            print(e)
            symbols = []
        return symbols
    
    @st.cache(allow_output_mutation=True)
    def get_all_watchlists(self):
        all_watchlists = rh.get_all_watchlists()
        return all_watchlists
    
    @st.cache(allow_output_mutation=True)
    def get_all_unique_symbols(self):
        all_unique_symbols = set()
        try:
            all_watchlists = rh.get_all_watchlists()
            for watchlist in all_watchlists['results']:
                watchlist_name = watchlist['display_name']
                instruments = self.get_watchlist_by_name(watchlist_name)
                all_unique_symbols.update(instruments)
        except Exception as e:
            logging.error(f"Error getting all unique symbols: {e}")
            print(e)
        return all_unique_symbols
    
    def get_portfolio_holdings(self):
        holdings = self.rh.account.build_holdings()
        portfolio_data = []
        for ticker, details in holdings.items():
            portfolio_data.append({
                'Ticker': ticker,
                'Equity': float(details['equity']),
                'Percentage': float(details['percentage'])
            })
        return portfolio_data

    def get_crypto_holdings(self):
        crypto_positions = self.rh.crypto.get_crypto_positions()
        crypto_data = []
        for position in crypto_positions:
            quantity = float(position['quantity'])
            if quantity > 0:  # Only include if quantity is greater than zero
                crypto_data.append({
                    'Currency': position['currency']['code'],
                    'Quantity': quantity
                })
        return crypto_data
    
    def get_option_holdings(self):
        option_positions = self.rh.options.get_open_option_positions()
        option_data = []
        for position in option_positions:
            # Assuming 'quantity' and 'average_price' are keys in the position dictionary
            quantity = float(position['quantity'])
            average_price = float(position['average_price'])
            if quantity > 0:  # Only include if quantity is greater than zero
                option_data.append({
                    'Option': position['chain_symbol'],  # Replace 'chain_symbol' with actual key for option symbol
                    'Value': quantity * average_price
                })
        return option_data
    
    def get_crypto_value_holdings(self):
        crypto_positions = rh.crypto.get_crypto_positions()
        crypto_data = []
        for position in crypto_positions:
            quantity = float(position['quantity'])
            if quantity > 0:  # Only include if quantity is greater than zero
                currency_code = position['currency']['code']
                crypto_quote = self.get_current_crypto_price(currency_code)
                current_price = float(crypto_quote['mark_price'])
                total_value = quantity * current_price
                crypto_data.append({
                    'Currency': currency_code,
                    'Quantity': quantity,
                    'Value': total_value
                })
        return crypto_data
    
    def get_current_crypto_price(self, symbol):
        try:
            crypto_quote = self.rh.crypto.get_crypto_quote(symbol)
            current_price = float(crypto_quote['mark_price'])
            return current_price
        except Exception as e:
            logging.error(f"Error getting real-time crypto data for {symbol}: {e}")
            print(e)
            return None

    
    def get_portfolio_value(self):
        try:
            portfolio_info = self.rh.profiles.load_portfolio_profile()
            market_value = portfolio_info['market_value']
            return market_value
        except Exception as e:
            logging.error(f"Error getting portfolio value: {e}")
            print(e)
            return None

    def get_holdings_info(self):
        try:
            holdings_info = self.rh.account.build_holdings()
            return holdings_info
        except Exception as e:
            logging.error(f"Error getting holdings info: {e}")
            print(e)
            return None
        
    def get_option_calls(self):
        try:
            url = 'https://api.robinhood.com/options/instruments/'
            payload = {'type': 'call'}
            option_calls_data = self.rh.helper.request_get(url, 'regular', payload)
            return option_calls_data
        except Exception as e:
            logging.error(f"Error getting option calls: {e}")
            print(e)
            return None
        
    def get_my_option_positions(self):
        try:
            url = 'https://api.robinhood.com/options/positions/'
            option_positions_data = self.rh.helper.request_get(url, 'results')
            return option_positions_data
        except Exception as e:
            logging.error(f"Error getting option positions: {e}")
            print(e)
            return None
        
    def get_option_instrument_info(self, option_url):
        try:
            option_instrument_data = self.rh.helper.request_get(option_url, 'regular')
            return option_instrument_data
        except Exception as e:
            logging.error(f"Error getting option instrument info: {e}")
            print(e)
            return None

    def get_crypto_historicals(self, symbol, interval, span):
        try:
            historical_data = self.rh.crypto.get_crypto_historicals(symbol, interval=interval, span=span)
            return historical_data
        except Exception as e:
            logging.error(f"Error getting crypto historicals for {symbol}: {e}")
            print(e)
            return None

    async def stream_crypto_data(self, symbol):
        while True:
            try:
                current_data = self.rh.crypto.get_crypto_quote(symbol)
                print(current_data)
                await asyncio.sleep(1)  # Streaming at 1-second intervals
            except Exception as e:
                logging.error(f"Error streaming crypto data for {symbol}: {e}")
                print(e)

    def execute_crypto_order(self, symbol, quantity, transaction_type):
        try:
            if transaction_type == 'buy':
                order = self.rh.orders.order_buy_crypto_by_quantity(symbol, quantity)
            elif transaction_type == 'sell':
                order = self.rh.orders.order_sell_crypto_by_quantity(symbol, quantity)
            return order
        except Exception as e:
            logging.error(f"Error executing {transaction_type} order for {symbol}: {e}")
            print(e)
            return None

    async def stream_crypto_historicals(self, symbol, interval, span):
        while True:
            try:
                historical_data = self.rh.crypto.get_crypto_historicals(symbol, interval=interval, span=span)
                yield historical_data
                await asyncio.sleep(60)  # Pause for 60 seconds (or your preferred interval)
            except Exception as e:
                logging.error(f"Error streaming crypto historicals for {symbol}: {e}")
                print(e)
                await asyncio.sleep(60)  # Wait before retrying

    def execute_crypto_order(self, symbol, amount, order_type):
        try:
            if order_type == 'buy':
                order_response = self.rh.orders.order_buy_crypto_by_quantity(symbol, amount)
            elif order_type == 'sell':
                order_response = self.rh.orders.order_sell_crypto_by_quantity(symbol, amount)
            return order_response
        except Exception as e:
            logging.error(f"Error executing {order_type} order for {symbol}: {e}")
            print(e)
            return None

    def get_crypto_instrument_info(self, symbol):
        try:
            crypto_info = self.rh.crypto.get_crypto_info(symbol)
            return crypto_info
        except Exception as e:
            logging.error(f"Error getting crypto instrument info for {symbol}: {e}")
            print(e)
            return None

    def get_real_time_crypto_quote(self, symbol):
        try:
            real_time_quote = self.rh.crypto.get_crypto_quote(symbol)
            return real_time_quote
        except Exception as e:
            logging.error(f"Error getting real-time quote for {symbol}: {e}")
            print(e)
            return None

    def get_real_time_crypto_data(self, ticker, debug=False):
        """
        Fetches the current crypto price along with other details like open, high, low, close prices, and volume
        from the RobinhoodManager.
        """
        try:
            crypto_quote = self.rh.get_crypto_quote(ticker)

            current_data_point = {
                'Timestamp': pd.Timestamp(crypto_quote['updated_at']),
                'Open': float(crypto_quote['open_price']),
                'High': float(crypto_quote['high_price']),
                'Low': float(crypto_quote['low_price']),
                'Close': float(crypto_quote['mark_price']),  # assuming mark price as close
                'Volume': float(crypto_quote['volume']),
            }

            if debug:
                print("[UPDATE] Current crypto data: ")
                print(current_data_point)
            return current_data_point

        except Exception as e:
            print(f" not able to get data ")
            print(e)
        return None
            
    

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std()
        return (returns.mean() - risk_free_rate) / downside_deviation

    def calculate_value_at_risk(self, returns, confidence_level=0.05):
        return returns.quantile(confidence_level, interpolation='higher')

    def calculate_conditional_value_at_risk(self, returns, confidence_level=0.05):
        var = self.calculate_value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()

    def calculate_profit(self, buy_price, sell_price, quantity):
        commission_rate = 0.0035  # Assuming 0.35% commission
        cost = buy_price * quantity
        revenue = sell_price * quantity
        commission_cost = cost * commission_rate
        commission_revenue = revenue * commission_rate
        profit = (revenue - commission_revenue) - (cost + commission_cost)
        return profit
        
    def calculate_bollinger_bands(self, data, window_size=20, num_std_dev=2):
        sma = data.rolling(window=window_size).mean()
        std_dev = data.rolling(window=window_size).std()
        bollinger_upper = sma + (std_dev * num_std_dev)
        bollinger_lower = sma - (std_dev * num_std_dev)
        return bollinger_upper, bollinger_lower

    def enhanced_alerts(self, historical_data, indicators_settings):
        alerts = []

        # SMA Alert
        if 'sma' in indicators_settings:
            sma = self.calculate_moving_average(historical_data, indicators_settings['sma']['window'])
            if sma.iloc[-1] < indicators_settings['sma']['threshold']:
                alerts.append("SMA Alert: Below threshold.")

        # RSI Alert
        if 'rsi' in indicators_settings:
            rsi = self.calculate_rsi(historical_data, indicators_settings['rsi']['window'])
            if rsi.iloc[-1] > indicators_settings['rsi']['threshold']:
                alerts.append("RSI Alert: Above threshold.")

        # Bollinger Bands Alert
        if 'bollinger' in indicators_settings:
            upper_band, lower_band = self.calculate_bollinger_bands(historical_data, indicators_settings['bollinger']['window'])
            if historical_data.iloc[-1] > upper_band.iloc[-1]:
                alerts.append("Bollinger Upper Alert: Price above upper band.")
            elif historical_data.iloc[-1] < lower_band.iloc[-1]:
                alerts.append("Bollinger Lower Alert: Price below lower band.")

        return alerts

    def calculate_moving_average(self, data, window_size):
        return data.rolling(window=window_size).mean()

    def calculate_rsi(self, data, window_size=14):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        gain = up.rolling(window=window_size).mean()
        loss = down.abs().rolling(window=window_size).mean()
        RS = gain / loss
        return 100 - (100 / (1 + RS))

    def portfolio_optimization(self, symbols, historical_returns, risk_free_rate=0.01):
        # Calculate mean returns and covariance
        mean_returns = historical_returns.mean()
        cov_matrix = historical_returns.cov()

        # Sharpe ratio as the optimization function
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Negative because we want to maximize Sharpe ratio

        # Constraints and bounds
        num_assets = len(symbols)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        initial_guess = num_assets * [1. / num_assets,]

        # Optimization
        result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        # Output the optimized weights
        return result.x
    
if __name__ == "__main__":
    manager = RobinhoodManager(username='manojbhat09@gmail.com', password='MENkeys796@09@')
    watchlist = manager.get_watchlist_by_name("Tech")
    print(watchlist)  # Should print the symbols in the Tech watchlist

    # Get historical data for a cryptocurrency
    historical_data = manager.get_crypto_historicals('BTC', '5minute', 'week')
    print(historical_data)

    # Start streaming crypto data (This would need to be run in an async loop)
    # asyncio.run(manager.stream_crypto_data('BTC'))

    # Execute a crypto order
    order = manager.execute_crypto_order('BTC', 1, 'buy')
    print(order)

    # Calculate profit
    profit = manager.calculate_profit(50000, 51000, 1)  # Example values
    print(f"Profit: {profit}")

    # Example: Calculate Sharpe Ratio for a set of returns
    sample_returns = [0.01, 0.02, -0.01, 0.03, 0.02]
    sharpe_ratio = manager.calculate_sharpe_ratio(sample_returns)
    print(f"Sharpe Ratio: {sharpe_ratio}")