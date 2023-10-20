import backtrader as bt
import backtrader.analyzers as btanalyzers

class LSTMStrategy(bt.Strategy):
    params = (
        ('stop_loss', 0.03),  # 3% stop loss
        ('take_profit', 0.05),  # 5% profit target
        ('risk_percentage', 0.01),  # risk 1% of total portfolio per trade
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None

    def next(self):
        # Use the LSTM model to predict the next price
        input_data = torch.tensor(self.data_close[-len(X_train):], dtype=torch.float32)
        with torch.no_grad():
            predicted_price = model(input_data)

        # Define a threshold for buy/sell decisions
        threshold = 0.02  # e.g., 2%

        # Buy strategy
        if predicted_price > self.data_close[0] * (1 + threshold) and not self.position:
            size = self.calculate_position_size(self.data_close[0], self.data_close[0] * (1 - self.params.stop_loss))
            self.buy(size=size, exectype=bt.Order.Market)
            self.sell(size=size, price=self.data_close[0] * (1 + self.params.take_profit), exectype=bt.Order.Limit, oco=self.order)
            self.sell(size=size, price=self.data_close[0] * (1 - self.params.stop_loss), exectype=bt.Order.Stop, oco=self.order)

        # Sell strategy
        elif predicted_price < self.data_close[0] * (1 - threshold) and self.position:
            size = self.calculate_position_size(self.data_close[0], self.data_close[0] * (1 + self.params.stop_loss))
            self.sell(size=size, exectype=bt.Order.Market)
            self.buy(size=size, price=self.data_close[0] * (1 - self.params.take_profit), exectype=bt.Order.Limit, oco=self.order)
            self.buy(size=size, price=self.data_close[0] * (1 + self.params.stop_loss), exectype=bt.Order.Stop, oco=self.order)

    def calculate_position_size(self, entry_price, stop_loss_level):
        risk_per_share = abs(entry_price - stop_loss_level)
        capital_risked = self.broker.getvalue() * self.params.risk_percentage
        size = capital_risked / risk_per_share
        return size

# Create a Backtrader Cerebro engine
cerebro = bt.Cerebro()

# Add the LSTM strategy to the Cerebro engine
cerebro.addstrategy(LSTMStrategy)

# Load the TSLA data into a Backtrader data feed
data = bt.feeds.PandasData(dataname=df, datetime='date', open='open_price', high='high_price', low='low_price', close='close_price', volume='volume')
cerebro.adddata(data)

# Set the initial cash for the backtest
cerebro.broker.set_cash(100000)

# Add analyzers for Sharpe Ratio, Drawdown, and CAGR
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual_return')

# Add TradeAnalyzer to get details about trades
cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')

# Run the backtest
results = cerebro.run()
trade_analysis = results[0].analyzers.tradeanalyzer.get_analysis()

# Print the performance metrics
print(f"Sharpe Ratio: {results[0].analyzers.sharpe.get_analysis()['sharperatio']:.3f}")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
annual_returns = results[0].analyzers.annual_return.get_analysis()
cagr = (annual_returns[list(annual_returns.keys())[-1]]/annual_returns[list(annual_returns.keys())[0]])**(1/len(annual_returns))-1
print(f"CAGR: {cagr*100:.2f}%")

print(f"Total Trades: {trade_analysis['total']['total']}")
print(f"Winning Trades: {trade_analysis['won']['total']}")
print(f"Losing Trades: {trade_analysis['lost']['total']}")