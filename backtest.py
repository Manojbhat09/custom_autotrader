import backtrader as bt
import backtrader.analyzers as btanalyzers

class LSTMStrategy(bt.Strategy):
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
            self.order = self.buy()

        # Sell strategy
        elif predicted_price < self.data_close[0] * (1 - threshold) and self.position:
            self.order = self.sell()

        # Hold strategy is implicit (i.e., do nothing if conditions for buy/sell are not met)

    def stop(self):
        # Print daily performance at the end of the day
        print(f"Net Profit/Loss for the day: {self.broker.getvalue() - 100000:.2f}")
        print(f"Number of Trades: {len(self)}")
        # ... add more daily performance metrics as needed

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