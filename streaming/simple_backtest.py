import pickle
import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker

ticker.MAXTICKS = 1500
# Load data
data = pickle.load(open("realtimedata", "rb"))
predictions = pickle.load(open("predictions2", "rb"))
realtime_data = pickle.load(open("realtimedata2", "rb"))
past_predictions = pickle.load(open("predictions", "rb"))

# Define a custom data feed to integrate predictions
class PredictiveData(bt.feed.DataBase):
    lines = ('prediction',)

    def start(self):
        super(PredictiveData, self).start()
        self.idx = 0

    def _load(self):
        if self.idx >= len(predictions):
            return False

        self.lines.datetime[0] = bt.date2num(predictions['Timestamp'].iloc[self.idx])
        self.lines.open[0] = self.lines.close[0] = self.lines.high[0] = self.lines.low[0] = realtime_data['Close'].iloc[self.idx]
        self.lines.volume[0] = realtime_data['Volume'].iloc[self.idx]
        self.lines.prediction[0] = predictions['Prediction'].iloc[self.idx]

        self.idx += 1
        return True

# Trading Strategy
class MyStrategy(bt.Strategy):
    params = (('stop_loss', 0.02), ('take_profit', 0.02), ('risk_percentage', 0.01))

    def log(self, text, dt=None):
        ''' Logging function '''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {text}')

    def __init__(self):
        self.order = None
        self.position_size = 0

    def next(self):
        current_price = self.data.close[0]
        predicted_price = self.data.prediction[0]
        self.log(f'Current Price: {current_price}, Predicted Price: {predicted_price}')

        if self.order:
            return

        # Buy strategy
        if predicted_price > current_price * (1 + self.p.stop_loss) and not self.position:
            size = self.calculate_position_size(current_price, current_price * (1 - self.p.stop_loss))
            self.order = self.buy(size=size)
            self.log(f'BUY ORDER PLACED AT {current_price}, SIZE {size}')
            self.sell(size=size, price=current_price * (1 + self.p.take_profit), exectype=bt.Order.Limit)
            self.sell(size=size, price=current_price * (1 - self.p.stop_loss), exectype=bt.Order.Stop)

        # Sell strategy
        elif predicted_price < current_price * (1 - self.p.stop_loss) and self.position:
            size = self.calculate_position_size(current_price, current_price * (1 + self.p.stop_loss))
            self.order = self.sell(size=size)
            self.log(f'SELL ORDER PLACED AT {current_price}, SIZE {size}')
            self.buy(size=size, price=current_price * (1 - self.p.take_profit), exectype=bt.Order.Limit)
            self.buy(size=size, price=current_price * (1 + self.p.stop_loss), exectype=bt.Order.Stop)

    def calculate_position_size(self, entry_price, stop_loss_level):
        risk_per_share = abs(entry_price - stop_loss_level)
        capital_risked = self.broker.getvalue() * self.p.risk_percentage
        size = capital_risked / risk_per_share
        return size

# Backtrader setup
cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)

# Add data feed
data_feed = PredictiveData(dataname=realtime_data)
cerebro.adddata(data_feed)

# Add analyzers
cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trade_analyzer")
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe_ratio", riskfreerate=0.0)
cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(btanalyzers.Returns, _name="returns")

# Set initial capital
cerebro.broker.setcash(100000)

# Run the strategy
results = cerebro.run()

# Print analyzers
first_strategy = results[0]
# Extracting the strategy
strategy = results[0]

print("Trade Analysis:", first_strategy.analyzers.trade_analyzer.get_analysis())
print("Sharpe Ratio:", first_strategy.analyzers.sharpe_ratio.get_analysis())
print("Drawdown:", first_strategy.analyzers.drawdown.get_analysis())
print("Returns:", first_strategy.analyzers.returns.get_analysis())

# Plotting
# plot_path = "./backtrader_plot.png"
# cerebro.plot(style='candlestick', barup='green', bardown='red')
# plt.savefig(plot_path)
# print(f"Plot saved to {plot_path}")

# Plot standard backtrader plot
fig = cerebro.plot()[0][0]
fig.savefig("standard_plot.png")

# Additional Plots

# Transactions
transactions = strategy.analyzers.transactions.get_analysis()
tx_df = pd.DataFrame([(item[0].date(), item[1], item[2]) for item in transactions.items()], columns=['date', 'price', 'size'])
tx_df.set_index('date', inplace=True)

# Plot trades on price chart
plt.figure(figsize=(12, 6))
plt.plot(realtime_data['Close'], label='Close Price', alpha=0.5)
plt.scatter(tx_df[tx_df['size'] > 0].index, tx_df[tx_df['size'] > 0]['price'], label='Buy', marker='^', color='green')
plt.scatter(tx_df[tx_df['size'] < 0].index, tx_df[tx_df['size'] < 0]['price'], label='Sell', marker='v', color='red')
plt.title('Trade Entry and Exit Points')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig("trades_on_price_chart.png")

# Profit/Loss per Trade
trade_results = strategy.analyzers.trade_analyzer.get_analysis()
trades = pd.DataFrame(trade_results['closed']).T
trades['pnl'] = trades['pnl'].astype(float)

plt.figure(figsize=(12, 6))
trades['pnl'].plot(kind='bar', color='blue')
plt.title('Profit/Loss per Trade')
plt.xlabel('Trade Number')
plt.ylabel('Profit/Loss')
plt.savefig("profit_loss_per_trade.png")

# Histogram of Returns
returns = pd.Series(strategy.analyzers.returns.get_analysis()['returns'])
plt.figure(figsize=(12, 6))
returns.hist(bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.savefig("histogram_of_returns.png")

print("Plots saved successfully.")