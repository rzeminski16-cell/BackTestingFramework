# Alternative System Usage

This guide covers how to use the simplified backtesting system in the `backtesting/` module.

---

## Quick Start

### Running a Backtest

```python
from backtesting.engine import BacktestEngine
from backtesting.strategies.examples.sma_crossover import SMACrossoverStrategy
from backtesting.loader import load_data

# Load data
data = load_data('raw_data/daily/AAPL_daily.csv')

# Create strategy
strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)

# Configure backtest
config = {
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1%
}

# Run backtest
engine = BacktestEngine(config)
result = engine.run(data, strategy)

# View results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Total Trades: {result.trade_count}")
```

---

## Creating a Strategy

### Strategy Template

```python
from backtesting.strategies.base import BaseStrategy


class MyStrategy(BaseStrategy):
    """Simple strategy description."""

    def __init__(self, param1: int = 20, param2: float = 2.0):
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, data):
        """
        Generate trading signals for entire dataset.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of (index, signal_type, price) tuples
        """
        signals = []

        # Calculate indicators
        data['sma'] = data['close'].rolling(self.param1).mean()

        for i in range(self.param1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]

            # Entry condition
            if previous['close'] < previous['sma'] and current['close'] > current['sma']:
                signals.append((i, 'BUY', current['close']))

            # Exit condition
            elif previous['close'] > previous['sma'] and current['close'] < current['sma']:
                signals.append((i, 'SELL', current['close']))

        return signals
```

---

## Included Example Strategies

### SMA Crossover

```python
from backtesting.strategies.examples.sma_crossover import SMACrossoverStrategy

strategy = SMACrossoverStrategy(
    fast_period=10,
    slow_period=30
)
```

Buys when fast SMA crosses above slow SMA, sells on cross below.

### RSI Strategy

```python
from backtesting.strategies.examples.rsi_strategy import RSIStrategy

strategy = RSIStrategy(
    period=14,
    oversold=30,
    overbought=70
)
```

Mean reversion: buys when RSI < oversold, sells when RSI > overbought.

### Bollinger Bands

```python
from backtesting.strategies.examples.bollinger import BollingerStrategy

strategy = BollingerStrategy(
    period=20,
    num_std=2.0
)
```

Buys at lower band, sells at upper band.

---

## Configuration Options

```python
config = {
    # Capital settings
    'initial_capital': 100000,
    'position_size': 0.95,  # Use 95% of capital per trade

    # Commission
    'commission': 0.001,  # 0.1% per trade

    # Risk management
    'stop_loss': 0.05,  # 5% stop loss (optional)
}
```

---

## Accessing Results

```python
result = engine.run(data, strategy)

# Performance metrics
print(f"Total Return: {result.total_return:.2%}")
print(f"Annualized Return: {result.annualized_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")

# Trade statistics
print(f"Total Trades: {result.trade_count}")
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Profit Factor: {result.profit_factor:.2f}")

# Trade list
for trade in result.trades:
    print(f"{trade.entry_date} -> {trade.exit_date}: {trade.pnl_pct:.2%}")

# Equity curve
equity = result.equity_curve  # pandas Series
```

---

## Basic Optimization

```python
from backtesting.optimization.optimizer import GridOptimizer

# Define parameter ranges
param_grid = {
    'fast_period': [5, 10, 15, 20],
    'slow_period': [20, 30, 40, 50],
}

# Run optimization
optimizer = GridOptimizer(engine, data, SMACrossoverStrategy)
best_params, best_result = optimizer.optimize(
    param_grid,
    metric='sharpe_ratio'
)

print(f"Best Parameters: {best_params}")
print(f"Best Sharpe: {best_result.sharpe_ratio:.2f}")
```

---

## Data Loading

```python
from backtesting.loader import load_data

# Load single file
data = load_data('raw_data/daily/AAPL_daily.csv')

# Load with date filter
data = load_data(
    'raw_data/daily/AAPL_daily.csv',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Expected columns
# date, open, high, low, close, volume
# Additional indicator columns optional
```

---

## Comparison with Main System

### Running Same Strategy on Both

```python
# ALTERNATIVE SYSTEM
from backtesting.engine import BacktestEngine as AltEngine
from backtesting.strategies.examples.sma_crossover import SMACrossoverStrategy

alt_engine = AltEngine({'initial_capital': 100000})
alt_result = alt_engine.run(data, SMACrossoverStrategy(10, 30))


# MAIN SYSTEM
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Config.config import BacktestConfig
from strategies.sma_strategy import SMAStrategy  # You'd need to create this

config = BacktestConfig(
    symbol='AAPL',
    initial_capital=100000,
    data_path='raw_data/daily/AAPL_daily.csv'
)
main_engine = SingleSecurityEngine(config, SMAStrategy(10, 30))
main_result = main_engine.run()


# Compare results
print(f"Alt Return: {alt_result.total_return:.2%}")
print(f"Main Return: {main_result.total_return_pct:.2%}")
```

Results should be similar; differences may arise from:
- Execution price assumptions
- Commission calculation details
- Position sizing methods

---

## Limitations

The alternative system does not support:

- Portfolio mode with capital contention
- Vulnerability scoring
- Walk-forward optimization
- Factor analysis integration
- Advanced exit rules (partial exits, trailing stops via ADJUST_STOP)
- Full GUI suite

For these features, use the main system (`Classes/`).

---

## When to Graduate to Main System

Consider moving to the main system when you need:

```
□ Portfolio backtesting across multiple securities
□ Walk-forward parameter validation
□ E-ratio or factor analysis
□ Vulnerability-based position scoring
□ Complex exit rule testing
□ Integration with full GUI applications
□ Production-ready strategy deployment
```

---

## Related Documentation

- [Overview](OVERVIEW.md) — System comparison
- [Main System Quick Start](../overview/QUICK_START.md) — Getting started with main system
- [Strategy Development](../strategy-development/STRATEGY_GUIDE.md) — Creating strategies for main system
