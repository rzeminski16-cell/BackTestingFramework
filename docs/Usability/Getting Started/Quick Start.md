---
tags:
  - usability/getting-started
  - tutorial
---

# Quick Start

Run your first backtest in under 5 minutes.

---

## Option A: Using the GUI

1. Launch the main GUI:
   ```bash
   python ctk_main_gui.py
   ```
2. Click **Backtesting**
3. Select a strategy from the registered strategy list
4. Choose a security (e.g. `AAPL`)
5. Set your capital (e.g. `100000`) and leave other defaults
6. Click **Run Backtest**
7. Results appear in the GUI. Reports are saved to `logs/backtests/`

---

## Option B: Using the Python API

```python
from pathlib import Path
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Config.config import BacktestConfig, CommissionConfig
from Classes.Data.data_loader import DataLoader
from Classes.Analysis.performance_metrics import PerformanceMetrics
from strategies.your_strategy import YourStrategy

# 1. Configure
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig()       # Default: 0.1% per trade
)

# 2. Load data
loader = DataLoader(Path('raw_data/daily'))
data = loader.load_csv('AAPL')

# 3. Create strategy
strategy = YourStrategy()

# 4. Run
engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)

# 5. Inspect results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Trades: {result.num_trades}")

# Full metrics (Sharpe, Sortino, drawdown, ...) come from PerformanceMetrics:
metrics = PerformanceMetrics.calculate_metrics(result)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

---

## What Happens Under the Hood

When you run a backtest, the engine:

1. Loads CSV data and validates required columns
2. Calls `strategy.prepare_data()` to pre-calculate any custom indicators
3. Iterates bar-by-bar through the data
4. On each bar, asks the strategy for a [[Signal Types|signal]]
5. Executes trades at the closing price, deducting [[Configuration Options|commission]]
6. Tracks equity, positions, and trade history
7. Returns a result object with [[Metrics Glossary|50+ performance metrics]]

For the full execution flow, see [[Backtest Execution Flow]].

---

## Next Steps

| Want to... | Go to... |
|---|---|
| Test across multiple securities | [[Portfolio Backtest]] |
| Find optimal parameters | [[Walk-Forward Optimisation]] |
| Build your own strategy | [[Adding a New Strategy]] |
| Understand the output metrics | [[Metrics Glossary]] |
