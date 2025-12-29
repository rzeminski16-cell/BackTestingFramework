# Backtesting Guide

This guide covers how to run backtests effectively, interpret results, and use both the GUI and Python API.

## How Backtesting Works

A backtest simulates running a trading strategy on historical data:

1. **Load Data** - Read historical prices from CSV files
2. **Process Bar by Bar** - For each day:
   - Strategy analyzes data up to that point (no future data)
   - Strategy generates a signal (BUY, SELL, HOLD)
   - Trades execute at the closing price
   - Stop losses and take profits are checked
3. **Record Results** - All trades logged with entry/exit details
4. **Calculate Metrics** - Performance metrics computed from trade history

### Execution Model

- **Trades execute at closing prices** - Matches TradingView behavior
- **No lookahead bias** - Strategies only see past data
- **One position at a time** - In single-security mode
- **Long only** - Buying and selling, not short selling
- **Configurable commission** - Percentage or fixed per trade

---

## Using the GUI

### Launch

```bash
python run_gui.py
```

### Step 1: Select Mode

**Single Security Mode:**
- Tests one security at a time
- Each gets isolated capital
- Results per security

**Portfolio Mode:**
- Tests multiple securities together
- Shared capital pool
- Position limits apply
- See [Portfolio Mode Guide](PORTFOLIO_MODE.md) for details

### Step 2: Select Securities

- Select from the list (populated from `raw_data/` folder)
- Ctrl+click for multiple selections
- Use "Select All" for all securities

### Step 3: Choose Strategy

Select from the dropdown. Available strategies are discovered from the `strategies/` folder.

**Configure Parameters:**
Click "Configure Strategy Parameters" to adjust:
- Volume filter settings
- Stop loss multipliers
- Grace periods
- Risk percentage

### Step 4: Set Commission

| Mode | Description | Example |
|------|-------------|---------|
| Percentage | % of trade value | 0.001 = 0.1% |
| Fixed | Amount per trade | $3 per trade |

### Step 5: Set Capital

Initial capital for the backtest (e.g., $100,000).

### Step 6: Set Date Range (Optional)

Leave blank to use all available data, or set specific start/end dates.

### Step 7: Run and Review

1. Name your backtest
2. Click "Run Backtest"
3. View results in the results window
4. Check logs in `logs/backtests/{backtest_name}/`

---

## Using Python

For automation and more control:

### Basic Backtest

```python
from pathlib import Path
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Configure
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001  # 0.1%
    )
)

# Load data
loader = DataLoader(Path('raw_data'))
data = loader.load_csv('AAPL')

# Create strategy
strategy = AlphaTrendStrategy()

# Run backtest
engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)

# View results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Number of Trades: {result.num_trades}")
print(f"Win Rate: {result.win_rate * 100:.1f}%")
```

### With Date Range

```python
from datetime import datetime

config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.001),
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

### Viewing Detailed Metrics

```python
from Classes.Analysis.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics.calculate_metrics(result)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
```

### Saving Trade Logs

```python
from Classes.Analysis.trade_logger import TradeLogger

logger = TradeLogger(Path('logs'))
logger.log_trades(
    symbol='AAPL',
    strategy_name=strategy.get_name(),
    trades=result.trades
)
```

### Generating Excel Reports

```python
from Classes.Analysis.excel_report_generator import ExcelReportGenerator

generator = ExcelReportGenerator(
    output_directory=Path('logs/reports'),
    initial_capital=100000.0
)
report_path = generator.generate_report(result)
print(f"Report saved to: {report_path}")
```

---

## Understanding Results

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Total Return %** | Overall profit/loss | Positive |
| **Win Rate** | % of profitable trades | 40-60% for trend strategies |
| **Profit Factor** | Gross profit / gross loss | > 1.5 |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Sortino Ratio** | Return vs downside risk | > 1.0 |
| **Max Drawdown** | Largest peak-to-trough decline | < 25% |
| **Calmar Ratio** | Return / max drawdown | > 1.0 |
| **Average Trade** | Mean profit per trade | Positive |
| **Average Duration** | Mean trade length | Strategy-dependent |

### Reading the Equity Curve

- **Upward slope** = Making money
- **Flat periods** = No positions or breaking even
- **Dips** = Drawdowns (losing periods)

A good equity curve rises steadily without large drops.

### Trade Log Columns

| Column | Description |
|--------|-------------|
| Entry Date | When the trade opened |
| Entry Price | Purchase price |
| Exit Date | When the trade closed |
| Exit Price | Sale price |
| Exit Reason | Why closed (signal, stop loss, etc.) |
| P/L | Profit/loss in currency |
| P/L % | Profit/loss as percentage |
| Duration | Days held |

---

## Strategy Configuration

### AlphaTrend Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volume_short_ma` | 4 | Short volume MA period |
| `volume_long_ma` | 30 | Long volume MA period |
| `atr_stop_loss_multiple` | 2.5 | Stop loss = entry - (ATR × this) |
| `grace_period_bars` | 14 | Bars to ignore EMA exit after entry |
| `risk_percent` | 2.0 | % of equity to risk per trade |

### Adjusting Parameters in Python

```python
strategy = AlphaTrendStrategy(
    volume_short_ma=4,
    volume_long_ma=30,
    atr_stop_loss_multiple=2.5,
    grace_period_bars=14,
    risk_percent=2.0
)
```

### Using Presets

Load from `config/strategy_presets/`:

```json
{
  "strategy_class": "AlphaTrendStrategy",
  "parameters": {
    "volume_short_ma": 4,
    "volume_long_ma": 30,
    "atr_stop_loss_multiple": 2.5
  }
}
```

---

## Common Workflows

### Workflow 1: Test a Strategy on One Security

1. Start GUI: `python run_gui.py`
2. Select single security mode
3. Choose security (e.g., AAPL)
4. Select AlphaTrendStrategy
5. Set commission to 0.1% (0.001)
6. Run backtest
7. Review results and trade logs

### Workflow 2: Compare Parameters

```python
parameters_to_test = [
    {'atr_stop_loss_multiple': 2.0},
    {'atr_stop_loss_multiple': 2.5},
    {'atr_stop_loss_multiple': 3.0},
]

for params in parameters_to_test:
    strategy = AlphaTrendStrategy(**params)
    result = engine.run('AAPL', data, strategy)
    print(f"ATR Multiple: {params['atr_stop_loss_multiple']}")
    print(f"  Return: {result.total_return_pct:.2f}%")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
```

### Workflow 3: Test Across Multiple Securities

```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

for symbol in symbols:
    data = loader.load_csv(symbol)
    result = engine.run(symbol, data, strategy)
    print(f"{symbol}: {result.total_return_pct:.2f}% return")
```

---

## Best Practices

### 1. Use Enough Data
- At least 2-3 years for reliable results
- More data = more confidence
- But very old data may not reflect current markets

### 2. Account for Costs
- Set realistic commission (check your broker)
- Real trading has slippage (not simulated)
- The framework uses no slippage by default

### 3. Avoid Overfitting
- If results look too good, be skeptical
- Use walk-forward optimization
- Test on out-of-sample data

### 4. Check Drawdowns
- Maximum drawdown shows worst-case scenario
- Ask: "Could I handle this psychologically?"
- Consider position sizing to reduce drawdown

### 5. Review Individual Trades
- Don't just look at totals
- Check if exit reasons make sense
- Look for patterns in winners and losers

---

## Troubleshooting

### "Required columns missing"

Your CSV doesn't have columns the strategy needs:
1. Check `strategy.required_columns()` for requirements
2. Verify your CSV has those columns
3. Column names are case-insensitive

### "No trades generated"

Strategy never triggered a buy signal:
- Conditions may be too strict
- Data may not contain the patterns needed
- Date range may be too short
- Try relaxing parameters temporarily

### Results don't match TradingView

Small differences are normal due to:
- Indicator calculation differences
- Rounding differences
- Commission handling

For closer matching:
- Export indicator values from TradingView directly
- Verify commission settings match

### Slow backtest

- Use `prepare_data()` for pre-calculations
- Consider Numba JIT for indicator calculations
- Reduce the number of securities in portfolio mode

---

## Output Locations

```
logs/
├── backtests/
│   ├── single_security/
│   │   └── {backtest_name}/
│   │       ├── {strategy}_{symbol}_trades.csv
│   │       ├── {strategy}_{symbol}_parameters.json
│   │       └── reports/
│   │           └── {strategy}_{symbol}_report.xlsx
│   └── portfolio/
│       └── {backtest_name}/
│           ├── trades/
│           │   └── {symbol}_trades.csv
│           └── reports/
```
