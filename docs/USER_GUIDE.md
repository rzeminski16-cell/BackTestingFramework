# User Guide

This guide explains what the Backtesting Framework does, when to use it, and how to use it effectively.

## What Is This Framework?

The Backtesting Framework is a tool for testing trading strategies on historical data. You give it:
- Historical price data (CSV files)
- A trading strategy (entry/exit rules)
- Configuration (capital, commission, etc.)

It simulates running that strategy on past data and tells you:
- How many trades would have been made
- What the profit/loss would have been
- Performance metrics like Sharpe ratio, max drawdown, win rate

This helps you evaluate whether a strategy is worth using before risking real money.

## When Should You Use It?

Use this framework when you want to:

1. **Test a trading idea** - "Would buying when price crosses above the 50-day moving average have been profitable on AAPL?"

2. **Compare strategies** - "Which performs better: my momentum strategy or my mean-reversion strategy?"

3. **Find optimal parameters** - "What's the best moving average period for this strategy? 20? 50? 100?"

4. **Test across multiple securities** - "Does this strategy work on tech stocks in general, or just AAPL?"

5. **Generate performance reports** - Create professional Excel reports with metrics for analysis or sharing.

## Key Concepts

### Backtest
A simulation that applies a strategy to historical data and records what trades would have occurred.

### Strategy
The trading rules that decide when to buy and sell. Strategies look at price data and indicators to generate signals.

### Signal
A decision from the strategy:
- **BUY** - Open a new position
- **SELL** - Close the current position
- **HOLD** - Do nothing

### Position
An open trade. When you buy, you have a position. When you sell, the position closes and becomes a completed trade.

### Trade
A completed transaction with entry date/price and exit date/price. The framework calculates profit/loss for each trade.

### Equity Curve
A chart showing how your portfolio value changes over time during the backtest.

## How It Works

1. **Load Data** - The framework reads your CSV file containing historical prices
2. **Process Bar by Bar** - For each day/bar in the data:
   - The strategy sees only data up to that point (no future data)
   - The strategy generates a signal (BUY, SELL, or HOLD)
   - Trades execute at the closing price
   - Stop losses and take profits are checked
3. **Record Results** - All trades are logged with entry/exit details
4. **Calculate Metrics** - Performance metrics are computed from the trade history

### Execution Model

The framework matches TradingView's default behavior:
- **Trades execute at closing prices** - When a signal occurs, the trade happens at that bar's close
- **No lookahead bias** - Strategies only see past data, never future prices
- **One position at a time** - In single-security mode, you're either in a trade or out
- **Long only** - The framework supports buying and selling, not short selling

---

## Using the GUI

The GUI is the easiest way to run backtests, especially when starting out.

### Starting the GUI

```bash
python run_gui.py
```

### Step 1: Select Mode and Securities

- **Single Security Mode**: Test on one security at a time. Each gets isolated capital.
- **Portfolio Mode**: Test across multiple securities with shared capital.

Select one or more securities from the list. These come from CSV files in your `raw_data/` folder.

### Step 2: Choose a Strategy

Select a strategy from the dropdown. The available strategies are discovered from the `strategies/` folder.

**Configure Parameters**: Click "Configure Strategy Parameters" to adjust:
- Volume filter settings
- Stop loss multipliers
- Grace periods
- Risk percentage

Each parameter has a description explaining what it does.

### Step 3: Set Commission and Capital

- **Commission Mode**:
  - *Percentage*: A percentage of trade value (e.g., 0.1% = 0.001)
  - *Fixed*: A fixed amount per trade (e.g., £3)
- **Initial Capital**: Starting money for the backtest (e.g., £100,000)

### Step 4: Set Date Range (Optional)

Leave blank to use all available data, or set specific start/end dates.

### Step 5: Review and Run

- Give your backtest a name (it will be prefixed with the strategy name)
- Review your settings
- Click "Run Backtest"

### Viewing Results

After the backtest completes:
- Results appear in a window showing key metrics
- Trade logs are saved to `logs/backtests/{backtest_name}/`
- Excel reports (if enabled) are saved alongside the logs

---

## Using Python Code

For more control or automation, use the framework programmatically.

### Basic Backtest

```python
from pathlib import Path
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from strategies.alphatrend_strategy import AlphaTrendStrategy

# 1. Configure the backtest
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001  # 0.1%
    )
)

# 2. Load your data
loader = DataLoader(Path('raw_data'))
data = loader.load_csv('AAPL')

# 3. Create a strategy
strategy = AlphaTrendStrategy()

# 4. Run the backtest
engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)

# 5. View results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Number of Trades: {result.num_trades}")
print(f"Win Rate: {result.win_rate * 100:.1f}%")
```

### Viewing Detailed Metrics

```python
from Classes.Analysis.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics.calculate_metrics(result)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
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

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Total Return %** | Overall profit/loss as percentage | Positive |
| **Win Rate** | Percentage of trades that were profitable | 40-60% for trend strategies |
| **Profit Factor** | Gross profit / gross loss | > 1.5 |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Max Drawdown** | Largest peak-to-trough decline | < 20-25% |
| **Average Trade** | Mean profit/loss per trade | Positive |

### Reading the Equity Curve

The equity curve shows portfolio value over time:
- **Upward slope** = Making money
- **Flat periods** = No open positions or breaking even
- **Dips** = Drawdowns (losing periods)

A good equity curve rises steadily without large drops.

### Trade Log Columns

| Column | Description |
|--------|-------------|
| Entry Date | When the trade was opened |
| Entry Price | Price at which you bought |
| Exit Date | When the trade was closed |
| Exit Price | Price at which you sold |
| Exit Reason | Why the trade closed (signal, stop loss, etc.) |
| P/L | Profit or loss in currency |
| P/L % | Profit or loss as percentage |
| Duration | How many days the trade was held |

---

## Common Workflows

### Workflow 1: Testing a Strategy on One Security

1. Start the GUI: `python run_gui.py`
2. Select single security mode
3. Choose your security (e.g., AAPL)
4. Select AlphaTrendStrategy (or your strategy)
5. Set commission to 0.1% (0.001)
6. Run the backtest
7. Review results in the results window
8. Check trade logs in `logs/backtests/`

### Workflow 2: Testing a Strategy on Multiple Securities

1. Start the GUI: `python run_gui.py`
2. Select portfolio mode
3. Choose multiple securities (Ctrl+click to select)
4. Set position limits (e.g., max 3 positions)
5. Run the backtest
6. Compare performance across securities

### Workflow 3: Finding Optimal Parameters

1. Start the optimization GUI: `python optimize_gui.py`
2. Select your strategy
3. Choose securities to optimize on
4. Pick speed mode (Quick for testing, Full for final results)
5. Enable sensitivity analysis
6. Run optimization
7. Review Excel report in `logs/optimization_reports/`
8. Use the recommended parameters

### Workflow 4: Comparing Two Strategies

1. Run a backtest with Strategy A, note the results
2. Run a backtest with Strategy B on the same data
3. Compare metrics side-by-side
4. Look at Sharpe ratio, max drawdown, and consistency

---

## Data Preparation

### CSV File Format

Your CSV files should have these columns:

```csv
date,open,high,low,close,volume,atr_14,ema_50
2020-01-02,300.00,301.50,299.00,301.00,50000000,5.23,295.50
2020-01-03,301.00,303.00,300.50,302.50,48000000,5.18,296.10
...
```

**Required columns:**
- `date` (or `time`) - Timestamp
- `close` - Closing price

**Recommended columns:**
- `open`, `high`, `low` - OHLC data
- `volume` - Trading volume

**Strategy-dependent columns:**
- `atr_14` - 14-period Average True Range
- `ema_50` - 50-period Exponential Moving Average
- Other indicators your strategy needs

### Pre-calculating Indicators

The framework expects indicators to be pre-calculated in your CSV files. This provides:
- Faster backtesting (no recalculation each time)
- Exact indicator values (match your data source)
- Flexibility to use any indicators

You can calculate indicators in Python with pandas/TA-Lib, or export them from TradingView.

### Column Name Handling

Column names are automatically normalized:
- Converted to lowercase
- `time` is renamed to `date`
- Leading/trailing spaces are trimmed

---

## Included Strategy: AlphaTrend

The framework includes a production-ready strategy called **AlphaTrend**.

### What It Does

AlphaTrend is a trend-following strategy that:
1. **Enters** when the AlphaTrend indicator signals an uptrend AND volume confirms
2. **Exits** when price falls below the 50-day EMA (with grace period and momentum protection)
3. Uses ATR-based stop losses for risk management

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volume_short_ma` | 4 | Short-term volume MA |
| `volume_long_ma` | 30 | Long-term volume MA |
| `atr_stop_loss_multiple` | 2.5 | Stop loss distance in ATR |
| `grace_period_bars` | 14 | Bars to ignore EMA exit after entry |
| `risk_percent` | 2.0 | Percentage of equity to risk per trade |

### Required Data Columns

AlphaTrend needs these columns in your CSV:
- `date`, `open`, `high`, `low`, `close`, `volume`
- `atr_14` - 14-period ATR
- `ema_50` - 50-period EMA

---

## Output Locations

The framework saves outputs to organized folders:

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
└── optimization_reports/
    └── {optimization_name}/
        └── {strategy}_{symbol}_optimization.xlsx
```

---

## Tips for Better Backtests

### 1. Use Enough Data
- At least 2-3 years for reliable results
- More data = more confidence in results
- But very old data may not reflect current markets

### 2. Account for Costs
- Set realistic commission (check your broker)
- Consider that real-world execution has slippage
- The framework uses no slippage by default

### 3. Watch for Overfitting
- If results look too good, they probably are
- Use walk-forward optimization to validate
- Test on out-of-sample data

### 4. Check Drawdowns
- Maximum drawdown shows worst-case scenario
- Ask: "Could I psychologically handle this drawdown?"
- Consider position sizing to reduce drawdown

### 5. Look at Individual Trades
- Don't just look at totals - review individual trades
- Check if exit reasons make sense
- Look for patterns in winners and losers

---

## Troubleshooting

### "Required columns missing"

Your CSV file doesn't have columns the strategy needs. Check:
1. What columns your strategy requires (`required_columns()` method)
2. What columns your CSV file has
3. That column names match (case-insensitive)

### "No trades generated"

The strategy never triggered a buy signal. Possible causes:
- Strategy conditions too strict
- Data doesn't contain the patterns the strategy looks for
- Date range too short

### "Results don't match TradingView"

Small differences are normal due to:
- Indicator calculation differences
- Rounding differences
- Commission handling

For closer matching:
- Export indicator values from TradingView directly
- Verify commission settings match
- Compare bar-by-bar if needed

### GUI doesn't start

On Linux, you may need to install tkinter:
```bash
sudo apt-get install python3-tk
```

---

## Next Steps

- **[Strategies Guide](STRATEGIES.md)** - Learn how to create your own strategies
- **[Configuration Guide](CONFIGURATION.md)** - All configuration options explained
- **[Optimization Guide](OPTIMIZATION.md)** - Find optimal strategy parameters
- **[Portfolio Mode Guide](PORTFOLIO_MODE.md)** - Multi-security backtesting
