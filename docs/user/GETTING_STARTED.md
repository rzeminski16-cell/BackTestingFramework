# Getting Started

This guide will help you install the Backtesting Framework, run your first backtest, and understand the key concepts.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/BackTestingFramework.git
cd BackTestingFramework
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas`, `numpy` - Data handling
- `openpyxl` - Excel report generation
- `scikit-optimize` - Parameter optimization
- `tkinter` - GUI (usually included with Python)

**Linux users:** You may need to install tkinter separately:
```bash
sudo apt-get install python3-tk
```

### Step 3: Verify Installation

```bash
python ctk_main_gui.py
```

If the main launcher GUI opens, you're ready to go. The launcher provides access to all framework features:
- **Backtesting** - Run backtests with configurable strategies
- **Optimization** - Walk-forward parameter optimization
- **Edge Analysis** - E-ratio and R-multiple analysis from trade logs
- **Factor Analysis** - Multi-factor performance analysis
- **Vulnerability Modeler** - Vulnerability score optimization

---

## Key Concepts

Before running your first backtest, understand these fundamental concepts:

### Backtest
A simulation that applies a trading strategy to historical data and records what trades would have occurred. This helps evaluate whether a strategy is worth using before risking real money.

### Strategy
The trading rules that decide when to buy and sell. Strategies analyze price data and indicators to generate trading signals.

### Signal
A decision from the strategy:
- **BUY** - Open a new position
- **SELL** - Close the current position
- **HOLD** - Do nothing
- **PARTIAL_EXIT** - Close part of the position
- **ADJUST_STOP** - Move the stop loss

### Position
An open trade. When you buy, you have a position. When you sell, the position closes and becomes a completed trade.

### Trade
A completed transaction with entry date/price and exit date/price. The framework calculates profit/loss for each trade.

### Equity Curve
A chart showing how your portfolio value changes over time during the backtest.

---

## Your First Backtest (GUI)

The GUI is the easiest way to get started.

### Step 1: Launch the GUI

```bash
python ctk_main_gui.py
```

Click on the **Backtesting** card to open the backtest wizard. Alternatively, run directly:

```bash
python ctk_backtest_gui.py
```

### Step 2: Select Mode and Securities

- **Single Security Mode**: Test on one security at a time
- **Portfolio Mode**: Test across multiple securities with shared capital

Select one or more securities from the list. These come from CSV files in your `raw_data/` folder.

### Step 3: Choose a Strategy

Select a strategy from the dropdown:
- **AlphaTrendStrategy** - Production-ready trend-following strategy
- **RandomBaseStrategy** - Baseline random strategy for comparison (E-ratio should hover around 1.0)

Click "Configure Strategy Parameters" to adjust settings like stop loss multipliers and risk percentage.

### Step 4: Set Commission and Capital

- **Commission Mode**:
  - *Percentage*: A percentage of trade value (e.g., 0.1% = 0.001)
  - *Fixed*: A fixed amount per trade (e.g., $3)
- **Initial Capital**: Starting money for the backtest (e.g., $100,000)

### Step 5: Run the Backtest

1. Give your backtest a name
2. Click "Run Backtest"
3. View results in the results window

Results are saved to `logs/backtests/{backtest_name}/`.

---

## Your First Backtest (Python)

For more control, use the framework programmatically:

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

---

## Understanding Results

### Key Metrics

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Total Return %** | Overall profit/loss as percentage | Positive |
| **Win Rate** | Percentage of profitable trades | 40-60% for trend strategies |
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

---

## Data Requirements

### CSV File Format

Place your data files in the `raw_data/` folder. Each file should be named by symbol (e.g., `AAPL.csv`).

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

### Example CSV Format

```csv
date,open,high,low,close,volume,atr_14,ema_50
2020-01-02,300.00,301.50,299.00,301.00,50000000,5.23,295.50
2020-01-03,301.00,303.00,300.50,302.50,48000000,5.18,296.10
```

### Getting Data

Use the Data Collection GUI to fetch data from Alpha Vantage:

```bash
python data_collection_gui.py
```

See the [Data Collection Guide](DATA_COLLECTION.md) for details.

---

## How the Framework Works

### Execution Model

1. **Load Data** - The framework reads your CSV file
2. **Process Bar by Bar** - For each day in the data:
   - The strategy sees only data up to that point (no future data)
   - The strategy generates a signal (BUY, SELL, or HOLD)
   - Trades execute at the closing price
   - Stop losses and take profits are checked
3. **Record Results** - All trades are logged
4. **Calculate Metrics** - Performance metrics are computed

### Key Principles

- **No lookahead bias** - Strategies only see past data, never future prices
- **Trades execute at closing prices** - Matches TradingView's default behavior
- **One position at a time** - In single-security mode, you're either in a trade or out
- **Long only** - The framework supports buying and selling, not short selling

---

## Included Strategy: AlphaTrend

The framework includes a production-ready strategy called **AlphaTrend**.

### What It Does

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
- Real-world execution has slippage (not simulated)

### 3. Watch for Overfitting
- If results look too good, they probably are
- Use walk-forward optimization to validate
- Test on out-of-sample data

### 4. Check Drawdowns
- Maximum drawdown shows worst-case scenario
- Ask: "Could I psychologically handle this drawdown?"

### 5. Look at Individual Trades
- Don't just look at totals - review individual trades
- Check if exit reasons make sense
- Look for patterns in winners and losers

---

## Troubleshooting

### "Required columns missing"

Your CSV file doesn't have columns the strategy needs:
1. Check what columns your strategy requires
2. Verify your CSV has those columns
3. Column names are case-insensitive

### "No trades generated"

The strategy never triggered a buy signal:
- Strategy conditions may be too strict
- Data may not contain the patterns the strategy looks for
- Date range may be too short

### GUI doesn't start

On Linux, install tkinter:
```bash
sudo apt-get install python3-tk
```

---

## Next Steps

- **[Tools & Applications](TOOLS.md)** - Explore all available tools
- **[Data Collection](DATA_COLLECTION.md)** - Gather market data
- **[Running Backtests](BACKTESTING.md)** - Advanced backtesting options
- **[Optimization](OPTIMIZATION.md)** - Find optimal parameters
