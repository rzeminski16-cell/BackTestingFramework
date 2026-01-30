# Quick Start Guide

Get from zero to your first backtest in under 10 minutes.

---

## Prerequisites

Before starting, ensure you have:

1. **Python 3.9+** installed
2. **Required packages** installed (see Installation below)
3. **Historical data** in the `raw_data/daily/` folder

---

## Installation

### 1. Clone or Download the Framework

If you haven't already, get the framework onto your machine.

### 2. Install Dependencies

From the framework root directory:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- pandas, numpy (data handling)
- customtkinter (GUI)
- openpyxl (Excel reports)
- scipy, scikit-learn (optimization)

### 3. Verify Installation

Launch the main menu to verify everything works:

```bash
python ctk_main_gui.py
```

You should see the launcher with application cards.

---

## Your First Backtest

### Option A: Using the GUI (Recommended)

**Step 1: Launch the Backtest GUI**

```bash
python ctk_backtest_gui.py
```

Or click "Backtesting" from the main launcher.

**Step 2: Select Mode**

Choose **Single Security** for your first test.

**Step 3: Select a Security**

Pick a ticker from the available list (data must exist in `raw_data/daily/`).

**Step 4: Select Strategy**

Choose **AlphaTrendStrategy** (the included production strategy).

**Step 5: Configure Settings**

Use the defaults for your first run:
- Initial Capital: 100,000
- Commission: 0.1% (percentage mode)

**Step 6: Run Backtest**

Click "Run Backtest" and wait for completion.

**Step 7: Review Results**

The GUI displays:
- Key metrics (Total Return, Sharpe Ratio, Win Rate, etc.)
- Equity curve chart
- Trade log with entry/exit details

Results are saved to `logs/backtests/single_security/`.

---

### Option B: Using Python Directly

For those who prefer code:

```python
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Config.config import BacktestConfig, CommissionConfig
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Configure the backtest
config = BacktestConfig(
    symbol="AAPL",
    initial_capital=100000,
    commission=CommissionConfig(mode="percentage", value=0.001),
    data_path="raw_data/daily/AAPL_daily.csv"
)

# Create strategy instance
strategy = AlphaTrendStrategy()

# Run backtest
engine = SingleSecurityEngine(config, strategy)
result = engine.run()

# View results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Win Rate: {result.win_rate:.1f}%")
print(f"Total Trades: {result.total_trades}")
```

---

## Understanding the Results

### Key Metrics to Check

| Metric | What It Tells You | Good Range |
|--------|-------------------|------------|
| **Total Return %** | Overall profit/loss | Positive, beats benchmark |
| **Sharpe Ratio** | Risk-adjusted return | Above 1.0 |
| **Max Drawdown %** | Worst peak-to-trough drop | Below 25% |
| **Win Rate** | Percentage of profitable trades | 40-60% for trend strategies |
| **Profit Factor** | Gross profit / Gross loss | Above 1.5 |

### Trade Log

Each trade shows:
- Entry/exit dates and prices
- Quantity and direction
- Profit/loss in dollars and percentage
- Exit reason (signal, stop loss, time exit)

### Equity Curve

The chart shows your capital over time. Look for:
- Steady upward trend (good)
- Deep drawdowns (concerning)
- Flat periods (no trades or break-even)

---

## Common First Issues

### "No data found for symbol"

**Cause**: Data file doesn't exist or wrong path.

**Fix**: Check that `raw_data/daily/{SYMBOL}_daily.csv` exists.

### "Required columns missing"

**Cause**: Data file missing columns the strategy needs.

**Fix**: AlphaTrend requires: `date, open, high, low, close, volume, atr_14, ema_50, mfi_14`

### "No trades generated"

**Cause**: Strategy conditions never triggered in the date range.

**Fix**:
- Expand date range
- Try a different security
- Check strategy parameters aren't too restrictive

### GUI doesn't launch

**Cause**: Missing tkinter or customtkinter.

**Fix**:
```bash
pip install customtkinter --upgrade
# On Linux: sudo apt-get install python3-tk
```

---

## Next Steps

After your first successful backtest:

1. **Try Portfolio Mode** — Test across multiple securities with shared capital
   - See [Portfolio Mode](../concepts/PORTFOLIO_MODE.md)

2. **Optimize Parameters** — Find better parameter values
   - See [Optimization](../concepts/OPTIMIZATION.md)

3. **Analyze Edge** — Validate your entry quality with E-ratio
   - See [Edge Analysis](../concepts/EDGE_ANALYSIS.md)

4. **Create Your Own Strategy** — Build a custom trading strategy
   - See [Strategy Development](../strategy-development/STRATEGY_GUIDE.md)

---

## Quick Reference: Launch Commands

| Application | Command |
|-------------|---------|
| Main Launcher | `python ctk_main_gui.py` |
| Backtest GUI | `python ctk_backtest_gui.py` |
| Univariate Optimization | `python ctk_univariate_optimization_gui.py` |
| Walk-Forward Optimization | `python ctk_optimization_gui.py` |
| Edge Analysis | `python ctk_edge_analysis_gui.py` |
| Vulnerability Modeler | `python ctk_vulnerability_gui.py` |
| Factor Analysis | `python ctk_factor_analysis_gui.py` |

---

## File Locations

| What | Where |
|------|-------|
| Price data | `raw_data/daily/{SYMBOL}_daily.csv` |
| Backtest results | `logs/backtests/single_security/` |
| Portfolio results | `logs/backtests/portfolio/` |
| Optimization reports | `logs/optimization_reports/` |
| Strategy files | `strategies/` |
| Configuration | `config/` |
