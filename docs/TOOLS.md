# Tools & Executables Guide

This guide documents all executable scripts and applications available in the framework.

## Main Applications

These are the primary tools you'll use for backtesting and analysis.

### Backtest GUI

**File:** `run_gui.py` (or `backtest_gui.py`)

**Purpose:** Main graphical interface for running backtests.

**How to Run:**
```bash
python run_gui.py
```

**Features:**
- Select single security or portfolio mode
- Choose and configure strategies
- Set commission and capital
- Run backtests and view results
- Automatic trade logging and report generation

**When to Use:** This is your primary tool for running backtests interactively.

---

### Optimization GUI

**File:** `optimize_gui.py` (or `optimization_gui.py`)

**Purpose:** Walk-forward optimization with Bayesian parameter search.

**How to Run:**
```bash
python optimize_gui.py
```

**Features:**
- Select strategy and securities to optimize
- Choose speed mode (Quick/Fast/Full)
- Enable sensitivity analysis
- View real-time optimization progress
- Generate Excel reports with recommendations

**When to Use:** When you want to find optimal strategy parameters that won't overfit.

---

### AlphaTrend Explorer

**File:** `alphatrend_explorer.py`

**Purpose:** Interactive visualization tool for understanding the AlphaTrend indicator.

**How to Run:**
```bash
streamlit run alphatrend_explorer.py
```

**Requirements:** Requires Streamlit (`pip install streamlit plotly`)

**Features:**
- Interactive parameter adjustment with real-time chart updates
- Component breakdown (ATR bands, MFI, signal generation)
- Side-by-side comparison of different parameter sets
- Synthetic or real data support
- Parameter sensitivity heatmaps
- Educational explanations of each indicator component

**When to Use:** When you want to understand how AlphaTrend works or experiment with parameters visually before backtesting.

---

### Vulnerability Score Modeler

**File:** `vulnerability_gui.py`

**Purpose:** Analyze and optimize vulnerability scoring for portfolio capital contention.

**How to Run:**
```bash
python vulnerability_gui.py
```

**Features:**
- Load completed backtest results
- Configure vulnerability score parameters
- Simulate position swaps and measure P/L impact
- Visualize trade lifecycles with vulnerability overlays
- Test different parameter presets (aggressive, conservative, momentum-focused)
- Export/import configurations
- Sensitivity analysis and Monte Carlo simulation

**When to Use:** When using portfolio mode with vulnerability scoring and you want to tune the scoring parameters to improve capital allocation.

---

## Utility Scripts

These scripts perform specific tasks and are run from the command line.

### Generate Excel Report

**File:** `generate_excel_report.py`

**Purpose:** Generate a standalone Excel report from a backtest.

**How to Run:**
```bash
python generate_excel_report.py
```

**What It Does:**
- Runs a sample backtest on AAPL
- Generates a comprehensive Excel report with:
  - Summary dashboard
  - Trade log
  - Performance analysis
  - Charts and visualizations

**When to Use:** As a template for generating Excel reports programmatically, or to quickly generate a sample report.

---

### Compare Trade Logs

**File:** `compare_trade_logs.py`

**Purpose:** Compare trade logs from the framework against TradingView exports.

**How to Run:**
```bash
python compare_trade_logs.py \
  --framework logs/backtest/trades.csv \
  --tradingview logs/tradingview_export.csv
```

**What It Does:**
- Finds exact matches (same entry date)
- Finds approximate matches (entry dates within N days)
- Calculates differences in prices and P/L
- Generates comparison report

**When to Use:** When validating that your backtest results match TradingView's results.

---

### Export AlphaTrend Indicators

**File:** `export_alphatrend_indicators.py`

**Purpose:** Export indicator values from your data to CSV for analysis.

**How to Run:**
```bash
python export_alphatrend_indicators.py --symbol AAPL
```

**What It Does:**
- Reads pre-calculated indicators from raw data
- Exports all indicator values to a new CSV file

**When to Use:** When you need to analyze indicator values separately or compare them with external sources.

---

### Process Chart Exports

**File:** `process_chart_exports.py`

**Purpose:** Process TradingView chart exports to prepare data for backtesting.

**How to Run:**
```bash
python process_chart_exports.py
```

**What It Does:**
- Standardizes column names and schema
- Fills in Candle Position values (works backwards from non-null values)
- Prepares data for use with the backtesting framework

**When to Use:** When importing new data exported from TradingView.

---

### Benchmark Performance

**File:** `benchmark_performance.py`

**Purpose:** Measure backtest execution performance.

**How to Run:**
```bash
python benchmark_performance.py
```

**What It Does:**
- Runs a backtest and measures execution time
- Reports timing for data loading, preparation, and execution
- Useful for performance tuning

**When to Use:** When you want to verify performance optimizations or benchmark on your hardware.

---

### Compare AlphaTrend Trades

**File:** `compare_alphatrend_trades.py`

**Purpose:** Compare AlphaTrend strategy trades between different sources.

**How to Run:**
```bash
python compare_alphatrend_trades.py
```

**What It Does:**
- Loads trades from framework and comparison source
- Identifies matching and differing trades
- Analyzes discrepancies in entry/exit timing and prices

**When to Use:** For detailed validation of AlphaTrend strategy execution.

---

### Backtest CLI

**File:** `backtest.py`

**Purpose:** Command-line examples for programmatic backtesting.

**How to Run:**
```bash
python backtest.py
```

**Note:** Most examples in this file have been deprecated. Use `run_gui.py` or write your own Python scripts using the framework's API.

**When to Use:** As a reference for programmatic usage patterns.

---

## Quick Reference

| Tool | Command | Purpose |
|------|---------|---------|
| **Backtest GUI** | `python run_gui.py` | Run backtests interactively |
| **Optimization GUI** | `python optimize_gui.py` | Find optimal parameters |
| **AlphaTrend Explorer** | `streamlit run alphatrend_explorer.py` | Visualize AlphaTrend indicator |
| **Vulnerability Modeler** | `python vulnerability_gui.py` | Tune vulnerability scoring |
| **Excel Report** | `python generate_excel_report.py` | Generate sample report |
| **Compare Trades** | `python compare_trade_logs.py` | Validate against TradingView |
| **Export Indicators** | `python export_alphatrend_indicators.py` | Export indicator values |
| **Process Exports** | `python process_chart_exports.py` | Prepare TradingView data |
| **Benchmark** | `python benchmark_performance.py` | Measure performance |

---

## Dependencies by Tool

| Tool | Extra Dependencies |
|------|-------------------|
| Backtest GUI | tkinter (usually included with Python) |
| Optimization GUI | tkinter, scikit-optimize |
| AlphaTrend Explorer | streamlit, plotly |
| Vulnerability Modeler | tkinter, matplotlib |
| All others | Standard framework dependencies |

**Install optional dependencies:**
```bash
# For AlphaTrend Explorer
pip install streamlit plotly

# For Vulnerability Modeler visualizations
pip install matplotlib
```

---

## Common Workflows

### Workflow 1: Quick Backtest
```bash
python run_gui.py
# → Select security → Choose strategy → Run
```

### Workflow 2: Optimize Then Backtest
```bash
# 1. Find optimal parameters
python optimize_gui.py
# → Review Excel report → Note recommended parameters

# 2. Run backtest with those parameters
python run_gui.py
# → Configure strategy with recommended parameters → Run
```

### Workflow 3: Understand AlphaTrend Before Using
```bash
# 1. Explore the indicator
streamlit run alphatrend_explorer.py
# → Adjust parameters → See how signals change

# 2. Run backtest with chosen parameters
python run_gui.py
```

### Workflow 4: Portfolio with Vulnerability Tuning
```bash
# 1. Run initial portfolio backtest
python run_gui.py
# → Portfolio mode → Multiple securities → Run

# 2. Tune vulnerability scoring
python vulnerability_gui.py
# → Load results → Adjust parameters → Simulate impact

# 3. Re-run with tuned parameters
python run_gui.py
# → Use tuned vulnerability preset
```

### Workflow 5: Validate Against TradingView
```bash
# 1. Export data from TradingView
# 2. Process the export
python process_chart_exports.py

# 3. Run backtest
python run_gui.py

# 4. Export TradingView trade list
# 5. Compare
python compare_trade_logs.py --framework logs/trades.csv --tradingview tv_trades.csv
```

---

## Related Guides

- **[User Guide](USER_GUIDE.md)** - Getting started with the framework
- **[Strategies Guide](STRATEGIES.md)** - Creating and configuring strategies
- **[Optimization Guide](OPTIMIZATION.md)** - Parameter optimization details
- **[Portfolio Mode Guide](PORTFOLIO_MODE.md)** - Multi-security backtesting
- **[Configuration Guide](CONFIGURATION.md)** - All configuration options
