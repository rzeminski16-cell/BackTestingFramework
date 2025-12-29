# Tools & Applications Guide

This guide documents all executable scripts and applications available in the framework.

## Main Applications

### Backtest GUI

**File:** `run_gui.py`

**Purpose:** Main graphical interface for running backtests.

```bash
python run_gui.py
```

**Features:**
- Single security or portfolio mode selection
- Strategy selection and configuration
- Commission and capital settings
- Real-time backtest execution
- Automatic trade logging and report generation

**When to Use:** Your primary tool for running backtests interactively.

---

### Optimization GUI

**File:** `optimize_gui.py`

**Purpose:** Walk-forward optimization with Bayesian parameter search.

```bash
python optimize_gui.py
```

**Features:**
- Strategy and securities selection
- Speed modes: Quick (25 iterations), Fast (50), Full (100)
- Sensitivity analysis
- Real-time optimization progress
- Excel reports with recommendations

**When to Use:** When finding optimal strategy parameters that won't overfit.

---

### Data Collection GUI

**File:** `data_collection_gui.py`

**Purpose:** Collect market data from Alpha Vantage API.

```bash
python data_collection_gui.py
```

**Features:**
- Daily and weekly OHLCV data
- 50+ technical indicators
- Fundamental data (income statements, balance sheets)
- Insider trading activity
- Forex currency pairs
- Options data

**When to Use:** When you need to gather historical market data.

---

### AlphaTrend Explorer

**File:** `alphatrend_explorer.py`

**Purpose:** Interactive visualization tool for the AlphaTrend indicator.

```bash
streamlit run alphatrend_explorer.py
```

**Requires:** `pip install streamlit plotly`

**Features:**
- Interactive parameter adjustment
- Real-time chart updates
- Component breakdown (ATR bands, MFI, signal generation)
- Side-by-side parameter comparison
- Parameter sensitivity heatmaps

**When to Use:** When understanding how AlphaTrend works or experimenting with parameters visually.

---

### Vulnerability Score Modeler

**File:** `vulnerability_gui.py`

**Purpose:** Analyze and optimize vulnerability scoring for portfolio capital allocation.

```bash
python vulnerability_gui.py
```

**Features:**
- Load completed backtest results
- Configure vulnerability score parameters
- Simulate position swaps and measure P/L impact
- Visualize trade lifecycles
- Test different parameter presets
- Export/import configurations

**When to Use:** When using portfolio mode and tuning vulnerability scoring to improve capital allocation.

---

## Analysis Tools

### Backtest Analysis

**File:** `run_backtest_analysis.py`

**Purpose:** Drill down into backtest results to understand strategy performance.

```bash
# Basic usage
python run_backtest_analysis.py logs/AlphaTrendStrategy_Backtest

# With custom thresholds
python run_backtest_analysis.py logs/MyStrategy \
    --gb-threshold 7.5 \
    --min-trades 5
```

**Features:**
- Fundamental features extraction (quarterly classification)
- Technical indicators at each trade
- Trade classification (Good/Bad/Indeterminate)
- Period classification (Calmar-based)
- Weekly indicator data for validation

**When to Use:** When analyzing why a strategy performs well or poorly in different conditions.

---

### Fundamental Data Fetcher

**File:** `run_fundamental_data_fetch.py`

**Purpose:** Fetch financial metrics from Alpha Vantage for backtest analysis.

```bash
# Create config file
python run_fundamental_data_fetch.py --create-config

# Fetch data
python run_fundamental_data_fetch.py logs/AlphaTrendStrategy
```

**Features:**
- Point-in-time historical data
- 20+ fundamental metrics per quarter
- EPS, revenue, margins, valuation ratios
- Cash flow metrics
- Balance sheet metrics

**When to Use:** When correlating strategy performance with fundamental data.

---

## Utility Scripts

### Generate Excel Report

**File:** `generate_excel_report.py`

**Purpose:** Generate a standalone Excel report from a backtest.

```bash
python generate_excel_report.py
```

**When to Use:** As a template for generating Excel reports programmatically.

---

### Compare Trade Logs

**File:** `compare_trade_logs.py`

**Purpose:** Compare trade logs from the framework against TradingView exports.

```bash
python compare_trade_logs.py \
  --framework logs/backtest/trades.csv \
  --tradingview logs/tradingview_export.csv
```

**When to Use:** When validating that your backtest results match TradingView.

---

### Export AlphaTrend Indicators

**File:** `export_alphatrend_indicators.py`

**Purpose:** Export indicator values from your data to CSV.

```bash
python export_alphatrend_indicators.py --symbol AAPL
```

**When to Use:** When analyzing indicator values separately.

---

### Process Chart Exports

**File:** `process_chart_exports.py`

**Purpose:** Process TradingView chart exports for backtesting.

```bash
python process_chart_exports.py
```

**When to Use:** When importing data exported from TradingView.

---

### Benchmark Performance

**File:** `benchmark_performance.py`

**Purpose:** Measure backtest execution performance.

```bash
python benchmark_performance.py
```

**When to Use:** When verifying performance optimizations.

---

## Quick Reference

| Tool | Command | Purpose |
|------|---------|---------|
| **Backtest GUI** | `python run_gui.py` | Run backtests interactively |
| **Optimization GUI** | `python optimize_gui.py` | Find optimal parameters |
| **Data Collection** | `python data_collection_gui.py` | Collect market data |
| **AlphaTrend Explorer** | `streamlit run alphatrend_explorer.py` | Visualize AlphaTrend |
| **Vulnerability Modeler** | `python vulnerability_gui.py` | Tune vulnerability scoring |
| **Backtest Analysis** | `python run_backtest_analysis.py [folder]` | Analyze backtest results |
| **Fundamental Data** | `python run_fundamental_data_fetch.py [folder]` | Fetch financial metrics |

---

## Dependencies by Tool

| Tool | Extra Dependencies |
|------|-------------------|
| Backtest GUI | tkinter (usually included) |
| Optimization GUI | tkinter, scikit-optimize |
| Data Collection GUI | customtkinter, requests |
| AlphaTrend Explorer | streamlit, plotly |
| Vulnerability Modeler | tkinter, matplotlib |

**Install optional dependencies:**
```bash
# For AlphaTrend Explorer
pip install streamlit plotly

# For Data Collection GUI
pip install customtkinter

# For Vulnerability Modeler visualizations
pip install matplotlib
```

---

## Common Workflows

### Quick Backtest
```bash
python run_gui.py
# → Select security → Choose strategy → Run
```

### Optimize Then Backtest
```bash
# 1. Find optimal parameters
python optimize_gui.py
# → Review Excel report → Note recommended parameters

# 2. Run backtest with those parameters
python run_gui.py
# → Configure strategy with recommended parameters → Run
```

### Understand AlphaTrend Before Using
```bash
# 1. Explore the indicator
streamlit run alphatrend_explorer.py
# → Adjust parameters → See how signals change

# 2. Run backtest with chosen parameters
python run_gui.py
```

### Collect Data Then Backtest
```bash
# 1. Collect data
python data_collection_gui.py
# → Select symbols → Choose indicators → Fetch

# 2. Run backtest
python run_gui.py
# → Select new securities → Run
```

### Deep Analysis Workflow
```bash
# 1. Run backtest
python run_gui.py

# 2. Analyze results
python run_backtest_analysis.py logs/my_backtest

# 3. Fetch fundamental data
python run_fundamental_data_fetch.py logs/my_backtest

# 4. Review analysis output
```
