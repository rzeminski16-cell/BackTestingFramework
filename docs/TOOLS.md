# Tools & Executables Guide

This guide documents all executable scripts and applications available in the framework.

## Directory Structure

The codebase is organized into these directories:

```
BackTestingFramework/
├── apps/           # GUI applications
├── tools/          # Command-line utilities
├── scripts/        # Test and utility scripts
└── Classes/        # Core framework code
```

## Main Applications (apps/)

These are the primary tools you'll use for backtesting and analysis.

### Backtest GUI

**File:** `apps/ctk_backtest_gui.py`

**Purpose:** Main graphical interface for running backtests (CustomTkinter-based).

**How to Run:**
```bash
python apps/ctk_backtest_gui.py
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

**File:** `apps/ctk_optimization_gui.py`

**Purpose:** Walk-forward optimization with Bayesian parameter search (CustomTkinter-based).

**How to Run:**
```bash
python apps/ctk_optimization_gui.py
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

**File:** `apps/alphatrend_explorer.py`

**Purpose:** Interactive visualization tool for understanding the AlphaTrend indicator.

**How to Run:**
```bash
streamlit run apps/alphatrend_explorer.py
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

**File:** `apps/ctk_vulnerability_gui.py`

**Purpose:** Analyze and optimize vulnerability scoring for portfolio capital contention (CustomTkinter-based).

**How to Run:**
```bash
python apps/ctk_vulnerability_gui.py
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

### Data Collection GUI

**File:** `apps/data_collection_gui.py`

**Purpose:** Unified interface for collecting market data from Alpha Vantage.

**How to Run:**
```bash
python apps/data_collection_gui.py
```

**Features:**
- Collect daily/weekly price data
- Fetch fundamental data
- Download insider transaction data
- Options data collection
- Forex rates
- Session logging and validation

**When to Use:** When you need to gather or update market data for backtesting.

---

## Command-Line Tools (tools/)

These scripts perform specific tasks and are run from the command line.

### Run Backtest Analysis

**File:** `tools/run_backtest_analysis.py`

**Purpose:** Post-backtest analysis with fundamental data integration.

**How to Run:**
```bash
python tools/run_backtest_analysis.py logs/MyStrategy
python tools/run_backtest_analysis.py logs/MyStrategy -o analysis_output/
python tools/run_backtest_analysis.py logs/MyStrategy --gb-threshold 10.0
```

**What It Does:**
- Analyzes completed backtest results
- Fetches point-in-time fundamental data
- Generates technical features at trade points
- Creates comprehensive analysis reports

**When to Use:** For deep analysis of backtest results with fundamental context.

---

### Generate Excel Report

**File:** `tools/generate_excel_report.py`

**Purpose:** Generate a standalone Excel report from a backtest.

**How to Run:**
```bash
python tools/generate_excel_report.py
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

**File:** `tools/compare_trade_logs.py`

**Purpose:** Compare trade logs from the framework against TradingView exports.

**How to Run:**
```bash
python tools/compare_trade_logs.py \
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

**File:** `tools/export_alphatrend_indicators.py`

**Purpose:** Export indicator values from your data to CSV for analysis.

**How to Run:**
```bash
python tools/export_alphatrend_indicators.py --symbol AAPL
```

**What It Does:**
- Reads pre-calculated indicators from raw data
- Exports all indicator values to a new CSV file

**When to Use:** When you need to analyze indicator values separately or compare them with external sources.

---

### Run Fundamental Data Fetch

**File:** `tools/run_fundamental_data_fetch.py`

**Purpose:** Fetch fundamental data from Alpha Vantage for multiple symbols.

**How to Run:**
```bash
python tools/run_fundamental_data_fetch.py
```

**What It Does:**
- Fetches income statements, balance sheets, cash flow
- Calculates derived metrics (EPS TTM, FCF, etc.)
- Saves quarterly data to CSV files

**When to Use:** When you need fundamental data for analysis or strategy constraints.

---

### Benchmark Performance

**File:** `tools/benchmark_performance.py`

**Purpose:** Measure backtest execution performance.

**How to Run:**
```bash
python tools/benchmark_performance.py
```

**What It Does:**
- Runs a backtest and measures execution time
- Reports timing for data loading, preparation, and execution
- Useful for performance tuning

**When to Use:** When you want to verify performance optimizations or benchmark on your hardware.

---

### Compare AlphaTrend Trades

**File:** `tools/compare_alphatrend_trades.py`

**Purpose:** Compare AlphaTrend strategy trades between different sources.

**How to Run:**
```bash
python tools/compare_alphatrend_trades.py
```

**What It Does:**
- Loads trades from framework and comparison source
- Identifies matching and differing trades
- Analyzes discrepancies in entry/exit timing and prices

**When to Use:** For detailed validation of AlphaTrend strategy execution.

---

## Test Scripts (scripts/)

These scripts are for testing and validation.

### Test ATR Stop Loss

**File:** `scripts/test_atr_stop_loss.py`

**Purpose:** Validate ATR-based stop loss calculations.

---

### Test Slippage

**File:** `scripts/test_slippage.py`

**Purpose:** Test slippage calculation logic.

---

### Test Currency Detection

**File:** `scripts/test_currency_detection.py`

**Purpose:** Validate currency detection for forex pairs.

---

### Test Vulnerability Score

**File:** `scripts/test_vulnerability_score.py`

**Purpose:** Comprehensive testing of vulnerability scoring system.

---

## Quick Reference

| Tool | Command | Purpose |
|------|---------|---------|
| **Backtest GUI** | `python apps/ctk_backtest_gui.py` | Run backtests interactively |
| **Optimization GUI** | `python apps/ctk_optimization_gui.py` | Find optimal parameters |
| **AlphaTrend Explorer** | `streamlit run apps/alphatrend_explorer.py` | Visualize AlphaTrend indicator |
| **Vulnerability Modeler** | `python apps/ctk_vulnerability_gui.py` | Tune vulnerability scoring |
| **Data Collection** | `python apps/data_collection_gui.py` | Collect market data |
| **Backtest Analysis** | `python tools/run_backtest_analysis.py` | Analyze backtest results |
| **Excel Report** | `python tools/generate_excel_report.py` | Generate sample report |
| **Compare Trades** | `python tools/compare_trade_logs.py` | Validate against TradingView |
| **Export Indicators** | `python tools/export_alphatrend_indicators.py` | Export indicator values |
| **Benchmark** | `python tools/benchmark_performance.py` | Measure performance |

---

## Dependencies by Tool

| Tool | Extra Dependencies |
|------|-------------------|
| Backtest GUI | customtkinter |
| Optimization GUI | customtkinter, scikit-optimize |
| AlphaTrend Explorer | streamlit, plotly |
| Vulnerability Modeler | customtkinter, matplotlib |
| Data Collection GUI | customtkinter |
| All others | Standard framework dependencies |

**Install optional dependencies:**
```bash
# For GUI applications
pip install customtkinter

# For AlphaTrend Explorer
pip install streamlit plotly

# For visualizations
pip install matplotlib
```

---

## Common Workflows

### Workflow 1: Quick Backtest
```bash
python apps/ctk_backtest_gui.py
# → Select security → Choose strategy → Run
```

### Workflow 2: Optimize Then Backtest
```bash
# 1. Find optimal parameters
python apps/ctk_optimization_gui.py
# → Review Excel report → Note recommended parameters

# 2. Run backtest with those parameters
python apps/ctk_backtest_gui.py
# → Configure strategy with recommended parameters → Run
```

### Workflow 3: Understand AlphaTrend Before Using
```bash
# 1. Explore the indicator
streamlit run apps/alphatrend_explorer.py
# → Adjust parameters → See how signals change

# 2. Run backtest with chosen parameters
python apps/ctk_backtest_gui.py
```

### Workflow 4: Portfolio with Vulnerability Tuning
```bash
# 1. Run initial portfolio backtest
python apps/ctk_backtest_gui.py
# → Portfolio mode → Multiple securities → Run

# 2. Tune vulnerability scoring
python apps/ctk_vulnerability_gui.py
# → Load results → Adjust parameters → Simulate impact

# 3. Re-run with tuned parameters
python apps/ctk_backtest_gui.py
# → Use tuned vulnerability preset
```

### Workflow 5: Validate Against TradingView
```bash
# 1. Run backtest
python apps/ctk_backtest_gui.py

# 2. Export TradingView trade list
# 3. Compare
python tools/compare_trade_logs.py --framework logs/trades.csv --tradingview tv_trades.csv
```

---

## Related Guides

- **[User Guide](USER_GUIDE.md)** - Getting started with the framework
- **[Strategies Guide](STRATEGIES.md)** - Creating and configuring strategies
- **[Optimization Guide](OPTIMIZATION.md)** - Parameter optimization details
- **[Portfolio Mode Guide](PORTFOLIO_MODE.md)** - Multi-security backtesting
- **[Configuration Guide](CONFIGURATION.md)** - All configuration options
- **[Data Collection Guide](DATA_COLLECTION_GUIDE.md)** - Collecting market data
- **[Backtest Analysis Guide](BACKTEST_ANALYSIS_GUIDE.md)** - Post-backtest analysis
