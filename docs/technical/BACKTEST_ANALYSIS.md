# Backtest Analysis Guide

Technical guide for the Backtest Analysis Framework - drilling down into backtesting results to understand strategy performance.

## Overview

The Backtest Analysis Framework processes trade logs to generate:

1. **Fundamental Features CSVs** - Per-security files with quarterly rows and yearly performance classifications
2. **Technical Features Master CSV** - All filtered trades with technical indicators and classification flags
3. **Summary Reports** - Statistics and coverage reports

---

## Quick Start

### Command Line

```bash
# Basic usage
python run_backtest_analysis.py logs/AlphaTrendStrategy_Backtest

# With custom output
python run_backtest_analysis.py logs/MyStrategy -o analysis_output/my_analysis

# With custom thresholds
python run_backtest_analysis.py logs/MyStrategy \
    --gb-threshold 7.5 \
    --min-trades 5 \
    --calmar-threshold 0.75
```

### Programmatic

```python
from pathlib import Path
from Classes.Analysis.backtest_analyzer import BacktestAnalyzer, AnalysisConfig

config = AnalysisConfig(
    raw_data_directory=Path("raw_data"),
    output_directory=Path("analysis_output/my_analysis"),
    gb_profit_good_threshold=7.5,
    min_trades_per_year=5,
    calmar_good_threshold=0.75,
)

analyzer = BacktestAnalyzer(config)
results = analyzer.analyze(
    trade_logs_folder=Path("logs/AlphaTrendStrategy_Backtest"),
    initial_capital=10000.0
)

print(f"Output: {results['output_directory']}")
```

---

## Output Structure

```
analysis_output/{strategy_name}/
├── analysis_config.json              # Configuration used
├── fundamental_features/             # Per-security CSVs
│   ├── AAPL_fundamental_features.csv
│   ├── GOOG_fundamental_features.csv
│   └── ...
├── technical_features/
│   └── technical_features_master.csv # All filtered trades
├── weekly_data/                      # Weekly indicators
│   ├── AAPL_weekly_indicators.csv
│   └── ...
└── summaries/
    ├── overall_summary.csv
    ├── technical_features_summary.csv
    └── trade_counts_by_year.csv
```

---

## Fundamental Features CSV

Each security gets a CSV with quarterly rows.

### Quarter Definitions (Calendar)

| Quarter | Months | Date Range |
|---------|--------|------------|
| Q1 | Jan, Feb, Mar | Jan 1 - Mar 31 |
| Q2 | Apr, May, Jun | Apr 1 - Jun 30 |
| Q3 | Jul, Aug, Sep | Jul 1 - Sep 30 |
| Q4 | Oct, Nov, Dec | Oct 1 - Dec 31 |

### Columns

| Column | Description |
|--------|-------------|
| `symbol` | Security symbol |
| `year` | Year |
| `quarter` | Quarter number (1-4) |
| `quarter_name` | Q1, Q2, Q3, Q4 |
| `period_GB_flag` | Yearly: "good", "indeterminate", "bad", "no_trades" |
| `calmar_ratio` | Calmar ratio for the year |
| `max_drawdown_pct` | Max drawdown % for the year |
| `num_trades` | Trades in the year |
| `return_pct` | Total return % for the year |

### period_GB_flag Classification

Calculated per year, assigned to all quarters:

| Flag | Criteria |
|------|----------|
| `good` | Calmar > 0.5 AND max_dd ≤ 25% |
| `indeterminate` | Calmar > 0.5 AND max_dd > 25% |
| `bad` | Calmar ≤ 0.5 |
| `Unknown` | No trades that year |

---

## Technical Features Master CSV

All trades passing filtering criteria with technical indicators.

### Trade Filtering

Only trades from security-year combinations with at least N trades (default: 4) are included. Trades assigned to entry date year.

### Classification Flags

**GB_Flag (Trade Quality):**

| Flag | Criteria |
|------|----------|
| `G` (Good) | Profit ≥ 5% |
| `I` (Indeterminate) | 0% ≤ Profit < 5% |
| `B` (Bad) | Profit < 0% |

**Outcomes_Flag (Post-Exit Behavior):**

| Flag | Criteria |
|------|----------|
| `FullRideGood` | Profit ≥ 5% AND price 21d later ≤ exit + 10% |
| `EarlyExitGood` | Profit ≥ 0% AND no major extension |
| `MissedOpportunity` | Small profit/loss but price surges >10% after |
| `Neutral` | Doesn't fit above |
| `Unknown` | Insufficient future data |

### Technical Indicators

| Column | Description |
|--------|-------------|
| `rsi_14w` | 14-week RSI (SMA smoothed) |
| `high_low_distance_52w` | Position in 52-week range |
| `bb_position` | Bollinger Bands position (-1 to +1) |
| `bb_width` | Bollinger Band width % |
| `volume_trend` | Current vs 50-week MA ratio |
| `atr_14w` | 14-week Average True Range |
| `atr_pct` | ATR as % of price |
| `price_vs_sma200` | Distance from 200-day SMA % |
| `above_sma200` | 1 if above, 0 if below |
| `volume_ratio` | Entry volume vs 20-day avg |
| `days_since_52w_high` | Trading days since high |
| `days_since_52w_low` | Trading days since low |
| `weeks_since_52w_high` | Weeks since high |
| `weeks_since_52w_low` | Weeks since low |

### Post-Exit Data

| Column | Description |
|--------|-------------|
| `price_21d_after_exit` | Close 21 trading days after exit |
| `max_price_21d_after_exit` | Max high in 21 days after exit |

---

## Configuration

### AnalysisConfig

```python
@dataclass
class AnalysisConfig:
    # GB_Flag Thresholds
    gb_profit_good_threshold: float = 5.0

    # Outcomes_Flag Thresholds
    full_ride_profit_min: float = 5.0
    full_ride_price_extension: float = 10.0
    early_exit_major_extension: float = 5.0
    missed_opportunity_surge: float = 10.0
    days_after_exit: int = 21

    # Period Classification
    calmar_good_threshold: float = 0.5
    max_dd_good_threshold: float = 25.0

    # Trade Filtering
    min_trades_per_year: int = 4

    # Indicator Parameters (weeks)
    rsi_period_weeks: int = 14
    rsi_sma_period_weeks: int = 14
    high_low_period_weeks: int = 52
    bb_period_weeks: int = 20
    bb_std_dev: float = 2.0
    volume_ma_period_weeks: int = 50
    atr_period_weeks: int = 14

    # Daily parameters
    sma_200_period_days: int = 200
    volume_ratio_period_days: int = 20

    # Output
    save_weekly_data: bool = True

    # Paths
    raw_data_directory: Path = None
    output_directory: Path = None
```

### CLI Arguments

```
positional arguments:
  trade_logs_folder     Path to trade log CSVs

optional arguments:
  -o, --output          Output directory
  -r, --raw-data        Raw data directory (default: raw_data)
  -c, --capital         Initial capital (default: 10000)

  --gb-threshold        GB_Flag "G" threshold (default: 5.0%)
  --calmar-threshold    Calmar "good" threshold (default: 0.5)
  --max-dd-threshold    Max DD "good" threshold (default: 25.0%)
  --min-trades          Min trades per security/year (default: 4)
  --days-after          Days after exit (default: 21)

  --full-ride-profit    FullRideGood min profit (default: 5.0%)
  --full-ride-extension FullRideGood extension (default: 10.0%)
  --early-exit-extension Major extension (default: 5.0%)
  --missed-opp-surge    MissedOpportunity surge (default: 10.0%)

  --no-weekly           Skip weekly data output
  -v, --verbose         Verbose output
```

---

## Fundamental Data Integration

Enhance analysis with Alpha Vantage fundamental data.

### Fetch Fundamental Data

```bash
# Create config
python run_fundamental_data_fetch.py --create-config

# Add API key
nano alpha_vantage_config.json

# Fetch data
python run_fundamental_data_fetch.py logs/AlphaTrendStrategy
```

### Available Metrics

| Category | Metrics |
|----------|---------|
| EPS | TTM, Growth Rate, Surprise Trend |
| Revenue | TTM, Growth Rate |
| Margins | Operating, Gross |
| Valuation | P/E, PEG, P/B, P/CF |
| Cash Flow | FCF, FCF Trend, FCF Yield |
| Leverage | Debt-to-Equity, Current Ratio |
| Returns | ROE, ROA |
| Other | Dividend Yield, Beta, Analyst Target |

### Merge with Backtest Results

```python
import pandas as pd

# Load backtest data
backtest_df = pd.read_csv(
    "analysis_output/fundamental_features/AAPL_fundamental_features.csv"
)

# Load Alpha Vantage data
av_df = pd.read_csv(
    "analysis_output/fundamental_data/AAPL_fundamental_data.csv"
)

# Merge on year and quarter
merged = backtest_df.merge(av_df, on=['symbol', 'year', 'quarter'], how='left')

# Analyze
good_periods = merged[merged['period_GB_flag'] == 'good']
print(f"Avg P/E in good periods: {good_periods['pe_ratio_trailing'].mean():.2f}")
```

---

## Analysis Workflow

### 1. Run Analysis

```bash
python run_backtest_analysis.py logs/AlphaTrendStrategy
```

### 2. Review Summaries

- `overall_summary.csv` - High-level statistics
- `trade_counts_by_year.csv` - Filtering impact

### 3. Fetch Fundamental Data

```bash
python run_fundamental_data_fetch.py logs/AlphaTrendStrategy
```

### 4. Analyze Technical Features

```python
import pandas as pd

df = pd.read_csv("analysis_output/technical_features/technical_features_master.csv")

# Good trades
good = df[df['GB_Flag'] == 'G']
print(f"Good trades: {len(good)}")
print(f"Avg RSI at entry: {good['rsi_14w'].mean():.1f}")
print(f"Avg BB position: {good['bb_position'].mean():.2f}")

# Bad trades
bad = df[df['GB_Flag'] == 'B']
print(f"\nBad trades: {len(bad)}")
print(f"Avg RSI at entry: {bad['rsi_14w'].mean():.1f}")
print(f"Avg BB position: {bad['bb_position'].mean():.2f}")
```

### 5. Find Patterns

```python
# Missed opportunities
missed = df[df['Outcomes_Flag'] == 'MissedOpportunity']
print(f"Missed opportunities: {len(missed)}")
print(f"Avg exit P/L: {missed['pl_pct'].mean():.1f}%")

# What could have been
print(f"Avg 21d post-exit price: {missed['max_price_21d_after_exit'].mean():.2f}")
```

---

## Technical Notes

### Weekly Data Calculation

- Daily data resampled to weekly (Friday week-end)
- Indicators calculated on weekly bars
- Looked up for each trade entry

### Calmar Ratio

```
Calmar = Yearly Return % / Max Drawdown %
```

- Uses actual yearly return (not annualized)
- Max drawdown from cumulative P/L within year
- When max_dd = 0 (all gains), returns 999.99

### Price After Exit

- `price_21d_after_exit`: Close 21 trading days after
- Uses last available if fewer than 21 days remain
- `max_price_21d_after_exit`: Max high in period

---

## Troubleshooting

### "Raw data file not found"

Ensure raw data matches trade log symbols:
- Trade log: `MyStrategy_AAPL_trades.csv`
- Raw data: `raw_data/AAPL.csv`

### Low Inclusion Rate

If many trades filtered out:
- Check `trade_counts_by_year.csv`
- Lower `--min-trades` threshold
- Ensure trades span multiple years

### Missing Indicators

Some indicators need warmup:
- RSI: 14 weeks
- Bollinger Bands: 20 weeks
- SMA200: 200 days

Early trades may have NaN for some indicators.

---

## Single Security Analysis

```python
from Classes.Analysis.backtest_analyzer import BacktestAnalyzer, AnalysisConfig

config = AnalysisConfig(raw_data_directory=Path("raw_data"))
analyzer = BacktestAnalyzer(config)

results = analyzer.analyze_single_security(
    trade_log_file=Path("logs/MyStrategy/MyStrategy_AAPL_trades.csv"),
    symbol="AAPL",
    output_dir=Path("analysis_output/AAPL_analysis"),
)
```
