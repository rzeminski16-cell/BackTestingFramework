# Backtest Analysis Guide

This guide covers the Backtest Analysis Framework - a tool for drilling down into backtesting results to understand where strategies perform best and worst.

## Overview

The Backtest Analysis Framework processes trade logs to generate:

1. **Fundamental Features CSVs** - Per-security files with quarterly rows and yearly performance classifications
2. **Technical Features Master CSV** - All filtered trades with technical indicators and classification flags
3. **Summary Reports** - Statistics and coverage reports

## Quick Start

### Command Line Usage

```bash
# Basic usage - analyze a strategy's trade logs
python tools/run_backtest_analysis.py logs/AlphaTrendStrategy_Fundamental_Constraints_Base

# With custom output directory
python tools/run_backtest_analysis.py logs/MyStrategy -o analysis_output/my_analysis

# With custom thresholds
python tools/run_backtest_analysis.py logs/MyStrategy \
    --gb-threshold 7.5 \
    --min-trades 5 \
    --calmar-threshold 0.75
```

### Programmatic Usage

```python
from pathlib import Path
from Classes.Analysis.backtest_analyzer import BacktestAnalyzer, AnalysisConfig

# Create configuration
config = AnalysisConfig(
    raw_data_directory=Path("raw_data"),
    output_directory=Path("analysis_output/my_analysis"),

    # Custom thresholds
    gb_profit_good_threshold=7.5,  # 7.5% profit for "Good"
    min_trades_per_year=5,
    calmar_good_threshold=0.75,
)

# Run analysis
analyzer = BacktestAnalyzer(config)
results = analyzer.analyze(
    trade_logs_folder=Path("logs/AlphaTrendStrategy_Fundamental_Constraints_Base"),
    initial_capital=10000.0
)

print(f"Output saved to: {results['output_directory']}")
```

### Quick Function

```python
from Classes.Analysis.backtest_analyzer.analyzer import run_analysis

results = run_analysis(
    trade_logs_folder="logs/AlphaTrendStrategy",
    raw_data_directory="raw_data",
    output_directory="analysis_output/AlphaTrendStrategy",
    gb_profit_good_threshold=7.5,  # Override defaults
)
```

## Output Structure

```
analysis_output/{strategy_name}/
├── analysis_config.json              # Configuration used for this analysis
├── fundamental_features/             # Per-security CSVs
│   ├── AAPL_fundamental_features.csv
│   ├── GOOG_fundamental_features.csv
│   └── ...
├── technical_features/
│   └── technical_features_master.csv # All filtered trades with indicators
├── weekly_data/                      # Weekly indicator data for validation
│   ├── AAPL_weekly_indicators.csv
│   ├── GOOG_weekly_indicators.csv
│   └── ...
└── summaries/
    ├── overall_summary.csv           # High-level statistics
    ├── technical_features_summary.csv # Per-symbol and per-year breakdown
    └── trade_counts_by_year.csv      # Before/after filtering counts
```

## Fundamental Features CSV

Each security gets a CSV with quarterly rows covering the full trading range.

### Quarter Definitions (Calendar Quarters)

| Quarter | Months | Date Range |
|---------|--------|------------|
| Q1 | January, February, March | Jan 1 - Mar 31 |
| Q2 | April, May, June | Apr 1 - Jun 30 |
| Q3 | July, August, September | Jul 1 - Sep 30 |
| Q4 | October, November, December | Oct 1 - Dec 31 |

### Columns

| Column | Description |
|--------|-------------|
| `symbol` | Security symbol |
| `year` | Year |
| `quarter` | Quarter number (1-4) |
| `quarter_name` | Quarter name (Q1, Q2, Q3, Q4) |
| `period_GB_flag` | Yearly classification: "good", "indeterminate", "bad", or "no_trades" |
| `calmar_ratio` | Calmar ratio for the year |
| `max_drawdown_pct` | Maximum drawdown percentage for the year |
| `num_trades` | Number of trades in the year |
| `return_pct` | Total return percentage for the year |

### period_GB_flag Classification

The `period_GB_flag` is calculated **per year** (not per quarter) and assigned to all quarters in that year:

| Flag | Criteria |
|------|----------|
| `good` | Calmar ratio > 0.5 AND max drawdown ≤ 25% |
| `indeterminate` | Calmar ratio > 0.5 AND max drawdown > 25% |
| `bad` | Calmar ratio ≤ 0.5 |
| `Unknown` | No trades in that year |

### Example

```csv
symbol,year,quarter,quarter_name,period_GB_flag,calmar_ratio,max_drawdown_pct,num_trades,return_pct
AAPL,2020,1,Q1,good,1.74,2.09,3,3.65
AAPL,2020,2,Q2,good,1.74,2.09,3,3.65
AAPL,2020,3,Q3,good,1.74,2.09,3,3.65
AAPL,2020,4,Q4,good,1.74,2.09,3,3.65
AAPL,2021,1,Q1,bad,-1.27,5.41,4,-6.88
...
```

### Usage

You can manually append fundamental data (EPS, revenue growth, etc.) to these files for further analysis. The quarterly breakdown aligns with standard financial reporting periods.

## Technical Features Master CSV

A single CSV containing all trades that pass the filtering criteria.

### Trade Filtering

Only trades from security-year combinations with **at least N trades** (default: 4) are included. This ensures sufficient sample size for meaningful analysis.

**Important**: Trades are assigned to the year of their **entry date**, not exit date. A trade that enters in December 2020 and exits in January 2021 counts toward 2020. The P/L is realized on exit but attributed to the entry year.

### Columns

#### Original Trade Data
All columns from the trade log are preserved, including:
- `trade_id`, `symbol`, `entry_date`, `exit_date`
- `entry_price`, `exit_price`, `pl`, `pl_pct`
- `duration_days`, `entry_reason`, `exit_reason`
- And more...

#### Classification Flags

| Column | Description |
|--------|-------------|
| `GB_Flag` | Trade quality: "G" (Good), "I" (Indeterminate), "B" (Bad) |
| `Outcomes_Flag` | Post-exit behavior classification |

#### Technical Indicators at Entry

| Column | Description |
|--------|-------------|
| `rsi_14w` | 14-week RSI (SMA smoothed) |
| `high_low_distance_52w` | Position within 52-week range: (Price - Low) / (High - Low) |
| `bb_position` | Bollinger Bands position: normalized -1 (lower band) to +1 (upper band), 0 at middle |
| `bb_width` | Bollinger Band width: (Upper - Lower) / Middle × 100 |
| `volume_trend` | Current volume vs 50-week MA ratio |
| `atr_14w` | 14-week Average True Range |
| `atr_pct` | ATR as percentage of price |
| `price_vs_sma200` | Price distance from 200-day SMA (%) |
| `above_sma200` | 1 if price > SMA200, 0 otherwise |
| `volume_ratio` | Entry day volume vs 20-day average |
| `days_since_52w_high` | Trading days since 52-week high |
| `days_since_52w_low` | Trading days since 52-week low |
| `weeks_since_52w_high` | Weeks since 52-week high |
| `weeks_since_52w_low` | Weeks since 52-week low |

#### Post-Exit Price Data

| Column | Description |
|--------|-------------|
| `price_21d_after_exit` | Closing price 21 trading days after exit |
| `max_price_21d_after_exit` | Maximum high price in 21 days after exit |

### GB_Flag Classification

| Flag | Criteria |
|------|----------|
| `G` (Good) | Profit ≥ 5% |
| `I` (Indeterminate) | Profit ≥ 0% AND < 5% |
| `B` (Bad) | Profit < 0% (loss) |

### Outcomes_Flag Classification

| Flag | Criteria |
|------|----------|
| `FullRideGood` | Profit ≥ 5% AND price 21 days later ≤ exit price + 10% of entry price |
| `EarlyExitGood` | Profit ≥ 0% AND no major extension (price doesn't go > exit + 5%) |
| `MissedOpportunity` | Exit at small profit (<5%) or loss, but price surges >10% after exit |
| `Neutral` | Doesn't fit the above categories |
| `Unknown` | Insufficient future price data |

## Weekly Indicator Data

Weekly OHLCV data with calculated indicators, saved for manual validation.

### Columns

| Column | Description |
|--------|-------------|
| `week_end` | Week ending date (Friday) |
| `open`, `high`, `low`, `close`, `volume` | Weekly OHLCV |
| `rsi_14w` | 14-week RSI (SMA smoothed) |
| `high_52w`, `low_52w` | Rolling 52-week high/low |
| `high_low_distance_52w` | Position within 52-week range |
| `weeks_since_52w_high`, `weeks_since_52w_low` | Weeks since extremes |
| `bb_middle`, `bb_upper`, `bb_lower` | Bollinger Bands (20 periods, 2 std dev) |
| `bb_position`, `bb_width` | BB metrics |
| `volume_ma_50w` | 50-week volume moving average |
| `volume_trend` | Current vs MA ratio |
| `atr_14w`, `atr_pct` | ATR and ATR as % of price |

## Configuration Options

### AnalysisConfig Parameters

```python
@dataclass
class AnalysisConfig:
    # GB_Flag Thresholds
    gb_profit_good_threshold: float = 5.0  # 5% profit for "G"

    # Outcomes_Flag Thresholds
    full_ride_profit_min: float = 5.0           # Min profit for FullRideGood
    full_ride_price_extension: float = 10.0     # % of entry price tolerance
    early_exit_major_extension: float = 5.0     # Major extension threshold
    missed_opportunity_surge: float = 10.0      # Price surge threshold
    days_after_exit: int = 21                   # Days to check after exit

    # Period Classification Thresholds
    calmar_good_threshold: float = 0.5          # Min Calmar for "good"
    max_dd_good_threshold: float = 25.0         # Max DD for "good"

    # Trade Filtering
    min_trades_per_year: int = 4                # Min trades for inclusion

    # Indicator Parameters (in weeks)
    rsi_period_weeks: int = 14
    rsi_sma_period_weeks: int = 14
    high_low_period_weeks: int = 52
    bb_period_weeks: int = 20
    bb_std_dev: float = 2.0
    volume_ma_period_weeks: int = 50
    atr_period_weeks: int = 14

    # Daily indicator parameters
    sma_200_period_days: int = 200
    volume_ratio_period_days: int = 20

    # Output Options
    save_weekly_data: bool = True

    # Paths
    raw_data_directory: Path = None  # Required
    output_directory: Path = None    # Optional
```

### CLI Arguments

```
positional arguments:
  trade_logs_folder     Path to folder containing trade log CSV files

optional arguments:
  -o, --output          Output directory
  -r, --raw-data        Raw price data directory (default: raw_data)
  -c, --capital         Initial capital for calculations (default: 10000)

  --gb-threshold        Profit threshold for GB_Flag "G" (default: 5.0%)
  --calmar-threshold    Calmar ratio threshold for "good" (default: 0.5)
  --max-dd-threshold    Max drawdown threshold for "good" (default: 25.0%)
  --min-trades          Min trades per security/year (default: 4)
  --days-after          Days after exit for Outcomes_Flag (default: 21)

  --full-ride-profit    Min profit for FullRideGood (default: 5.0%)
  --full-ride-extension Price extension tolerance (default: 10.0%)
  --early-exit-extension Major extension threshold (default: 5.0%)
  --missed-opp-surge    Surge threshold for MissedOpportunity (default: 10.0%)

  --no-weekly           Skip saving weekly indicator data
  -v, --verbose         Enable verbose output
```

## Examples

### Analyzing Different Profit Thresholds

```bash
# Conservative: 10% profit threshold for "Good"
python tools/run_backtest_analysis.py logs/MyStrategy --gb-threshold 10.0

# Aggressive: 3% profit threshold for "Good"
python tools/run_backtest_analysis.py logs/MyStrategy --gb-threshold 3.0
```

### Stricter Period Classification

```bash
# Require higher Calmar and lower drawdown for "good" periods
python tools/run_backtest_analysis.py logs/MyStrategy \
    --calmar-threshold 1.0 \
    --max-dd-threshold 15.0
```

### More Inclusive Trade Filtering

```bash
# Include periods with just 3+ trades
python tools/run_backtest_analysis.py logs/MyStrategy --min-trades 2
```

### Single Security Analysis

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

## Workflow

### Typical Analysis Workflow

1. **Run analysis** on your trade logs folder
2. **Review overall_summary.csv** for high-level statistics
3. **Examine trade_counts_by_year.csv** to understand filtering impact
4. **Append fundamental data** to fundamental features CSVs
5. **Analyze technical_features_master.csv** to find patterns
6. **Validate weekly_data** CSVs if indicators seem off

### Finding Strategy Strengths

Look for patterns in `technical_features_master.csv`:
- Filter by `GB_Flag == 'G'` to see winning trades
- Check which indicator ranges correlate with good outcomes
- Compare `rsi_14w`, `bb_position`, `volume_trend` between G, I, B trades

### Identifying Problem Areas

- Filter `period_GB_flag == 'bad'` in fundamental features
- Look for `Outcomes_Flag == 'MissedOpportunity'` trades
- Check if certain indicator ranges correlate with losses

## Technical Notes

### Weekly Data Calculation

- Daily data is resampled to weekly using Friday as week end
- Indicators are calculated on weekly bars, then looked up for each trade
- This provides a longer-term view of market conditions at trade entry

### Calmar Ratio Calculation

```
Calmar Ratio = Yearly Return % / Max Drawdown %
```

- Uses actual yearly return (not annualized)
- Max drawdown calculated from cumulative P/L within the year
- When max drawdown is 0 (all gains), returns 999.99

### Price After Exit

- `price_21d_after_exit`: Closing price 21 trading days after exit
- If fewer than 21 days of data remain, uses last available price
- `max_price_21d_after_exit`: Maximum high price in that period

## Troubleshooting

### "Raw data file not found for {symbol}"

Ensure the raw data directory contains CSV files matching your trade log symbols:
- Trade log: `MyStrategy_AAPL_trades.csv`
- Raw data: `raw_data/AAPL.csv`

### Low Inclusion Rate

If many trades are filtered out:
- Check `trade_counts_by_year.csv` to see which symbol-years are excluded
- Consider lowering `--min-trades` threshold
- Ensure trades span multiple years per security

### Missing Indicators

Some indicators require warmup periods:
- RSI needs 14 weeks of data
- Bollinger Bands need 20 weeks
- SMA200 needs 200 days

Early trades may have NaN for some indicators.

## Fundamental Data Integration

The backtest analysis can be enhanced with fundamental data from Alpha Vantage. See the [Fundamental Data Guide](FUNDAMENTAL_DATA_GUIDE.md) for complete details.

### Quick Setup

```bash
# 1. Create Alpha Vantage config
python tools/run_fundamental_data_fetch.py --create-config

# 2. Add your API key
nano alpha_vantage_config.json

# 3. Fetch fundamental data
python tools/run_fundamental_data_fetch.py logs/AlphaTrendStrategy
```

### Available Metrics

The fundamental data fetcher provides 20+ metrics per quarter:

| Category | Metrics |
|----------|---------|
| **EPS** | TTM, Growth Rate (YoY), Surprise Trend |
| **Revenue** | TTM, Growth Rate (YoY) |
| **Margins** | Operating Margin, Gross Margin |
| **Valuation** | P/E (Trailing & Forward), PEG, P/B, P/CF |
| **Cash Flow** | FCF, FCF Trend, FCF Yield |
| **Leverage** | Debt-to-Equity, Current Ratio, Interest Coverage |
| **Returns** | ROE, ROA |
| **Other** | Dividend Yield, Beta, Analyst Target Price |

### Merging with Backtest Results

```python
import pandas as pd

# Load backtest fundamental features
backtest_df = pd.read_csv("analysis_output/fundamental_features/AAPL_fundamental_features.csv")

# Load Alpha Vantage data
av_df = pd.read_csv("analysis_output/fundamental_data/AAPL_fundamental_data.csv")

# Merge on year and quarter
merged = backtest_df.merge(av_df, on=['symbol', 'year', 'quarter'], how='left')

# Now analyze strategy performance vs fundamentals
good_periods = merged[merged['period_GB_flag'] == 'good']
print(f"Average P/E in good periods: {good_periods['pe_ratio_trailing'].mean():.2f}")
```

### Output Structure

After running both analysis tools:

```
analysis_output/{strategy}/
├── fundamental_features/      # From run_backtest_analysis.py
│   ├── AAPL_fundamental_features.csv
│   └── ...
├── fundamental_data/          # From run_fundamental_data_fetch.py
│   ├── AAPL_fundamental_data.csv
│   └── logs/
│       ├── decisions.log
│       └── issues.log
├── technical_features/
│   └── technical_features_master.csv
└── summaries/
    └── ...
```
