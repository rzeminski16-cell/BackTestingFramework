# Fundamental Data Fetcher Guide

This guide covers the Fundamental Data Fetcher - a tool for collecting point-in-time fundamental metrics from Alpha Vantage for use in backtest analysis.

## Overview

The Fundamental Data Fetcher pulls financial data from Alpha Vantage's premium API and calculates key metrics for each security/quarter combination. This data can be merged with the backtest analyzer's fundamental features to create a comprehensive analysis dataset.

## Prerequisites

### 1. Alpha Vantage Premium API Key

You need a premium Alpha Vantage subscription. Get one at: https://www.alphavantage.co/premium/

### 2. Configuration File

Create a configuration file with your API key:

```bash
# Generate sample config
python run_fundamental_data_fetch.py --create-config
```

This creates `alpha_vantage_config.json`:

```json
{
  "api_key": "YOUR_API_KEY_HERE",
  "rate_limit_per_minute": 75,
  "cache_directory": "cache/alpha_vantage",
  "cache_expiry_days": 1,
  "request_timeout": 30,
  "max_retries": 3,
  "retry_delay": 2.0
}
```

Edit this file and replace `YOUR_API_KEY_HERE` with your actual API key.

**Rate Limit Settings by Plan:**

| Plan | Calls/Minute |
|------|--------------|
| Premium (lowest) | 75 |
| Premium (mid) | 150-600 |
| Premium (highest) | 1200 |

## Quick Start

```bash
# Fetch fundamental data for all securities in your trade logs
python run_fundamental_data_fetch.py logs/AlphaTrendStrategy

# With custom date range
python run_fundamental_data_fetch.py logs/AlphaTrendStrategy --start-year 2015 --end-year 2024

# With custom output directory
python run_fundamental_data_fetch.py logs/AlphaTrendStrategy -o analysis_output/fundamentals

# For specific symbols only
python run_fundamental_data_fetch.py logs/AlphaTrendStrategy --symbols AAPL MSFT GOOG
```

## Output Files

### Directory Structure

```
analysis_output/{strategy_name}/
├── fundamental_data/
│   ├── AAPL_fundamental_data.csv
│   ├── GOOG_fundamental_data.csv
│   ├── MSFT_fundamental_data.csv
│   └── logs/
│       ├── decisions.log          # All user decisions made
│       ├── remembered_answers.json # Answers to reuse
│       ├── issues.log             # Problems encountered
│       └── fundamental_fetch_*.log # Detailed run log
```

### Fundamental Data CSV Columns

#### Identification
| Column | Description |
|--------|-------------|
| `symbol` | Stock ticker |
| `year` | Year |
| `quarter` | Quarter (1-4) |
| `quarter_end_date` | End date of quarter |
| `is_recent` | Whether this quarter uses current Forward P/E |
| `price_at_quarter_end` | Stock price at quarter end |

#### EPS Metrics
| Column | Description |
|--------|-------------|
| `eps_ttm` | Trailing 12-month diluted EPS (sum of last 4 quarters) |
| `eps_growth_yoy_pct` | Year-over-year EPS growth percentage |
| `eps_surprise_trend` | Beat/Miss trend for last 4 quarters (e.g., "3/4 Beat") |

#### Revenue & Profitability
| Column | Description |
|--------|-------------|
| `revenue_ttm` | Trailing 12-month revenue |
| `revenue_growth_yoy_pct` | Year-over-year revenue growth percentage |
| `operating_income_ttm` | Trailing 12-month operating income |
| `operating_margin_ttm_pct` | Operating margin (Operating Income / Revenue × 100) |
| `gross_profit_ttm` | Trailing 12-month gross profit |
| `gross_margin_ttm_pct` | Gross margin (Gross Profit / Revenue × 100) |
| `ebitda_ttm` | Trailing 12-month EBITDA |

#### Valuation Ratios
| Column | Description |
|--------|-------------|
| `pe_ratio_trailing` | Price / TTM EPS |
| `pe_ratio_forward` | Price / Forward EPS (recent quarters only, else NaN) |
| `peg_ratio` | P/E / EPS Growth Rate (recent quarters: from API; historical: calculated) |
| `price_to_book` | Price / Book Value per Share |
| `price_to_cash_flow` | Price / Operating Cash Flow per Share |

#### Cash Flow Metrics
| Column | Description |
|--------|-------------|
| `operating_cash_flow_ttm` | TTM operating cash flow |
| `capex_ttm` | TTM capital expenditures |
| `fcf_ttm` | Free Cash Flow (OCF - CapEx) |
| `fcf_growth_yoy_pct` | Year-over-year FCF growth |
| `fcf_yield_pct` | FCF / Market Cap × 100 |

#### Balance Sheet Metrics
| Column | Description |
|--------|-------------|
| `debt_to_equity` | Total Debt / Shareholders' Equity |
| `current_ratio` | Current Assets / Current Liabilities |
| `interest_coverage` | Operating Income / Interest Expense |

#### Additional Metrics (Recent Quarters Only)
| Column | Description |
|--------|-------------|
| `roe_ttm_pct` | Return on Equity (TTM) |
| `roa_ttm_pct` | Return on Assets (TTM) |
| `dividend_yield_pct` | Annual dividend / Price × 100 |
| `beta` | Market beta (volatility vs. market) |
| `analyst_target_price` | Consensus analyst price target |

## Point-in-Time Data

### Historical vs Current Data

The fetcher calculates **point-in-time historical data** where possible:

| Metric Type | Historical? | Notes |
|-------------|-------------|-------|
| EPS (TTM) | ✅ Yes | Calculated from quarterly earnings history |
| EPS Growth | ✅ Yes | Calculated from historical EPS |
| Revenue/Margins | ✅ Yes | Calculated from quarterly income statements |
| P/E Trailing | ✅ Yes | Uses historical price and historical EPS |
| FCF & Trends | ✅ Yes | Calculated from quarterly cash flow statements |
| Debt/Equity | ✅ Yes | From quarterly balance sheets |
| **Forward P/E** | ⚠️ Recent only | Current analyst estimates not available historically |
| **PEG Ratio** | ⚠️ Partial | Recent: from API; Historical: calculated from P/E and growth |
| **ROE/ROA** | ❌ Current | API only provides current values |
| **Beta** | ❌ Current | API only provides current value |

### "Recent" Quarter Definition

Quarters within the last 8 quarters (~2 years) are considered "recent" and receive Forward P/E from the API. Older quarters have `pe_ratio_forward = NaN`.

## Interactive Mode

### Handling Ambiguous Data

When the fetcher encounters ambiguous data, it will prompt you for a decision:

```
============================================================
DECISION REQUIRED: Which EPS field should be used?
============================================================
Context: {"symbol": "AAPL", "available_fields": ["reportedEPS", "estimatedEPS"]}

Options:
  [1] reportedEPS (default)
  [2] estimatedEPS

Enter choice [1-2] (default: 1):
```

### Answer Memory

Your answers are remembered for the session:

- Same question type → Same answer (no re-prompting)
- Answers saved to `logs/remembered_answers.json`
- Use `--clear-answers` to reset

### Logs

All issues are logged but don't stop processing:

- `decisions.log` - All decisions made
- `issues.log` - Problems encountered (missing data, API errors)
- `*.log` - Full execution log

## API Endpoints Used

| Endpoint | Data Retrieved |
|----------|----------------|
| `OVERVIEW` | Current ratios (P/E, P/B, ROE, Beta, etc.) |
| `INCOME_STATEMENT` | Quarterly/annual revenue, income, margins |
| `BALANCE_SHEET` | Quarterly/annual debt, equity, assets |
| `CASH_FLOW` | Quarterly/annual OCF, CapEx, dividends |
| `EARNINGS` | EPS history with analyst estimates |
| `TIME_SERIES_DAILY_ADJUSTED` | Historical prices (if not in local files) |

## Caching

API responses are cached to:
- Reduce API calls
- Speed up re-runs
- Avoid rate limit issues

Cache settings in config:
- `cache_directory`: Where to store cache files
- `cache_expiry_days`: How long to keep cached responses

```bash
# Clear cache before fetching
python run_fundamental_data_fetch.py logs/MyStrategy --clear-cache
```

## Programmatic Usage

```python
from pathlib import Path
from Classes.Analysis.backtest_analyzer import (
    AlphaVantageConfig,
    AlphaVantageClient,
    FundamentalDataFetcher,
    InteractiveHandler,
)

# Load configuration
config = AlphaVantageConfig.load()

# Initialize components
client = AlphaVantageClient(config)
handler = InteractiveHandler(log_dir=Path("logs"))
fetcher = FundamentalDataFetcher(
    client=client,
    interactive_handler=handler,
    price_data_dir=Path("raw_data")  # Optional: for historical prices
)

# Fetch data for a single symbol
df = fetcher.fetch_fundamental_data("AAPL", start_year=2020, end_year=2024)
print(df.head())

# Fetch for multiple symbols
symbols = ["AAPL", "MSFT", "GOOG"]
results = fetcher.fetch_for_multiple_symbols(
    symbols=symbols,
    start_year=2020,
    end_year=2024,
    output_dir=Path("output/fundamentals")
)
```

## Merging with Backtest Analysis

After fetching fundamental data, merge it with the backtest analyzer's output:

```python
import pandas as pd
from pathlib import Path

# Load backtest fundamental features
backtest_df = pd.read_csv("analysis_output/fundamental_features/AAPL_fundamental_features.csv")

# Load Alpha Vantage fundamental data
av_df = pd.read_csv("analysis_output/fundamental_data/AAPL_fundamental_data.csv")

# Merge on year and quarter
merged = backtest_df.merge(
    av_df,
    on=['symbol', 'year', 'quarter'],
    how='left'
)

# Now you have period_GB_flag alongside EPS, P/E, etc.
print(merged[['year', 'quarter', 'period_GB_flag', 'eps_ttm', 'pe_ratio_trailing']].head())
```

## Troubleshooting

### "API key not found"

```bash
# Create config file
python run_fundamental_data_fetch.py --create-config

# Edit with your API key
nano alpha_vantage_config.json
```

### "Rate limit exceeded"

- Check your subscription tier
- Increase `rate_limit_per_minute` in config only if your plan allows
- The client handles rate limiting automatically

### "No data for symbol"

Some symbols may not have fundamental data:
- ETFs don't have fundamental data
- New IPOs may have limited history
- Delisted stocks may be unavailable

Check `issues.log` for details.

### Slow fetching

- Each symbol requires ~5-6 API calls
- Use local price data (`--raw-data` option) to reduce API calls
- Caching speeds up re-runs significantly

## CLI Reference

```
usage: run_fundamental_data_fetch.py [-h] [-o OUTPUT] [--start-year START_YEAR]
                                     [--end-year END_YEAR] [--config CONFIG]
                                     [--raw-data RAW_DATA] [--symbols SYMBOLS...]
                                     [--create-config] [--clear-cache]
                                     [--clear-answers] [-v]
                                     [trade_logs_folder]

positional arguments:
  trade_logs_folder     Path to folder containing trade log CSV files

optional arguments:
  -o, --output          Output directory for fundamental data files
  --start-year          First year to fetch data for (default: 2010)
  --end-year            Last year to fetch data for (default: current year)
  --config              Path to Alpha Vantage config file
  --raw-data            Directory containing raw price CSV files (default: raw_data)
  --symbols             Specific symbols to fetch (overrides trade log extraction)
  --create-config       Create a sample configuration file and exit
  --clear-cache         Clear API response cache before fetching
  --clear-answers       Clear remembered answers before starting
  -v, --verbose         Enable verbose output
```
