# Data Collection Guide

This guide explains how to use the Data Collection System to gather historical market data for backtesting.

## Overview

The Data Collection System is a GUI-based tool that automates the process of fetching historical market data from Alpha Vantage. It replaces the previous manual data collection process with a unified, professional-grade interface.

### What It Does

- Fetches historical price data (daily and weekly timeframes)
- Retrieves 50+ technical indicators calculated by Alpha Vantage
- Gathers fundamental data (income statements, balance sheets, cash flow)
- Collects insider trading activity
- Downloads forex currency pair data
- Retrieves historical options data

### Data Output

All collected data is saved as CSV files in the `raw_data/` directory, ready for use with the backtesting framework.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `customtkinter>=5.2.0` - Modern GUI framework
- `requests>=2.28.0` - HTTP client for API calls

### 2. Configure Your API Key

Create a settings file at `config/data_collection/settings.json`:

```json
{
    "api": {
        "api_key": "YOUR_ALPHA_VANTAGE_API_KEY",
        "requests_per_minute": 75,
        "base_url": "https://www.alphavantage.co/query"
    },
    "cache": {
        "enabled": true,
        "directory": ".cache/alpha_vantage",
        "ttl_hours": 24
    }
}
```

You can copy from the template:
```bash
cp config/data_collection/settings.json.template config/data_collection/settings.json
```

Then edit the file and replace `YOUR_ALPHA_VANTAGE_API_KEY` with your actual key.

### 3. Launch the Application

```bash
python apps/data_collection_gui.py
```

---

## Ticker Presets

The system includes a pre-configured ticker universe in `config/data_collection/tickers.json`. This file contains all securities from SECURITIES.md organized by category.

### Using Presets in the GUI

Each tab with a ticker selector includes a "Load Preset" dropdown. Available presets include:

| Preset | Description |
|--------|-------------|
| All Stocks | Complete stock universe (165+ tickers) |
| Large Cap Only | Large cap stocks across all sectors |
| All ETFs | Complete ETF universe (46 tickers) |
| Tech Sector | Electronic Technology + Technology Services |
| Healthcare Sector | Healthcare stocks only |
| Finance Sector | Financial stocks only |
| US Tradeable | US-listed stocks only (excludes non-US) |
| Sector: [Name] | Individual sector breakdowns |
| Forex Pairs | Common currency pairs |

### Editing the Ticker File

The `tickers.json` file structure:

```json
{
    "categories": {
        "stocks": {
            "consumer_discretionary": {
                "large_cap": ["AMZN", "SHOP", ...],
                "mid_cap": [...],
                "small_cap": [...]
            },
            ...
        },
        "etfs": {...},
        "crypto": [...]
    },
    "presets": {
        "all_stocks": {
            "description": "All stocks from the universe",
            "tickers": [...]
        },
        ...
    },
    "forex_pairs": ["EUR/USD", "GBP/USD", ...]
}
```

Add custom presets by adding entries to the `presets` section.

---

## API Subscription Tiers

The system is optimized for Alpha Vantage's Premium tier (75 requests/minute). If you have a different subscription, update the `requests_per_minute` setting accordingly:

| Tier | Requests/Minute | Recommended Setting |
|------|-----------------|---------------------|
| Free | 5 | 5 |
| Premium | 75 | 75 |
| Enterprise | 150+ | Match your limit |

The system automatically handles rate limiting with queuing and exponential backoff on errors.

---

## GUI Overview

The application has six tabs, each dedicated to a specific data type:

### Tab Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Daily] [Weekly] [Fundamental] [Insider] [Forex] [Options] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Tab-specific controls and options                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Progress bar and status                                    │
├─────────────────────────────────────────────────────────────┤
│  Log output area                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tab Details

### Daily Data Tab

Collects daily OHLCV (Open, High, Low, Close, Volume) data with optional technical indicators.

**Configuration Options:**

| Option | Description |
|--------|-------------|
| Symbols | Comma-separated list (e.g., `AAPL, MSFT, GOOGL`) |
| Date Range | Preset (1Y, 5Y, 10Y, Max) or custom start/end dates |
| Output Size | `compact` (100 days) or `full` (20+ years) |
| Adjusted | Use split/dividend adjusted prices |
| Technical Indicators | Select from 50+ available indicators |

**Technical Indicators Available:**

The system supports all Alpha Vantage technical indicators:

- **Trend**: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, VWAP
- **Momentum**: RSI, STOCH, STOCHF, WILLR, ADX, ADXR, CCI, AROON, MOM, ROC, ROCR, TRIX, ULTOSC, DX, MINUS_DI, PLUS_DI, MINUS_DM, PLUS_DM, MFI, BOP, PPO, APO, CMO
- **Volatility**: ATR, NATR, BBANDS, MIDPOINT, MIDPRICE
- **Volume**: AD, ADOSC, OBV
- **Other**: MACD, MACDEXT, STOCHRSI, HT_TRENDLINE, HT_SINE, HT_TRENDMODE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, SAR

**Output File Format:**

```
raw_data/daily/AAPL_daily_2020-01-01_2024-01-01.csv
```

Columns: `date, open, high, low, close, volume, [indicator columns...]`

---

### Weekly Data Tab

Collects weekly OHLCV data with optional technical indicators. Uses the same configuration options as Daily Data.

**Output File Format:**

```
raw_data/weekly/AAPL_weekly_2020-01-01_2024-01-01.csv
```

---

### Fundamental Data Tab

Retrieves company fundamental data from financial statements.

**Available Data Categories:**

| Category | Description |
|----------|-------------|
| Income Statement | Revenue, earnings, expenses (annual/quarterly) |
| Balance Sheet | Assets, liabilities, equity (annual/quarterly) |
| Cash Flow | Operating, investing, financing flows (annual/quarterly) |
| Earnings | EPS, estimates, surprises |
| Company Overview | Sector, industry, market cap, description |

**Configuration Options:**

| Option | Description |
|--------|-------------|
| Symbols | Comma-separated list of stock tickers |
| Categories | Select which data types to fetch |
| Period | Annual, Quarterly, or Both |

**Output File Format:**

```
raw_data/fundamental/AAPL_income_statement_annual.csv
raw_data/fundamental/AAPL_balance_sheet_quarterly.csv
```

---

### Insider Data Tab

Collects insider trading transactions (Form 4 filings).

**Data Collected:**

- Transaction date
- Insider name and title
- Transaction type (buy/sell)
- Shares traded
- Price per share
- Value of transaction
- Shares owned after transaction

**Output File Format:**

```
raw_data/insider/AAPL_insider_transactions.csv
```

---

### Forex Tab

Retrieves foreign exchange rate data for currency pairs.

**Configuration Options:**

| Option | Description |
|--------|-------------|
| Currency Pairs | Select from major pairs (EUR/USD, GBP/USD, etc.) |
| Timeframe | Intraday (1min-60min), Daily, Weekly, Monthly |
| Output Size | Compact or Full history |

**Popular Currency Pairs:**

- EUR/USD, GBP/USD, USD/JPY, USD/CHF
- AUD/USD, USD/CAD, NZD/USD
- EUR/GBP, EUR/JPY, GBP/JPY

**Output File Format:**

```
raw_data/forex/EURUSD_daily.csv
```

---

### Options Tab

Retrieves historical options chain data.

**Configuration Options:**

| Option | Description |
|--------|-------------|
| Symbols | Underlying stock symbols |
| Date Range | Historical date range for options data |
| Contract Types | Calls, Puts, or Both |
| Expiration Filter | Filter by expiration date range |

**Data Collected:**

- Strike price
- Expiration date
- Contract type (call/put)
- Bid/Ask prices
- Volume and open interest
- Implied volatility
- Greeks (delta, gamma, theta, vega)

**Note:** Alpha Vantage provides 15+ years of historical options data. If data cannot be fully gathered for a requested range, the system will flag this in the logs.

**Output File Format:**

Options data is organized by ticker in subfolders:

```
raw_data/options/AAPL/AAPL_20240119_calls.csv
raw_data/options/AAPL/AAPL_20240119_puts.csv
raw_data/options/MSFT/MSFT_20240119_calls.csv
```

---

## Logging System

The Data Collection System maintains comprehensive logs for auditing and debugging.

### Log Types

| Log Type | Location | Description |
|----------|----------|-------------|
| Session | `logs/data_collection/session/` | Overall session activity |
| API | `logs/data_collection/api/` | All API requests and responses |
| Validation | `logs/data_collection/validation/` | Data quality checks |
| Decisions | `logs/data_collection/decisions/` | Why data was included/excluded |
| Metrics | `logs/data_collection/metrics/` | Performance statistics |
| Errors | `logs/data_collection/errors/` | Error details and stack traces |
| Data Issues | `logs/data_collection/data_issues/` | Missing or problematic data |

### Log File Naming

```
session_YYYYMMDD_HHMMSS.log
api_YYYYMMDD_HHMMSS.log
```

### Viewing Logs

Logs are displayed in real-time in the GUI's log output area. Full logs are saved to the files listed above.

---

## Data Validation

The system validates all collected data before saving:

### Validation Checks

| Check Type | Description |
|------------|-------------|
| Type Validation | Ensures correct data types (dates, numbers) |
| Range Validation | Prices > 0, volumes >= 0, percentages in range |
| Temporal Validation | Dates are sequential, no gaps or duplicates |
| Completeness | Required fields are present |
| Consistency | OHLC relationship valid (Low <= Open,Close <= High) |

### Handling Validation Failures

When validation fails, the system:
1. Logs the issue to `data_issues/` log
2. Displays a warning in the GUI
3. Offers options based on your configuration:
   - **Skip**: Exclude invalid rows
   - **Fill**: Interpolate missing values
   - **Flag**: Include with warning markers
   - **Fail**: Abort the collection

Configure default behavior in settings:

```json
{
    "validation": {
        "on_error": "skip",
        "strict_mode": false
    }
}
```

---

## Output Directory Structure

```
raw_data/
├── daily/
│   ├── AAPL_daily_2020-01-01_2024-01-01.csv
│   ├── MSFT_daily_2020-01-01_2024-01-01.csv
│   └── ...
├── weekly/
│   ├── AAPL_weekly_2020-01-01_2024-01-01.csv
│   └── ...
├── fundamental/
│   ├── AAPL_income_statement_annual.csv
│   ├── AAPL_balance_sheet_quarterly.csv
│   └── ...
├── insider/
│   ├── AAPL_insider_transactions.csv
│   └── ...
├── forex/
│   ├── EURUSD_daily.csv
│   └── ...
└── options/
    ├── AAPL/
    │   ├── AAPL_20240119_calls.csv
    │   ├── AAPL_20240119_puts.csv
    │   └── ...
    ├── MSFT/
    │   └── ...
    └── ...
```

### Metadata Files

Each data file is accompanied by a metadata JSON file:

```
AAPL_daily_2020-01-01_2024-01-01.csv
AAPL_daily_2020-01-01_2024-01-01_metadata.json
```

Metadata includes:
- Collection timestamp
- API endpoint used
- Request parameters
- Row count
- Date range covered
- Validation status

---

## Caching

The system caches API responses to reduce redundant requests.

### Cache Configuration

```json
{
    "cache": {
        "enabled": true,
        "directory": ".cache/alpha_vantage",
        "ttl_hours": 24
    }
}
```

### Cache Behavior

- Cached responses are reused within the TTL period
- Cache is automatically invalidated after TTL expires
- Force refresh by disabling cache temporarily in settings

### Clearing the Cache

```bash
rm -rf .cache/alpha_vantage/*
```

---

## Rate Limiting

The system respects Alpha Vantage's rate limits to avoid API errors.

### How It Works

1. Requests are queued and processed at the configured rate
2. A token bucket algorithm ensures even distribution
3. On rate limit errors (HTTP 429), exponential backoff is applied

### Rate Limit Settings

```json
{
    "api": {
        "requests_per_minute": 75,
        "retry_attempts": 3,
        "retry_base_delay_seconds": 2
    }
}
```

### Monitoring Rate Usage

The GUI displays:
- Current queue size
- Requests made this minute
- Estimated time remaining

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| Invalid API Key | Wrong or expired key | Check settings.json |
| Rate Limit Exceeded | Too many requests | Reduce requests_per_minute |
| Symbol Not Found | Invalid ticker | Verify symbol exists on Alpha Vantage |
| No Data Available | Symbol has no data for range | Try different date range |
| Network Error | Connection issues | Check internet, retry |

### Retry Logic

Failed requests automatically retry with exponential backoff:
- Attempt 1: Immediate
- Attempt 2: Wait 2 seconds
- Attempt 3: Wait 4 seconds
- Attempt 4: Wait 8 seconds

After all retries fail, the error is logged and the symbol is skipped.

---

## Best Practices

### 1. Start Small

Test with 1-2 symbols before running large batch jobs:

```
AAPL, MSFT
```

### 2. Use Appropriate Date Ranges

- For initial testing: Use `1Y` or `compact` output
- For production: Use `full` output for maximum history

### 3. Monitor Progress

Watch the log output for:
- Successful completions
- Warnings about missing data
- Errors that need attention

### 4. Schedule Large Jobs

Large data collection jobs take time. For 100+ symbols:
- Expect 2+ hours for full historical data
- Run during off-hours if possible
- Use caching to resume interrupted jobs

### 5. Validate Output

After collection, spot-check a few files:
- Open in Excel or pandas
- Verify date ranges
- Check for gaps

---

## Troubleshooting

### Application Won't Start

1. Check Python version (3.8+ required)
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Check for import errors in console

### No Data Being Saved

1. Verify API key is valid
2. Check the logs for errors
3. Ensure output directories exist and are writable

### Slow Performance

1. Reduce concurrent symbol count
2. Check your rate limit isn't too low
3. Enable caching if disabled

### GUI Freezes

The GUI runs data collection in background threads. If it freezes:
1. Wait for current operation to complete
2. Check logs for stuck requests
3. Restart the application if needed

---

## Configuration Reference

### Full Settings Schema

```json
{
    "api": {
        "api_key": "string (required)",
        "base_url": "string (default: https://www.alphavantage.co/query)",
        "requests_per_minute": "integer (default: 75)",
        "timeout_seconds": "integer (default: 30)",
        "retry_attempts": "integer (default: 3)",
        "retry_base_delay_seconds": "integer (default: 2)"
    },
    "cache": {
        "enabled": "boolean (default: true)",
        "directory": "string (default: .cache/alpha_vantage)",
        "ttl_hours": "integer (default: 24)"
    },
    "validation": {
        "enabled": "boolean (default: true)",
        "strict_mode": "boolean (default: false)",
        "on_error": "string: skip|fill|flag|fail (default: skip)"
    },
    "output": {
        "base_directory": "string (default: raw_data)",
        "create_metadata": "boolean (default: true)",
        "overwrite_existing": "boolean (default: false)"
    },
    "logging": {
        "level": "string: DEBUG|INFO|WARNING|ERROR (default: INFO)",
        "directory": "string (default: logs/data_collection)"
    }
}
```

---

## Integration with Backtesting

Once data is collected, it's ready for use with the backtesting framework:

1. **Daily/Weekly Data**: Used directly by strategies for price analysis
2. **Technical Indicators**: Pre-calculated indicators save computation time
3. **Fundamental Data**: Used by fundamental-aware strategies
4. **Insider Data**: Used for sentiment and conviction signals
5. **Options Data**: Used for volatility analysis and options strategies

### Example Workflow

```bash
# 1. Collect data
python apps/data_collection_gui.py

# 2. Run backtest
python run_gui.py
# Select securities from raw_data/daily/
# Choose strategy and run
```

---

## Getting Help

If you encounter issues:

1. Check this documentation
2. Review the logs in `logs/data_collection/`
3. Verify your Alpha Vantage subscription and API key
4. Check Alpha Vantage's [API documentation](https://www.alphavantage.co/documentation/)
