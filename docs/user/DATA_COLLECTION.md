# Data Collection Guide

This guide explains how to use the Data Collection System to gather historical market data for backtesting.

## Overview

The Data Collection System fetches historical market data from Alpha Vantage. It provides:
- Daily and weekly OHLCV (Open, High, Low, Close, Volume) data
- 50+ pre-calculated technical indicators
- Fundamental data (financial statements)
- Insider trading activity
- Forex currency pair data
- Options data

All data is saved as CSV files in the `raw_data/` directory.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `customtkinter>=5.2.0`, `requests>=2.28.0`

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

Get a free API key at [alphavantage.co](https://www.alphavantage.co/support/#api-key).

### 3. Launch the Application

```bash
python data_collection_gui.py
```

---

## API Subscription Tiers

The system is optimized for Alpha Vantage's Premium tier (75 requests/minute). Adjust settings based on your plan:

| Tier | Requests/Minute | Setting |
|------|-----------------|---------|
| Free | 5 | 5 |
| Premium | 75 | 75 |
| Enterprise | 150+ | Match your limit |

---

## Data Types

### Daily Data

Daily OHLCV data with optional technical indicators.

**Configuration Options:**

| Option | Description |
|--------|-------------|
| Symbols | Comma-separated list (e.g., `AAPL, MSFT, GOOGL`) |
| Date Range | Preset (1Y, 5Y, 10Y, Max) or custom dates |
| Output Size | `compact` (100 days) or `full` (20+ years) |
| Adjusted | Use split/dividend adjusted prices |
| Technical Indicators | Select from 50+ indicators |

**Available Indicators:**

- **Trend**: SMA, EMA, WMA, DEMA, TEMA, KAMA, VWAP
- **Momentum**: RSI, STOCH, WILLR, ADX, CCI, AROON, MOM, ROC
- **Volatility**: ATR, NATR, BBANDS
- **Volume**: AD, ADOSC, OBV

**Output:**
```
raw_data/daily/AAPL.csv
```

---

### Weekly Data

Weekly OHLCV data with the same indicator options as daily.

**Output:**
```
raw_data/weekly/AAPL.csv
```

---

### Fundamental Data

Company financial statements and metrics.

**Available Data:**

| Category | Description |
|----------|-------------|
| Income Statement | Revenue, earnings, expenses |
| Balance Sheet | Assets, liabilities, equity |
| Cash Flow | Operating, investing, financing flows |
| Earnings | EPS, estimates, surprises |
| Company Overview | Sector, industry, market cap |

**Output:**
```
raw_data/fundamentals/AAPL_income_statement.csv
raw_data/fundamentals/AAPL_balance_sheet.csv
```

---

### Insider Data

Insider trading transactions (Form 4 filings).

**Data Collected:**
- Transaction date and type (buy/sell)
- Insider name and title
- Shares traded and price
- Value of transaction
- Shares owned after transaction

**Output:**
```
raw_data/insider_transactions/AAPL_insider.csv
```

---

### Forex Data

Foreign exchange rate data for currency pairs.

**Popular Pairs:**
- EUR/USD, GBP/USD, USD/JPY, USD/CHF
- AUD/USD, USD/CAD, NZD/USD
- EUR/GBP, EUR/JPY, GBP/JPY

**Timeframes:**
- Intraday (1min-60min)
- Daily, Weekly, Monthly

**Output:**
```
raw_data/forex/EURUSD.csv
```

---

### Options Data

Historical options chain data.

**Data Collected:**
- Strike price and expiration date
- Contract type (call/put)
- Bid/Ask prices
- Volume and open interest
- Implied volatility
- Greeks (delta, gamma, theta, vega)

**Output (organized by ticker):**
```
raw_data/options/AAPL/AAPL_20240119_calls.csv
raw_data/options/AAPL/AAPL_20240119_puts.csv
```

---

## Ticker Presets

The system includes pre-configured ticker groups in `config/data_collection/tickers.json`.

**Available Presets:**

| Preset | Description |
|--------|-------------|
| All Stocks | Complete stock universe (165+ tickers) |
| Large Cap Only | Large cap stocks across all sectors |
| All ETFs | Complete ETF universe (46 tickers) |
| Tech Sector | Electronic Technology + Technology Services |
| Healthcare Sector | Healthcare stocks only |
| Finance Sector | Financial stocks only |
| US Tradeable | US-listed stocks only |
| Forex Pairs | Common currency pairs |

Use "Load Preset" in the GUI to quickly select groups of tickers.

---

## Output Directory Structure

```
raw_data/
├── daily/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ...
├── weekly/
│   ├── AAPL.csv
│   └── ...
├── fundamentals/
│   ├── AAPL_income_statement.csv
│   ├── AAPL_balance_sheet.csv
│   └── ...
├── insider_transactions/
│   ├── AAPL_insider.csv
│   └── ...
├── forex/
│   ├── EURUSD.csv
│   └── ...
└── options/
    ├── AAPL/
    │   ├── AAPL_20240119_calls.csv
    │   └── AAPL_20240119_puts.csv
    └── ...
```

---

## Data Validation

The system validates all collected data before saving:

| Check | Description |
|-------|-------------|
| Type Validation | Correct data types (dates, numbers) |
| Range Validation | Prices > 0, volumes >= 0 |
| Temporal Validation | Sequential dates, no gaps |
| Completeness | Required fields present |
| Consistency | OHLC relationship valid (Low <= High) |

**When validation fails:**
- Issue logged to `logs/data_collection/data_issues/`
- Warning displayed in GUI
- Default behavior: Skip invalid rows

---

## Caching

API responses are cached to reduce redundant requests.

**Configuration:**
```json
{
    "cache": {
        "enabled": true,
        "directory": ".cache/alpha_vantage",
        "ttl_hours": 24
    }
}
```

**Clear cache:**
```bash
rm -rf .cache/alpha_vantage/*
```

---

## Rate Limiting

The system respects Alpha Vantage's rate limits:

1. Requests are queued and processed at the configured rate
2. On rate limit errors (HTTP 429), exponential backoff is applied
3. Retries: 2s, 4s, 8s delays

**Monitor in GUI:**
- Current queue size
- Requests made this minute
- Estimated time remaining

---

## Logging

Logs are saved to `logs/data_collection/`:

| Log Type | Description |
|----------|-------------|
| Session | Overall session activity |
| API | All API requests and responses |
| Validation | Data quality checks |
| Errors | Error details and stack traces |
| Data Issues | Missing or problematic data |

---

## Troubleshooting

### "Invalid API Key"
- Check `settings.json` has correct API key
- Verify key at [alphavantage.co](https://www.alphavantage.co)

### "Rate Limit Exceeded"
- Reduce `requests_per_minute` in settings
- Wait for rate limit to reset
- The client handles this automatically

### "Symbol Not Found"
- Verify symbol exists on Alpha Vantage
- Check for correct format (e.g., `AAPL` not `AAPL.US`)

### "No Data Available"
- Symbol may have no data for requested range
- Try different date range
- Some symbols have limited history

### Slow Performance
- Reduce concurrent symbol count
- Enable caching if disabled
- Check your API rate limit isn't too low

---

## Best Practices

### 1. Start Small
Test with 1-2 symbols before running large batch jobs.

### 2. Use Appropriate Date Ranges
- For testing: Use `1Y` or `compact` output
- For production: Use `full` for maximum history

### 3. Monitor Progress
Watch the log output for warnings and errors.

### 4. Schedule Large Jobs
For 100+ symbols:
- Expect 2+ hours for full historical data
- Run during off-hours
- Use caching to resume interrupted jobs

### 5. Validate Output
After collection, spot-check a few files:
- Open in Excel or pandas
- Verify date ranges
- Check for gaps

---

## Integration with Backtesting

Once data is collected, it's ready for use:

```bash
# 1. Collect data
python data_collection_gui.py

# 2. Run backtest
python run_gui.py
# → Select securities from raw_data/
# → Choose strategy and run
```
