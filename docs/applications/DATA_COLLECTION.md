# Data Collection

The Data Collection application gathers historical market data from Alpha Vantage API for use in backtesting.

---

## Launching

```bash
python apps/data_collection_gui.py
```

Or access from the main launcher by clicking "Data Collection".

---

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA COLLECTION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TICKER SELECTION                                                          │
│  ────────────────                                                          │
│  Preset: [All Stocks (165)       ▼]     [Load]                            │
│                                                                             │
│  Or select individually:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ☑ AAPL   Apple Inc.                                                │   │
│  │ ☑ MSFT   Microsoft Corporation                                     │   │
│  │ ☑ GOOGL  Alphabet Inc.                                             │   │
│  │ ☐ AMZN   Amazon.com Inc.                                           │   │
│  │ ...                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Selected: 165 tickers                                                     │
│                                                                             │
│  DATA TYPES                                                                │
│  ──────────                                                                │
│  ☑ Daily Prices (OHLCV + 50+ indicators)                                  │
│  ☐ Weekly Prices                                                           │
│  ☑ Fundamental Data (financials, ratios)                                  │
│  ☐ Insider Transactions                                                    │
│  ☐ Options Chains                                                          │
│  ☐ Forex Pairs                                                             │
│                                                                             │
│  OPTIONS                                                                   │
│  ───────                                                                   │
│  ○ Full history (all available data)                                      │
│  ● Update only (fetch new data since last update)                         │
│                                                                             │
│  [Start Collection]                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Types

### Daily Prices

The primary data type for backtesting:

```
Output: raw_data/daily/{TICKER}_daily.csv

Contents:
├── Date columns: date
├── Price columns: open, high, low, close, adjusted_close
├── Volume: volume
└── 50+ Technical Indicators:
    ├── Moving Averages: sma_10, sma_20, sma_50, sma_200, ema_12, ema_26, ema_50
    ├── Momentum: rsi_14, macd, macd_signal, macd_hist, stoch_k, stoch_d
    ├── Volatility: atr_14, bollinger_upper, bollinger_middle, bollinger_lower
    ├── Volume: obv, adl, mfi_14
    └── Trend: adx, plus_di, minus_di, cci_20
```

### Weekly Prices

Longer timeframe data:

```
Output: raw_data/weekly/{TICKER}_weekly.csv

Contents: Same structure as daily, aggregated to weekly bars
```

### Fundamental Data

Financial statements and ratios:

```
Output: raw_data/fundamentals/{TICKER}/
├── income_statement.csv      Revenue, earnings, margins
├── balance_sheet.csv         Assets, liabilities, equity
├── cash_flow.csv             Operating, investing, financing
├── earnings.csv              Quarterly EPS, surprises
└── overview.json             Company info, ratios, metrics
```

### Insider Transactions

Insider buying and selling:

```
Output: raw_data/insider_transactions/{TICKER}.csv

Contents:
├── transaction_date
├── filing_date
├── insider_name
├── insider_title
├── transaction_type (buy/sell)
├── shares
└── value
```

### Options Chains

Historical options data:

```
Output: raw_data/options/{TICKER}/{YEAR}/
├── calls/
│   └── {expiration_date}.csv
└── puts/
    └── {expiration_date}.csv

Contents: strike, bid, ask, volume, open_interest, implied_volatility
```

### Forex Pairs

Currency exchange rates:

```
Output: raw_data/forex/{PAIR}.csv

Example pairs: USDGBP, USDEUR, USDJPY
Use for: Converting P/L to different currencies
```

---

## Ticker Presets

Pre-configured ticker lists in `config/data_collection/tickers.json`:

| Preset | Description | Count |
|--------|-------------|-------|
| All Stocks | Complete US equity universe | 165+ |
| All ETFs | Exchange-traded funds | 46 |
| Large Cap | Mega and large cap stocks | 50 |
| Tech Sector | Technology companies | 35 |
| Healthcare | Healthcare companies | 25 |
| Financials | Financial sector | 30 |
| Custom | User-defined list | Variable |

---

## Collection Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COLLECTION PROGRESS                                      │
│                                                                             │
│  Status: Collecting data...                                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Progress: ████████████████████░░░░░░░░░░  65%                     │   │
│  │  Current: MSFT (108 of 165)                                        │   │
│  │  Data type: Daily prices                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Statistics:                                                               │
│  ├── Completed: 107 tickers                                                │
│  ├── Remaining: 58 tickers                                                 │
│  ├── Errors: 2 (see log)                                                   │
│  └── Estimated time: 12 minutes                                            │
│                                                                             │
│  API Usage:                                                                │
│  ├── Requests this minute: 45 / 75                                        │
│  └── Rate limit: Premium (75 req/min)                                     │
│                                                                             │
│  Recent Activity:                                                          │
│  ├── ✓ AAPL: 2,518 bars fetched                                           │
│  ├── ✓ GOOGL: 2,518 bars fetched                                          │
│  ├── ✗ BRK.A: API error (symbol format)                                   │
│  └── ... MSFT: Fetching...                                                 │
│                                                                             │
│  [Pause]  [Cancel]                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Configuration

### Alpha Vantage Setup

1. Get an API key from [Alpha Vantage](https://www.alphavantage.co/)
2. Configure in `config/data_collection/settings.json`:

```json
{
  "api_key": "YOUR_API_KEY",
  "rate_limit": 75,
  "timeout": 30
}
```

### Rate Limits

| Tier | Requests/Minute | Best For |
|------|-----------------|----------|
| Free | 5 | Testing, small updates |
| Premium | 75 | Regular data collection |
| Enterprise | Higher | Large-scale operations |

The collector automatically handles rate limiting.

---

## Update vs Full History

### Full History

Downloads all available historical data:
- Use for: Initial setup, new tickers
- Time: ~30 seconds per ticker
- Data: 20+ years where available

### Update Only

Fetches only new data since last collection:
- Use for: Daily/weekly updates
- Time: ~5 seconds per ticker
- Data: New bars only

```
Last collection: 2024-01-10
Today: 2024-01-15

Update mode fetches: Jan 11, 12, 15 (3 new bars)
Full mode fetches: All 2,500+ bars (redundant)
```

---

## Output Structure

```
raw_data/
├── daily/
│   ├── AAPL_daily.csv
│   ├── MSFT_daily.csv
│   ├── GOOGL_daily.csv
│   └── ...
│
├── weekly/
│   ├── AAPL_weekly.csv
│   └── ...
│
├── fundamentals/
│   ├── AAPL/
│   │   ├── income_statement.csv
│   │   ├── balance_sheet.csv
│   │   ├── cash_flow.csv
│   │   ├── earnings.csv
│   │   └── overview.json
│   └── ...
│
├── insider_transactions/
│   ├── AAPL.csv
│   └── ...
│
├── options/
│   ├── AAPL/
│   │   └── 2024/
│   │       ├── calls/
│   │       └── puts/
│   └── ...
│
└── forex/
    ├── USDGBP.csv
    └── ...
```

---

## Data Quality

### Automatic Validation

The collector validates data as it's fetched:

```
Validation Checks:
├── Date sequence: No gaps in trading days
├── Price sanity: Open/High/Low/Close relationships valid
├── Volume: Non-negative values
├── Indicators: No NaN values in critical columns
└── File integrity: CSV properly formatted
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing dates | Market holidays, halts | Expected behavior |
| Symbol not found | Delisted or wrong format | Check ticker validity |
| Partial data | API timeout | Re-run collection |
| Rate limit errors | Too many requests | Wait or upgrade tier |

---

## Logging

Collection logs are saved to `logs/data_collection/`:

```
logs/data_collection/
├── collection_2024-01-15_14-30-00.log    Full session log
├── errors_2024-01-15_14-30-00.csv        Failed tickers with reasons
└── summary_2024-01-15_14-30-00.json      Collection statistics
```

---

## Best Practices

### Initial Setup

1. Start with a small preset (Large Cap) to test
2. Verify data quality before expanding
3. Collect daily prices first, fundamentals later

### Regular Updates

1. Run updates weekly or before major backtests
2. Use "Update only" mode for efficiency
3. Check error logs for failed tickers

### Before Backtesting

1. Verify required columns exist for your strategy
2. Check date range covers your test period
3. Ensure no gaps in critical securities

---

## Related Documentation

- [Quick Start](../overview/QUICK_START.md) — Getting started with backtesting
- [Backtesting Concepts](../concepts/BACKTESTING.md) — How data is used
- [Configuration Reference](../reference/CONFIGURATION.md) — All settings
