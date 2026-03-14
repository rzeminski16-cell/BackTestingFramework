---
tags:
  - usability/extending
  - data
---

# Adding a New Security

How to prepare and add historical data for a new security.

---

## Step 1: Prepare the CSV

Create a CSV file in `raw_data/daily/` named after the ticker symbol:

```
raw_data/daily/TSLA.csv
```

### Required Columns

| Column | Type | Description |
|---|---|---|
| `date` | Date (YYYY-MM-DD) | Trading date |
| `close` | Float | Closing price |

### Recommended Columns

| Column | Type | Description |
|---|---|---|
| `open` | Float | Opening price |
| `high` | Float | Intraday high |
| `low` | Float | Intraday low |
| `volume` | Integer | Trading volume |

### Strategy-Specific Columns

Depending on which strategy you want to run, you'll need additional pre-calculated indicator columns. For example, `BaseAlphaTrendStrategy` requires:

| Column | Description |
|---|---|
| `atr_14_atr` | 14-period Average True Range |
| `mfi_14_mfi` | 14-period Money Flow Index |

> [!tip] Check Required Columns
> Call `strategy.required_columns()` on any strategy instance to see exactly which columns it needs.

---

## Step 2: Validate Data Quality

Ensure:
- Dates are sorted in ascending order (oldest first)
- No duplicate dates
- No missing closing prices
- Indicator columns have values after their warmup period (initial NaN rows are acceptable)

The `DataLoader` will validate columns at load time and raise clear errors for any missing columns.

---

## Step 3: Add Security Metadata (Optional)

If you want the system to know this security's currency and other attributes, add an entry to `config/security_metadata.json`:

```json
{
  "TSLA": {
    "currency": "USD",
    "exchange": "NASDAQ",
    "name": "Tesla Inc"
  }
}
```

This is used for multi-currency portfolio backtests — the [[Data Layer]] uses this to look up FX rates.

---

## Step 4: Add to Baskets (Optional)

To include the security in portfolio backtests, add its ticker to a basket file in `config/baskets/`. See [[Creating a Security Basket]].

---

## Data Sources

The framework includes `yfinance` for downloading data. For indicators, you can use the `ta` library to pre-calculate them and save to CSV.

Additional data directories:
- `raw_data/weekly/` — weekly timeframe data
- `raw_data/fundamentals/` — fundamental financial data
- `raw_data/insider_transactions/` — insider trading records
- `raw_data/options/` — options chain data
- `raw_data/forex/` — FX rate data

---

## Next Steps

- [[Adding a New Indicator]] — add indicators to existing securities
- [[Creating a Security Basket]] — group securities for portfolio testing
- [[Available Securities]] — see what's already included
