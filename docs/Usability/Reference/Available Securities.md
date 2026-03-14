---
tags:
  - usability/reference
  - data
---

# Available Securities

Securities included with the framework and their data coverage.

---

## Data Location

All daily price data lives in `raw_data/daily/` as CSV files. Each file is named `{SYMBOL}.csv`.

---

## Checking Available Securities

To see all available securities:

```bash
ls raw_data/daily/*.csv
```

Or in Python:

```python
from pathlib import Path
securities = [f.stem for f in Path('raw_data/daily').glob('*.csv')]
print(sorted(securities))
```

---

## Data Format

Each CSV contains at minimum:

| Column | Description |
|---|---|
| `date` | Trading date |
| `close` | Closing price |

Most files also include `open`, `high`, `low`, `volume`, and pre-calculated indicators like `atr_14_atr`, `mfi_14_mfi`, etc.

---

## Additional Data

| Directory | Type | Format |
|---|---|---|
| `raw_data/weekly/` | Weekly OHLCV | CSV |
| `raw_data/fundamentals/` | EPS, P/E, revenue growth, FCF | CSV |
| `raw_data/insider_transactions/` | Insider buy/sell records | CSV |
| `raw_data/options/` | Options chain data (IV, skew) | CSV |
| `raw_data/forex/` | FX rate time series | CSV |

---

## Security Metadata

`config/security_metadata.json` maps tickers to their currency and exchange:

```json
{
  "AAPL": { "currency": "USD", "exchange": "NASDAQ", "name": "Apple Inc" }
}
```

This is used by the [[Data Layer]] for multi-currency portfolio backtests.

---

## Pre-Built Baskets

Security groupings for portfolio testing live in `config/baskets/`. See [[Creating a Security Basket]] for creating your own.

---

## Next Steps

- [[Adding a New Security]] — add data for a new ticker
- [[Adding a New Indicator]] — add indicators to existing data
