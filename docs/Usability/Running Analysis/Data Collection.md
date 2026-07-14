---
tags:
  - usability/running-analysis
  - guide
---

# Data Collection

The unified Alpha Vantage collection system: fetches, validates, and writes
every data family the framework consumes — daily/weekly prices with
indicators, fundamentals, forex, benchmarks, macro, commodities, corporate
actions — into `raw_data/`, with caching, rate limiting, and logging.

---

## Launching

```bash
python apps/data_collection_gui.py     # or the "Data Collection" launcher card
```

## One-time setup

1. Copy the settings template and add your Alpha Vantage API key:

   ```bash
   cp config/data_collection/settings.json.template config/data_collection/settings.json
   # edit config/data_collection/settings.json -> api.api_key
   ```

   `settings.json` is **gitignored on purpose** — it holds your key. Never
   commit it. The file also sets the per-minute rate limit (match your API
   tier), retry policy, cache directory/expiry, and validation thresholds.

2. Maintain your universe in `config/data_collection/tickers.json` —
   symbols, exchanges, and which data families to collect per symbol.

## Collecting

Pick the data families and symbols in the GUI and run. The system:

- **Caches** raw API responses under `cache/alpha_vantage/` (expiry is
  configurable; the cache is rebuildable and stays out of git), so re-runs
  only hit the API for missing or stale data.
- **Rate-limits** to your configured requests/minute and retries transient
  failures with exponential backoff.
- **Validates** what it writes: gap detection (max trading-day gaps),
  minimum history length, outlier checks, OHLCV consistency.
- **Logs** every run to a per-run log file so failures are auditable.

Output lands in `raw_data/<family>/`, e.g. `raw_data/daily/AAPL_daily.csv`
with the pre-computed indicator columns strategies declare in
`required_columns()`.

## Where each family is used

| Family | Consumed by |
|---|---|
| daily / weekly | Backtesting, optimisation, every analysis tool |
| forex | Multi-currency conversion in the engines (`raw_data/forex`) |
| benchmarks | Benchmark comparison sheets in reports |
| fundamentals | Fundamental entry rules + backtest analyzer features |
| macro / commodities / breadth | Data Preparation run packages (modelling features) |

## Troubleshooting

- **"Missing required columns" when backtesting** — the strategy needs an
  indicator column your collection didn't include; re-collect that family
  with the indicator enabled (the error message lists exactly which).
- **Rate-limit errors** — lower `rate_limit_per_minute` to your API tier.
- **Stale data** — the cache serves anything younger than
  `cache_expiry_hours`; lower it or clear `cache/alpha_vantage/`.

## Related

- [[Adding a New Security]] — the manual path for one-off symbols
- [[../../DATA_PREPARATION|Data Preparation]] — builds modelling run packages from collected data
