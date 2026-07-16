---
tags:
  - usability/reference
  - cli
---

# Command Line Interface (btf)

Every core flow is scriptable via `python -m btf` (or plain `btf` after
`pip install -e .`). Commands are headless, print a readable summary, can
emit machine-readable outputs (`--json`, `--trades-csv`), and exit non-zero
on failure — suitable for shell scripts, cron, and CI.

```bash
python -m btf --help
python -m btf <command> --help
```

---

## btf list

```bash
python -m btf list                 # everything
python -m btf list strategies      # registered strategy classes
python -m btf list securities      # symbols found in raw_data/daily
python -m btf list baskets
python -m btf list benchmarks
```

## btf backtest

One symbol runs the single-security engine; several symbols (or `--basket`)
run the portfolio engine.

```bash
# Single security, with Excel report + trade log + metrics JSON
python -m btf backtest --strategy AlphaTrendV1Strategy --symbols AAPL \
    --start 2020-01-01 --end 2024-12-31 \
    --report --trades-csv out/trades.csv --json out/metrics.json

# Portfolio from a basket
python -m btf backtest --strategy AlphaTrendV1Strategy --basket Technology

# Execution-realism variants (see Configuration Options)
python -m btf backtest --strategy AlphaTrendV1Strategy --symbols AAPL \
    --intrabar-stops --next-bar-open

# Portfolio: randomise which same-day signals get scarce capital
# (seeded + date-salted, so one seed reproduces the whole run)
python -m btf backtest --strategy AlphaTrendV1Strategy --basket Technology \
    --randomize-signal-order --signal-seed 42

# Strategy parameter overrides (repeatable)
python -m btf backtest --strategy AlphaTrendV1Strategy --symbols AAPL \
    --param atr_multiplier=2.5 --param ma_length=20
```

Key flags: `--capital`, `--commission` (fraction; add `--fixed-commission`
for a flat amount), `--slippage` (percent), `--base-currency`, `--data-dir`,
`--no-fx` (single-currency data outside the security registry).

## btf optimize

Walk-forward optimization for one security, using
`config/optimization_config.yaml` for search settings. Prints per-window OOS
metrics including the **Deflated Sharpe Ratio** (probability the winning
in-sample Sharpe survives its own parameter search; > 0.95 is a pass).

```bash
python -m btf optimize --strategy AlphaTrendV1Strategy --symbol AAPL \
    --mode rolling --report
```

## btf montecarlo

Bootstrap Monte Carlo from a trade log (per-trade returns or R-multiples) or
from an equity curve (daily returns — calendar-time risk).

```bash
python -m btf montecarlo --trade-log logs/<run>/trades.csv \
    --simulations 5000 --steps 200 --risk 0.01 --json mc.json

python -m btf montecarlo --daily-curve logs/<run>/equity_curve.csv \
    --steps 252 --block-size 10
```

`--periods-per-year` (your trades/year, or 252 for daily pools) enables the
annualized per-path Sharpe/CAGR/Calmar distributions.

## btf ingest

Convert raw CSVs into typed, validated Parquet files (written next to the
CSVs). Loads are several times faster and schema problems (unparseable
dates, duplicate dates, non-positive closes) are reported once at ingest
instead of surfacing mid-backtest. The `DataLoader` automatically prefers a
Parquet file while it is at least as new as its CSV, so nothing else changes.

```bash
python -m btf ingest --data-dir raw_data/daily
python -m btf ingest --data-dir raw_data/daily --force   # rebuild all
```

## btf signals — live-paper bridge

Report the strategy's action on the **latest collected bar** per symbol, by
replaying the full backtest through the real engine — research and paper
trading share one code path, so a signal here is exactly what the backtest
would have done today. Designed for a cron job after data collection.

```bash
python -m btf signals --strategy AlphaTrendV1Strategy --basket Technology \
    --json today_signals.json
```

Per-symbol actions: `ENTER` (entry signal fired on the latest bar, with
side/price/stop), `EXIT` (position closed on the latest bar with the
reason), `HOLDING` (position still open, with entry date/price and current
stop), `FLAT`, or `ERROR`. This is a stateless v1 — each run replays from
history; a persistent paper account (day-over-day fills and P&L) is the
natural next increment.

## btf new-strategy

Scaffold a ready-to-edit strategy file. Strategies are **auto-discovered**
(any concrete `BaseStrategy` subclass in `strategies/`, plus
`btf.strategies` entry points from installed packages), so no registry edit
is needed.

```bash
python -m btf new-strategy MeanReversionStrategy --indicators atr_14 rsi_14
python -m btf new-strategy PanicFadeStrategy --direction short
```

The generated file instantiates and backtests immediately (entries are
TODO), includes risk-based position sizing, ATR stops on the correct side,
and the parameter documentation the base class requires.

## btf dashboard

```bash
python -m btf dashboard                                    # run picker
python -m btf dashboard --run processed_data/runs/<run>/<model_run>
```

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Run failed (engine/report error — details on stderr) |
| 2 | Bad arguments (unknown strategy, missing data, invalid input) |
