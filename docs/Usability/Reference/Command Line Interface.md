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
