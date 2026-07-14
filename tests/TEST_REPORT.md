# BackTestingFramework — Test Report

**Date:** 2026-07-13
**Environment:** Python 3.11, fresh dependency install from `requirements.txt`

## Summary

```
975 passed, 2 warnings (Python 3.11)
```

All 44 test modules pass, covering: engines (single-security, portfolio,
integration), strategies (AlphaTrend family, short-only base, random control),
core & optimization metrics, currency conversion, data layer & collection,
data preparation, modelling & evaluation, Monte Carlo, pattern analysis,
rejection explorer, report generation and charts, vulnerability scoring/trace,
and walk-forward optimization.

Added 2026-07-13 (P0 correctness fixes):

- `test_short_accounting.py` — ledger-vs-trade-log invariant
  (`final_equity − initial == Σ trade.pl`) for LONG and SHORT on both
  engines, with commission/slippage/partial exits; direction-aware
  mark-to-market and break-even stop checks.
- `test_stable_metrics_cadence.py` — RAR% cadence invariance (calendar-daily
  vs trading-day vs per-trade sampling) and agreement with the Modelling
  stage's Adjusted RAR.
- `test_metric_consistency.py` — golden cross-layer tests pinning the
  report adapter's unit contract (win_rate fraction vs percent, dollar
  win/loss aggregates) against the centralized metrics module, so any
  re-implemented formula that drifts fails CI.

Added 2026-07-14 (P2 analytics depth):

- `test_engine_parity.py` — the single-security and portfolio engines must
  produce identical trades and equity paths on the same inputs (LONG and
  SHORT, with partials, trailing stops, and costs).
- `test_risk_metrics.py` — exposure, VaR/CVaR, rolling helpers, and
  engine-tracked MAE/MFE per trade.
- `test_execution_realism.py` — intrabar stop/TP triggers with gap-aware
  fills (both engines) and next-bar-open execution timing.
- `test_deflated_sharpe.py` — PSR/DSR math (a best-of-200-noise-strategies
  winner must NOT look significant) and WindowResult integration.
- `test_monte_carlo_upgrades.py` — annualized per-path Sharpe/CAGR/Calmar
  distributions and the daily-returns pool loader.
- `test_benchmark.py` gained benchmark-relative max-drawdown coverage.

Added 2026-07-14 (P3 product polish):

- `test_cli.py` — the `btf` CLI: parser/subcommands, strategy registry,
  hermetic end-to-end single and portfolio backtests on synthetic data
  (trade-log CSV + metrics JSON assertions), next-bar-open flag, and
  Monte Carlo from both a trade log and a daily equity curve.

Added 2026-07-14 (P4):

- `test_p4_features.py` — strategy auto-discovery (all known strategies
  found, no abstracts), the `new-strategy` scaffold (generates a working,
  instantiable strategy; correct stop side; overwrite protection), the
  Parquet store (ingest, loader preference, CSV/Parquet equivalence,
  staleness fallback, validation warnings), and the `btf signals` bridge
  (ENTER / EXIT / HOLDING / FLAT / ERROR classification).
- `test_engine_properties.py` — hypothesis property tests: ledger
  conservation, finite self-consistent equity curves, stop monotonicity
  against adversarial adjustment proposals, partial-exit quantity
  invariants, intrabar fill bounds, and Monte Carlo seed reproducibility
  across randomly generated scenarios.

The two warnings are benign numpy `RuntimeWarning`s from intentional
edge-case tests (correlation of identical values).

## How to run

```bash
pip install -r requirements.txt pytest
python -m pytest tests/ -q
```

## Known gaps (see docs/SYSTEM_EVALUATION.md)

- No CI pipeline runs this suite automatically on push (roadmap P1).
- Legacy `backtesting/` package has no coverage (slated for removal, P1).

## Historical note

An earlier version of this report (2025-12-28, branch
`claude/test-framework-DuNLP`) recorded 39 passing tests and the fix of a
`CommissionConfig.percent` AttributeError in `single_security_engine.py`.
The suite has since grown ~21×.
