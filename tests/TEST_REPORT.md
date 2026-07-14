# BackTestingFramework — Test Report

**Date:** 2026-07-13
**Environment:** Python 3.11, fresh dependency install from `requirements.txt`

## Summary

```
866 passed, 2 warnings (Python 3.11)
```

All 35 test modules pass, covering: engines (single-security, portfolio,
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
