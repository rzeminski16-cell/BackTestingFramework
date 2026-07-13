# BackTestingFramework — Test Report

**Date:** 2026-07-13
**Environment:** Python 3.11, fresh dependency install from `requirements.txt`

## Summary

```
829 passed, 2 warnings in 144.10s (0:02:24)
```

All 32 test modules pass, covering: engines (single-security, portfolio,
integration), strategies (AlphaTrend family, short-only base, random control),
core & optimization metrics, currency conversion, data layer & collection,
data preparation, modelling & evaluation, Monte Carlo, pattern analysis,
rejection explorer, report generation and charts, vulnerability scoring/trace,
and walk-forward optimization.

The two warnings are benign numpy `RuntimeWarning`s from intentional
edge-case tests (correlation of identical values).

## How to run

```bash
pip install -r requirements.txt pytest
python -m pytest tests/ -q
```

## Known gaps (see docs/SYSTEM_EVALUATION.md)

- No test asserts equity-curve ↔ trade-ledger consistency for SHORT trades
  (the current short cash-ledger bug, Evaluation §2.2, is not covered).
- No cross-tool consistency test for RAR% annualisation (Evaluation §2.3).
- No CI pipeline runs this suite automatically on push.

## Historical note

An earlier version of this report (2025-12-28, branch
`claude/test-framework-DuNLP`) recorded 39 passing tests and the fix of a
`CommissionConfig.percent` AttributeError in `single_security_engine.py`.
The suite has since grown ~21×.
