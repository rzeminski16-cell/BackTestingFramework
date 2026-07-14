---
tags:
  - usability/running-analysis
  - guide
---

# Rule Tester

Answer "what if I had filtered these trades?" without re-running backtests:
load a trade log, build entry- or exit-rules from price/indicator features,
and compare before/after performance side by side.

---

## Launching

```bash
python ctk_rule_tester_gui.py        # or the "Rule Tester" card in the launcher
```

## Prerequisites

- One or more **trade-log CSVs** from backtests.
- The matching **raw price data** under `raw_data/daily/` — the tool
  auto-loads price files for the tickers found in the trade log so rules can
  reference indicator values around each entry/exit.

## Workflow

1. **Load trade logs.** Ticker names are extracted from filenames and the
   corresponding price data is loaded automatically.
2. **Pick up to 3 features** from the price data (e.g. `rsi_14`, `atr_14`,
   `volume`). A histogram of each feature's value at trade entry shows where
   your winners and losers actually sit before you pick thresholds.
3. **Build rules** — comparisons against thresholds (e.g. `rsi_14 < 70`),
   combined, in one of two modes:
   - **Entry rules**: keep only trades whose entry bar satisfied the rule —
     simulates a stricter entry filter.
   - **Exit rules**: evaluate registered strategy exit-rule variants.
4. **Set the lookback** — rules can look N bars *before* the entry
   (e.g. "RSI was below 70 at any point in the 5 bars before entry").
5. **Compare.** The before/after panel shows trade count, win rate, total
   and average P/L, and the stable metrics (RAR%, R², R-Cubed — all
   annualized on the calendar-daily convention).
6. **Export** — filtered trades and a comparison report as CSV/Excel.

## Interpreting honestly

The Rule Tester evaluates a rule **on the same trades that inspired it** —
this is in-sample exploration, not validation. Treat improvements as
hypotheses:

- Prefer rules that remove a *few* clearly bad trades over rules that
  reshuffle everything (check the trade count delta).
- A rule worth keeping should then be added to the strategy itself and
  validated with [[Walk-Forward Optimisation]] or the
  [[../../MODELLING_USER_GUIDE|Modelling & Evaluation]] stage, which applies
  purged out-of-sample testing.

## Related

- [[Edge Analysis]] — validates entry quality with E-ratios
- [[Metrics Glossary]] — RAR%, R-Cubed definitions
