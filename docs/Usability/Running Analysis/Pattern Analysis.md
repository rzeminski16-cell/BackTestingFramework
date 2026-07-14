---
tags:
  - usability/running-analysis
  - guide
---

# Pattern Analysis

Measures the **density of MA-crossover signals in the windows before each
trade entry**, split by trade outcome — do winning trades tend to fire after
quiet periods or after clusters of signals? Uses the same crossover logic as
the AlphaTrend family (`MA(L)` shifted by `offset` crossing the unshifted
`MA(L)`), so the findings map directly onto those strategies.

---

## Launching

```bash
python ctk_pattern_analysis_gui.py     # or the "Pattern Analysis" launcher card
# headless:
python tools/run_pattern_analysis.py --help
```

## Prerequisites

- A **trade-log CSV** from a backtest.
- The matching **raw price data** (`raw_data/daily/` by default) for every
  symbol in the log.

## Workflow

1. **Select the trade log** and the price-data directory.
2. **Configure lookback windows** — the bar windows before each entry in
   which crossover signals are counted (e.g. 5/10/20 bars).
3. **Build MA/offset combos** — which moving-average lengths and shift
   offsets to scan for crossovers.
4. **Run.** Analysis executes on a worker thread; the right-hand summary
   card fills in when it finishes.
5. **Open the Excel report** (default `pattern_analysis_report.xlsx`) — a
   multi-sheet breakdown of signal density per window, per combo, split by
   outcome class (winners vs losers), with per-trade detail sheets.

## Reading the results

You are looking for **separation**: a window/combo where the signal density
distribution for winning trades is visibly different from losing trades. If
winners consistently enter after (say) 2+ crossovers in the prior 10 bars and
losers after 0–1, that density is a candidate entry filter — take it to the
[[Rule Tester]] to quantify the before/after effect, then validate
out-of-sample before trusting it.

No separation is also an answer: crossover clustering carries no information
for this strategy, so stop tuning it.

## Related

- [[Rule Tester]] — turn a discovered pattern into a filter and measure it
- [[Edge Analysis]] — complementary entry-quality validation
