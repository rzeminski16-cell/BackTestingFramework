---
tags:
  - usability/running-analysis
  - analysis
---

# Edge Analysis

Validate the quality of a strategy's entries using E-ratio and R-multiple analysis.

---

## When to Use

- Determining whether entries have genuine predictive power
- Comparing entry quality across strategies or parameter sets
- Understanding risk/reward characteristics at a per-trade level

---

## Key Concepts

### E-Ratio (Edge Ratio)

Measures how much price moves in your favour vs against you after entry, over a fixed number of bars.

$$E = \frac{\text{Maximum Favourable Excursion (MFE)}}{\text{Maximum Adverse Excursion (MAE)}}$$

- **E > 1.0** → price tends to move in your favour after entry (positive edge)
- **E ≈ 1.0** → random — no edge
- **E < 1.0** → entries are worse than random

### R-Multiples

Each trade's profit or loss expressed as a multiple of the initial risk (distance to stop loss).

- **+2R** = made twice your initial risk
- **-1R** = lost your full initial risk (stopped out)
- **Positive average R** = the strategy has positive expectancy

---

## GUI Workflow

1. Launch `python ctk_edge_analysis_gui.py` (or via the main launcher)
2. Select strategy, security, and parameters
3. Configure the analysis window (number of bars after entry to measure)
4. Click **Run**

---

## Output

- **E-ratio chart** — edge ratio at different bar horizons after entry
- **R-multiple distribution** — histogram showing how trades cluster
- **Average R** — the strategy's expected payoff per unit of risk

---

## Next Steps

- [[Factor Analysis]] — identify what conditions produce good vs bad trades
- [[Metrics Glossary]] — related metrics like profit factor, win rate
