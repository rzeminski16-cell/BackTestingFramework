---
tags:
  - usability/running-analysis
  - analysis
  - factor-analysis
---

# Factor Analysis

Identify which market conditions and indicators drive good vs bad trades for your strategy.

---

## When to Use

- Understanding *why* a strategy wins or loses
- Discovering regime-dependent behaviour (works in trends, fails in chop)
- Finding potential entry filters to improve performance

---

## Analysis Tiers

The factor analysis system operates in three tiers of increasing sophistication.

### Tier 1 — Exploratory

Quick visual exploration of factor relationships.

- Correlation analysis between factors and trade outcomes
- Distribution comparisons (winning trades vs losing trades)
- Scatter plots and visual patterns

### Tier 2 — Statistical Testing

Hypothesis-driven analysis with statistical rigour.

- **t-tests / Mann-Whitney U** — are factor distributions significantly different for winners vs losers?
- **Effect size** — how large is the practical difference?
- **P-values** — statistical significance

### Tier 3 — Machine Learning

Predictive modelling to find complex factor interactions.

- **Feature importance** — which factors best predict trade outcome (Random Forest, XGBoost)
- **Classification models** — can we predict winners from pre-trade conditions?
- **Interaction analysis** — do factor combinations matter more than individual factors?

---

## Factor Categories

| Category | Examples |
|---|---|
| Technical | RSI at entry, ATR, momentum, distance from moving averages |
| Fundamental | P/E ratio, revenue growth, EPS |
| Insider | Net insider buying/selling prior to entry |
| Options | Implied volatility, put/call skew |
| Regime | Market trend direction, volatility regime |

---

## GUI Workflow

1. Launch `python ctk_factor_analysis_gui.py` (or via the main launcher)
2. Select a **completed backtest** to analyse (from `logs/backtests/`)
3. Choose which factor categories to include
4. Select the analysis tier
5. Click **Run Analysis**

---

## Output

- Per-factor statistical summaries
- Feature importance rankings
- Scenario detection — conditions associated with the best/worst trades
- Visual reports (distributions, correlations, importance charts)

---

## Next Steps

- [[Factor Analysis Component]] — implementation details
- [[Edge Analysis]] — simpler entry quality validation
