---
tags:
  - implementation/component
  - analysis
  - factor-analysis
---

# Factor Analysis Component

Identifies which market conditions and indicators correlate with trade outcomes.

---

## Key Classes

| Class | File | Purpose |
|---|---|---|
| `FactorAnalyzer` | `Classes/FactorAnalysis/analyzer.py` | Main analyser orchestrating all tiers |
| Scenario detection modules | `Classes/FactorAnalysis/` | Identifies market condition patterns |

---

## Three-Tier Architecture

### Tier 1 — Exploratory

Fast, visual analysis:
- Correlation matrix between factors and trade P/L
- Distribution plots comparing winning vs losing trades
- Scatter plots to spot nonlinear relationships

### Tier 2 — Statistical Testing

Hypothesis-driven validation:
- **t-tests** — are factor means significantly different for winners vs losers?
- **Mann-Whitney U** — non-parametric alternative (doesn't assume normal distribution)
- **Effect size** — practical significance (Cohen's d)
- **P-value correction** — adjusts for multiple comparisons

### Tier 3 — Machine Learning

Predictive modelling:
- **Random Forest / XGBoost** — tree-based feature importance
- **Classification** — can we predict trade outcome from pre-trade factors?
- **Interaction detection** — do factor combinations matter more than individual factors?

---

## Factor Categories

| Category | Source Data | Examples |
|---|---|---|
| Technical | `raw_data/daily/` | RSI, ATR, momentum, distance from MAs |
| Fundamental | `raw_data/fundamentals/` | P/E ratio, EPS growth, revenue growth |
| Insider | `raw_data/insider_transactions/` | Net insider buying, transaction volume |
| Options | `raw_data/options/` | Implied volatility, put/call skew |
| Regime | Derived from price data | Trend direction, volatility regime |

---

## Scenario Detection

The analyser identifies "scenarios" — specific market conditions that correlate with unusually good or bad trade outcomes. For example:

- "Trades entered when RSI < 30 **and** ATR > 2x average had 75% win rate"
- "Trades during low-volatility regimes underperformed by 40%"

This uses a combination of statistical binning and interaction analysis.

---

## Configuration

Analysis settings are stored in `config/trade_analysis_config.json`, controlling which factors to include, significance thresholds, and analysis depth.

---

## Related

- [[Factor Analysis]] — user guide
- [[Edge Analysis]] — simpler entry quality analysis
