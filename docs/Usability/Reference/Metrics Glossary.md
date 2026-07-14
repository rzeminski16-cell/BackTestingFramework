---
tags:
  - usability/reference
  - metrics
---

# Metrics Glossary

All performance metrics calculated by the framework, grouped by category.

---

## Return Metrics

| Metric | Description |
|---|---|
| **Total Return** | Absolute profit/loss in base currency |
| **Total Return %** | Percentage return on initial capital |
| **Annualised Return** | Total return scaled to a per-year basis |
| **CAGR** | Compound Annual Growth Rate |
| **Average Trade Return** | Mean P/L per trade |
| **Average Trade Return %** | Mean percentage P/L per trade |

---

## Risk Metrics

| Metric | Description |
|---|---|
| **Max Drawdown** | Largest peak-to-trough decline in equity |
| **Max Drawdown %** | Max drawdown as percentage of peak equity |
| **Max Drawdown Duration** | Longest time spent below a previous equity peak |
| **Average Drawdown** | Mean drawdown across all drawdown periods |
| **Volatility (Annualised)** | Standard deviation of returns, annualised |
| **Downside Deviation** | Standard deviation of negative returns only |

---

## Risk-Adjusted Return Metrics

| Metric | Description |
|---|---|
| **Sharpe Ratio** | (Return - Risk-Free Rate) / Volatility. Measures return per unit of total risk |
| **Sortino Ratio** | (Return - Risk-Free Rate) / Downside Deviation. Like Sharpe but only penalises downside risk |
| **Calmar Ratio** | Annualised Return / Max Drawdown. Higher = better return relative to worst-case |
| **Recovery Factor** | Total Return / Max Drawdown. How many times the strategy recovered its worst loss |
| **Profit Factor** | Gross Wins / Gross Losses. Above 1.0 = profitable overall |

---

## Trade Statistics

| Metric | Description |
|---|---|
| **Number of Trades** | Total completed trades |
| **Win Rate** | Percentage of trades that were profitable |
| **Loss Rate** | Percentage of trades that were unprofitable |
| **Average Win** | Mean profit on winning trades |
| **Average Loss** | Mean loss on losing trades |
| **Largest Win** | Best single trade P/L |
| **Largest Loss** | Worst single trade P/L |
| **Average Win/Loss Ratio** | Average Win / Average Loss (absolute values) |
| **Consecutive Wins (Max)** | Longest streak of winning trades |
| **Consecutive Losses (Max)** | Longest streak of losing trades |

---

## Duration Metrics

| Metric | Description |
|---|---|
| **Average Trade Duration** | Mean holding period across all trades |
| **Average Winning Trade Duration** | Mean holding period for winners |
| **Average Losing Trade Duration** | Mean holding period for losers |
| **Total Time in Market** | Percentage of the backtest period where a position was open |

---

## Expectancy Metrics

| Metric | Description |
|---|---|
| **Expectancy** | Average amount won/lost per trade. (Win Rate × Avg Win) - (Loss Rate × Avg Loss) |
| **Expectancy per Day** | Expectancy divided by average trade duration |
| **R-Multiple (Mean)** | Average trade return expressed as a multiple of initial risk |
| **R-Multiple (Median)** | Median R-multiple |

---

## Edge & Excursion Metrics

| Metric | Description |
|---|---|
| **E-Ratio** | Maximum Favourable Excursion / Maximum Adverse Excursion. Above 1.0 = positive edge |
| **MFE (Max Favourable Excursion)** | Best unrealised move during a trade, % vs entry (≥ 0). Tracked bar-by-bar off the bar's high/low, direction-aware |
| **MAE (Max Adverse Excursion)** | Worst unrealised move during a trade, % vs entry (≤ 0). Large \|MAE\| on winners suggests stops are wider than needed |
| **Avg MFE / Avg MAE** | Means across all trades that tracked excursions |

---

## Risk Metrics (Equity Curve)

| Metric | Description |
|---|---|
| **Exposure (Time in Market)** | Percentage of bars with capital deployed |
| **VaR 95%** | Historical one-bar Value-at-Risk: the 5% worst-tail begins at this one-bar loss (%) |
| **CVaR 95%** | Expected shortfall: the *average* loss beyond the 95% VaR (%) |
| **Rolling Sharpe / Volatility** | Windowed (default 63-bar) versions for stability inspection |

---

## Stable (Regression-Based) Metrics

These measure underlying performance in a way that is less sensitive to
lucky start/end points. **Annualisation convention:** every equity curve is
first resampled onto a forward-filled *calendar-daily* grid, so
`BARS_PER_YEAR = 365` always matches the true cadence — RAR% is comparable
across the backtest reports, the Rule Tester, and the Modelling stage.

| Metric | Description |
|---|---|
| **RAR% (Regressed Annual Return)** | Fit OLS to ln(equity) vs time on the calendar-daily grid; RAR% = (e^(slope × 365) − 1) × 100. An annual return estimate using *every* point on the curve, not just the endpoints |
| **R² (Log-Equity Regression)** | Goodness of fit of that regression — how "straight" the equity growth is (1.0 = perfectly steady) |
| **Adjusted RAR% (RAR% × R²)** | RAR% penalised by noisiness; the primary selection metric in the Modelling stage |
| **R-Cubed** | RAR% / (Average of the 5 largest drawdowns × average drawdown length / 365). Return per unit of drawdown depth *and* duration |
| **Robust Sharpe** | RAR% / annualised std-dev of rolling 30-calendar-day returns |

---

## Multiple-Testing Control

| Metric | Description |
|---|---|
| **PSR (Probabilistic Sharpe Ratio)** | Probability that the true Sharpe exceeds a benchmark given the sample length and return moments |
| **DSR (Deflated Sharpe Ratio)** | PSR against the Sharpe you'd expect from the *luckiest* of the N configurations a parameter search examined. Shown per walk-forward window ("In-Sample DSR"); **> 0.95 = the result survives its own search**. A high in-sample Sharpe with a low DSR is selection bias, not skill |

---

## Portfolio-Specific Metrics

| Metric | Description |
|---|---|
| **Trades Skipped (No Capital)** | Signals that couldn't be taken due to insufficient capital |
| **Positions Swapped** | Number of times a vulnerable position was closed for a new signal |
| **Capital Utilisation** | Average percentage of capital deployed |

---

## How Metrics Are Calculated

All metrics are computed by `Classes/Core/performance_metrics.py` and `Classes/Analysis/performance_metrics.py`. The engine passes the trade list and equity curve to these modules after a backtest completes.

For implementation details, see [[Reporting]].
