# Performance Metrics Glossary

The framework calculates 50+ performance metrics. This glossary defines each metric and provides guidance on interpretation.

---

## Return Metrics

### Total Return

**Definition**: Overall percentage gain or loss from start to end of backtest.

```
Total Return = (Final Capital - Initial Capital) / Initial Capital × 100
```

| Value | Interpretation |
|-------|----------------|
| Positive | Strategy was profitable |
| Negative | Strategy lost money |
| vs Benchmark | Compare to buy-and-hold or index |

---

### Annualized Return

**Definition**: Return normalized to a yearly basis.

```
Annualized Return = ((1 + Total Return) ^ (365 / Days)) - 1
```

| Value | Interpretation |
|-------|----------------|
| > 10% | Strong annual performance |
| 5-10% | Moderate performance |
| < 5% | Consider alternatives |

---

### CAGR (Compound Annual Growth Rate)

**Definition**: Geometric mean annual growth rate.

```
CAGR = (Final Value / Initial Value) ^ (1 / Years) - 1
```

Accounts for compounding, useful for multi-year comparisons.

---

## Risk Metrics

### Max Drawdown

**Definition**: Largest peak-to-trough decline during the backtest.

```
Max Drawdown = (Trough Value - Peak Value) / Peak Value × 100
```

| Value | Interpretation |
|-------|----------------|
| < 10% | Low risk |
| 10-20% | Moderate risk |
| 20-30% | High risk |
| > 30% | Very high risk |

---

### Average Drawdown

**Definition**: Mean of all drawdown periods.

Lower values indicate more consistent equity growth.

---

### Drawdown Duration

**Definition**: Number of days from peak to recovery.

```
Max Drawdown Duration: Longest time to recover from a drawdown
Avg Drawdown Duration: Average recovery time
```

---

### Volatility (Standard Deviation)

**Definition**: Standard deviation of daily returns.

```
Volatility = StdDev(Daily Returns) × √252  (annualized)
```

Higher volatility = larger daily swings.

---

### Downside Deviation

**Definition**: Standard deviation of negative returns only.

Used in Sortino ratio calculation. Focuses on "bad" volatility.

---

## Risk-Adjusted Return Metrics

### Sharpe Ratio

**Definition**: Return per unit of total risk.

```
Sharpe Ratio = (Return - Risk-Free Rate) / Volatility
```

| Value | Interpretation |
|-------|----------------|
| < 0 | Negative risk-adjusted return |
| 0-0.5 | Poor |
| 0.5-1.0 | Acceptable |
| 1.0-2.0 | Good |
| > 2.0 | Excellent |

---

### Sortino Ratio

**Definition**: Return per unit of downside risk.

```
Sortino Ratio = (Return - Risk-Free Rate) / Downside Deviation
```

Like Sharpe but only penalizes downside volatility. Generally higher than Sharpe.

| Value | Interpretation |
|-------|----------------|
| < 1.0 | Poor |
| 1.0-2.0 | Good |
| > 2.0 | Excellent |

---

### Calmar Ratio

**Definition**: Return relative to max drawdown.

```
Calmar Ratio = Annualized Return / |Max Drawdown|
```

| Value | Interpretation |
|-------|----------------|
| < 0.5 | Poor risk/reward |
| 0.5-1.0 | Acceptable |
| 1.0-3.0 | Good |
| > 3.0 | Excellent |

---

### Information Ratio

**Definition**: Excess return over benchmark per unit of tracking error.

```
Information Ratio = (Strategy Return - Benchmark Return) / Tracking Error
```

Measures skill in beating a benchmark.

---

## Trade Metrics

### Total Trades

**Definition**: Number of completed round-trip trades (entry + exit).

More trades = more data for statistical significance, but also more commission costs.

---

### Win Rate

**Definition**: Percentage of profitable trades.

```
Win Rate = Winning Trades / Total Trades × 100
```

| Strategy Type | Typical Win Rate |
|--------------|------------------|
| Trend following | 30-45% |
| Mean reversion | 50-70% |
| Random | ~50% |

Note: Win rate alone is not meaningful—must consider average win/loss size.

---

### Profit Factor

**Definition**: Gross profit divided by gross loss.

```
Profit Factor = Sum(Winning Trades) / |Sum(Losing Trades)|
```

| Value | Interpretation |
|-------|----------------|
| < 1.0 | Losing system |
| 1.0-1.5 | Marginal edge |
| 1.5-2.0 | Good edge |
| > 2.0 | Strong edge |

---

### Average Trade

**Definition**: Mean profit/loss per trade.

```
Average Trade = Total P/L / Total Trades
```

Must be positive after commissions for a viable strategy.

---

### Average Win / Average Loss

**Definition**: Mean profit of winning trades / Mean loss of losing trades.

```
Win/Loss Ratio = Average Win / |Average Loss|
```

Higher is better. Combined with win rate determines expectancy.

---

### Expectancy

**Definition**: Expected value per trade.

```
Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
```

Or equivalently:

```
Expectancy = Average Trade
```

Must be positive for a viable system.

---

### Largest Win / Largest Loss

**Definition**: Best and worst individual trades.

Large outliers may indicate:
- Lucky/unlucky events
- Potential for outsized gains
- Risk of catastrophic loss

---

### Consecutive Wins / Losses

**Definition**: Maximum streak of winning/losing trades.

Long losing streaks test psychological resilience and capital reserves.

---

## Time-Based Metrics

### Average Holding Period

**Definition**: Mean number of days/bars per trade.

Indicates strategy timeframe (intraday, swing, position).

---

### Average Bars in Winning/Losing Trades

**Definition**: Mean duration of winners vs losers.

Winners running longer than losers is generally positive.

---

### Trades Per Year

**Definition**: Average number of trades annualized.

```
Trades Per Year = Total Trades / Years in Backtest
```

More trades = more opportunities but also more costs.

---

## Portfolio-Specific Metrics

### Capital Utilization

**Definition**: Average percentage of capital deployed.

```
Capital Utilization = Avg(Allocated Capital) / Total Capital × 100
```

Low utilization means capital sitting idle.

---

### Position Count Statistics

- **Avg Positions**: Mean concurrent positions
- **Max Positions**: Maximum concurrent positions
- **Days Fully Invested**: Days at max allocation

---

### Capital Contention Events

**Definition**: Times signals were skipped due to capital/position limits.

High contention may indicate:
- Position limits too restrictive
- Strategy generates too many signals
- Need for better signal prioritization

---

## E-Ratio Metrics

### E-Ratio (Edge Ratio)

**Definition**: Ratio of favorable to adverse price movement after entry.

```
E-Ratio(n) = Avg Normalized MFE(n) / Avg Normalized MAE(n)
```

Where:
- MFE = Maximum Favorable Excursion (best price reached)
- MAE = Maximum Adverse Excursion (worst price reached)
- Normalized by ATR for cross-market comparison

| Value | Interpretation |
|-------|----------------|
| < 1.0 | Negative edge (entries work against you) |
| = 1.0 | No edge (random) |
| > 1.0 | Positive edge (entries in your favor) |

---

### Peak E-Ratio Horizon

**Definition**: Number of days at which E-ratio is highest.

Suggests optimal holding period for the strategy's entries.

---

## R-Multiple Metrics

### R-Multiple

**Definition**: Trade result expressed in units of initial risk.

```
R-Multiple = Trade P/L / Initial Risk Amount
```

Example: Risk $100, make $300 = 3R

---

### Average R-Multiple

**Definition**: Mean R-multiple across all trades.

```
Average R = Sum(R-Multiples) / Total Trades
```

Must be positive for a viable system.

---

### R-Multiple Distribution

Statistics about the distribution of trade outcomes:
- **Median R**: Middle value (less affected by outliers)
- **R Std Dev**: Consistency of outcomes
- **Skewness**: Asymmetry (positive = fat right tail)

---

## Statistical Metrics

### T-Statistic

**Definition**: Statistical significance of returns.

```
T-Stat = (Mean Return - 0) / (Std Dev / √n)
```

| Value | Interpretation |
|-------|----------------|
| < 1.96 | Not statistically significant (95% level) |
| > 1.96 | Significant at 95% level |
| > 2.58 | Significant at 99% level |

---

### P-Value

**Definition**: Probability results occurred by chance.

| Value | Interpretation |
|-------|----------------|
| > 0.05 | Not significant |
| < 0.05 | Significant at 95% level |
| < 0.01 | Significant at 99% level |

---

## Quick Reference Table

| Metric | Good Value | Excellent Value |
|--------|------------|-----------------|
| Sharpe Ratio | > 1.0 | > 2.0 |
| Sortino Ratio | > 1.5 | > 3.0 |
| Calmar Ratio | > 1.0 | > 3.0 |
| Profit Factor | > 1.5 | > 2.0 |
| Max Drawdown | < 20% | < 10% |
| Win Rate (trend) | > 40% | > 50% |
| Expectancy | Positive | > 1R |
| E-Ratio | > 1.1 | > 1.3 |

---

## Related Documentation

- [Backtesting Concepts](../concepts/BACKTESTING.md) — How metrics are calculated
- [Edge Analysis](../concepts/EDGE_ANALYSIS.md) — E-ratio methodology
- [Backtest GUI](../applications/BACKTEST_GUI.md) — Viewing metrics
