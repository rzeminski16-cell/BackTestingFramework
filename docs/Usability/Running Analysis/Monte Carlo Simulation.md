---
tags:
  - usability/running-analysis
  - guide
---

# Monte Carlo Simulation

Bootstrap your strategy's historical trade returns into thousands of synthetic
equity curves to see the *distribution* of outcomes — drawdown risk,
probability of loss, and how sensitive results are to trade ordering — rather
than the single path one backtest happens to produce.

---

## Launching

```bash
python ctk_monte_carlo_gui.py        # or the "Monte Carlo" card in the launcher
```

Headless equivalent:

```bash
python -m btf montecarlo --trade-log logs/<run>/trades.csv --simulations 5000
```

## Prerequisites

One or more **trade-log CSVs** produced by a backtest (the framework schema
with `pl_pct`, or any CSV — the loader falls back to user-selected columns).

## The wizard

1. **Load Data** — select one or more trade-log CSVs; they are concatenated
   into a single return pool.
2. **Data Review** — summary statistics (count, mean, std, skew) and the
   return-source picker: **% returns** (`pl_pct`) or **R-multiples**
   (computed from entry/stop when present). R-multiples pair naturally with
   the risk-per-trade knob; % returns replay realised outcomes.
3. **Simulation** — initial capital, number of simulations, trades per path,
   risk per trade (fraction of equity), sizing (**compounding** or **fixed**
   on starting capital), and sampling (**simple bootstrap** or **block
   bootstrap** — blocks preserve short-run streaks/autocorrelation).
4. **Advanced** — optional drawdown-responsive risk reduction (cut risk to a
   reduced level while below a drawdown threshold), per-trade costs, return
   clipping (tames extreme R-multiples from very tight stops), random seed.
5. **Run** — progress plus a tabbed results window: equity-curve fan with
   percentile bands, final-equity and drawdown distributions, and the
   summary metrics table.

## Reading the results

| Metric | Meaning |
|---|---|
| Median / P5 / P95 final equity | The central outcome and the realistic band around it |
| Probability of loss | Share of paths ending below starting capital |
| Probability of ruin | Share of paths that ever dropped to half of starting capital |
| Median / P95 max drawdown | Typical and bad-case drawdown — budget capital for P95, not the median |
| Median annualized Sharpe / CAGR / Calmar | Per-path distributions; set **periods per year** (your trades/year) to enable |

A strategy whose single backtest looks great but whose P5 path is ruinous is
telling you the historical trade *ordering* was lucky.

## Daily-returns mode (CLI)

Bootstrapping per-trade returns ignores how long trades take. To simulate
calendar-time risk, feed an equity-curve CSV instead:

```bash
python -m btf montecarlo --daily-curve logs/<run>/equity_curve.csv \
    --steps 252 --simulations 5000
```

Each step is then one *day* (`risk` is fixed at 1.0 and
`periods-per-year` defaults to 252, so annualized distributions are exact).

## Related

- [[Reading Reports]] — where trade logs come from
- [[Metrics Glossary]] — definitions for every metric shown
