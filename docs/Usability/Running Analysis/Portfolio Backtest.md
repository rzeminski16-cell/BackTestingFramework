---
tags:
  - usability/running-analysis
  - backtest
  - portfolio
---

# Portfolio Backtest

Test a strategy across multiple securities sharing a single capital pool.

---

## When to Use

- Evaluating a strategy across a diversified set of securities
- Understanding the impact of capital contention (not enough capital for all signals)
- Testing portfolio-level performance and drawdowns

---

## GUI Workflow

1. Launch via `python ctk_main_gui.py` → **Backtesting** → select **Portfolio** mode
2. **Select strategy**
3. **Choose a basket** — pre-defined groups of securities from `config/baskets/` (or select individual securities)
4. **Configure portfolio settings:**

| Field | Description | Default |
|---|---|---|
| Initial Capital | Total shared capital pool | 100,000 |
| Capital Contention Mode | How to handle signals when capital is exhausted | DEFAULT |
| Commission / Slippage | Same as single security | 0.1% / 0.1% |

5. Click **Run Backtest**

---

## Capital Contention Modes

When a new BUY signal arrives but no capital is available, the system must decide what to do. This is controlled by the [[Configuration Options|capital contention mode]]:

### `DEFAULT`
Simply ignore new signals when capital is exhausted. First-come, first-served.

### `VULNERABILITY_SCORE`
Score existing positions against a compound-growth target price and potentially
**swap** the position furthest below its target for the new signal. The score
is the percentage distance of the current (averaged) price below a target:

`P_target(t) = P_entry * (1 + g_d)^t * perf_ratio^(-alpha) * (1 + beta * min(r14, 0))`

where `g_d` comes from `target_monthly_growth`, `perf_ratio` measures realized
daily return vs. `g_d`, and `r14` is the mean of recent daily returns.

Immunities:
- **Age immunity** — positions younger than `min_trade_age_days` are never swapped
- **At/above target** — positions whose current reference price is at or above target are immune (score = 0)

---

## Python API

```python
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Config.config import PortfolioConfig, CommissionConfig
from Classes.Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode
)
from Classes.Data.data_loader import DataLoader
from strategies.your_strategy import YourStrategy
from pathlib import Path

config = PortfolioConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(),
    capital_contention=CapitalContentionConfig.vulnerability_score_mode(
        min_trade_age_days=100,
        target_monthly_growth=0.05,
        alpha=1.0,
        beta=1.0,
    )
)

loader = DataLoader(Path('raw_data/daily'))
symbols = ['AAPL', 'MSFT', 'GOOGL']
data = {sym: loader.load_csv(sym) for sym in symbols}

strategy = YourStrategy()
engine = PortfolioEngine(config)
result = engine.run(data, strategy)
```

---

## Output

- **Portfolio-level equity curve** — aggregated across all securities
- **Per-security trade logs** — same detail as single-security
- **Capital allocation history** — who held capital at each point
- **Vulnerability score history** — (if using vulnerability scoring)

Reports saved to `logs/backtests/portfolio/`.

---

## Security Baskets

Pre-defined groups of securities live in `config/baskets/` as JSON files. See [[Creating a Security Basket]] to create your own.

---

## Next Steps

- [[Walk-Forward Optimisation]] — optimise parameters across the portfolio
- [[Portfolio Execution Flow]] — understand the engine's multi-security logic
