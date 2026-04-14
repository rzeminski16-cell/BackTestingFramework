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
Score existing positions and potentially **swap** a weak position for the new signal. A position becomes vulnerable when its score drops below the `swap_threshold`.

The score decays over time:
- **Immunity period** — new positions are protected for N days (score = 100)
- **Fast decay** — applied to stagnant positions (P/L below threshold)
- **Slow decay** — applied to profitable positions

See [[Vulnerability Scoring]] for the full scoring guide.

### `ENHANCED_VULNERABILITY`
Feature-based scoring with configurable weights for multiple factors (momentum, volatility, drawdown from peak, etc.). See [[Vulnerability Scoring]] for details.

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
        immunity_days=7,
        swap_threshold=50.0
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

- [[Vulnerability Scoring]] — tune the capital contention scorer
- [[Walk-Forward Optimisation]] — optimise parameters across the portfolio
- [[Portfolio Execution Flow]] — understand the engine's multi-security logic
