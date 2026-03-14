---
tags:
  - usability/running-analysis
  - backtest
---

# Single Security Backtest

Test a strategy against one security's historical data in isolation.

---

## When to Use

- Validating a strategy idea against a single security
- Debugging strategy logic before scaling to a portfolio
- Generating detailed trade-by-trade analysis

---

## GUI Workflow

1. Launch `python ctk_backtest_gui.py` (or via the main launcher)
2. **Select strategy** from the dropdown
3. **Choose security** — picks from CSV files in `raw_data/daily/`
4. **Set parameters:**

| Field | Description | Default |
|---|---|---|
| Initial Capital | Starting account balance | 100,000 |
| Commission Mode | `PERCENTAGE` or `FIXED` | PERCENTAGE |
| Commission Value | 0.001 = 0.1% per trade | 0.001 |
| Slippage | Percentage slippage on execution | 0.1% |
| Start / End Date | Filter the data range (optional) | Full range |

5. **Adjust strategy parameters** — each strategy exposes its own tuneable parameters (see [[Configuration Options]])
6. Click **Run Backtest**

---

## Python API

```python
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from strategies.base_alphatrend_strategy import BaseAlphaTrendStrategy
from pathlib import Path

config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001),
    slippage_percent=0.1
)

loader = DataLoader(Path('raw_data/daily'))
data = loader.load_csv('AAPL')

strategy = BaseAlphaTrendStrategy(atr_multiplier=2.5, risk_percent=2.0)

engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)
```

---

## Output

The engine returns a `BacktestResult` containing:

- **Trade list** — every trade with entry/exit dates, prices, P/L, duration
- **Equity curve** — capital value at every bar
- **Performance metrics** — 50+ metrics (see [[Metrics Glossary]])

Reports are saved to `logs/backtests/single_security/`:
- Excel report (`.xlsx`) with charts and metric breakdowns
- Trade log (`.csv`) with one row per trade

---

## Key Behaviours

> [!warning] No Lookahead Bias
> The engine feeds data to the strategy one bar at a time. The strategy cannot see future bars. All indicators must be pre-calculated in the raw CSV data.

- Trades execute at the **closing price** of the signal bar
- Commission is deducted on both entry and exit
- Slippage is applied as a percentage on execution price
- Stop losses are checked **before** the strategy generates signals on each bar
- Position sizing is determined by the strategy's `position_size()` method

---

## Next Steps

- [[Portfolio Backtest]] — test across multiple securities
- [[Reading Reports]] — interpret the output files
- [[Backtest Execution Flow]] — understand the engine's bar-by-bar logic
