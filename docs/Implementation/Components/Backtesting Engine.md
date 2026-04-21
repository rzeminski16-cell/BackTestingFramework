---
tags:
  - implementation/component
  - engine
---

# Backtesting Engine

The core engines that execute strategies bar-by-bar against historical data.

---

## Engines

| Engine | File | Purpose |
|---|---|---|
| `SingleSecurityEngine` | `Classes/Engine/single_security_engine.py` | Runs one strategy on one security |
| `PortfolioEngine` | `Classes/Engine/portfolio_engine.py` | Runs one strategy across multiple securities with shared capital |

---

## SingleSecurityEngine

### Initialisation

Takes a `BacktestConfig` plus optional `CurrencyConverter` and `SecurityRegistry`. Creates a `PositionManager` and `TradeExecutor` internally.

### Execution (`run()`)

1. Validate data (non-empty, has `date` and `close`)
2. Filter by date range if configured
3. Call `strategy.prepare_data(data)` — pre-calculates custom indicators once
4. Iterate bar-by-bar through the data:
   - Build `StrategyContext` for the current bar
   - Check stop loss and take profit
   - Call `strategy.generate_signal(context)`
   - Execute the resulting signal via `TradeExecutor`
   - Record equity for the equity curve
5. Close any remaining open position at the last bar
6. Return `BacktestResult`

### Key Design Decisions

- **Execution at close** — all trades execute at the current bar's closing price
- **Stop loss checked first** — before strategy signals, the engine checks if the stop loss was breached
- **Single position** — one open position at a time per security
- **Slippage applied** — as a percentage on the execution price

---

## PortfolioEngine

Extends the single-security concept to multiple securities sharing one capital pool.

### Additional Responsibilities

- **Capital allocation** — tracks available capital across all securities
- **Capital contention** — handles the case when a BUY signal arrives but capital is exhausted
- **Per-security tracking** — maintains separate trade histories per symbol
- **Aggregated equity** — combines all positions into a portfolio equity curve

### Capital Contention Handling

When a new BUY signal arrives and capital is insufficient:

| Mode | Behaviour |
|---|---|
| `DEFAULT` | Skip the signal |
| `VULNERABILITY_SCORE` | Score all open positions against a compound-growth target price. If at least one non-immune position is below target, close the one furthest below target and enter the new trade |

Scores are computed by `VulnerabilityScoreCalculator` in
`Classes/Engine/vulnerability_score.py`. See [[Configuration Options]] for the
configurable parameters (`min_trade_age_days`, `target_monthly_growth`,
`alpha`, `beta`, `avg_window_days`, `pullback_window_days`).

---

## BacktestResult

The result object returned by both engines, containing:

- `trades` — list of completed `Trade` objects
- `equity_curve` — list of equity values per bar
- `num_trades` — total trades
- Computed metrics: `total_return_pct`, `sharpe_ratio`, `max_drawdown`, etc.

---

## Related

- [[Position Management]] — how positions and trades are managed
- [[Backtest Execution Flow]] — the full bar-by-bar flow diagram
- [[Portfolio Execution Flow]] — multi-security flow with capital contention
