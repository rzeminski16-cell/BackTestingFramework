---
tags:
  - usability/reference
  - configuration
---

# Configuration Options

All configuration fields for backtesting and portfolio testing.

---

## BacktestConfig

Used for [[Single Security Backtest|single security backtests]].

| Field | Type | Default | Description |
|---|---|---|---|
| `initial_capital` | float | 100,000 | Starting account balance |
| `commission` | CommissionConfig | 0.1% percentage | Commission settings (see below) |
| `start_date` | datetime | None | Start of backtest period (None = from beginning of data) |
| `end_date` | datetime | None | End of backtest period (None = to end of data) |
| `position_size_limit` | float | 1.0 | Max position size as fraction of capital (1.0 = 100%) |
| `base_currency` | str | "GBP" | Account base currency |
| `slippage_percent` | float | 0.1 | Slippage applied on all trade executions (%) |

---

## CommissionConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | CommissionMode | PERCENTAGE | `PERCENTAGE` or `FIXED` |
| `value` | float | 0.001 | Commission value (0.001 = 0.1% for percentage, or e.g. 3.0 for fixed) |

> [!example] Commission Examples
> - **0.1% per trade**: `CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)`
> - **Fixed fee**: `CommissionConfig(mode=CommissionMode.FIXED, value=3.0)`

---

## PortfolioConfig

Used for [[Portfolio Backtest|portfolio backtests]]. Includes all BacktestConfig fields plus:

| Field | Type | Default | Description |
|---|---|---|---|
| `capital_contention` | CapitalContentionConfig | DEFAULT mode | How to handle signals when capital is exhausted |
| `basket_name` | str | None | Name of security basket (for logging) |

---

## CapitalContentionConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | CapitalContentionMode | DEFAULT | `DEFAULT`, `VULNERABILITY_SCORE`, or `ENHANCED_VULNERABILITY` |
| `vulnerability_config` | VulnerabilityScoreConfig | (see below) | Parameters for vulnerability scoring |

### Capital Contention Modes

| Mode | Behaviour |
|---|---|
| `DEFAULT` | Ignore new signals when no capital available |
| `VULNERABILITY_SCORE` | Score positions, swap weak ones for new signals |
| `ENHANCED_VULNERABILITY` | Feature-weighted scoring with multiple factors |

---

## VulnerabilityScoreConfig

Parameters for the `VULNERABILITY_SCORE` mode:

| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `immunity_days` | int | 7 | 1–30 | Days new positions are protected from swapping |
| `min_profit_threshold` | float | 0.02 | 0.0–0.20 | P/L below this = "stagnant" (fast decay) |
| `decay_rate_fast` | float | 5.0 | 1.0–20.0 | Points lost per day for stagnant trades |
| `decay_rate_slow` | float | 1.0 | 0.1–5.0 | Points lost per day for performing trades |
| `swap_threshold` | float | 50.0 | 10–90 | Score below which position can be swapped |
| `base_score` | float | 100.0 | — | Starting score |

---

## OptimizationConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `metric` | str | "total_return" | Metric to optimise |
| `per_security` | bool | False | Optimise per security or globally |
| `min_trades` | int | 1 | Minimum trades required for valid result |
| `maximize` | bool | True | Maximise the metric (False for metrics like max_drawdown) |

### Valid Optimisation Metrics

`total_return`, `total_return_pct`, `sharpe_ratio`, `profit_factor`, `win_rate`, `max_drawdown`, `avg_win`, `avg_loss`, `num_trades`

---

## Strategy Parameters

Strategy-specific parameters are defined in `config/strategy_parameters.json`. Each parameter has:

| Field | Description |
|---|---|
| `type` | `float` or `int` |
| `default` | Default value |
| `configurable` | Whether the GUI exposes it |
| `optimization` | Min, max, step for optimisation |
| `description` | Human-readable description |
| `category` | Grouping for the GUI |

See [[Creating Strategy Presets]] for saving parameter combinations.

---

## Next Steps

- [[Single Security Backtest]] — use these config options
- [[Portfolio Backtest]] — portfolio-specific configuration
- [[Vulnerability Scoring]] — deep dive into scoring parameters
