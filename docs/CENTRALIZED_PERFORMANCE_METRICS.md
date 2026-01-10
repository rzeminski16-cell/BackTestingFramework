# Centralized Performance Metrics

## Overview

The BackTestingFramework now uses a **centralized performance metrics module** located at:

```
Classes/Core/performance_metrics.py
```

This module provides the **single source of truth** for all performance metric calculations across the framework, ensuring consistency in:

- Risk-free rate (standardized at **3.5%** - UK base rate approximation)
- Trading days per year (**252** days)
- Edge case handling (profit factor caps, zero divisions, etc.)
- Annualization methodology

## Why Centralization?

Previously, different systems in the framework calculated metrics differently:

| Issue | Before | After |
|-------|--------|-------|
| Risk-free rate | 2% in some files, 3.5% in others | **3.5% everywhere** |
| Profit factor cap | Some returned infinity, others 999.99 | **999.99 everywhere** |
| Sharpe calculation | Minor variations in excess returns | **Standardized formula** |
| Max drawdown | Different handling for portfolios | **Consistent methodology** |

## Systems Using Centralized Metrics

All the following systems now use `CentralizedPerformanceMetrics`:

1. **Backtesting** (`backtesting/metrics/performance.py`)
2. **Analysis** (`Classes/Analysis/performance_metrics.py`)
3. **Univariate Optimization** (`Classes/Optimization/univariate_optimizer.py`)
4. **Multivariate Optimization** (`Classes/Optimization/optimizer.py`)
5. **Walk-Forward Optimization** (`Classes/Optimization/walk_forward_optimizer.py`)
6. **Sensitivity Analysis** (`Classes/Optimization/sensitivity_analyzer.py`)
7. **Vulnerability Scorer** (`Classes/VulnerabilityScorer/analyzer.py`)
8. **Factor Analysis** (via Analysis module)
9. **Pre-Trade Analysis** (via Analysis module)
10. **Alpha Trend Modeller** (visualization only, no metric changes needed)
11. **Edge Analysis** (specialized E-ratio/R-multiple metrics, documented reference)

## Usage

### Basic Usage

```python
from Classes.Core import CentralizedPerformanceMetrics

# Calculate all metrics from equity curve and trades
metrics = CentralizedPerformanceMetrics.calculate_all_metrics(
    equity_curve=equity_df,      # DataFrame with 'equity' column
    trades=trade_list,           # List of trade objects
    initial_capital=100000       # Initial capital
)

# Access individual metrics
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
```

### Individual Metric Calculations

```python
from Classes.Core import CentralizedPerformanceMetrics

# Equity curve metrics
sharpe = CentralizedPerformanceMetrics.calculate_sharpe_ratio(equity_df)
sortino = CentralizedPerformanceMetrics.calculate_sortino_ratio(equity_df)
max_dd, max_dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(equity_df)
volatility = CentralizedPerformanceMetrics.calculate_volatility(equity_df)

# Trade-based metrics
win_rate = CentralizedPerformanceMetrics.calculate_win_rate(trades)
profit_factor = CentralizedPerformanceMetrics.calculate_profit_factor(trades)
expectancy = CentralizedPerformanceMetrics.calculate_expectancy(trades)
```

### Using Custom Parameters

While standardized values are recommended, you can override parameters:

```python
# Custom risk-free rate (not recommended for cross-system comparison)
sharpe = CentralizedPerformanceMetrics.calculate_sharpe_ratio(
    equity_df,
    risk_free_rate=0.05,  # 5% instead of standard 3.5%
    trading_days=252
)
```

## Available Metrics

### Return Metrics
| Metric | Key | Description |
|--------|-----|-------------|
| Total Return | `total_return` | Total percentage return over period |
| Total Return ($) | `total_return_dollars` | Total dollar return |
| Annual Return | `annual_return` | Compound Annual Growth Rate (CAGR) |
| Final Equity | `final_equity` | Final portfolio value |

### Risk-Adjusted Return Metrics
| Metric | Key | Description |
|--------|-----|-------------|
| Sharpe Ratio | `sharpe_ratio` | Risk-adjusted return (rf=3.5%) |
| Sortino Ratio | `sortino_ratio` | Downside risk-adjusted return |
| Calmar Ratio | `calmar_ratio` | Annual return / Max drawdown |
| Recovery Factor | `recovery_factor` | Net profit / Max drawdown |

### Risk Metrics
| Metric | Key | Description |
|--------|-----|-------------|
| Max Drawdown ($) | `max_drawdown` | Maximum peak-to-trough decline |
| Max Drawdown (%) | `max_drawdown_pct` | Maximum decline as percentage |
| Volatility | `volatility` | Annualized standard deviation |
| Downside Deviation | `downside_deviation` | Volatility of negative returns |

### Trade Metrics
| Metric | Key | Description |
|--------|-----|-------------|
| Total Trades | `total_trades` | Number of completed trades |
| Win Rate | `win_rate` | Percentage of winning trades |
| Profit Factor | `profit_factor` | Gross profit / Gross loss |
| Avg Trade Return | `avg_trade_return` | Average return per trade |
| Avg Win | `avg_win` | Average winning trade return |
| Avg Loss | `avg_loss` | Average losing trade return |
| Expectancy | `expectancy` | Expected return per trade |
| Max Consecutive Wins | `max_consecutive_wins` | Longest winning streak |
| Max Consecutive Losses | `max_consecutive_losses` | Longest losing streak |
| Avg Trade Duration | `avg_trade_duration` | Average holding period (days) |

### Daily Metrics
| Metric | Key | Description |
|--------|-----|-------------|
| Best Day | `best_day` | Best single day return |
| Worst Day | `worst_day` | Worst single day return |

## Standardized Constants

The following constants are standardized across the framework:

```python
from Classes.Core import (
    DEFAULT_RISK_FREE_RATE,    # 0.035 (3.5%)
    TRADING_DAYS_PER_YEAR,     # 252
    MAX_PROFIT_FACTOR,         # 999.99
)
```

## Backward Compatibility

Existing code using the old `PerformanceMetrics` classes will continue to work. These classes now delegate to `CentralizedPerformanceMetrics` internally:

```python
# Old code still works - delegates to centralized class
from Classes.Analysis.performance_metrics import PerformanceMetrics

# These now use centralized calculations
sharpe = PerformanceMetrics.calculate_sharpe_ratio(equity_df)
sortino = PerformanceMetrics.calculate_sortino_ratio(equity_df)
```

```python
# Old backtesting code still works
from backtesting.metrics.performance import PerformanceMetrics

metrics = PerformanceMetrics(backtest_result)
sharpe = metrics.get('sharpe_ratio')  # Uses centralized calculation
```

## Metric Definitions

The module provides `MetricDefinition` dataclasses for formatting:

```python
from Classes.Core import METRIC_DEFINITIONS

# Get definition for a metric
sharpe_def = METRIC_DEFINITIONS['sharpe_ratio']
print(sharpe_def.name)           # "Sharpe Ratio"
print(sharpe_def.description)    # "Risk-adjusted return (annualized, rf=3.5%)"
print(sharpe_def.higher_is_better)  # True

# Format a value
formatted = sharpe_def.format(1.5)  # "1.500"
```

## Generating Summary Reports

```python
from Classes.Core import CentralizedPerformanceMetrics

metrics = CentralizedPerformanceMetrics.calculate_all_metrics(
    equity_curve=equity_df,
    trades=trades
)

# Generate formatted summary
summary = CentralizedPerformanceMetrics.get_summary_text(metrics, "My Strategy")
print(summary)
```

Output:
```
My Strategy
============================================================

Returns:
  Total Return:      15.32%
  Annual Return:     12.45%
  Final Equity:      $115,320.00

Risk Metrics:
  Max Drawdown:      8.25%
  Volatility:        15.67%

Risk-Adjusted Returns:
  Sharpe Ratio:      1.234
  Sortino Ratio:     1.567
  Calmar Ratio:      1.509

Trade Statistics:
  Total Trades:      156
  Win Rate:          58.33%
  Profit Factor:     1.85
  Expectancy:        0.98%
============================================================
```

## Specialized Metrics Not in Centralized Module

Some systems have specialized metrics that are NOT in the centralized module:

### Edge Analysis
- **E-ratio**: Entry edge measurement (MFE/MAE normalized by ATR)
- **R-multiple**: Risk-reward ratio (Exit-Entry)/(Entry-StopLoss)

These are trade-specific analytical tools, not standard portfolio metrics.

### Vulnerability Scorer
- **Consistency Percentage**: % of exit points where vulnerability P/L >= natural P/L
- **Trade-based Sharpe**: Per-trade Sharpe (different from daily equity curve Sharpe)

These use trade-by-trade analysis specific to vulnerability scoring.

### Factor Analysis
- Statistical metrics (t-tests, effect sizes, feature importance)
- These are statistical analysis tools, not performance metrics

## Migration Guide

If you have custom code that calculated metrics directly, update to use centralized:

### Before
```python
# Old way - custom calculation
returns = equity_curve['equity'].pct_change().dropna()
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
```

### After
```python
# New way - centralized calculation
from Classes.Core import CentralizedPerformanceMetrics

sharpe = CentralizedPerformanceMetrics.calculate_sharpe_ratio(equity_curve)
```

This ensures:
- Correct risk-free rate adjustment
- Consistent annualization
- Proper handling of edge cases
- Cross-system comparability

## Troubleshooting

### Import Errors

If you get import errors, ensure the Core module is initialized:

```python
# This should work
from Classes.Core import CentralizedPerformanceMetrics

# If not, try
from Classes.Core.performance_metrics import CentralizedPerformanceMetrics
```

### Different Results After Migration

If you see different metric values after migration, this is expected! The old calculations may have used:
- Different risk-free rates (2% vs 3.5%)
- Different profit factor caps
- Inconsistent edge case handling

The new values are **more accurate and consistent** across the framework.

## Contributing

When adding new metrics to the framework:

1. Add the calculation to `Classes/Core/performance_metrics.py`
2. Add a `MetricDefinition` entry in `METRIC_DEFINITIONS`
3. Update the `calculate_all_metrics()` method if appropriate
4. Update this documentation

---

*Last updated: January 2026*
*Version: 1.0*
