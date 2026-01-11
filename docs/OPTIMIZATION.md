# Optimization Guide

This guide explains how to find optimal strategy parameters using the framework's walk-forward optimization.

## Why Optimize?

Every strategy has parameters (e.g., moving average periods, stop loss distances). Different parameter values give different results. Optimization finds parameter values that perform well.

**The Problem:** You can always find parameters that work great on past data. But those parameters often fail on new data. This is called **overfitting**.

**The Solution:** Walk-forward optimization tests parameters on data they've never seen, so you know they'll work on future data too.

## How Walk-Forward Optimization Works

1. **Split data into training and testing periods**
2. **Optimize on training data** - Find the best parameters
3. **Test on testing data** - See how those parameters perform on unseen data
4. **Roll forward** - Move the window and repeat
5. **Analyze stability** - Check if good parameters stay good

```
|-------- Training --------|---- Testing ----|
                           |-------- Training --------|---- Testing ----|
                                                      |-------- Training --------|---- Testing ----|
```

If parameters work well on testing data (that was never used for optimization), they're more likely to work on future data.

---

## Using the Optimization GUI

### Starting the GUI

```bash
python optimize_gui.py
```

### Step-by-Step

1. **Select Strategy**
   Choose the strategy to optimize (e.g., AlphaTrendStrategy)

2. **Select Securities**
   Pick one or more securities to optimize on. Use "Select All" for comprehensive testing.

3. **Choose Speed Mode**
   - **Quick (25 iterations)** - Fast, for initial testing
   - **Fast (50 iterations)** - Balance of speed and quality
   - **Full (100 iterations)** - Best results, takes longer

4. **Enable Sensitivity Analysis** (recommended)
   Tests how robust parameters are to small changes

5. **Set Start Date (optional)**
   Filter to use only recent data

6. **Click "Start Optimization"**
   Watch progress in the log window

### Reading Results

When complete, you'll see:
- **Console summary** with recommended parameters
- **Excel report** in `logs/optimization_reports/`

---

## Understanding the Results

### Key Metrics

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **In-Sample Result** | Performance during training | Higher is better |
| **Out-of-Sample Result** | Performance on unseen data | Close to in-sample |
| **Degradation %** | How much performance drops | < 15% |
| **Success Rate** | % of windows that meet constraints | > 75% |

### Interpreting Degradation

Degradation = (In-Sample - Out-of-Sample) / In-Sample × 100%

| Degradation | Meaning |
|-------------|---------|
| < 15% | Excellent - Parameters are robust |
| 15-30% | Moderate - Use with caution |
| > 30% | Poor - Strategy may be overfit |

### Example Results

**Good results:**
```
Avg In-Sample Sortino: 2.45
Avg Out-Sample Sortino: 2.31
Degradation: 5.7%
Success Rate: 85%
```
→ Low degradation + high success rate = safe to use

**Bad results:**
```
Avg In-Sample Sortino: 3.12
Avg Out-Sample Sortino: 1.89
Degradation: 39.4%
Success Rate: 45%
```
→ High degradation + low success rate = overfit

---

## Excel Report Contents

The optimization creates an Excel file with 6 sheets:

### 1. Summary
- Overall success rate
- Average performance metrics
- Recommended parameters

### 2. Window Results
- Results for each training/testing window
- In-sample vs out-of-sample comparison
- Degradation per window

### 3. Parameter Stability
- How parameters varied across windows
- Stability assessment (Very Stable / Stable / Unstable)
- Standard deviation of each parameter

### 4. Sensitivity Analysis
- How performance changes with parameter variations
- Most and least robust parameters
- Robustness ratings

### 5. In-Sample vs Out-Sample
- Direct comparison chart
- Overfitting detection

### 6. Recommendations
- Suggested parameter values
- Robust parameter ranges
- Action items

---

## Using Recommended Parameters

### From the Excel Report

The "Recommendations" sheet shows:
```
Parameter: atr_stop_loss_multiple
Recommended Value: 2.5
Robust Range: 2.2 - 2.8
```

Use any value within the robust range. Don't obsess over exact values.

### In Your Strategy

```python
# Use recommended values
strategy = AlphaTrendStrategy(
    volume_short_ma=4,        # From report
    volume_long_ma=28,        # From report
    atr_stop_loss_multiple=2.5,  # From report
    risk_percent=2.0
)
```

### Via Presets

Save recommended parameters to `config/strategy_presets/`:
```json
{
  "strategy_class": "AlphaTrendStrategy",
  "parameters": {
    "volume_short_ma": 4,
    "volume_long_ma": 28,
    "atr_stop_loss_multiple": 2.5
  }
}
```

---

## Configuration Options

Edit `config/optimization_config.yaml`:

### Speed Modes

```yaml
bayesian_optimization:
  speed_mode: fast  # quick, fast, or full
```

| Mode | Iterations | Time | When to Use |
|------|-----------|------|-------------|
| quick | 25 | ~5 min | Initial testing |
| fast | 50 | ~15 min | Development |
| full | 100 | ~30 min | Final optimization |

### Parameter Ranges

Parameter ranges are loaded from the **centralized configuration** at `config/strategy_parameters.json`. This ensures all systems use consistent parameter definitions.

The walk-forward optimizer automatically loads parameters from the centralized config. You can override specific ranges in `config/optimization_config.yaml` if needed:

```yaml
# These values override the centralized config
strategy_parameters:
  AlphaTrendStrategy:
    volume_short_ma:
      min: 2
      max: 10
      type: int

    atr_stop_loss_multiple:
      min: 1.5
      max: 4.0
      type: float
```

**Programmatic access:**
```python
from config.strategy_config import StrategyConfig

# Get optimization ranges for all parameters
opt_params = StrategyConfig.get_optimization_params('AlphaTrendStrategy')
# Returns: {'volume_short_ma': {'type': 'int', 'min': 2, 'max': 20, ...}, ...}
```

### Constraints

Set minimum requirements:

```yaml
optimization:
  constraints:
    min_profit_factor: 1.5      # Must make 50% more than lose
    max_drawdown_percent: 25.0  # Max 25% drawdown
    min_trades_per_year: 1      # At least 1 trade per year
```

### Optimization Metric

What to maximize:

```yaml
optimization:
  primary_metric: sortino_ratio  # or sharpe_ratio, profit_factor, total_return_pct
```

---

## Optimization Workflow

### Phase 1: Quick Test

1. Run with **Quick mode**
2. Skip sensitivity analysis
3. Check if strategy works at all
4. Time: ~5 minutes

### Phase 2: Refinement

1. Run with **Fast mode**
2. Enable sensitivity analysis
3. Review parameter stability
4. Adjust parameter ranges if needed
5. Time: ~15 minutes

### Phase 3: Final Optimization

1. Run with **Full mode**
2. Enable sensitivity analysis
3. Generate final recommendations
4. Time: ~30 minutes

### Phase 4: Validation

1. Run a regular backtest with recommended parameters
2. Compare results to out-of-sample metrics
3. Test on new securities not used in optimization

---

## Common Issues

### "Not enough data to create windows"

**Cause:** Your data is too short.

**Solution:** You need at least:
- Training period (365 days) + Testing period (365 days) = 730 days
- Recommended: 3+ years of data

### All windows fail constraints

**Cause:** Constraints are too strict or strategy doesn't work.

**Solutions:**
1. Relax constraints (lower min_profit_factor, increase max_drawdown)
2. Run a regular backtest first to check if strategy works
3. Widen parameter ranges

### Parameters are unstable

**Cause:** Optimal values vary widely across windows.

**Solutions:**
1. Simplify the strategy
2. Narrow parameter ranges to known-good areas
3. Increase training window size
4. Strategy may inherently be unstable

### Optimization is slow

**Solutions:**
1. Use Quick mode for testing
2. Skip sensitivity analysis during development
3. Optimize fewer securities at once
4. On Linux/Mac: Use multiple CPU cores (`n_jobs: 4`)

---

## Best Practices

### 1. Don't Chase Perfect Parameters

- Use robust ranges, not exact values
- Round to sensible numbers (2.5, not 2.4873)
- Slightly suboptimal but stable is better than "optimal" but fragile

### 2. Validate on Fresh Data

- After optimization, test on data that was never seen
- If possible, paper trade for a period before going live
- Re-optimize periodically (every 6-12 months, not weekly)

### 3. Watch for Red Flags

- Degradation > 30% = overfit
- Unstable parameters = unreliable strategy
- Low success rate = constraints too strict or strategy doesn't work

### 4. Document Everything

- Save Excel reports
- Record which parameters you chose
- Track actual performance vs. predicted

### 5. Re-optimize Carefully

- Don't re-optimize after every losing trade
- Recommended: Once every 6-12 months
- Frequent re-optimization causes overfitting

---

## Programmatic Optimization

For advanced users who want to run optimization from Python:

```python
from pathlib import Path
from Classes.Optimization.optimizer import StrategyOptimizer
from Classes.Config.config import BacktestConfig, OptimizationConfig
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Define parameter grid
param_grid = {
    'volume_short_ma': [2, 4, 6, 8],
    'volume_long_ma': [20, 25, 30, 35],
    'atr_stop_loss_multiple': [2.0, 2.5, 3.0]
}

# Configure
opt_config = OptimizationConfig(
    metric='sharpe_ratio',
    per_security=False,
    maximize=True,
    min_trades=5
)

backtest_config = BacktestConfig(initial_capital=100000.0)

# Run optimization
optimizer = StrategyOptimizer(opt_config, backtest_config)
results = optimizer.optimize(
    strategy_class=AlphaTrendStrategy,
    param_grid=param_grid,
    symbol='AAPL',
    data=data
)

# Get best parameters
best = results.get_best()
print(f"Best parameters: {best.parameters}")
print(f"Sharpe Ratio: {best.metric_value:.2f}")
```

---

## Related Guides

- **[User Guide](USER_GUIDE.md)** - Getting started
- **[Configuration Guide](CONFIGURATION.md)** - All configuration options
- **[Strategies Guide](STRATEGIES.md)** - Strategy parameters
