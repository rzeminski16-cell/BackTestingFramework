# Optimization Guide

This guide explains how to find optimal strategy parameters using the framework's optimization tools.

---

## Optimization Methods Overview

The framework provides two optimization approaches:

| Method | Best For | Approach |
|--------|----------|----------|
| **Univariate Optimization** | Understanding individual parameter impact | Test one parameter at a time, others at control values |
| **Walk-Forward Optimization** | Finding robust parameters for live trading | Multi-parameter search with out-of-sample validation |

---

## Univariate Optimization

Univariate optimization tests **one parameter at a time** while keeping all other parameters at fixed "control" values. This helps you understand how each parameter individually affects strategy performance.

### Why Use Univariate Optimization?

- **Understand parameter sensitivity**: See exactly how each parameter affects performance
- **Identify optimal ranges**: Find the sweet spot for each parameter
- **Avoid overfitting**: Simpler than multi-parameter optimization
- **Visual analysis**: Line charts show clear relationships between parameter values and metrics

### Launch

From the main menu or directly:

```bash
python ctk_main_gui.py
# Click "Univariate Optimization"
```

Or launch directly:

```bash
python ctk_univariate_optimization_gui.py
```

### Step-by-Step Workflow

#### 1. Select Securities

Choose between two modes:

| Mode | Description |
|------|-------------|
| **Single Security** | Test on one ticker |
| **Portfolio** | Test on multiple tickers |

For portfolio mode, you can run securities:
- **Together**: Combined results across all securities
- **Separately**: Individual results per security

#### 2. Select Strategy

Choose the strategy to optimize (e.g., AlphaTrendStrategy).

#### 3. Configure Parameters

For each strategy parameter, you'll see a control panel:

```
┌─────────────────────────────────────────────────────────┐
│ ☑ volume_short_ma                                       │
│                                                          │
│ Control Value: [4]                                       │
│                                                          │
│ Range: [2] ════════════════════════════════════ [20]    │
│        Min           Slider                    Max       │
│                                                          │
│ Interval: [1]                                            │
└─────────────────────────────────────────────────────────┘
```

- **Checkbox**: Enable/disable parameter for optimization
- **Control Value**: The baseline value (used when testing other parameters)
- **Range Slider**: Set min and max values to test
- **Interval**: Step size between test values

**Example**: If min=2, max=10, interval=2, the optimizer tests: 2, 4, 6, 8, 10

#### 4. Select Metrics

Choose which performance metrics to calculate:

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall percentage gain/loss |
| **Annual Return** | Annualized return percentage |
| **Sharpe Ratio** | Risk-adjusted return (volatility) |
| **Sortino Ratio** | Risk-adjusted return (downside only) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | Return / Max Drawdown |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Total Trades** | Number of trades executed |
| **Avg Trade Return** | Mean return per trade |
| **Expectancy** | Expected value per trade |
| **Volatility** | Standard deviation of returns |
| **Final Equity** | Ending portfolio value |

**Tip**: Select 3-5 key metrics to keep charts readable.

#### 5. Configure Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **Initial Capital** | Starting portfolio value | 100,000 |
| **Commission (%)** | Trading cost percentage | 0.1% |
| **Slippage (%)** | Execution slippage | 0.1% |
| **Start Date** | Filter data from this date (optional) | - |
| **End Date** | Filter data to this date (optional) | - |

**Date Range**: Leave blank to use all available data, or specify dates in `YYYY-MM-DD` format to test on a specific timeframe.

#### 6. Run Optimization

Click **"Run Optimization"** to start. Progress is shown in the log panel:

```
Strategy: AlphaTrendStrategy
Securities: AAPL, MSFT, GOOGL
Run Mode: together
Parameters to optimize: 3
Metrics: total_return, sharpe_ratio, max_drawdown
Date Range: 2020-01-01 to 2023-12-31

Testing parameter: volume_short_ma
  Testing value 2/10...
  Testing value 3/10...
  ...
```

#### 7. Export Results

When complete, click **"Export to Excel"** to generate the report.

### Excel Report Contents

The report contains multiple sheets:

#### Summary Sheet

Contains:
- **Optimization Settings**: Strategy, securities, date range, capital settings
- **Control Values**: Baseline value for each parameter
- **Best Values by Metric**: Optimal parameter value for each metric

Example:
```
BEST VALUES BY METRIC
Parameter          | Control | Total Return | Sharpe Ratio | Max Drawdown
-------------------|---------|--------------|--------------|-------------
volume_short_ma    |    4    |      6       |      4       |      3
volume_long_ma     |   30    |     25       |     30       |     35
atr_stop_multiple  |  2.5    |    2.0       |    2.5       |    3.0
```

#### Parameter Sheets

One sheet per optimized parameter, containing:

1. **Data Table**: All tested values with metric results

```
Parameter Value | Total Return | Sharpe Ratio | Max Drawdown | Win Rate
----------------|--------------|--------------|--------------|----------
        2       |    12.5%     |     1.2      |    -15.3%    |   45%
        3       |    18.2%     |     1.5      |    -12.1%    |   48%
        4       |    15.8%     |     1.4      |    -13.5%    |   47%
        ...
```

2. **Line Charts**: Visual representation of how each metric changes across parameter values

```
    Sharpe Ratio
    │
1.5 │         ●
    │       ●   ●
1.3 │     ●       ●
    │   ●           ●
1.1 │ ●               ●
    └─────────────────────
      2   3   4   5   6   7   8
           volume_short_ma
```

The charts feature:
- Clear x-axis (parameter values) and y-axis (metric values)
- Simple black line with small circular markers
- Professional, minimal styling

### Interpreting Results

#### Finding Optimal Values

Look for parameter values where:
- Multiple metrics perform well simultaneously
- Performance is stable (not a sharp peak)
- Values make intuitive sense for the strategy

#### Identifying Sensitivity

Parameters are **sensitive** if small changes cause large performance swings:

```
High Sensitivity (Risky)          Low Sensitivity (Robust)
         │                                  │
    ●    │                         ●────●────●────●────●
   / \   │                                  │
  ●   ●  │                                  │
 /     \ │                                  │
●       ●│                                  │
─────────┴──────                   ─────────┴──────
```

Prefer parameters with **low sensitivity** - they're more robust to market changes.

#### Common Patterns

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| **Flat line** | Parameter has little impact | Use default or remove from strategy |
| **Clear peak** | Optimal value exists | Use peak value, but verify stability |
| **Monotonic increase** | Higher is always better | Consider if there's a natural limit |
| **U-shape** | Extremes are bad | Use middle values |
| **Noisy/jagged** | Unstable or overfit | Increase data, simplify strategy |

### Best Practices

1. **Start with control values from backtesting**: Use values that already work reasonably well

2. **Use sensible ranges**: Don't test values that don't make sense (e.g., negative periods)

3. **Appropriate intervals**: Too fine = slow and noisy; too coarse = miss optimal values

4. **Multiple securities**: Test on several securities to ensure robustness

5. **Out-of-sample validation**: After finding optimal values, test on data not used in optimization

6. **Don't over-optimize**: If you need very specific values to get good results, the strategy may be overfit

### Example Workflow

```
1. Initial Setup
   - Select: AAPL, MSFT, GOOGL (portfolio, together)
   - Strategy: AlphaTrendStrategy
   - Date Range: 2020-01-01 to 2023-12-31

2. Parameter Configuration
   - volume_short_ma: control=4, range=2-10, interval=1
   - volume_long_ma: control=30, range=15-50, interval=5
   - atr_stop_multiple: control=2.5, range=1.5-4.0, interval=0.5

3. Metrics Selection
   - Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor

4. Run and Export

5. Analysis
   - volume_short_ma: Best at 4-5, stable curve → keep default
   - volume_long_ma: Best at 25-30, slight improvement → use 28
   - atr_stop_multiple: Best at 2.0-2.5, sensitive above 3.0 → use 2.5

6. Validation
   - Run backtest with new values on 2024 data (out-of-sample)
   - Compare results to original parameters
```

---

## Walk-Forward Optimization

Walk-forward optimization is a more sophisticated approach that helps prevent overfitting by testing parameters on data they've never seen.

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

### Launch

```bash
python optimize_gui.py
```

### Step-by-Step

1. **Select Strategy** - Choose the strategy to optimize (e.g., AlphaTrendStrategy)

2. **Select Securities** - Pick one or more securities. Use "Select All" for comprehensive testing.

3. **Choose Speed Mode**
   - **Quick (25 iterations)** - Fast, for initial testing
   - **Fast (50 iterations)** - Balance of speed and quality
   - **Full (100 iterations)** - Best results, takes longer

4. **Enable Sensitivity Analysis** (recommended) - Tests how robust parameters are to small changes

5. **Set Start Date (optional)** - Filter to use only recent data

6. **Click "Start Optimization"** - Watch progress in the log window

### Results

When complete:
- Console summary with recommended parameters
- Excel report in `logs/optimization_reports/`

---

## Understanding Results

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **In-Sample Result** | Performance during training | Higher is better |
| **Out-of-Sample Result** | Performance on unseen data | Close to in-sample |
| **Degradation %** | How much performance drops | < 15% |
| **Success Rate** | % of windows meeting constraints | > 75% |

### Interpreting Degradation

```
Degradation = (In-Sample - Out-of-Sample) / In-Sample × 100%
```

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
Low degradation + high success rate = safe to use

**Bad results:**
```
Avg In-Sample Sortino: 3.12
Avg Out-Sample Sortino: 1.89
Degradation: 39.4%
Success Rate: 45%
```
High degradation + low success rate = overfit

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

Parameter ranges are automatically loaded from `config/strategy_parameters.json` - the single source of truth for all strategy parameters. This ensures consistency across backtesting, univariate optimization, and walk-forward optimization.

To override specific ranges for optimization, edit `config/optimization_config.yaml`:

```yaml
# Overrides for specific parameters (optional)
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

> **Note**: If you don't specify overrides, the default ranges from `config/strategy_parameters.json` are used automatically.

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

## Best Practices

### 1. Don't Chase Perfect Parameters

- Use robust ranges, not exact values
- Round to sensible numbers (2.5, not 2.4873)
- Slightly suboptimal but stable is better than "optimal" but fragile

### 2. Validate on Fresh Data

- After optimization, test on data that was never seen
- Paper trade for a period before going live
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

## Common Issues

### "Not enough data to create windows"

**Cause:** Data is too short.

**Solution:** You need at least:
- Training period (365 days) + Testing period (365 days) = 730 days
- Recommended: 3+ years of data

### All windows fail constraints

**Cause:** Constraints are too strict or strategy doesn't work.

**Solutions:**
1. Relax constraints (lower min_profit_factor, increase max_drawdown)
2. Run a regular backtest first to verify strategy works
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

## Programmatic Optimization

For advanced users:

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
