# Walk-Forward Optimization Guide

## Overview

The walk-forward optimization feature prevents overfitting by:
1. Dividing data into rolling training/testing windows
2. Optimizing parameters only on in-sample (training) data using Bayesian optimization
3. Testing those parameters on truly out-of-sample (testing) data
4. Rolling forward through time and repeating
5. Analyzing parameter stability and robustness

This approach ensures that optimized parameters will perform well on unseen future data.

## Quick Start

### 1. Launch the Optimization GUI

```bash
python optimize_gui.py
```

Or:

```bash
python -m optimization_gui
```

### 2. Configure Your Optimization

1. **Select Strategy**: Choose the strategy you want to optimize (e.g., AlphaTrendStrategy)
2. **Select Securities**: Choose one or more securities to optimize
   - Click "Select All" to test all available securities
   - Hold Ctrl/Cmd to select multiple individual securities
3. **Enable Sensitivity Analysis**: Keep this checked to analyze parameter robustness
4. **Click "Start Optimization"**: Begin the walk-forward optimization process

### 3. Monitor Progress

The GUI will show:
- Current window being optimized
- Progress bar with percentage completion
- Real-time log of optimization steps
- Results for each window

### 4. Review Results

When complete, you'll get:
- Console output summarizing results
- Excel report saved to `logs/optimization_reports/`
- Recommended parameters with robust ranges

## Configuration

All optimization settings are in `config/optimization_config.yaml`.

### Walk-Forward Settings

```yaml
walk_forward:
  training_period_days: 365    # 1 year training window
  testing_period_days: 365     # 1 year testing window
  step_size_min_days: 7        # Minimum roll-forward step
  step_size_max_days: 30       # Maximum roll-forward step
```

**Explanation**:
- **Training period**: How much historical data to use for optimization
- **Testing period**: How much future data to test the optimized parameters on
- **Step size**: How far to move forward between windows (randomized to prevent bias)

### Bayesian Optimization Settings

```yaml
bayesian_optimization:
  n_iterations: 100            # Number of parameter sets to test per window
  n_initial_points: 20         # Random explorations before Bayesian search
  random_state: 42             # Seed for reproducibility (or null for random)
```

**Explanation**:
- More iterations = better optimization but slower
- Initial points ensure good coverage of parameter space
- Random state allows reproducible results

### Optimization Constraints

```yaml
optimization:
  primary_metric: sortino_ratio    # What to maximize

  constraints:
    min_profit_factor: 1.5         # Wins must exceed losses by 50%
    max_drawdown_percent: 25.0     # Maximum acceptable drawdown
    min_trades_per_year: 1         # Minimum trading activity
```

**Explanation**:
- **Primary metric**: What you're optimizing for (Sortino ratio = risk-adjusted returns)
- **Constraints**: Hard requirements that must be met
- If constraints are violated, that parameter set is rejected

### Parameter Ranges

Define the search space for each strategy parameter:

```yaml
strategy_parameters:
  AlphaTrendStrategy:
    volume_short_ma:
      min: 2
      max: 10
      type: int
      step: 1

    atr_stop_loss_multiple:
      min: 1.5
      max: 4.0
      type: float
      step: 0.25
```

**Explanation**:
- **min/max**: Bounds for the parameter
- **type**: int or float
- **step**: Optional, used for grid search fallback

### Sensitivity Analysis

```yaml
sensitivity_analysis:
  float_variation_percent: 10.0   # Test ±10% for floats
  int_variation_amount: 1         # Test ±1 for integers
  test_mode: combinations         # "individual" or "combinations"
  max_sensitivity_tests: 1000     # Limit on total tests
```

**Explanation**:
- Tests how robust parameters are to small changes
- **Individual mode**: Test each parameter separately (faster)
- **Combinations mode**: Test all parameter variations together (more thorough)

## Understanding the Results

### Excel Report Sheets

The generated Excel report contains 6 sheets:

#### 1. Summary
- High-level overview
- Success rate across all windows
- Average in-sample vs out-of-sample performance
- Recommended parameters

#### 2. Window Results
- Detailed results for each walk-forward window
- In-sample and out-of-sample metrics
- Performance degradation percentages
- Charts comparing in-sample vs out-of-sample

#### 3. Parameter Stability
- How parameter values varied across windows
- Stability assessment (Very Stable / Stable / Unstable)
- Full table of parameters for all windows

#### 4. Sensitivity Analysis
- Overall robustness assessment
- Most and least robust parameters
- Parameter-by-parameter sensitivity metrics

#### 5. In-Sample vs Out-Sample
- Direct comparison of training vs testing performance
- Overfitting detection (degradation > 30% = overfitting)

#### 6. Recommendations
- Recommended parameters for live trading
- Robust parameter ranges
- Key findings
- Actionable next steps

### Interpreting Results

#### ✅ Good Signs (Parameters are Robust)
- **Sortino/Sharpe degradation < 15%**: Excellent generalization
- **High success rate (>80%)**: Parameters work in most market conditions
- **Low parameter standard deviation**: Optimal values are consistent
- **Sensitivity analysis shows "Robust"**: Small changes don't hurt performance

#### ⚠️ Warning Signs (Use Caution)
- **Degradation 15-30%**: Moderate overfitting, monitor carefully
- **Moderate success rate (50-80%)**: Works in some conditions
- **Parameter values jump around**: Unstable optimization
- **Sensitivity shows "Moderate"**: Be careful with parameter selection

#### ❌ Red Flags (Strategy May Be Overfit)
- **Degradation > 30%**: Severe overfitting
- **Low success rate (<50%)**: Constraints too strict or strategy doesn't work
- **Parameters vary wildly**: No stable optimal values exist
- **Sensitivity shows "Unstable"**: Performance is fragile

### Example Interpretation

```
Avg In-Sample Sortino: 2.45
Avg Out-Sample Sortino: 2.31
Sortino Degradation: 5.7%

Success Rate: 85% (17/20 windows)
```

**Interpretation**:
- ✅ Excellent! Only 5.7% degradation means parameters generalize well
- ✅ 85% success rate indicates robust performance
- ✅ Safe to use these parameters for live trading

```
Avg In-Sample Sortino: 3.12
Avg Out-Sample Sortino: 1.89
Sortino Degradation: 39.4%

Success Rate: 45% (9/20 windows)
```

**Interpretation**:
- ❌ 39.4% degradation = severe overfitting
- ❌ Only 45% success rate = unreliable
- ❌ Do NOT use for live trading - strategy needs rework

## Optimization Workflow

### Step 1: Initial Optimization
1. Run optimization with default settings
2. Review the Excel report
3. Check degradation percentage and success rate

### Step 2: Assess Results

**If results are good (degradation < 15%, success rate > 80%)**:
- Proceed to forward testing with recommended parameters
- Monitor sensitivity analysis for robust ranges
- Document parameter ranges for live trading

**If results are moderate (degradation 15-30%)**:
- Review sensitivity analysis
- Use most robust parameters only
- Consider adding more constraints
- Increase training window size

**If results are poor (degradation > 30%)**:
- Strategy may be too complex (overfitting)
- Try simplifying the strategy
- Increase training period
- Reduce parameter search space
- Add stricter constraints

### Step 3: Parameter Selection

From the report, use:
1. **Recommended Value**: Median value across all windows
2. **Robust Range**: Min-max range where performance is stable

Example:
```
Parameter: atr_stop_loss_multiple
Recommended Value: 2.5
Robust Range: 2.2 - 2.8
```

For live trading, you can safely use any value between 2.2 and 2.8.

### Step 4: Forward Testing

Before live trading:
1. Test recommended parameters on new data (after optimization period)
2. Verify performance matches out-of-sample results
3. Monitor for 1-3 months
4. Re-optimize periodically (every 6-12 months)

## Advanced Features

### Custom Constraints

Edit `optimization_config.yaml` to change constraints:

```yaml
constraints:
  min_profit_factor: 2.0          # More aggressive
  max_drawdown_percent: 15.0      # Tighter risk control
  min_trades_per_year: 5          # More active trading
```

### Custom Parameter Ranges

Add or modify parameter ranges:

```yaml
strategy_parameters:
  AlphaTrendStrategy:
    new_parameter:
      min: 10
      max: 100
      type: int

    # Or use specific values
    another_parameter:
      values: [0.5, 1.0, 1.5, 2.0, 2.5]
      type: float
```

### Optimization Metrics

Change what you're optimizing for:

```yaml
optimization:
  primary_metric: sharpe_ratio     # Instead of sortino_ratio
  # Other options: profit_factor, calmar_ratio, total_return_pct
```

## Troubleshooting

### "Not enough data to create walk-forward windows"
**Solution**: Your data is too short. You need at least:
- Training period (365 days) + Testing period (365 days) = ~730 days minimum
- Recommended: 3+ years of data for reliable results

### Optimization is very slow
**Solutions**:
1. Reduce `n_iterations` in config (e.g., from 100 to 50)
2. Use `individual` instead of `combinations` for sensitivity analysis
3. Reduce `max_sensitivity_tests`
4. Optimize fewer securities at once

### All windows fail constraints
**Solutions**:
1. Relax constraints (lower `min_profit_factor`, increase `max_drawdown_percent`)
2. Check if your strategy is working at all (run a regular backtest first)
3. Expand parameter search ranges

### Parameters are unstable (high standard deviation)
**Causes**:
1. Too many parameters (overfitting)
2. Parameter ranges too wide
3. Strategy inherently unstable

**Solutions**:
1. Simplify the strategy
2. Narrow parameter ranges to known-good areas
3. Increase training window size
4. Add more constraints

### Sensitivity analysis shows "Unstable" for most parameters
**Interpretation**: Strategy performance is fragile
**Solutions**:
1. Focus on parameters marked as "Robust"
2. Fix sensitive parameters to their median values
3. Consider strategy redesign

## Best Practices

### 1. Start Simple
- Begin with a few securities
- Use default config settings
- Understand results before customizing

### 2. Validate Externally
- Always forward test on new data
- Compare out-of-sample results to real performance
- Re-optimize periodically (but not too often)

### 3. Watch for Red Flags
- High degradation = overfitting
- Unstable parameters = unreliable strategy
- Low success rate = strategy doesn't work

### 4. Use Robust Ranges
- Don't obsess over exact "optimal" values
- Use any value within the robust range
- Round to sensible numbers (e.g., 2.5 instead of 2.4873)

### 5. Document Everything
- Save optimization reports
- Track which parameters you used when
- Record actual performance vs. predicted

### 6. Re-optimize Carefully
- Don't re-optimize too frequently (causes overfitting)
- Recommended: Once every 6-12 months
- Always compare new results to old results

## Common Questions

### Q: How long does optimization take?
**A**: Depends on:
- Data size: More data = longer
- Number of windows: Default ~5-10 windows
- Iterations: 100 per window by default
- Sensitivity tests: Up to 1000

**Typical times**:
- Single security: 10-30 minutes
- Multiple securities: 30-120 minutes

### Q: Can I optimize multiple strategies at once?
**A**: No, optimize one strategy at a time. Run the GUI multiple times for different strategies.

### Q: What if I don't have enough data?
**A**: You need at least 2-3 years of data. With less:
- Reduce training and testing periods in config
- Results will be less reliable
- Consider using longer timeframes (weekly instead of daily)

### Q: Should I always use the "optimal" parameters?
**A**: No! Use the robust range, not exact values:
- ✅ Good: "Use 2.0 to 3.0, I'll pick 2.5"
- ❌ Bad: "Use exactly 2.4873916"

Overly precise values suggest overfitting.

### Q: How do I know if my strategy is good?
**A**: Look for:
- Out-of-sample Sortino > 1.5
- Degradation < 15%
- Success rate > 75%
- Stable parameters across windows

### Q: When should I re-optimize?
**A**:
- ✅ Good times: After 6-12 months, market regime changes
- ❌ Bad times: After every losing trade, every week

Frequent re-optimization causes overfitting!

## Support

For issues or questions:
1. Check this guide first
2. Review the Excel report "Recommendations" sheet
3. Check the console output for error messages
4. Review `config/optimization_config.yaml` settings

## Technical Details

### Bayesian Optimization
Uses Gaussian Process regression to intelligently search parameter space:
1. Starts with random explorations
2. Builds a model of parameter → performance
3. Focuses search on promising areas
4. More efficient than grid search (100 tests instead of 10,000+)

### Walk-Forward Methodology
Prevents overfitting by:
1. Training only on historical data (in-sample)
2. Testing on future data that optimization never saw (out-of-sample)
3. Rolling forward through time
4. Measuring degradation from in-sample to out-of-sample

If degradation is low, parameters are robust and will work in live trading.

### Sensitivity Analysis
Tests robustness by:
1. Varying each parameter slightly (±10% or ±1)
2. Measuring performance changes
3. Identifying fragile vs. robust parameters

Robust parameters show stable performance (green in report).
Sensitive parameters show large swings (red in report).

## Conclusion

Walk-forward optimization with Bayesian optimization is the gold standard for preventing overfitting. By following this guide and interpreting results carefully, you can develop strategies with robust parameters that will perform well in live trading.

Remember: The goal is not to find "perfect" parameters, but to find **robust** parameters that work across different market conditions.
