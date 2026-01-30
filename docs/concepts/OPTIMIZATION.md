# Optimization

Optimization helps find strategy parameters that improve performance without overfitting to historical data.

---

## The Optimization Challenge

Every strategy has parameters (e.g., moving average period, ATR multiplier, stop loss percentage). Finding good values is critical:

```
                    THE OPTIMIZATION DILEMMA

    Too Little Optimization              Too Much Optimization
    ┌─────────────────────┐              ┌─────────────────────┐
    │                     │              │                     │
    │  Uses arbitrary     │              │  Perfect fit to     │
    │  parameter values   │              │  historical data    │
    │                     │              │                     │
    │  May leave money    │              │  Fails on new data  │
    │  on the table       │              │  (overfitting)      │
    │                     │              │                     │
    └─────────────────────┘              └─────────────────────┘
              │                                    │
              │                                    │
              └──────────────┬─────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  GOAL: Find robust  │
                  │  parameters that    │
                  │  work on unseen     │
                  │  data               │
                  └─────────────────────┘
```

---

## Optimization Approaches

The framework provides two complementary approaches:

### Univariate Optimization

Tests one parameter at a time while holding others constant:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      UNIVARIATE OPTIMIZATION                                │
│                                                                             │
│   Control Values: atr_mult=2.0, period=14, threshold=50                    │
│                                                                             │
│   Test 1: Vary atr_mult (1.0 to 5.0, step 0.5)                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  atr_mult  │  1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0  │  │
│   │  Sharpe    │  0.8   1.1   1.3   1.4   1.2   1.0   0.9   0.7   0.5  │  │
│   │            │              ▲     ★                                    │  │
│   │            │           control  best                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Test 2: Vary period (5 to 30, step 5)                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  period    │   5    10    14    20    25    30                      │  │
│   │  Sharpe    │  0.9   1.2   1.3   1.4   1.3   1.1                     │  │
│   │            │             ▲      ★                                    │  │
│   │            │          control  best                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Benefits:                                                                 │
│   • Clear visualization of parameter impact                                │
│   • Easy to understand parameter sensitivity                               │
│   • Fast execution (fewer combinations)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Best For:**
- Understanding individual parameter behavior
- Initial exploration of parameter ranges
- Identifying parameters that matter most

### Walk-Forward Optimization

Tests parameters across rolling time windows to prevent overfitting:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WALK-FORWARD OPTIMIZATION                               │
│                                                                             │
│   Historical Data: 2018 ─────────────────────────────────────────► 2023   │
│                                                                             │
│   Period 1:                                                                 │
│   ┌────────────────────────────────┬─────────────────┐                     │
│   │  TRAINING (In-Sample)          │  TEST (OOS)     │                     │
│   │  2018-01 to 2019-12           │  2020-01 to 06  │                     │
│   │  Find best params             │  Validate       │                     │
│   └────────────────────────────────┴─────────────────┘                     │
│                                                                             │
│   Period 2:                                                                 │
│        ┌────────────────────────────────┬─────────────────┐                │
│        │  TRAINING (In-Sample)          │  TEST (OOS)     │                │
│        │  2018-07 to 2020-06           │  2020-07 to 12  │                │
│        └────────────────────────────────┴─────────────────┘                │
│                                                                             │
│   Period 3:                                                                 │
│              ┌────────────────────────────────┬─────────────────┐          │
│              │  TRAINING (In-Sample)          │  TEST (OOS)     │          │
│              │  2019-01 to 2020-12           │  2021-01 to 06  │          │
│              └────────────────────────────────┴─────────────────┘          │
│                                                                             │
│   ... continues rolling forward                                            │
│                                                                             │
│   Benefits:                                                                 │
│   • Tests parameter stability over time                                    │
│   • Out-of-sample validation built in                                      │
│   • Reveals if strategy degrades over time                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Best For:**
- Final parameter selection
- Testing robustness across market conditions
- Avoiding overfitting

---

## Bayesian Optimization

Within walk-forward periods, the framework uses Bayesian optimization to efficiently search parameter space:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BAYESIAN SEARCH PROCESS                                │
│                                                                             │
│   Parameter Space:                                                          │
│   ├── atr_mult: 1.0 to 5.0                                                 │
│   ├── period: 5 to 50                                                      │
│   └── threshold: 30 to 70                                                  │
│                                                                             │
│   Iteration 1-10: Random Exploration                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Try random combinations to understand the landscape                │  │
│   │  Results: Build initial model of parameter → performance            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Iteration 11+: Guided Search                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Model predicts: "This region likely has better performance"        │  │
│   │  Algorithm balances:                                                │  │
│   │  • Exploitation: Test near known good values                       │  │
│   │  • Exploration: Test uncertain regions                             │  │
│   │  Each result improves the model                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Advantages over Grid Search:                                             │
│   • 10x fewer iterations needed                                            │
│   • Finds good regions faster                                              │
│   • Handles high-dimensional spaces                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Optimization Workflow

### Recommended Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION WORKFLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    STEP 1: BASELINE
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Run backtest with default parameters                                  │
    │  Record: Sharpe, Return, Drawdown, Win Rate                           │
    │  This is your baseline to beat                                        │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    STEP 2: UNIVARIATE EXPLORATION
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Test each parameter independently                                     │
    │  Identify:                                                             │
    │  • Which parameters have the most impact                              │
    │  • Reasonable ranges for each parameter                               │
    │  • Parameters that don't matter (flat response)                       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    STEP 3: WALK-FORWARD OPTIMIZATION
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Configure:                                                            │
    │  • Training period length (e.g., 2 years)                             │
    │  • Test period length (e.g., 6 months)                                │
    │  • Number of walk-forward periods                                     │
    │                                                                         │
    │  Run optimization on important parameters                              │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    STEP 4: ANALYZE RESULTS
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Check for:                                                            │
    │  • Parameter stability (do values change wildly?)                     │
    │  • Out-of-sample vs in-sample performance gap                         │
    │  • Consistent improvement across periods                              │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    STEP 5: FINAL VALIDATION
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Run full backtest with optimized parameters                          │
    │  Compare to baseline                                                  │
    │  Verify improvement is real, not noise                               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

---

## Optimization Metrics

Choose the right metric for your goals:

| Metric | Optimizes For | Best When |
|--------|---------------|-----------|
| **Sharpe Ratio** | Risk-adjusted returns | Want consistent performance |
| **Total Return** | Maximum profit | Risk tolerance is high |
| **Calmar Ratio** | Return vs max drawdown | Drawdown is primary concern |
| **Sortino Ratio** | Return vs downside risk | Upside volatility is okay |
| **Profit Factor** | Profit efficiency | Want reliable edge |
| **Win Rate** | Trade success rate | Psychology matters |

### Multi-Metric Consideration

Avoid optimizing for a single metric in isolation:

```
Parameter Set A:          Parameter Set B:
├── Sharpe: 2.1 ★         ├── Sharpe: 1.8
├── Drawdown: -35%        ├── Drawdown: -15% ★
├── Win Rate: 35%         ├── Win Rate: 52% ★
└── Trades: 12            └── Trades: 45

Which is better? Depends on your priorities.
Consider creating a composite score or setting minimum thresholds.
```

---

## Detecting Overfitting

Signs that optimization has gone too far:

### Large In-Sample vs Out-of-Sample Gap

```
In-Sample Performance:    Out-of-Sample Performance:
├── Sharpe: 2.5           ├── Sharpe: 0.8  ⚠️ Large drop
├── Return: +45%          ├── Return: +8%  ⚠️ Large drop
└── Win Rate: 68%         └── Win Rate: 42%

This gap suggests overfitting to historical data.
```

### Unstable Parameters Across Periods

```
Walk-Forward Results:
Period 1: atr_mult = 2.5, period = 14
Period 2: atr_mult = 4.2, period = 8   ⚠️ Big change
Period 3: atr_mult = 1.8, period = 32  ⚠️ Big change
Period 4: atr_mult = 3.9, period = 11  ⚠️ Big change

Wildly different "optimal" values suggest noise, not signal.
```

### Edge Case Parameters

```
Optimized values at extremes:
├── atr_mult = 0.51 (near minimum of 0.5)  ⚠️ Edge
├── period = 49 (near maximum of 50)        ⚠️ Edge
└── threshold = 70 (at maximum)             ⚠️ Edge

Parameters at range limits often indicate overfitting.
```

---

## Sensitivity Analysis

Understanding how sensitive performance is to parameter changes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SENSITIVITY ANALYSIS                                   │
│                                                                             │
│   GOOD: Flat/Stable Response                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Sharpe │                                                           │  │
│   │   2.0   │         ════════════════════                             │  │
│   │   1.5   │    ═════                    ═════                        │  │
│   │   1.0   │ ═══                              ═══                     │  │
│   │         │                                                           │  │
│   │         └─────────────────────────────────────────────────────────  │  │
│   │              1.0   1.5   2.0   2.5   3.0   3.5   4.0  (atr_mult)  │  │
│   │                                                                     │  │
│   │   Wide range of good values = robust parameter                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   BAD: Spike Response                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Sharpe │                                                           │  │
│   │   2.0   │                     ║                                    │  │
│   │   1.5   │                    ╔╝                                    │  │
│   │   1.0   │                   ╔╝ ╚╗                                   │  │
│   │   0.5   │ ═════════════════╝    ╚═════════════════                │  │
│   │         │                                                           │  │
│   │         └─────────────────────────────────────────────────────────  │  │
│   │              1.0   1.5   2.0   2.5   3.0   3.5   4.0  (atr_mult)  │  │
│   │                                                                     │  │
│   │   Only one good value = fragile, likely overfitting                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Options

### Univariate Optimization

| Setting | Description | Default |
|---------|-------------|---------|
| `control_values` | Baseline values when testing other params | Strategy defaults |
| `min_value` | Minimum parameter value to test | Parameter-specific |
| `max_value` | Maximum parameter value to test | Parameter-specific |
| `step` | Increment between test values | Parameter-specific |
| `metric` | Metric to optimize | Sharpe Ratio |

### Walk-Forward Optimization

| Setting | Description | Default |
|---------|-------------|---------|
| `train_period_months` | Training window length | 24 |
| `test_period_months` | Test window length | 6 |
| `n_iterations` | Bayesian search iterations | 100 |
| `metric` | Metric to optimize | Sharpe Ratio |

---

## Output Reports

### Univariate Report

Excel workbook containing:
- Summary sheet with control values and best values
- Per-parameter sheets with data tables and charts
- Line charts showing metric vs parameter value

### Walk-Forward Report

Excel workbook containing:
- Summary of all periods
- Per-period optimal parameters
- In-sample vs out-of-sample comparison
- Parameter stability analysis
- Final recommendations

---

## Related Documentation

- [Applications: Univariate Optimization](../applications/OPTIMIZATION_GUI.md) — Using the univariate GUI
- [Applications: Walk-Forward Optimization](../applications/OPTIMIZATION_GUI.md) — Using the walk-forward GUI
- [Configuration: Optimization](../reference/CONFIGURATION.md#optimization) — All optimization settings
- [Metrics Glossary](../reference/METRICS_GLOSSARY.md) — Metric definitions
