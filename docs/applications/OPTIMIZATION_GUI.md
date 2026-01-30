# Optimization Tools

The framework provides two optimization applications: Walk-Forward Optimization for robust parameter discovery, and Univariate Optimization for understanding individual parameter impact.

---

## Walk-Forward Optimization

### Purpose

Find parameters that perform well across multiple time periods, reducing overfitting risk.

### Launching

```bash
python ctk_optimization_gui.py
```

### Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WALK-FORWARD OPTIMIZATION FLOW                           │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────┐
   │   Select    │ →   │  Configure  │ →   │    Run      │ →   │  Review  │
   │  Securities │     │  Parameters │     │Optimization │     │  Report  │
   └─────────────┘     └─────────────┘     └─────────────┘     └──────────┘
```

### Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   WALK-FORWARD SETTINGS                                     │
│                                                                             │
│   Time Periods:                                                            │
│   ─────────────                                                            │
│   Training Period:    [ 12 ] months     In-sample optimization            │
│   Test Period:        [  3 ] months     Out-of-sample validation          │
│   Step Size:          [  3 ] months     Period advancement                │
│                                                                             │
│   Search Settings:                                                         │
│   ───────────────                                                          │
│   Optimization Method: ● Bayesian   ○ Grid   ○ Random                      │
│   Iterations:          [ 100 ]                                             │
│   Target Metric:       [ Sharpe Ratio  ▼]                                  │
│                                                                             │
│   Parameter Ranges:                                                        │
│   ────────────────                                                         │
│   Parameter           Min      Max      Type                               │
│   atr_multiplier      [ 0.5]   [ 5.0]   continuous                        │
│   atr_period          [  5 ]   [  50]   integer                           │
│   mfi_period          [  5 ]   [  30]   integer                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Walk-Forward Visualization

```
Historical Data: 2018-01-01 to 2023-12-31
────────────────────────────────────────────────────────────────────────────────

Period 1: Train 2018-01 to 2018-12 │ Test 2019-01 to 2019-03
          ████████████████████████   ▒▒▒▒▒▒

Period 2: Train 2018-04 to 2019-03 │ Test 2019-04 to 2019-06
               ████████████████████████   ▒▒▒▒▒▒

Period 3: Train 2018-07 to 2019-06 │ Test 2019-07 to 2019-09
                    ████████████████████████   ▒▒▒▒▒▒

...continues rolling forward...

████ = Training (find optimal parameters)
▒▒▒▒ = Testing (validate on unseen data)
```

### Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  WALK-FORWARD RESULTS                                       │
│                                                                             │
│   AGGREGATED OUT-OF-SAMPLE PERFORMANCE                                     │
│   ─────────────────────────────────────                                    │
│   Combined Sharpe:      1.18                                               │
│   Combined Return:      +45.2%                                              │
│   Average Period Return: +3.8%                                              │
│   Winning Periods:      9 of 12 (75%)                                      │
│                                                                             │
│   PARAMETER STABILITY                                                      │
│   ───────────────────                                                      │
│   Parameter          Mean     Std Dev   Stability                         │
│   atr_multiplier     1.85      0.32      High                             │
│   atr_period         16.2      2.8       High                             │
│   mfi_period         12.8      4.1       Medium                           │
│                                                                             │
│   RECOMMENDED PARAMETERS                                                   │
│   ──────────────────────                                                   │
│   Based on consistency across periods:                                     │
│   • atr_multiplier: 1.8                                                    │
│   • atr_period: 16                                                         │
│   • mfi_period: 13                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Univariate Optimization

### Purpose

Understand how each parameter individually affects performance, visualized with line charts.

### Launching

```bash
python ctk_univariate_optimization_gui.py
```

### How It Works

Tests one parameter at a time while keeping others at "control" values:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIVARIATE APPROACH                                      │
│                                                                             │
│   Control Values (baseline):                                               │
│   ├── atr_multiplier: 1.5                                                  │
│   ├── atr_period: 14                                                       │
│   └── mfi_period: 14                                                       │
│                                                                             │
│   Test 1: Vary atr_multiplier (0.5 to 5.0, step 0.5)                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   atr_mult=0.5, atr_period=14, mfi_period=14  → Sharpe=0.82        │  │
│   │   atr_mult=1.0, atr_period=14, mfi_period=14  → Sharpe=1.15        │  │
│   │   atr_mult=1.5, atr_period=14, mfi_period=14  → Sharpe=1.28        │  │
│   │   atr_mult=2.0, atr_period=14, mfi_period=14  → Sharpe=1.35  ★Max  │  │
│   │   atr_mult=2.5, atr_period=14, mfi_period=14  → Sharpe=1.22        │  │
│   │   ...                                                               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Test 2: Vary atr_period (5 to 50, step 5)                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   atr_mult=1.5, atr_period=5,  mfi_period=14  → Sharpe=0.95        │  │
│   │   atr_mult=1.5, atr_period=10, mfi_period=14  → Sharpe=1.18        │  │
│   │   atr_mult=1.5, atr_period=15, mfi_period=14  → Sharpe=1.32  ★Max  │  │
│   │   ...                                                               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Test 3: Vary mfi_period...                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   UNIVARIATE SETTINGS                                       │
│                                                                             │
│   Control Values (used when testing other parameters):                     │
│   ──────────────────────────────────────────────────                       │
│   Parameter           Control Value                                         │
│   atr_multiplier      [ 1.5  ]                                             │
│   atr_period          [  14  ]                                             │
│   mfi_period          [  14  ]                                             │
│                                                                             │
│   Test Ranges:                                                             │
│   ────────────                                                             │
│   Parameter           Min       Max       Interval                         │
│   atr_multiplier      [ 0.5]    [ 5.0]    [ 0.5]    (10 tests)            │
│   atr_period          [  5 ]    [  50]    [  5 ]    (10 tests)            │
│   mfi_period          [  5 ]    [  30]    [  5 ]    (6 tests)             │
│                                                                             │
│   Metrics to Calculate:                                                    │
│   ────────────────────                                                     │
│   ☑ Sharpe Ratio                                                           │
│   ☑ Total Return %                                                         │
│   ☑ Max Drawdown %                                                         │
│   ☐ Win Rate                                                               │
│   ☐ Profit Factor                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Line Chart Output

Each parameter generates a chart showing metric vs parameter value:

```
Sharpe Ratio vs atr_multiplier
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1.4 ┤                     ●                                               │
│      │                  ●     ●                                            │
│  1.2 ┤               ●           ●                                         │
│      │            ●                 ●                                      │
│  1.0 ┤         ●                       ●                                   │
│      │      ●                             ●                                │
│  0.8 ┤   ●                                                                 │
│      │                                                                     │
│  0.6 ┤                                                                     │
│      ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────────               │
│      0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5  5.0                     │
│                        atr_multiplier                                      │
│                                                                             │
│  Best value: 2.0 (Sharpe = 1.35)                                          │
│  Pattern: Clear peak → optimal value exists                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Interpreting Patterns

| Pattern | Meaning | Action |
|---------|---------|--------|
| **Clear peak** | Optimal value exists | Use the peak value |
| **Flat line** | Parameter has little impact | Use default, focus elsewhere |
| **Monotonic up/down** | Higher/lower always better | Check for natural limits |
| **Noisy/jagged** | Unstable or overfitting | Need more data or simpler approach |

### Excel Report

The output Excel file contains:

| Sheet | Contents |
|-------|----------|
| Summary | Control values, best values, settings |
| {Parameter}_Results | Data table + line chart for each parameter |

---

## Choosing Between Methods

| Scenario | Recommended Method |
|----------|-------------------|
| Quick parameter exploration | Univariate |
| Understanding parameter sensitivity | Univariate |
| Finding robust production parameters | Walk-Forward |
| Validating against overfitting | Walk-Forward |
| Initial strategy development | Univariate first, then Walk-Forward |

---

## Best Practices

### Walk-Forward

- Use at least 3-4 periods for statistical relevance
- Training period should be 3-4x longer than test period
- Check parameter stability across periods
- Be skeptical of parameters that vary wildly

### Univariate

- Start with reasonable control values (defaults or domain knowledge)
- Use fine enough intervals to see the shape of the curve
- Test each parameter independently first
- Combine insights carefully—interactions may exist

### Both Methods

- Always validate on out-of-sample data
- Don't optimize too many parameters
- Prefer robust, stable parameters over "optimal" ones
- Re-run periodically as market conditions change

---

## Output Locations

```
Walk-Forward:
logs/optimization_reports/walk_forward_{strategy}_{timestamp}/
├── optimization_results.json
├── period_details/
└── walk_forward_report.xlsx

Univariate:
logs/optimization_reports/univariate_{strategy}_{timestamp}.xlsx
```

---

## Related Documentation

- [Optimization Concepts](../concepts/OPTIMIZATION.md) — Theory and methodology
- [Backtest GUI](BACKTEST_GUI.md) — Testing optimized parameters
- [Configuration Reference](../reference/CONFIGURATION.md) — All settings
