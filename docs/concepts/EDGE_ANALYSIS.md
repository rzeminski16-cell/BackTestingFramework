# Edge Analysis

Edge analysis validates whether your trading strategy has genuine predictive power through statistical measures of entry quality.

---

## What Is Edge?

A trading strategy has "edge" if its entries systematically lead to favorable price movement. Without edge, a strategy is no better than random trading—and commission costs will eventually erode capital.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EDGE CONCEPT                                       │
│                                                                             │
│   WITH EDGE:                           WITHOUT EDGE:                       │
│                                                                             │
│   Entry ──▶ Price tends to            Entry ──▶ Price moves               │
│             move favorably                       randomly                  │
│                                                                             │
│   ┌──────────────────────────┐         ┌──────────────────────────┐       │
│   │         ╱                │         │      ╱    ╲              │       │
│   │        ╱                 │         │     ╱      ╲    ╱        │       │
│   │   ────•                  │         │   ─•        ╲  ╱         │       │
│   │      Entry               │         │   Entry      ╲╱          │       │
│   └──────────────────────────┘         └──────────────────────────┘       │
│                                                                             │
│   Favorable moves > Adverse moves      Favorable ≈ Adverse                │
│   E-ratio > 1.0                        E-ratio ≈ 1.0                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## E-Ratio (Edge Ratio)

The E-ratio measures entry quality by comparing favorable vs adverse price excursions at each time horizon.

### Definition

```
E-ratio(n) = Average Normalized MFE(n) / Average Normalized MAE(n)

Where:
├── MFE(n) = Maximum Favorable Excursion at day n
│            (How much price moved in your favor by day n)
│
├── MAE(n) = Maximum Adverse Excursion at day n
│            (How much price moved against you by day n)
│
└── Normalized = Divided by ATR at entry
                 (Makes values comparable across securities)
```

### Interpretation

| E-ratio Value | Meaning | Implication |
|---------------|---------|-------------|
| **> 1.0** | Favorable moves exceed adverse | Strategy has positive edge |
| **= 1.0** | Favorable equals adverse | No edge (random entries) |
| **< 1.0** | Adverse moves exceed favorable | Negative edge (worse than random) |

### E-Ratio Over Time

The framework calculates E-ratio at each horizon (day 1, day 2, ... day N):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      E-RATIO ACROSS HORIZONS                                │
│                                                                             │
│   E-ratio                                                                   │
│     1.6 │                      ★                                           │
│     1.4 │                 ╭────╮                                            │
│     1.2 │            ╭───╯    ╰───╮                                        │
│     1.0 │───────────╯─────────────╰──────────────────────────              │
│     0.8 │                              ╰───╮                               │
│         │                                                                   │
│         └───────────────────────────────────────────────────────           │
│             1   3   5   7   9  11  13  15  17  19  21  (days)              │
│                                                                             │
│   ★ Peak E-ratio at day 8 = 1.42                                           │
│   → Entries have maximum edge at 8-day horizon                             │
│   → Consider targeting exits around this timeframe                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insights from E-Ratio Curve:**
- **Peak location**: Optimal holding period
- **Peak magnitude**: Strength of edge
- **Decay rate**: How quickly edge diminishes
- **Crossover point**: When edge becomes negative

---

## R-Multiple Analysis

R-Multiple measures trade outcomes relative to initial risk:

```
R-Multiple = (Exit Price - Entry Price) / (Entry Price - Stop Loss)

Example:
├── Entry: $100
├── Stop Loss: $95
├── Risk (1R): $5
│
├── Exit at $115: R = ($115 - $100) / $5 = +3.0R (Won 3x risk)
├── Exit at $95:  R = ($95 - $100) / $5 = -1.0R (Lost 1x risk)
└── Exit at $105: R = ($105 - $100) / $5 = +1.0R (Won 1x risk)
```

### R-Multiple Distribution

Analyzing the distribution reveals strategy characteristics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     R-MULTIPLE DISTRIBUTION                                 │
│                                                                             │
│   Losing Trades                        Winning Trades                      │
│   ┌─────────────────────────┐          ┌─────────────────────────┐        │
│   │                         │          │                         │        │
│   │  ████                   │          │       ████              │        │
│   │  ████ ███               │          │  ███  ████  ██          │        │
│   │  ████ ███  ██           │          │  ███  ████  ██  █       │        │
│   └──────────────────────────          └─────────────────────────┘        │
│      -3R  -2R  -1R                        +1R  +2R  +3R +4R +5R           │
│                                                                             │
│   Key Metrics:                                                             │
│   ├── Average Winner: +2.3R                                                │
│   ├── Average Loser: -1.1R                                                 │
│   ├── Win Rate: 48%                                                        │
│   └── Expectancy: (0.48 × 2.3) + (0.52 × -1.1) = +0.53R per trade         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Validation Framework

The edge analysis includes comprehensive validation to ensure reliable results:

### Data Validation

| Check | Purpose | Expected Result |
|-------|---------|-----------------|
| Price data integrity | No gaps or NaN values | All pass |
| ATR computation | Valid volatility normalization | ATR > 0 |
| Date alignment | Trade dates match price data | ≤ 1 day difference |
| Entry price consistency | Logged price matches data | < 5% mismatch |

### Statistical Validation

| Check | Purpose | Expected Result |
|-------|---------|-----------------|
| Sample size | Sufficient trades for statistics | ≥ 20 trades per horizon |
| Outlier impact | Results not driven by few trades | < 20% difference when winsorized |
| Survivorship | Consistent sample across horizons | ~100% at all horizons |
| Random baseline | Comparison to no-edge scenario | E-ratio > 1.0 for edge |

### Warning Signs

```
RED FLAGS in E-ratio Analysis:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ⚠️ Entry price mismatches > 5%                                            │
│     → Data quality issue, results may be unreliable                        │
│                                                                             │
│  ⚠️ Winsorized E-ratio differs > 20% from raw                              │
│     → Results driven by outliers, not consistent edge                      │
│                                                                             │
│  ⚠️ E-ratio < 1.0 at most horizons                                         │
│     → Strategy may have negative edge                                      │
│                                                                             │
│  ⚠️ Sample size < 20 trades                                                │
│     → Insufficient data for reliable statistics                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Using Edge Analysis

### Step 1: Generate Trade Data

Run backtests to generate trade logs:
- Single security or portfolio mode
- Sufficient trades (aim for 50+ for reliable statistics)
- Multiple securities if testing strategy robustness

### Step 2: Load into Edge Analysis GUI

Launch the Edge Analysis GUI:
```bash
python ctk_edge_analysis_gui.py
```

Configure:
- Path to price data (raw_data/daily/)
- Load one or more trade log files
- Set maximum horizon (default: 30 days)

### Step 3: Interpret Results

**E-Ratio Tab:**
- Look for E-ratio consistently > 1.0
- Identify peak horizon (optimal holding period)
- Compare to winsorized version (gold line)

**R-Multiple Tab:**
- Check win rate and average R
- Look for favorable expectancy
- Identify distribution shape (fat tails = high variance)

**Validation Tab:**
- Review all checks pass
- Investigate any failures or warnings
- Check for data quality issues

### Step 4: Apply Insights

```
E-ratio peaks at day 8
├── Implication: Entries have max edge at 8 days
├── Action: Consider time-based exit around day 8
└── Verify: Run backtest with 8-day max hold

R-Multiple: Average winner 2.3R, loser 1.1R
├── Implication: Good reward/risk ratio
├── Action: Ensure stops aren't too tight
└── Verify: Check stop loss hit rate
```

---

## Comparing Strategies

E-ratio enables objective strategy comparison:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STRATEGY COMPARISON                                     │
│                                                                             │
│                Strategy A      Strategy B      Random Baseline             │
│   Peak E-ratio:    1.42           1.18              1.02                   │
│   Peak horizon:    Day 8          Day 15            N/A                    │
│   Decay rate:      Gradual        Sharp             N/A                    │
│                                                                             │
│   E-ratio                                                                   │
│     1.5 │      A                                                           │
│     1.4 │     ╭─╮                                                          │
│     1.3 │    ╱   ╲                                                         │
│     1.2 │   ╱     ╲ ............B....                                      │
│     1.1 │  ╱       ╲........'''      ''....                                │
│     1.0 │─╱─────────────────────────────────── Random                      │
│         │                                                                   │
│         └───────────────────────────────────────────────────────           │
│             5        10        15        20        25   (days)             │
│                                                                             │
│   Conclusion:                                                               │
│   Strategy A has stronger, more concentrated edge                          │
│   Strategy B has weaker but longer-lasting edge                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Rule Testing

The Rule Tester extends edge analysis by testing specific entry and exit rules:

### Entry Rule Testing

Test how different entry conditions affect edge:

```
Test: "Only enter when RSI < 30"
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   All Entries:           Filtered Entries (RSI < 30):                      │
│   ├── Trades: 150        ├── Trades: 45                                    │
│   ├── E-ratio: 1.18      ├── E-ratio: 1.52  ✓ Better                      │
│   └── Win Rate: 48%      └── Win Rate: 58%  ✓ Better                      │
│                                                                             │
│   Conclusion: RSI filter improves entry quality                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Exit Rule Testing

Test how different exit conditions affect results:

```
Test: Exit rules comparison
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    Signal Exit    Time Exit (10d)   Trailing Stop          │
│   Avg Holding:       12 days         10 days           8 days              │
│   Avg R-Multiple:    +1.8R           +1.4R            +2.1R                │
│   Win Rate:          52%             48%              45%                  │
│                                                                             │
│   Best for: Your risk tolerance and trading style                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [Applications: Edge Analysis GUI](../applications/ANALYSIS_TOOLS.md) — Using the GUI
- [Strategies](STRATEGIES.md) — How strategies generate signals
- [Metrics Glossary](../reference/METRICS_GLOSSARY.md) — Metric definitions
- [Factor Analysis](FACTOR_ANALYSIS.md) — Deeper analysis of trade factors
