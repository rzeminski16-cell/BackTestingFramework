# Testing Your Strategy

This document covers how to validate your strategy and avoid common pitfalls.

---

## Validation Checklist

Before using a strategy for serious analysis:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY VALIDATION CHECKLIST                            │
│                                                                             │
│   □ Basic Functionality                                                    │
│     □ Strategy loads without errors                                        │
│     □ All required columns are available                                   │
│     □ Signals are generated (not all HOLD)                                 │
│     □ Trades execute correctly                                             │
│                                                                             │
│   □ Correctness                                                            │
│     □ No lookahead bias                                                    │
│     □ Signals match intended logic                                         │
│     □ Stop losses trigger correctly                                        │
│     □ Position sizing is reasonable                                        │
│                                                                             │
│   □ Edge Validation                                                        │
│     □ Beats random baseline                                                │
│     □ E-ratio > 1.0                                                        │
│     □ Positive expectancy                                                  │
│                                                                             │
│   □ Robustness                                                             │
│     □ Works on multiple securities                                         │
│     □ Works across different time periods                                  │
│     □ Parameters are stable (not overfit)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TESTING WORKFLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    1. UNIT TEST
       │
       │  Test strategy logic in isolation
       │  Verify signals for known inputs
       │
       ▼
    2. SINGLE SECURITY TEST
       │
       │  Run backtest on one ticker
       │  Check metrics are reasonable
       │  Verify trade log makes sense
       │
       ▼
    3. RANDOM BASELINE COMPARISON
       │
       │  Compare to RandomControlStrategy
       │  Your strategy should outperform
       │
       ▼
    4. EDGE ANALYSIS
       │
       │  Run E-ratio analysis
       │  Confirm E-ratio > 1.0 at relevant horizons
       │
       ▼
    5. MULTI-SECURITY TEST
       │
       │  Test on 10-20 securities
       │  Check if edge persists across symbols
       │
       ▼
    6. OUT-OF-SAMPLE TEST
       │
       │  Reserve recent data for final validation
       │  Don't optimize on this data
       │
       ▼
    7. WALK-FORWARD VALIDATION
       │
       │  Test parameter stability over time
       │  Ensure not overfit to specific period
       │
       ▼
    8. PORTFOLIO TEST
       │
       │  Run with realistic capital constraints
       │  Verify performance under real conditions

```

---

## Unit Testing Signals

Test your strategy logic with known inputs:

```python
import unittest
import pandas as pd
from your_strategy import YourStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import SignalType


class TestYourStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = YourStrategy(param1=2.0, param2=14)

    def test_buy_signal_on_crossover(self):
        """Test that BUY signal generated on SMA crossover."""
        # Create test data with crossover
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'close': [100, 101, 102, 103, 104],
            'sma_fast': [99, 100, 101, 102, 103],
            'sma_slow': [100, 100, 100, 100, 100],
            'atr_14': [2, 2, 2, 2, 2],
            # Previous: fast < slow, Current: fast > slow = crossover
        })

        # Prepare data
        data = self.strategy.prepare_data(data)

        # Create context for last bar
        context = self._create_context(data, bar_index=4, position=None)

        # Generate signal
        signal = self.strategy.generate_signal(context)

        # Assert
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertIsNotNone(signal.stop_loss)

    def test_hold_when_no_crossover(self):
        """Test HOLD when no crossover occurs."""
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'close': [100, 100, 100, 100, 100],
            'sma_fast': [99, 99, 99, 99, 99],  # Always below
            'sma_slow': [100, 100, 100, 100, 100],
            'atr_14': [2, 2, 2, 2, 2],
        })

        data = self.strategy.prepare_data(data)
        context = self._create_context(data, bar_index=4, position=None)
        signal = self.strategy.generate_signal(context)

        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def _create_context(self, data, bar_index, position):
        """Helper to create StrategyContext for testing."""
        # Implementation depends on your context class
        pass


if __name__ == '__main__':
    unittest.main()
```

---

## Random Baseline Comparison

Your strategy should beat random entries:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  RANDOM BASELINE COMPARISON                                 │
│                                                                             │
│   Test Setup:                                                              │
│   1. Run YourStrategy on AAPL 2020-2023                                    │
│   2. Run RandomControlStrategy on same data/settings                       │
│   3. Compare metrics                                                       │
│                                                                             │
│   Expected Results:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   Metric           YourStrategy    RandomControl    Difference      │  │
│   │   ──────────────────────────────────────────────────────────────── │  │
│   │   Total Return        +24.5%          +2.1%          +22.4%        │  │
│   │   Sharpe Ratio         1.42           0.15           +1.27         │  │
│   │   Win Rate             55%            50%            +5%           │  │
│   │   E-ratio              1.35           1.02           +0.33         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   If your strategy doesn't significantly beat random, it may not have      │
│   a real edge.                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## E-Ratio Validation

Use Edge Analysis to validate entry quality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     E-RATIO INTERPRETATION                                  │
│                                                                             │
│   E-ratio measures how price moves after entry:                            │
│   E-ratio = Average Favorable Movement / Average Adverse Movement          │
│                                                                             │
│   Results:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │  E-ratio > 1.0  →  Positive edge (price tends to move in your favor)│  │
│   │  E-ratio = 1.0  →  No edge (random)                                 │  │
│   │  E-ratio < 1.0  →  Negative edge (you're entering at bad times)     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   What to check:                                                           │
│   • Peak E-ratio value (higher is better)                                  │
│   • Horizon of peak (suggests optimal holding period)                      │
│   • Consistency across securities                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Pitfalls

### 1. Lookahead Bias

Using future information in trading decisions:

```python
# WRONG: Using future data
def generate_signal(self, context):
    # This uses tomorrow's close (lookahead!)
    tomorrow = context.historical_data.iloc[-1]
    if tomorrow['close'] > context.current_bar['close']:
        return Signal(SignalType.BUY, ...)

# CORRECT: Only use current and past data
def generate_signal(self, context):
    current = context.current_bar
    previous = context.get_bar(-1)
    # Only historical data
```

### 2. Overfitting

Strategy works great historically but fails forward:

```
Signs of Overfitting:
├── Parameters are very specific (e.g., 13.7 days, not 14)
├── Dramatically different optimal parameters per security
├── Out-of-sample performance much worse than in-sample
├── Walk-forward shows unstable parameters
└── Strategy has many parameters relative to trade count
```

Solutions:
- Use walk-forward optimization
- Prefer simple strategies with few parameters
- Reserve data for out-of-sample testing
- Require parameter stability across periods

### 3. Survivorship Bias

Only testing on stocks that still exist:

```
Problem:
Testing on today's S&P 500 ignores companies that went bankrupt,
were delisted, or acquired. This inflates results.

Solutions:
├── Use historical constituent data when available
├── Include delisted securities in test universe
└── Be aware that results may be optimistic
```

### 4. Data Snooping

Testing many variations until something works:

```
Problem:
Testing 100 strategies on the same data, one will look good by chance.

Solutions:
├── Pre-register your hypothesis before testing
├── Apply multiple testing correction (Bonferroni, FDR)
├── Use fresh data for final validation
└── Be skeptical of marginal results
```

---

## Out-of-Sample Testing

Reserve data for final validation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OUT-OF-SAMPLE TESTING                                    │
│                                                                             │
│   Full Data: 2015-01-01 to 2023-12-31                                      │
│                                                                             │
│   IN-SAMPLE                         OUT-OF-SAMPLE                          │
│   (Development)                     (Final Validation)                     │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ░░░░░░░░░░░░                          │
│   2015-01-01         2021-12-31     2022-01-01    2023-12-31               │
│                                                                             │
│   Use in-sample to:                 Use out-of-sample to:                  │
│   • Develop strategy                • Final validation only                │
│   • Optimize parameters             • Test once, not repeatedly            │
│   • Run factor analysis             • Confirm edge persists                │
│   • Iterate and refine                                                     │
│                                                                             │
│   RULE: Never optimize on out-of-sample data!                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Robustness Checks

### Multi-Security Test

Test on diverse securities:

```
Test Universe:
├── Large caps: AAPL, MSFT, GOOGL, AMZN
├── Mid caps: CRM, PANW, DDOG
├── Different sectors: JPM, XOM, UNH
└── High/low volatility: TSLA vs PG

Check:
├── Win rate consistent across securities
├── Sharpe ratio positive for most
├── No single security dominates results
└── Similar E-ratio patterns
```

### Multi-Period Test

Test across different market conditions:

```
Periods to Test:
├── Bull market: 2017, 2019, 2021
├── Bear market: 2008, 2022
├── High volatility: March 2020
├── Low volatility: 2017-2018
└── Sideways: 2015

Check:
├── Strategy performs in at least some conditions
├── Understand when it fails
├── Drawdowns are manageable in adverse periods
```

---

## Performance Benchmarks

What metrics suggest a viable strategy:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Sharpe Ratio | > 0.5 | > 1.0 | > 1.5 |
| Profit Factor | > 1.2 | > 1.5 | > 2.0 |
| Win Rate | > 35% | > 45% | > 55% |
| Max Drawdown | < 30% | < 20% | < 15% |
| E-ratio (peak) | > 1.1 | > 1.3 | > 1.5 |

Note: These are guidelines, not absolute requirements. Context matters.

---

## Final Validation Steps

Before considering a strategy "production ready":

```
□ Passed all unit tests
□ Beats random baseline significantly
□ E-ratio > 1.0 at relevant horizons
□ Works on at least 10 different securities
□ Works across at least 3 different time periods
□ Out-of-sample performance within 30% of in-sample
□ Walk-forward shows stable parameters
□ Drawdowns acceptable for your risk tolerance
□ Commission and slippage accounted for
□ Understand when/why the strategy fails
```

---

## Related Documentation

- [Strategy Guide](STRATEGY_GUIDE.md) — Creating strategies
- [Edge Analysis Concepts](../concepts/EDGE_ANALYSIS.md) — E-ratio methodology
- [Optimization](../concepts/OPTIMIZATION.md) — Parameter optimization
- [Factor Analysis](../concepts/FACTOR_ANALYSIS.md) — Understanding performance drivers
