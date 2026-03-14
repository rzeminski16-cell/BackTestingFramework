---
tags:
  - usability/extending
  - data
  - indicators
---

# Adding a New Indicator

How to add a pre-calculated technical indicator to the raw data files so strategies can use it.

---

## Why Pre-Calculated?

The framework requires indicators to exist as columns in the CSV data files. This is by design:

1. **No lookahead bias** — indicators are calculated once on the full dataset before backtesting, using the same logic every time
2. **Performance** — no repeated calculation during bar-by-bar iteration
3. **Reproducibility** — the same CSV always produces the same backtest

---

## Step 1: Calculate the Indicator

Use pandas and/or the `ta` library to compute the indicator across the full dataset:

```python
import pandas as pd
import ta

# Load existing data
df = pd.read_csv('raw_data/daily/AAPL.csv')

# Example: Add Bollinger Bands
bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['bb_upper_20'] = bb.bollinger_hband()
df['bb_lower_20'] = bb.bollinger_lband()
df['bb_mid_20'] = bb.bollinger_mavg()

# Save back
df.to_csv('raw_data/daily/AAPL.csv', index=False)
```

---

## Step 2: Naming Convention

Use descriptive, lowercase names that include the period:

```
{indicator}_{period}_{suffix}
```

Examples:
- `atr_14_atr` — ATR with 14-period
- `ema_50` — 50-period EMA
- `rsi_14` — 14-period RSI
- `bb_upper_20` — Upper Bollinger Band (20-period)

> [!warning] Period is Baked In
> The period is part of the column name. You **cannot** optimise indicator periods without regenerating the CSV with a different period. Only strategy-level parameters (multipliers, thresholds) are optimisable at runtime.

---

## Step 3: Add to All Relevant Securities

If a strategy requires the indicator, **every security** you plan to test must have it. Apply the same calculation script to all CSV files.

---

## Step 4: Reference in Strategy

Add the column name to your strategy's `required_columns()`:

```python
def required_columns(self) -> List[str]:
    return ['date', 'close', 'bb_upper_20', 'bb_lower_20', 'atr_14_atr']
```

The `DataLoader` will validate that these columns exist when data is loaded.

---

## Next Steps

- [[Adding a New Strategy]] — use your new indicator in a strategy
- [[Adding a New Security]] — ensure new securities include the indicator
