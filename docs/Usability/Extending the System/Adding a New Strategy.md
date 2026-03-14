---
tags:
  - usability/extending
  - strategy
---

# Adding a New Strategy

Step-by-step guide to creating a new trading strategy using the `BaseStrategy` framework.

---

## Overview

Every strategy is a Python class that extends `BaseStrategy` and implements six required methods. The engine calls these methods bar-by-bar to generate trading decisions.

---

## Step 1: Create the File

Create a new file in `strategies/`:

```
strategies/my_ema_crossover_strategy.py
```

---

## Step 2: Implement Required Methods

```python
from typing import List, Optional
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class MyEmaCrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.

    Entry: Fast EMA crosses above slow EMA.
    Exit: Fast EMA crosses below slow EMA.
    Stop Loss: ATR-based.

    RAW DATA INDICATORS (NOT OPTIMIZABLE):
        - ema_20: 20-period EMA (from raw CSV)
        - ema_50: 50-period EMA (from raw CSV)
        - atr_14_atr: 14-period ATR (from raw CSV)

    OPTIMIZABLE PARAMETERS:
        - atr_multiplier: Stop loss distance in ATR units
        - risk_percent: Equity percentage risked per trade
    """

    def __init__(self, atr_multiplier: float = 2.0, risk_percent: float = 2.0):
        super().__init__(
            atr_multiplier=atr_multiplier,
            risk_percent=risk_percent
        )
        self.atr_multiplier = atr_multiplier
        self.risk_percent = risk_percent

    # --- REQUIRED 1: Trade Direction ---
    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    # --- REQUIRED 2: Required Columns ---
    def required_columns(self) -> List[str]:
        return ['date', 'close', 'ema_20', 'ema_50', 'atr_14_atr']

    # --- REQUIRED 3: Entry Signal ---
    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        ema_fast = context.get_indicator_value('ema_20')
        ema_slow = context.get_indicator_value('ema_50')

        if ema_fast is None or ema_slow is None:
            return None

        # Check crossover: fast > slow now AND fast <= slow on previous bar
        prev = context.previous_bar
        if prev is None:
            return None

        if ema_fast > ema_slow and prev.get('ema_20', 0) <= prev.get('ema_50', 0):
            return Signal.buy(
                size=1.0,
                stop_loss=self.calculate_initial_stop_loss(context),
                reason="EMA 20/50 bullish crossover",
                direction=self.trade_direction
            )
        return None

    # --- REQUIRED 4: Initial Stop Loss ---
    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        atr = context.get_indicator_value('atr_14_atr')
        if atr and atr > 0:
            return context.current_price - (atr * self.atr_multiplier)
        return context.current_price * 0.95  # 5% fallback

    # --- REQUIRED 5: Position Size ---
    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        equity = context.total_equity
        risk_amount = equity * (self.risk_percent / 100)
        stop_distance = context.current_price - signal.stop_loss

        if stop_distance <= 0:
            return (context.available_capital * 0.1) / context.current_price

        stop_distance_base = stop_distance * context.fx_rate
        return risk_amount / stop_distance_base

    # --- REQUIRED 6: Exit Signal ---
    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        ema_fast = context.get_indicator_value('ema_20')
        ema_slow = context.get_indicator_value('ema_50')

        if ema_fast is not None and ema_slow is not None:
            if ema_fast < ema_slow:
                return Signal.sell(reason="EMA 20/50 bearish crossover")
        return None
```

---

## Step 3: Register Parameters (Optional)

Add parameter definitions to `config/strategy_parameters.json` so the GUI and optimiser know about your strategy:

```json
"MyEmaCrossoverStrategy": {
  "display_name": "EMA Crossover Strategy",
  "description": "Buys on EMA 20/50 crossover, exits on crossunder.",
  "trade_direction": "LONG",
  "required_indicators": ["ema_20", "ema_50", "atr_14_atr"],
  "parameters": {
    "atr_multiplier": {
      "type": "float",
      "default": 2.0,
      "configurable": true,
      "optimization": { "min": 0.5, "max": 5.0, "step": 0.25 },
      "description": "ATR multiplier for stop loss",
      "category": "Stop Loss"
    },
    "risk_percent": {
      "type": "float",
      "default": 2.0,
      "configurable": true,
      "optimization": { "min": 0.5, "max": 10.0, "step": 0.5 },
      "description": "Equity percentage risked per trade",
      "category": "Position Sizing"
    }
  }
}
```

---

## Required vs Optional Methods

### Required (must implement)

| Method | Purpose |
|---|---|
| `trade_direction` | Returns `TradeDirection.LONG` or `SHORT` |
| `required_columns()` | Lists all CSV columns needed (must include `date`, `close`) |
| `generate_entry_signal()` | Returns `Signal.buy()` or `None` |
| `calculate_initial_stop_loss()` | Returns the stop loss price |
| `position_size()` | Returns number of shares to buy |
| `generate_exit_signal()` | Returns `Signal.sell()` or `None` |

### Optional (override if needed)

| Method | Purpose | Default |
|---|---|---|
| `fundamental_rules` | Fundamental entry filter | Always pass |
| `_prepare_data_impl()` | Pre-calculate custom indicators | No-op |
| `should_adjust_stop()` | Trailing stop logic | No adjustment |
| `should_partial_exit()` | Partial profit-taking | No partial exits |
| `should_pyramid()` | Add to existing position (max 1) | No pyramiding |

---

## Important Rules

> [!warning] Raw Data Requirement
> All indicators listed in `required_columns()` **must exist** in the CSV file. The framework does not calculate standard indicators at runtime. If you need `ema_20` in your strategy, it must be a column in `raw_data/daily/AAPL.csv`.

> [!warning] No Lookahead Bias
> In `_prepare_data_impl()`, only use **causal** operations:
> - Allowed: `.rolling()`, `.expanding()`, `.shift(n)` where n >= 0
> - Forbidden: `.shift(-n)` where n > 0, global `.mean()`, `.std()`
>
> The framework automatically checks for look-ahead bias patterns.

> [!warning] Optimisable vs Non-Optimisable
> - **Indicator periods** (e.g. `ema_20`) are baked into the CSV column names — you cannot optimise them without regenerating the raw data
> - **Strategy parameters** (e.g. `atr_multiplier`, `risk_percent`) are computed at runtime — these can be optimised

---

## Testing Your Strategy

```python
# Quick test
from strategies.my_ema_crossover_strategy import MyEmaCrossoverStrategy

strategy = MyEmaCrossoverStrategy()
print(strategy)  # Shows name, direction, and parameters
print(strategy.required_columns())  # Verify columns
```

Then run a [[Single Security Backtest]] to validate.

---

## Template

See [[New Strategy Template]] for a copy-paste starting point.

---

## Next Steps

- [[Signal Types]] — all available signal types
- [[Strategy Framework]] — implementation details of `BaseStrategy`
- [[Creating Strategy Presets]] — save parameter configurations
