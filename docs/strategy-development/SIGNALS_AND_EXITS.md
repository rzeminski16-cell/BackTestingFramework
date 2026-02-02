# Signals and Exits

This document covers signal types, exit rules, and position management in detail.

---

## Signal Types

The framework supports five signal types for different trading actions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL TYPES                                        │
│                                                                             │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                  │
│   │     BUY      │   │    SELL      │   │    HOLD      │                  │
│   │              │   │              │   │              │                  │
│   │ Open new     │   │ Close entire │   │ No action    │                  │
│   │ position     │   │ position     │   │              │                  │
│   └──────────────┘   └──────────────┘   └──────────────┘                  │
│                                                                             │
│   ┌──────────────┐   ┌──────────────┐                                     │
│   │ ADJUST_STOP  │   │ PARTIAL_EXIT │                                     │
│   │              │   │              │                                     │
│   │ Move stop    │   │ Close part   │                                     │
│   │ loss level   │   │ of position  │                                     │
│   └──────────────┘   └──────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## BUY Signal

Opens a new long position:

```python
from Classes.Models.signal import Signal, SignalType

# Basic BUY
Signal(SignalType.BUY)

# BUY with entry price (recommended)
Signal(
    SignalType.BUY,
    price=current['close']
)

# BUY with stop loss (recommended)
Signal(
    SignalType.BUY,
    price=current['close'],
    stop_loss=current['close'] - current['atr'] * 2
)

# BUY with all details
Signal(
    SignalType.BUY,
    price=current['close'],
    stop_loss=current['close'] * 0.95,
    take_profit=current['close'] * 1.10,  # Optional
    quantity=100,                          # Optional - override position sizing
    metadata={'reason': 'breakout'}        # Optional - for logging
)
```

### When BUY Executes

The engine only executes BUY when:
- No current position exists
- Sufficient capital available
- (Portfolio mode) Position slot available

---

## SELL Signal

Closes the entire current position:

```python
# Basic SELL
Signal(SignalType.SELL)

# SELL with reason (for logging/analysis)
Signal(
    SignalType.SELL,
    reason='signal_exit'
)

# Common exit reasons
Signal(SignalType.SELL, reason='stop_loss')      # Stop hit
Signal(SignalType.SELL, reason='take_profit')    # Target hit
Signal(SignalType.SELL, reason='time_exit')      # Max holding time
Signal(SignalType.SELL, reason='signal_exit')    # Exit signal
Signal(SignalType.SELL, reason='trailing_stop')  # Trailing stop hit
```

### Exit Reasons

| Reason | Description | When to Use |
|--------|-------------|-------------|
| `signal_exit` | Strategy exit signal triggered | Normal exit condition |
| `stop_loss` | Initial stop loss hit | Protective exit |
| `trailing_stop` | Trailing stop triggered | Locking in profits |
| `take_profit` | Target price reached | Profit taking |
| `time_exit` | Maximum holding period | Time-based rules |
| `end_of_data` | Backtest ended with position | Automatic |

---

## HOLD Signal

Indicates no action—maintain current state:

```python
# Basic HOLD
Signal(SignalType.HOLD)

# Common usage patterns
def generate_signal(self, context) -> Signal:
    # Not enough history
    if len(context.historical_data) < 20:
        return Signal(SignalType.HOLD)

    # Already in position, no exit conditions
    if context.position and not self._should_exit(context):
        return Signal(SignalType.HOLD)

    # No entry conditions met
    if not self._should_enter(context):
        return Signal(SignalType.HOLD)
```

---

## ADJUST_STOP Signal

Moves the stop loss to a new level:

```python
# Move stop loss
Signal(
    SignalType.ADJUST_STOP,
    stop_loss=new_stop_price
)

# Example: Trailing stop implementation
def _check_trailing_stop(self, context) -> Signal:
    position = context.position
    current = context.current_bar

    # Calculate new stop based on current price
    new_stop = current['close'] - (current['atr'] * 2)

    # Only adjust if new stop is higher (for longs)
    if new_stop > position.stop_loss:
        return Signal(SignalType.ADJUST_STOP, stop_loss=new_stop)

    return Signal(SignalType.HOLD)
```

### Trailing Stop Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRAILING STOP EXAMPLE                                  │
│                                                                             │
│   Price                                                                     │
│   $115 ┤                        ╭─╮                                        │
│        │                       ╭╯ ╰╮                                       │
│   $110 ┤                    ╭──╯   ╰─ SELL (trailing stop hit)             │
│        │                 ╭──╯                                              │
│   $105 ┤    Stop ········╭╯·····▲ Stop raised to $105                     │
│        │         ╭───────╯                                                 │
│   $100 ┤  BUY ──►│ Stop ····▲ Stop raised to $100                         │
│        │         │                                                         │
│    $95 ┤    Initial Stop ▼                                                │
│        ┼────────────────────────────────────────────────                   │
│                                                                             │
│   Stop trails price up, never moves down                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PARTIAL_EXIT Signal

Closes part of the position:

```python
# Exit specific quantity
Signal(
    SignalType.PARTIAL_EXIT,
    quantity=50  # Shares to exit
)

# Exit percentage
def _partial_exit_half(self, context) -> Signal:
    current_qty = context.position.quantity
    exit_qty = current_qty // 2
    return Signal(SignalType.PARTIAL_EXIT, quantity=exit_qty)
```

### Scale-Out Example

```python
def generate_signal(self, context) -> Signal:
    if context.position is None:
        return self._check_entry(context)

    position = context.position
    current = context.current_bar
    profit_pct = (current['close'] - position.entry_price) / position.entry_price

    # Scale out at 5% profit
    if profit_pct >= 0.05 and not self._scaled_out_1:
        self._scaled_out_1 = True
        return Signal(
            SignalType.PARTIAL_EXIT,
            quantity=position.quantity // 3
        )

    # Scale out at 10% profit
    if profit_pct >= 0.10 and not self._scaled_out_2:
        self._scaled_out_2 = True
        return Signal(
            SignalType.PARTIAL_EXIT,
            quantity=position.quantity // 2
        )

    # Final exit at signal
    return self._check_exit(context)
```

---

## Exit Rule Patterns

### Fixed Stop Loss

```python
def _entry_with_fixed_stop(self, context) -> Signal:
    price = context.current_bar['close']
    stop = price * 0.95  # 5% below entry

    return Signal(SignalType.BUY, price=price, stop_loss=stop)
```

### ATR-Based Stop

```python
def _entry_with_atr_stop(self, context) -> Signal:
    price = context.current_bar['close']
    atr = context.current_bar['atr_14']
    stop = price - (atr * self.atr_multiplier)

    return Signal(SignalType.BUY, price=price, stop_loss=stop)
```

### Support-Based Stop

```python
def _entry_with_support_stop(self, context) -> Signal:
    price = context.current_bar['close']
    recent_low = context.get_bars(20)['low'].min()
    stop = recent_low * 0.99  # Just below recent support

    return Signal(SignalType.BUY, price=price, stop_loss=stop)
```

### Time-Based Exit

```python
def __init__(self, max_holding_days: int = 10):
    super().__init__()
    self.max_holding_days = max_holding_days
    self._days_in_trade = 0

def generate_signal(self, context) -> Signal:
    if context.position:
        self._days_in_trade += 1
        if self._days_in_trade >= self.max_holding_days:
            self._days_in_trade = 0
            return Signal(SignalType.SELL, reason='time_exit')
```

### Signal-Based Exit

```python
def _check_signal_exit(self, context) -> Signal:
    current = context.current_bar
    prev = context.get_previous_bar(1)

    # Exit when fast SMA crosses below slow SMA
    if prev and current['sma_fast'] < current['sma_slow']:
        if prev['sma_fast'] >= prev['sma_slow']:
            return Signal(SignalType.SELL, reason='signal_exit')

    return Signal(SignalType.HOLD)
```

### Profit Target Exit

```python
def _check_profit_target(self, context) -> Signal:
    position = context.position
    current_price = context.current_bar['close']

    profit_pct = (current_price - position.entry_price) / position.entry_price

    if profit_pct >= self.profit_target:
        return Signal(SignalType.SELL, reason='take_profit')

    return Signal(SignalType.HOLD)
```

---

## Combined Exit Logic

Real strategies often combine multiple exit rules:

```python
def _check_exit(self, context) -> Signal:
    """Check all exit conditions in priority order."""

    # Priority 1: Stop loss (handled by engine, but we can check)
    if self._is_stop_hit(context):
        return Signal(SignalType.SELL, reason='stop_loss')

    # Priority 2: Profit target
    signal = self._check_profit_target(context)
    if signal.signal_type == SignalType.SELL:
        return signal

    # Priority 3: Time-based exit
    signal = self._check_time_exit(context)
    if signal.signal_type == SignalType.SELL:
        return signal

    # Priority 4: Signal-based exit
    signal = self._check_signal_exit(context)
    if signal.signal_type == SignalType.SELL:
        return signal

    # Priority 5: Adjust trailing stop
    signal = self._check_trailing_stop(context)
    if signal.signal_type == SignalType.ADJUST_STOP:
        return signal

    return Signal(SignalType.HOLD)
```

---

## Stop Loss Execution

The engine automatically checks stop losses each bar:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STOP LOSS EXECUTION                                      │
│                                                                             │
│   Each bar, before strategy.generate_signal():                             │
│                                                                             │
│   1. Check if bar low ≤ position.stop_loss                                 │
│   2. If yes, close position at stop_loss price                             │
│   3. Strategy doesn't need to check this                                   │
│                                                                             │
│   Example:                                                                  │
│   ├── Entry: $100, Stop: $95                                               │
│   ├── Day 3: Low = $94 (below stop)                                        │
│   └── Position closed at $95 (stop price)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Best Practices

### Always Define Stop Loss

```python
# GOOD: Stop loss defined
Signal(SignalType.BUY, price=100, stop_loss=95)

# AVOID: No stop loss (defaults may be used or no protection)
Signal(SignalType.BUY, price=100)
```

### Use Meaningful Exit Reasons

```python
# GOOD: Clear reason for analysis
Signal(SignalType.SELL, reason='momentum_reversal')

# AVOID: Generic or no reason
Signal(SignalType.SELL)
```

### Handle Edge Cases

```python
def generate_signal(self, context) -> Signal:
    # Ensure we have data
    if context.get_previous_bar(1) is None:
        return Signal(SignalType.HOLD)

    # Ensure indicator values are valid
    current = context.current_bar
    if pd.isna(current.get('indicator')):
        return Signal(SignalType.HOLD)
```

---

## Related Documentation

- [Strategy Guide](STRATEGY_GUIDE.md) — Getting started
- [Strategy Structure](STRATEGY_STRUCTURE.md) — Class architecture
- [Testing Your Strategy](TESTING_YOUR_STRATEGY.md) — Validation
