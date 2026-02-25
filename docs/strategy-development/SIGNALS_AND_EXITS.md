# Signals and Exits

This document covers signal types, exit rules, and position management in detail.
All code examples use the actual framework API as defined in the source code.

---

## Signal Types

The framework supports six signal types for different trading actions.
Signals must be created using factory methods -- never call the `Signal()` constructor directly.

```
+-----------------------------------------------------------------------------+
|                            SIGNAL TYPES                                      |
|                                                                              |
|   +------------+   +------------+   +------------+   +------------+         |
|   |    BUY     |   |    SELL    |   |    HOLD    |   |  PYRAMID   |         |
|   |            |   |            |   |            |   |            |         |
|   | Open new   |   | Close full |   | No action  |   | Add to     |         |
|   | position   |   | position   |   |            |   | position   |         |
|   +------------+   +------------+   +------------+   +------------+         |
|                                                                              |
|   +---------------+   +---------------+                                      |
|   |  ADJUST_STOP  |   | PARTIAL_EXIT  |                                      |
|   |               |   |               |                                      |
|   | Move stop     |   | Close part    |                                      |
|   | loss level    |   | of position   |                                      |
|   +---------------+   +---------------+                                      |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Factory Methods Summary

| Factory Method | Signal Type | When to Use |
|---|---|---|
| `Signal.buy(...)` | BUY | Open a new position (long or short) |
| `Signal.sell(...)` | SELL | Close the entire position |
| `Signal.hold(...)` | HOLD | No action this bar |
| `Signal.partial_exit(...)` | PARTIAL_EXIT | Close a fraction of the position |
| `Signal.adjust_stop(...)` | ADJUST_STOP | Move the stop loss to a new level |
| `Signal.pyramid(...)` | PYRAMID | Add to an existing position (max 1 per trade) |

### Signal Fields

Every `Signal` object has the following fields:

| Field | Type | Description |
|---|---|---|
| `type` | `SignalType` | BUY, SELL, HOLD, PARTIAL_EXIT, ADJUST_STOP, or PYRAMID |
| `size` | `float` | Position size as fraction of capital (0.0-1.0) for BUY/PYRAMID |
| `stop_loss` | `Optional[float]` | Stop loss price for BUY signals |
| `take_profit` | `Optional[float]` | Take profit price for BUY signals |
| `new_stop_loss` | `Optional[float]` | New stop loss price for ADJUST_STOP signals |
| `direction` | `TradeDirection` | LONG or SHORT (defaults to LONG) |
| `reason` | `str` | Human-readable explanation |
| `fraction` | N/A | Not a separate field -- partial exit fraction is stored in `size` |

IMPORTANT: Signal has NO `price` field, NO `quantity` field. The signal field
for the type is `.type`, not `.signal_type`.

---

## BUY Signal

Opens a new position (long or short). Created with `Signal.buy()`.

```python
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection

# Basic BUY (long, with stop loss)
Signal.buy(
    size=1.0,
    stop_loss=context.current_price * 0.95,
    reason="Breakout above resistance"
)

# BUY with stop loss and take profit
Signal.buy(
    size=1.0,
    stop_loss=context.current_price - atr * 2,
    take_profit=context.current_price + atr * 4,
    reason="ATR breakout entry"
)

# SHORT entry
Signal.buy(
    size=1.0,
    stop_loss=context.current_price * 1.05,
    take_profit=context.current_price * 0.90,
    reason="Bearish reversal",
    direction=TradeDirection.SHORT
)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `size` | (required) | Fraction of capital to allocate (0.0-1.0) |
| `stop_loss` | `None` | Stop loss price. If omitted, `calculate_initial_stop_loss()` is called |
| `take_profit` | `None` | Take profit price (optional) |
| `reason` | `""` | Explanation for logging and analysis |
| `direction` | `TradeDirection.LONG` | Trade direction: LONG or SHORT |

### When BUY Executes

The engine only executes BUY when:
- No current position exists
- Sufficient capital is available
- Fundamental rules pass (checked automatically after entry signal)
- (Portfolio mode) A position slot is available

### Typical Usage in generate_entry_signal()

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    # Need previous bar for crossover detection
    prev = context.previous_bar
    if prev is None:
        return None

    current = context.current_bar

    # EMA crossover entry
    if (prev['ema_20'] <= prev['ema_50'] and
            current['ema_20'] > current['ema_50']):
        return Signal.buy(
            size=1.0,
            stop_loss=self.calculate_initial_stop_loss(context),
            reason="EMA 20/50 bullish crossover",
            direction=self.trade_direction
        )

    return None
```

---

## SELL Signal

Closes the entire current position. Created with `Signal.sell()`.

```python
# Basic SELL
Signal.sell()

# SELL with reason (recommended for analysis)
Signal.sell(reason="Signal exit: EMA crossover bearish")

# Common exit reasons
Signal.sell(reason="signal_exit")       # Exit signal triggered
Signal.sell(reason="take_profit")       # Profit target reached
Signal.sell(reason="time_exit")         # Max holding time exceeded
Signal.sell(reason="trailing_stop")     # Trailing stop triggered
Signal.sell(reason="momentum_reversal") # Custom reason
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `reason` | `""` | Explanation for logging and analysis |

### Exit Reasons

| Reason | Description | When to Use |
|---|---|---|
| `signal_exit` | Strategy exit signal triggered | Normal technical exit condition |
| `stop_loss` | Stop loss hit | Handled automatically by engine |
| `trailing_stop` | Trailing stop triggered | Trailing stop logic |
| `take_profit` | Target price reached | Profit taking |
| `time_exit` | Maximum holding period reached | Time-based exit rules |
| `end_of_data` | Backtest ended with open position | Automatic (engine-generated) |

### Typical Usage in generate_exit_signal()

```python
def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
    current = context.current_bar
    prev = context.previous_bar

    if prev is None:
        return None

    # Exit on bearish EMA crossover
    if (prev['ema_20'] >= prev['ema_50'] and
            current['ema_20'] < current['ema_50']):
        return Signal.sell(reason="EMA 20/50 bearish crossover")

    return None
```

---

## HOLD Signal

Indicates no action -- maintain the current state. Created with `Signal.hold()`.

```python
# Basic HOLD
Signal.hold()

# HOLD with reason (useful for debugging)
Signal.hold(reason="Waiting for confirmation bar")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `reason` | `""` | Explanation (useful for debugging) |

### When HOLD is Used

You typically do not need to return `Signal.hold()` explicitly. The strategy
methods (`generate_entry_signal`, `generate_exit_signal`) return `None` when
no action is needed. The engine's `generate_signal()` method returns
`Signal.hold()` automatically when no entry, exit, or pyramid condition is met.

However, `Signal.hold()` is returned explicitly in some internal situations:

```python
# Inside BaseStrategy.generate_signal() - when fundamentals fail
if not fundamental_result.passed:
    return Signal.hold(
        reason=f"Fundamentals failed: {fundamental_result.reason}"
    )
```

---

## PARTIAL_EXIT Signal

Closes a fraction of the current position. Created with `Signal.partial_exit()`.

IMPORTANT: Uses `fraction` (0.0-1.0), NOT a share quantity.

```python
# Exit half the position
Signal.partial_exit(fraction=0.5, reason="Taking partial profit at 10% gain")

# Exit one-third of the position
Signal.partial_exit(fraction=0.33, reason="Scale-out level 1")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `fraction` | (required) | Fraction of position to exit (0.0-1.0, exclusive-inclusive) |
| `reason` | `""` | Explanation for logging |

The `fraction` parameter must satisfy `0.0 < fraction <= 1.0`. Values outside
this range raise a `ValueError`.

### Typical Usage in should_partial_exit()

The engine calls `should_partial_exit(context)` on every bar when in a position.
This method returns a `float` (the fraction), NOT a Signal object.

```python
def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
    """Return fraction of position to exit, or None."""

    # Take 50% off the table at 10% profit
    pl_pct = context.get_position_pl_pct()
    if pl_pct > 10.0:
        return 0.5

    return None
```

### Scale-Out Example

```python
def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
    """Multi-level scale-out based on profit percentage."""
    pl_pct = context.get_position_pl_pct()
    position = context.position

    # Track scale-out state using remaining quantity
    original_qty = position.initial_quantity
    remaining_ratio = position.current_quantity / original_qty

    # Level 1: Exit 1/3 at 5% profit (only if still holding full position)
    if pl_pct >= 5.0 and remaining_ratio > 0.9:
        return 0.33

    # Level 2: Exit another 1/2 at 10% profit (of remaining position)
    if pl_pct >= 10.0 and remaining_ratio > 0.5:
        return 0.5

    return None
```

---

## ADJUST_STOP Signal

Moves the stop loss to a new price level. Created with `Signal.adjust_stop()`.

```python
# Move stop loss to a new price
Signal.adjust_stop(new_stop_loss=105.0, reason="Trailing stop adjustment")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `new_stop_loss` | (required) | New stop loss price (absolute level) |
| `reason` | `""` | Explanation for logging |

### Direction-Aware Enforcement

The engine enforces that stops can only move in the protective direction:

- LONG positions: Stop can only move UP (new_stop must be > current stop)
- SHORT positions: Stop can only move DOWN (new_stop must be < current stop)

If you return a stop that moves in the wrong direction, the engine ignores it.

### Typical Usage in should_adjust_stop()

The engine calls `should_adjust_stop(context)` on every bar when in a position.
This method returns a `float` (the new stop price), NOT a Signal object.

```python
def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
    """ATR-based trailing stop."""
    atr = context.get_indicator_value('atr_14')
    if atr is None:
        return None

    new_stop = context.current_price - (atr * 2.0)

    # The engine enforces direction-aware logic, but we can check here too
    if context.position.stop_loss is not None and new_stop > context.position.stop_loss:
        return new_stop

    return None
```

### Trailing Stop Diagram

```
+-----------------------------------------------------------------------------+
|                      TRAILING STOP EXAMPLE (LONG)                            |
|                                                                              |
|   Price                                                                      |
|   $115 |                        .---.                                        |
|        |                       /     \                                       |
|   $110 |                    .-'       '-- SELL (close <= trailing stop)       |
|        |                 .-'                                                  |
|   $105 |    Stop ........'........^ Stop raised to $105                      |
|        |         .------'                                                    |
|   $100 |  BUY -->|  Stop ........^ Stop raised to $100                       |
|        |         |                                                           |
|    $95 |    Initial Stop v                                                   |
|        +----------------------------------------------------                 |
|                                                                              |
|   For LONG: stop trails price UP, never moves down.                          |
|   For SHORT: stop trails price DOWN, never moves up.                         |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## PYRAMID Signal

Adds to an existing position. Created with `Signal.pyramid()`. Maximum one
pyramid per trade -- the engine enforces this limit.

```python
# Add to position with 50% of original capital allocation
Signal.pyramid(size=0.5, reason="Price momentum continuation")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `size` | (required) | Additional position size as fraction of capital (0.0-1.0) |
| `reason` | `""` | Explanation for logging |

### What Happens When You Pyramid

When a pyramid is executed, the engine automatically:

1. Calculates the additional quantity based on `size` and available capital
2. Updates the average entry price (weighted average of original + pyramid)
3. Moves the stop loss to break-even (accounting for commissions)
4. Sets `position.has_pyramided = True` (preventing further pyramids)

```
+-----------------------------------------------------------------------------+
|                      PYRAMID EXECUTION FLOW                                  |
|                                                                              |
|   1. Entry: 100 shares @ $50.00, stop @ $47.00                              |
|      |                                                                       |
|   2. Price rises to $55.00 (+10%)                                            |
|      |                                                                       |
|   3. Pyramid: 50 shares @ $55.00                                             |
|      |                                                                       |
|   4. New avg entry: ($50*100 + $55*50) / 150 = $51.67                        |
|      |                                                                       |
|   5. Stop moves to break-even: ~$51.67 + commissions                         |
|      (was $47.00, now protects total investment)                              |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Typical Usage in should_pyramid()

The engine calls `should_pyramid(context)` on every bar when in a position and
not yet pyramided. This method returns a `Signal` object (or `None`).

```python
def should_pyramid(self, context: StrategyContext) -> Optional[Signal]:
    """Pyramid when position shows strong momentum."""
    # Only pyramid if profitable
    pl_pct = context.get_position_pl_pct()

    if pl_pct > 5.0:
        return Signal.pyramid(
            size=0.5,
            reason="Adding on strength at +5%"
        )

    return None
```

---

## Strategy Method Architecture

Strategies extend `BaseStrategy` and implement specific methods. The engine
calls these methods automatically -- you never call them yourself.

```
+-----------------------------------------------------------------------------+
|                    ENGINE CALL FLOW (EACH BAR)                                |
|                                                                              |
|   +-- No Position? -----------------------------------------+                |
|   |                                                          |                |
|   |   1. generate_entry_signal(context) -> Optional[Signal]  |                |
|   |   2. check_fundamentals(context)   [automatic]           |                |
|   |   3. calculate_initial_stop_loss(context) -> float       |                |
|   |   4. position_size(context, signal) -> float             |                |
|   |                                                          |                |
|   +----------------------------------------------------------+                |
|                                                                              |
|   +-- In Position? -----------------------------------------+                |
|   |                                                          |                |
|   |   1. Check stop loss      [automatic, uses close price]  |                |
|   |   2. Check take profit    [automatic, uses close price]  |                |
|   |   3. should_adjust_stop(context) -> Optional[float]      |                |
|   |   4. should_partial_exit(context) -> Optional[float]     |                |
|   |   5. generate_exit_signal(context) -> Optional[Signal]   |                |
|   |   6. should_pyramid(context) -> Optional[Signal]         |                |
|   |                                                          |                |
|   +----------------------------------------------------------+                |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Required Methods

| Method | Returns | Called When |
|---|---|---|
| `generate_entry_signal(context)` | `Optional[Signal]` | Every bar when no position |
| `generate_exit_signal(context)` | `Optional[Signal]` | Every bar when in position |
| `calculate_initial_stop_loss(context)` | `float` | When a BUY signal is generated |
| `position_size(context, signal)` | `float` | When a BUY signal is generated |

### Optional Methods

| Method | Returns | Called When |
|---|---|---|
| `should_adjust_stop(context)` | `Optional[float]` | Every bar when in position |
| `should_partial_exit(context)` | `Optional[float]` | Every bar when in position |
| `should_pyramid(context)` | `Optional[Signal]` | Every bar when in position and not yet pyramided |

IMPORTANT: Strategies do NOT override `generate_signal()`. That method is
implemented by `BaseStrategy` and orchestrates all the calls above.

---

## StrategyContext Reference

`StrategyContext` is an immutable dataclass (`frozen=True`) passed to every
strategy method. It provides safe, look-ahead-free access to market data.

### Fields

| Field | Type | Description |
|---|---|---|
| `data` | `DataFrame` or `HistoricalDataView` | Bars 0 to current_index only (no future data) |
| `current_index` | `int` | Index of the current bar |
| `current_price` | `float` | Close price of the current bar |
| `current_date` | `pd.Timestamp` | Date/time of the current bar |
| `position` | `Optional[Position]` | Current position, or `None` if flat |
| `available_capital` | `float` | Available capital in base currency |
| `total_equity` | `float` | Total equity (capital + position value) in base currency |
| `symbol` | `str` | Security symbol being traded |
| `fx_rate` | `float` | Exchange rate from security currency to base currency |

### Properties and Methods

| Member | Returns | Description |
|---|---|---|
| `has_position` | `bool` | Whether currently in an open position |
| `current_bar` | `pd.Series` | Data for the current bar |
| `previous_bar` | `Optional[pd.Series]` | Data for the previous bar (None if first bar) |
| `get_bar(offset)` | `Optional[pd.Series]` | Bar at offset from current (negative=past, 0=current) |
| `get_indicator_value(column, offset=0)` | `Optional[float]` | Indicator value at current bar or with offset |
| `get_position_pl()` | `float` | Current position P/L in currency (0 if no position) |
| `get_position_pl_pct()` | `float` | Current position P/L as percentage (0 if no position) |

### DOES NOT EXIST

The following are common mistakes. These methods and fields do not exist:

- `context.get_previous_bar(n)` -- use `context.get_bar(-n)` instead
- `context.get_bars(n)` -- use `context.data.iloc[...]` for slicing
- `context.get_indicator(name)` -- use `context.get_indicator_value(column)` instead
- `context.parameters` -- not a field on context
- `context.historical_data` -- use `context.data` instead
- `context.position.unrealized_pnl` -- use `context.get_position_pl()` instead
- `context.position.quantity` -- use `context.position.current_quantity` instead

### Usage Examples

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    # Check we have enough history
    prev = context.previous_bar
    if prev is None:
        return None

    # Access the current bar
    current = context.current_bar

    # Get indicator values
    ema_50 = context.get_indicator_value('ema_50')
    rsi = context.get_indicator_value('rsi_14')
    atr = context.get_indicator_value('atr_14')

    # Get a value from 3 bars ago
    old_close = context.get_indicator_value('close', offset=-3)

    # Use current_price (shorthand for close)
    price = context.current_price

    # Check available capital
    if context.available_capital < price * 10:
        return None

    # ...entry logic...
    return None
```

---

## Position Model Reference

When `context.position` is not `None`, it provides these fields:

| Field | Type | Description |
|---|---|---|
| `entry_price` | `float` | Average entry price (updated on pyramid) |
| `entry_date` | `datetime` | Position entry date |
| `initial_quantity` | `float` | Original position size before any changes |
| `current_quantity` | `float` | Current position size after partial exits |
| `direction` | `TradeDirection` | LONG or SHORT |
| `stop_loss` | `Optional[float]` | Current stop loss price |
| `take_profit` | `Optional[float]` | Take profit price |
| `has_pyramided` | `bool` | Whether position has been pyramided |
| `total_commission_paid` | `float` | Total commissions paid on this position |

### Position Methods

| Method | Returns | Description |
|---|---|---|
| `calculate_pl(price)` | `float` | Unrealized P/L at given price |
| `calculate_pl_pct(price)` | `float` | Unrealized P/L as percentage |
| `duration_days(date)` | `int` | Days since position was opened |

IMPORTANT: The position field for quantity is `current_quantity`, NOT `quantity`.
Use `context.position.current_quantity` to check the current position size.

---

## Stop Loss Execution

The engine automatically checks stop losses each bar BEFORE calling any
strategy methods. Stop losses are checked against the CLOSE PRICE, not the
bar low or bar high.

```
+-----------------------------------------------------------------------------+
|                    STOP LOSS EXECUTION                                        |
|                                                                              |
|   Each bar, BEFORE strategy methods are called:                              |
|                                                                              |
|   1. Engine gets current_price = bar['close']                                |
|   2. Calls position_manager.check_stop_loss(current_price)                   |
|   3. For LONG:  stop hit if close <= stop_loss                               |
|   4. For SHORT: stop hit if close >= stop_loss                               |
|   5. If hit, position closed at close price                                  |
|   6. Strategy methods are NOT called on this bar                             |
|                                                                              |
|   Example (LONG):                                                            |
|   +-- Entry: $100, Stop: $95                                                 |
|   +-- Day 3: Close = $94 (<= stop)                                           |
|   +-- Position closed at $94 (close price, not stop price)                   |
|                                                                              |
|   Example (SHORT):                                                           |
|   +-- Entry: $100, Stop: $105                                                |
|   +-- Day 3: Close = $106 (>= stop)                                          |
|   +-- Position closed at $106 (close price, not stop price)                  |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Take Profit Execution

Take profit is also checked using the close price, after the stop loss check:

- LONG: take profit hit if close >= take_profit
- SHORT: take profit hit if close <= take_profit

---

## Direction-Aware Behavior

The framework is fully direction-aware. Long and short positions have
mirror-image behavior for stops, exits, and P/L calculations.

```
+-----------------------------------------------------------------------------+
|                DIRECTION-AWARE RULES                                         |
|                                                                              |
|   LONG POSITIONS                  |   SHORT POSITIONS                        |
|   --------------------------------|----------------------------------        |
|   Stop loss:  close <= stop       |   Stop loss:  close >= stop              |
|   Take profit: close >= TP        |   Take profit: close <= TP               |
|   Trailing: stop only moves UP    |   Trailing: stop only moves DOWN         |
|   P/L = (current - entry) * qty   |   P/L = (entry - current) * qty         |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Long Trailing Stop Example

```python
def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
    atr = context.get_indicator_value('atr_14')
    if atr is None:
        return None

    if context.position.direction.is_long:
        # LONG: trail stop upward
        new_stop = context.current_price - (atr * 2.0)
        if context.position.stop_loss is None or new_stop > context.position.stop_loss:
            return new_stop
    else:
        # SHORT: trail stop downward
        new_stop = context.current_price + (atr * 2.0)
        if context.position.stop_loss is None or new_stop < context.position.stop_loss:
            return new_stop

    return None
```

---

## Exit Rule Patterns

### Fixed Percentage Stop Loss

```python
def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
    if self.trade_direction.is_long:
        return context.current_price * 0.95  # 5% below entry
    else:
        return context.current_price * 1.05  # 5% above entry
```

### ATR-Based Stop Loss

```python
def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
    atr = context.get_indicator_value('atr_14')

    if self.trade_direction.is_long:
        return context.current_price - (atr * self.params.get('atr_multiplier', 2.0))
    else:
        return context.current_price + (atr * self.params.get('atr_multiplier', 2.0))
```

### Support/Resistance-Based Stop Loss

```python
def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
    # Look back 20 bars for recent low (long) or high (short)
    lows = []
    highs = []
    for offset in range(-20, 0):
        bar = context.get_bar(offset)
        if bar is not None:
            lows.append(bar['low'])
            highs.append(bar['high'])

    if self.trade_direction.is_long:
        recent_low = min(lows) if lows else context.current_price * 0.95
        return recent_low * 0.99  # Just below recent support
    else:
        recent_high = max(highs) if highs else context.current_price * 1.05
        return recent_high * 1.01  # Just above recent resistance
```

### Time-Based Exit

```python
def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
    max_days = self.params.get('max_holding_days', 10)
    days_held = context.position.duration_days(context.current_date)

    if days_held >= max_days:
        return Signal.sell(reason="time_exit")

    # ...other exit checks...
    return None
```

### Signal-Based Exit

```python
def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
    current = context.current_bar
    prev = context.previous_bar

    if prev is None:
        return None

    # Exit when fast SMA crosses below slow SMA (for longs)
    if (prev['sma_fast'] >= prev['sma_slow'] and
            current['sma_fast'] < current['sma_slow']):
        return Signal.sell(reason="signal_exit")

    return None
```

### Profit Target Exit

```python
def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
    profit_pct = context.get_position_pl_pct()
    target = self.params.get('profit_target_pct', 15.0)

    if profit_pct >= target:
        return Signal.sell(reason="take_profit")

    return None
```

---

## Combined Exit Logic

Real strategies combine multiple exit rules inside `generate_exit_signal()`.
Stop loss and take profit are handled automatically by the engine and do not
need to be checked in this method.

```python
def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
    """Check all exit conditions in priority order.

    Note: Stop loss and take profit are checked by the engine automatically
    BEFORE this method is called. No need to duplicate that logic here.
    """

    # Priority 1: Time-based exit
    days_held = context.position.duration_days(context.current_date)
    if days_held >= self.params.get('max_holding_days', 20):
        return Signal.sell(reason="time_exit")

    # Priority 2: Signal-based exit (e.g., indicator crossover)
    current = context.current_bar
    prev = context.previous_bar
    if prev is not None:
        if (prev['ema_20'] >= prev['ema_50'] and
                current['ema_20'] < current['ema_50']):
            return Signal.sell(reason="signal_exit")

    # No exit condition met
    return None
```

### Checking Signal Type

When checking the type of a returned signal, use `.type` (not `.signal_type`):

```python
signal = self._some_check(context)
if signal is not None and signal.type == SignalType.SELL:
    return signal
```

---

## Complete Strategy Example

Putting it all together -- a strategy that uses entries, exits, trailing stops,
partial exits, and pyramiding:

```python
from typing import Optional, List
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-following strategy with EMA crossover entry,
    ATR trailing stop, partial profit taking, and pyramiding.
    """

    def __init__(self, atr_multiplier: float = 2.5, **params):
        self.atr_multiplier = atr_multiplier
        super().__init__(**params)

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close', 'low', 'high', 'ema_20', 'ema_50', 'atr_14', 'rsi_14']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        prev = context.previous_bar
        if prev is None:
            return None

        current = context.current_bar

        # Enter on EMA crossover with RSI confirmation
        if (prev['ema_20'] <= prev['ema_50'] and
                current['ema_20'] > current['ema_50'] and
                current['rsi_14'] > 50):
            return Signal.buy(
                size=1.0,
                stop_loss=self.calculate_initial_stop_loss(context),
                reason="EMA crossover + RSI confirmation",
                direction=self.trade_direction
            )

        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        atr = context.get_indicator_value('atr_14')
        return context.current_price - (atr * self.atr_multiplier)

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        # Risk-based position sizing: risk 2% of equity per trade
        risk_pct = 0.02
        risk_amount = context.total_equity * risk_pct
        stop_distance = abs(context.current_price - signal.stop_loss)
        if stop_distance == 0:
            return 0
        quantity = risk_amount / (stop_distance * context.fx_rate)
        return quantity

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        current = context.current_bar
        prev = context.previous_bar

        if prev is None:
            return None

        # Exit on bearish EMA crossover
        if (prev['ema_20'] >= prev['ema_50'] and
                current['ema_20'] < current['ema_50']):
            return Signal.sell(reason="Bearish EMA crossover")

        # Exit if RSI shows extreme overbought
        if current['rsi_14'] > 85:
            return Signal.sell(reason="RSI extreme overbought")

        return None

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """Trail stop using 2x ATR below current price."""
        atr = context.get_indicator_value('atr_14')
        if atr is None:
            return None

        new_stop = context.current_price - (atr * 2.0)

        # For LONG: only move stop up (engine enforces this too)
        if context.position.stop_loss is not None and new_stop > context.position.stop_loss:
            return new_stop

        return None

    def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
        """Take 50% off at +10% profit."""
        pl_pct = context.get_position_pl_pct()
        remaining_ratio = context.position.current_quantity / context.position.initial_quantity

        if pl_pct > 10.0 and remaining_ratio > 0.9:
            return 0.5

        return None

    def should_pyramid(self, context: StrategyContext) -> Optional[Signal]:
        """Add to position on continued momentum."""
        pl_pct = context.get_position_pl_pct()

        if pl_pct > 5.0:
            return Signal.pyramid(
                size=0.5,
                reason="Adding on +5% momentum"
            )

        return None
```

---

## Best Practices

### Always Use Factory Methods

```python
# CORRECT: Use factory methods
Signal.buy(size=1.0, stop_loss=95.0, reason="Breakout")
Signal.sell(reason="Trend reversal")
Signal.hold()
Signal.partial_exit(fraction=0.5, reason="Scale out")
Signal.adjust_stop(new_stop_loss=102.0, reason="Trail")
Signal.pyramid(size=0.5, reason="Add on strength")

# WRONG: Never call the constructor directly
# Signal(SignalType.BUY, ...)   <-- Do not do this
```

### Always Define Stop Loss

```python
# GOOD: Stop loss defined in entry signal
Signal.buy(
    size=1.0,
    stop_loss=self.calculate_initial_stop_loss(context),
    reason="Entry with protection"
)

# ACCEPTABLE: Omit stop_loss and let engine call calculate_initial_stop_loss()
# The engine will fill it in automatically if missing
Signal.buy(size=1.0, reason="Engine will set stop")
```

### Use Meaningful Exit Reasons

```python
# GOOD: Clear reason for post-trade analysis
Signal.sell(reason="Bearish divergence on RSI")

# AVOID: No reason makes analysis harder
Signal.sell()
```

### Handle Edge Cases

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    # Ensure previous bar exists
    if context.previous_bar is None:
        return None

    # Ensure indicator values are valid
    ema = context.get_indicator_value('ema_50')
    if ema is None:
        return None

    # Check for NaN in bar data
    import math
    current = context.current_bar
    if math.isnan(current.get('atr_14', float('nan'))):
        return None

    # ...entry logic...
    return None
```

### Use context.current_price Instead of Accessing Close Directly

```python
# GOOD: Use the convenience property
price = context.current_price

# ALSO FINE: Access from current_bar
price = context.current_bar['close']

# Both are equivalent -- current_price is just a shorthand
```

---

## Related Documentation

- [Strategy Guide](STRATEGY_GUIDE.md) -- Getting started
- [Strategy Structure](STRATEGY_STRUCTURE.md) -- Class architecture
- [Testing Your Strategy](TESTING_YOUR_STRATEGY.md) -- Validation
