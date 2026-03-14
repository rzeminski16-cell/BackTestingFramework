---
tags:
  - usability/reference
  - signals
  - strategy
---

# Signal Types

Every bar, the strategy returns one of these signals to the engine.

---

## Signal Overview

| Signal | Action | When Generated |
|---|---|---|
| `BUY` | Open a new position | Entry conditions met, no current position |
| `SELL` | Close the entire position | Exit conditions met |
| `PARTIAL_EXIT` | Close a fraction of the position | Partial profit-taking conditions met |
| `ADJUST_STOP` | Move the stop loss to a new level | Trailing stop conditions met |
| `PYRAMID` | Add to an existing position | Pyramiding conditions met (max 1 per trade) |
| `HOLD` | Do nothing | No conditions met, or waiting |

---

## Signal Factory Methods

Signals are created using static factory methods on the `Signal` class:

```python
from Classes.Models.signal import Signal

# Entry
Signal.buy(size=1.0, stop_loss=95.0, reason="EMA crossover", direction=TradeDirection.LONG)

# Full exit
Signal.sell(reason="Stop loss hit")

# Partial exit (close 50% of position)
Signal.partial_exit(fraction=0.5, reason="Take partial profit at 10%")

# Move stop loss
Signal.adjust_stop(new_stop_loss=102.0, reason="Trail stop to breakeven")

# Add to position (max 1 per trade)
Signal.pyramid(size=0.5, reason="Momentum continuation")

# No action
Signal.hold()
```

---

## Signal Fields

| Field | Type | Description |
|---|---|---|
| `type` | `SignalType` | One of the types above |
| `size` | `float` | Position size fraction (BUY/PYRAMID) or exit fraction (PARTIAL_EXIT) |
| `stop_loss` | `float` | Stop loss price for BUY signals |
| `take_profit` | `float` | Take profit price (optional) |
| `new_stop_loss` | `float` | Updated stop loss for ADJUST_STOP |
| `direction` | `TradeDirection` | LONG or SHORT |
| `reason` | `str` | Human-readable explanation |
| `metadata` | `dict` | Additional strategy-specific data |

---

## Engine Processing Order

On each bar, the engine processes signals in this order:

1. **Check stop loss** — if price breaches the stop, generate SELL
2. **Check take profit** — if price hits take profit, generate SELL
3. **Call strategy** — `generate_signal()` which internally:
   - If no position → calls `generate_entry_signal()` then checks `fundamental_rules`
   - If in position → calls `generate_exit_signal()`, then `should_pyramid()`
4. **Check trailing stop** — calls `should_adjust_stop()`
5. **Check partial exit** — calls `should_partial_exit()`

For the full flow, see [[Backtest Execution Flow]].

---

## Pyramiding Rules

> [!warning] Max One Pyramid Per Trade
> The engine enforces a maximum of one pyramid per trade. When pyramiding occurs, the stop loss is automatically moved to **breakeven** (accounting for the increased position size and commission costs).

---

## Next Steps

- [[Adding a New Strategy]] — use these signals in a custom strategy
- [[Strategy Framework]] — how signals flow through the engine
