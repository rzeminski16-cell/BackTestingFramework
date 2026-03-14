---
tags:
  - implementation/flow
  - engine
---

# Backtest Execution Flow

The complete bar-by-bar execution sequence for a single-security backtest.

---

## End-to-End Flow

```mermaid
flowchart TD
    START([Start]) --> INIT[Engine initialises PositionManager, TradeExecutor]
    INIT --> LOAD["DataLoader reads CSV"]
    LOAD --> VALIDATE["Validate data: columns, date range"]
    VALIDATE --> PREP["strategy.prepare_data(data) — pre-calculate indicators"]
    PREP --> LOOP["For each bar i = 0..N"]

    LOOP --> CTX["Build StrategyContext for bar i"]
    CTX --> FX["Look up FX rate for this bar"]
    FX --> SL_CHECK{Has open position with stop loss?}

    SL_CHECK -->|Yes| SL_HIT{"close <= stop_loss?"}
    SL_HIT -->|Yes| SL_EXIT["Execute SELL at stop loss price\nRecord trade, free capital"]
    SL_HIT -->|No| TP_CHECK{"Has take profit?"}
    SL_CHECK -->|No| SIGNAL

    TP_CHECK -->|Yes| TP_HIT{"close >= take_profit?"}
    TP_HIT -->|Yes| TP_EXIT["Execute SELL at take profit\nRecord trade, free capital"]
    TP_HIT -->|No| TRAIL
    TP_CHECK -->|No| TRAIL

    TRAIL["should_adjust_stop() → move stop?"] --> PARTIAL["should_partial_exit() → partial exit?"]
    PARTIAL --> SIGNAL["strategy.generate_signal(context)"]

    SL_EXIT --> EQUITY
    TP_EXIT --> EQUITY

    SIGNAL --> SIG_TYPE{Signal type?}
    SIG_TYPE -->|BUY| SIZE["strategy.position_size(context, signal)"]
    SIZE --> EXEC_BUY["TradeExecutor: Open position\nDeduct commission + slippage"]
    SIG_TYPE -->|SELL| EXEC_SELL["TradeExecutor: Close position\nDeduct commission, record trade"]
    SIG_TYPE -->|PYRAMID| EXEC_PYR["TradeExecutor: Add to position\nMove stop to breakeven"]
    SIG_TYPE -->|HOLD| EQUITY

    EXEC_BUY --> EQUITY
    EXEC_SELL --> EQUITY
    EXEC_PYR --> EQUITY

    EQUITY["Record equity = capital + unrealised P/L"]
    EQUITY --> NEXT{More bars?}
    NEXT -->|Yes| LOOP
    NEXT -->|No| CLOSE["Close any remaining position at last bar"]
    CLOSE --> METRICS["Calculate 50+ performance metrics"]
    METRICS --> REPORT["Generate Excel report + CSV trade log"]
    REPORT --> DONE([Return BacktestResult])
```

---

## Bar Processing Detail

On each bar, the engine processes in this exact order:

### 1. Build Context
`StrategyContext` is created with:
- Current bar data, price, and index
- Current position (if any)
- Available capital and total equity
- FX rate from `CurrencyConverter`

### 2. Check Stop Loss
If there's an open position with a stop loss and the current close price has breached it, the position is **immediately closed** at the stop loss price. The strategy is not consulted.

### 3. Check Take Profit
Same as stop loss, but for the take profit level (if set).

### 4. Trailing Stop
`strategy.should_adjust_stop(context)` is called. If it returns a new price, the stop is moved — but only in the protective direction (up for LONG, down for SHORT).

### 5. Partial Exit
`strategy.should_partial_exit(context)` is called. If it returns a fraction, that portion of the position is closed.

### 6. Strategy Signal
`strategy.generate_signal(context)` is called. Internally, this:
- If no position: calls `generate_entry_signal()` → checks `fundamental_rules` → returns BUY or HOLD
- If in position: calls `generate_exit_signal()` → checks `should_pyramid()` → returns SELL, PYRAMID, or HOLD

### 7. Execute Signal
The `TradeExecutor` opens, closes, or modifies the position based on the signal.

### 8. Record Equity
Current equity (capital + unrealised P/L) is appended to the equity curve.

---

## Key Invariants

> [!warning] Order Matters
> Stop loss and take profit are checked **before** the strategy generates signals. This means a strategy cannot override a stop loss hit on the same bar.

- Only one position open at a time per security
- Maximum one pyramid per trade
- Trades execute at the close price (adjusted for slippage)
- Commission is deducted on every entry and exit

---

## Related

- [[Backtesting Engine]] — component details
- [[Position Management]] — position lifecycle
- [[Strategy Framework]] — signal generation logic
