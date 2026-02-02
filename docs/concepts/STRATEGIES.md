# Strategies

This document explains how trading strategies work within the framework.

---

## What Is a Strategy?

A strategy is a set of rules that determine:
1. **When to enter** a trade (buy conditions)
2. **When to exit** a trade (sell conditions, stop loss, time limits)
3. **How much to trade** (position sizing)

The framework evaluates these rules bar-by-bar against historical data and executes trades accordingly.

---

## Strategy Flow

Every strategy follows this pattern during backtesting:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STRATEGY EXECUTION FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

    INITIALIZATION
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  1. Strategy receives parameters (e.g., period=14, atr_mult=2.0)       │
    │  2. Strategy reports required columns (e.g., ['close', 'atr_14'])      │
    │  3. Data is loaded and validated                                       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    DATA PREPARATION (Optional)
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Strategy can pre-calculate indicators or derived values               │
    │  Example: Calculate AlphaTrend line from ATR and MFI                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    BAR-BY-BAR LOOP
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  For each bar (day) in the data:                                       │
    │                                                                         │
    │      ┌─────────────────────────────────────────────────────────────┐   │
    │      │                    StrategyContext                          │   │
    │      │                                                             │   │
    │      │  Contains:                                                  │   │
    │      │  • current_bar: Today's OHLCV and indicators               │   │
    │      │  • position: Current open position (if any)                │   │
    │      │  • bar_index: Position in data series                      │   │
    │      │  • data: Full historical data up to current bar            │   │
    │      └─────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │      ┌─────────────────────────────────────────────────────────────┐   │
    │      │            strategy.generate_signal(context)                │   │
    │      │                                                             │   │
    │      │  Strategy evaluates:                                        │   │
    │      │  • Entry conditions (if no position)                       │   │
    │      │  • Exit conditions (if has position)                       │   │
    │      │  • Stop loss adjustments                                   │   │
    │      │                                                             │   │
    │      │  Returns: Signal (BUY, SELL, HOLD, PARTIAL_EXIT, ADJUST)   │   │
    │      └─────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

---

## Signal Types

Strategies communicate their decisions through signals:

| Signal Type | Meaning | When Used |
|-------------|---------|-----------|
| **BUY** | Open a new long position | Entry conditions met, no current position |
| **SELL** | Close entire position | Exit conditions met, position exists |
| **HOLD** | No action | No conditions triggered |
| **PARTIAL_EXIT** | Close part of position | Scaling out of a winner |
| **ADJUST_STOP** | Move stop loss | Trailing stop or risk reduction |

### Signal Properties

A signal can include additional information:

```
Signal(
    signal_type = SignalType.BUY,
    price = 150.00,              # Execution price (usually close)
    stop_loss = 145.00,          # Initial stop loss level
    quantity = 100,              # Optional: specific share count
    reason = "Trend breakout"    # Optional: for logging
)
```

---

## Included Strategies

The framework includes production-ready strategies you can use immediately.

### AlphaTrend Strategy

Located at: `strategies/alphatrend_strategy.py`

A trend-following strategy based on the AlphaTrend indicator:

**Core Concept:**
- Uses ATR (Average True Range) to adapt to volatility
- Uses MFI (Money Flow Index) to confirm trend direction
- Enters on trend breakouts with volume confirmation
- Exits via trailing stop, signal, or time limit

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 14 | ATR calculation period |
| `atr_multiplier` | 2.0 | ATR multiplier for trend line |
| `mfi_period` | 14 | MFI calculation period |
| `mfi_threshold` | 50 | MFI level for trend confirmation |
| `volume_threshold` | 1.2 | Volume ratio for confirmation |
| `max_hold_days` | 10 | Maximum days to hold |
| `risk_per_trade` | 0.02 | Risk percentage (2%) |

**Required Data Columns:**
- date, open, high, low, close, volume
- atr_14 (ATR indicator)
- ema_50 (50-day EMA)
- mfi_14 (MFI indicator)

**Exit Conditions:**
1. Price crosses below AlphaTrend line
2. Stop loss triggered
3. Maximum holding period reached
4. Momentum reversal detected

### Random Control Strategy

Located at: `strategies/random_base_strategy.py`

A baseline strategy that enters and exits randomly:

**Purpose:**
- Provides a benchmark for comparison
- E-ratio should be approximately 1.0 (no edge)
- If your strategy's E-ratio isn't significantly better, it may lack real edge

---

## Strategy Parameters

### Parameter Definition

Strategies define their parameters with metadata:

```
Parameter Definition Structure:
┌─────────────────────────────────────────────────────────────────────────────┐
│  'atr_multiplier': {                                                        │
│      'type': float,          # Data type                                   │
│      'default': 2.0,         # Default value                               │
│      'min': 0.5,             # Minimum for optimization                    │
│      'max': 5.0,             # Maximum for optimization                    │
│      'step': 0.1,            # Step size for optimization                  │
│      'description': 'ATR multiplier for trend line calculation'           │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

This metadata enables:
- GUI parameter input with validation
- Automatic optimization range detection
- Documentation generation

### Parameter Registry

All strategy parameters are centrally registered in `config/strategy_parameters.json`:

```
config/strategy_parameters.json
┌─────────────────────────────────────────────────────────────────────────────┐
│  {                                                                          │
│    "AlphaTrendStrategy": {                                                 │
│      "atr_period": {"type": "int", "default": 14, ...},                   │
│      "atr_multiplier": {"type": "float", "default": 2.0, ...},            │
│      ...                                                                    │
│    },                                                                       │
│    "YourCustomStrategy": {                                                 │
│      ...                                                                    │
│    }                                                                        │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Entry Logic

A typical entry decision evaluates multiple conditions:

```
                         ENTRY DECISION TREE
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ Already have position?  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                   Yes                      No
                    │                       │
                    ▼                       ▼
             Return HOLD          ┌─────────────────────┐
                                  │ Trend filter okay?  │
                                  │ (e.g., above EMA)   │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                                  No                   Yes
                                  │                     │
                                  ▼                     ▼
                           Return HOLD        ┌─────────────────────┐
                                              │ Entry signal valid? │
                                              │ (e.g., breakout)    │
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────┴──────────┐
                                              │                     │
                                              No                   Yes
                                              │                     │
                                              ▼                     ▼
                                       Return HOLD        ┌─────────────────────┐
                                                          │ Volume confirms?    │
                                                          └──────────┬──────────┘
                                                                     │
                                                          ┌──────────┴──────────┐
                                                          │                     │
                                                          No                   Yes
                                                          │                     │
                                                          ▼                     ▼
                                                   Return HOLD         Return BUY signal
                                                                       with stop loss
```

---

## Exit Logic

Exit decisions protect profits and limit losses:

```
                         EXIT DECISION TREE
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Have open position?   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    No                     Yes
                    │                       │
                    ▼                       ▼
             Return HOLD          ┌─────────────────────┐
                                  │ Stop loss triggered?│
                                  │ (price ≤ stop)      │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                                 Yes                    No
                                  │                     │
                                  ▼                     ▼
                           Return SELL        ┌─────────────────────┐
                           (reason: stop)     │ Exit signal valid?  │
                                              │ (e.g., trend reversal)│
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────┴──────────┐
                                              │                     │
                                             Yes                    No
                                              │                     │
                                              ▼                     ▼
                                       Return SELL        ┌─────────────────────┐
                                       (reason: signal)   │ Time limit reached? │
                                                          │ (max_hold_days)     │
                                                          └──────────┬──────────┘
                                                                     │
                                                          ┌──────────┴──────────┐
                                                          │                     │
                                                         Yes                    No
                                                          │                     │
                                                          ▼                     ▼
                                                   Return SELL          Check for stop
                                                   (reason: time)       adjustment, else HOLD
```

---

## Stop Loss Management

Strategies can implement various stop loss approaches:

### Fixed Stop Loss

Set at entry, never moves:
```
Stop Loss = Entry Price - (ATR × Multiplier)
```

### Trailing Stop Loss

Moves up as price increases, never down:
```
If current_high > previous_high:
    new_stop = current_high - (ATR × Multiplier)
    stop_loss = max(stop_loss, new_stop)
```

### Time-Decay Stop

Tightens as holding period increases:
```
days_held = current_bar - entry_bar
decay_factor = 1 - (days_held / max_hold_days × decay_rate)
stop_loss = entry_price + (initial_risk × decay_factor)
```

---

## Position Sizing

Strategies control position size through the `calculate_position_size()` method:

### Risk-Based (Default)

```
Risk Amount = Capital × Risk Percentage
Position Size = Risk Amount / (Entry - Stop Loss)

Example:
Capital = $100,000
Risk = 2%
Entry = $100, Stop = $95

Risk Amount = $100,000 × 0.02 = $2,000
Position Size = $2,000 / $5 = 400 shares
```

### Custom Sizing

Strategies can override to implement custom logic:
- Volatility-adjusted sizing
- Kelly criterion
- Fixed fractional
- Sector exposure limits

---

## Data Requirements

Each strategy declares required data columns:

```
Required Columns Declaration:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  AlphaTrendStrategy requires:                                              │
│  • date        - Timestamp                                                 │
│  • open        - Open price                                                │
│  • high        - High price                                                │
│  • low         - Low price                                                 │
│  • close       - Close price                                               │
│  • volume      - Trading volume                                            │
│  • atr_14      - 14-period ATR                                             │
│  • ema_50      - 50-period EMA                                             │
│  • mfi_14      - 14-period MFI                                             │
│                                                                             │
│  Data must have these columns or backtest will fail validation             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

If your data is missing required columns, either:
1. Add the indicators to your data collection
2. Use a strategy that doesn't require them
3. Pre-calculate them in the strategy's `prepare_data()` method

---

## Strategy Behavior in Different Modes

### Single Security Mode

- Strategy operates on one security at a time
- Full capital available for each trade
- Simple position tracking (0 or 1 position)

### Portfolio Mode

- Same strategy runs on multiple securities
- Capital is shared and limited
- Strategy may generate signal but not execute due to:
  - No available capital
  - Position limit reached
  - Lower priority than competing signals

The strategy itself doesn't change—the engine handles capital allocation.

---

## Related Documentation

- [Strategy Development Guide](../strategy-development/STRATEGY_GUIDE.md) — Creating custom strategies
- [Signal Types](../strategy-development/SIGNALS_AND_EXITS.md) — Detailed signal documentation
- [Testing Strategies](../strategy-development/TESTING_YOUR_STRATEGY.md) — Validation approaches
- [Included Strategy Files](../../strategies/) — Source code
