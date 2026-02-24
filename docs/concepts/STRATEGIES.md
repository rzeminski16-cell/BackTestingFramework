# Strategies

This document explains how trading strategies work within the framework.

---

## What Is a Strategy?

A strategy is a set of rules that determine:
1. **When to enter** a trade (buy conditions)
2. **When to exit** a trade (sell conditions, stop loss)
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
    │  1. Strategy receives parameters (e.g., atr_multiplier=2.0)            │
    │  2. Strategy reports required columns                                  │
    │     (e.g., ['date', 'open', 'high', 'low', 'close', 'volume',         │
    │             'atr_14_atr', 'mfi_14_mfi'])                               │
    │  3. Data is loaded and validated                                       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    DATA PREPARATION (Optional)
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  prepare_data() calls _prepare_data_impl() to pre-calculate            │
    │  strategy-specific indicators or derived values.                       │
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
    │      │  - data: DataFrame (bars 0 to current_index)               │   │
    │      │  - current_index: Position in data series                  │   │
    │      │  - current_price: Close price of current bar               │   │
    │      │  - current_date: Timestamp of current bar                  │   │
    │      │  - position: Current open position (if any)                │   │
    │      │  - available_capital: Capital in base currency              │   │
    │      │  - total_equity: Total equity in base currency             │   │
    │      │  - symbol: Security symbol                                 │   │
    │      │  - fx_rate: Exchange rate to base currency                 │   │
    │      └─────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │      ┌─────────────────────────────────────────────────────────────┐   │
    │      │            strategy.generate_signal(context)                │   │
    │      │                                                             │   │
    │      │  Orchestrates:                                              │   │
    │      │  1. If no position:                                        │   │
    │      │     - Calls generate_entry_signal()                        │   │
    │      │     - Checks fundamental_rules                             │   │
    │      │     - Ensures stop_loss is set                             │   │
    │      │  2. If in position:                                        │   │
    │      │     - Calls generate_exit_signal()                         │   │
    │      │     - Checks should_pyramid()                              │   │
    │      │                                                             │   │
    │      │  Returns: Signal (BUY, SELL, HOLD, PARTIAL_EXIT,           │   │
    │      │           ADJUST_STOP, PYRAMID)                            │   │
    │      └─────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

---

## Signal Types

Strategies communicate their decisions through signals created via factory methods:

| Signal Type      | Factory Method                            | When Used                                      |
|------------------|-------------------------------------------|-------------------------------------------------|
| **BUY**          | `Signal.buy(size, stop_loss, ...)`        | Entry conditions met, no current position       |
| **SELL**         | `Signal.sell(reason)`                     | Exit conditions met, position exists             |
| **HOLD**         | `Signal.hold(reason)`                     | No action                                       |
| **PARTIAL_EXIT** | `Signal.partial_exit(fraction, reason)`   | Scaling out of a winner (fraction 0-1)           |
| **ADJUST_STOP**  | `Signal.adjust_stop(new_stop_loss, reason)` | Trailing stop or risk reduction                |
| **PYRAMID**      | `Signal.pyramid(size, reason)`            | Adding to an existing position (max 1 per trade) |

### Signal Factory Methods

Signals are created exclusively through static factory methods on the `Signal` class
(located at `Classes/Models/signal.py`). Do not construct `Signal(...)` directly.

```
Signal.buy(
    size=1.0,                          # Position size fraction (0.0-1.0)
    stop_loss=145.00,                  # Initial stop loss price
    take_profit=None,                  # Optional take profit price
    reason="AlphaTrend buy signal",    # Optional: for logging
    direction=TradeDirection.LONG      # LONG or SHORT
)

Signal.sell(reason="Stop loss hit")

Signal.hold(reason="No conditions met")

Signal.partial_exit(
    fraction=0.5,                      # Fraction of position to exit (0-1)
    reason="Taking partial profit"
)

Signal.adjust_stop(
    new_stop_loss=148.00,              # New stop loss price
    reason="Trailing stop update"
)

Signal.pyramid(
    size=0.5,                          # Additional size fraction
    reason="Adding to winner"
)
```

### Signal Fields

A `Signal` dataclass has the following fields:

| Field            | Type             | Description                                      |
|------------------|------------------|--------------------------------------------------|
| `type`           | `SignalType`     | BUY, SELL, HOLD, PARTIAL_EXIT, ADJUST_STOP, PYRAMID |
| `size`           | `float`          | Position size fraction for BUY/PYRAMID           |
| `stop_loss`      | `Optional[float]`| Stop loss price for BUY signals                  |
| `take_profit`    | `Optional[float]`| Take profit price for BUY signals                |
| `new_stop_loss`  | `Optional[float]`| New stop loss price for ADJUST_STOP              |
| `reason`         | `str`            | Human-readable explanation                       |
| `direction`      | `TradeDirection` | LONG or SHORT (for BUY signals)                  |

---

## BaseStrategy (Abstract Base Class)

Located at: `Classes/Strategy/base_strategy.py`

All strategies inherit from `BaseStrategy`. The class defines the contract every
strategy must fulfill and provides the orchestration logic via `generate_signal()`.

### Required Abstract Methods

Every strategy **must** implement these six methods:

```
Required Methods:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. trade_direction (property)                                             │
│     Returns: TradeDirection.LONG or TradeDirection.SHORT                    │
│                                                                             │
│  2. required_columns()                                                     │
│     Returns: List[str] - column names needed from raw CSV data             │
│                                                                             │
│  3. generate_entry_signal(context: StrategyContext)                        │
│     Returns: Signal.buy() or None                                          │
│                                                                             │
│  4. calculate_initial_stop_loss(context: StrategyContext)                  │
│     Returns: float - stop loss price                                       │
│                                                                             │
│  5. position_size(context: StrategyContext, signal: Signal)                │
│     Returns: float - number of shares/units to buy                         │
│                                                                             │
│  6. generate_exit_signal(context: StrategyContext)                         │
│     Returns: Signal.sell() or None                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Optional Overrides

These methods have default behavior and may be overridden when needed:

```
Optional Methods:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  fundamental_rules (property)                                              │
│     Returns: BaseFundamentalRules                                          │
│     Default: AlwaysPassFundamentalRules (no filtering)                     │
│                                                                             │
│  prepare_data(data) -> pd.DataFrame                                        │
│     Pre-calculate custom indicators (delegates to _prepare_data_impl)      │
│     Default: Returns data unchanged                                        │
│                                                                             │
│  should_adjust_stop(context) -> Optional[float]                            │
│     Trailing stop logic. Return new stop price or None.                    │
│     Default: None (no trailing stop)                                       │
│                                                                             │
│  should_partial_exit(context) -> Optional[float]                           │
│     Partial profit-taking. Return fraction (0-1) or None.                  │
│     Default: None (no partial exits)                                       │
│                                                                             │
│  should_pyramid(context) -> Optional[Signal]                               │
│     Return Signal.pyramid() or None. Max 1 pyramid per trade.             │
│     Default: None (no pyramiding)                                          │
│                                                                             │
│  should_check_stop_loss(context) -> bool                                   │
│     Default: True if position has a stop loss                              │
│                                                                             │
│  should_check_take_profit(context) -> bool                                 │
│     Default: True if position has a take profit                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### generate_signal() Orchestration

The engine calls `generate_signal(context)` on every bar. This method orchestrates
the strategy logic -- you do not override it directly:

```
                    generate_signal(context) Flow
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Has open position?    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    No                     Yes
                    │                       │
                    ▼                       ▼
          ┌──────────────────┐   ┌──────────────────────┐
          │ generate_entry   │   │ generate_exit_signal()│
          │ _signal()        │   └──────────┬───────────┘
          └────────┬─────────┘              │
                   │                        ├── If SELL: return SELL
            Got BUY signal?                 │
                   │                        ▼
            ┌──────┴──────┐      ┌──────────────────────┐
            No           Yes     │ should_pyramid()      │
            │             │      │ (if not already       │
            ▼             ▼      │  pyramided)           │
         HOLD    check_fundamentals └──────────┬────────┘
                         │                     │
                  ┌──────┴──────┐        ├── If PYRAMID:
                  No           Yes       │    return PYRAMID
                  │             │        │
                  ▼             ▼        ▼
               HOLD     Ensure stop    HOLD
                        loss is set,
                        return BUY
```

---

## StrategyContext

Located at: `Classes/Strategy/strategy_context.py`

An immutable dataclass (`frozen=True`) passed to every strategy method. Ensures
strategies cannot accidentally modify state.

### Fields

| Field              | Type                           | Description                                 |
|--------------------|--------------------------------|---------------------------------------------|
| `data`             | `DataFrame` or `HistoricalDataView` | Bars 0 to current_index (no future data) |
| `current_index`    | `int`                          | Index of the current bar in the data        |
| `current_price`    | `float`                        | Close price of current bar                  |
| `current_date`     | `pd.Timestamp`                 | Date of current bar                         |
| `position`         | `Optional[Position]`           | Current open position (None if flat)        |
| `available_capital` | `float`                       | Available capital in base currency          |
| `total_equity`     | `float`                        | Total equity in base currency               |
| `symbol`           | `str`                          | Security symbol being traded                |
| `fx_rate`          | `float`                        | Exchange rate to base currency              |

### Properties and Methods

| Member                                | Returns             | Description                                  |
|---------------------------------------|---------------------|----------------------------------------------|
| `has_position`                        | `bool`              | True if currently in a position              |
| `current_bar`                         | `pd.Series`         | Current bar's OHLCV and indicator data       |
| `previous_bar`                        | `Optional[pd.Series]` | Previous bar (None if at first bar)        |
| `get_bar(offset)`                     | `Optional[pd.Series]` | Bar at offset (negative=past, 0=current)   |
| `get_indicator_value(column, offset)` | `Optional[float]`   | Indicator value at offset (default 0)        |
| `get_position_pl()`                   | `float`             | Current position P/L in currency             |
| `get_position_pl_pct()`              | `float`             | Current position P/L as percentage           |

---

## Included Strategies

The framework includes production-ready strategies you can use immediately.

### AlphaTrend Strategy

Located at: `strategies/base_alphatrend_strategy.py`
Class: `BaseAlphaTrendStrategy`

A trend-following strategy based on the AlphaTrend indicator:

**Core Concept:**
- Uses ATR (Average True Range) to adapt to volatility
- Uses MFI (Money Flow Index) to confirm trend direction
- Enters on AlphaTrend crossover signal (AlphaTrend crosses above smoothed line)
- Exits via stop loss only

**Direction:** LONG only

**Parameters (all optimizable):**

| Parameter            | Type    | Default | Description                                    |
|----------------------|---------|---------|------------------------------------------------|
| `atr_multiplier`     | `float` | 2.0     | ATR multiplier for stop loss                   |
| `risk_percent`       | `float` | 2.0     | Percent of equity to risk per trade            |
| `alpha_atr_multiplier` | `float` | 1.0   | Base multiplier for ATR bands in AlphaTrend    |
| `smoothing_length`   | `int`   | 3       | EMA period for AlphaTrend smoothing            |
| `percentile_period`  | `int`   | 100     | Lookback period for dynamic MFI thresholds     |

**Required Data Columns:**

```
Required Columns:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  BaseAlphaTrendStrategy requires:                                          │
│  - date          Timestamp                                                 │
│  - open          Open price                                                │
│  - high          High price                                                │
│  - low           Low price                                                 │
│  - close         Close price                                               │
│  - volume        Trading volume                                            │
│  - atr_14_atr    14-period ATR (pre-calculated in raw data)                │
│  - mfi_14_mfi    14-period MFI (pre-calculated in raw data)                │
│                                                                             │
│  Data must have these columns or backtest will fail validation             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Exit Conditions:**
1. Stop loss triggered (close price <= stop loss)

There is no time-based exit, no signal-based exit, and no momentum reversal exit.
The `generate_exit_signal()` method returns `None` -- all exits are handled by
the stop loss mechanism.

**Stop Loss:** `Entry Price - (ATR14 x atr_multiplier)`

**Position Sizing:** Risk-based. The number of shares is calculated so that if the
stop loss is hit, the loss equals `risk_percent`% of total equity.

```
Position Size = (Equity x risk_percent / 100) / (Stop Distance x fx_rate)
```

### Random Control Strategy

Located at: `strategies/random_control_strategy.py`
Class: `RandomControlStrategy`

A baseline strategy that enters and exits randomly:

**Purpose:**
- Provides a benchmark for comparison
- E-ratio should be approximately 1.0 (no edge)
- If your strategy's E-ratio is not significantly better, it may lack real edge

**Direction:** LONG only

**Parameters:**

| Parameter            | Type             | Default | Description                           |
|----------------------|------------------|---------|---------------------------------------|
| `entry_probability`  | `float`          | 0.10    | Probability of entry each bar (0-1)   |
| `exit_probability`   | `float`          | 0.10    | Probability of exit each bar (0-1)    |
| `atr_multiplier`     | `float`          | 2.0     | ATR multiplier for stop loss          |
| `risk_percent`       | `float`          | 2.0     | Percent of equity to risk per trade   |
| `random_seed`        | `Optional[int]`  | None    | Seed for reproducibility              |

**Required Data Columns:** `date`, `close`, `atr_14_atr`

**Stop Loss:** `Entry Price - (ATR14 x atr_multiplier)`

**Position Sizing:** Risk-based (same formula as AlphaTrend).

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
│      'description': 'ATR multiplier for stop loss calculation'             │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

This metadata enables:
- GUI parameter input with validation
- Automatic optimization range detection
- Documentation generation

### Parameter Registry

All strategy parameters are centrally registered in `config/strategy_parameters.json`.
Parameter presets are saved in `config/strategy_presets/`.

```
config/strategy_parameters.json
┌─────────────────────────────────────────────────────────────────────────────┐
│  {                                                                          │
│    "BaseAlphaTrendStrategy": {                                             │
│      "atr_multiplier": {"type": "float", "default": 2.0, ...},            │
│      "risk_percent": {"type": "float", "default": 2.0, ...},              │
│      "alpha_atr_multiplier": {"type": "float", "default": 1.0, ...},      │
│      "smoothing_length": {"type": "int", "default": 3, ...},              │
│      "percentile_period": {"type": "int", "default": 100, ...},           │
│      ...                                                                    │
│    },                                                                       │
│    "RandomControlStrategy": {                                              │
│      "entry_probability": {"type": "float", "default": 0.10, ...},        │
│      "exit_probability": {"type": "float", "default": 0.10, ...},         │
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
                                  │ Entry signal valid? │
                                  │ (strategy-specific) │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                                  No                   Yes
                                  │                     │
                                  ▼                     ▼
                           Return HOLD        ┌─────────────────────┐
                                              │ Fundamentals pass?  │
                                              │ (fundamental_rules) │
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

### Example Entry Implementation

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    # Skip warmup period
    if context.current_index < 100:
        return None

    # Check entry condition
    filtered_buy = context.get_indicator_value('filtered_buy')
    if filtered_buy:
        return Signal.buy(
            size=1.0,
            stop_loss=self.calculate_initial_stop_loss(context),
            reason="AlphaTrend buy signal",
            direction=self.trade_direction
        )
    return None
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
                                  │ (close price <=     │
                                  │  stop loss)         │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                                 Yes                    No
                                  │                     │
                                  ▼                     ▼
                           Engine forces      ┌─────────────────────┐
                           SELL               │ Exit signal valid?  │
                           (reason: stop)     │ (generate_exit      │
                                              │  _signal)           │
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────┴──────────┐
                                              │                     │
                                             Yes                    No
                                              │                     │
                                              ▼                     ▼
                                       Return SELL         Check for stop
                                       (reason: signal)    adjustment and
                                                           pyramiding,
                                                           else HOLD
```

**Important:** Stop loss is checked using the CLOSE price, not the bar low.

---

## Stop Loss Management

Strategies can implement various stop loss approaches:

### Fixed Stop Loss

Set at entry, never moves:
```
Stop Loss = Entry Price - (ATR x Multiplier)
```

### Trailing Stop Loss

Implemented via `should_adjust_stop()`. The engine enforces that for LONG
positions, the stop can only move UP; for SHORT positions, it can only move DOWN.

```python
def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
    atr = context.get_indicator_value('atr_14')
    new_stop = context.current_price - (atr * 2.0)
    if new_stop > context.position.stop_loss:
        return new_stop
    return None
```

---

## Position Sizing

Strategies control position size through the `position_size()` method:

### Risk-Based (Default Approach)

```
Risk Amount = Equity x Risk Percentage
Position Size = Risk Amount / (Stop Distance x FX Rate)

Example:
Equity = $100,000
Risk = 2%
Entry = $100, Stop = $95, FX Rate = 1.0

Risk Amount = $100,000 x 0.02 = $2,000
Stop Distance = $100 - $95 = $5
Position Size = $2,000 / ($5 x 1.0) = 400 shares
```

### Custom Sizing

Strategies can override `position_size()` to implement custom logic:
- Volatility-adjusted sizing
- Kelly criterion
- Fixed fractional
- Sector exposure limits

---

## Position Model

Located at: `Classes/Models/position.py`

The `Position` dataclass tracks an open position:

| Field                   | Type             | Description                              |
|-------------------------|------------------|------------------------------------------|
| `entry_price`           | `float`          | Average entry price (updated on pyramid) |
| `entry_date`            | `datetime`       | Position entry date                      |
| `initial_quantity`      | `float`          | Original position size before pyramiding |
| `current_quantity`      | `float`          | Current size after partials/pyramids     |
| `direction`             | `TradeDirection` | LONG or SHORT                            |
| `stop_loss`             | `Optional[float]`| Current stop loss price                  |
| `take_profit`           | `Optional[float]`| Take profit price                        |
| `has_pyramided`         | `bool`           | Whether position has been pyramided      |
| `total_commission_paid` | `float`          | Total commission paid on this position   |
| `entry_fx_rate`         | `float`          | FX rate at entry                         |
| `security_currency`     | `str`            | Currency the security is denominated in  |
| `entry_equity`          | `float`          | Total portfolio equity at time of entry  |

**Key Methods:**
- `calculate_pl(price)` -- unrealized P/L at given price
- `calculate_pl_pct(price)` -- unrealized P/L as percentage
- `duration_days(date)` -- position duration in days

---

## Data Requirements

Each strategy declares required data columns via `required_columns()`:

```
Required Columns Declaration:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  BaseAlphaTrendStrategy requires:                                          │
│  - date          Timestamp                                                 │
│  - open          Open price                                                │
│  - high          High price                                                │
│  - low           Low price                                                 │
│  - close         Close price                                               │
│  - volume        Trading volume                                            │
│  - atr_14_atr    14-period ATR (from raw data CSV)                         │
│  - mfi_14_mfi    14-period MFI (from raw data CSV)                         │
│                                                                             │
│  Data must have these columns or backtest will fail validation             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

If your data is missing required columns, either:
1. Add the indicators to your data collection
2. Use a strategy that does not require them
3. Pre-calculate them in the strategy's `_prepare_data_impl()` method

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

The strategy itself does not change -- the engine handles capital allocation.

---

## Related Documentation

- [Strategy Development Guide](../strategy-development/STRATEGY_GUIDE.md) -- Creating custom strategies
- [Signal Types](../strategy-development/SIGNALS_AND_EXITS.md) -- Detailed signal documentation
- [Testing Strategies](../strategy-development/TESTING_YOUR_STRATEGY.md) -- Validation approaches
- [Included Strategy Files](../../strategies/) -- Source code
