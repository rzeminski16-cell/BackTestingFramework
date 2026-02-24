# Strategy Development Guide

This guide covers everything you need to create custom trading strategies for the BackTesting Framework.

---

## Overview

A trading strategy in this framework is a Python class that inherits from `BaseStrategy` and:
1. Declares which direction it trades (LONG or SHORT)
2. Specifies which data columns it needs from raw CSV files
3. Decides when to enter positions (BUY)
4. Calculates stop losses and position sizes
5. Decides when to exit positions (SELL)
6. Optionally manages trailing stops, partial exits, and pyramiding

```
+-----------------------------------------------------------------------------+
|                          STRATEGY FLOW                                       |
|                                                                             |
|   Market Data                 Engine                     Strategy            |
|   ----------                  ------                     --------            |
|                                                                             |
|   Bar 1 ---------> generate_signal(context) -----> generate_entry_signal()  |
|                    (no position)                   returns None              |
|                    Result: HOLD                    (conditions not met)      |
|                                                                             |
|   Bar 2 ---------> generate_signal(context) -----> generate_entry_signal()  |
|                    (no position)                   returns Signal.buy()      |
|                    check_fundamentals() ---------> PASSED                   |
|                    ensure stop_loss set             |                        |
|                    Result: BUY                      V                        |
|                                           calculate_initial_stop_loss()     |
|                                           position_size()                   |
|                                                                             |
|   Bar 3 ---------> generate_signal(context) -----> generate_exit_signal()   |
|                    (in position)                   returns None              |
|                    check should_pyramid() -------> returns None              |
|                    Result: HOLD                                              |
|                                                                             |
|   Bar 4 ---------> generate_signal(context) -----> generate_exit_signal()   |
|                    (in position)                   returns Signal.sell()     |
|                    Result: SELL                                              |
|                                                                             |
+-----------------------------------------------------------------------------+
```

The engine calls `generate_signal(context)` on every bar. This orchestrator method
is defined in `BaseStrategy` and should NOT be overridden. It delegates to your
`generate_entry_signal()` and `generate_exit_signal()` implementations.

---

## Getting Started

### Strategy Location

Place your strategy files in the `strategies/` directory:

```
strategies/
    base_alphatrend_strategy.py     # Included production strategy (AlphaTrend)
    random_control_strategy.py      # Random baseline for comparison
    your_strategy.py                # Your custom strategy
```

### Parameter Configuration

Strategy parameters are defined in `config/strategy_parameters.json`. Each strategy
has an entry with parameter definitions including types, defaults, and optimization
ranges. When you create a new strategy, add its parameters to this file.

### Minimal Strategy Example

```python
from typing import List, Optional
import pandas as pd

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class SimpleMovingAverageStrategy(BaseStrategy):
    """Buy when price crosses above SMA, sell when crosses below."""

    def __init__(self, sma_period: int = 20, risk_percent: float = 2.0):
        # Pass all params to BaseStrategy which stores them in self.params
        # and calls _validate_parameters()
        super().__init__(sma_period=sma_period, risk_percent=risk_percent)
        self.sma_period = sma_period
        self.risk_percent = risk_percent

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'open', 'high', 'low', 'close', 'volume', 'atr_14_atr']

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-calculate SMA indicator."""
        df = data.copy()
        df['sma'] = df['close'].rolling(window=self.sma_period).mean()
        df['atr_14'] = df['atr_14_atr']
        return df

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Buy when price crosses above SMA."""
        current = context.current_bar
        previous = context.previous_bar

        if previous is None:
            return None

        if previous['close'] < previous['sma'] and current['close'] > current['sma']:
            return Signal.buy(
                size=1.0,
                stop_loss=self.calculate_initial_stop_loss(context),
                reason="Price crossed above SMA",
                direction=self.trade_direction
            )
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """Stop loss at 2x ATR below current price."""
        atr = context.get_indicator_value('atr_14')
        if atr is not None and atr > 0:
            return context.current_price - (atr * 2.0)
        return context.current_price * 0.95  # 5% fallback

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """Risk-based position sizing."""
        equity = context.total_equity
        risk_amount = equity * (self.risk_percent / 100)
        stop_distance = context.current_price - signal.stop_loss

        if stop_distance <= 0:
            return (context.available_capital * 0.1) / context.current_price

        stop_distance_base = stop_distance * context.fx_rate
        return risk_amount / stop_distance_base

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Sell when price crosses below SMA."""
        current = context.current_bar
        previous = context.previous_bar

        if previous is None:
            return None

        if previous['close'] > previous['sma'] and current['close'] < current['sma']:
            return Signal.sell(reason="Price crossed below SMA")
        return None
```

---

## Strategy Components

### Required Abstract Methods

Every strategy MUST implement all six of these:

| Method | Signature | Returns |
|--------|-----------|---------|
| `trade_direction` | `@property` | `TradeDirection.LONG` or `TradeDirection.SHORT` |
| `required_columns()` | `() -> List[str]` | Column names needed from raw CSV data |
| `generate_entry_signal()` | `(context: StrategyContext) -> Optional[Signal]` | `Signal.buy()` or `None` |
| `calculate_initial_stop_loss()` | `(context: StrategyContext) -> float` | Stop loss price |
| `position_size()` | `(context: StrategyContext, signal: Signal) -> float` | Number of shares |
| `generate_exit_signal()` | `(context: StrategyContext) -> Optional[Signal]` | `Signal.sell()` or `None` |

### Optional Overrides

| Method | Signature | Default Behavior |
|--------|-----------|------------------|
| `fundamental_rules` | `@property -> BaseFundamentalRules` | `AlwaysPassFundamentalRules` (no filtering) |
| `_prepare_data_impl()` | `(data: pd.DataFrame) -> pd.DataFrame` | Returns data unchanged |
| `should_adjust_stop()` | `(context: StrategyContext) -> Optional[float]` | Returns `None` (no trailing stop) |
| `should_partial_exit()` | `(context: StrategyContext) -> Optional[float]` | Returns `None` (no partial exits) |
| `should_pyramid()` | `(context: StrategyContext) -> Optional[Signal]` | Returns `None` (no pyramiding) |
| `should_check_stop_loss()` | `(context: StrategyContext) -> bool` | `True` if position has a stop loss |
| `should_check_take_profit()` | `(context: StrategyContext) -> bool` | `True` if position has a take profit |
| `_validate_parameters()` | `() -> None` | No validation |

### The __init__ Pattern

All strategies follow this initialization pattern:

```python
def __init__(self, my_param: float = 1.0, other_param: int = 10):
    # 1. Pass all params as kwargs to BaseStrategy
    super().__init__(my_param=my_param, other_param=other_param)
    # 2. Store params as instance attributes for convenient access
    self.my_param = my_param
    self.other_param = other_param
```

`BaseStrategy.__init__(**params)` does three things:
1. Stores params in `self.params` (a dict)
2. Calls `self._validate_parameters()` (override for custom checks)
3. Calls `self._validate_strategy_structure()` (checks abstract methods)

### How generate_signal() Works

The engine calls `generate_signal(context)` on every bar. This method is defined
in `BaseStrategy` and orchestrates the full decision flow. You do NOT override it.

```
+-----------------------------------------------------------------------+
|                  generate_signal(context)                              |
|                  [called by engine every bar]                          |
+-----------------------------------------------------------------------+
|                                                                       |
|   Has position?                                                       |
|       |                                                               |
|       +-- NO ----> generate_entry_signal(context)                     |
|       |                |                                              |
|       |                +-- None ---------> HOLD                       |
|       |                |                                              |
|       |                +-- Signal.buy() -> check_fundamentals()       |
|       |                                        |                      |
|       |                                        +-- FAILED -> HOLD     |
|       |                                        |                      |
|       |                                        +-- PASSED             |
|       |                                             |                 |
|       |                                             V                 |
|       |                                   ensure stop_loss is set     |
|       |                                   (calls calculate_initial_   |
|       |                                    stop_loss if needed)       |
|       |                                             |                 |
|       |                                             V                 |
|       |                                         return BUY            |
|       |                                                               |
|       +-- YES ---> generate_exit_signal(context)                      |
|                        |                                              |
|                        +-- Signal.sell() -> return SELL                |
|                        |                                              |
|                        +-- None --> should_pyramid(context)            |
|                                        |                              |
|                                        +-- Signal.pyramid() ->        |
|                                        |   return PYRAMID             |
|                                        |   (max 1 per trade)          |
|                                        |                              |
|                                        +-- None -> HOLD               |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## The Strategy Context

Your methods receive an immutable `StrategyContext` (frozen dataclass) containing
everything needed to make decisions:

```
StrategyContext (frozen=True)
|
|-- data                  DataFrame or HistoricalDataView
|                         (bars 0 to current_index only, no future data)
|
|-- current_index         int - index of current bar in data
|
|-- current_price         float - close price of current bar
|
|-- current_date          pd.Timestamp - date of current bar
|
|-- position              Optional[Position] - current open position or None
|   |-- entry_price       float - average entry price
|   |-- entry_date        datetime
|   |-- current_quantity  float - shares held now
|   |-- initial_quantity  float - shares at original entry
|   |-- direction         TradeDirection (LONG or SHORT)
|   |-- stop_loss         Optional[float]
|   |-- take_profit       Optional[float]
|   |-- has_pyramided     bool
|   +-- is_open           bool (property)
|
|-- available_capital     float - cash available (base currency)
|
|-- total_equity          float - capital + position value (base currency)
|
|-- symbol                str - security symbol being traded
|
|-- fx_rate               float - exchange rate to base currency
|                         (e.g., 0.8 means $1 = 0.8 base)
|
+-- Properties and Methods:
    |-- has_position           -> bool
    |-- current_bar            -> pd.Series (current bar's data)
    |-- previous_bar           -> Optional[pd.Series] (bar before current)
    |-- get_bar(offset)        -> Optional[pd.Series]
    |                             (negative=past, 0=current)
    |-- get_indicator_value(column, offset=0) -> Optional[float]
    |-- get_position_pl()      -> float (unrealized P/L in currency)
    +-- get_position_pl_pct()  -> float (unrealized P/L as percentage)
```

### Accessing Data

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    # Current bar (pd.Series with all columns)
    close = context.current_bar['close']
    volume = context.current_bar['volume']

    # Current price shortcut
    price = context.current_price

    # Previous bar (None if at first bar)
    prev = context.previous_bar
    if prev is not None:
        prev_close = prev['close']

    # Bar at arbitrary offset (negative = past, 0 = current)
    bar_3_ago = context.get_bar(-3)
    if bar_3_ago is not None:
        old_close = bar_3_ago['close']

    # Get a single indicator value (with optional offset)
    atr = context.get_indicator_value('atr_14')
    prev_atr = context.get_indicator_value('atr_14', offset=-1)

    # Check position state
    if context.has_position:
        entry_price = context.position.entry_price
        shares_held = context.position.current_quantity
        current_stop = context.position.stop_loss
        current_pl = context.get_position_pl()         # P/L in currency
        current_pl_pct = context.get_position_pl_pct()  # P/L as percentage
```

### What Does NOT Exist on StrategyContext

Do not attempt to use any of the following -- they are not part of the API:

- `context.get_previous_bar(n)` -- use `context.get_bar(-n)` instead
- `context.get_bars(n)` -- use `context.data.iloc[...]` if you need a range
- `context.get_indicator(name)` -- use `context.get_indicator_value(column)`
- `context.parameters` -- access `self.params` on your strategy instance
- `context.historical_data` -- use `context.data`
- `context.position.unrealized_pnl` -- use `context.get_position_pl()`
- `context.position.quantity` -- use `context.position.current_quantity`

---

## Signal Types

Strategies communicate intentions by returning `Signal` objects. Signals are created
ONLY through factory methods -- never by calling the `Signal()` constructor directly.

There are six signal types: BUY, SELL, HOLD, PARTIAL_EXIT, ADJUST_STOP, PYRAMID.

Access the type via `signal.type` (a `SignalType` enum value).

### Signal.buy()

Opens a new position:

```python
Signal.buy(
    size=1.0,                  # Position size fraction (required, 0.0-1.0)
    stop_loss=95.0,            # Stop loss price (optional, but ensured by engine)
    take_profit=None,          # Take profit price (optional)
    reason="Entry conditions", # Human-readable reason (optional)
    direction=TradeDirection.LONG  # Trade direction (default: LONG)
)
```

Note: `size` is a required positional argument with no default value.

### Signal.sell()

Closes the entire current position:

```python
Signal.sell(reason="Exit signal triggered")
```

### Signal.hold()

No action -- maintain current state:

```python
Signal.hold(reason="Waiting for setup")
```

### Signal.partial_exit()

Close a fraction of the position (used in `should_partial_exit()`):

```python
Signal.partial_exit(
    fraction=0.5,              # Fraction to exit (0.0-1.0, required)
    reason="Taking half profit"
)
```

Note: Uses `fraction`, NOT `quantity`. A value of 0.5 means exit 50% of the position.

### Signal.adjust_stop()

Move the stop loss to a new level (used in `should_adjust_stop()`):

```python
Signal.adjust_stop(
    new_stop_loss=102.50,      # New stop loss price (required)
    reason="Trailing stop update"
)
```

### Signal.pyramid()

Add to an existing position (used in `should_pyramid()`):

```python
Signal.pyramid(
    size=0.5,                  # Additional size fraction (required)
    reason="Momentum continuation"
)
```

When pyramiding occurs, the engine automatically adjusts the stop loss to break-even.
Maximum one pyramid per trade.

---

## Data Preparation

Use `_prepare_data_impl()` to pre-calculate custom indicators before backtesting
begins. This method is called once per security, and the results are added as
columns to the DataFrame that strategies access via `context.data`.

```python
def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
    """Calculate custom indicators before backtesting."""
    df = data.copy()

    # Normalize raw column names if needed
    df['atr_14'] = df['atr_14_atr']

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Custom indicator
    df['momentum'] = df['close'].pct_change(5)

    return df
```

### Important Rules

1. `_prepare_data_impl()` runs once before backtesting -- calculations are vectorized
   and fast, not repeated every bar.

2. The parent method `prepare_data()` (which calls `_prepare_data_impl()`) automatically:
   - Validates that all `required_columns()` exist in the raw data
   - Checks new columns for potential look-ahead bias

3. PREVENT LOOK-AHEAD BIAS -- only use causal operations:
   - ALLOWED: `.rolling()`, `.expanding()`, `.shift(n)` where n >= 0, `.ewm()`
   - FORBIDDEN: `.shift(-n)` where n > 0, global `.mean()`, `.std()` on full series

4. All standard indicators (ATR, EMA, RSI, MFI, etc.) must come from the raw CSV
   data. Custom calculations in `_prepare_data_impl()` are only for strategy-specific
   derived indicators.

---

## Position Sizing

### Method Signature

Position sizing is handled by the required `position_size()` method:

```python
def position_size(self, context: StrategyContext, signal: Signal) -> float:
    """
    Args:
        context: Current market context (has available_capital, total_equity, etc.)
        signal: The BUY signal with stop_loss already set

    Returns:
        Number of shares/units to buy (must be > 0)
    """
```

### Risk-Based Sizing (Standard Pattern)

The standard approach sizes the position so that if the stop loss is hit, you
lose exactly `risk_percent%` of equity:

```
Position Size = (Equity * Risk%) / (Stop Distance * FX Rate)
```

```python
def position_size(self, context: StrategyContext, signal: Signal) -> float:
    equity = context.total_equity
    risk_amount = equity * (self.risk_percent / 100)

    stop_distance = context.current_price - signal.stop_loss
    if stop_distance <= 0:
        # Fallback: use 10% of available capital
        return (context.available_capital * 0.1) / context.current_price

    # Convert stop distance to base currency
    stop_distance_base = stop_distance * context.fx_rate

    shares = risk_amount / stop_distance_base
    return shares
```

### Capital-Fraction Sizing (Alternative)

```python
def position_size(self, context: StrategyContext, signal: Signal) -> float:
    # Use 20% of available capital per position
    capital_to_use = context.available_capital * 0.20
    return capital_to_use / context.current_price
```

---

## Optional Features

### Trailing Stop Loss

Override `should_adjust_stop()` to implement trailing stops. Return the new stop
price, or `None` to leave it unchanged. The engine enforces that LONG stops can
only move UP and SHORT stops can only move DOWN.

```python
def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
    atr = context.get_indicator_value('atr_14')
    if atr is None:
        return None

    new_stop = context.current_price - (atr * 2.0)

    # Only move stop up (for LONG positions)
    if new_stop > context.position.stop_loss:
        return new_stop
    return None
```

### Partial Exits

Override `should_partial_exit()` to take partial profits. Return a fraction
(0.0 to 1.0) of the position to exit, or `None`.

```python
def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
    # Take 50% off at 10% profit
    if context.get_position_pl_pct() > 10.0:
        return 0.5
    return None
```

### Pyramiding

Override `should_pyramid()` to add to winning positions. Return `Signal.pyramid()`
or `None`. Maximum one pyramid per trade -- the engine tracks this automatically.
When a pyramid occurs, the stop loss is moved to break-even.

```python
def should_pyramid(self, context: StrategyContext) -> Optional[Signal]:
    if context.get_position_pl_pct() > 5.0:
        return Signal.pyramid(
            size=0.5,  # Add 50% of original allocation
            reason="Price momentum continuation"
        )
    return None
```

### Fundamental Rules

Override the `fundamental_rules` property to add a fundamental filter. Fundamentals
are checked every time `generate_entry_signal()` returns a BUY signal. If the check
fails, the trade is skipped.

```python
from Classes.Strategy.fundamental_rules import BaseFundamentalRules

@property
def fundamental_rules(self) -> BaseFundamentalRules:
    return MyCustomFundamentalRules()
```

The default is `AlwaysPassFundamentalRules`, which never blocks a trade.

### Custom Parameter Validation

Override `_validate_parameters()` to validate your strategy's parameters at
initialization time:

```python
def _validate_parameters(self) -> None:
    if self.risk_percent <= 0 or self.risk_percent > 100:
        raise ValueError(
            f"risk_percent must be between 0 and 100, got {self.risk_percent}"
        )
    if self.atr_multiplier <= 0:
        raise ValueError(
            f"atr_multiplier must be positive, got {self.atr_multiplier}"
        )
```

---

## Included Strategies

### BaseAlphaTrendStrategy

Production-ready trend-following strategy using the AlphaTrend indicator:

```
Location: strategies/base_alphatrend_strategy.py

Trade Direction: LONG only

Parameters:
    atr_multiplier ......... ATR multiplier for stop loss (default: 2.0)
    risk_percent ........... Percent of equity to risk per trade (default: 2.0)
    alpha_atr_multiplier ... Base multiplier for ATR bands (default: 1.0)
    smoothing_length ....... EMA period for AlphaTrend smoothing (default: 3)
    percentile_period ...... Lookback for dynamic MFI thresholds (default: 100)

Required columns from raw CSV:
    date, open, high, low, close, volume, atr_14_atr, mfi_14_mfi

Entry: AlphaTrend buy signal (crossover above smoothed AlphaTrend line)
Exit:  Stop loss only (no technical exit signal)
Stop:  ATR-based (current_price - atr_14 * atr_multiplier)

Features NOT used:
    - No volume confirmation filter
    - No SMA-based exits
    - No time-based exits
    - No trailing stop
    - No partial exits
    - No pyramiding
```

### RandomControlStrategy

Baseline strategy with random entries and exits for comparison:

```
Location: strategies/random_control_strategy.py

Trade Direction: LONG only

Parameters:
    entry_probability ...... Probability of entry each bar (default: 0.10)
    exit_probability ....... Probability of exit each bar (default: 0.10)
    atr_multiplier ......... ATR multiplier for stop loss (default: 2.0)
    risk_percent ........... Percent of equity to risk per trade (default: 2.0)
    random_seed ............ Seed for reproducibility (default: None)

Required columns from raw CSV:
    date, close, atr_14_atr

Purpose:
    - Validate that your strategy beats random
    - E-ratio should be approximately 1.0 for random
    - Returns should be near zero (minus transaction costs)
```

---

## Strategy Development Workflow

```
+-----------------------------------------------------------------------------+
|                    DEVELOPMENT WORKFLOW                                       |
+-----------------------------------------------------------------------------+

    1. DEFINE HYPOTHESIS
       |
       |  "AlphaTrend crossover with adaptive volatility
       |   bands identifies trend starts"
       |
       V
    2. IMPLEMENT STRATEGY
       |
       |  - Inherit from BaseStrategy
       |  - Implement all 6 required methods
       |  - Override _prepare_data_impl() if needed
       |  - Add parameters to config/strategy_parameters.json
       |
       V
    3. INITIAL BACKTEST
       |
       |  Test on single security
       |  Check basic metrics (profitable? Sharpe > 0?)
       |
       V
    4. VALIDATE EDGE
       |
       |  Run Edge Analysis (E-ratio > 1.0?)
       |  Compare to RandomControlStrategy
       |
       V
    5. OPTIMIZE PARAMETERS
       |
       |  Univariate optimization to understand each parameter
       |  Walk-forward optimization for robustness
       |
       V
    6. PORTFOLIO TEST
       |
       |  Test across multiple securities
       |  Check if edge persists with capital constraints
       |
       V
    7. FACTOR ANALYSIS
       |
       |  Identify what conditions produce best results
       |  Add filters to improve strategy
       |
       V
    8. ITERATE
       |
       |  Refine based on analysis
       |  Re-test to confirm improvements
       |
       V
    9. PRODUCTION READY
```

---

## Quick Reference: Common Patterns

### Checking Indicator Crossovers

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    current = context.current_bar
    previous = context.previous_bar
    if previous is None:
        return None

    # Bullish crossover: fast crosses above slow
    if (previous['sma_20'] <= previous['sma_50'] and
            current['sma_20'] > current['sma_50']):
        return Signal.buy(
            size=1.0,
            stop_loss=self.calculate_initial_stop_loss(context),
            reason="SMA bullish crossover",
            direction=self.trade_direction
        )
    return None
```

### ATR-Based Stop Loss

```python
def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
    atr = context.get_indicator_value('atr_14')
    if atr is not None and atr > 0:
        return context.current_price - (atr * self.atr_multiplier)
    return context.current_price * 0.95  # 5% fallback
```

### Warmup Period Guard

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    # Skip bars before indicators are valid
    if context.current_index < self.warmup_period:
        return None
    # ... rest of entry logic
```

---

## Next Steps

- [Strategy Structure](STRATEGY_STRUCTURE.md) -- Detailed anatomy of a strategy
- [Signals and Exits](SIGNALS_AND_EXITS.md) -- Signal types and exit rules
- [Testing Your Strategy](TESTING_YOUR_STRATEGY.md) -- Validation techniques

---

## Related Documentation

- [Backtesting Concepts](../concepts/BACKTESTING.md) -- How strategies are executed
- [Optimization](../concepts/OPTIMIZATION.md) -- Finding optimal parameters
- [Edge Analysis](../concepts/EDGE_ANALYSIS.md) -- Validating entry quality
