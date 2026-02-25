# Strategy Structure

This document provides a detailed breakdown of strategy class architecture and lifecycle.

---

## Class Hierarchy

```
+-----------------------------------------------------------------------------+
|                         CLASS HIERARCHY                                       |
|                                                                              |
|                        BaseStrategy (ABC)                                    |
|                    Classes/Strategy/base_strategy.py                         |
|                             |                                                |
|              +--------------+--------------+                                 |
|              |              |              |                                 |
|              v              v              v                                 |
|   BaseAlphaTrend   RandomControl    YourStrategy                            |
|     Strategy         Strategy         (Custom)                               |
|   (Production)      (Baseline)                                               |
|                                                                              |
+-----------------------------------------------------------------------------+
```

All strategies inherit from `BaseStrategy` located at `Classes/Strategy/base_strategy.py`.

Key supporting classes:

- `Signal` (`Classes/Models/signal.py`) -- trading signal with factory methods
- `StrategyContext` (`Classes/Strategy/strategy_context.py`) -- immutable bar context
- `Position` (`Classes/Models/position.py`) -- open position state
- `TradeDirection` (`Classes/Models/trade_direction.py`) -- LONG or SHORT enum

---

## Complete Strategy Template

```python
from typing import List, Optional
import pandas as pd

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class YourStrategy(BaseStrategy):
    """
    Your strategy description.

    This strategy enters when [conditions] and exits when [conditions].

    STRATEGY STRUCTURE:
    - Trade Direction: LONG only
    - Fundamental Rules: None (always pass)
    - Entry Rules: [describe entry conditions]
    - Initial Stop Loss: ATR-based (ATR14 x atr_multiplier)
    - Position Sizing: Risk-based sizing
    - Trailing Stop: [None or describe]
    - Partial Exit: [None or describe]
    - Full Exit Rules: [describe exit conditions]
    - Pyramiding: [None or describe]

    RAW DATA INDICATORS (NOT OPTIMIZABLE):
        - atr_14_atr: Average True Range (14-period)

    OPTIMIZABLE PARAMETERS:
        - atr_multiplier: ATR multiplier for stop loss (default: 2.0)
        - risk_percent: Percent of equity to risk per trade (default: 2.0)
        - lookback: Number of bars for signal lookback (default: 20)
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        risk_percent: float = 2.0,
        lookback: int = 20,
    ):
        """
        Initialize strategy with parameters.

        All parameters MUST be passed to super().__init__() as keyword
        arguments. super().__init__() stores them in self.params and
        calls self._validate_parameters().

        Args:
            atr_multiplier: ATR multiplier for stop loss distance
            risk_percent: Percent of equity to risk per trade
            lookback: Number of bars for signal lookback
        """
        super().__init__(
            atr_multiplier=atr_multiplier,
            risk_percent=risk_percent,
            lookback=lookback,
        )
        self.atr_multiplier = atr_multiplier
        self.risk_percent = risk_percent
        self.lookback = lookback

    # =========================================================================
    # OPTIONAL: Parameter Validation
    # =========================================================================

    def _validate_parameters(self) -> None:
        """
        Custom parameter validation. Called automatically by
        super().__init__().

        Raises:
            ValueError: If parameters are invalid
        """
        if self.params.get('atr_multiplier', 2.0) <= 0:
            raise ValueError("atr_multiplier must be positive")
        if self.params.get('risk_percent', 2.0) <= 0:
            raise ValueError("risk_percent must be positive")

    # =========================================================================
    # REQUIRED: Trade Direction
    # =========================================================================

    @property
    def trade_direction(self) -> TradeDirection:
        """Return LONG or SHORT. This is a required abstract property."""
        return TradeDirection.LONG

    # =========================================================================
    # REQUIRED: Data Columns
    # =========================================================================

    def required_columns(self) -> List[str]:
        """
        Return list of column names that MUST exist in the raw CSV data.

        Must include 'date' and 'close' at minimum.

        Returns:
            List of required column names (lowercase)
        """
        return ['date', 'open', 'high', 'low', 'close', 'volume', 'atr_14_atr']

    # =========================================================================
    # OPTIONAL: Data Preparation
    # =========================================================================

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate custom indicators before the backtest loop.

        This is called once via prepare_data(), which validates raw columns
        first and checks for look-ahead bias afterward.

        IMPORTANT -- only use causal operations:
        - ALLOWED: .rolling(), .expanding(), .shift(n) where n >= 0
        - FORBIDDEN: .shift(-n) where n > 0, global .mean()/.std()

        Args:
            data: Raw OHLCV data with pre-calculated standard indicators

        Returns:
            DataFrame with custom indicator columns added
        """
        df = data.copy()

        # Normalize column names
        df['atr_14'] = df['atr_14_atr']

        # Moving averages
        df['sma_fast'] = df['close'].rolling(window=10).mean()
        df['sma_slow'] = df['close'].rolling(window=self.lookback).mean()

        # Custom indicator
        df['momentum'] = df['close'].pct_change(5) * 100

        return df

    # =========================================================================
    # REQUIRED: Entry Signal
    # =========================================================================

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate entry signal. Called every bar when NOT in a position.

        Return Signal.buy() when entry conditions are met, or None to skip.
        The engine will call calculate_initial_stop_loss() to set the stop
        if the signal does not already include one.

        Fundamental rules are checked AFTER this returns a BUY signal.
        If fundamentals fail, the trade is skipped.

        Args:
            context: Current market context (immutable)

        Returns:
            Signal.buy() if entry conditions met, None otherwise
        """
        # Skip warmup period
        if context.current_index < self.lookback:
            return None

        current = context.current_bar
        previous = context.previous_bar

        if previous is None:
            return None

        # Check for NaN in indicators
        if pd.isna(current.get('sma_fast')) or pd.isna(current.get('sma_slow')):
            return None

        # Entry condition: fast SMA crosses above slow SMA
        cross_above = (
            previous['sma_fast'] <= previous['sma_slow']
            and current['sma_fast'] > current['sma_slow']
        )

        if cross_above:
            return Signal.buy(
                size=1.0,
                stop_loss=self.calculate_initial_stop_loss(context),
                reason="SMA crossover (fast above slow)",
                direction=self.trade_direction,
            )

        return None

    # =========================================================================
    # REQUIRED: Initial Stop Loss
    # =========================================================================

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """
        Calculate initial stop loss price.

        Called when generating entry signals. The engine also calls this
        as a fallback if generate_entry_signal() returns a BUY signal
        without a stop_loss set.

        Args:
            context: Current market context

        Returns:
            Stop loss price (absolute price level)
        """
        atr = context.get_indicator_value('atr_14')

        if atr is not None and atr > 0:
            return context.current_price - (atr * self.atr_multiplier)
        else:
            # Fallback: 5% stop loss
            return context.current_price * 0.95

    # =========================================================================
    # REQUIRED: Position Sizing
    # =========================================================================

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate number of shares/units to buy.

        Called when a BUY signal is generated.

        Args:
            context: Current market context
            signal: BUY signal with stop_loss set

        Returns:
            Number of shares to buy (must be > 0)
        """
        if signal.stop_loss is None:
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        equity = context.total_equity
        risk_amount = equity * (self.risk_percent / 100)
        stop_distance = context.current_price - signal.stop_loss

        if stop_distance <= 0:
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        # Convert stop distance to base currency
        stop_distance_base = stop_distance * context.fx_rate

        return risk_amount / stop_distance_base

    # =========================================================================
    # REQUIRED: Exit Signal
    # =========================================================================

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate exit signal. Called every bar when IN a position.

        Return Signal.sell() when exit conditions are met, or None
        to let the position continue (stop loss and take profit are
        checked separately by the engine).

        Args:
            context: Current market context

        Returns:
            Signal.sell() if exit conditions met, None otherwise
        """
        current = context.current_bar
        previous = context.previous_bar

        if previous is None:
            return None

        # Exit condition: fast SMA crosses below slow SMA
        cross_below = (
            previous['sma_fast'] >= previous['sma_slow']
            and current['sma_fast'] < current['sma_slow']
        )

        if cross_below:
            return Signal.sell(reason="SMA crossover (fast below slow)")

        # Time-based exit using position duration
        position = context.position
        if position.duration_days(context.current_date) >= 60:
            return Signal.sell(reason="Time exit (60 days)")

        return None

    # =========================================================================
    # OPTIONAL: Trailing Stop
    # =========================================================================

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """
        Trailing stop logic. Called every bar when in a position.

        For LONG positions, the engine enforces that the stop can only
        move UP. For SHORT positions, the stop can only move DOWN.

        Args:
            context: Current market context

        Returns:
            New stop loss price, or None if no adjustment needed
        """
        atr = context.get_indicator_value('atr_14')
        if atr is None or atr <= 0:
            return None

        new_stop = context.current_price - (atr * self.atr_multiplier)

        # Only return if the new stop is higher (engine enforces this too)
        if context.position.stop_loss is not None and new_stop > context.position.stop_loss:
            return new_stop

        return None

    # =========================================================================
    # OPTIONAL: Partial Exit
    # =========================================================================

    def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
        """
        Partial profit-taking logic. Called every bar when in a position.

        Args:
            context: Current market context

        Returns:
            Fraction of position to exit (0.0-1.0), or None
        """
        # Take 50% profit at 10% gain
        if context.get_position_pl_pct() > 10.0:
            return 0.5
        return None

    # =========================================================================
    # OPTIONAL: Pyramiding
    # =========================================================================

    def should_pyramid(self, context: StrategyContext) -> Optional[Signal]:
        """
        Pyramiding logic. Called every bar when in a position.

        Maximum ONE pyramid per trade is enforced by the engine.
        When pyramiding occurs, the stop loss is automatically moved
        to break-even (accounting for increased position size and
        commission costs).

        Args:
            context: Current market context

        Returns:
            Signal.pyramid() if should add to position, None otherwise
        """
        # Pyramid when price moves 5% in our favor
        if context.get_position_pl_pct() > 5.0:
            return Signal.pyramid(
                size=0.5,  # Add 50% of original position
                reason="Price momentum continuation",
            )
        return None
```

---

## Strategy Lifecycle

```
+-----------------------------------------------------------------------------+
|                      STRATEGY LIFECYCLE                                      |
+-----------------------------------------------------------------------------+

    INITIALIZATION
    |
    +-- __init__(params)
    |   +-- super().__init__(**params) stores params in self.params
    |   +-- _validate_parameters() is called automatically
    |   +-- Strategy stores params as instance attributes
    |
    v
    DATA PREPARATION
    |
    +-- prepare_data(data) [called by engine]
    |   +-- Validates all required_columns() exist in raw data
    |   +-- Calls _prepare_data_impl(data) for custom indicators
    |   +-- Checks new columns for look-ahead bias
    |
    v
    BAR LOOP (for each bar in data)
    |
    +-- Engine builds StrategyContext for current bar
    |
    +-- Engine calls generate_signal(context) [DO NOT OVERRIDE]
    |   |
    |   +-- If NO position:
    |   |   +-- Calls generate_entry_signal(context)
    |   |   +-- If BUY signal returned:
    |   |   |   +-- Checks fundamental_rules
    |   |   |   +-- If fundamentals pass:
    |   |   |   |   +-- Ensures stop_loss is set via calculate_initial_stop_loss()
    |   |   |   |   +-- Returns BUY signal
    |   |   |   +-- If fundamentals fail:
    |   |   |       +-- Returns HOLD
    |   |   +-- If None returned:
    |   |       +-- Returns HOLD
    |   |
    |   +-- If IN position:
    |       +-- Calls generate_exit_signal(context)
    |       +-- If SELL signal returned:
    |       |   +-- Returns SELL
    |       +-- If None returned:
    |           +-- Calls should_pyramid(context) [if not already pyramided]
    |           +-- Returns PYRAMID signal or HOLD
    |
    +-- Engine separately handles (via PositionManager):
    |   +-- should_check_stop_loss(context) --> check stop loss hit
    |   +-- should_check_take_profit(context) --> check take profit hit
    |   +-- should_adjust_stop(context) --> trailing stop adjustment
    |   +-- should_partial_exit(context) --> partial position exit
    |
    +-- Engine calls position_size(context, signal) when opening position
    |
    +-- Continue to next bar...
    |
    v
    BACKTEST END
    |
    +-- Results returned to caller
```

### Key Lifecycle Notes

- There is NO `reset()` method. Strategy state is not cleared between backtests.
- There is NO `on_position_opened()` or `on_position_closed()` lifecycle hook.
- Strategies do NOT override `generate_signal()`. They implement the individual
  methods: `generate_entry_signal()`, `generate_exit_signal()`, etc.
- The engine orchestrates all the method calls. The strategy only provides
  the individual decision methods.

---

## Signal Reference

Signals are created exclusively through factory methods. Never use the `Signal()`
constructor directly.

```
+-----------------------------------------------------------------------------+
|                          SIGNAL FACTORY METHODS                               |
+-----------------------------------------------------------------------------+
|                                                                              |
|   Signal.buy(size, stop_loss, take_profit, reason, direction)               |
|       -> Open a new position                                                 |
|       -> size: fraction of capital (0.0-1.0)                                 |
|       -> direction: TradeDirection.LONG or TradeDirection.SHORT               |
|                                                                              |
|   Signal.sell(reason)                                                        |
|       -> Close entire position                                               |
|                                                                              |
|   Signal.hold(reason)                                                        |
|       -> No action                                                           |
|                                                                              |
|   Signal.partial_exit(fraction, reason)                                      |
|       -> Close part of position                                              |
|       -> fraction: 0.0-1.0 (NOT quantity)                                    |
|                                                                              |
|   Signal.adjust_stop(new_stop_loss, reason)                                  |
|       -> Adjust stop loss level                                              |
|                                                                              |
|   Signal.pyramid(size, reason)                                               |
|       -> Add to existing position (max 1 per trade)                          |
|       -> size: fraction of capital (0.0-1.0)                                 |
|                                                                              |
+-----------------------------------------------------------------------------+
|                                                                              |
|   Signal types (SignalType enum):                                            |
|       BUY, SELL, HOLD, PARTIAL_EXIT, ADJUST_STOP, PYRAMID                   |
|                                                                              |
|   Access the type via signal.type (NOT signal.signal_type)                   |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## StrategyContext Reference

`StrategyContext` is a frozen (immutable) dataclass passed to every strategy method.
The strategy cannot modify it.

```
+-----------------------------------------------------------------------------+
|                       STRATEGY CONTEXT                                       |
+-----------------------------------------------------------------------------+
|                                                                              |
|   FIELDS (set by engine):                                                    |
|   -----------------------                                                    |
|   data              DataFrame or HistoricalDataView (bars 0..current_index)  |
|   current_index     int -- index of current bar in data                      |
|   current_price     float -- close price of current bar                      |
|   current_date      pd.Timestamp -- date of current bar                      |
|   position          Optional[Position] -- current open position or None      |
|   available_capital float -- available capital in base currency              |
|   total_equity      float -- total equity in base currency                   |
|   symbol            str -- security symbol being traded                      |
|   fx_rate           float -- exchange rate (security currency -> base)       |
|                                                                              |
|   PROPERTIES:                                                                |
|   -----------                                                                |
|   has_position      bool -- True if position is open                         |
|   current_bar       pd.Series -- current bar data (all columns)              |
|   previous_bar      Optional[pd.Series] -- previous bar (None if first bar)  |
|                                                                              |
|   METHODS:                                                                   |
|   --------                                                                   |
|   get_bar(offset)            Optional[pd.Series]                             |
|       offset: negative=past, 0=current. Returns None if out of bounds.       |
|                                                                              |
|   get_indicator_value(column, offset=0)    Optional[float]                   |
|       Get a single indicator value from a bar. Returns None if missing.      |
|                                                                              |
|   get_position_pl()          float                                           |
|       Current position P/L in currency (0 if no position).                   |
|                                                                              |
|   get_position_pl_pct()      float                                           |
|       Current position P/L as percentage (0 if no position).                 |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### What Does NOT Exist on StrategyContext

These are common mistakes. None of these exist:

- `context.get_previous_bar(n)` -- use `context.get_bar(-n)` instead
- `context.get_bars(n)` -- use `context.data.iloc[start:end]` for slices
- `context.get_indicator(name)` -- use `context.get_indicator_value(column)` instead
- `context.parameters` -- access params via `self.param_name` on the strategy
- `context.historical_data` -- use `context.data` instead

---

## Position Reference

The `Position` dataclass tracks an open position. Accessed via `context.position`.

```
+-----------------------------------------------------------------------------+
|                          POSITION FIELDS                                     |
+-----------------------------------------------------------------------------+
|                                                                              |
|   entry_price          float -- average entry price (updated on pyramid)     |
|   entry_date           datetime -- position entry date                       |
|   initial_quantity     float -- original position size before pyramiding     |
|   current_quantity     float -- current size after partial exits             |
|   direction            TradeDirection -- LONG or SHORT                       |
|   stop_loss            Optional[float] -- current stop loss price            |
|   take_profit          Optional[float] -- take profit price                  |
|   has_pyramided        bool -- whether position has been pyramided           |
|   total_commission_paid float -- total commission on this position           |
|   entry_fx_rate        float -- FX rate at entry                             |
|   security_currency    str -- currency the security is denominated in        |
|   entry_equity         float -- total portfolio equity at time of entry      |
|                                                                              |
|   METHODS:                                                                   |
|   --------                                                                   |
|   calculate_pl(price)        float -- unrealized P/L at given price          |
|   calculate_pl_pct(price)    float -- unrealized P/L as percentage           |
|   duration_days(date)        int -- days since entry                         |
|   calculate_value(price)     float -- current position value                 |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## State Management

### Strategy State vs Context State

```
+-----------------------------------------------------------------------------+
|                     STATE MANAGEMENT                                         |
|                                                                              |
|   STRATEGY STATE (self.*)           CONTEXT STATE (context.*)               |
|   -----------------------          ------------------------                  |
|   * Parameters (self.params)       * data (read-only DataFrame)             |
|   * Instance attributes            * current_bar (read-only)                |
|   * Internal counters/flags        * position (read-only)                   |
|   * Cached calculations            * available_capital                      |
|                                    * total_equity                           |
|                                    * current_price                          |
|                                    * fx_rate                                |
|                                                                              |
|   Persists across bars             Fresh each bar (immutable)               |
|   You control                      Engine controls                          |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Stateful Strategy Example

If your strategy needs to track state across bars, use instance attributes.
Be aware there is no `reset()` method -- if you run multiple backtests with
the same strategy instance, state from the previous run will carry over.

```python
class StatefulStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20):
        super().__init__(lookback=lookback)
        self.lookback = lookback
        self.bars_since_entry = 0

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close', 'atr_14_atr']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        # Reset counter when not in a position
        self.bars_since_entry = 0

        if context.current_index < self.lookback:
            return None

        # Entry logic...
        return None

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        self.bars_since_entry += 1

        # Time-based exit after 20 bars
        if self.bars_since_entry >= 20:
            return Signal.sell(reason="Time exit")

        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        return context.current_price * 0.95

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return context.available_capital * 0.1 / context.current_price
```

### Stateless Strategy Example

Prefer stateless strategies when possible. Use context data and position
fields (like `duration_days`) instead of tracking state manually.

```python
class StatelessStrategy(BaseStrategy):
    def __init__(self, sma_period: int = 50):
        super().__init__(sma_period=sma_period)
        self.sma_period = sma_period

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close', 'atr_14_atr']

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['sma'] = df['close'].rolling(window=self.sma_period).mean()
        return df

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        sma = context.get_indicator_value('sma')
        if sma is not None and context.current_price > sma:
            return Signal.buy(
                size=1.0,
                stop_loss=self.calculate_initial_stop_loss(context),
                reason="Price above SMA",
                direction=self.trade_direction,
            )
        return None

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        sma = context.get_indicator_value('sma')
        if sma is not None and context.current_price < sma:
            return Signal.sell(reason="Price below SMA")
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        atr = context.get_indicator_value('atr_14')
        if atr and atr > 0:
            return context.current_price - (atr * 2.0)
        return context.current_price * 0.95

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return context.available_capital * 0.1 / context.current_price
```

---

## Error Handling

Handle edge cases gracefully in your strategy methods. Return `None` (for
entry/exit signals) or safe default values when data is missing or insufficient.

```python
def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
    # Handle insufficient history (warmup period)
    if context.current_index < self.lookback:
        return None

    # Handle missing indicator data
    sma_fast = context.get_indicator_value('sma_fast')
    sma_slow = context.get_indicator_value('sma_slow')
    if sma_fast is None or sma_slow is None:
        return None

    # Handle NaN values
    if pd.isna(sma_fast) or pd.isna(sma_slow):
        return None

    # Handle zero/invalid ATR
    atr = context.get_indicator_value('atr_14')
    if atr is None or atr <= 0:
        return None

    # Handle missing previous bar
    previous = context.previous_bar
    if previous is None:
        return None

    # Normal entry logic continues...
    if previous['sma_fast'] <= previous['sma_slow'] and sma_fast > sma_slow:
        return Signal.buy(
            size=1.0,
            stop_loss=self.calculate_initial_stop_loss(context),
            reason="SMA crossover",
            direction=self.trade_direction,
        )

    return None


def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
    # Always provide a fallback stop loss
    atr = context.get_indicator_value('atr_14')
    if atr is not None and atr > 0:
        return context.current_price - (atr * self.atr_multiplier)
    # Fallback: percentage-based stop
    return context.current_price * 0.95


def position_size(self, context: StrategyContext, signal: Signal) -> float:
    # Guard against invalid stop loss
    if signal.stop_loss is None:
        capital_to_use = context.available_capital * 0.1
        return capital_to_use / context.current_price

    stop_distance = context.current_price - signal.stop_loss
    if stop_distance <= 0:
        capital_to_use = context.available_capital * 0.1
        return capital_to_use / context.current_price

    # Risk-based sizing
    risk_amount = context.total_equity * (self.risk_percent / 100)
    stop_distance_base = stop_distance * context.fx_rate
    return risk_amount / stop_distance_base
```

---

## File Organization

For simple strategies, keep everything in one file:

```
strategies/
+-- my_strategy.py               # Single-file strategy
+-- random_control_strategy.py   # Baseline strategy
+-- base_alphatrend_strategy.py  # Production strategy
```

For complex strategies, split into modules:

```
strategies/
+-- my_strategy/
|   +-- __init__.py              # Export main strategy class
|   +-- strategy.py              # Main strategy class (extends BaseStrategy)
|   +-- indicators.py            # Custom indicator calculations
|   +-- entry_rules.py           # Entry condition helpers
|   +-- exit_rules.py            # Exit condition helpers
|   +-- position_sizing.py       # Sizing logic helpers
```

### What Does NOT Exist in the API

Avoid these common mistakes when writing strategies:

| Incorrect                          | Correct Alternative                              |
|------------------------------------|--------------------------------------------------|
| `get_name()` as static method     | Instance method `get_name()` (returns class name)|
| `get_parameter_info()`            | Does not exist; document params in docstring     |
| `on_position_opened()`            | Does not exist; no lifecycle hooks               |
| `on_position_closed()`            | Does not exist; no lifecycle hooks               |
| `reset()`                         | Does not exist; no reset mechanism               |
| Override `generate_signal()`      | Implement individual methods instead             |
| Override `prepare_data()`         | Override `_prepare_data_impl()` instead          |
| `Signal(SignalType.BUY, ...)`     | `Signal.buy(...)` factory method                 |
| `Signal(SignalType.HOLD)`         | `Signal.hold()`                                  |
| `context.historical_data`        | `context.data`                                   |
| `context.get_previous_bar(n)`    | `context.get_bar(-n)`                            |
| `context.get_bars(n)`            | `context.data.iloc[start:end]`                   |
| `context.get_indicator(name)`    | `context.get_indicator_value(column)`            |
| `signal.signal_type`             | `signal.type`                                    |
| `Signal.partial_exit(quantity=)` | `Signal.partial_exit(fraction=)`                 |

---

## Related Documentation

- [Strategy Guide](STRATEGY_GUIDE.md) -- Getting started
- [Signals and Exits](SIGNALS_AND_EXITS.md) -- Signal types
- [Testing Your Strategy](TESTING_YOUR_STRATEGY.md) -- Validation
