# Strategy Structure

This document provides a detailed breakdown of strategy class architecture and lifecycle.

---

## Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASS HIERARCHY                                     │
│                                                                             │
│                        BaseStrategy                                         │
│                        (Abstract Base)                                      │
│                             │                                               │
│              ┌──────────────┼──────────────┐                               │
│              │              │              │                               │
│              ▼              ▼              ▼                               │
│     AlphaTrendStrategy  RandomStrategy  YourStrategy                       │
│     (Production)        (Baseline)      (Custom)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

All strategies inherit from `BaseStrategy` located at `Classes/Strategy/base_strategy.py`.

---

## Complete Strategy Template

```python
from typing import Optional
import pandas as pd

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal, SignalType
from Classes.Models.position import Position


class YourStrategy(BaseStrategy):
    """
    Your strategy description.

    This strategy enters when [conditions] and exits when [conditions].
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        param1: float = 10.0,
        param2: int = 14,
        param3: bool = True
    ):
        """
        Initialize strategy with parameters.

        Args:
            param1: Description of param1
            param2: Description of param2
            param3: Description of param3
        """
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

        # Internal state (reset each backtest)
        self._bars_in_trade = 0

    # =========================================================================
    # REQUIRED STATIC METHODS
    # =========================================================================

    @staticmethod
    def get_name() -> str:
        """Return strategy name for display."""
        return "YourStrategy"

    @staticmethod
    def required_columns() -> list[str]:
        """Return list of required data columns."""
        return [
            'date',
            'open', 'high', 'low', 'close',
            'volume',
            'atr_14',  # If you need ATR
        ]

    @staticmethod
    def get_parameter_info() -> dict:
        """Return parameter definitions for GUI and optimization."""
        return {
            'param1': {
                'type': float,
                'default': 10.0,
                'min': 1.0,
                'max': 50.0,
                'step': 0.5,
                'description': 'What param1 controls'
            },
            'param2': {
                'type': int,
                'default': 14,
                'min': 5,
                'max': 50,
                'step': 1,
                'description': 'What param2 controls'
            },
            'param3': {
                'type': bool,
                'default': True,
                'description': 'What param3 enables/disables'
            }
        }

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate indicators before backtest loop.

        This runs once before the backtest starts. Use vectorized
        operations for efficiency.

        Args:
            df: Raw OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        # Moving averages
        df['sma_fast'] = df['close'].rolling(window=10).mean()
        df['sma_slow'] = df['close'].rolling(window=self.param2).mean()

        # Custom indicator
        df['my_indicator'] = self._calculate_custom_indicator(df)

        return df

    def _calculate_custom_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Helper method for complex calculations."""
        # Your calculation logic
        return df['close'].pct_change(5) * 100

    # =========================================================================
    # SIGNAL GENERATION (Core Logic)
    # =========================================================================

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal for current bar.

        This is called for every bar in the backtest. It should:
        1. Check current state (position or no position)
        2. Evaluate entry/exit conditions
        3. Return appropriate Signal

        Args:
            context: Current market state and position info

        Returns:
            Signal indicating desired action
        """
        current = context.current_bar

        # No position - check for entry
        if context.position is None:
            return self._check_entry(context)

        # Has position - check for exit or adjustment
        else:
            return self._check_exit(context)

    def _check_entry(self, context: StrategyContext) -> Signal:
        """Check entry conditions when not in a position."""
        current = context.current_bar
        previous = context.get_previous_bar(1)

        if previous is None:
            return Signal(SignalType.HOLD)

        # Entry condition example: fast SMA crosses above slow SMA
        cross_above = (
            previous['sma_fast'] <= previous['sma_slow'] and
            current['sma_fast'] > current['sma_slow']
        )

        # Volume filter (if enabled)
        volume_ok = True
        if self.param3:
            avg_volume = context.get_bars(20)['volume'].mean()
            volume_ok = current['volume'] > avg_volume

        if cross_above and volume_ok:
            # Calculate stop loss
            stop_loss = current['close'] - (current['atr_14'] * self.param1)

            return Signal(
                SignalType.BUY,
                price=current['close'],
                stop_loss=stop_loss
            )

        return Signal(SignalType.HOLD)

    def _check_exit(self, context: StrategyContext) -> Signal:
        """Check exit conditions when in a position."""
        current = context.current_bar
        position = context.position

        # Track bars in trade
        self._bars_in_trade += 1

        # Exit condition 1: Fast SMA crosses below slow SMA
        previous = context.get_previous_bar(1)
        if previous:
            cross_below = (
                previous['sma_fast'] >= previous['sma_slow'] and
                current['sma_fast'] < current['sma_slow']
            )
            if cross_below:
                self._bars_in_trade = 0
                return Signal(SignalType.SELL, reason='signal_exit')

        # Exit condition 2: Time-based (max 20 bars)
        if self._bars_in_trade >= 20:
            self._bars_in_trade = 0
            return Signal(SignalType.SELL, reason='time_exit')

        # Trailing stop adjustment
        new_stop = self._calculate_trailing_stop(context)
        if new_stop > position.stop_loss:
            return Signal(SignalType.ADJUST_STOP, stop_loss=new_stop)

        return Signal(SignalType.HOLD)

    def _calculate_trailing_stop(self, context: StrategyContext) -> float:
        """Calculate trailing stop level."""
        current = context.current_bar
        return current['close'] - (current['atr_14'] * self.param1)

    # =========================================================================
    # OPTIONAL: Position Sizing
    # =========================================================================

    def calculate_position_size(self, context: StrategyContext) -> float:
        """
        Custom position sizing logic.

        Default behavior uses config settings. Override for custom logic.

        Args:
            context: Current market state

        Returns:
            Number of shares to buy
        """
        capital = context.available_capital
        price = context.current_bar['close']
        atr = context.current_bar['atr_14']

        # Risk 2% of capital
        risk_amount = capital * 0.02

        # Stop distance based on ATR
        stop_distance = atr * self.param1

        # Calculate shares
        if stop_distance > 0:
            shares = risk_amount / stop_distance
        else:
            shares = (capital * 0.10) / price  # Fallback: 10% allocation

        return int(shares)

    # =========================================================================
    # OPTIONAL: Lifecycle Hooks
    # =========================================================================

    def on_position_opened(self, position: Position) -> None:
        """Called after a position is opened."""
        self._bars_in_trade = 0
        # Additional initialization for this trade

    def on_position_closed(self, trade) -> None:
        """Called after a position is closed."""
        self._bars_in_trade = 0
        # Cleanup or logging

    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        super().reset()
        self._bars_in_trade = 0
```

---

## Strategy Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STRATEGY LIFECYCLE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    INITIALIZATION
    │
    ├── __init__(params)
    │   └── Store parameters, initialize state
    │
    ▼
    BACKTEST START
    │
    ├── reset()
    │   └── Clear any state from previous runs
    │
    ├── prepare_data(df)
    │   └── Pre-calculate all indicators
    │
    ▼
    BAR LOOP (for each bar in data)
    │
    ├── generate_signal(context)
    │   ├── If no position → check entry
    │   └── If has position → check exit
    │
    ├── [If BUY signal executed]
    │   └── on_position_opened(position)
    │
    ├── [If SELL signal executed]
    │   └── on_position_closed(trade)
    │
    └── Continue to next bar...
    │
    ▼
    BACKTEST END
    │
    └── Results returned to caller
```

---

## State Management

### Strategy State vs Context State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STATE MANAGEMENT                                        │
│                                                                             │
│   STRATEGY STATE (self.*)           CONTEXT STATE (context.*)              │
│   ─────────────────────             ────────────────────────               │
│   • Parameters                      • current_bar (read-only)              │
│   • Internal counters               • position (read-only)                 │
│   • Flags                           • historical_data (read-only)          │
│   • Cached calculations             • available_capital                    │
│                                                                             │
│   Persists across bars              Fresh each bar                         │
│   Reset between backtests           Provided by engine                     │
│   You control                       Engine controls                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stateful vs Stateless

```python
# STATEFUL: Tracks information across bars
class StatefulStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.bars_since_signal = 0
        self.last_signal_type = None

    def generate_signal(self, context):
        self.bars_since_signal += 1
        # Use self.bars_since_signal in logic

    def reset(self):
        super().reset()
        self.bars_since_signal = 0
        self.last_signal_type = None


# STATELESS: Only uses current context
class StatelessStrategy(BaseStrategy):
    def generate_signal(self, context):
        # Only uses context data, no internal state
        if context.current_bar['close'] > context.current_bar['sma']:
            return Signal(SignalType.BUY, ...)
```

---

## Error Handling

Handle edge cases gracefully:

```python
def generate_signal(self, context) -> Signal:
    current = context.current_bar

    # Handle missing data
    if pd.isna(current.get('sma_fast')) or pd.isna(current.get('sma_slow')):
        return Signal(SignalType.HOLD)

    # Handle insufficient history
    if len(context.historical_data) < self.param2:
        return Signal(SignalType.HOLD)

    # Handle zero values
    if current.get('atr_14', 0) <= 0:
        return Signal(SignalType.HOLD)

    # Normal logic continues...
```

---

## File Organization

For complex strategies, split into modules:

```
strategies/
├── my_strategy/
│   ├── __init__.py           # Export main class
│   ├── strategy.py           # Main strategy class
│   ├── indicators.py         # Custom indicator calculations
│   ├── entry_rules.py        # Entry logic
│   ├── exit_rules.py         # Exit logic
│   └── position_sizing.py    # Sizing logic
│
└── my_strategy.py            # Or keep simple strategies in one file
```

---

## Related Documentation

- [Strategy Guide](STRATEGY_GUIDE.md) — Getting started
- [Signals and Exits](SIGNALS_AND_EXITS.md) — Signal types
- [Testing Your Strategy](TESTING_YOUR_STRATEGY.md) — Validation
