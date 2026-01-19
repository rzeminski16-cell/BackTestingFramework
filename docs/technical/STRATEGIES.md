# Strategy Development Guide

This guide explains how to create custom trading strategies for the Backtesting Framework.

## How Strategies Work

A strategy is a set of rules that decide when to buy and sell. The framework calls your strategy once for each bar (day) of data, and your strategy returns a signal.

```
For each bar in data:
    1. Framework creates a context with current price, indicators, position status
    2. Strategy's generate_signal() is called with this context
    3. Strategy returns a Signal (BUY, SELL, HOLD, etc.)
    4. Framework executes the signal
```

---

## Creating a Strategy

### Minimum Required Code

```python
from typing import List
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal

class MyStrategy(BaseStrategy):
    def __init__(self, my_param: int = 50):
        super().__init__(my_param=my_param)
        self.my_param = my_param

    def required_columns(self) -> List[str]:
        """Tell the framework what data columns you need."""
        return ['date', 'close', f'sma_{self.my_param}']

    def generate_signal(self, context: StrategyContext) -> Signal:
        """Called every bar. Return a trading signal."""

        if not context.has_position:
            # Not in a trade - check for entry
            sma = context.get_indicator_value(f'sma_{self.my_param}')
            if context.current_price > sma:
                return Signal.buy(
                    size=0.2,  # Use 20% of capital
                    stop_loss=context.current_price * 0.95,
                    reason="Price above SMA"
                )
        else:
            # In a trade - check for exit
            sma = context.get_indicator_value(f'sma_{self.my_param}')
            if context.current_price < sma:
                return Signal.sell(reason="Price below SMA")

        return Signal.hold()
```

### Where to Put Your Strategy

1. Create a file in `strategies/` (e.g., `strategies/my_strategy.py`)
2. The GUI will automatically discover it
3. Or import it directly in your Python code

---

## The Strategy Context

The `context` parameter passed to `generate_signal()` provides access to:

### Current Bar Data

```python
context.current_price      # Current closing price
context.current_date       # Current date
context.current_index      # Bar index (0, 1, 2, ...)
```

### Indicator Values

```python
# Get any column value from your data
sma = context.get_indicator_value('sma_50')
atr = context.get_indicator_value('atr_14')
rsi = context.get_indicator_value('rsi_14')
```

### Position Information

```python
context.has_position       # True if you have an open position
context.position           # The Position object (if open)
context.available_capital  # How much cash you have
context.total_equity       # Cash + position value
```

### Historical Data

```python
# Get data from previous bars (only past data, never future)
historical = context.historical_data

# Last 10 closes
last_10_closes = historical['close'].tail(10)

# Get a specific past bar
prev_close = context.get_bar(-1)['close']  # Yesterday's close
```

### Position P/L

```python
if context.has_position:
    pl_pct = context.get_position_pl_pct()   # P/L as percentage
    pl_dollars = context.get_position_pl()    # P/L in currency
    entry_price = context.position.entry_price
```

---

## Signal Types

### BUY Signal

Opens a new position.

```python
Signal.buy(
    size=0.2,                                  # Position size (fraction of capital)
    stop_loss=context.current_price * 0.95,    # Stop loss price
    take_profit=context.current_price * 1.10,  # Take profit price (optional)
    reason="Entry signal triggered"            # For logging
)
```

### SELL Signal

Closes the current position.

```python
Signal.sell(reason="Exit signal triggered")
```

### HOLD Signal

Do nothing.

```python
Signal.hold()
```

### PARTIAL_EXIT Signal

Close part of your position.

```python
Signal.partial_exit(
    fraction=0.5,  # Close 50% of position
    reason="Taking partial profit"
)
```

### ADJUST_STOP Signal

Move your stop loss.

```python
Signal.adjust_stop(
    new_stop=context.current_price * 0.98,
    reason="Trailing stop"
)
```

---

## Advanced Features

### Custom Position Sizing

Override the `position_size()` method:

```python
def position_size(self, context: StrategyContext, signal: Signal) -> float:
    """Calculate position size based on risk."""
    if signal.stop_loss is None:
        return 0.1  # Default 10%

    # Risk-based sizing: risk 2% of equity
    risk_amount = context.total_equity * 0.02
    stop_distance = context.current_price - signal.stop_loss

    if stop_distance <= 0:
        return 0.1

    shares = risk_amount / stop_distance
    position_value = shares * context.current_price

    # Return as fraction of capital
    return min(position_value / context.total_equity, 0.95)
```

### Trailing Stop Loss

Override the `should_adjust_stop()` method:

```python
def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
    """Trail stop loss with EMA after 5% profit."""
    if not context.has_position:
        return None

    if context.get_position_pl_pct() >= 5.0:
        ema = context.get_indicator_value('ema_14')
        if ema and ema > context.position.stop_loss:
            return ema  # New stop loss

    return None
```

### Partial Profit Taking

Override the `should_partial_exit()` method:

```python
def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
    """Take 50% profit at 10% gain."""
    if not context.has_position:
        return None

    if context.get_position_pl_pct() >= 10.0:
        if not context.position.partial_exits:  # Haven't taken profit yet
            return 0.5  # Exit 50% of position

    return None
```

### Data Preparation (Performance Optimization)

Override `prepare_data()` to pre-calculate values once:

```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate indicators for faster backtesting."""
    df = data.copy()

    # Calculate custom indicators once
    df['my_indicator'] = df['close'].rolling(20).mean()
    df['signal_line'] = df['my_indicator'].ewm(span=9).mean()

    return df
```

This runs once before the backtest, not on every bar.

---

## Example: Simple Moving Average Crossover

```python
from typing import List, Optional
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
import pandas as pd

class MACrossover(BaseStrategy):
    """Buy when fast MA crosses above slow MA, sell when it crosses below."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(self.slow_period).mean()
        return df

    def generate_signal(self, context: StrategyContext) -> Signal:
        fast_ma = context.get_indicator_value('fast_ma')
        slow_ma = context.get_indicator_value('slow_ma')

        if fast_ma is None or slow_ma is None:
            return Signal.hold()

        if not context.has_position:
            if fast_ma > slow_ma:
                return Signal.buy(
                    size=0.9,
                    stop_loss=context.current_price * 0.95,
                    reason=f"Fast MA ({fast_ma:.2f}) > Slow MA ({slow_ma:.2f})"
                )
        else:
            if fast_ma < slow_ma:
                return Signal.sell(reason="Fast MA crossed below Slow MA")

        return Signal.hold()
```

---

## Example: RSI Mean Reversion

```python
from typing import List
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal

class RSIMeanReversion(BaseStrategy):
    """Buy when RSI is oversold, sell when overbought."""

    def __init__(self, oversold: int = 30, overbought: int = 70):
        super().__init__(oversold=oversold, overbought=overbought)
        self.oversold = oversold
        self.overbought = overbought

    def required_columns(self) -> List[str]:
        return ['date', 'close', 'rsi_14']

    def generate_signal(self, context: StrategyContext) -> Signal:
        rsi = context.get_indicator_value('rsi_14')

        if rsi is None:
            return Signal.hold()

        if not context.has_position:
            if rsi < self.oversold:
                return Signal.buy(
                    size=0.3,
                    stop_loss=context.current_price * 0.93,
                    take_profit=context.current_price * 1.05,
                    reason=f"RSI oversold ({rsi:.1f})"
                )
        else:
            if rsi > self.overbought:
                return Signal.sell(reason=f"RSI overbought ({rsi:.1f})")

        return Signal.hold()
```

---

## Example: Trend Following with Trailing Stop

```python
from typing import List, Optional
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
import pandas as pd

class TrendFollower(BaseStrategy):
    """
    Enter on breakout above recent high.
    Trail stop loss with ATR.
    """

    def __init__(self, lookback: int = 20, atr_multiplier: float = 2.0):
        super().__init__(lookback=lookback, atr_multiplier=atr_multiplier)
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier

    def required_columns(self) -> List[str]:
        return ['date', 'close', 'high', 'atr_14']

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['highest_high'] = df['high'].rolling(self.lookback).max()
        return df

    def generate_signal(self, context: StrategyContext) -> Signal:
        highest = context.get_indicator_value('highest_high')
        atr = context.get_indicator_value('atr_14')

        if highest is None or atr is None:
            return Signal.hold()

        if not context.has_position:
            if context.current_price > highest:
                stop = context.current_price - (atr * self.atr_multiplier)
                return Signal.buy(
                    size=0.5,
                    stop_loss=stop,
                    reason="Breakout above recent high"
                )

        return Signal.hold()

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """Trail stop with ATR."""
        if not context.has_position:
            return None

        atr = context.get_indicator_value('atr_14')
        if atr is None:
            return None

        new_stop = context.current_price - (atr * self.atr_multiplier)

        if new_stop > context.position.stop_loss:
            return new_stop

        return None
```

---

## Best Practices

### 1. Keep It Simple

- Start with simple rules
- Add complexity only if it improves results
- Complex strategies are more likely to be overfit

### 2. Use Pre-calculated Indicators

- Put indicator calculations in `prepare_data()`
- This runs once, not on every bar
- Much faster for large datasets

### 3. Always Return a Signal

- `generate_signal()` must return a Signal object
- Use `Signal.hold()` when no action is needed
- Never return `None`

### 4. Handle Missing Data

Check if indicator values exist before using them:

```python
sma = context.get_indicator_value('sma_50')
if sma is None:
    return Signal.hold()
```

### 5. Log Your Reasons

Use the `reason` parameter in signals for debugging:

```python
Signal.buy(
    size=0.5,
    stop_loss=price * 0.95,
    reason=f"RSI={rsi:.1f}, Price above SMA"
)
```

### 6. Test Incrementally

- Test with a simple version first
- Add features one at a time
- Verify each addition improves results

---

## Debugging Strategies

### Print Debug Info

```python
def generate_signal(self, context: StrategyContext) -> Signal:
    print(f"Date: {context.current_date}, Price: {context.current_price}")
    print(f"Position: {context.has_position}")

    # Your logic here
```

### Check Why No Trades

If your strategy generates no trades:

1. Print entry conditions to see why they're not met
2. Check if required indicators have valid values
3. Verify the data covers enough bars for your indicators
4. Try relaxing conditions temporarily

### Verify Signal Execution

Check trade logs to see:
- When signals were generated
- What reasons were recorded
- Entry and exit prices

---

## Performance Optimization

### Use Numba for Intensive Calculations

```python
from numba import njit

@njit(cache=True)
def calculate_indicator(close: np.ndarray, period: int) -> np.ndarray:
    # Optimized calculation
    result = np.empty_like(close)
    for i in range(period, len(close)):
        result[i] = some_calculation(close[i-period:i])
    return result
```

### Vectorize Where Possible

```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Vectorized operations are faster than loops
    df['signal'] = np.where(df['fast_ma'] > df['slow_ma'], 1, 0)
    return df
```

### Cache Expensive Calculations

```python
def __init__(self, ...):
    super().__init__(...)
    self._cache = {}

def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    cache_key = hash(tuple(data.index))
    if cache_key in self._cache:
        return self._cache[cache_key]

    df = self._expensive_calculation(data)
    self._cache[cache_key] = df
    return df
```

---

## Included Strategies

The framework includes two strategies for production use and baseline comparison.

### BaseAlphaTrendStrategy

A simplified AlphaTrend strategy with time-based exits.

**What It Does:**
1. **Enters** when AlphaTrend indicator signals uptrend (crosses above smoothed line)
2. **Exits** after holding for `max_hold_days` (time-based exit)
3. Uses ATR-based stop losses with risk-based position sizing

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_multiplier` | 2.0 | Stop loss = entry - (ATR14 × this) |
| `risk_percent` | 2.0 | % of equity to risk per trade |
| `max_hold_days` | 10 | Maximum days to hold position |
| `alpha_atr_multiplier` | 1.0 | ATR multiplier for AlphaTrend bands |
| `smoothing_length` | 3 | EMA period for smoothing |
| `percentile_period` | 100 | Lookback for MFI thresholds |

**Required Columns:**
- `date`, `open`, `high`, `low`, `close`, `volume`
- `atr_14_atr` - 14-period ATR
- `mfi_14_mfi` - 14-period Money Flow Index

**Usage:**

```python
from strategies.base_alphatrend_strategy import BaseAlphaTrendStrategy

strategy = BaseAlphaTrendStrategy(
    atr_multiplier=2.0,
    risk_percent=2.0,
    max_hold_days=10
)
```

### RandomControlStrategy

A random baseline strategy for comparison purposes.

**What It Does:**
1. **Enters** randomly with configurable probability (default 10%)
2. **Exits** randomly with configurable probability (default 10%)
3. Uses ATR-based stop losses with risk-based position sizing

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entry_probability` | 0.10 | Probability of entry each bar |
| `exit_probability` | 0.10 | Probability of exit each bar |
| `atr_multiplier` | 2.0 | Stop loss = entry - (ATR14 × this) |
| `risk_percent` | 2.0 | % of equity to risk per trade |
| `random_seed` | None | Random seed for reproducibility |

**Required Columns:**
- `date`, `close`
- `atr_14_atr` - 14-period ATR

**Usage:**

```python
from strategies.random_control_strategy import RandomControlStrategy

strategy = RandomControlStrategy(
    entry_probability=0.10,
    exit_probability=0.10,
    atr_multiplier=2.0,
    risk_percent=2.0
)
```
