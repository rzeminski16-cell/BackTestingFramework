# Strategy Development Guide

This guide covers everything you need to create custom trading strategies for the BackTesting Framework.

---

## Overview

A trading strategy in this framework is a Python class that:
1. Receives market data bar-by-bar
2. Decides when to enter positions (BUY)
3. Decides when to exit positions (SELL)
4. Manages position parameters (stops, targets)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STRATEGY FLOW                                       │
│                                                                             │
│   Market Data                Strategy                    Engine             │
│   ──────────                 ────────                    ──────             │
│                                                                             │
│   Bar 1 ──────────────►  generate_signal()  ────►  HOLD                   │
│                          (no position)               (wait)                │
│                                                                             │
│   Bar 2 ──────────────►  generate_signal()  ────►  HOLD                   │
│                          (conditions not met)        (wait)                │
│                                                                             │
│   Bar 3 ──────────────►  generate_signal()  ────►  BUY                    │
│                          (entry conditions!)         (open position)       │
│                                                                             │
│   Bar 4 ──────────────►  generate_signal()  ────►  HOLD                   │
│                          (in position)               (maintain)            │
│                                                                             │
│   Bar 5 ──────────────►  generate_signal()  ────►  SELL                   │
│                          (exit conditions!)          (close position)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Strategy Location

Place your strategy files in the `strategies/` directory:

```
strategies/
├── alphatrend_strategy.py     # Included production strategy
├── random_base_strategy.py    # Random baseline for comparison
└── your_strategy.py           # Your custom strategy
```

### Minimal Strategy Example

```python
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Models.signal import Signal, SignalType

class SimpleMovingAverageStrategy(BaseStrategy):
    """Buy when price crosses above SMA, sell when crosses below."""

    def __init__(self, sma_period: int = 20):
        super().__init__()
        self.sma_period = sma_period

    @staticmethod
    def get_name() -> str:
        return "SimpleMovingAverageStrategy"

    @staticmethod
    def required_columns() -> list[str]:
        return ['date', 'open', 'high', 'low', 'close', 'volume']

    @staticmethod
    def get_parameter_info() -> dict:
        return {
            'sma_period': {
                'type': int,
                'default': 20,
                'min': 5,
                'max': 200,
                'description': 'SMA lookback period'
            }
        }

    def prepare_data(self, df):
        """Pre-calculate SMA indicator."""
        df['sma'] = df['close'].rolling(window=self.sma_period).mean()
        return df

    def generate_signal(self, context) -> Signal:
        """Generate trading signal for current bar."""
        current = context.current_bar
        previous = context.get_previous_bar(1)

        if previous is None:
            return Signal(SignalType.HOLD)

        # Entry: price crosses above SMA
        if context.position is None:
            if previous['close'] < previous['sma'] and current['close'] > current['sma']:
                return Signal(
                    SignalType.BUY,
                    price=current['close'],
                    stop_loss=current['close'] * 0.95  # 5% stop
                )

        # Exit: price crosses below SMA
        else:
            if previous['close'] > previous['sma'] and current['close'] < current['sma']:
                return Signal(SignalType.SELL)

        return Signal(SignalType.HOLD)
```

---

## Strategy Components

Every strategy must implement these elements:

### Required Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_name()` | Strategy identifier for display | `str` |
| `required_columns()` | Data columns needed | `list[str]` |
| `get_parameter_info()` | Parameter definitions for GUI/optimization | `dict` |
| `generate_signal(context)` | Core logic—decides BUY/SELL/HOLD | `Signal` |

### Optional Methods

| Method | Purpose | Default Behavior |
|--------|---------|------------------|
| `prepare_data(df)` | Pre-calculate indicators | Returns df unchanged |
| `calculate_position_size(context)` | Custom position sizing | Uses config defaults |
| `on_position_opened(position)` | Hook after entry | No action |
| `on_position_closed(trade)` | Hook after exit | No action |

---

## The Strategy Context

Your `generate_signal()` method receives a `StrategyContext` object with everything needed to make decisions:

```
StrategyContext
├── current_bar          Current bar data (dict)
│   ├── date
│   ├── open, high, low, close
│   ├── volume
│   └── [indicators...]
│
├── position             Current position or None
│   ├── entry_price
│   ├── entry_date
│   ├── quantity
│   ├── stop_loss
│   └── unrealized_pnl
│
├── historical_data      All previous bars (DataFrame)
│
├── parameters           Strategy parameters
│
└── Methods:
    ├── get_previous_bar(n)    Get bar n days ago
    ├── get_bars(n)            Get last n bars as DataFrame
    └── get_indicator(name)    Get indicator value
```

### Accessing Data

```python
def generate_signal(self, context) -> Signal:
    # Current bar
    close = context.current_bar['close']
    volume = context.current_bar['volume']

    # Previous bar
    prev = context.get_previous_bar(1)
    if prev:
        prev_close = prev['close']

    # Last 10 bars
    recent = context.get_bars(10)
    avg_volume = recent['volume'].mean()

    # Check position
    if context.position:
        entry_price = context.position.entry_price
        current_pnl = context.position.unrealized_pnl
```

---

## Signal Types

Your strategy returns `Signal` objects to communicate intentions:

### BUY Signal

Opens a new long position:

```python
Signal(
    SignalType.BUY,
    price=current['close'],           # Entry price
    stop_loss=current['close'] * 0.95 # Stop loss level
)
```

### SELL Signal

Closes the current position:

```python
Signal(
    SignalType.SELL,
    reason="exit_signal"  # Optional reason for logging
)
```

### HOLD Signal

No action—maintain current state:

```python
Signal(SignalType.HOLD)
```

### ADJUST_STOP Signal

Move stop loss to new level:

```python
Signal(
    SignalType.ADJUST_STOP,
    stop_loss=new_stop_price
)
```

### PARTIAL_EXIT Signal

Close part of the position:

```python
Signal(
    SignalType.PARTIAL_EXIT,
    quantity=50  # Shares to exit
)
```

---

## Parameter Definition

Define parameters for GUI display and optimization:

```python
@staticmethod
def get_parameter_info() -> dict:
    return {
        'atr_multiplier': {
            'type': float,
            'default': 2.0,
            'min': 0.5,
            'max': 5.0,
            'step': 0.1,
            'description': 'ATR multiplier for stop loss'
        },
        'lookback_period': {
            'type': int,
            'default': 14,
            'min': 5,
            'max': 50,
            'step': 1,
            'description': 'Indicator lookback period'
        },
        'use_volume_filter': {
            'type': bool,
            'default': True,
            'description': 'Require above-average volume'
        }
    }
```

This enables:
- GUI parameter controls
- Optimization range definitions
- Default value documentation

---

## Data Preparation

Use `prepare_data()` to pre-calculate indicators:

```python
def prepare_data(self, df):
    """Calculate all indicators before backtesting."""

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Momentum
    df['momentum'] = df['close'].pct_change(5)

    # Volatility
    df['atr'] = self._calculate_atr(df, 14)

    # Your custom indicators
    df['my_indicator'] = self._custom_calculation(df)

    return df
```

Benefits of `prepare_data()`:
- Calculations run once, not every bar
- Vectorized operations are faster
- Cleaner `generate_signal()` code

---

## Position Sizing

### Default Behavior

The engine calculates position size based on config:

```
Position Size = (Capital × Risk%) / Stop Distance
```

### Custom Sizing

Override for strategy-specific logic:

```python
def calculate_position_size(self, context) -> float:
    """Calculate shares to buy."""
    capital = context.available_capital
    price = context.current_bar['close']
    atr = context.current_bar['atr']

    # Risk 2% of capital, use 2x ATR for stop
    risk_amount = capital * 0.02
    stop_distance = atr * 2

    shares = risk_amount / stop_distance

    # Cap at 20% of capital
    max_shares = (capital * 0.20) / price

    return min(shares, max_shares)
```

---

## Included Strategies

### AlphaTrendStrategy

Production-ready trend-following strategy:

```
Location: strategies/alphatrend_strategy.py

Features:
├── Adaptive ATR-based trend following
├── Dynamic MFI thresholds
├── Volume confirmation filter
├── SMA-based exits with momentum protection
├── Time-based exits (max 10 days)
└── Risk-based position sizing (2%)

Required columns:
├── date, open, high, low, close, volume
├── atr_14, ema_50, mfi_14
```

### RandomControlStrategy

Baseline for comparison—entries are random:

```
Location: strategies/random_base_strategy.py

Purpose:
├── Validate that your strategy beats random
├── E-ratio should be approximately 1.0
└── Returns should be near zero (minus costs)
```

---

## Strategy Development Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT WORKFLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    1. DEFINE HYPOTHESIS
       │
       │  "Price crossing above 20-SMA with high volume
       │   indicates trend start"
       │
       ▼
    2. IMPLEMENT STRATEGY
       │
       │  Create strategy class with entry/exit logic
       │  Define parameters
       │
       ▼
    3. INITIAL BACKTEST
       │
       │  Test on single security
       │  Check basic metrics (profitable? Sharpe > 0?)
       │
       ▼
    4. VALIDATE EDGE
       │
       │  Run Edge Analysis (E-ratio > 1.0?)
       │  Compare to RandomControlStrategy
       │
       ▼
    5. OPTIMIZE PARAMETERS
       │
       │  Univariate optimization to understand each parameter
       │  Walk-forward optimization for robustness
       │
       ▼
    6. PORTFOLIO TEST
       │
       │  Test across multiple securities
       │  Check if edge persists with capital constraints
       │
       ▼
    7. FACTOR ANALYSIS
       │
       │  Identify what conditions produce best results
       │  Add filters to improve strategy
       │
       ▼
    8. ITERATE
       │
       │  Refine based on analysis
       │  Re-test to confirm improvements
       │
       ▼
    9. PRODUCTION READY
```

---

## Next Steps

- [Strategy Structure](STRATEGY_STRUCTURE.md) — Detailed anatomy of a strategy
- [Signals and Exits](SIGNALS_AND_EXITS.md) — Signal types and exit rules
- [Testing Your Strategy](TESTING_YOUR_STRATEGY.md) — Validation techniques

---

## Related Documentation

- [Backtesting Concepts](../concepts/BACKTESTING.md) — How strategies are executed
- [Optimization](../concepts/OPTIMIZATION.md) — Finding optimal parameters
- [Edge Analysis](../concepts/EDGE_ANALYSIS.md) — Validating entry quality
