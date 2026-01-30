# Alternative Backtesting System

The framework includes a second backtesting implementation in the `backtesting/` directory. This module provides a more streamlined, modular architecture for simpler use cases.

---

## Two Systems Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM COMPARISON                                        │
│                                                                             │
│   MAIN SYSTEM (Classes/)              ALTERNATIVE SYSTEM (backtesting/)    │
│   ────────────────────────            ──────────────────────────────────   │
│                                                                             │
│   Full-featured                       Streamlined                          │
│   ├── Portfolio mode                  ├── Simpler architecture             │
│   ├── Vulnerability scoring           ├── Easier to understand             │
│   ├── Walk-forward optimization       ├── Quick experiments                │
│   ├── Factor analysis integration     ├── Basic optimization               │
│   ├── Advanced exit rules             └── Example strategies               │
│   └── Production-ready                                                     │
│                                                                             │
│   Best for:                           Best for:                            │
│   • Production strategies             • Learning the framework             │
│   • Complex multi-asset testing       • Simple strategy tests              │
│   • Advanced analysis needs           • Quick prototyping                  │
│   • Full workflow integration         • Educational use                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## When to Use Each System

| Use Case | Main System | Alternative System |
|----------|-------------|-------------------|
| Learning backtesting basics | | Recommended |
| Quick strategy prototype | | Recommended |
| Single security test | Either | Either |
| Portfolio backtesting | Recommended | Limited |
| Walk-forward optimization | Recommended | |
| Factor analysis | Recommended | |
| Vulnerability scoring | Recommended | |
| Production deployment | Recommended | |

---

## Module Structure

```
backtesting/
├── __init__.py           # Package exports
│
├── engine.py             # Core backtest engine
├── portfolio.py          # Portfolio management
├── position.py           # Position tracking
│
├── loader.py             # Simplified data loading
│
├── metrics/
│   └── performance.py    # Basic performance metrics
│
├── optimization/
│   ├── optimizer.py      # Parameter optimization
│   └── report.py         # Optimization reports
│
├── strategies/
│   ├── base.py           # Base strategy class
│   └── examples/
│       ├── sma_crossover.py   # SMA crossover example
│       ├── rsi_strategy.py    # RSI mean reversion
│       └── bollinger.py       # Bollinger bands
│
└── gui/
    └── simple_gui.py     # Simplified GUI layer
```

---

## Key Differences

### Strategy Interface

The alternative system uses a slightly simpler strategy interface:

```python
# MAIN SYSTEM (Classes/Strategy/)
class MainStrategy(BaseStrategy):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def required_columns() -> list[str]: ...
    @staticmethod
    def get_parameter_info() -> dict: ...
    def prepare_data(self, df): ...
    def generate_signal(self, context) -> Signal: ...

# ALTERNATIVE SYSTEM (backtesting/strategies/)
class AltStrategy(BaseStrategy):
    def __init__(self, params): ...
    def generate_signals(self, data) -> list[Signal]: ...
```

### Execution Model

```
MAIN SYSTEM:
├── Bar-by-bar processing with context
├── Full position management
├── Capital contention handling
└── Vulnerability scoring support

ALTERNATIVE SYSTEM:
├── Vectorized signal generation
├── Simpler position tracking
├── Basic capital management
└── Faster for simple cases
```

### Configuration

```
MAIN SYSTEM:
├── BacktestConfig, PortfolioConfig classes
├── Commission modes (percentage, fixed)
├── Capital contention modes
└── Extensive configuration options

ALTERNATIVE SYSTEM:
├── Simple dict-based configuration
├── Basic commission support
├── Fewer options
└── Easier to get started
```

---

## Migration Path

If you start with the alternative system and need more features:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MIGRATION GUIDE                                         │
│                                                                             │
│   Alternative System                  Main System                          │
│   ──────────────────                  ───────────                          │
│                                                                             │
│   class MyStrategy:                   class MyStrategy(BaseStrategy):      │
│       def generate_signals(data):         @staticmethod                   │
│           signals = []                    def get_name(): return "..."    │
│           for i in range(len(data)):                                       │
│               if condition:               @staticmethod                   │
│                   signals.append(BUY)     def required_columns(): ...     │
│           return signals                                                   │
│                                           def generate_signal(context):   │
│                                               if condition:               │
│                                                   return Signal(BUY, ...) │
│                                               return Signal(HOLD)         │
│                                                                             │
│   Key Changes:                                                              │
│   1. Inherit from Classes/Strategy/base_strategy.py                        │
│   2. Add static methods (get_name, required_columns, get_parameter_info)   │
│   3. Change from batch signals to single-bar generate_signal               │
│   4. Use StrategyContext instead of raw DataFrame                          │
│   5. Move to strategies/ directory                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Availability

| Feature | Main System | Alternative |
|---------|-------------|-------------|
| Single security backtest | Yes | Yes |
| Portfolio backtest | Yes | Basic |
| Walk-forward optimization | Yes | No |
| Univariate optimization | Yes | Basic |
| E-ratio analysis | Yes | No |
| Factor analysis | Yes | No |
| Vulnerability scoring | Yes | No |
| GUI applications | Full suite | Basic |
| Excel reports | Comprehensive | Simple |
| Custom position sizing | Yes | Limited |
| Multiple exit types | Yes | Basic |

---

## Recommendation

**Start with the main system** (`Classes/`) unless:
- You're learning and want simpler code to study
- You need quick one-off experiments
- You're building a custom workflow from scratch

The main system is more complex but provides everything needed for serious strategy development and testing.

---

## Related Documentation

- [Usage Guide](USAGE.md) — How to use the alternative system
- [Main System Architecture](../overview/ARCHITECTURE.md) — Main system details
- [Strategy Development](../strategy-development/STRATEGY_GUIDE.md) — Creating strategies
