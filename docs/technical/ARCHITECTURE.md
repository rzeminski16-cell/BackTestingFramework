# Architecture Overview

This document provides a technical overview of the Backtesting Framework's architecture, components, and data flow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Points                             │
│  run_gui.py │ optimize_gui.py │ data_collection_gui.py │ CLI    │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                        Configuration                             │
│         BacktestConfig │ PortfolioConfig │ OptimizationConfig   │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                        Core Engines                              │
│            SingleSecurityEngine │ PortfolioEngine               │
│            PositionManager │ TradeExecutor                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                        Strategy Layer                            │
│         BaseStrategy │ StrategyContext │ Signal │ Order         │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
│    DataLoader │ SecurityRegistry │ CurrencyConverter │ Validator│
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                       Analysis Layer                             │
│   PerformanceMetrics │ TradeLogger │ ExcelReportGenerator       │
│   BacktestAnalyzer │ VulnerabilityScorer                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
BackTestingFramework/
├── Classes/                    # Core framework modules
│   ├── Config/                # Configuration dataclasses
│   │   └── config.py          # BacktestConfig, PortfolioConfig, etc.
│   │
│   ├── Data/                  # Data loading and validation
│   │   ├── data_loader.py     # CSV loading and normalization
│   │   ├── data_validator.py  # Data quality checks
│   │   ├── security_registry.py # Security metadata
│   │   └── currency_converter.py # Multi-currency support
│   │
│   ├── DataCollection/        # Alpha Vantage API integration
│   │   ├── api_client.py      # API request handling
│   │   ├── data_fetcher.py    # Data type fetchers
│   │   └── rate_limiter.py    # Rate limit management
│   │
│   ├── Engine/                # Backtesting engines
│   │   ├── single_security_engine.py  # Single security backtest
│   │   ├── portfolio_engine.py        # Multi-security backtest
│   │   ├── position_manager.py        # Position tracking
│   │   └── trade_executor.py          # Trade execution
│   │
│   ├── Strategy/              # Strategy base classes
│   │   ├── base_strategy.py   # Abstract strategy class
│   │   └── strategy_context.py # Context for signal generation
│   │
│   ├── Models/                # Core data models
│   │   ├── trade.py           # Trade dataclass
│   │   ├── position.py        # Position dataclass
│   │   ├── signal.py          # Signal enum and class
│   │   ├── order.py           # Order dataclass
│   │   └── backtest_result.py # Result container
│   │
│   ├── Optimization/          # Walk-forward optimization
│   │   ├── optimizer.py       # Core optimizer
│   │   ├── walk_forward.py    # Walk-forward logic
│   │   └── sensitivity.py     # Sensitivity analysis
│   │
│   ├── Analysis/              # Performance and reporting
│   │   ├── performance_metrics.py    # Metric calculations
│   │   ├── trade_logger.py           # Trade log CSV output
│   │   ├── excel_report_generator.py # Excel reports
│   │   └── backtest_analyzer/        # Deep-dive analysis
│   │
│   ├── Indicators/            # Indicator calculations
│   │   └── indicator_engine.py # Vectorized calculations
│   │
│   ├── GUI/                   # GUI components
│   │   ├── backtest_gui.py    # Main backtest GUI
│   │   └── optimization_gui.py # Optimization GUI
│   │
│   └── VulnerabilityScorer/   # Portfolio capital allocation
│       ├── scorer.py          # Score calculation
│       └── feature_calculator.py # Feature extraction
│
├── strategies/                # Trading strategies
│   └── alphatrend_strategy.py # Production AlphaTrend strategy
│
├── raw_data/                  # Market data storage
│   ├── daily/                 # Daily OHLCV + indicators
│   ├── weekly/                # Weekly data
│   ├── fundamentals/          # Financial statements
│   ├── options/               # Options chains
│   └── insider_transactions/  # Insider trading
│
├── logs/                      # Output storage
│   ├── backtests/             # Backtest results
│   └── optimization_reports/  # Optimization outputs
│
├── config/                    # Configuration files
│   ├── baskets/              # Security baskets
│   ├── strategy_presets/     # Parameter presets
│   └── vulnerability_presets/ # Vulnerability configs
│
└── tests/                     # Unit tests
```

---

## Core Components

### BacktestConfig

Configuration for single-security backtests.

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission: CommissionConfig = None
    start_date: datetime = None
    end_date: datetime = None
    base_currency: str = 'GBP'
```

### PortfolioConfig

Configuration for multi-security backtests.

```python
@dataclass
class PortfolioConfig:
    initial_capital: float = 100000.0
    max_positions: int = None
    position_size_limit: float = 0.3
    total_allocation_limit: float = 0.9
    capital_contention: CapitalContentionConfig = None
    commission: CommissionConfig = None
    base_currency: str = 'GBP'
```

### SingleSecurityEngine

Executes backtests on a single security.

**Key Methods:**
- `run(symbol, data, strategy)` - Run backtest
- `_process_bar()` - Process single data bar
- `_execute_signal()` - Execute trading signal

**Execution Flow:**
1. Load and prepare data
2. Initialize position manager
3. For each bar:
   - Create strategy context
   - Generate signal from strategy
   - Execute signal if actionable
   - Check stop loss / take profit
4. Close any remaining positions
5. Calculate metrics and return result

### PortfolioEngine

Executes backtests across multiple securities with shared capital.

**Key Methods:**
- `run(data_dict, strategy)` - Run portfolio backtest
- `_process_date()` - Process all securities for a date
- `_handle_capital_contention()` - Resolve capital conflicts

**Execution Flow:**
1. Align all securities to common date index
2. For each date:
   - Generate signals for all securities
   - Handle capital contention (if needed)
   - Execute signals with available capital
   - Update positions and equity
3. Close remaining positions
4. Aggregate results

### PositionManager

Tracks open and closed positions.

**Key Methods:**
- `open_position()` - Create new position
- `close_position()` - Close existing position
- `partial_exit()` - Close portion of position
- `update_stop_loss()` - Adjust stop level

### TradeExecutor

Executes trades and calculates P/L.

**Key Methods:**
- `execute_buy()` - Execute buy order
- `execute_sell()` - Execute sell order
- `calculate_pl()` - Calculate profit/loss

---

## Strategy Framework

### BaseStrategy

Abstract base class for all strategies.

```python
class BaseStrategy(ABC):
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Columns the strategy needs in the data."""
        pass

    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> Signal:
        """Generate a trading signal for the current bar."""
        pass

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optional: Pre-calculate indicators."""
        return data

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """Optional: Calculate position size."""
        return signal.size

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """Optional: Return new stop loss level."""
        return None
```

### StrategyContext

Provides data to the strategy for signal generation.

```python
@dataclass
class StrategyContext:
    current_date: datetime
    current_price: float
    current_index: int
    has_position: bool
    position: Optional[Position]
    available_capital: float
    total_equity: float
    historical_data: pd.DataFrame

    def get_indicator_value(self, column: str) -> Optional[float]:
        """Get indicator value at current bar."""

    def get_bar(self, offset: int = 0) -> pd.Series:
        """Get data bar at offset from current."""

    def get_position_pl_pct(self) -> float:
        """Get current position P/L percentage."""
```

### Signal

Trading signal types and creation.

```python
class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    PARTIAL_EXIT = "partial_exit"
    ADJUST_STOP = "adjust_stop"

@dataclass
class Signal:
    type: SignalType
    size: float = None
    stop_loss: float = None
    take_profit: float = None
    reason: str = None

    @classmethod
    def buy(cls, size, stop_loss, take_profit=None, reason=None) -> Signal

    @classmethod
    def sell(cls, reason=None) -> Signal

    @classmethod
    def hold(cls) -> Signal

    @classmethod
    def partial_exit(cls, fraction, reason=None) -> Signal

    @classmethod
    def adjust_stop(cls, new_stop, reason=None) -> Signal
```

---

## Data Models

### Trade

Completed trade record.

```python
@dataclass
class Trade:
    trade_id: str
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: float
    entry_value: float
    exit_value: float
    pl: float
    pl_pct: float
    duration_days: int
    entry_reason: str
    exit_reason: str
    commission: float
    currency: str = 'USD'
    fx_rate_entry: float = 1.0
    fx_rate_exit: float = 1.0
    pl_base_currency: float = None
```

### Position

Open position record.

```python
@dataclass
class Position:
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_reason: str
    partial_exits: List[PartialExit] = None
    highest_price: float = None  # For trailing stops
```

### BacktestResult

Complete backtest output.

```python
@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    trades: List[Trade]
    equity_curve: pd.DataFrame
    num_trades: int
    total_return_pct: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pct: float
    initial_capital: float
    final_capital: float
```

---

## Data Flow

### Backtest Execution Flow

```
1. Configuration
   BacktestConfig → Engine

2. Data Loading
   raw_data/*.csv → DataLoader → DataFrame

3. Data Preparation
   DataFrame → Strategy.prepare_data() → Prepared DataFrame

4. Bar-by-Bar Processing
   For each bar:
   ├─ Create StrategyContext
   ├─ Strategy.generate_signal(context) → Signal
   ├─ TradeExecutor.execute(signal) → Trade/Position
   ├─ PositionManager.update()
   └─ Update equity curve

5. Result Generation
   trades, equity_curve → BacktestResult
   BacktestResult → PerformanceMetrics
   BacktestResult → TradeLogger / ExcelReportGenerator
```

### Portfolio Execution Flow

```
1. Configuration
   PortfolioConfig → PortfolioEngine

2. Data Loading
   {symbol: DataFrame} → data_dict

3. Date Alignment
   data_dict → Unified date index

4. Daily Processing
   For each date:
   ├─ Generate signals for all securities
   ├─ Check capital availability
   ├─ Handle contention (DEFAULT or VULNERABILITY_SCORE)
   ├─ Execute signals
   ├─ Update positions
   └─ Update portfolio equity

5. Result Aggregation
   Per-symbol results + portfolio metrics → PortfolioResult
```

---

## Performance Optimizations

### Numba JIT Compilation

The AlphaTrend strategy uses Numba for 5-20x speedup:

```python
@njit(cache=True)
def calculate_alphatrend_jit(close, high, low, atr, mfi, period, multiplier):
    # Vectorized calculation with JIT compilation
    ...
```

### Vectorized Indicator Calculations

Indicators are pre-calculated in `prepare_data()` rather than per-bar:

```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Calculate once for entire dataset
    df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
    df['slow_ma'] = df['close'].rolling(self.slow_period).mean()
    return df
```

### Caching

- Currency conversion rates are cached
- API responses are cached (24-hour TTL by default)
- Strategy-calculated columns are stored in DataFrame

---

## Extension Points

### Custom Strategies

Create new strategies by extending `BaseStrategy`:

```python
class MyStrategy(BaseStrategy):
    def required_columns(self) -> List[str]:
        return ['date', 'close', 'rsi_14']

    def generate_signal(self, context: StrategyContext) -> Signal:
        # Your logic here
        return Signal.hold()
```

### Custom Indicators

Add indicators in `prepare_data()`:

```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df['my_indicator'] = your_calculation(df)
    return df
```

### Custom Analysis

Extend the analysis system:

```python
from Classes.Analysis.performance_metrics import PerformanceMetrics

# Add custom metrics
custom_metrics = PerformanceMetrics.calculate_metrics(result)
custom_metrics['my_metric'] = calculate_my_metric(result)
```

---

## Testing

### Unit Tests

Located in `tests/`:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_engine.py

# Run with coverage
python -m pytest --cov=Classes tests/
```

### Integration Tests

Test complete workflows:

```python
def test_full_backtest():
    config = BacktestConfig(initial_capital=100000.0)
    loader = DataLoader(Path('tests/fixtures'))
    data = loader.load_csv('TEST_SYMBOL')
    strategy = AlphaTrendStrategy()
    engine = SingleSecurityEngine(config)

    result = engine.run('TEST_SYMBOL', data, strategy)

    assert result.num_trades > 0
    assert result.total_return_pct is not None
```
