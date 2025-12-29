# API Reference

Complete reference for the core classes and methods in the Backtesting Framework.

## Engine Classes

### SingleSecurityEngine

Executes backtests on a single security.

```python
from Classes.Engine.single_security_engine import SingleSecurityEngine
```

#### Constructor

```python
SingleSecurityEngine(
    config: BacktestConfig,
    currency_converter: CurrencyConverter = None,
    security_registry: SecurityRegistry = None
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | BacktestConfig | Backtest configuration |
| `currency_converter` | CurrencyConverter | Optional currency conversion |
| `security_registry` | SecurityRegistry | Optional security metadata |

#### Methods

**run()**
```python
def run(
    symbol: str,
    data: pd.DataFrame,
    strategy: BaseStrategy
) -> BacktestResult
```
Run backtest on a single security.

---

### PortfolioEngine

Executes backtests across multiple securities with shared capital.

```python
from Classes.Engine.portfolio_engine import PortfolioEngine
```

#### Constructor

```python
PortfolioEngine(
    config: PortfolioConfig,
    currency_converter: CurrencyConverter = None,
    security_registry: SecurityRegistry = None
)
```

#### Methods

**run()**
```python
def run(
    data_dict: Dict[str, pd.DataFrame],
    strategy: BaseStrategy
) -> PortfolioResult
```
Run backtest on multiple securities.

---

## Data Classes

### DataLoader

Loads and normalizes CSV data files.

```python
from Classes.Data.data_loader import DataLoader
```

#### Constructor

```python
DataLoader(data_directory: Path)
```

#### Methods

**load_csv()**
```python
def load_csv(symbol: str) -> pd.DataFrame
```
Load CSV file for a symbol.

**list_available()**
```python
def list_available() -> List[str]
```
List all available symbols.

---

### CurrencyConverter

Handles multi-currency P/L calculations.

```python
from Classes.Data.currency_converter import CurrencyConverter
```

#### Constructor

```python
CurrencyConverter(base_currency: str = 'GBP')
```

#### Methods

**load_rates_directory()**
```python
def load_rates_directory(directory: Path)
```
Load all rate files from a directory.

**convert()**
```python
def convert(
    amount: float,
    from_currency: str,
    to_currency: str,
    date: datetime
) -> float
```
Convert amount between currencies on a specific date.

---

### SecurityRegistry

Manages security metadata.

```python
from Classes.Data.security_registry import SecurityRegistry
```

#### Constructor

```python
SecurityRegistry(metadata_path: Path)
```

#### Methods

**get_currency()**
```python
def get_currency(symbol: str) -> str
```
Get the currency for a symbol.

**get_sector()**
```python
def get_sector(symbol: str) -> str
```
Get the sector for a symbol.

---

## Strategy Classes

### BaseStrategy

Abstract base class for all strategies.

```python
from Classes.Strategy.base_strategy import BaseStrategy
```

#### Abstract Methods

**required_columns()**
```python
@abstractmethod
def required_columns(self) -> List[str]
```
Return list of required data columns.

**generate_signal()**
```python
@abstractmethod
def generate_signal(self, context: StrategyContext) -> Signal
```
Generate trading signal for current bar.

#### Optional Override Methods

**prepare_data()**
```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame
```
Pre-calculate indicators. Called once before backtest.

**position_size()**
```python
def position_size(self, context: StrategyContext, signal: Signal) -> float
```
Calculate position size. Returns fraction of capital.

**should_adjust_stop()**
```python
def should_adjust_stop(self, context: StrategyContext) -> Optional[float]
```
Return new stop loss level, or None.

**should_partial_exit()**
```python
def should_partial_exit(self, context: StrategyContext) -> Optional[float]
```
Return fraction to exit, or None.

#### Utility Methods

**get_name()**
```python
def get_name(self) -> str
```
Get strategy class name.

**get_parameters()**
```python
def get_parameters(self) -> Dict[str, Any]
```
Get current parameter values.

---

### StrategyContext

Provides data to strategies for signal generation.

```python
from Classes.Strategy.strategy_context import StrategyContext
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `current_date` | datetime | Current bar date |
| `current_price` | float | Current closing price |
| `current_index` | int | Bar index (0, 1, 2, ...) |
| `has_position` | bool | Whether position is open |
| `position` | Position | Current position (or None) |
| `available_capital` | float | Available cash |
| `total_equity` | float | Cash + position value |
| `historical_data` | DataFrame | All data up to current bar |

#### Methods

**get_indicator_value()**
```python
def get_indicator_value(self, column: str) -> Optional[float]
```
Get indicator value at current bar.

**get_bar()**
```python
def get_bar(self, offset: int = 0) -> pd.Series
```
Get data bar at offset from current (negative = past).

**get_position_pl_pct()**
```python
def get_position_pl_pct(self) -> float
```
Get current position P/L as percentage.

**get_position_pl()**
```python
def get_position_pl(self) -> float
```
Get current position P/L in currency.

---

## Model Classes

### Signal

Trading signal with metadata.

```python
from Classes.Models.signal import Signal, SignalType
```

#### Signal Types

```python
class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    PARTIAL_EXIT = "partial_exit"
    ADJUST_STOP = "adjust_stop"
```

#### Factory Methods

**buy()**
```python
@classmethod
def buy(
    cls,
    size: float,
    stop_loss: float = None,
    take_profit: float = None,
    reason: str = None
) -> Signal
```

**sell()**
```python
@classmethod
def sell(cls, reason: str = None) -> Signal
```

**hold()**
```python
@classmethod
def hold(cls) -> Signal
```

**partial_exit()**
```python
@classmethod
def partial_exit(cls, fraction: float, reason: str = None) -> Signal
```

**adjust_stop()**
```python
@classmethod
def adjust_stop(cls, new_stop: float, reason: str = None) -> Signal
```

---

### Trade

Completed trade record.

```python
from Classes.Models.trade import Trade
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `trade_id` | str | Unique identifier |
| `symbol` | str | Security symbol |
| `entry_date` | datetime | Entry date |
| `entry_price` | float | Entry price |
| `exit_date` | datetime | Exit date |
| `exit_price` | float | Exit price |
| `quantity` | float | Shares/units |
| `pl` | float | Profit/loss |
| `pl_pct` | float | P/L as percentage |
| `duration_days` | int | Days held |
| `entry_reason` | str | Entry reason |
| `exit_reason` | str | Exit reason |
| `commission` | float | Commission paid |

---

### Position

Open position record.

```python
from Classes.Models.position import Position
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `symbol` | str | Security symbol |
| `entry_date` | datetime | Entry date |
| `entry_price` | float | Entry price |
| `quantity` | float | Shares/units |
| `stop_loss` | float | Stop loss price |
| `take_profit` | float | Take profit price |
| `entry_reason` | str | Entry reason |
| `partial_exits` | List | Partial exit records |
| `highest_price` | float | Highest price since entry |

---

### BacktestResult

Complete backtest output.

```python
from Classes.Models.backtest_result import BacktestResult
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `symbol` | str | Security symbol |
| `strategy_name` | str | Strategy name |
| `trades` | List[Trade] | All trades |
| `equity_curve` | DataFrame | Equity over time |
| `num_trades` | int | Number of trades |
| `total_return_pct` | float | Total return % |
| `win_rate` | float | Win rate (0-1) |
| `sharpe_ratio` | float | Sharpe ratio |
| `max_drawdown_pct` | float | Max drawdown % |
| `initial_capital` | float | Starting capital |
| `final_capital` | float | Ending capital |

---

## Analysis Classes

### PerformanceMetrics

Calculates performance metrics from backtest results.

```python
from Classes.Analysis.performance_metrics import PerformanceMetrics
```

#### Class Methods

**calculate_metrics()**
```python
@classmethod
def calculate_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.035
) -> Dict[str, float]
```
Calculate all metrics from a backtest result.

#### Available Metrics

- `total_return_pct` - Total return percentage
- `num_trades` - Number of trades
- `win_rate` - Percentage of winning trades
- `profit_factor` - Gross profit / gross loss
- `sharpe_ratio` - Risk-adjusted return
- `sortino_ratio` - Downside risk-adjusted return
- `calmar_ratio` - Return / max drawdown
- `max_drawdown_pct` - Maximum drawdown percentage
- `avg_trade_pct` - Average trade return
- `avg_winner_pct` - Average winning trade
- `avg_loser_pct` - Average losing trade
- `best_trade_pct` - Best single trade
- `worst_trade_pct` - Worst single trade
- `avg_duration_days` - Average trade duration
- `max_consecutive_wins` - Max winning streak
- `max_consecutive_losses` - Max losing streak

---

### TradeLogger

Saves trade logs to CSV files.

```python
from Classes.Analysis.trade_logger import TradeLogger
```

#### Constructor

```python
TradeLogger(output_directory: Path)
```

#### Methods

**log_trades()**
```python
def log_trades(
    symbol: str,
    strategy_name: str,
    trades: List[Trade],
    backtest_name: str = None
) -> Path
```
Save trades to CSV. Returns output path.

---

### ExcelReportGenerator

Generates comprehensive Excel reports.

```python
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
```

#### Constructor

```python
ExcelReportGenerator(
    output_directory: Path,
    initial_capital: float,
    risk_free_rate: float = 0.035
)
```

#### Methods

**generate_report()**
```python
def generate_report(
    result: BacktestResult,
    filename: str = None
) -> Path
```
Generate Excel report. Returns output path.

---

## Configuration Classes

### BacktestConfig

```python
from Classes.Config.config import BacktestConfig
```

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission: CommissionConfig = None
    start_date: datetime = None
    end_date: datetime = None
    base_currency: str = 'GBP'
```

---

### PortfolioConfig

```python
from Classes.Config.config import PortfolioConfig
```

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

---

### CommissionConfig

```python
from Classes.Config.config import CommissionConfig, CommissionMode
```

```python
class CommissionMode(Enum):
    PERCENTAGE = "percentage"
    FIXED = "fixed"

@dataclass
class CommissionConfig:
    mode: CommissionMode
    value: float
```

---

### CapitalContentionConfig

```python
from Classes.Config.config import CapitalContentionConfig, CapitalContentionMode
```

```python
class CapitalContentionMode(Enum):
    DEFAULT = "default"
    VULNERABILITY_SCORE = "vulnerability_score"

@dataclass
class CapitalContentionConfig:
    mode: CapitalContentionMode = CapitalContentionMode.DEFAULT
    swap_threshold: float = 50.0
```

---

## Usage Examples

### Basic Backtest

```python
from pathlib import Path
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Analysis.performance_metrics import PerformanceMetrics
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Configure
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.001)
)

# Load data
loader = DataLoader(Path('raw_data'))
data = loader.load_csv('AAPL')

# Create strategy
strategy = AlphaTrendStrategy()

# Run backtest
engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)

# Calculate metrics
metrics = PerformanceMetrics.calculate_metrics(result)

# Display results
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

### Portfolio Backtest

```python
from Classes.Config.config import PortfolioConfig
from Classes.Engine.portfolio_engine import PortfolioEngine

# Configure portfolio
config = PortfolioConfig(
    initial_capital=100000.0,
    max_positions=3,
    position_size_limit=0.3
)

# Load multiple securities
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data_dict = {s: loader.load_csv(s) for s in symbols}

# Run
engine = PortfolioEngine(config)
results = engine.run(data_dict, strategy)

# Display results
for symbol, result in results.symbol_results.items():
    print(f"{symbol}: {result.total_return_pct:.2f}%")
print(f"Portfolio: {results.portfolio_return_pct:.2f}%")
```

### Generate Report

```python
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
from Classes.Analysis.trade_logger import TradeLogger

# Log trades
logger = TradeLogger(Path('logs'))
logger.log_trades('AAPL', strategy.get_name(), result.trades)

# Generate Excel report
generator = ExcelReportGenerator(
    output_directory=Path('logs/reports'),
    initial_capital=100000.0
)
report_path = generator.generate_report(result)
print(f"Report saved: {report_path}")
```
