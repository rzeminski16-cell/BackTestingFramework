# Backtesting Framework

A professional-grade Python framework for backtesting trading strategies across multiple securities with detailed trade logging and optimization capabilities.

## Features

### Core Capabilities
- ✅ **Graphical User Interface**: Easy-to-use GUI for configuring and running backtests
- ✅ **Single Security Backtesting**: Test strategies on individual securities
- ✅ **Portfolio Backtesting**: Test strategies across multiple securities with shared capital
- ✅ **Parameter Optimization**: Grid search to find optimal strategy parameters
- ✅ **Detailed Trade Logging**: Complete trade logs for analysis and reporting
- ✅ **Performance Metrics**: Comprehensive performance analysis (Sharpe ratio, max drawdown, etc.)

### Advanced Strategy Features
- **Partial Position Exits**: Scale out of positions at different profit levels
- **Trailing Stop Losses**: Dynamic stops based on price or indicator values
- **Conditional Orders**: If-then logic (e.g., "move stop to breakeven after 5% profit")
- **Multi-condition Entry/Exit**: Complex signal logic with multiple indicators
- **Flexible Position Sizing**: Custom position sizing algorithms

### Execution Model
- Executes trades at **closing prices** (matches TradingView default behavior)
- No lookahead bias (strategies only see historical data)
- Configurable commission (percentage or fixed per trade)
- No slippage (for accuracy matching TradingView)
- LONG positions only

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- tkinter (for GUI - usually included with Python)

```bash
# Install dependencies
pip install pandas numpy

# If tkinter is not installed (Linux):
# sudo apt-get install python3-tk
```

## Quick Start

### Option 1: Using the GUI (Recommended for Beginners)

The easiest way to run backtests is using the graphical interface:

```bash
python run_gui.py
# or
python backtest_gui.py
```

The GUI allows you to:
- ✅ Select securities from a list (single or multiple)
- ✅ Choose between single security or portfolio mode
- ✅ Select and configure strategies with visual parameter editors
- ✅ Set commission (percentage or fixed)
- ✅ Configure date ranges
- ✅ Name your backtests (format: `{strategy_name}_{your_name}`)
- ✅ View results in real-time
- ✅ Automatically save trade logs

![GUI Screenshot](docs/gui_screenshot.png)

### Option 2: Using Python Code

For programmatic control or integration into scripts:

### 1. Prepare Your Data

Place CSV files in the `raw_data/` directory. Each file should be named by symbol (e.g., `AAPL.csv`) and contain:
- `time` or `date` column (will be auto-converted to `date`)
- `close` column (required)
- Any indicators your strategy needs (e.g., `sma_50`, `rsi_14`, `ema_20`)

Column names are case-insensitive and will be normalized to lowercase.

### 2. Create a Strategy

```python
from typing import List
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal

class MyStrategy(BaseStrategy):
    def __init__(self, ma_period: int = 50):
        super().__init__(ma_period=ma_period)
        self.ma_period = ma_period

    def required_columns(self) -> List[str]:
        return ['date', 'close', f'sma_{self.ma_period}']

    def generate_signal(self, context: StrategyContext) -> Signal:
        if not context.has_position:
            # Entry: Price crosses above MA
            if context.current_price > context.get_indicator_value(f'sma_{self.ma_period}'):
                return Signal.buy(
                    size=0.2,  # 20% of capital
                    stop_loss=context.current_price * 0.95,  # 5% stop
                    reason="Price above SMA"
                )
        else:
            # Exit: Price crosses below MA
            if context.current_price < context.get_indicator_value(f'sma_{self.ma_period}'):
                return Signal.sell(reason="Price below SMA")

        return Signal.hold()
```

### 3. Run a Backtest

```python
from pathlib import Path
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Analysis.performance_metrics import PerformanceMetrics

# Configure
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001  # 0.1%
    )
)

# Load data
data_loader = DataLoader(Path('raw_data'))
data = data_loader.load_csv('AAPL')

# Run backtest
strategy = MyStrategy(ma_period=50)
engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)

# Display metrics
metrics = PerformanceMetrics.calculate_metrics(result)
PerformanceMetrics.print_metrics(metrics)
```

## Usage Examples

### Single Security Backtest

```python
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Engine.single_security_engine import SingleSecurityEngine
from strategies.examples import PartialExitStrategy

# Fixed commission per trade
commission = CommissionConfig(mode=CommissionMode.FIXED, value=3.0)
config = BacktestConfig(initial_capital=100000.0, commission=commission)

# Load data and run
data_loader = DataLoader(Path('raw_data'))
data = data_loader.load_csv('AAPL')
strategy = PartialExitStrategy(rsi_period=14, position_size=0.2, first_target_pct=0.10, second_target_pct=0.20)

engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)
```

### Portfolio Backtest

```python
from Classes.Config.config import PortfolioConfig
from Classes.Engine.portfolio_engine import PortfolioEngine

# Configure portfolio constraints
config = PortfolioConfig(
    initial_capital=100000.0,
    max_positions=3,  # Max 3 concurrent positions
    position_size_limit=0.3,  # Max 30% per position
    total_allocation_limit=0.9  # Max 90% total
)

# Load multiple securities
symbols = ['AAPL', 'MSFT', 'JPM']
data_dict = {}
for symbol in symbols:
    data_dict[symbol] = data_loader.load_csv(symbol)

# Run portfolio backtest
strategy = PartialExitStrategy()
engine = PortfolioEngine(config)
results = engine.run(data_dict, strategy)

# Results is a dict mapping symbol -> BacktestResult
for symbol, result in results.items():
    print(f"{symbol}: {result.num_trades} trades, P/L: ${result.total_return:.2f}")
```

### Parameter Optimization

```python
from Classes.Optimization.optimizer import StrategyOptimizer
from Classes.Config.config import OptimizationConfig

# Configure optimization
opt_config = OptimizationConfig(
    metric='total_return_pct',
    per_security=False,  # Find global best
    min_trades=5,
    maximize=True
)

# Define parameter grid
param_grid = {
    'ma_period': [20, 50, 100],
    'position_size': [0.1, 0.2, 0.3],
    'stop_loss_pct': [0.03, 0.05, 0.08]
}

# Run optimization
optimizer = StrategyOptimizer(opt_config, backtest_config)
results = optimizer.optimize(PartialExitStrategy, param_grid, 'AAPL', data)

# Get best parameters
best = results.get_best()
print(f"Best parameters: {best.parameters}")
print(f"Return: {best.backtest_result.total_return_pct:.2f}%")
```

## Advanced Strategy Features

### Partial Exits (Scaling Out)

```python
class MyStrategy(BaseStrategy):
    def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
        """Take 50% profit at 10% gain"""
        if context.get_position_pl_pct() >= 10.0:
            return 0.5  # Exit 50% of position
        return None
```

### Trailing Stops

```python
class MyStrategy(BaseStrategy):
    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """Trail stop with EMA after 5% profit"""
        if context.get_position_pl_pct() >= 5.0:
            ema = context.get_indicator_value('ema_14')
            if ema and ema > context.position.stop_loss:
                return ema  # Move stop up to EMA
        return None
```

### Conditional Orders

```python
class MyStrategy(BaseStrategy):
    def generate_signal(self, context: StrategyContext) -> Signal:
        # Entry: Multi-condition
        if not context.has_position:
            price = context.current_price
            sma = context.get_indicator_value('sma_200')
            rsi = context.get_indicator_value('rsi_14')

            if price > sma and rsi < 70:  # Uptrend + not overbought
                return Signal.buy(size=0.2, stop_loss=price*0.95)

        return Signal.hold()
```

## Configuration

### Commission Settings

```python
# Percentage-based commission
commission = CommissionConfig(
    mode=CommissionMode.PERCENTAGE,
    value=0.001  # 0.1% of trade value
)

# Fixed commission per trade
commission = CommissionConfig(
    mode=CommissionMode.FIXED,
    value=3.0  # £3 per trade
)
```

### Date Range Filtering

```python
from datetime import datetime

config = BacktestConfig(
    initial_capital=100000.0,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

## Output and Logging

### Trade Logs

Trades are automatically logged to CSV files:

```python
from Classes.Analysis.trade_logger import TradeLogger

logger = TradeLogger(Path('logs'))
logger.log_trades('AAPL', strategy.get_name(), result.trades)
```

CSV format:
```
symbol,entry_date,entry_price,exit_date,exit_price,quantity,pl,pl_pct,duration_days,exit_reason,...
```

### Performance Metrics

```python
from Classes.Analysis.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics.calculate_metrics(result)
# Returns dict with:
# - total_return, total_return_pct
# - num_trades, win_rate
# - avg_win, avg_loss, profit_factor
# - sharpe_ratio, max_drawdown
# - etc.
```

## Project Structure

```
BackTestingFramework/
├── Classes/
│   ├── Config/          # Configuration classes
│   ├── Models/          # Data models (Signal, Trade, Position, Order)
│   ├── Data/            # Data loading and validation
│   ├── Strategy/        # Strategy base classes
│   ├── Engine/          # Backtesting engines
│   ├── Optimization/    # Parameter optimization
│   └── Analysis/        # Performance metrics and logging
├── strategies/
│   └── examples/        # Example strategies
├── raw_data/            # CSV data files
├── logs/                # Trade logs output
├── config/              # Configuration files
├── backtest_gui.py      # GUI application
├── run_gui.py           # GUI launcher script
├── backtest.py          # Command-line examples
└── ARCHITECTURE.md      # Detailed architecture documentation
```

## Example Strategies

The framework includes two example strategies:

1. **AdvancedTrailingStopStrategy**: Multi-condition entry with trailing stops
2. **PartialExitStrategy**: Demonstrates scaling out at profit targets

See `strategies/examples/` for full implementations.

## Running Examples

### Using the GUI

```bash
# Launch the GUI
python run_gui.py
```

**GUI Features:**
1. **Security Selection**: Choose one or more securities from your `raw_data/` folder
2. **Mode Selection**: Toggle between single security and portfolio mode
3. **Strategy Configuration**:
   - Select from available strategies
   - Click "Configure Strategy Parameters" to customize
4. **Commission Settings**: Choose percentage or fixed commission
5. **Portfolio Settings**: (Portfolio mode only)
   - Max concurrent positions
   - Position size limit per security
   - Total allocation limit
6. **Backtest Naming**: Name follows format `{strategy_name}_{your_input}`
7. **Real-time Results**: View performance metrics as backtest runs
8. **Auto-save Logs**: Trade logs saved to `logs/{backtest_name}/`

### Using Command Line

```bash
# Run all examples
python backtest.py

# Or import and run individual examples
from backtest import example_single_security_backtest
example_single_security_backtest()
```

## Creating Custom Strategies

### Minimum Requirements

```python
class MyStrategy(BaseStrategy):
    def required_columns(self) -> List[str]:
        """List of required columns from CSV"""
        return ['date', 'close', 'sma_50']

    def generate_signal(self, context: StrategyContext) -> Signal:
        """Generate trading signal"""
        # Your logic here
        return Signal.hold()
```

### Optional Overrides

- `position_size()`: Custom position sizing
- `should_check_stop_loss()`: When to check stop loss
- `should_check_take_profit()`: When to check take profit
- `should_adjust_stop()`: Trailing stop logic
- `should_partial_exit()`: Partial exit logic

## Accuracy and TradingView Compatibility

This framework is designed to closely match TradingView Pine Script results:

- ✅ Executes at close prices (same as Pine Script default)
- ✅ No lookahead bias (point-in-time data only)
- ✅ Bar-by-bar execution model
- ✅ Configurable commission (match your broker)
- ✅ No slippage by default

For best accuracy:
1. Export data from TradingView with same timeframe
2. Use same commission settings as your Pine Script
3. Verify indicator calculations match

## Performance Metrics

Calculated metrics include:
- **Total Return**: Dollar and percentage returns
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Total wins / total losses
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Mean profit/loss per trade
- **Trade Duration**: Average holding period

## Optimization

The optimizer supports:
- Grid search over parameter combinations
- Global optimization (best params for all securities)
- Per-security optimization (best params per security)
- Customizable metrics (return, Sharpe, profit factor, etc.)

```python
# Per-security optimization
opt_config = OptimizationConfig(metric='sharpe_ratio', per_security=True)
results = optimizer.optimize_multiple(MyStrategy, param_grid, data_dict)

# Each security gets its own best parameters
for symbol, opt_result in results.items():
    print(f"{symbol}: {opt_result.get_best_parameters()}")
```

## Contributing

To add new features:
1. Follow the existing architecture (see ARCHITECTURE.md)
2. Add tests for new functionality
3. Update examples if adding strategy features

## License

[Your License Here]

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
