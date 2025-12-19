# Backtesting Framework

A Python framework for backtesting trading strategies on historical data. Test strategies on single securities or portfolios, optimize parameters, and generate detailed performance reports.

## What It Does

- **Backtest trading strategies** on historical price data
- **Run single-security or portfolio backtests** with shared capital management
- **Optimize strategy parameters** using walk-forward analysis to prevent overfitting
- **Generate Excel reports** with performance metrics and charts
- **Use the GUI or Python code** - whatever suits your workflow

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/BackTestingFramework.git
cd BackTestingFramework

# Install dependencies
pip install -r requirements.txt

# On Linux, you may need tkinter:
# sudo apt-get install python3-tk
```

### Run a Backtest (GUI)

```bash
python run_gui.py
```

The GUI walks you through:
1. Selecting securities from your `raw_data/` folder
2. Choosing a strategy (e.g., AlphaTrendStrategy)
3. Setting commission and capital
4. Running the backtest and viewing results

### Run a Backtest (Python)

```python
from pathlib import Path
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Configure
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.001)
)

# Load data and run
loader = DataLoader(Path('raw_data'))
data = loader.load_csv('AAPL')
strategy = AlphaTrendStrategy()
engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)

# View results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Number of Trades: {result.num_trades}")
```

## Data Requirements

Place CSV files in the `raw_data/` folder. Each file should be named by symbol (e.g., `AAPL.csv`) and contain:

| Column | Required | Description |
|--------|----------|-------------|
| `date` or `time` | Yes | Timestamp |
| `close` | Yes | Closing price |
| `open`, `high`, `low` | Recommended | OHLC data |
| `volume` | Recommended | Trading volume |
| `atr_14`, `ema_50`, etc. | Strategy-dependent | Pre-calculated indicators |

## Project Structure

```
BackTestingFramework/
├── Classes/                  # Core framework code
│   ├── Config/              # Configuration classes
│   ├── Data/                # Data loading
│   ├── Engine/              # Backtesting engines
│   ├── Strategy/            # Strategy base classes
│   ├── Optimization/        # Parameter optimization
│   └── Analysis/            # Metrics and reporting
├── strategies/              # Trading strategies
├── raw_data/                # Your CSV data files
├── logs/                    # Backtest results and trade logs
├── config/                  # Configuration files
├── docs/                    # Detailed documentation
├── run_gui.py              # GUI launcher
├── optimize_gui.py         # Optimization GUI
└── backtest.py             # CLI examples
```

## Documentation

| Guide | Description |
|-------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete guide to using the framework |
| [Tools & Executables](docs/TOOLS.md) | All available applications and scripts |
| [Strategies](docs/STRATEGIES.md) | Creating and configuring trading strategies |
| [Configuration](docs/CONFIGURATION.md) | All configuration options explained |
| [Optimization](docs/OPTIMIZATION.md) | Finding optimal strategy parameters |
| [Portfolio Mode](docs/PORTFOLIO_MODE.md) | Multi-security backtesting |

## Key Features

### Execution Model
- Trades execute at **closing prices** (matches TradingView behavior)
- No lookahead bias - strategies only see historical data
- Configurable commission (percentage or fixed per trade)
- Long positions only

### Strategy Features
- Stop loss and take profit levels
- Trailing stops with indicator-based adjustments
- Partial position exits (scale out of trades)
- Custom position sizing

### Optimization
- Walk-forward optimization with Bayesian search
- Sensitivity analysis to identify robust parameters
- Prevents overfitting by testing on out-of-sample data

### Reporting
- Excel reports with 50+ performance metrics
- Equity curves and drawdown charts
- Trade-by-trade logs

## Requirements

- Python 3.8+
- pandas, numpy
- openpyxl (Excel reports)
- scikit-optimize (optimization)
- tkinter (GUI)

## License

[Your License Here]
