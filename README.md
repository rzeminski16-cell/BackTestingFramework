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

### Launch the Main Menu (Recommended)

```bash
python ctk_main_gui.py
```

The main launcher provides access to all framework features:
- **Backtesting** - Run backtests with configurable strategies
- **Optimization** - Walk-forward parameter optimization
- **Edge Analysis** - E-ratio and R-multiple analysis
- **Factor Analysis** - Performance factor analysis
- **Vulnerability Modeler** - Vulnerability score optimization

### Run a Backtest (GUI)

```bash
python ctk_backtest_gui.py
```

The GUI walks you through:
1. Selecting securities from your `raw_data/` folder
2. Choosing a strategy (e.g., AlphaTrendStrategy, RandomBaseStrategy)
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
├── ctk_main_gui.py          # Main launcher (start here)
├── ctk_backtest_gui.py      # Backtesting GUI
├── ctk_optimization_gui.py  # Optimization GUI
├── ctk_edge_analysis_gui.py # Edge Analysis GUI (E-ratio, R-multiple)
├── ctk_factor_analysis_gui.py # Factor Analysis GUI
├── ctk_vulnerability_gui.py # Vulnerability Modeler GUI
└── run_fundamental_data_fetch.py # Alpha Vantage data fetcher
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
| [Backtest Analysis](docs/BACKTEST_ANALYSIS_GUIDE.md) | Drill down into strategy performance |
| [Fundamental Data](docs/FUNDAMENTAL_DATA_GUIDE.md) | Fetch financial metrics from Alpha Vantage |

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

### Backtest Analysis
- **Drill down into results** to understand when strategies work best
- **Technical indicators at each trade** (RSI, Bollinger Bands, ATR, etc.)
- **Trade classification** (Good/Indeterminate/Bad based on profit)
- **Outcome analysis** (FullRideGood, EarlyExitGood, MissedOpportunity)
- **Period classification** based on Calmar ratio and drawdown

### Fundamental Data Integration
- **Alpha Vantage API** integration for premium financial data
- **Point-in-time metrics** (EPS, P/E, Revenue Growth, FCF, etc.)
- **20+ fundamental indicators** per quarter
- **Interactive mode** with answer memory for data ambiguities
- **Caching** to minimize API calls

## Requirements

- Python 3.8+
- pandas, numpy
- openpyxl (Excel reports)
- scikit-optimize (optimization)
- tkinter (GUI)

## License

[Your License Here]
