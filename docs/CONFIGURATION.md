# Configuration Guide

This guide covers all configuration options for backtests, portfolios, optimization, and reporting.

## Backtest Configuration

Used for single-security backtests.

### Python Usage

```python
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from datetime import datetime

config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001
    ),
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    base_currency='GBP'
)
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 100000.0 | Starting capital |
| `commission` | CommissionConfig | None | Commission settings |
| `start_date` | datetime | None | Filter data to start from this date |
| `end_date` | datetime | None | Filter data to end at this date |
| `base_currency` | str | 'GBP' | Base currency for P/L calculations |

### Commission Configuration

Two modes available:

**Percentage-based:**
```python
CommissionConfig(
    mode=CommissionMode.PERCENTAGE,
    value=0.001  # 0.1% of trade value
)
```

**Fixed per trade:**
```python
CommissionConfig(
    mode=CommissionMode.FIXED,
    value=3.0  # £3 per trade
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | CommissionMode | PERCENTAGE or FIXED |
| `value` | float | Commission rate (percentage) or amount (fixed) |

---

## Portfolio Configuration

Used for multi-security backtests with shared capital.

### Python Usage

```python
from Classes.Config.config import PortfolioConfig, CapitalContentionConfig, CapitalContentionMode

config = PortfolioConfig(
    initial_capital=100000.0,
    max_positions=3,
    position_size_limit=0.3,
    total_allocation_limit=0.9,
    capital_contention=CapitalContentionConfig(
        mode=CapitalContentionMode.DEFAULT
    ),
    base_currency='GBP'
)
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 100000.0 | Total starting capital for portfolio |
| `max_positions` | int | None | Maximum concurrent open positions |
| `position_size_limit` | float | 0.3 | Max fraction of capital per position (e.g., 0.3 = 30%) |
| `total_allocation_limit` | float | 0.9 | Max total capital in positions (e.g., 0.9 = 90%) |
| `capital_contention` | CapitalContentionConfig | DEFAULT | How to handle competing signals |
| `commission` | CommissionConfig | None | Commission settings |
| `base_currency` | str | 'GBP' | Base currency for all calculations |

### Capital Contention Modes

When multiple securities signal at once and there's not enough capital:

**DEFAULT mode:**
```python
CapitalContentionConfig(mode=CapitalContentionMode.DEFAULT)
```
First-come-first-served. Later signals are rejected if capital runs out.

**VULNERABILITY_SCORE mode:**
```python
CapitalContentionConfig(
    mode=CapitalContentionMode.VULNERABILITY_SCORE,
    swap_threshold=50.0
)
```
Evaluates existing positions for "vulnerability" (weakness). May close weak positions to make room for new signals.

---

## Optimization Configuration

Controls parameter optimization behavior.

### Config File Location

`config/optimization_config.yaml`

### Structure

```yaml
walk_forward:
  training_period_days: 365
  testing_period_days: 365
  step_size_min_days: 7
  step_size_max_days: 30

bayesian_optimization:
  n_iterations: 100
  n_initial_points: 20
  n_jobs: 1
  speed_mode: fast

optimization:
  primary_metric: sortino_ratio
  constraints:
    min_profit_factor: 1.5
    max_drawdown_percent: 25.0
    min_trades_per_year: 1

sensitivity_analysis:
  float_variation_percent: 10.0
  int_variation_amount: 1
  test_mode: combinations
  max_sensitivity_tests: 1000

strategy_parameters:
  AlphaTrendStrategy:
    volume_short_ma:
      min: 2
      max: 10
      type: int
    atr_stop_loss_multiple:
      min: 1.5
      max: 4.0
      type: float
```

### Walk-Forward Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_period_days` | int | Days of data used for optimization |
| `testing_period_days` | int | Days of data used for validation |
| `step_size_min_days` | int | Minimum window roll-forward step |
| `step_size_max_days` | int | Maximum window roll-forward step |

### Bayesian Optimization Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_iterations` | int | Parameter sets to test per window |
| `n_initial_points` | int | Random explorations before Bayesian search |
| `n_jobs` | int | CPU cores for parallel processing (Linux/Mac only) |
| `speed_mode` | str | "quick" (25 iter), "fast" (50 iter), "full" (100 iter) |
| `random_state` | int/null | Seed for reproducibility |

### Optimization Constraints

| Parameter | Type | Description |
|-----------|------|-------------|
| `primary_metric` | str | Metric to maximize (sortino_ratio, sharpe_ratio, profit_factor, total_return_pct) |
| `min_profit_factor` | float | Minimum acceptable profit factor |
| `max_drawdown_percent` | float | Maximum acceptable drawdown |
| `min_trades_per_year` | int | Minimum trading frequency |

### Parameter Ranges

Define search bounds for each strategy parameter:

```yaml
strategy_parameters:
  StrategyName:
    parameter_name:
      min: 2        # Lower bound
      max: 10       # Upper bound
      type: int     # int or float
      step: 1       # Optional: step size for grid search
```

Or use explicit values:
```yaml
    parameter_name:
      values: [1.0, 1.5, 2.0, 2.5, 3.0]
      type: float
```

---

## Strategy Parameters

Each strategy has its own configurable parameters. Here are the AlphaTrend parameters:

### AlphaTrendStrategy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volume_short_ma` | int | 4 | Short-term volume moving average period |
| `volume_long_ma` | int | 30 | Long-term volume moving average period |
| `volume_alignment_window` | int | 14 | Bars to wait for volume condition after signal |
| `stop_loss_percent` | float | 0.0 | Percentage stop loss (0 = use ATR instead) |
| `atr_stop_loss_multiple` | float | 2.5 | Stop loss = entry - (ATR × this) |
| `grace_period_bars` | int | 14 | Bars to ignore EMA exit after entry |
| `momentum_gain_pct` | float | 2.0 | % gain threshold for momentum protection |
| `momentum_lookback` | int | 7 | Bars to check for momentum |
| `risk_percent` | float | 2.0 | % of equity to risk per trade |

### Using Strategy Parameters

**In Python:**
```python
strategy = AlphaTrendStrategy(
    volume_short_ma=4,
    volume_long_ma=30,
    atr_stop_loss_multiple=2.5,
    risk_percent=2.0
)
```

**Via GUI:**
Click "Configure Strategy Parameters" in the GUI to adjust values.

**Via Presets:**
Save/load parameter sets from `config/strategy_presets/`:
```json
{
  "strategy_class": "AlphaTrendStrategy",
  "parameters": {
    "volume_short_ma": 4,
    "volume_long_ma": 30,
    "atr_stop_loss_multiple": 2.5,
    "risk_percent": 2.0
  }
}
```

---

## Currency Configuration

For multi-currency backtesting.

### Security Metadata

`config/security_metadata.json`:
```json
{
  "AAPL": {
    "currency": "USD",
    "type": "stock",
    "sector": "technology"
  },
  "MSFT": {
    "currency": "USD",
    "type": "stock",
    "sector": "technology"
  },
  "VOD.L": {
    "currency": "GBP",
    "type": "stock",
    "sector": "telecom"
  }
}
```

### Currency Rate Files

Place in `currency_rates/` folder:

**USDGBP.csv:**
```csv
date,rate
2020-01-02,0.7634
2020-01-03,0.7627
```

The rate is: 1 USD = X GBP.

### Using Currency Conversion

```python
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Data.security_registry import SecurityRegistry

# Load currency data
registry = SecurityRegistry(Path('config/security_metadata.json'))
converter = CurrencyConverter(base_currency='GBP')
converter.load_rates_directory(Path('currency_rates/'))

# Pass to engine
engine = SingleSecurityEngine(
    config=config,
    currency_converter=converter,
    security_registry=registry
)
```

---

## Excel Report Configuration

Settings for report generation.

### Python Usage

```python
from Classes.Analysis.excel_report_generator import ExcelReportGenerator

generator = ExcelReportGenerator(
    output_directory=Path('logs/reports'),
    initial_capital=100000.0,
    risk_free_rate=0.02,      # 2% annual risk-free rate
    benchmark_name="S&P 500"  # For display purposes
)
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_directory` | Path | Required | Where to save reports |
| `initial_capital` | float | Required | Starting capital (for calculations) |
| `risk_free_rate` | float | 0.02 | Annual risk-free rate for Sharpe ratio |
| `benchmark_name` | str | "S&P 500" | Benchmark name for display |

---

## Vulnerability Scoring Configuration

For portfolio capital contention with vulnerability scoring.

### Preset Files

Located in `config/vulnerability_presets/`:

**default.json:**
```json
{
  "name": "Default",
  "description": "Balanced approach",
  "parameters": {
    "immunity_days": 7,
    "base_score": 100,
    "swap_threshold": 50
  },
  "features": {
    "days_held": {
      "enabled": true,
      "weight": -1.5
    },
    "current_pl_pct": {
      "enabled": true,
      "weight": 1.0
    },
    "pl_momentum_7d": {
      "enabled": true,
      "weight": 3.0
    }
  }
}
```

### Available Presets

| Preset | Description |
|--------|-------------|
| `default` | Balanced approach |
| `aggressive` | Quick to swap weak positions |
| `conservative` | Protect positions longer |
| `momentum_focused` | Emphasize recent momentum |

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `immunity_days` | int | New positions can't be swapped for this many days |
| `base_score` | float | Starting score (100 = fully protected) |
| `swap_threshold` | float | Score below which position can be swapped |

---

## GUI Settings

The GUI stores your last-used settings automatically.

### Location

Settings are stored in the user's home directory as hidden files or in a config folder.

### Reset Settings

To reset to defaults, delete the settings file or use "Reset" buttons in the GUI.

---

## Environment Variables

The framework respects these environment variables:

| Variable | Description |
|----------|-------------|
| `BACKTEST_DATA_DIR` | Override default data directory |
| `BACKTEST_LOG_DIR` | Override default log directory |

---

## Configuration Tips

### For Development/Testing

```python
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.0)  # No commission
)
```

```yaml
# optimization_config.yaml
bayesian_optimization:
  speed_mode: quick  # Faster iterations
  n_jobs: 4          # Use multiple cores
```

### For Production/Final Results

```python
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.001)  # Realistic commission
)
```

```yaml
# optimization_config.yaml
bayesian_optimization:
  speed_mode: full   # Maximum iterations
  n_jobs: 1          # More stable
```

### For TradingView Matching

```python
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001  # Match your TradingView commission
    )
    # No slippage (default)
    # Trades at close prices (default)
)
```

---

## Related Guides

- **[User Guide](USER_GUIDE.md)** - Getting started
- **[Strategies Guide](STRATEGIES.md)** - Strategy configuration
- **[Optimization Guide](OPTIMIZATION.md)** - Optimization settings
- **[Portfolio Mode Guide](PORTFOLIO_MODE.md)** - Portfolio configuration
