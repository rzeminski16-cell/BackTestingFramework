# Configuration Reference

Complete reference for all configuration options in the Backtesting Framework.

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
| `start_date` | datetime | None | Filter data start date |
| `end_date` | datetime | None | Filter data end date |
| `base_currency` | str | 'GBP' | Base currency for P/L |

### Commission Configuration

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
| `initial_capital` | float | 100000.0 | Total starting capital |
| `max_positions` | int | None | Maximum concurrent positions |
| `position_size_limit` | float | 0.3 | Max fraction per position (0.3 = 30%) |
| `total_allocation_limit` | float | 0.9 | Max total capital in positions |
| `capital_contention` | CapitalContentionConfig | DEFAULT | Competing signal handling |
| `commission` | CommissionConfig | None | Commission settings |
| `base_currency` | str | 'GBP' | Base currency |

### Capital Contention Modes

**DEFAULT mode:**
```python
CapitalContentionConfig(mode=CapitalContentionMode.DEFAULT)
```
First-come-first-served. Later signals rejected if capital runs out.

**VULNERABILITY_SCORE mode:**
```python
CapitalContentionConfig(
    mode=CapitalContentionMode.VULNERABILITY_SCORE,
    swap_threshold=50.0
)
```
Evaluates and may close weak positions to make room for new signals.

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
| `training_period_days` | int | Days for optimization |
| `testing_period_days` | int | Days for validation |
| `step_size_min_days` | int | Minimum window roll-forward |
| `step_size_max_days` | int | Maximum window roll-forward |

### Bayesian Optimization Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_iterations` | int | Parameter sets per window |
| `n_initial_points` | int | Random explorations before Bayesian |
| `n_jobs` | int | CPU cores (Linux/Mac only) |
| `speed_mode` | str | "quick" (25), "fast" (50), "full" (100) |
| `random_state` | int/null | Seed for reproducibility |

### Optimization Constraints

| Parameter | Type | Description |
|-----------|------|-------------|
| `primary_metric` | str | Metric to maximize |
| `min_profit_factor` | float | Minimum acceptable profit factor |
| `max_drawdown_percent` | float | Maximum acceptable drawdown |
| `min_trades_per_year` | int | Minimum trading frequency |

**Available metrics:** `sortino_ratio`, `sharpe_ratio`, `profit_factor`, `total_return_pct`

### Parameter Ranges

```yaml
strategy_parameters:
  StrategyName:
    parameter_name:
      min: 2        # Lower bound
      max: 10       # Upper bound
      type: int     # int or float
      step: 1       # Optional step size
```

Or explicit values:
```yaml
    parameter_name:
      values: [1.0, 1.5, 2.0, 2.5, 3.0]
      type: float
```

---

## Strategy Parameters

### AlphaTrendStrategy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volume_short_ma` | int | 4 | Short-term volume MA period |
| `volume_long_ma` | int | 30 | Long-term volume MA period |
| `volume_alignment_window` | int | 14 | Bars for volume condition |
| `stop_loss_percent` | float | 0.0 | % stop loss (0 = use ATR) |
| `atr_stop_loss_multiple` | float | 2.5 | Stop loss = entry - (ATR × this) |
| `grace_period_bars` | int | 14 | Bars to ignore EMA exit |
| `momentum_gain_pct` | float | 2.0 | % gain for momentum protection |
| `momentum_lookback` | int | 7 | Bars for momentum check |
| `risk_percent` | float | 2.0 | % of equity to risk |

### Usage

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
Click "Configure Strategy Parameters" to adjust values.

**Via Presets:**
`config/strategy_presets/`:
```json
{
  "strategy_class": "AlphaTrendStrategy",
  "parameters": {
    "volume_short_ma": 4,
    "volume_long_ma": 30,
    "atr_stop_loss_multiple": 2.5
  }
}
```

---

## Currency Configuration

### Security Metadata

`config/security_metadata.json`:
```json
{
  "AAPL": {
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

The rate format: 1 USD = X GBP.

### Usage

```python
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Data.security_registry import SecurityRegistry

registry = SecurityRegistry(Path('config/security_metadata.json'))
converter = CurrencyConverter(base_currency='GBP')
converter.load_rates_directory(Path('currency_rates/'))

engine = SingleSecurityEngine(
    config=config,
    currency_converter=converter,
    security_registry=registry
)
```

---

## Data Collection Configuration

### Settings File

`config/data_collection/settings.json`:

```json
{
    "api": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://www.alphavantage.co/query",
        "requests_per_minute": 75,
        "timeout_seconds": 30,
        "retry_attempts": 3,
        "retry_base_delay_seconds": 2
    },
    "cache": {
        "enabled": true,
        "directory": ".cache/alpha_vantage",
        "ttl_hours": 24
    },
    "validation": {
        "enabled": true,
        "strict_mode": false,
        "on_error": "skip"
    },
    "output": {
        "base_directory": "raw_data",
        "create_metadata": true,
        "overwrite_existing": false
    },
    "logging": {
        "level": "INFO",
        "directory": "logs/data_collection"
    }
}
```

### API Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | Required | Alpha Vantage API key |
| `requests_per_minute` | 75 | Rate limit |
| `timeout_seconds` | 30 | Request timeout |
| `retry_attempts` | 3 | Retry count on failure |

### Validation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `on_error` | "skip" | skip, fill, flag, or fail |
| `strict_mode` | false | Stricter validation |

---

## Excel Report Configuration

```python
from Classes.Analysis.excel_report_generator import ExcelReportGenerator

generator = ExcelReportGenerator(
    output_directory=Path('logs/reports'),
    initial_capital=100000.0,
    risk_free_rate=0.035,      # 3.5% annual
    benchmark_name="S&P 500"
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_directory` | Required | Report save location |
| `initial_capital` | Required | For calculations |
| `risk_free_rate` | 0.035 | For Sharpe ratio |
| `benchmark_name` | "S&P 500" | Display name |

---

## Vulnerability Scoring Configuration

### Preset Files

`config/vulnerability_presets/`:

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
      "weight": -5.0
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
| `immunity_days` | int | Days before position can be swapped |
| `base_score` | float | Starting score (100) |
| `swap_threshold` | float | Score below which swap is allowed |

---

## Environment Variables

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
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.0)
)
```

```yaml
bayesian_optimization:
  speed_mode: quick
  n_jobs: 4
```

### For Production/Final Results

```python
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.001)
)
```

```yaml
bayesian_optimization:
  speed_mode: full
  n_jobs: 1
```

### For TradingView Matching

```python
config = BacktestConfig(
    initial_capital=100000.0,
    commission=CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001  # Match TradingView commission
    )
    # No slippage (default)
    # Trades at close prices (default)
)
```
