# Configuration Reference

This document covers all configuration files and options in the BackTesting Framework.

---

## Configuration File Locations

```
config/
├── strategy_parameters.json       # Strategy parameter definitions
├── strategy_presets/              # Saved strategy configurations
│   └── {strategy_name}/
│       └── {preset_name}.json
│
├── baskets/                       # Security basket definitions
│   ├── All_Stocks.json
│   ├── Large_Cap_15.json
│   └── ...
│
├── vulnerability_presets/         # Vulnerability scoring presets
│   ├── base.json
│   └── optimised.json
│
├── factor_analysis/               # Factor analysis configuration
│   └── default.json
│
├── optimization_config.yaml       # Optimization settings
│
├── security_metadata.json         # Security characteristics
│
├── trade_analysis_config.json     # Trade analysis thresholds
│
└── data_collection/               # Data collection settings
    ├── settings.json              # API configuration
    └── tickers.json               # Ticker presets
```

---

## Strategy Parameters

### File: `config/strategy_parameters.json`

Defines parameters for all registered strategies:

```json
{
  "AlphaTrendStrategy": {
    "parameters": {
      "atr_multiplier": {
        "type": "float",
        "default": 1.5,
        "min": 0.5,
        "max": 5.0,
        "step": 0.1,
        "description": "ATR multiplier for trend calculation"
      },
      "atr_period": {
        "type": "int",
        "default": 14,
        "min": 5,
        "max": 50,
        "step": 1,
        "description": "ATR lookback period"
      },
      "risk_per_trade": {
        "type": "float",
        "default": 0.02,
        "min": 0.01,
        "max": 0.10,
        "step": 0.005,
        "description": "Risk percentage per trade"
      }
    },
    "required_columns": ["date", "open", "high", "low", "close", "volume", "atr_14", "ema_50", "mfi_14"]
  }
}
```

### Parameter Types

| Type | JSON Type | Example |
|------|-----------|---------|
| `int` | `"int"` | Periods, counts |
| `float` | `"float"` | Multipliers, percentages |
| `bool` | `"bool"` | Feature flags |

---

## Strategy Presets

### Location: `config/strategy_presets/{strategy_name}/{preset_name}.json`

Saved parameter configurations:

```json
{
  "name": "conservative",
  "description": "Lower risk settings",
  "strategy": "AlphaTrendStrategy",
  "parameters": {
    "atr_multiplier": 2.0,
    "atr_period": 14,
    "risk_per_trade": 0.015,
    "max_holding_days": 7
  }
}
```

---

## Security Baskets

### Location: `config/baskets/{basket_name}.json`

Groups of securities for portfolio testing:

```json
{
  "name": "Large_Cap_15",
  "description": "Top 15 large-cap stocks",
  "symbols": [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "JPM", "XOM", "PG", "HD"
  ]
}
```

### Available Baskets

| Basket | Description | Count |
|--------|-------------|-------|
| `All_Stocks` | Full equity universe | 165+ |
| `Large_Cap_15` | Top 15 by market cap | 15 |
| `Balanced_Large_Cap` | Diversified large caps | 20 |
| `Tech_Focused` | Technology sector | 25 |
| `All_ETFs` | Exchange-traded funds | 46 |

---

## Backtest Configuration

### Python Classes

```python
from Classes.Config.config import BacktestConfig, PortfolioConfig, CommissionConfig

# Single security backtest
config = BacktestConfig(
    symbol='AAPL',
    initial_capital=100000,
    commission=CommissionConfig(mode='percentage', value=0.001),
    data_path='raw_data/daily/AAPL_daily.csv',
    start_date='2020-01-01',  # Optional
    end_date='2023-12-31',    # Optional
)

# Portfolio backtest
portfolio_config = PortfolioConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    initial_capital=100000,
    commission=CommissionConfig(mode='percentage', value=0.001),
    max_positions=3,
    position_size_limit=0.40,
    total_allocation_limit=0.95,
    capital_contention_mode='VULNERABILITY_SCORE',  # or 'DEFAULT'
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `symbol` / `symbols` | str / list | Required | Security(ies) to test |
| `initial_capital` | float | 100000 | Starting capital |
| `data_path` | str | Required | Path to price data |
| `start_date` | str | None | Start of test period (YYYY-MM-DD) |
| `end_date` | str | None | End of test period |

### Commission Configuration

| Option | Type | Values | Description |
|--------|------|--------|-------------|
| `mode` | str | `'percentage'`, `'fixed'` | Commission type |
| `value` | float | 0.001 or 5.0 | Rate or amount |

### Portfolio-Specific Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_positions` | int | 3 | Maximum concurrent positions |
| `position_size_limit` | float | 0.40 | Max % per position |
| `total_allocation_limit` | float | 0.95 | Max % total allocated |
| `capital_contention_mode` | str | 'DEFAULT' | Contention resolution |

---

## Vulnerability Scoring

### Location: `config/vulnerability_presets/{preset_name}.json`

Position scoring weights:

```json
{
  "name": "optimised",
  "description": "Optimized scoring weights",
  "weights": {
    "profitability": 0.35,
    "age": 0.25,
    "distance_to_stop": 0.18,
    "momentum": 0.12,
    "signal_strength": 0.10
  },
  "thresholds": {
    "min_score_to_keep": 0.30,
    "swap_improvement_required": 0.15
  }
}
```

### Weight Components

| Component | Description | Higher Weight = |
|-----------|-------------|-----------------|
| `profitability` | Current unrealized P/L | Favor profitable positions |
| `age` | Days in position | Close older positions |
| `distance_to_stop` | Current price vs stop | Favor safer positions |
| `momentum` | Recent price direction | Favor trending positions |
| `signal_strength` | Entry signal quality | Favor strong entries |

---

## Optimization Configuration

### Location: `config/optimization_config.yaml`

Walk-forward and Bayesian optimization settings:

```yaml
walk_forward:
  training_months: 12
  test_months: 3
  step_months: 3
  min_trades_per_period: 20

bayesian:
  n_iterations: 100
  n_initial_points: 20
  acquisition_function: "EI"  # Expected Improvement
  kappa: 2.5

sensitivity:
  enabled: true
  n_samples: 50

metrics:
  primary: "sharpe_ratio"
  secondary:
    - "total_return"
    - "max_drawdown"
    - "profit_factor"
```

---

## Factor Analysis Configuration

### Location: `config/factor_analysis/default.json`

```json
{
  "trade_classification": {
    "good_threshold_pct": 2.0,
    "bad_threshold_pct": -1.0
  },
  "temporal_alignment": {
    "fundamental_delay_days": 0,
    "insider_delay_days": 3
  },
  "factors": {
    "technical": {
      "enabled": true,
      "include": ["RSI_14", "MACD", "Bollinger_%B", "Momentum_5d", "Volume_Ratio"]
    },
    "fundamental": {
      "enabled": true,
      "include": ["P/E", "P/B", "ROE", "Revenue_Growth"]
    },
    "insider": {
      "enabled": true,
      "lookback_days": 30
    },
    "regime": {
      "enabled": true
    }
  },
  "analysis": {
    "tier1": {"enabled": true},
    "tier2": {"enabled": true, "significance_level": 0.05},
    "tier3": {"enabled": true, "n_estimators": 100}
  },
  "scenarios": {
    "mode": "binary",
    "min_trades": 10,
    "max_factors": 2
  }
}
```

---

## Trade Analysis Configuration

### Location: `config/trade_analysis_config.json`

```json
{
  "trade_classification": {
    "good_profit_threshold": 0.02,
    "bad_loss_threshold": -0.01
  },
  "period_classification": {
    "good_calmar_threshold": 1.0,
    "bad_calmar_threshold": 0.5
  },
  "e_ratio": {
    "max_horizon_days": 30,
    "min_trades_per_horizon": 20,
    "winsorize_percentile": 0.99
  }
}
```

---

## Data Collection Configuration

### Location: `config/data_collection/settings.json`

```json
{
  "api_key": "YOUR_ALPHA_VANTAGE_KEY",
  "rate_limit": 75,
  "timeout_seconds": 30,
  "retry_attempts": 3,
  "output_directory": "raw_data"
}
```

### Location: `config/data_collection/tickers.json`

```json
{
  "presets": {
    "All_Stocks": ["AAPL", "MSFT", "GOOGL", ...],
    "All_ETFs": ["SPY", "QQQ", "IWM", ...],
    "Tech_Sector": ["AAPL", "MSFT", "GOOGL", "NVDA", ...],
    "Forex": ["USDGBP", "USDEUR", "USDJPY", ...]
  }
}
```

---

## Security Metadata

### Location: `config/security_metadata.json`

```json
{
  "AAPL": {
    "name": "Apple Inc.",
    "sector": "Technology",
    "market_cap": "Large",
    "currency": "USD"
  },
  "MSFT": {
    "name": "Microsoft Corporation",
    "sector": "Technology",
    "market_cap": "Large",
    "currency": "USD"
  }
}
```

---

## Environment Variables

Some settings can be overridden via environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `ALPHA_VANTAGE_API_KEY` | Data collection API key | `abc123` |
| `BACKTEST_DATA_PATH` | Default data directory | `/data/prices` |
| `BACKTEST_OUTPUT_PATH` | Default output directory | `/data/results` |

---

## Related Documentation

- [Quick Start](../overview/QUICK_START.md) — Getting started
- [Backtest GUI](../applications/BACKTEST_GUI.md) — Configuration in GUI
- [Optimization Tools](../applications/OPTIMIZATION_GUI.md) — Optimization settings
