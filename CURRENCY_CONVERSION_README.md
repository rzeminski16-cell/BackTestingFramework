# Currency Conversion Feature

## Overview

The backtesting framework now supports multi-currency backtesting with automatic currency conversion. The base currency (currency of account) is always **GBP**, but securities can be traded in different currencies (USD, EUR, etc.).

## Key Components

### 1. Security Currency Configuration

Each security now has a `currency` field in the security metadata configuration (`config/security_metadata.json`):

```json
{
  "AAPL": {
    "type": "stock",
    "sector": "technology",
    "description": "Apple Inc.",
    "currency": "USD"
  }
}
```

### 2. Currency Conversion Rates

Currency conversion rates are loaded from CSV files in the `currency_rates/` directory. The expected filename format is `XXXGBP.csv` (e.g., `USDGBP.csv` for USD to GBP conversion).

**CSV Format:**
```csv
date,rate
2020-01-02,0.7634
2020-01-03,0.7627
...
```

Where:
- `date`: The date of the exchange rate
- `rate`: Exchange rate from the security currency to GBP (e.g., 1 USD = 0.7634 GBP)

### 3. How It Works

1. **Position Values**: When calculating position values, the framework automatically converts from the security's currency to GBP using the exchange rate for that date.

2. **Trade P&L**: All profit/loss calculations are performed in the security's currency, then converted to GBP at the time of the transaction.

3. **Capital Management**: The account capital is maintained in GBP. When buying securities:
   - Calculate the cost in the security's currency
   - Convert to GBP using the current exchange rate
   - Deduct from GBP capital

4. **Portfolio Aggregation**: In portfolio backtests, positions in different currencies are all converted to GBP before aggregating total equity.

## Usage Example

**IMPORTANT:** You must pass both `security_registry` and `currency_converter` to the engine for currency conversion to work!

```python
from pathlib import Path
from Classes.Config.config import BacktestConfig, PortfolioConfig
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Data.security_registry import SecurityRegistry
from Classes.Data.data_loader import DataLoader
from Classes.Strategy.moving_average_crossover import MovingAverageCrossover

# STEP 1: Initialize security registry (contains currency metadata)
security_registry = SecurityRegistry(
    metadata_file=Path('config/security_metadata.json')
)

# STEP 2: Initialize currency converter and load FX rates
currency_converter = CurrencyConverter(base_currency='GBP')
currency_converter.load_rates_directory(Path('currency_rates/'))

# STEP 3: Create backtest configuration
config = BacktestConfig(
    initial_capital=100000.0,  # GBP
    base_currency='GBP'
)

# STEP 4: Initialize engine with BOTH currency_converter AND security_registry
# ⚠️ If you don't pass these, currencies will default to GBP!
engine = SingleSecurityEngine(
    config=config,
    currency_converter=currency_converter,  # ← Required for FX conversion
    security_registry=security_registry     # ← Required to read security currencies
)

# STEP 5: Load data and run backtest
data_loader = DataLoader(data_dir=Path('raw_data/'))
aapl_data = data_loader.load_symbol('AAPL')  # USD-denominated stock

strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
result = engine.run(symbol='AAPL', data=aapl_data, strategy=strategy)

# Results are in GBP with proper FX conversion
print(f"Total Return (GBP): £{result.total_return:.2f}")
```

## Portfolio Example

```python
# Portfolio backtest with multi-currency securities
portfolio_config = PortfolioConfig(
    initial_capital=100000.0,  # GBP
    base_currency='GBP',
    max_positions=3,
    position_size_limit=0.3
)

portfolio_engine = PortfolioEngine(
    config=portfolio_config,
    currency_converter=currency_converter,
    security_registry=security_registry
)

# Load data for multiple securities
data_dict = {
    'AAPL': data_loader.load_symbol('AAPL'),  # USD
    'MSFT': data_loader.load_symbol('MSFT'),  # USD
    'JPM': data_loader.load_symbol('JPM'),    # USD
}

results = portfolio_engine.run(data_dict=data_dict, strategy=strategy)

# All results are in GBP
for symbol, result in results.items():
    print(f"{symbol}: £{result.total_return:.2f}")
```

## Important Notes

### 1. Exchange Rate Data

- You must provide exchange rate data for any currency that differs from GBP
- Exchange rates should be in the format: 1 unit of foreign currency = X GBP
- The framework uses the most recent exchange rate available on or before the transaction date
- Missing exchange rates will default to 1.0 (i.e., no conversion), which may produce inaccurate results

### 2. Obtaining Exchange Rate Data

You can obtain historical exchange rate data from sources like:
- Bank of England (BoE) for GBP rates
- European Central Bank (ECB)
- Federal Reserve Economic Data (FRED)
- Financial data APIs (Yahoo Finance, Alpha Vantage, etc.)

### 3. Backward Compatibility

If you don't provide a currency converter or security registry, the framework will:
- Default all currencies to GBP
- Apply no currency conversion (exchange rate = 1.0)
- Work as before with all calculations in a single implied currency

### 4. Accuracy Considerations

- Currency conversion adds a source of return/risk to your backtest
- Exchange rate movements can significantly impact returns
- Historical exchange rates reflect actual market conditions
- Consider the impact of FX volatility on your strategy's performance

## File Structure

```
BackTestingFramework/
├── config/
│   └── security_metadata.json          # Security metadata with currency field
├── currency_rates/                     # Currency conversion data
│   ├── USDGBP.csv                      # USD to GBP rates
│   ├── EURGBP.csv                      # EUR to GBP rates (if needed)
│   └── ...
├── Classes/
│   ├── Data/
│   │   ├── currency_converter.py       # Currency conversion logic
│   │   └── security_registry.py        # Security metadata with currency
│   ├── Engine/
│   │   ├── single_security_engine.py   # Updated with FX conversion
│   │   └── portfolio_engine.py         # Updated with FX conversion
│   └── Config/
│       └── config.py                   # Updated with base_currency field
```

## Technical Details

### Currency Converter Class

The `CurrencyConverter` class provides:

- `load_rates(currency_pair, file_path)`: Load rates for a specific currency pair
- `load_rates_directory(directory)`: Load all rate files from a directory
- `get_rate(from_currency, to_currency, date)`: Get exchange rate for a specific date
- `convert(amount, from_currency, to_currency, date)`: Convert an amount
- `has_rate(currency)`: Check if rates are available for a currency

### Engine Integration

Both `SingleSecurityEngine` and `PortfolioEngine` now accept:

```python
def __init__(self,
             config: BacktestConfig/PortfolioConfig,
             currency_converter: Optional[CurrencyConverter] = None,
             security_registry: Optional[SecurityRegistry] = None):
```

Internal helper methods:
- `_get_fx_rate(symbol, date)`: Get FX rate for a symbol on a specific date
- `_convert_to_base_currency(amount, symbol, date)`: Convert amount to GBP

## Example Output

```
Backtest Results for AAPL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initial Capital: £100,000.00 (GBP)
Final Equity: £125,450.75 (GBP)
Total Return: £25,450.75 (25.45%)
Total Trades: 42
Win Rate: 57.14%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

All monetary values are now clearly in GBP, ensuring accurate multi-currency portfolio tracking.
