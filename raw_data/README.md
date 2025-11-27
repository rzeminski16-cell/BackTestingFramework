# Raw Data Folder

## File Naming Convention
Each CSV file should be named with the security symbol: `SYMBOL.csv`

Examples:
- AAPL.csv
- MSFT.csv
- GC=F.csv (for Gold Futures)
- BTC-USD.csv (for Bitcoin)

## Required CSV Schema
Each CSV file should contain ALL the data you might need for your strategies,
including OHLC prices, volume, and any technical indicators you've calculated.

### Minimum Required Columns:
| Column | Description | Format | Example |
|--------|-------------|--------|---------|
| date | Trading date | YYYY-MM-DD | 2024-01-15 |
| close | Closing price | Float | 151.75 |

### Common Additional Columns:
| Column | Description | Example |
|--------|-------------|---------|
| open | Opening price | 150.25 |
| high | High price | 152.80 |
| low | Low price | 149.50 |
| volume | Trading volume | 50000000 |
| sma_20 | 20-period Simple Moving Average | 149.50 |
| sma_50 | 50-period Simple Moving Average | 148.25 |
| ema_12 | 12-period Exponential Moving Average | 151.00 |
| rsi_14 | 14-period Relative Strength Index | 65.5 |
| macd | MACD line | 2.5 |
| macd_signal | MACD signal line | 2.1 |
| bb_upper | Bollinger Band Upper | 155.0 |
| bb_middle | Bollinger Band Middle | 150.0 |
| bb_lower | Bollinger Band Lower | 145.0 |
| atr_14 | 14-period Average True Range | 3.5 |

## Example CSV Format:
```
date,open,high,low,close,volume,sma_20,sma_50,rsi_14
2024-01-02,185.50,186.40,184.20,185.90,45000000,184.50,183.20,62.5
2024-01-03,186.00,187.50,185.80,186.25,48000000,184.75,183.30,63.2
2024-01-04,186.30,188.00,186.10,187.50,52000000,185.10,183.45,65.8
```

## Important Notes:
1. **Include ALL indicators your strategies might use** - The framework will filter to only what's needed
2. **Header row is required** - First row must contain column names
3. **Date format must be YYYY-MM-DD** - Other formats will cause errors
4. **Sorted by date** - Data should be in chronological order (oldest first)
5. **Consistent column names** - Use the same naming across all securities
6. **Handle NaN values** - The framework will drop rows with missing values in required columns

## Column Name Mapping:
If your raw data uses different column names (e.g., 'Close' instead of 'close'), 
you can provide a mapping when preparing data:

```python
column_mapping = {
    'date': 'Date',           # Strategy expects 'date', raw data has 'Date'
    'close': 'Close',         # Strategy expects 'close', raw data has 'Close'
    'sma_50': 'SMA_50'        # Strategy expects 'sma_50', raw data has 'SMA_50'
}

data_prep.prepare_data_for_strategy(strategy, security_metadata, column_mapping)
```

## Where to Get Data with Indicators:
- **TradingView**: Export charts with indicators
- **Yahoo Finance + pandas_ta**: Download data and calculate indicators
- **Alpaca Markets**: Historical data with indicators
- **Polygon.io**: Technical indicators API
- **TA-Lib**: Calculate indicators locally

## Example Python Code to Generate Data with Indicators:
```python
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# Download data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="5y")
df = df.reset_index()

# Standardize column names
df.columns = [col.lower() for col in df.columns]
df = df.rename(columns={'date': 'date'})

# Calculate indicators
df['sma_20'] = ta.sma(df['close'], length=20)
df['sma_50'] = ta.sma(df['close'], length=50)
df['rsi_14'] = ta.rsi(df['close'], length=14)
df['ema_12'] = ta.ema(df['close'], length=12)

# MACD
macd = ta.macd(df['close'])
df['macd'] = macd['MACD_12_26_9']
df['macd_signal'] = macd['MACDs_12_26_9']

# Bollinger Bands
bbands = ta.bbands(df['close'], length=20)
df['bb_upper'] = bbands['BBU_20_2.0']
df['bb_middle'] = bbands['BBM_20_2.0']
df['bb_lower'] = bbands['BBL_20_2.0']

# Save
df.to_csv('raw_data/AAPL.csv', index=False)
```

## Data Quality Checklist:
- [ ] All CSV files are in the raw_data folder
- [ ] Each file is named correctly (SYMBOL.csv)
- [ ] All columns your strategies need are present
- [ ] Dates are in YYYY-MM-DD format
- [ ] Data is sorted chronologically
- [ ] Column names are consistent across all files
- [ ] Indicators are properly calculated (no lookahead bias)
