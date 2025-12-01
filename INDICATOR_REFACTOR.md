# Indicator Refactoring - Reading from Raw Data

## Overview

The backtesting framework has been refactored to read standard technical indicators directly from raw data CSV files instead of calculating them. This simplifies the framework and ensures consistency with external calculation sources (e.g., TradingView).

## Standard Indicators (Read from Raw Data)

The following indicators are now read from raw data with **fixed settings** and **cannot be altered via parameters**:

| Indicator | Column Name | Settings | Description |
|-----------|-------------|----------|-------------|
| ATR | `atr_14` | 14-period | Average True Range |
| EMA | `ema_50` | 50-period | Exponential Moving Average |
| SMA | `sma_200` | 200-period | Simple Moving Average |
| RSI | `rsi_14` | 14-period | Relative Strength Index |
| CMF | `cmf` | Static | Chaikin Money Flow |
| Parabolic SAR | `par_sar` | Static | Parabolic SAR |
| Bollinger Bands Base | `base_bb` | Static | Middle Bollinger Band |
| Bollinger Bands Upper | `upper_bb` | Static | Upper Bollinger Band |
| Bollinger Bands Lower | `lower_bb` | Static | Lower Bollinger Band |

## Changes Made

### 1. IndicatorEngine (`Classes/Indicators/indicator_engine.py`)

**Before:**
- Calculated all indicators using vectorized operations
- Had methods like `calculate_alphatrend_indicators()` and `calculate_basic_indicators()`
- Complex calculation logic with configurable periods

**After:**
- Simplified to validation-only class
- Provides `validate_indicators()` to check if required columns exist
- Provides `get_available_indicators()` to list present indicators
- Provides `prepare_data()` as a no-op for backward compatibility

### 2. AlphaTrendStrategy (`strategies/alphatrend_strategy.py`)

**Before:**
- Had `common_period` parameter (default: 14) for ATR/MFI
- Had `exit_ema_period` parameter (default: 50) for exit EMA
- Calculated ATR internally with configurable period

**After:**
- Removed `common_period` parameter (always uses `atr_14` from raw data)
- Removed `exit_ema_period` parameter (always uses `ema_50` from raw data)
- Reads `atr_14` and `ema_50` directly from CSV columns
- Still calculates custom AlphaTrend-specific indicators (MFI, alphatrend bands, etc.)

**Required Columns:**
```python
['date', 'open', 'high', 'low', 'close', 'volume', 'atr_14', 'ema_50']
```

### 3. Example Strategies

**Note:** AdvancedTrailingStopStrategy and PartialExitStrategy have been removed from the framework.

The framework now includes AlphaTrendStrategy as the primary example strategy, which demonstrates:
- Reading pre-calculated indicators from raw data (atr_14, ema_50)
- Custom strategy-specific indicator calculations
- Risk-based position sizing
- Volume filtering

**Required Columns:**
```python
['date', 'close', 'rsi_14']
```

### 4. BaseStrategy (`Classes/Strategy/base_strategy.py`)

**Documentation Updated:**
- Clarified that standard indicators are read from raw data
- Updated examples to show reading from raw data
- Explained difference between standard indicators (read from data) and custom indicators (calculated)

### 5. Export Script (`export_alphatrend_indicators.py`)

**Before:**
- Calculated indicators using IndicatorEngine
- Required strategy preset parameters

**After:**
- Reads pre-calculated indicators from raw data
- Validates which standard indicators are present
- Exports all available indicators to CSV

### 6. Strategy Preset (`config/strategy_presets/AlphaTrendStrategy__Default.json`)

**Before:**
```json
{
  "common_period": 14,
  "exit_ema_period": 50,
  ...
}
```

**After:**
```json
{
  // common_period removed (atr_14 is always 14)
  // exit_ema_period removed (ema_50 is always 50)
  ...
}
```

## Migration Guide for Custom Strategies

If you have custom strategies that calculate indicators, follow these steps:

### Step 1: Update required_columns()

**Before:**
```python
def required_columns(self) -> List[str]:
    return ['date', 'close']
```

**After:**
```python
def required_columns(self) -> List[str]:
    return [
        'date', 'close',
        'atr_14',   # Pre-calculated from raw data
        'ema_50',   # Pre-calculated from raw data
        'rsi_14'    # Pre-calculated from raw data
    ]
```

### Step 2: Remove indicator calculation parameters

**Before:**
```python
def __init__(self, atr_period: int = 14, ema_period: int = 50):
    self.atr_period = atr_period
    self.ema_period = ema_period
```

**After:**
```python
def __init__(self):
    # No period parameters needed - always read from raw data
    pass
```

### Step 3: Update generate_signal() to read from raw data

**Before:**
```python
def generate_signal(self, context: StrategyContext) -> Signal:
    atr = context.get_indicator_value(f'atr_{self.atr_period}')
    ema = context.get_indicator_value(f'ema_{self.ema_period}')
```

**After:**
```python
def generate_signal(self, context: StrategyContext) -> Signal:
    atr = context.get_indicator_value('atr_14')  # Always 14-period
    ema = context.get_indicator_value('ema_50')  # Always 50-period
```

### Step 4: Remove prepare_data() if only calculating standard indicators

**Before:**
```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    from Classes.Indicators.indicator_engine import IndicatorEngine
    return IndicatorEngine.calculate_basic_indicators(
        data,
        atr_periods=[self.atr_period],
        ema_periods=[self.ema_period]
    )
```

**After:**
```python
# Remove prepare_data() entirely - indicators come from raw data
# OR keep it only for custom strategy-specific calculations:
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Verify required indicators exist
    if 'atr_14' not in df.columns:
        raise ValueError("Missing atr_14 in raw data")

    # Calculate only custom strategy-specific indicators
    df['custom_signal'] = ...  # Your custom logic
    return df
```

## Benefits

1. **Simplicity**: No need to configure indicator periods - they're standardized
2. **Consistency**: Indicators calculated externally (e.g., TradingView) match exactly
3. **Performance**: No calculation overhead during backtesting
4. **Reliability**: Single source of truth for indicator values
5. **Maintainability**: Less code to maintain and test

## Raw Data Requirements

Your raw data CSV files must now include the standard indicator columns:

```csv
date,open,high,low,close,volume,atr_14,ema_50,sma_200,rsi_14,cmf,par_sar,base_bb,upper_bb,lower_bb
2024-01-01,100,102,99,101,1000,2.5,100.5,98.2,55.3,0.05,99.5,100,102,98
...
```

Not all indicators are required for all strategies - only include the ones your strategy uses.

## Custom Indicators

If your strategy needs custom calculations (like AlphaTrend's MFI thresholds or adaptive bands), you can still calculate those in `prepare_data()`. The refactoring only applies to standard indicators that are commonly used across multiple strategies.

## Questions or Issues?

If you encounter any issues with the refactoring, check:

1. Do your raw data files include the required indicator columns?
2. Are the column names correct (e.g., `atr_14`, not `ATR_14` or `atr`)?
3. Does your strategy's `required_columns()` list all needed indicators?
4. Are you reading indicators with the correct column names (e.g., `atr_14`, not `atr_{period}`)?
