# AlphaTrend Enhanced Long-Only Strategy

## Overview

The AlphaTrend Enhanced Strategy is a sophisticated long-only trading system based on the AlphaTrend indicator originally developed by KivancOzbilgic. This implementation extends the base indicator with advanced risk management, adaptive features, and performance optimizations.

**Strategy Type:** Long-only (buy and hold until exit signal)
**Time Frame:** Works on any time frame (daily, weekly, etc.)
**Market Focus:** Trend-following with momentum confirmation

## Core Philosophy

The strategy combines three key elements:
1. **Trend Detection:** Uses AlphaTrend indicator to identify trend direction
2. **Volume Confirmation:** Requires volume alignment to filter false signals
3. **Risk Management:** Employs adaptive stops and position sizing

## Entry Conditions

A trade is entered when **ALL** of the following conditions are met:

### 1. AlphaTrend Signal
- AlphaTrend line crosses above its smoothed EMA-3
- This creates a "filtered buy signal" indicating bullish trend reversal
- The signal must be fresh (not already acted upon)

### 2. Volume Alignment
- Short-term volume MA (default: 4-period) > Long-term volume MA (default: 30-period)
- Volume condition must occur within the alignment window (default: 14 bars)
- Can occur before, during, or shortly after the AlphaTrend signal
- Ensures sufficient market participation

### 3. Position Availability
- No existing open position (long-only, one position at a time)
- Sufficient capital available for the trade

### Entry Timing
The strategy allows flexible entry timing:
- Can enter ON the signal bar if volume was confirmed in the previous 14 bars
- Can enter UP TO 14 bars after the signal if volume confirms later
- This prevents missing trades due to temporary volume conditions

## Exit Conditions

The strategy exits a position when **ANY** of the following occur:

### 1. EMA-50 Exit (Primary)
- Price closes below the 50-period Exponential Moving Average
- **Protection mechanisms:**
  - **Grace Period:** Ignores EMA exit for first 14 bars after entry (prevents premature exits)
  - **Momentum Protection:** Ignores EMA exit if price gained 2%+ in last 7 bars (protects strong moves)

### 2. Stop Loss (Risk Management)
- ATR-based stop: Entry price - (ATR × 2.5) [default]
- Alternative: Percentage-based stop (configurable)
- Trailing stop can be enabled through strategy methods

### 3. End of Backtest
- Any open position is closed at the last available price

## Technical Indicators

### AlphaTrend Indicator
The core of the strategy, AlphaTrend combines price action with volatility:

```
Upper Band = Low - (ATR × Adaptive Coefficient)
Lower Band = High + (ATR × Adaptive Coefficient)
```

**Adaptive Coefficient:**
- Base multiplier: 1.0 (static)
- Adjusted by volatility ratio: Current ATR / Long-term ATR EMA
- Higher volatility = wider bands = fewer false signals

**AlphaTrend Calculation:**
- In uptrend: AT = max(previous AT, upper band)
- In downtrend: AT = min(previous AT, lower band)
- Smoothed with 3-period EMA for cleaner signals

### Money Flow Index (MFI)
Used to determine momentum direction:
- 14-period MFI calculation
- Dynamic thresholds based on 100-bar percentiles
- Upper threshold: 70th percentile
- Lower threshold: 30th percentile
- Bullish when MFI ≥ (upper + lower) / 2

### Volume Filter
Simple but effective volume confirmation:
```
Short MA (4) > Long MA (30) = Volume increasing
```

### Exit Indicator
- 50-period EMA: Standard trend-following exit
- Pre-calculated from raw data, not computed during backtest

## Strategy Parameters

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volume_short_ma` | 4 | Short-term volume moving average period |
| `volume_long_ma` | 30 | Long-term volume moving average period |
| `volume_alignment_window` | 14 | Bars to wait for volume condition after signal |
| `stop_loss_percent` | 0.0 | Percentage-based stop loss (if ATR stop disabled) |
| `atr_stop_loss_multiple` | 2.5 | Multiple of ATR for stop loss distance |
| `grace_period_bars` | 14 | Bars to ignore EMA exit after entry |
| `momentum_gain_pct` | 2.0 | % gain threshold to ignore EMA exit |
| `momentum_lookback` | 7 | Bars to look back for momentum calculation |
| `risk_percent` | 2.0 | Percentage of equity to risk per trade |

### Static Parameters (Not Configurable)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `atr_multiplier` | 1.0 | Base multiplier before adaptive adjustment |
| `source` | 'close' | Price source for calculations |
| `smoothing_length` | 3 | EMA smoothing for AlphaTrend |
| `percentile_period` | 100 | Lookback for dynamic MFI thresholds |

## Position Sizing

The strategy uses **risk-based position sizing** to ensure consistent risk per trade:

```
Position Size = (Total Equity × Risk %) / Stop Distance
```

**Example:**
- Total Equity: $100,000
- Risk Percent: 2.0%
- Risk Amount: $2,000
- Entry Price: $50
- Stop Loss: $48
- Stop Distance: $2
- Position Size: $2,000 / $2 = 1,000 shares

**Multi-Currency Support:**
- Automatically converts stop distance to base currency
- Handles FX rate fluctuations
- See `CURRENCY_CONVERSION_README.md` for details

**Constraints:**
- Position value cannot exceed available capital
- Respects position size limits from configuration
- Accounts for commission costs (0.1% default)

## Performance Optimizations

### Vectorized Indicator Calculation

**Problem:** Original implementation calculated indicators on-the-fly during backtesting
- Time Complexity: O(n²) - recalculating for every bar
- Performance: Extremely slow on large datasets

**Solution:** Pre-calculate ALL indicators once before backtesting
- Time Complexity: O(n) - calculate each indicator once
- Performance: **10-100x speedup**
- Method: `prepare_data()` uses pandas/numpy vectorization

### Implementation Details

```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate all indicators using vectorized operations."""
    # AlphaTrend bands
    df['up_band'] = df['low'] - df['atr_14'] * df['adaptive_coeff']
    df['down_band'] = df['high'] + df['atr_14'] * df['adaptive_coeff']

    # MFI calculation (vectorized)
    df['positive_flow'] = np.where(df['price_change'] > 0, df['raw_money_flow'], 0)
    positive_mf = df['positive_flow'].rolling(window=14).sum()

    # Volume filter
    df['vol_short_ma'] = df['volume'].rolling(window=4).mean()
    df['volume_condition'] = df['vol_short_ma'] > df['vol_long_ma']

    return df
```

During backtesting, `generate_signal()` simply reads pre-calculated values:
```python
def generate_signal(self, context: StrategyContext) -> Signal:
    indicators = self._get_indicators(context)  # Just reads columns
    if indicators['filtered_buy'] and indicators['volume_condition']:
        return Signal.buy(...)
```

### Data Leakage Prevention

The strategy is designed to prevent future data leakage:
- All indicators use only historical data
- Signals can only trigger on or after the bar they appear
- Volume alignment looks backward or forward from signal, never uses future prices
- See `DATA_LEAKAGE_AUDIT.md` for comprehensive analysis

## Risk Management Features

### 1. Adaptive Stop Loss
- Uses ATR-based stops that adjust to market volatility
- Wider stops in volatile markets = fewer stop-outs
- Tighter stops in calm markets = better capital efficiency

### 2. Grace Period Protection
Prevents premature exits after entry:
- Ignores EMA exit signal for first 14 bars
- Allows trade to develop without immediate exit
- Only stop loss can close position during grace period

### 3. Momentum Protection
Protects winning trades during pullbacks:
- If price gained 2%+ in last 7 bars, ignore EMA exit
- Allows strong trends to continue despite minor corrections
- Prevents selling during healthy retracements

### 4. Risk-Based Sizing
- Consistent risk per trade (2% default)
- Larger positions in lower-volatility setups
- Smaller positions in higher-volatility setups
- Automatically adjusts to account balance

## Usage Example

### Basic Configuration

```python
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Create strategy with default parameters
strategy = AlphaTrendStrategy()

# Or customize parameters
strategy = AlphaTrendStrategy(
    volume_short_ma=5,           # Faster volume filter
    volume_long_ma=25,
    volume_alignment_window=10,   # Tighter window
    atr_stop_loss_multiple=3.0,  # Wider stop
    grace_period_bars=20,         # Longer grace period
    momentum_gain_pct=3.0,        # Higher momentum threshold
    risk_percent=1.5              # More conservative risk
)
```

### Running a Backtest

```python
from Classes.Config.config import BacktestConfig
from Classes.Engine.single_security_engine import SingleSecurityEngine

# Configure backtest
config = BacktestConfig(
    initial_capital=100000,
    commission_percent=0.1,
    slippage_percent=0.05,
    position_size_limit=0.95
)

# Create engine and run
engine = SingleSecurityEngine(config)
result = engine.run(
    symbol='AAPL',
    strategy=strategy,
    data=data  # DataFrame with OHLCV + atr_14 + ema_50
)

# Analyze results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Number of Trades: {len(result.trades)}")
```

### Required Data Columns

Your input data must include these pre-calculated indicators:

```python
required_columns = [
    'date',      # Timestamp
    'open',      # Open price
    'high',      # High price
    'low',       # Low price
    'close',     # Close price
    'volume',    # Volume
    'atr_14',    # 14-period Average True Range
    'ema_50'     # 50-period Exponential Moving Average
]
```

The strategy will calculate all AlphaTrend-specific indicators in `prepare_data()`.

## Parameter Tuning Guidelines

### For More Aggressive Trading
- Decrease `volume_long_ma` (e.g., 20) - faster to signal volume increase
- Decrease `grace_period_bars` (e.g., 10) - exit sooner
- Decrease `momentum_gain_pct` (e.g., 1.5%) - less momentum protection
- Increase `risk_percent` (e.g., 2.5%) - larger positions

### For More Conservative Trading
- Increase `volume_long_ma` (e.g., 40) - slower to signal, fewer trades
- Increase `grace_period_bars` (e.g., 20) - more time for trades to work
- Increase `momentum_gain_pct` (e.g., 3.0%) - more momentum protection
- Decrease `risk_percent` (e.g., 1.0%) - smaller positions
- Increase `atr_stop_loss_multiple` (e.g., 3.0) - wider stops

### For Different Time Frames
**Daily Charts:**
- Use default parameters as a starting point
- Consider `volume_alignment_window=10-14`

**Weekly Charts:**
- Increase `volume_alignment_window=4-6` (fewer bars in a month)
- Increase `grace_period_bars=4-6`
- Adjust `momentum_lookback=3-4`

**Intraday Charts:**
- Decrease `volume_alignment_window=20-30` (more bars per day)
- May need to adjust MFI and ATR periods (not currently configurable)

## Performance Metrics

When evaluating this strategy, focus on:

1. **Win Rate:** Typically 40-60% (trend-following nature)
2. **Average Win/Loss Ratio:** Should be > 1.5 (winners bigger than losers)
3. **Maximum Drawdown:** Monitor for acceptable risk level
4. **Sharpe Ratio:** Risk-adjusted returns (target > 1.0)
5. **Trade Frequency:** Depends on market conditions and parameters

## Common Issues and Solutions

### Too Many Trades
- Increase `volume_long_ma` to filter more signals
- Decrease `volume_alignment_window` for stricter timing
- Consider adding minimum trade duration

### Too Few Trades
- Decrease `volume_long_ma` for faster signals
- Increase `volume_alignment_window` for more flexibility
- Check if data has sufficient liquidity/volume

### Frequent Stop-Outs
- Increase `atr_stop_loss_multiple` for wider stops
- Increase `grace_period_bars` to avoid early exits
- Check if entry logic needs refinement

### Holding Losers Too Long
- Decrease `grace_period_bars`
- Decrease `momentum_gain_pct`
- Consider adding time-based exits

## Comparison with Original AlphaTrend

| Feature | Original | Enhanced Version |
|---------|----------|------------------|
| **Trend Detection** | Basic AlphaTrend | Adaptive AlphaTrend with volatility adjustment |
| **Volume Filter** | None | Configurable short/long MA with alignment window |
| **Exit Logic** | Simple crossover | EMA-50 with grace period and momentum protection |
| **Position Sizing** | Fixed | Risk-based with currency conversion |
| **Performance** | O(n²) recalculation | O(n) vectorized pre-calculation |
| **Stop Loss** | Basic | ATR-based adaptive stops |
| **Multi-Currency** | No | Full FX conversion support |

## Further Reading

- **EXCEL_REPORTING_GUIDE.md** - Understanding backtest metrics and reports
- **CURRENCY_CONVERSION_README.md** - Multi-currency trading support
- **DATA_LEAKAGE_AUDIT.md** - How the framework prevents look-ahead bias
- **ARCHITECTURE.md** - Framework design and extension points

## Contributing

To extend this strategy:
1. Inherit from `AlphaTrendStrategy`
2. Override `generate_signal()` for custom entry/exit logic
3. Add new parameters to `__init__()` and pass to `super()`
4. Update `prepare_data()` if adding new indicators

Example:
```python
class CustomAlphaTrend(AlphaTrendStrategy):
    def __init__(self, my_param=10, **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param

    def generate_signal(self, context):
        # Your custom logic
        base_signal = super().generate_signal(context)
        # Add your modifications
        return modified_signal
```

## License

This strategy is part of the BackTestingFramework. See repository root for license information.

## Credits

- **Original AlphaTrend Indicator:** KivancOzbilgic
- **Enhanced Implementation:** BackTestingFramework contributors
- **Performance Optimizations:** Vectorization and pre-calculation techniques

---

*Last Updated: December 2025*
