"""
AlphaTrend Enhanced Long-Only Strategy

Based on AlphaTrend indicator by KivancOzbilgic with enhancements:
- Adaptive ATR multiplier based on volatility
- Dynamic MFI thresholds using percentile analysis (MFI read from raw data)
- Volume filter with alignment windows
- SMA-based exits with grace period and momentum protection
- Risk-based position sizing

Entry: AlphaTrend buy signal + volume condition within alignment window
Exit: Price closes below SMA-50 (with protections) or stop loss hit
Stop Loss: Percentage-based stop loss (x% below entry price)

PERFORMANCE OPTIMIZED:
- Standard indicators (ATR, MFI, SMA) read directly from raw data
- Custom AlphaTrend signals calculated using vectorized operations
- O(n) complexity instead of O(n²)
- Numba JIT compilation for state-dependent loops (5-20x speedup)

RAW DATA REQUIREMENTS (from Alpha Vantage):
- atr_14_atr: Average True Range (14-period)
- sma_50_sma: Simple Moving Average (50-period) for exits
- mfi_14_mfi: Money Flow Index (14-period) for momentum

All standard indicators are read from raw data - no calculations performed.
If an indicator is missing, a clear error will be raised.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal

# Try to import Numba for JIT compilation
# Falls back to pure Python if not available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator if Numba is not available
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator


# ============================================================================
# NUMBA JIT-COMPILED FUNCTIONS
# These functions are compiled to machine code for significant speedup.
# They maintain identical calculations to the pure Python versions.
# ============================================================================

@njit(cache=True)
def _alphatrend_numba(up_band, down_band, momentum_bullish):
    """
    JIT-compiled AlphaTrend calculation.

    This is mathematically identical to the pure Python version but
    runs 5-20x faster due to machine code compilation.

    Args:
        up_band: Upper band values (numpy array)
        down_band: Lower band values (numpy array)
        momentum_bullish: Boolean array of momentum bullish flags

    Returns:
        AlphaTrend values as numpy array
    """
    n = len(up_band)
    alphatrend = np.zeros(n)

    # Initialize first value
    alphatrend[0] = up_band[0]

    # State-dependent loop - compiled to efficient machine code
    for i in range(1, n):
        if momentum_bullish[i]:
            # Uptrend: AlphaTrend can only move up or stay same
            if up_band[i] < alphatrend[i-1]:
                alphatrend[i] = max(alphatrend[i-1], up_band[i])
            else:
                alphatrend[i] = up_band[i]
        else:
            # Downtrend: AlphaTrend can only move down or stay same
            if down_band[i] > alphatrend[i-1]:
                alphatrend[i] = min(alphatrend[i-1], down_band[i])
            else:
                alphatrend[i] = down_band[i]

    return alphatrend


@njit(cache=True)
def _filter_signals_numba(cross_up, cross_down):
    """
    JIT-compiled signal filtering for alternating buy/sell signals.

    Ensures we only get alternating signals (buy, sell, buy, sell, ...)
    by filtering out consecutive signals of the same type.

    Args:
        cross_up: Boolean array of cross-up signals
        cross_down: Boolean array of cross-down signals

    Returns:
        Tuple of (filtered_buy, filtered_sell) as boolean numpy arrays
    """
    n = len(cross_up)
    filtered_buy = np.zeros(n, dtype=np.bool_)
    filtered_sell = np.zeros(n, dtype=np.bool_)

    # 0 = no signal yet, 1 = last was buy, 2 = last was sell
    last_signal = 0

    for i in range(n):
        if cross_up[i] and last_signal != 1:
            filtered_buy[i] = True
            last_signal = 1
        elif cross_down[i] and last_signal != 2:
            filtered_sell[i] = True
            last_signal = 2

    return filtered_buy, filtered_sell


class AlphaTrendStrategy(BaseStrategy):
    """
    AlphaTrend Enhanced Strategy with volume filter and risk management.

    RAW DATA INDICATORS (read from CSV, no calculations):
        - atr_14_atr: Average True Range (14-period) - for bands and stop loss
        - sma_50_sma: Simple Moving Average (50-period) - for exits
        - mfi_14_mfi: Money Flow Index (14-period) - for momentum detection

    CUSTOM CALCULATIONS (strategy-specific, computed in prepare_data):
        - AlphaTrend bands using ATR from raw data
        - AlphaTrend signal generation
        - Volume MA comparison

    Parameters:
        volume_short_ma: Volume short MA period (default: 4)
        volume_long_ma: Volume long MA period (default: 30)
        volume_alignment_window: Bars to wait for volume condition after signal (default: 14)
        stop_loss_percent: Percentage below current price for stop loss (default: 0)
        atr_stop_loss_multiple: Multiple of ATR for stop loss (default: 2.5)
        grace_period_bars: Bars to ignore SMA exit after entry (default: 14)
        momentum_gain_pct: % gain to ignore SMA exit (default: 2.0)
        momentum_lookback: Bars for momentum calculation (default: 7)
        risk_percent: Percent of equity to risk per trade (default: 2.0)

    Static Parameters (not configurable via GUI):
        atr_multiplier: Base multiplier for ATR bands (fixed: 1.0)
        source: Price source for calculations (fixed: 'close')
        smoothing_length: EMA smoothing period for AlphaTrend (fixed: 3)
        percentile_period: Lookback for dynamic MFI thresholds (fixed: 100)
    """

    def __init__(self,
                 volume_short_ma: int = 4,
                 volume_long_ma: int = 30,
                 volume_alignment_window: int = 14,
                 stop_loss_percent: float = 0.0,
                 atr_stop_loss_multiple: float = 2.5,
                 grace_period_bars: int = 14,
                 momentum_gain_pct: float = 2.0,
                 momentum_lookback: int = 7,
                 risk_percent: float = 2.0):
        """Initialize AlphaTrend strategy with parameters."""
        super().__init__(
            volume_short_ma=volume_short_ma,
            volume_long_ma=volume_long_ma,
            volume_alignment_window=volume_alignment_window,
            stop_loss_percent=stop_loss_percent,
            atr_stop_loss_multiple=atr_stop_loss_multiple,
            grace_period_bars=grace_period_bars,
            momentum_gain_pct=momentum_gain_pct,
            momentum_lookback=momentum_lookback,
            risk_percent=risk_percent
        )

        # Static parameters (not configurable via GUI)
        self.atr_multiplier = 1.0
        self.source = 'close'
        self.smoothing_length = 3
        self.percentile_period = 100

        # Store configurable parameters as instance variables
        self.volume_short_ma = volume_short_ma
        self.volume_long_ma = volume_long_ma
        self.volume_alignment_window = volume_alignment_window
        self.stop_loss_percent = stop_loss_percent
        self.atr_stop_loss_multiple = atr_stop_loss_multiple
        self.grace_period_bars = grace_period_bars
        self.momentum_gain_pct = momentum_gain_pct
        self.momentum_lookback = momentum_lookback
        self.risk_percent = risk_percent

        # Track bars since entry for grace period and momentum (per-symbol for portfolio mode)
        self._bars_since_entry: Dict[str, int] = {}
        self._entry_bar_open: Dict[str, float] = {}

        # Track the most recent signal bar index (per-symbol for portfolio mode)
        self._signal_bar_idx: Dict[str, int] = {}

    def required_columns(self) -> List[str]:
        """
        Required columns from CSV data (all indicators read from raw data).

        Returns list of columns that MUST exist in the raw data.
        If any column is missing, a MissingColumnError will be raised.
        """
        return [
            # OHLCV data
            'date', 'open', 'high', 'low', 'close', 'volume',
            # Indicators from Alpha Vantage raw data
            'atr_14_atr',   # Average True Range - for bands and stop loss
            'sma_50_sma',   # Simple Moving Average - for exit signals
            'mfi_14_mfi',   # Money Flow Index - for momentum detection
        ]

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate custom AlphaTrend indicators ONCE before backtesting.

        RAW DATA INDICATORS (read directly, no calculations):
        - atr_14_atr: Average True Range from Alpha Vantage
        - sma_50_sma: Simple Moving Average from Alpha Vantage
        - mfi_14_mfi: Money Flow Index from Alpha Vantage

        CUSTOM CALCULATIONS (strategy-specific):
        - Adaptive coefficient based on volatility ratio
        - AlphaTrend bands using raw ATR
        - AlphaTrend signal generation
        - Volume filter

        Args:
            data: Raw OHLCV data with pre-calculated indicators from Alpha Vantage

        Returns:
            Data with AlphaTrend-specific indicators added as columns

        Raises:
            ValueError: If required indicators are missing from raw data
        """
        df = data.copy()

        # Verify required indicators exist in raw data
        required_indicators = ['atr_14_atr', 'sma_50_sma', 'mfi_14_mfi']
        missing = [col for col in required_indicators if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required indicators in raw data: {missing}\n"
                f"Please re-run data collection with these indicators enabled.\n"
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        # ==== ADAPTIVE COEFFICIENT ====
        # Calculate long-term ATR average for volatility ratio using raw ATR
        df['atr_ema_long'] = df['atr_14_atr'].ewm(span=14 * 3, adjust=False).mean()
        df['volatility_ratio'] = df['atr_14_atr'] / df['atr_ema_long']
        df['adaptive_coeff'] = self.atr_multiplier * df['volatility_ratio']

        # ==== ALPHATREND BANDS ====
        # Use atr_14_atr from raw data for band calculations
        df['up_band'] = df['low'] - df['atr_14_atr'] * df['adaptive_coeff']
        df['down_band'] = df['high'] + df['atr_14_atr'] * df['adaptive_coeff']

        # ==== MONEY FLOW INDEX ====
        # MFI is now read directly from raw data (mfi_14_mfi)
        # No calculation needed - just use the raw data column
        df['mfi'] = df['mfi_14_mfi']

        # ==== DYNAMIC MFI THRESHOLDS ====
        # Vectorized percentile calculations on raw MFI data
        df['mfi_upper'] = df['mfi'].rolling(window=self.percentile_period).quantile(0.70)
        df['mfi_lower'] = df['mfi'].rolling(window=self.percentile_period).quantile(0.30)
        df['mfi_threshold'] = (df['mfi_upper'] + df['mfi_lower']) / 2

        # Momentum bullish condition
        df['momentum_bullish'] = df['mfi'] >= df['mfi_threshold']

        # ==== ALPHATREND CALCULATION (VECTORIZED) ====
        df['alphatrend'] = self._calculate_alphatrend_vectorized(
            df['up_band'].values,
            df['down_band'].values,
            df['momentum_bullish'].values
        )

        # ==== SMOOTHED ALPHATREND ====
        df['smooth_at'] = df['alphatrend'].ewm(span=self.smoothing_length, adjust=False).mean()

        # ==== ALPHATREND SIGNALS ====
        # Vectorized crossover detection
        df['at_cross_up'] = (df['alphatrend'] > df['smooth_at']) & (df['alphatrend'].shift(1) <= df['smooth_at'].shift(1))
        df['at_cross_down'] = (df['alphatrend'] < df['smooth_at']) & (df['alphatrend'].shift(1) >= df['smooth_at'].shift(1))

        # ==== FILTER FOR ALTERNATING SIGNALS (VECTORIZED) ====
        df['filtered_buy'], df['filtered_sell'] = self._filter_alternating_signals_vectorized(
            df['at_cross_up'].values,
            df['at_cross_down'].values
        )

        # ==== VOLUME FILTER ====
        # Vectorized volume MA calculations
        df['vol_short_ma'] = df['volume'].rolling(window=self.volume_short_ma).mean()
        df['vol_long_ma'] = df['volume'].rolling(window=self.volume_long_ma).mean()
        df['volume_condition'] = df['vol_short_ma'] > df['vol_long_ma']

        return df

    @staticmethod
    def _calculate_alphatrend_vectorized(
        up_band: np.ndarray,
        down_band: np.ndarray,
        momentum_bullish: np.ndarray
    ) -> pd.Series:
        """
        Optimized AlphaTrend calculation using Numba JIT compilation.

        PERFORMANCE: Uses Numba-compiled _alphatrend_numba() for 5-20x speedup.
        Falls back to pure Python if Numba is not available.

        The calculation is mathematically identical regardless of which
        implementation is used - only performance differs.

        Args:
            up_band: Upper band values
            down_band: Lower band values
            momentum_bullish: Momentum bullish flags

        Returns:
            AlphaTrend values as Series
        """
        # Use the Numba-compiled version for maximum performance
        # This is 5-20x faster than pure Python loops
        alphatrend = _alphatrend_numba(
            up_band.astype(np.float64),
            down_band.astype(np.float64),
            momentum_bullish.astype(np.bool_)
        )
        return pd.Series(alphatrend)

    @staticmethod
    def _filter_alternating_signals_vectorized(
        cross_up: np.ndarray,
        cross_down: np.ndarray
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Filter signals to ensure alternating buy/sell using Numba JIT compilation.

        PERFORMANCE: Uses Numba-compiled _filter_signals_numba() for 5-20x speedup.
        Falls back to pure Python if Numba is not available.

        The calculation is mathematically identical regardless of which
        implementation is used - only performance differs.

        Args:
            cross_up: Cross up signals
            cross_down: Cross down signals

        Returns:
            Tuple of (filtered_buy, filtered_sell) as Series
        """
        # Use the Numba-compiled version for maximum performance
        filtered_buy, filtered_sell = _filter_signals_numba(
            cross_up.astype(np.bool_),
            cross_down.astype(np.bool_)
        )
        return pd.Series(filtered_buy), pd.Series(filtered_sell)

    def _get_indicators(self, context: StrategyContext) -> Optional[dict]:
        """
        Get pre-calculated indicators for current bar.

        PERFORMANCE: Simply reads values from pre-calculated columns.
        No computation happens here - all indicators were calculated in prepare_data()
        or read from raw data.

        Raw data indicators used:
        - atr_14_atr: Average True Range
        - sma_50_sma: Simple Moving Average (for exits)
        - mfi_14_mfi: Money Flow Index

        Returns:
            Dictionary with indicator values at current bar, or None if insufficient data
        """
        # Check if we have enough historical data
        if context.current_index < max(14, self.percentile_period):
            return None

        # Get current bar (all indicators are pre-calculated columns)
        current_bar = context.current_bar

        # Check for NaN values in critical raw data indicators
        if pd.isna(current_bar.get('atr_14_atr')) or pd.isna(current_bar.get('sma_50_sma')):
            return None

        # Return indicator values (mix of raw data and custom calculations)
        return {
            'atr_14_atr': current_bar['atr_14_atr'],  # Read from raw data
            'alphatrend': current_bar['alphatrend'],   # Custom calculation
            'smooth_at': current_bar['smooth_at'],     # Custom calculation
            'filtered_buy': current_bar['filtered_buy'],
            'filtered_sell': current_bar['filtered_sell'],
            'volume_condition': current_bar['volume_condition'],
            'sma_50': current_bar['sma_50_sma'],       # Read from raw data
            'mfi': current_bar['mfi']                  # Read from raw data (mfi_14_mfi)
        }

    def _has_active_signal(self, context: StrategyContext) -> bool:
        """
        Check if we have an active AlphaTrend signal within the volume alignment window.

        Returns:
            True if signal is active (on or within volume_alignment_window bars after signal)
        """
        symbol = context.symbol
        signal_bar_idx = self._signal_bar_idx.get(symbol, -1)
        if signal_bar_idx < 0:
            return False

        bars_since_signal = context.current_index - signal_bar_idx
        # CRITICAL: Ensure we're on or after the signal bar (bars_since_signal >= 0)
        # and within the forward window (bars_since_signal <= window)
        return 0 <= bars_since_signal <= self.volume_alignment_window

    def _check_volume_since_signal(self, context: StrategyContext) -> bool:
        """
        Check if volume condition was met within the correct time window.

        Logic:
        - ON signal bar: Check lookback window (signal - window) to signal bar
        - AFTER signal bar: Check from signal bar to current bar only

        This ensures trades can only occur on or after the signal bar, never before.

        PERFORMANCE: Uses pre-calculated volume_condition column.

        Returns:
            True if volume condition was met within the correct window
        """
        symbol = context.symbol
        signal_bar_idx = self._signal_bar_idx.get(symbol, -1)
        if signal_bar_idx < 0:
            return False

        current_idx = context.current_index

        # Determine the correct time window based on where we are relative to signal
        if current_idx == signal_bar_idx:
            # ON signal bar: check lookback window for volume confirmation
            # This allows entering on signal bar if volume was confirmed before/on signal
            vol_start = max(0, signal_bar_idx - self.volume_alignment_window)
            vol_end = signal_bar_idx + 1
        elif current_idx > signal_bar_idx:
            # AFTER signal bar: check from signal to current bar only
            # This ensures we only look forward from signal, never before it
            vol_start = signal_bar_idx
            vol_end = current_idx + 1
        else:
            # BEFORE signal bar: should never happen due to _has_active_signal check
            # but return False as safety measure
            return False

        # Check if volume condition was met in the determined window
        for i in range(vol_start, vol_end):
            if context.data.iloc[i]['volume_condition']:
                return True

        return False

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on AlphaTrend logic.

        PERFORMANCE: All indicators are pre-calculated, just reads values.
        """
        # Get pre-calculated indicators
        indicators = self._get_indicators(context)

        if not indicators:
            return Signal.hold()

        current_price = context.current_price
        current_idx = context.current_index

        # Get symbol for per-symbol state tracking
        symbol = context.symbol

        # Update signal tracking when a new AlphaTrend buy signal appears
        if indicators['filtered_buy']:
            self._signal_bar_idx[symbol] = current_idx

        # Reset signal when we get a sell signal (prevents re-entry on old buy signals)
        if indicators['filtered_sell']:
            self._signal_bar_idx[symbol] = -1

        # Entry logic
        if not context.has_position:
            # Reset bars counter when not in position
            self._bars_since_entry[symbol] = 0
            self._entry_bar_open[symbol] = None

            # Check if we have an active AlphaTrend signal
            at_signal_active = self._has_active_signal(context)

            # Only proceed if we have an active signal
            if at_signal_active:
                # Check if volume condition has been met since signal appeared
                vol_aligned = self._check_volume_since_signal(context)

                # Entry condition: AlphaTrend signal is active AND volume has aligned since signal
                # We only enter when both conditions are satisfied
                if vol_aligned:
                    # Calculate stop loss: use ATR-based if atr_stop_loss_multiple > 0, else percentage-based
                    if self.atr_stop_loss_multiple > 0:
                        stop_loss = current_price - (indicators['atr_14_atr'] * self.atr_stop_loss_multiple)
                    else:
                        stop_loss = current_price * (1 - self.stop_loss_percent / 100)

                    # Store entry bar open for momentum calculation
                    self._entry_bar_open[symbol] = context.current_bar['open']

                    # Reset signal to prevent re-entry on same signal
                    self._signal_bar_idx[symbol] = -1

                    return Signal.buy(
                        size=1.0,  # Will be overridden by position_size() method
                        stop_loss=stop_loss,
                        reason=f"AlphaTrend signal + Volume aligned"
                    )

        # Exit logic
        else:
            # Increment bars since entry
            self._bars_since_entry[symbol] = self._bars_since_entry.get(symbol, 0) + 1

            # Check SMA-50 exit condition (read from raw data as sma_50_sma)
            sma_exit = current_price < indicators['sma_50']

            # Grace period protection
            bars_since = self._bars_since_entry.get(symbol, 0)
            in_grace_period = bars_since <= self.grace_period_bars

            # Momentum protection
            has_momentum = False
            entry_bar_open = self._entry_bar_open.get(symbol)
            if entry_bar_open is not None and self.momentum_lookback <= context.current_index:
                lookback_idx = context.current_index - self.momentum_lookback
                if lookback_idx >= 0:
                    lookback_open = context.data.iloc[lookback_idx]['open']
                    price_gain_pct = ((current_price - lookback_open) / lookback_open) * 100
                    has_momentum = price_gain_pct >= self.momentum_gain_pct

            # Exit only if SMA condition met and NOT protected
            if sma_exit and not in_grace_period and not has_momentum:
                return Signal.sell(reason=f"Price < SMA(50)")

        return Signal.hold()

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size using risk-based sizing with currency conversion.

        Formula: Position size = (Equity * Risk%) / (Stop Distance in Base Currency)

        Works with both percentage-based and ATR-based stop losses.

        Note: This returns the IDEAL position size based on total equity.
        The portfolio engine handles capital availability checks and may
        reduce the position size or use vulnerability swaps if needed.
        """
        if signal.stop_loss is None:
            # Fallback to default sizing if no stop loss
            return super().position_size(context, signal)

        equity = context.total_equity  # In base currency (e.g., GBP)
        risk_amount = equity * (self.risk_percent / 100)  # In base currency

        # Calculate stop distance from the actual stop loss price
        # This works for both percentage-based and ATR-based stop losses
        stop_distance = context.current_price - signal.stop_loss  # In security currency

        if stop_distance <= 0:
            # Invalid stop distance, fallback to default
            return super().position_size(context, signal)

        # Convert stop distance to base currency
        stop_distance_base = stop_distance * context.fx_rate

        # Calculate shares based on risk in base currency
        # Capital availability is checked by the portfolio engine, not here
        shares = risk_amount / stop_distance_base

        return shares
