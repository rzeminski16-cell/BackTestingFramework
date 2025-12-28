"""
AlphaTrend Enhanced Long-Only Strategy

Based on AlphaTrend indicator by KivancOzbilgic with enhancements:
- Adaptive ATR multiplier based on volatility
- Dynamic MFI thresholds using percentile analysis
- Volume filter with alignment windows
- EMA-based exits with grace period and momentum protection
- Risk-based position sizing

Entry: AlphaTrend buy signal + volume condition within alignment window
Exit: Price closes below EMA-50 (with protections) or stop loss hit
Stop Loss: ATR-based or percentage-based stop loss

PERFORMANCE OPTIMIZED:
- Indicators pre-calculated using vectorized operations (10-100x speedup)
- No repeated calculations during backtest
- O(n) complexity instead of O(n²)
- Numba JIT compilation for state-dependent loops (5-20x speedup)

NOTE: All standard indicators (atr_14, ema_50) are read from raw data.
AlphaTrend signals are calculated (approved exception).
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection

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

    STRATEGY STRUCTURE:
    - Trade Direction: LONG only
    - Fundamental Rules: None (always pass)
    - Entry Rules: AlphaTrend buy signal + volume alignment
    - Initial Stop Loss: ATR-based (default) or percentage-based
    - Position Sizing: Risk-based sizing
    - Trailing Stop: None
    - Partial Exit: None
    - Full Exit Rules: Price below EMA-50 (with grace period and momentum protection)
    - Pyramiding: None

    Standard Indicators (read from raw data, static):
        - atr_14: Average True Range (14-period, static)
        - ema_50: Exponential Moving Average (50-period, static, used for exits)

    Custom Parameters (strategy-specific calculations):
        volume_short_ma: Volume short MA period (default: 4)
        volume_long_ma: Volume long MA period (default: 30)
        volume_alignment_window: Bars to wait for volume condition after signal (default: 14)
        stop_loss_percent: Percentage below current price for stop loss (default: 0)
        atr_stop_loss_multiple: Multiple of ATR for stop loss (default: 2.5)
        grace_period_bars: Bars to ignore EMA exit after entry (default: 14)
        momentum_gain_pct: % gain to ignore EMA exit (default: 2.0)
        momentum_lookback: Bars for momentum calculation (default: 7)
        risk_percent: Percent of equity to risk per trade (default: 2.0)

    Static Parameters (not configurable via GUI):
        atr_multiplier: Base multiplier for ATR bands (fixed: 1.0)
        source: Price source for calculations (fixed: 'close')
        smoothing_length: EMA smoothing period for AlphaTrend (fixed: 3)
        percentile_period: Lookback for dynamic thresholds (fixed: 100)
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
        self._entry_bar_idx: Dict[str, int] = {}

        # Track the most recent signal bar index (per-symbol for portfolio mode)
        self._signal_bar_idx: Dict[str, int] = {}

    @property
    def trade_direction(self) -> TradeDirection:
        """This is a LONG-only strategy."""
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        """Required columns from CSV data (including pre-calculated indicators)."""
        return [
            'date', 'open', 'high', 'low', 'close', 'volume',
            'atr_14',  # Pre-calculated ATR from raw data
            'ema_50'   # Pre-calculated EMA from raw data
        ]

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate custom AlphaTrend indicators ONCE before backtesting.

        Standard indicators (atr_14, ema_50) are read from raw data.
        Custom indicators specific to AlphaTrend are calculated here.

        APPROVED EXCEPTION: AlphaTrend signal calculations are allowed
        because they are strategy-specific and not standard indicators.

        Args:
            data: Raw OHLCV data with pre-calculated standard indicators

        Returns:
            Data with all AlphaTrend-specific indicators added as columns
        """
        df = data.copy()

        # ==== ADAPTIVE COEFFICIENT ====
        # Calculate long-term ATR average for volatility ratio
        df['atr_ema_long'] = df['atr_14'].ewm(span=14 * 3, adjust=False).mean()
        df['volatility_ratio'] = df['atr_14'] / df['atr_ema_long']
        df['adaptive_coeff'] = self.atr_multiplier * df['volatility_ratio']

        # ==== ALPHATREND BANDS ====
        # Use atr_14 from raw data for band calculations
        df['up_band'] = df['low'] - df['atr_14'] * df['adaptive_coeff']
        df['down_band'] = df['high'] + df['atr_14'] * df['adaptive_coeff']

        # ==== MONEY FLOW INDEX (MFI) ====
        # Vectorized MFI calculation (14-period to match atr_14)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['raw_money_flow'] = df['typical_price'] * df['volume']

        # Determine positive and negative money flow
        df['price_change'] = df['typical_price'].diff()
        df['positive_flow'] = np.where(df['price_change'] > 0, df['raw_money_flow'], 0)
        df['negative_flow'] = np.where(df['price_change'] < 0, df['raw_money_flow'], 0)

        # Sum over 14-period (vectorized)
        positive_mf = df['positive_flow'].rolling(window=14).sum()
        negative_mf = df['negative_flow'].rolling(window=14).sum()

        # MFI calculation with division by zero protection
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))

        # ==== DYNAMIC MFI THRESHOLDS ====
        # Vectorized percentile calculations
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

        Args:
            up_band: Upper band values
            down_band: Lower band values
            momentum_bullish: Momentum bullish flags

        Returns:
            AlphaTrend values as Series
        """
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

        Args:
            cross_up: Cross up signals
            cross_down: Cross down signals

        Returns:
            Tuple of (filtered_buy, filtered_sell) as Series
        """
        filtered_buy, filtered_sell = _filter_signals_numba(
            cross_up.astype(np.bool_),
            cross_down.astype(np.bool_)
        )
        return pd.Series(filtered_buy), pd.Series(filtered_sell)

    def _get_indicators(self, context: StrategyContext) -> Optional[dict]:
        """
        Get pre-calculated indicators for current bar.

        Returns:
            Dictionary with indicator values at current bar, or None if insufficient data
        """
        if context.current_index < max(14, self.percentile_period):
            return None

        current_bar = context.current_bar

        if pd.isna(current_bar.get('atr_14')) or pd.isna(current_bar.get('ema_50')):
            return None

        return {
            'atr_14': current_bar['atr_14'],
            'alphatrend': current_bar['alphatrend'],
            'smooth_at': current_bar['smooth_at'],
            'filtered_buy': current_bar['filtered_buy'],
            'filtered_sell': current_bar['filtered_sell'],
            'volume_condition': current_bar['volume_condition'],
            'ema_50': current_bar['ema_50'],
            'mfi': current_bar['mfi']
        }

    def _has_active_signal(self, context: StrategyContext) -> bool:
        """Check if we have an active AlphaTrend signal within the volume alignment window."""
        symbol = context.symbol
        signal_bar_idx = self._signal_bar_idx.get(symbol, -1)
        if signal_bar_idx < 0:
            return False

        bars_since_signal = context.current_index - signal_bar_idx
        return 0 <= bars_since_signal <= self.volume_alignment_window

    def _check_volume_since_signal(self, context: StrategyContext) -> bool:
        """Check if volume condition was met within the correct time window."""
        symbol = context.symbol
        signal_bar_idx = self._signal_bar_idx.get(symbol, -1)
        if signal_bar_idx < 0:
            return False

        current_idx = context.current_index

        if current_idx == signal_bar_idx:
            vol_start = max(0, signal_bar_idx - self.volume_alignment_window)
            vol_end = signal_bar_idx + 1
        elif current_idx > signal_bar_idx:
            vol_start = signal_bar_idx
            vol_end = current_idx + 1
        else:
            return False

        for i in range(vol_start, vol_end):
            if context.data.iloc[i]['volume_condition']:
                return True

        return False

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """
        Calculate initial stop loss price.

        Uses ATR-based stop if atr_stop_loss_multiple > 0,
        otherwise uses percentage-based stop.

        Args:
            context: Current market context

        Returns:
            Stop loss price
        """
        current_price = context.current_price
        atr = context.get_indicator_value('atr_14')

        if self.atr_stop_loss_multiple > 0 and atr is not None:
            return current_price - (atr * self.atr_stop_loss_multiple)
        else:
            return current_price * (1 - self.stop_loss_percent / 100)

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate entry signal based on AlphaTrend logic.

        Entry conditions:
        1. AlphaTrend buy signal is active (within alignment window)
        2. Volume condition has been met since signal

        Args:
            context: Current market context

        Returns:
            Signal.buy() if entry conditions met, None otherwise
        """
        indicators = self._get_indicators(context)
        if not indicators:
            return None

        symbol = context.symbol
        current_idx = context.current_index

        # Update signal tracking when a new AlphaTrend buy signal appears
        if indicators['filtered_buy']:
            self._signal_bar_idx[symbol] = current_idx

        # Reset signal when we get a sell signal
        if indicators['filtered_sell']:
            self._signal_bar_idx[symbol] = -1

        # Reset bars counter when not in position
        self._bars_since_entry[symbol] = 0
        self._entry_bar_open[symbol] = None

        # Check if we have an active AlphaTrend signal
        at_signal_active = self._has_active_signal(context)

        if at_signal_active:
            vol_aligned = self._check_volume_since_signal(context)

            if vol_aligned:
                # Store entry bar info
                self._entry_bar_open[symbol] = context.current_bar['open']
                self._entry_bar_idx[symbol] = current_idx

                # Reset signal to prevent re-entry on same signal
                self._signal_bar_idx[symbol] = -1

                return Signal.buy(
                    size=1.0,
                    stop_loss=self.calculate_initial_stop_loss(context),
                    reason="AlphaTrend signal + Volume aligned",
                    direction=self.trade_direction
                )

        return None

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate exit signal based on EMA-50 exit rule.

        Exit conditions:
        - Price closes below EMA-50
        - NOT in grace period (bars since entry <= grace_period_bars)
        - NOT protected by momentum (price gain >= momentum_gain_pct)

        Args:
            context: Current market context

        Returns:
            Signal.sell() if exit conditions met, None otherwise
        """
        indicators = self._get_indicators(context)
        if not indicators:
            return None

        symbol = context.symbol
        current_price = context.current_price

        # Update signal tracking
        if indicators['filtered_buy']:
            self._signal_bar_idx[symbol] = context.current_index
        if indicators['filtered_sell']:
            self._signal_bar_idx[symbol] = -1

        # Increment bars since entry
        self._bars_since_entry[symbol] = self._bars_since_entry.get(symbol, 0) + 1

        # Check EMA-50 exit condition
        ema_exit = current_price < indicators['ema_50']

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

        # Exit only if EMA condition met and NOT protected
        if ema_exit and not in_grace_period and not has_momentum:
            return Signal.sell(reason="Price < EMA(50)")

        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size using risk-based sizing with currency conversion.

        Formula: Position size = (Equity * Risk%) / (Stop Distance in Base Currency)

        Args:
            context: Current market context
            signal: BUY signal with stop_loss set

        Returns:
            Number of shares to buy
        """
        if signal.stop_loss is None:
            # Fallback to capital-based sizing
            capital_to_use = context.available_capital * signal.size
            return capital_to_use / context.current_price

        equity = context.total_equity
        risk_amount = equity * (self.risk_percent / 100)

        stop_distance = context.current_price - signal.stop_loss

        if stop_distance <= 0:
            capital_to_use = context.available_capital * signal.size
            return capital_to_use / context.current_price

        # Convert stop distance to base currency
        stop_distance_base = stop_distance * context.fx_rate

        shares = risk_amount / stop_distance_base

        return shares
