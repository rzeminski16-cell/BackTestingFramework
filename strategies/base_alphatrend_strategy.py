"""
Base Alpha Trend Strategy

A simplified AlphaTrend-based strategy with stop-loss-only exits.

Entry: AlphaTrend indicator buy signal (crossover of AlphaTrend above smoothed line)
Stop Loss: ATR14 x atr_multiplier
Position Sizing: Risk-based (lose risk_percent% of equity at stop loss)

No advanced features: No trailing stop, no partial exits, no pyramiding.

ALPHATREND INDICATOR CALCULATION:
================================
The AlphaTrend indicator creates dynamic bands around price using ATR and adapts
based on momentum (MFI). It generates buy signals when price momentum shifts bullish.

1. Adaptive Coefficient:
   - Calculate long-term ATR average (42-period EMA of ATR14)
   - Volatility ratio = ATR14 / long-term ATR average
   - Adaptive coefficient = atr_multiplier * volatility_ratio

2. Upper and Lower Bands:
   - Upper band (support in uptrends) = Low - ATR14 * adaptive_coefficient
   - Lower band (resistance in downtrends) = High + ATR14 * adaptive_coefficient

3. Momentum Detection (using MFI):
   - Calculate dynamic MFI thresholds using rolling percentiles (70th and 30th)
   - MFI threshold = average of upper and lower percentiles
   - Momentum is bullish when MFI >= threshold

4. AlphaTrend Line (state-dependent):
   - In uptrend (momentum bullish): Line ratchets UP (can only go up or stay same)
   - In downtrend (momentum bearish): Line ratchets DOWN (can only go down or stay same)

5. Buy Signal Generation:
   - Buy signal occurs when AlphaTrend crosses ABOVE the smoothed AlphaTrend line
   - Signals are filtered to ensure alternating buy/sell sequence

STRATEGY STRUCTURE:
- Trade Direction: LONG only
- Fundamental Rules: None (always pass)
- Entry Rules: AlphaTrend buy signal
- Initial Stop Loss: ATR-based (ATR14 x atr_multiplier)
- Position Sizing: Risk-based sizing (risk_percent% of equity at stop loss)
- Trailing Stop: None
- Partial Exit: None
- Full Exit Rules: Stop loss only
- Pyramiding: None

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - atr_14_atr: Average True Range (14-period) - used for bands and stop loss
    - mfi_14_mfi: Money Flow Index (14-period) - used for momentum detection

OPTIMIZABLE PARAMETERS:
    - atr_multiplier: ATR multiplier for stop loss (default: 2.0)
    - risk_percent: Percent of equity to risk per trade (default: 2.0)
    - alpha_atr_multiplier: Base multiplier for ATR bands in AlphaTrend (default: 1.0)
    - smoothing_length: EMA period for AlphaTrend smoothing (default: 3)
    - percentile_period: Lookback period for dynamic MFI thresholds (default: 100)
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
# ============================================================================

@njit(cache=True)
def _alphatrend_numba(up_band, down_band, momentum_bullish):
    """
    JIT-compiled AlphaTrend calculation.

    The AlphaTrend line follows these rules:
    - In uptrend (momentum bullish): Can only move UP or stay same (ratchets up)
    - In downtrend (momentum bearish): Can only move DOWN or stay same (ratchets down)

    This creates a trend-following line that doesn't whipsaw during the trend.

    Args:
        up_band: Upper band values (numpy array) - support levels
        down_band: Lower band values (numpy array) - resistance levels
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


class BaseAlphaTrendStrategy(BaseStrategy):
    """
    Base AlphaTrend Strategy with stop-loss-only exits.

    A simplified version of AlphaTrend that uses:
    - AlphaTrend indicator for entry signals
    - ATR-based stop loss
    - Risk-based position sizing

    No advanced features like trailing stops, partial exits, or pyramiding.
    """

    def __init__(self,
                 # Stop loss parameters
                 atr_multiplier: float = 2.0,
                 # Position sizing
                 risk_percent: float = 2.0,
                 # AlphaTrend calculation parameters
                 alpha_atr_multiplier: float = 1.0,
                 smoothing_length: int = 3,
                 percentile_period: int = 100):
        """
        Initialize Base AlphaTrend strategy.

        Args:
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)
            risk_percent: Percent of equity to risk per trade (default: 2.0)
            alpha_atr_multiplier: Base multiplier for ATR bands in AlphaTrend (default: 1.0)
            smoothing_length: EMA period for AlphaTrend smoothing (default: 3)
            percentile_period: Lookback for dynamic MFI thresholds (default: 100)
        """
        super().__init__(
            atr_multiplier=atr_multiplier,
            risk_percent=risk_percent,
            alpha_atr_multiplier=alpha_atr_multiplier,
            smoothing_length=smoothing_length,
            percentile_period=percentile_period
        )

        # Store parameters
        self.atr_multiplier = atr_multiplier
        self.risk_percent = risk_percent
        self.alpha_atr_multiplier = alpha_atr_multiplier
        self.smoothing_length = smoothing_length
        self.percentile_period = percentile_period

    @property
    def trade_direction(self) -> TradeDirection:
        """This is a LONG-only strategy."""
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        """
        Required columns from CSV data.

        Returns list of columns that MUST exist in the raw data.
        """
        return [
            # OHLCV data
            'date', 'open', 'high', 'low', 'close', 'volume',
            'atr_14_atr',  # Pre-calculated ATR from raw data
            'mfi_14_mfi'   # Pre-calculated Money Flow Index from raw data
        ]

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate AlphaTrend indicators ONCE before backtesting.

        Args:
            data: Raw OHLCV data with pre-calculated indicators

        Returns:
            Data with AlphaTrend-specific indicators added as columns
        """
        df = data.copy()

        # ==== NORMALIZE COLUMN NAMES ====
        df['atr_14'] = df['atr_14_atr']
        df['mfi_14'] = df['mfi_14_mfi']

        # ==== ADAPTIVE COEFFICIENT ====
        # Calculate long-term ATR average for volatility ratio
        df['atr_ema_long'] = df['atr_14'].ewm(span=42, adjust=False).mean()
        df['volatility_ratio'] = df['atr_14'] / df['atr_ema_long']
        df['adaptive_coeff'] = self.alpha_atr_multiplier * df['volatility_ratio']

        # ==== ALPHATREND BANDS ====
        df['up_band'] = df['low'] - df['atr_14'] * df['adaptive_coeff']
        df['down_band'] = df['high'] + df['atr_14'] * df['adaptive_coeff']

        # ==== DYNAMIC MFI THRESHOLDS ====
        df['mfi_upper'] = df['mfi_14'].rolling(window=self.percentile_period).quantile(0.70)
        df['mfi_lower'] = df['mfi_14'].rolling(window=self.percentile_period).quantile(0.30)
        df['mfi_threshold'] = (df['mfi_upper'] + df['mfi_lower']) / 2

        # Momentum bullish condition
        df['momentum_bullish'] = df['mfi_14'] >= df['mfi_threshold']

        # ==== ALPHATREND CALCULATION ====
        df['alphatrend'] = self._calculate_alphatrend_vectorized(
            df['up_band'].values,
            df['down_band'].values,
            df['momentum_bullish'].values
        )

        # ==== SMOOTHED ALPHATREND ====
        df['smooth_at'] = df['alphatrend'].ewm(span=self.smoothing_length, adjust=False).mean()

        # ==== ALPHATREND SIGNALS ====
        # Buy signal: AlphaTrend crosses above smoothed line
        df['at_cross_up'] = (df['alphatrend'] > df['smooth_at']) & (df['alphatrend'].shift(1) <= df['smooth_at'].shift(1))
        df['at_cross_down'] = (df['alphatrend'] < df['smooth_at']) & (df['alphatrend'].shift(1) >= df['smooth_at'].shift(1))

        # ==== FILTER FOR ALTERNATING SIGNALS ====
        df['filtered_buy'], df['filtered_sell'] = self._filter_alternating_signals_vectorized(
            df['at_cross_up'].values,
            df['at_cross_down'].values
        )

        return df

    @staticmethod
    def _calculate_alphatrend_vectorized(
        up_band: np.ndarray,
        down_band: np.ndarray,
        momentum_bullish: np.ndarray
    ) -> pd.Series:
        """Optimized AlphaTrend calculation using Numba JIT compilation."""
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
        """Filter signals to ensure alternating buy/sell using Numba JIT compilation."""
        filtered_buy, filtered_sell = _filter_signals_numba(
            cross_up.astype(np.bool_),
            cross_down.astype(np.bool_)
        )
        return pd.Series(filtered_buy), pd.Series(filtered_sell)

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """
        Calculate initial stop loss price using ATR.

        Stop Loss = Entry Price - (ATR14 x atr_multiplier)

        Args:
            context: Current market context

        Returns:
            Stop loss price
        """
        current_price = context.current_price
        atr = context.get_indicator_value('atr_14')

        if atr is not None and atr > 0:
            return current_price - (atr * self.atr_multiplier)
        else:
            # Fallback: 5% stop loss if ATR not available
            return current_price * 0.95

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate entry signal based on AlphaTrend buy signal.

        Entry occurs when AlphaTrend crosses above its smoothed line,
        indicating a shift to bullish momentum.

        Args:
            context: Current market context

        Returns:
            Signal.buy() if AlphaTrend buy signal, None otherwise
        """
        # Check warmup period
        if context.current_index < self.percentile_period:
            return None

        # Get the filtered buy signal
        filtered_buy = context.get_indicator_value('filtered_buy')

        if filtered_buy:
            return Signal.buy(
                size=1.0,
                stop_loss=self.calculate_initial_stop_loss(context),
                reason="AlphaTrend buy signal",
                direction=self.trade_direction
            )

        return None

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate exit signal.

        This strategy relies on stop loss for exits. No additional exit
        conditions are used.

        Args:
            context: Current market context

        Returns:
            None (exit handled by stop loss only)
        """
        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size using risk-based sizing.

        Formula: Position size = (Equity * Risk%) / (Stop Distance in Base Currency)

        This ensures that if the stop loss is hit, we only lose risk_percent% of equity.

        Args:
            context: Current market context
            signal: BUY signal with stop_loss set

        Returns:
            Number of shares to buy
        """
        if signal.stop_loss is None:
            # Fallback to using 10% of capital
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        equity = context.total_equity
        risk_amount = equity * (self.risk_percent / 100)

        stop_distance = context.current_price - signal.stop_loss

        if stop_distance <= 0:
            # Invalid stop, fallback to 10% of capital
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        # Convert stop distance to base currency
        stop_distance_base = stop_distance * context.fx_rate

        shares = risk_amount / stop_distance_base

        return shares
