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
Stop Loss: ATR-based stop loss (using atr_14 from raw data)

PERFORMANCE OPTIMIZED:
- Indicators pre-calculated using vectorized operations (10-100x speedup)
- No repeated calculations during backtest
- O(n) complexity instead of O(n²)

NOTE: All standard indicators (atr_14, ema_50) are read from raw data.
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal


class AlphaTrendStrategy(BaseStrategy):
    """
    AlphaTrend Enhanced Strategy with volume filter and risk management.

    Standard Indicators (read from raw data, static):
        - atr_14: Average True Range (14-period, static)
        - ema_50: Exponential Moving Average (50-period, static, used for exits)

    Custom Parameters (strategy-specific calculations):
        atr_multiplier: Base multiplier for ATR bands (default: 1.0)
        source: Price source for calculations - 'close', 'open', 'high', 'low' (default: 'close')
        smoothing_length: EMA smoothing period for AlphaTrend (default: 3)
        percentile_period: Lookback for dynamic thresholds (default: 100)
        volume_short_ma: Volume short MA period (default: 4)
        volume_long_ma: Volume long MA period (default: 30)
        volume_alignment_window: Bars to check for volume condition (default: 14)
        signal_lookback: Bars to look back for AlphaTrend signal (default: 9)
        stop_atr_multiplier: ATR multiplier for stop loss (default: 2.5)
        grace_period_bars: Bars to ignore EMA exit after entry (default: 14)
        momentum_gain_pct: % gain to ignore EMA exit (default: 2.0)
        momentum_lookback: Bars for momentum calculation (default: 7)
        risk_percent: Percent of equity to risk per trade (default: 2.0)
    """

    def __init__(self,
                 atr_multiplier: float = 1.0,
                 source: str = 'close',
                 smoothing_length: int = 3,
                 percentile_period: int = 100,
                 volume_short_ma: int = 4,
                 volume_long_ma: int = 30,
                 volume_alignment_window: int = 14,
                 signal_lookback: int = 9,
                 stop_atr_multiplier: float = 2.5,
                 grace_period_bars: int = 14,
                 momentum_gain_pct: float = 2.0,
                 momentum_lookback: int = 7,
                 risk_percent: float = 2.0):
        """Initialize AlphaTrend strategy with parameters."""
        super().__init__(
            atr_multiplier=atr_multiplier,
            source=source,
            smoothing_length=smoothing_length,
            percentile_period=percentile_period,
            volume_short_ma=volume_short_ma,
            volume_long_ma=volume_long_ma,
            volume_alignment_window=volume_alignment_window,
            signal_lookback=signal_lookback,
            stop_atr_multiplier=stop_atr_multiplier,
            grace_period_bars=grace_period_bars,
            momentum_gain_pct=momentum_gain_pct,
            momentum_lookback=momentum_lookback,
            risk_percent=risk_percent
        )

        # Store parameters as instance variables
        self.atr_multiplier = atr_multiplier
        self.source = source
        self.smoothing_length = smoothing_length
        self.percentile_period = percentile_period
        self.volume_short_ma = volume_short_ma
        self.volume_long_ma = volume_long_ma
        self.volume_alignment_window = volume_alignment_window
        self.signal_lookback = signal_lookback
        self.stop_atr_multiplier = stop_atr_multiplier
        self.grace_period_bars = grace_period_bars
        self.momentum_gain_pct = momentum_gain_pct
        self.momentum_lookback = momentum_lookback
        self.risk_percent = risk_percent

        # Track bars since entry for grace period and momentum
        self._bars_since_entry = 0
        self._entry_bar_open = None  # Store open price of entry bar for momentum calc

    def required_columns(self) -> List[str]:
        """Required columns from CSV data (including pre-calculated indicators)."""
        return [
            'date', 'open', 'high', 'low', 'close', 'volume',
            'atr_14',  # Pre-calculated ATR from raw data
            'ema_50'   # Pre-calculated EMA from raw data
        ]

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate custom AlphaTrend indicators ONCE before backtesting.

        Standard indicators (atr_14, ema_50) are read from raw data.
        Custom indicators specific to AlphaTrend are calculated here.

        PERFORMANCE OPTIMIZATION: This replaces the O(n²) on-the-fly calculation
        with O(n) vectorized calculation, providing 10-100x speedup.

        Args:
            data: Raw OHLCV data with pre-calculated standard indicators

        Returns:
            Data with all AlphaTrend-specific indicators added as columns
        """
        df = data.copy()

        # Verify required standard indicators exist
        if 'atr_14' not in df.columns:
            raise ValueError("Missing required indicator: atr_14 must be present in raw data")
        if 'ema_50' not in df.columns:
            raise ValueError("Missing required indicator: ema_50 must be present in raw data")

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
        Vectorized AlphaTrend calculation - replaces O(n) loop with optimized logic.

        Args:
            up_band: Upper band values
            down_band: Lower band values
            momentum_bullish: Momentum bullish flags

        Returns:
            AlphaTrend values as Series
        """
        n = len(up_band)
        alphatrend = np.zeros(n)

        # Initialize first value
        alphatrend[0] = up_band[0]

        # Optimized loop - unavoidable due to state dependency
        for i in range(1, n):
            if momentum_bullish[i]:
                # Uptrend
                alphatrend[i] = max(alphatrend[i-1], up_band[i]) if up_band[i] < alphatrend[i-1] else up_band[i]
            else:
                # Downtrend
                alphatrend[i] = min(alphatrend[i-1], down_band[i]) if down_band[i] > alphatrend[i-1] else down_band[i]

        return pd.Series(alphatrend)

    @staticmethod
    def _filter_alternating_signals_vectorized(
        cross_up: np.ndarray,
        cross_down: np.ndarray
    ) -> tuple[pd.Series, pd.Series]:
        """
        Filter signals to ensure alternating buy/sell - vectorized implementation.

        Args:
            cross_up: Cross up signals
            cross_down: Cross down signals

        Returns:
            Tuple of (filtered_buy, filtered_sell) as Series
        """
        n = len(cross_up)
        filtered_buy = np.zeros(n, dtype=bool)
        filtered_sell = np.zeros(n, dtype=bool)

        last_signal = None
        for i in range(n):
            if cross_up[i] and last_signal != 'buy':
                filtered_buy[i] = True
                last_signal = 'buy'
            elif cross_down[i] and last_signal != 'sell':
                filtered_sell[i] = True
                last_signal = 'sell'

        return pd.Series(filtered_buy), pd.Series(filtered_sell)

    def _get_indicators(self, context: StrategyContext) -> Optional[dict]:
        """
        Get pre-calculated indicators for current bar.

        PERFORMANCE: Simply reads values from pre-calculated columns.
        No computation happens here - all indicators were calculated in prepare_data().

        Returns:
            Dictionary with indicator values at current bar, or None if insufficient data
        """
        # Check if we have enough historical data
        if context.current_index < max(14, self.percentile_period):
            return None

        # Get current bar (all indicators are pre-calculated columns)
        current_bar = context.current_bar

        # Check for NaN values in critical indicators
        if pd.isna(current_bar.get('atr_14')) or pd.isna(current_bar.get('ema_50')):
            return None

        # Return pre-calculated indicator values
        return {
            'atr_14': current_bar['atr_14'],  # Read from raw data
            'alphatrend': current_bar['alphatrend'],
            'smooth_at': current_bar['smooth_at'],
            'filtered_buy': current_bar['filtered_buy'],
            'filtered_sell': current_bar['filtered_sell'],
            'volume_condition': current_bar['volume_condition'],
            'ema_50': current_bar['ema_50'],  # Read from raw data
            'mfi': current_bar['mfi']
        }

    def _check_alphatrend_signal_in_window(self, context: StrategyContext) -> bool:
        """
        Check if AlphaTrend buy signal occurred within lookback window.

        PERFORMANCE: Uses pre-calculated filtered_buy column.
        """
        current_idx = context.current_index
        start_idx = max(0, current_idx - self.signal_lookback + 1)

        # Check pre-calculated signals in window
        for i in range(start_idx, current_idx + 1):
            if context.data.iloc[i]['filtered_buy']:
                return True
        return False

    def _check_volume_alignment(self, context: StrategyContext) -> bool:
        """
        Check if volume condition was met within alignment window of any signal.

        For each AlphaTrend buy signal in the lookback window, check if volume
        condition was met within +/- volume_alignment_window bars.

        PERFORMANCE: Uses pre-calculated volume_condition column.
        """
        current_idx = context.current_index
        start_idx = max(0, current_idx - self.signal_lookback + 1)

        # Check each potential signal bar in lookback window
        for signal_offset in range(current_idx - start_idx + 1):
            signal_idx = start_idx + signal_offset

            # Check if there was a buy signal at this bar
            if context.data.iloc[signal_idx]['filtered_buy']:
                # Check volume condition within alignment window around this signal
                vol_start = max(0, signal_idx - self.volume_alignment_window)
                vol_end = min(len(context.data), signal_idx + self.volume_alignment_window + 1)

                # Check if volume condition met in window
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

        # Entry logic
        if not context.has_position:
            # Reset bars counter when not in position
            self._bars_since_entry = 0
            self._entry_bar_open = None

            # Check if AlphaTrend signal occurred in lookback window
            at_signal_in_window = self._check_alphatrend_signal_in_window(context)

            # Check if volume condition aligned with signal
            vol_aligned = self._check_volume_alignment(context)

            # Entry condition: Both AlphaTrend signal and volume condition met
            if at_signal_in_window and vol_aligned:
                # Calculate stop loss using atr_14 from raw data
                atr = indicators['atr_14']
                stop_loss = current_price - (atr * self.stop_atr_multiplier)

                # Store entry bar open for momentum calculation
                self._entry_bar_open = context.current_bar['open']

                return Signal.buy(
                    size=1.0,  # Will be overridden by position_size() method
                    stop_loss=stop_loss,
                    reason=f"AlphaTrend signal + Volume aligned"
                )

        # Exit logic
        else:
            # Increment bars since entry
            self._bars_since_entry += 1

            # Check EMA-50 exit condition (read from raw data)
            ema_exit = current_price < indicators['ema_50']

            # Grace period protection
            in_grace_period = self._bars_since_entry <= self.grace_period_bars

            # Momentum protection
            has_momentum = False
            if self._entry_bar_open is not None and self.momentum_lookback <= context.current_index:
                lookback_idx = context.current_index - self.momentum_lookback
                if lookback_idx >= 0:
                    lookback_open = context.data.iloc[lookback_idx]['open']
                    price_gain_pct = ((current_price - lookback_open) / lookback_open) * 100
                    has_momentum = price_gain_pct >= self.momentum_gain_pct

            # Exit only if EMA condition met and NOT protected
            if ema_exit and not in_grace_period and not has_momentum:
                return Signal.sell(reason=f"Price < EMA(50)")

        return Signal.hold()

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size using risk-based sizing with currency conversion.

        Formula: Position size = (Equity * Risk%) / (ATR * Multiplier in Base Currency)

        Uses atr_14 from raw data for consistent stop distance calculation.
        """
        if signal.stop_loss is None:
            # Fallback to default sizing if no stop loss
            return super().position_size(context, signal)

        # Get ATR from current bar (read from raw data)
        atr = context.current_bar.get('atr_14')
        if atr is None or pd.isna(atr):
            # Fallback to default sizing if ATR not available
            return super().position_size(context, signal)

        equity = context.total_equity  # In base currency (e.g., GBP)
        risk_amount = equity * (self.risk_percent / 100)  # In base currency

        # Calculate stop distance from ATR (matches TradingView)
        stop_distance = atr * self.stop_atr_multiplier  # In security currency

        if stop_distance <= 0:
            # Invalid stop distance, fallback to default
            return super().position_size(context, signal)

        # Convert stop distance to base currency
        stop_distance_base = stop_distance * context.fx_rate

        # Calculate shares based on risk in base currency
        shares = risk_amount / stop_distance_base

        # Ensure we don't exceed available capital (in base currency)
        max_shares = context.available_capital / (context.current_price * context.fx_rate)
        shares = min(shares, max_shares)

        return shares
