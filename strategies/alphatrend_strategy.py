"""
AlphaTrend Enhanced Long-Only Strategy

Based on AlphaTrend indicator by KivancOzbilgic with enhancements:
- Adaptive ATR multiplier based on volatility
- Dynamic MFI thresholds using percentile analysis
- Volume filter with alignment windows
- EMA-based exits with grace period and momentum protection
- Risk-based position sizing

Entry: AlphaTrend buy signal + volume condition within alignment window
Exit: Price closes below exit EMA (with protections) or stop loss hit
Stop Loss: ATR-based stop loss

PERFORMANCE OPTIMIZED:
- Indicators pre-calculated using vectorized operations (10-100x speedup)
- No repeated calculations during backtest
- O(n) complexity instead of O(n²)
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Indicators.indicator_engine import IndicatorEngine


class AlphaTrendStrategy(BaseStrategy):
    """
    AlphaTrend Enhanced Strategy with volume filter and risk management.

    Parameters:
        atr_multiplier: Base multiplier for ATR bands (default: 1.0)
        common_period: Period for ATR/MFI calculations (default: 14)
        source: Price source for calculations - 'close', 'open', 'high', 'low' (default: 'close')
        smoothing_length: EMA smoothing period for AlphaTrend (default: 3)
        percentile_period: Lookback for dynamic thresholds (default: 100)
        volume_short_ma: Volume short MA period (default: 4)
        volume_long_ma: Volume long MA period (default: 30)
        volume_alignment_window: Bars to check for volume condition (default: 14)
        signal_lookback: Bars to look back for AlphaTrend signal (default: 9)
        exit_ema_period: EMA period for exit signal (default: 50)
        stop_atr_multiplier: ATR multiplier for stop loss (default: 2.5)
        grace_period_bars: Bars to ignore EMA exit after entry (default: 14)
        momentum_gain_pct: % gain to ignore EMA exit (default: 2.0)
        momentum_lookback: Bars for momentum calculation (default: 7)
        risk_percent: Percent of equity to risk per trade (default: 2.0)
    """

    def __init__(self,
                 atr_multiplier: float = 1.0,
                 common_period: int = 14,
                 source: str = 'close',
                 smoothing_length: int = 3,
                 percentile_period: int = 100,
                 volume_short_ma: int = 4,
                 volume_long_ma: int = 30,
                 volume_alignment_window: int = 14,
                 signal_lookback: int = 9,
                 exit_ema_period: int = 50,
                 stop_atr_multiplier: float = 2.5,
                 grace_period_bars: int = 14,
                 momentum_gain_pct: float = 2.0,
                 momentum_lookback: int = 7,
                 risk_percent: float = 2.0):
        """Initialize AlphaTrend strategy with parameters."""
        super().__init__(
            atr_multiplier=atr_multiplier,
            common_period=common_period,
            source=source,
            smoothing_length=smoothing_length,
            percentile_period=percentile_period,
            volume_short_ma=volume_short_ma,
            volume_long_ma=volume_long_ma,
            volume_alignment_window=volume_alignment_window,
            signal_lookback=signal_lookback,
            exit_ema_period=exit_ema_period,
            stop_atr_multiplier=stop_atr_multiplier,
            grace_period_bars=grace_period_bars,
            momentum_gain_pct=momentum_gain_pct,
            momentum_lookback=momentum_lookback,
            risk_percent=risk_percent
        )

        # Store parameters as instance variables
        self.atr_multiplier = atr_multiplier
        self.common_period = common_period
        self.source = source
        self.smoothing_length = smoothing_length
        self.percentile_period = percentile_period
        self.volume_short_ma = volume_short_ma
        self.volume_long_ma = volume_long_ma
        self.volume_alignment_window = volume_alignment_window
        self.signal_lookback = signal_lookback
        self.exit_ema_period = exit_ema_period
        self.stop_atr_multiplier = stop_atr_multiplier
        self.grace_period_bars = grace_period_bars
        self.momentum_gain_pct = momentum_gain_pct
        self.momentum_lookback = momentum_lookback
        self.risk_percent = risk_percent

        # Track bars since entry for grace period and momentum
        self._bars_since_entry = 0
        self._entry_bar_open = None  # Store open price of entry bar for momentum calc

    def required_columns(self) -> List[str]:
        """Required columns from CSV data."""
        return ['date', 'open', 'high', 'low', 'close', 'volume']

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate all indicators ONCE before backtesting.

        PERFORMANCE OPTIMIZATION: This replaces the O(n²) on-the-fly calculation
        with O(n) vectorized calculation, providing 10-100x speedup.

        Args:
            data: Raw OHLCV data

        Returns:
            Data with all AlphaTrend indicators added as columns
        """
        return IndicatorEngine.calculate_alphatrend_indicators(
            data=data,
            atr_multiplier=self.atr_multiplier,
            common_period=self.common_period,
            source=self.source,
            smoothing_length=self.smoothing_length,
            percentile_period=self.percentile_period,
            volume_short_ma=self.volume_short_ma,
            volume_long_ma=self.volume_long_ma,
            volume_alignment_window=self.volume_alignment_window,
            signal_lookback=self.signal_lookback,
            exit_ema_period=self.exit_ema_period
        )

    def _get_indicators(self, context: StrategyContext) -> Optional[dict]:
        """
        Get pre-calculated indicators for current bar.

        PERFORMANCE: Simply reads values from pre-calculated columns.
        No computation happens here - all indicators were calculated in prepare_data().

        Returns:
            Dictionary with indicator values at current bar, or None if insufficient data
        """
        # Check if we have enough historical data
        if context.current_index < max(self.common_period, self.percentile_period):
            return None

        # Get current bar (all indicators are pre-calculated columns)
        current_bar = context.current_bar

        # Check for NaN values in critical indicators
        if pd.isna(current_bar.get('atr_stop')) or pd.isna(current_bar.get('exit_ema')):
            return None

        # Return pre-calculated indicator values
        return {
            'atr': current_bar['atr'],
            'atr_stop': current_bar['atr_stop'],
            'alphatrend': current_bar['alphatrend'],
            'smooth_at': current_bar['smooth_at'],
            'filtered_buy': current_bar['filtered_buy'],
            'filtered_sell': current_bar['filtered_sell'],
            'volume_condition': current_bar['volume_condition'],
            'exit_ema': current_bar['exit_ema'],
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
                # Calculate stop loss
                atr_stop = indicators['atr_stop']
                stop_loss = current_price - (atr_stop * self.stop_atr_multiplier)

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

            # Check EMA exit condition
            ema_exit = current_price < indicators['exit_ema']

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
                return Signal.sell(reason=f"Price < EMA({self.exit_ema_period})")

        return Signal.hold()

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size using risk-based sizing.

        Position size = (Equity * Risk%) / (Entry Price - Stop Loss)

        This ensures consistent risk per trade regardless of stop distance.
        """
        if signal.stop_loss is None:
            # Fallback to default sizing if no stop loss
            return super().position_size(context, signal)

        equity = context.total_equity
        risk_amount = equity * (self.risk_percent / 100)
        stop_distance = context.current_price - signal.stop_loss

        if stop_distance <= 0:
            # Invalid stop loss, fallback to default
            return super().position_size(context, signal)

        # Calculate shares based on risk
        shares = risk_amount / stop_distance

        # Ensure we don't exceed available capital
        max_shares = context.available_capital / context.current_price
        shares = min(shares, max_shares)

        return shares
