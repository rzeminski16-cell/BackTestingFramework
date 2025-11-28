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

        # Cached indicator data (calculated once per bar to avoid recalculation)
        self._indicators_cache = {}
        self._last_calculated_index = -1

        # Track bars since entry for grace period
        self._bars_since_entry = 0
        self._entry_bar_open = None  # Store open price of entry bar for momentum calc

    def required_columns(self) -> List[str]:
        """Required columns from CSV data."""
        return ['date', 'open', 'high', 'low', 'close', 'volume']

    def _calculate_indicators(self, data: pd.DataFrame, current_index: int) -> dict:
        """
        Calculate all indicators needed for the strategy.

        Returns a dictionary with all calculated indicators at current_index.
        """
        # Use cache if already calculated for this bar
        if current_index == self._last_calculated_index:
            return self._indicators_cache

        # Get data up to current point (no lookahead)
        df = data.iloc[:current_index + 1].copy()

        if len(df) < max(self.common_period, self.percentile_period):
            return {}

        indicators = {}

        # Calculate True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # EMA-based ATR (more responsive than SMA)
        df['atr'] = df['tr'].ewm(span=self.common_period, adjust=False).mean()

        # Standard ATR for stop loss (using SMA as in original)
        df['atr_stop'] = df['tr'].rolling(window=self.common_period).mean()

        # Adaptive coefficient based on volatility
        df['atr_ema_long'] = df['atr'].ewm(span=self.common_period * 3, adjust=False).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_ema_long']
        df['adaptive_coeff'] = self.atr_multiplier * df['volatility_ratio']

        # AlphaTrend bands
        df['up_band'] = df['low'] - df['atr'] * df['adaptive_coeff']
        df['down_band'] = df['high'] + df['atr'] * df['adaptive_coeff']

        # Calculate MFI (Money Flow Index)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['raw_money_flow'] = df['typical_price'] * df['volume']

        # Determine positive and negative money flow
        df['price_change'] = df['typical_price'].diff()
        df['positive_flow'] = np.where(df['price_change'] > 0, df['raw_money_flow'], 0)
        df['negative_flow'] = np.where(df['price_change'] < 0, df['raw_money_flow'], 0)

        # Sum over period
        positive_mf = df['positive_flow'].rolling(window=self.common_period).sum()
        negative_mf = df['negative_flow'].rolling(window=self.common_period).sum()

        # MFI calculation
        mfi_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))

        # Dynamic MFI thresholds using percentile
        df['mfi_upper'] = df['mfi'].rolling(window=self.percentile_period).quantile(0.70)
        df['mfi_lower'] = df['mfi'].rolling(window=self.percentile_period).quantile(0.30)
        df['mfi_threshold'] = (df['mfi_upper'] + df['mfi_lower']) / 2

        # Momentum bullish condition
        df['momentum_bullish'] = df['mfi'] >= df['mfi_threshold']

        # AlphaTrend calculation
        alphatrend = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                alphatrend.iloc[i] = df['up_band'].iloc[i]
            else:
                if df['momentum_bullish'].iloc[i]:
                    # Uptrend
                    if df['up_band'].iloc[i] < alphatrend.iloc[i-1]:
                        alphatrend.iloc[i] = alphatrend.iloc[i-1]
                    else:
                        alphatrend.iloc[i] = df['up_band'].iloc[i]
                else:
                    # Downtrend
                    if df['down_band'].iloc[i] > alphatrend.iloc[i-1]:
                        alphatrend.iloc[i] = alphatrend.iloc[i-1]
                    else:
                        alphatrend.iloc[i] = df['down_band'].iloc[i]

        df['alphatrend'] = alphatrend

        # Smoothed AlphaTrend
        df['smooth_at'] = df['alphatrend'].ewm(span=self.smoothing_length, adjust=False).mean()

        # AlphaTrend signals (crossovers)
        df['at_cross_up'] = (df['alphatrend'] > df['smooth_at']) & (df['alphatrend'].shift(1) <= df['smooth_at'].shift(1))
        df['at_cross_down'] = (df['alphatrend'] < df['smooth_at']) & (df['alphatrend'].shift(1) >= df['smooth_at'].shift(1))

        # Filter for alternating signals
        df['filtered_buy'] = False
        df['filtered_sell'] = False

        last_signal = None
        for i in range(len(df)):
            if df['at_cross_up'].iloc[i] and last_signal != 'buy':
                df.loc[df.index[i], 'filtered_buy'] = True
                last_signal = 'buy'
            elif df['at_cross_down'].iloc[i] and last_signal != 'sell':
                df.loc[df.index[i], 'filtered_sell'] = True
                last_signal = 'sell'

        # Volume filter
        df['vol_short_ma'] = df['volume'].rolling(window=self.volume_short_ma).mean()
        df['vol_long_ma'] = df['volume'].rolling(window=self.volume_long_ma).mean()
        df['volume_condition'] = df['vol_short_ma'] > df['vol_long_ma']

        # Exit EMA (using configured source)
        df['exit_ema'] = df[self.source].ewm(span=self.exit_ema_period, adjust=False).mean()

        # Store current bar indicators
        current = df.iloc[-1]
        indicators['atr'] = current['atr']
        indicators['atr_stop'] = current['atr_stop']
        indicators['alphatrend'] = current['alphatrend']
        indicators['smooth_at'] = current['smooth_at']
        indicators['filtered_buy'] = current['filtered_buy']
        indicators['filtered_sell'] = current['filtered_sell']
        indicators['volume_condition'] = current['volume_condition']
        indicators['exit_ema'] = current['exit_ema']
        indicators['mfi'] = current['mfi']

        # Store full series for lookback checks
        indicators['_df'] = df

        # Cache results
        self._indicators_cache = indicators
        self._last_calculated_index = current_index

        return indicators

    def _check_alphatrend_signal_in_window(self, df: pd.DataFrame, current_idx: int) -> bool:
        """Check if AlphaTrend buy signal occurred within lookback window."""
        start_idx = max(0, current_idx - self.signal_lookback + 1)
        window = df.iloc[start_idx:current_idx + 1]
        return window['filtered_buy'].any()

    def _check_volume_alignment(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        Check if volume condition was met within alignment window of any signal.

        For each AlphaTrend buy signal in the lookback window, check if volume
        condition was met within +/- volume_alignment_window bars.
        """
        start_idx = max(0, current_idx - self.signal_lookback + 1)

        # Check each potential signal bar in lookback window
        for signal_offset in range(current_idx - start_idx + 1):
            signal_idx = start_idx + signal_offset

            # Check if there was a buy signal at this bar
            if df.iloc[signal_idx]['filtered_buy']:
                # Check volume condition within alignment window around this signal
                vol_start = max(0, signal_idx - self.volume_alignment_window)
                vol_end = min(len(df), signal_idx + self.volume_alignment_window + 1)
                vol_window = df.iloc[vol_start:vol_end]

                if vol_window['volume_condition'].any():
                    return True

        return False

    def generate_signal(self, context: StrategyContext) -> Signal:
        """Generate trading signal based on AlphaTrend logic."""
        # Calculate indicators
        indicators = self._calculate_indicators(context.data, context.current_index)

        if not indicators:
            return Signal.hold()

        df = indicators['_df']
        current_price = context.current_price

        # Entry logic
        if not context.has_position:
            # Reset bars counter when not in position
            self._bars_since_entry = 0
            self._entry_bar_open = None

            # Check if AlphaTrend signal occurred in lookback window
            at_signal_in_window = self._check_alphatrend_signal_in_window(df, context.current_index)

            # Check if volume condition aligned with signal
            vol_aligned = self._check_volume_alignment(df, context.current_index)

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
