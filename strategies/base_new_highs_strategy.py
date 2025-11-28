"""
BaseNewHighs Strategy (Balanced Preset)

Translated from Pine Script "Multi-Preset Strategy V2" - Balanced configuration.

Entry Rules (ALL must be true):
1. Base Rules (Mandatory):
   - Close > highest high in past 14 candles
   - Close >= SMA(200)

2. Additional Rules:
   - MA Crossover: EMA(7) crossed over EMA(21) within last 14 bars
   - SAR Buy: Parabolic SAR < close (uptrend)

Exit Rules (ANY triggers exit):
   - Stop Loss: ATR-based (3 * ATR(14))
   - Sell Signal: EMA(14) > close * 1.03 (EMA is 3% above current price)

Position Sizing:
   - Risk 0.5% of equity per trade
   - Position size = (equity * 0.005) / (ATR(14) * 3.0)
"""
from typing import List
import numpy as np
import pandas as pd
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal


class BaseNewHighsStrategy(BaseStrategy):
    """
    Multi-indicator strategy with new highs confirmation.

    Parameters:
        new_high_n: Period for highest high lookback (default: 14)
        sma_length: SMA length (default: 200)
        ma_fast_length: Fast EMA length (default: 7)
        ma_slow_length: Slow EMA length (default: 21)
        ma_lookback_k: MA crossover lookback period (default: 14)
        sar_start: SAR start value (default: 0.02)
        sar_increment: SAR increment (default: 0.02)
        sar_maximum: SAR maximum (default: 0.2)
        ema_sell_length: EMA sell signal length (default: 14)
        ema_sell_threshold: EMA sell threshold % (default: 3.0)
        atr_length: ATR period (default: 14)
        atr_multiplier: ATR multiplier for stop loss (default: 3.0)
        risk_percent: Risk per trade as % of equity (default: 0.5)
    """

    def __init__(self,
                 new_high_n: int = 14,
                 sma_length: int = 200,
                 ma_fast_length: int = 7,
                 ma_slow_length: int = 21,
                 ma_lookback_k: int = 14,
                 sar_start: float = 0.02,
                 sar_increment: float = 0.02,
                 sar_maximum: float = 0.2,
                 ema_sell_length: int = 14,
                 ema_sell_threshold: float = 3.0,
                 atr_length: int = 14,
                 atr_multiplier: float = 3.0,
                 risk_percent: float = 0.5):
        """Initialize strategy with parameters."""
        super().__init__(
            new_high_n=new_high_n,
            sma_length=sma_length,
            ma_fast_length=ma_fast_length,
            ma_slow_length=ma_slow_length,
            ma_lookback_k=ma_lookback_k,
            sar_start=sar_start,
            sar_increment=sar_increment,
            sar_maximum=sar_maximum,
            ema_sell_length=ema_sell_length,
            ema_sell_threshold=ema_sell_threshold,
            atr_length=atr_length,
            atr_multiplier=atr_multiplier,
            risk_percent=risk_percent
        )

        self.new_high_n = new_high_n
        self.sma_length = sma_length
        self.ma_fast_length = ma_fast_length
        self.ma_slow_length = ma_slow_length
        self.ma_lookback_k = ma_lookback_k
        self.sar_start = sar_start
        self.sar_increment = sar_increment
        self.sar_maximum = sar_maximum
        self.ema_sell_length = ema_sell_length
        self.ema_sell_threshold = ema_sell_threshold
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_percent = risk_percent

    def required_columns(self) -> List[str]:
        """Required columns from data."""
        return ['date', 'open', 'high', 'low', 'close']

    def _calculate_current_indicators(self, data: pd.DataFrame) -> dict:
        """
        Calculate indicator values for the current bar (last row of data).
        Returns a dict with all indicator values.
        """
        if len(data) == 0:
            return None

        # SMA 200
        sma_200 = data['close'].rolling(window=self.sma_length).mean().iloc[-1]

        # EMA Fast and Slow for MA crossover
        ema_fast = data['close'].ewm(span=self.ma_fast_length, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.ma_slow_length, adjust=False).mean()

        ema_fast_current = ema_fast.iloc[-1]
        ema_slow_current = ema_slow.iloc[-1]

        # Check for MA crossover in last K bars
        # Must use FULL dataset EMA, then check last K bars
        ma_crossed_recently = False
        if len(data) >= self.ma_lookback_k:
            # Calculate EMA on full dataset
            ema_fast_full = data['close'].ewm(span=self.ma_fast_length, adjust=False).mean()
            ema_slow_full = data['close'].ewm(span=self.ma_slow_length, adjust=False).mean()

            # Check last K bars for crossover
            start_idx = len(data) - self.ma_lookback_k
            for i in range(start_idx + 1, len(data)):
                if ema_fast_full.iloc[i] > ema_slow_full.iloc[i] and \
                   ema_fast_full.iloc[i-1] <= ema_slow_full.iloc[i-1]:
                    ma_crossed_recently = True
                    break

        # EMA for sell signal
        ema_sell = data['close'].ewm(span=self.ema_sell_length, adjust=False).mean().iloc[-1]

        # ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_length).mean().iloc[-1]

        # Parabolic SAR - simplified calculation for current value
        sar = self._calculate_sar_current(data)

        return {
            'sma_200': sma_200,
            'ema_fast': ema_fast_current,
            'ema_slow': ema_slow_current,
            'ma_crossed_recently': ma_crossed_recently,
            'ema_sell': ema_sell,
            'atr': atr,
            'sar': sar
        }

    def _calculate_sar_current(self, data: pd.DataFrame) -> float:
        """Calculate Parabolic SAR for current bar."""
        if len(data) < 2:
            return data['low'].iloc[-1] if len(data) > 0 else 0.0

        # Initialize
        is_uptrend = True
        sar = data['low'].iloc[0]
        ep = data['high'].iloc[0]
        af = self.sar_start

        for i in range(1, len(data)):
            # Calculate SAR for current bar
            sar = sar + af * (ep - sar)

            # Check for reversal and update
            if is_uptrend:
                if i >= 2:
                    sar = min(sar, data['low'].iloc[i-1], data['low'].iloc[i-2])
                else:
                    sar = min(sar, data['low'].iloc[i-1])

                if data['low'].iloc[i] < sar:
                    is_uptrend = False
                    sar = ep
                    ep = data['low'].iloc[i]
                    af = self.sar_start
                else:
                    if data['high'].iloc[i] > ep:
                        ep = data['high'].iloc[i]
                        af = min(af + self.sar_increment, self.sar_maximum)
            else:
                if i >= 2:
                    sar = max(sar, data['high'].iloc[i-1], data['high'].iloc[i-2])
                else:
                    sar = max(sar, data['high'].iloc[i-1])

                if data['high'].iloc[i] > sar:
                    is_uptrend = True
                    sar = ep
                    ep = data['high'].iloc[i]
                    af = self.sar_start
                else:
                    if data['low'].iloc[i] < ep:
                        ep = data['low'].iloc[i]
                        af = min(af + self.sar_increment, self.sar_maximum)

        return sar

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on strategy rules.
        """
        # Check if we have enough data
        data_len = len(context.data)
        min_required_bars = max(self.sma_length, self.new_high_n, self.ma_lookback_k, self.atr_length)

        if data_len < min_required_bars:
            return Signal.hold()

        current_price = context.current_price

        # Calculate indicators for current bar
        indicators = self._calculate_current_indicators(context.data)

        if indicators is None:
            return Signal.hold()

        # Entry Logic (when not in position)
        if not context.has_position:
            # Rule 1: New High - Close > highest high in past N candles
            historical_highs = context.data['high'].iloc[-(self.new_high_n+1):-1]  # Exclude current bar
            new_high_rule = current_price > historical_highs.max()

            # Rule 2: SMA 200
            sma_200_rule = not pd.isna(indicators['sma_200']) and current_price >= indicators['sma_200']

            # Rule 3: MA Crossover
            ma_rule = indicators['ma_crossed_recently']

            # Rule 4: SAR Buy
            sar_buy_rule = not pd.isna(indicators['sar']) and indicators['sar'] < current_price

            # All rules must be true for entry
            if new_high_rule and sma_200_rule and ma_rule and sar_buy_rule:
                # Calculate stop loss based on ATR
                if pd.isna(indicators['atr']) or indicators['atr'] <= 0:
                    return Signal.hold()

                stop_distance = indicators['atr'] * self.atr_multiplier
                stop_loss = current_price - stop_distance

                # Build reason string
                reasons = [
                    f"New High (close > highest {self.new_high_n}d)",
                    f"Above SMA({self.sma_length})",
                    f"MA Cross (EMA{self.ma_fast_length}/EMA{self.ma_slow_length} within {self.ma_lookback_k}d)",
                    f"SAR Uptrend"
                ]
                reason = " | ".join(reasons)

                # Store ATR for position sizing
                self._current_atr = indicators['atr']

                return Signal.buy(
                    size=1.0,  # Placeholder, real size calculated in position_size()
                    stop_loss=stop_loss,
                    reason=reason
                )

        # Exit Logic (when in position)
        else:
            # Check sell signal: EMA > close * (1 + threshold%)
            if not pd.isna(indicators['ema_sell']):
                threshold_price = current_price * (1 + self.ema_sell_threshold / 100)
                if indicators['ema_sell'] > threshold_price:
                    return Signal.sell(reason=f"EMA({self.ema_sell_length}) > {self.ema_sell_threshold}% above price")

        return Signal.hold()

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size based on ATR and risk percentage.

        Position size = (equity * risk_percent / 100) / (ATR * atr_multiplier)
        """
        if signal.type != 'BUY':
            return 0.0

        # Use stored ATR from generate_signal
        if not hasattr(self, '_current_atr') or self._current_atr is None or self._current_atr <= 0:
            return super().position_size(context, Signal.buy(size=0.1))

        # Calculate risk amount
        risk_amount = context.total_equity * (self.risk_percent / 100)

        # Calculate stop distance
        stop_distance = self._current_atr * self.atr_multiplier

        # Calculate position size (number of shares)
        shares = risk_amount / stop_distance

        # Ensure we don't exceed available capital
        max_shares = context.available_capital / context.current_price
        shares = min(shares, max_shares)

        return shares

    def get_name(self) -> str:
        """Get strategy name."""
        return "BaseNewHighs"
