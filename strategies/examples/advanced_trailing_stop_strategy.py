"""
Advanced Strategy with Conditional Trailing Stops

Entry: Price above SMA-200 and RSI-14 not overbought
Exit: Price crosses below EMA-50 or stop loss hit
Trailing Stop: Move stop to breakeven after 5% profit, then trail with EMA-50

NOTE: All indicators (sma_200, ema_50, rsi_14) are read from raw data with fixed settings.
"""
from typing import List, Optional
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal


class AdvancedTrailingStopStrategy(BaseStrategy):
    """
    Advanced strategy demonstrating:
    - Multi-condition entry
    - Conditional trailing stops
    - Indicator-based stop loss

    Standard Indicators (read from raw data, static):
        - sma_200: Simple Moving Average (200-period, static)
        - ema_50: Exponential Moving Average (50-period, static)
        - rsi_14: Relative Strength Index (14-period, static)

    Parameters:
        rsi_threshold: RSI overbought threshold (default: 70)
        position_size: Position size (default: 0.25)
        initial_stop_pct: Initial stop loss (default: 0.08 = 8%)
        breakeven_profit_pct: P/L threshold to move stop to breakeven (default: 0.05 = 5%)
    """

    def __init__(self, rsi_threshold: float = 70,
                 position_size: float = 0.25, initial_stop_pct: float = 0.08,
                 breakeven_profit_pct: float = 0.05):
        """Initialize strategy."""
        super().__init__(
            rsi_threshold=rsi_threshold,
            position_size=position_size,
            initial_stop_pct=initial_stop_pct,
            breakeven_profit_pct=breakeven_profit_pct
        )
        self.rsi_threshold = rsi_threshold
        self.position_size_pct = position_size
        self.initial_stop_pct = initial_stop_pct
        self.breakeven_profit_pct = breakeven_profit_pct

    def required_columns(self) -> List[str]:
        """Required columns from data (including pre-calculated indicators)."""
        return [
            'date', 'close',
            'sma_200',  # Pre-calculated from raw data
            'ema_50',   # Pre-calculated from raw data
            'rsi_14'    # Pre-calculated from raw data
        ]

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal with multi-condition logic.

        All indicators are read from raw data (no calculations).
        """
        current_price = context.current_price

        # Get indicators from raw data
        sma = context.get_indicator_value('sma_200')
        ema = context.get_indicator_value('ema_50')
        rsi = context.get_indicator_value('rsi_14')

        if sma is None or ema is None or rsi is None:
            return Signal.hold()

        # Entry logic: Multi-condition
        if not context.has_position:
            # Condition 1: Price above long-term SMA (uptrend)
            # Condition 2: RSI not overbought
            if current_price > sma and rsi < self.rsi_threshold:
                # Additional confirmation: price above short-term EMA
                if current_price > ema:
                    stop_loss = current_price * (1 - self.initial_stop_pct)

                    return Signal.buy(
                        size=self.position_size_pct,
                        stop_loss=stop_loss,
                        reason=f"Price > SMA(200), RSI({rsi:.1f}) < {self.rsi_threshold}"
                    )

        # Exit logic: Price crosses below EMA
        else:
            if context.previous_bar is not None:
                previous_price = context.previous_bar['close']
                previous_ema = context.get_indicator_value('ema_50', offset=-1)

                if previous_ema is not None:
                    # Exit if price crosses below EMA
                    if previous_price >= previous_ema and current_price < ema:
                        return Signal.sell(reason=f"Price crossed below EMA(50)")

        return Signal.hold()

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """
        Implement conditional trailing stop logic.

        1. If P/L > breakeven_profit_pct, move stop to breakeven
        2. Then trail stop using EMA-50 (only move up, never down)
        """
        if not context.has_position:
            return None

        current_pl_pct = context.get_position_pl_pct()
        current_stop = context.position.stop_loss
        entry_price = context.position.entry_price

        # Get current EMA value from raw data
        ema = context.get_indicator_value('ema_50')
        if ema is None:
            return None

        # Stage 1: Move to breakeven after reaching profit threshold
        if current_pl_pct >= self.breakeven_profit_pct:
            # If current stop is below breakeven, move it to breakeven
            if current_stop < entry_price:
                return entry_price

            # Stage 2: Trail with EMA (only if EMA is above current stop)
            if ema > current_stop:
                return ema

        return None
