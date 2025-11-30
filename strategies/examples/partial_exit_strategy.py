"""
Strategy with Partial Position Exits

Entry: RSI-14 oversold and price bounces off support
Exit: Scale out at profit targets, final exit on signal

NOTE: RSI-14 is read from raw data with fixed 14-period setting.
"""
from typing import List, Optional
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal


class PartialExitStrategy(BaseStrategy):
    """
    Strategy demonstrating partial exits (scaling out).

    Entry when RSI-14 is oversold and recovering
    Takes partial profits at 10% and 20% gains
    Final exit when RSI-14 becomes overbought or stop loss hit

    Standard Indicators (read from raw data, static):
        - rsi_14: Relative Strength Index (14-period, static)

    Parameters:
        rsi_oversold: RSI oversold threshold (default: 30)
        rsi_overbought: RSI overbought threshold (default: 70)
        position_size: Initial position size (default: 0.3)
        first_target_pct: First profit target (default: 0.10 = 10%)
        second_target_pct: Second profit target (default: 0.20 = 20%)
        stop_loss_pct: Stop loss (default: 0.06 = 6%)
    """

    def __init__(self, rsi_oversold: float = 30,
                 rsi_overbought: float = 70, position_size: float = 0.3,
                 first_target_pct: float = 0.10, second_target_pct: float = 0.20,
                 stop_loss_pct: float = 0.06):
        """Initialize strategy."""
        super().__init__(
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            position_size=position_size,
            first_target_pct=first_target_pct,
            second_target_pct=second_target_pct,
            stop_loss_pct=stop_loss_pct
        )
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size_pct = position_size
        self.first_target_pct = first_target_pct
        self.second_target_pct = second_target_pct
        self.stop_loss_pct = stop_loss_pct

        # Track which partial exits have been taken
        self.first_exit_taken = False
        self.second_exit_taken = False

    def required_columns(self) -> List[str]:
        """Required columns from data (including pre-calculated indicators)."""
        return [
            'date', 'close',
            'rsi_14'  # Pre-calculated from raw data
        ]

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal.

        Entry: RSI crosses above oversold threshold
        Exit: RSI crosses above overbought threshold

        All indicators are read from raw data (no calculations).
        """
        current_price = context.current_price
        rsi = context.get_indicator_value('rsi_14')

        if rsi is None:
            return Signal.hold()

        # Entry: RSI crosses above oversold (recovery from oversold)
        if not context.has_position:
            # Reset partial exit flags when not in position
            self.first_exit_taken = False
            self.second_exit_taken = False

            if context.previous_bar is not None:
                previous_rsi = context.get_indicator_value('rsi_14', offset=-1)

                if previous_rsi is not None:
                    # Entry when RSI crosses above oversold level
                    if previous_rsi <= self.rsi_oversold and rsi > self.rsi_oversold:
                        stop_loss = current_price * (1 - self.stop_loss_pct)

                        return Signal.buy(
                            size=self.position_size_pct,
                            stop_loss=stop_loss,
                            reason=f"RSI({rsi:.1f}) recovering from oversold"
                        )

        # Exit: RSI becomes overbought
        else:
            if rsi >= self.rsi_overbought:
                return Signal.sell(reason=f"RSI({rsi:.1f}) overbought")

        return Signal.hold()

    def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
        """
        Check for partial exit opportunities.

        Takes 50% profit at first target
        Takes another 50% (25% of original) at second target
        """
        if not context.has_position:
            return None

        current_pl_pct = context.get_position_pl_pct()

        # First partial exit at 10% profit (take 50% off)
        if not self.first_exit_taken and current_pl_pct >= self.first_target_pct * 100:
            self.first_exit_taken = True
            return 0.5  # Exit 50% of position

        # Second partial exit at 20% profit (take 50% of remaining = 25% of original)
        if self.first_exit_taken and not self.second_exit_taken:
            if current_pl_pct >= self.second_target_pct * 100:
                self.second_exit_taken = True
                return 0.5  # Exit 50% of remaining position

        return None
