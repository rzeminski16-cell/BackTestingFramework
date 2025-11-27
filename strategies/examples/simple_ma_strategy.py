"""
Simple Moving Average Crossover Strategy

Entry: When price crosses above SMA
Exit: When price crosses below SMA or stop loss/take profit hit
"""
from typing import List
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal


class SimpleMAStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy.

    Parameters:
        ma_period: Moving average period (default: 50)
        position_size: Position size as fraction of capital (default: 0.2)
        stop_loss_pct: Stop loss percentage (default: 0.05 = 5%)
        take_profit_pct: Take profit percentage (default: 0.15 = 15%)
    """

    def __init__(self, ma_period: int = 50, position_size: float = 0.2,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.15):
        """
        Initialize strategy.

        Args:
            ma_period: Moving average period
            position_size: Position size (0.0-1.0)
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__(
            ma_period=ma_period,
            position_size=position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        self.ma_period = ma_period
        self.position_size_pct = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def required_columns(self) -> List[str]:
        """Required columns from data."""
        # We need the SMA column (e.g., sma_50)
        ma_column = f'sma_{self.ma_period}'
        return ['date', 'close', ma_column]

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal.

        Entry: Price crosses above MA
        Exit: Price crosses below MA
        """
        current_price = context.current_price
        ma_column = f'sma_{self.ma_period}'
        current_ma = context.get_indicator_value(ma_column)

        # Need previous bar for crossover detection
        if context.previous_bar is None or current_ma is None:
            return Signal.hold()

        previous_ma = context.get_indicator_value(ma_column, offset=-1)
        previous_price = context.previous_bar['close']

        if previous_ma is None:
            return Signal.hold()

        # Entry signal: price crosses above MA
        if not context.has_position:
            if previous_price <= previous_ma and current_price > current_ma:
                # Calculate stop loss and take profit
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)

                return Signal.buy(
                    size=self.position_size_pct,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Price crossed above SMA({self.ma_period})"
                )

        # Exit signal: price crosses below MA
        else:
            if previous_price >= previous_ma and current_price < current_ma:
                return Signal.sell(reason=f"Price crossed below SMA({self.ma_period})")

        return Signal.hold()

    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.ma_period <= 0:
            raise ValueError("MA period must be positive")
        if not 0 < self.position_size_pct <= 1.0:
            raise ValueError("Position size must be between 0 and 1")
        if self.stop_loss_pct < 0:
            raise ValueError("Stop loss percentage must be non-negative")
        if self.take_profit_pct < 0:
            raise ValueError("Take profit percentage must be non-negative")
