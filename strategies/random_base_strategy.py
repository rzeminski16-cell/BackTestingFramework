"""
Random Base Strategy

A baseline random strategy for comparison purposes. This strategy makes random
entry and exit decisions based on configurable probabilities.

Entry: Each day there is a configurable probability of opening a long position
Exit: Each day there is a configurable probability of exiting
Position Sizing: Fixed percentage of total capital
Stop Loss: ATR-based (ATR14 x multiplier)

This strategy serves as a baseline to compare other strategies against.
A good strategy should outperform this random baseline.

STRATEGY STRUCTURE:
- Trade Direction: LONG only
- Fundamental Rules: None (always pass)
- Entry Rules: Random probability check each bar
- Initial Stop Loss: ATR-based (ATR14 x stop_loss_atr_multiple)
- Position Sizing: Fixed percentage of capital
- Trailing Stop: None
- Partial Exit: None
- Full Exit Rules: Random probability check each bar
- Pyramiding: None

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - atr_14_atr: Average True Range (14-period) - used for stop loss calculation

OPTIMIZABLE PARAMETERS:
    - entry_probability: Probability of entry each bar (default: 0.10 = 10%)
    - exit_probability: Probability of exit each bar (default: 0.10 = 10%)
    - position_size_pct: Position size as % of capital (default: 5.0 = 5%)
    - stop_loss_atr_multiple: ATR multiplier for stop loss (default: 2.0)
    - random_seed: Random seed for reproducibility (default: None = random)
"""
import random
from typing import List, Optional
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class RandomBaseStrategy(BaseStrategy):
    """
    Random baseline strategy for comparison purposes.

    This strategy randomly enters and exits positions based on configurable
    probabilities. It serves as a baseline to compare other strategies against -
    a good strategy should significantly outperform this random approach.

    Parameters:
        entry_probability (float): Probability of entry each bar (0.0-1.0, default: 0.10)
        exit_probability (float): Probability of exit each bar (0.0-1.0, default: 0.10)
        position_size_pct (float): Position size as % of capital (default: 5.0)
        stop_loss_atr_multiple (float): ATR multiplier for stop loss (default: 2.0)
        random_seed (int, optional): Random seed for reproducibility (default: None)
    """

    def __init__(self, **params):
        """Initialize the random strategy with parameters."""
        # Set defaults before calling parent __init__
        self._entry_probability = params.get('entry_probability', 0.10)
        self._exit_probability = params.get('exit_probability', 0.10)
        self._position_size_pct = params.get('position_size_pct', 5.0)
        self._stop_loss_atr_multiple = params.get('stop_loss_atr_multiple', 2.0)
        self._random_seed = params.get('random_seed', None)

        # Initialize random number generator
        if self._random_seed is not None:
            random.seed(self._random_seed)

        super().__init__(**params)

    @property
    def trade_direction(self) -> TradeDirection:
        """This strategy only trades long."""
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        """
        Return list of required columns from raw data.

        We need:
        - date, close: Standard required columns
        - atr_14_atr: For stop loss calculation
        """
        return ['date', 'close', 'atr_14_atr']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate entry signal based on random probability.

        Each bar, there is an entry_probability chance of generating a buy signal.

        Args:
            context: Current market context

        Returns:
            Signal.buy() if random check passes, None otherwise
        """
        # Random entry check
        if random.random() < self._entry_probability:
            stop_loss = self.calculate_initial_stop_loss(context)
            return Signal.buy(
                size=self._position_size_pct / 100.0,  # Convert to fraction
                stop_loss=stop_loss,
                reason=f"Random entry (p={self._entry_probability:.0%})",
                direction=self.trade_direction
            )
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """
        Calculate initial stop loss using ATR.

        Stop loss = Current Price - (ATR14 x stop_loss_atr_multiple)

        Args:
            context: Current market context

        Returns:
            Stop loss price
        """
        atr = context.get_indicator_value('atr_14_atr')
        if atr is None or atr <= 0:
            # Fallback: 5% stop loss if ATR not available
            return context.current_price * 0.95

        stop_distance = atr * self._stop_loss_atr_multiple
        return context.current_price - stop_distance

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size as fixed percentage of capital.

        Position = (Capital x position_size_pct%) / Current Price

        Args:
            context: Current market context
            signal: Buy signal (contains size as fraction)

        Returns:
            Number of shares to buy
        """
        capital_to_use = context.available_capital * signal.size
        shares = capital_to_use / context.current_price
        return max(0, shares)

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate exit signal based on random probability.

        Each bar while in a position, there is an exit_probability chance
        of generating a sell signal.

        Args:
            context: Current market context

        Returns:
            Signal.sell() if random check passes, None otherwise
        """
        # Random exit check
        if random.random() < self._exit_probability:
            return Signal.sell(reason=f"Random exit (p={self._exit_probability:.0%})")
        return None

    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if not 0.0 <= self._entry_probability <= 1.0:
            raise ValueError(
                f"entry_probability must be between 0.0 and 1.0, "
                f"got {self._entry_probability}"
            )

        if not 0.0 <= self._exit_probability <= 1.0:
            raise ValueError(
                f"exit_probability must be between 0.0 and 1.0, "
                f"got {self._exit_probability}"
            )

        if not 0.0 < self._position_size_pct <= 100.0:
            raise ValueError(
                f"position_size_pct must be between 0 and 100, "
                f"got {self._position_size_pct}"
            )

        if self._stop_loss_atr_multiple <= 0:
            raise ValueError(
                f"stop_loss_atr_multiple must be positive, "
                f"got {self._stop_loss_atr_multiple}"
            )
