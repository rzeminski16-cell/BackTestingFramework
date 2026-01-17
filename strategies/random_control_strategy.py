"""
Random Control Strategy

A control/baseline strategy that makes random entry and exit decisions.
This strategy is used for comparison purposes - a profitable strategy
should significantly outperform this random baseline.

Entry: Each day there is a configurable probability of entering a long position
Exit: Each day there is a configurable probability of exiting
Stop Loss: ATR14 x atr_multiplier
Position Sizing: Risk-based (lose risk_percent% of equity at stop loss)

No advanced features: No trailing stop, no partial exits, no pyramiding.

STRATEGY STRUCTURE:
- Trade Direction: LONG only
- Fundamental Rules: None (always pass)
- Entry Rules: Random probability check each bar (entry_probability)
- Initial Stop Loss: ATR-based (ATR14 x atr_multiplier)
- Position Sizing: Risk-based sizing (risk_percent% of equity at stop loss)
- Trailing Stop: None
- Partial Exit: None
- Full Exit Rules: Random probability check each bar (exit_probability)
- Pyramiding: None

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - atr_14_atr: Average True Range (14-period) - used for stop loss calculation

OPTIMIZABLE PARAMETERS:
    - entry_probability: Probability of entry each bar (default: 0.10 = 10%)
    - exit_probability: Probability of exit each bar (default: 0.10 = 10%)
    - atr_multiplier: ATR multiplier for stop loss (default: 2.0)
    - risk_percent: Percent of equity to risk per trade (default: 2.0)
    - random_seed: Random seed for reproducibility (default: None = random)
"""
import random
from typing import List, Optional
import pandas as pd
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class RandomControlStrategy(BaseStrategy):
    """
    Random Control Strategy for baseline comparison.

    This strategy randomly enters and exits positions based on configurable
    probabilities. It uses risk-based position sizing to ensure consistent
    risk per trade.

    A good trading strategy should significantly outperform this random baseline.
    """

    def __init__(self,
                 # Entry/Exit probabilities
                 entry_probability: float = 0.10,
                 exit_probability: float = 0.10,
                 # Stop loss parameters
                 atr_multiplier: float = 2.0,
                 # Position sizing
                 risk_percent: float = 2.0,
                 # Reproducibility
                 random_seed: Optional[int] = None):
        """
        Initialize Random Control strategy.

        Args:
            entry_probability: Probability of entry each bar (0.0-1.0, default: 0.10)
            exit_probability: Probability of exit each bar (0.0-1.0, default: 0.10)
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)
            risk_percent: Percent of equity to risk per trade (default: 2.0)
            random_seed: Random seed for reproducibility (default: None = random)
        """
        super().__init__(
            entry_probability=entry_probability,
            exit_probability=exit_probability,
            atr_multiplier=atr_multiplier,
            risk_percent=risk_percent,
            random_seed=random_seed
        )

        # Store parameters
        self.entry_probability = entry_probability
        self.exit_probability = exit_probability
        self.atr_multiplier = atr_multiplier
        self.risk_percent = risk_percent
        self.random_seed = random_seed

        # Initialize random number generator
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if not 0.0 <= self.entry_probability <= 1.0:
            raise ValueError(
                f"entry_probability must be between 0.0 and 1.0, "
                f"got {self.entry_probability}"
            )

        if not 0.0 <= self.exit_probability <= 1.0:
            raise ValueError(
                f"exit_probability must be between 0.0 and 1.0, "
                f"got {self.exit_probability}"
            )

        if self.atr_multiplier <= 0:
            raise ValueError(
                f"atr_multiplier must be positive, got {self.atr_multiplier}"
            )

        if self.risk_percent <= 0 or self.risk_percent > 100:
            raise ValueError(
                f"risk_percent must be between 0 and 100, got {self.risk_percent}"
            )

    @property
    def trade_direction(self) -> TradeDirection:
        """This is a LONG-only strategy."""
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        """
        Required columns from CSV data.

        Returns list of columns that MUST exist in the raw data.
        """
        return ['date', 'close', 'atr_14_atr']

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by normalizing column names.

        Args:
            data: Raw OHLCV data with pre-calculated indicators

        Returns:
            Data with normalized column names
        """
        df = data.copy()

        # Normalize ATR column name
        df['atr_14'] = df['atr_14_atr']

        return df

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
        Generate entry signal based on random probability.

        Each bar, there is an entry_probability chance of generating a buy signal.

        Args:
            context: Current market context

        Returns:
            Signal.buy() if random check passes, None otherwise
        """
        # Skip warmup period (need ATR)
        if context.current_index < 14:
            return None

        # Random entry check
        if random.random() < self.entry_probability:
            return Signal.buy(
                size=1.0,
                stop_loss=self.calculate_initial_stop_loss(context),
                reason=f"Random entry (p={self.entry_probability:.0%})",
                direction=self.trade_direction
            )

        return None

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
        if random.random() < self.exit_probability:
            return Signal.sell(reason=f"Random exit (p={self.exit_probability:.0%})")

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
