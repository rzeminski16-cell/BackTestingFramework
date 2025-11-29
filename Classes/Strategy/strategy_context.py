"""
Strategy context - immutable data passed to strategy methods.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from ..Models.position import Position


@dataclass(frozen=True)
class StrategyContext:
    """
    Immutable context passed to strategy methods.

    This ensures strategies can't accidentally modify state and
    makes it clear what data is available to the strategy.

    PERFORMANCE: The data DataFrame contains ALL bars (full dataset) for efficiency.
    Strategies must use current_index to access only historical data (0 to current_index)
    to prevent lookahead bias. Helper methods enforce this automatically.

    Attributes:
        data: Full dataset with all bars and pre-calculated indicators
        current_index: Current bar index in the data (strategies must not access beyond this)
        current_price: Current close price
        current_date: Current date
        position: Current position (None if no position)
        available_capital: Available capital for new positions
        total_equity: Total equity (capital + position value)
    """
    data: pd.DataFrame
    current_index: int
    current_price: float
    current_date: pd.Timestamp
    position: Optional[Position]
    available_capital: float
    total_equity: float

    @property
    def has_position(self) -> bool:
        """Check if currently in a position."""
        return self.position is not None and self.position.is_open

    @property
    def current_bar(self) -> pd.Series:
        """Get current bar data."""
        return self.data.iloc[self.current_index]

    @property
    def previous_bar(self) -> Optional[pd.Series]:
        """Get previous bar data (None if at first bar)."""
        if self.current_index > 0:
            return self.data.iloc[self.current_index - 1]
        return None

    def get_bar(self, offset: int = 0) -> Optional[pd.Series]:
        """
        Get bar at offset from current position.

        Args:
            offset: Offset from current (negative = past, positive = future)
                   Note: Positive offsets should not be used in strategies!

        Returns:
            Bar data or None if out of bounds
        """
        index = self.current_index + offset
        if 0 <= index < len(self.data):
            return self.data.iloc[index]
        return None

    def get_indicator_value(self, column: str, offset: int = 0) -> Optional[float]:
        """
        Get indicator value at current bar or with offset.

        Args:
            column: Column name
            offset: Offset from current (negative = past)

        Returns:
            Indicator value or None if not available
        """
        bar = self.get_bar(offset)
        if bar is not None and column in bar:
            return bar[column]
        return None

    def get_position_pl(self) -> float:
        """
        Get current position P/L.

        Returns:
            P/L in currency (0 if no position)
        """
        if self.has_position:
            return self.position.calculate_pl(self.current_price)
        return 0.0

    def get_position_pl_pct(self) -> float:
        """
        Get current position P/L percentage.

        Returns:
            P/L as percentage (0 if no position)
        """
        if self.has_position:
            return self.position.calculate_pl_pct(self.current_price)
        return 0.0
