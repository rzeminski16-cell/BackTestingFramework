"""
Strategy context - immutable data passed to strategy methods.
"""
from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
from ..Models.position import Position
from ..Data.historical_data_view import HistoricalDataView


@dataclass(frozen=True)
class StrategyContext:
    """
    Immutable context passed to strategy methods.

    This ensures strategies can't accidentally modify state and
    makes it clear what data is available to the strategy.

    DATA LEAKAGE PROTECTION: The data DataFrame contains ONLY HISTORICAL DATA up to and
    including the current bar (0 to current_index). Future data is not accessible, preventing
    look-ahead bias. The current_index always points to the last bar in the data.

    Attributes:
        data: Historical dataset (bars 0 to current_index inclusive) with pre-calculated indicators.
              Can be either a DataFrame or HistoricalDataView (which provides look-ahead protection).
        current_index: Current bar index in the data (this is the last bar available)
        current_price: Current close price
        current_date: Current date
        position: Current position (None if no position)
        available_capital: Available capital for new positions (in base currency)
        total_equity: Total equity (capital + position value, in base currency)
        symbol: Security symbol being traded
        fx_rate: Exchange rate from security currency to base currency (1.0 if same currency)
                 Example: If trading USD security with GBP base, fx_rate=0.8 means $1 = £0.8
    """
    data: Union[pd.DataFrame, HistoricalDataView]
    current_index: int
    current_price: float
    current_date: pd.Timestamp
    position: Optional[Position]
    available_capital: float
    total_equity: float
    symbol: str = ""
    fx_rate: float = 1.0

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
            offset: Offset from current (negative = past, 0 = current)
                   Note: The data only contains historical bars, so positive offsets
                   beyond current_index will return None (no future data available)

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
