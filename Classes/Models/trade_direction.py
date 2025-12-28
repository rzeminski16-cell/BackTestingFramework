"""
Trade direction enum for specifying long or short trades.
"""
from enum import Enum


class TradeDirection(Enum):
    """Direction of a trade."""
    LONG = "LONG"    # Buy to open, sell to close
    SHORT = "SHORT"  # Sell to open, buy to close

    def __str__(self) -> str:
        return self.value

    @property
    def is_long(self) -> bool:
        """Check if direction is long."""
        return self == TradeDirection.LONG

    @property
    def is_short(self) -> bool:
        """Check if direction is short."""
        return self == TradeDirection.SHORT
