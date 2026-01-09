"""Position tracking for backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class PositionSide(Enum):
    """Position side enum."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_date: datetime
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.exit_price is None

    @property
    def pnl(self) -> float:
        """Calculate profit/loss for the position."""
        if self.exit_price is None:
            return 0.0

        if self.side == PositionSide.LONG:
            return (self.exit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.exit_price) * self.quantity

    @property
    def pnl_percent(self) -> float:
        """Calculate profit/loss percentage."""
        if self.exit_price is None:
            return 0.0

        if self.side == PositionSide.LONG:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100

    def close(self, exit_price: float, exit_date: datetime) -> None:
        """Close the position."""
        self.exit_price = exit_price
        self.exit_date = exit_date

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade (closed position)."""

    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    entry_date: datetime
    exit_date: datetime
    pnl: float
    pnl_percent: float

    @classmethod
    def from_position(cls, position: Position) -> "Trade":
        """Create a Trade from a closed Position."""
        if position.is_open:
            raise ValueError("Cannot create Trade from open position")

        return cls(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            quantity=position.quantity,
            entry_date=position.entry_date,
            exit_date=position.exit_date,
            pnl=position.pnl,
            pnl_percent=position.pnl_percent,
        )
