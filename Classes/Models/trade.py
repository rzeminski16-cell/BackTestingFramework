"""
Trade model for completed trades.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class Trade:
    """
    Represents a completed trade.

    Attributes:
        symbol: Security symbol
        entry_date: Entry date
        entry_price: Entry price
        exit_date: Exit date
        exit_price: Exit price
        quantity: Number of shares/units traded
        side: Trade side (always "LONG" for this framework)
        initial_stop_loss: Initial stop loss price
        final_stop_loss: Final stop loss price (may differ due to trailing)
        take_profit: Take profit price
        pl: Profit/Loss in currency
        pl_pct: Profit/Loss as percentage
        duration_days: Trade duration in days
        entry_reason: Reason for entry
        exit_reason: Reason for exit
        commission_paid: Total commission paid
        partial_exits: Number of partial exits before final exit
        metadata: Additional trade-specific data
    """
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: float
    side: str = "LONG"
    initial_stop_loss: Optional[float] = None
    final_stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pl: float = 0.0
    pl_pct: float = 0.0
    duration_days: int = 0
    entry_reason: str = ""
    exit_reason: str = ""
    commission_paid: float = 0.0
    partial_exits: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pl > 0

    @property
    def entry_value(self) -> float:
        """Calculate entry value."""
        return self.entry_price * self.quantity

    @property
    def exit_value(self) -> float:
        """Calculate exit value."""
        return self.exit_price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trade to dictionary for CSV export.

        Returns:
            Dictionary representation
        """
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.strftime('%Y-%m-%d') if isinstance(self.entry_date, datetime) else self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date.strftime('%Y-%m-%d') if isinstance(self.exit_date, datetime) else self.exit_date,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'side': self.side,
            'initial_stop_loss': self.initial_stop_loss,
            'final_stop_loss': self.final_stop_loss,
            'take_profit': self.take_profit,
            'pl': self.pl,
            'pl_pct': self.pl_pct,
            'duration_days': self.duration_days,
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
            'commission_paid': self.commission_paid,
            'partial_exits': self.partial_exits
        }

    @classmethod
    def from_position(cls, position, exit_date: datetime, exit_price: float,
                     exit_reason: str, commission_paid: float) -> 'Trade':
        """
        Create a Trade from a closed Position.

        Args:
            position: Position object
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: Reason for exit
            commission_paid: Total commission paid (entry + exit)

        Returns:
            Trade object
        """
        duration_days = (exit_date - position.entry_date).days

        # Calculate P/L including partial exits
        pl = 0.0

        # P/L from partial exits
        for partial_exit in position.partial_exits:
            pl += (partial_exit.price - position.entry_price) * partial_exit.quantity

        # P/L from final exit
        pl += (exit_price - position.entry_price) * position.current_quantity

        # Subtract total commission
        pl -= commission_paid

        # Calculate P/L percentage
        initial_value = position.entry_price * position.initial_quantity
        pl_pct = (pl / initial_value * 100) if initial_value > 0 else 0.0

        return cls(
            symbol=position.symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            quantity=position.initial_quantity,  # Initial quantity for trade record
            side="LONG",
            initial_stop_loss=position.stop_loss,  # Stop at entry
            final_stop_loss=position.stop_loss,    # Final stop (may have been adjusted)
            take_profit=position.take_profit,
            pl=pl,
            pl_pct=pl_pct,
            duration_days=duration_days,
            entry_reason=position.entry_reason,
            exit_reason=exit_reason,
            commission_paid=commission_paid,
            partial_exits=len(position.partial_exits)
        )

    def __str__(self) -> str:
        """String representation of trade."""
        return (f"{self.symbol} {self.side}: "
                f"{self.entry_date.strftime('%Y-%m-%d')} -> {self.exit_date.strftime('%Y-%m-%d')}, "
                f"P/L: {self.pl:.2f} ({self.pl_pct:.2f}%)")
