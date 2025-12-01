"""
Position model for tracking open positions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class PartialExit:
    """Record of a partial position exit."""
    date: datetime
    quantity: float
    price: float
    reason: str = ""


@dataclass
class Position:
    """
    Tracks an open position with support for partial exits.

    Attributes:
        symbol: Security symbol
        entry_date: Position entry date
        entry_price: Average entry price
        initial_quantity: Original position size
        current_quantity: Current position size (after partial exits)
        stop_loss: Current stop loss price
        take_profit: Take profit price
        partial_exits: List of partial exit records
        total_commission_paid: Total commission paid on this position
        entry_reason: Reason for entering position
        entry_fx_rate: FX rate at entry (for FX P&L calculation)
        security_currency: Currency the security is denominated in
        entry_equity: Total portfolio equity at time of entry
    """
    symbol: str
    entry_date: datetime
    entry_price: float
    initial_quantity: float
    current_quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    partial_exits: List[PartialExit] = field(default_factory=list)
    total_commission_paid: float = 0.0
    entry_reason: str = ""
    entry_fx_rate: float = 1.0  # FX rate at entry
    security_currency: str = "GBP"  # Currency security is denominated in
    entry_equity: float = 0.0  # Total portfolio equity at time of entry

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.current_quantity > 0

    @property
    def current_value(self) -> float:
        """
        Calculate current position value (excluding commission).

        Note: Needs current price from outside.
        """
        raise NotImplementedError("Use calculate_value(current_price) instead")

    def calculate_value(self, current_price: float) -> float:
        """
        Calculate current position value.

        Args:
            current_price: Current market price

        Returns:
            Current value of position
        """
        return self.current_quantity * current_price

    def calculate_pl(self, current_price: float) -> float:
        """
        Calculate unrealized P/L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P/L (including partial exits)
        """
        # P/L from current position
        current_pl = (current_price - self.entry_price) * self.current_quantity

        # P/L from partial exits
        partial_pl = sum(
            (exit.price - self.entry_price) * exit.quantity
            for exit in self.partial_exits
        )

        # Subtract commission
        total_pl = current_pl + partial_pl - self.total_commission_paid
        return total_pl

    def calculate_pl_pct(self, current_price: float) -> float:
        """
        Calculate unrealized P/L percentage.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P/L as percentage of initial investment
        """
        initial_value = self.entry_price * self.initial_quantity
        if initial_value == 0:
            return 0.0
        return (self.calculate_pl(current_price) / initial_value) * 100

    def add_partial_exit(self, exit: PartialExit) -> None:
        """
        Record a partial exit.

        Args:
            exit: Partial exit record
        """
        if exit.quantity > self.current_quantity:
            raise ValueError(
                f"Cannot exit {exit.quantity} shares, only {self.current_quantity} remaining"
            )
        self.partial_exits.append(exit)
        self.current_quantity -= exit.quantity

    def update_stop_loss(self, new_stop_loss: float) -> None:
        """
        Update stop loss price.

        Args:
            new_stop_loss: New stop loss price
        """
        self.stop_loss = new_stop_loss

    def duration_days(self, current_date: datetime) -> int:
        """
        Calculate position duration in days.

        Args:
            current_date: Current date

        Returns:
            Duration in days
        """
        return (current_date - self.entry_date).days

    def __str__(self) -> str:
        """String representation of position."""
        return (f"{self.symbol}: {self.current_quantity:.4f} @ {self.entry_price:.2f} "
                f"(SL: {self.stop_loss}, TP: {self.take_profit})")
