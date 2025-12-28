"""
Position model for tracking open positions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from .trade_direction import TradeDirection


@dataclass
class PartialExit:
    """Record of a partial position exit."""
    date: datetime
    quantity: float
    price: float
    reason: str = ""


@dataclass
class PyramidEntry:
    """Record of a pyramid addition to position."""
    date: datetime
    quantity: float
    price: float
    commission: float
    reason: str = ""


@dataclass
class Position:
    """
    Tracks an open position with support for partial exits and pyramiding.

    Attributes:
        symbol: Security symbol
        entry_date: Position entry date
        entry_price: Average entry price (updated on pyramid)
        initial_quantity: Original position size (before pyramiding)
        current_quantity: Current position size (after partial exits)
        direction: Trade direction (LONG or SHORT)
        stop_loss: Current stop loss price
        take_profit: Take profit price
        partial_exits: List of partial exit records
        pyramid_entries: List of pyramid entry records
        has_pyramided: Whether position has been pyramided
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
    direction: TradeDirection = TradeDirection.LONG
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    partial_exits: List[PartialExit] = field(default_factory=list)
    pyramid_entries: List[PyramidEntry] = field(default_factory=list)
    has_pyramided: bool = False
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
        # P/L from current position (direction-aware)
        if self.direction == TradeDirection.LONG:
            current_pl = (current_price - self.entry_price) * self.current_quantity
        else:  # SHORT
            current_pl = (self.entry_price - current_price) * self.current_quantity

        # P/L from partial exits
        if self.direction == TradeDirection.LONG:
            partial_pl = sum(
                (exit.price - self.entry_price) * exit.quantity
                for exit in self.partial_exits
            )
        else:  # SHORT
            partial_pl = sum(
                (self.entry_price - exit.price) * exit.quantity
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
            Unrealized P/L as percentage of total investment
        """
        total_investment = self.get_total_investment()
        if total_investment == 0:
            return 0.0
        return (self.calculate_pl(current_price) / total_investment) * 100

    def get_total_investment(self) -> float:
        """
        Calculate total investment including initial entry and pyramids.

        Note: After pyramiding, entry_price is the weighted average.
        This method calculates total investment as average price * total quantity.

        Returns:
            Total investment value
        """
        return self.entry_price * self.current_quantity

    @property
    def total_quantity(self) -> float:
        """
        Get total quantity including pyramids (before any exits).

        Returns:
            Total quantity
        """
        base_qty = self.initial_quantity
        pyramid_qty = sum(p.quantity for p in self.pyramid_entries)
        return base_qty + pyramid_qty

    def add_pyramid(self, pyramid: PyramidEntry) -> None:
        """
        Add a pyramid entry to the position.

        Updates:
        - Average entry price (weighted average)
        - Current quantity
        - Total commission paid
        - has_pyramided flag

        Args:
            pyramid: Pyramid entry record
        """
        if self.has_pyramided:
            raise ValueError("Position has already been pyramided (max 1 pyramid per trade)")

        # Calculate new average entry price
        old_value = self.entry_price * self.current_quantity
        new_value = pyramid.price * pyramid.quantity
        new_total_quantity = self.current_quantity + pyramid.quantity

        self.entry_price = (old_value + new_value) / new_total_quantity

        # Update quantities
        self.current_quantity = new_total_quantity

        # Update commission
        self.total_commission_paid += pyramid.commission

        # Record pyramid
        self.pyramid_entries.append(pyramid)
        self.has_pyramided = True

    def calculate_breakeven_stop(self, exit_commission_rate: float = 0.0) -> float:
        """
        Calculate break-even stop loss price.

        The break-even price accounts for:
        - Total entry costs (initial + pyramids)
        - Total commissions paid
        - Estimated exit commission

        For LONG: break-even is above average entry price
        For SHORT: break-even is below average entry price

        Args:
            exit_commission_rate: Commission rate for exit (e.g., 0.001 for 0.1%)

        Returns:
            Break-even stop loss price
        """
        # Total investment cost
        total_cost = self.get_total_investment()

        # Add all commissions
        total_cost += self.total_commission_paid

        # Estimate exit commission
        estimated_exit_value = self.current_quantity * self.entry_price
        estimated_exit_commission = estimated_exit_value * exit_commission_rate
        total_cost += estimated_exit_commission

        # Calculate break-even price per share
        breakeven_price = total_cost / self.current_quantity

        return breakeven_price

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
        direction = self.direction.value
        pyramid_str = " [PYRAMIDED]" if self.has_pyramided else ""
        return (f"{self.symbol} [{direction}]{pyramid_str}: {self.current_quantity:.4f} @ {self.entry_price:.2f} "
                f"(SL: {self.stop_loss}, TP: {self.take_profit})")
