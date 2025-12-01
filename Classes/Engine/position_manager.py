"""
Position manager for tracking and managing open positions.
"""
from typing import Optional
from datetime import datetime
from ..Models.position import Position, PartialExit
from ..Models.signal import Signal, SignalType


class PositionManager:
    """
    Manages open positions with support for:
    - Position tracking
    - Stop loss checks
    - Take profit checks
    - Partial exits
    - Trailing stops
    """

    def __init__(self):
        """Initialize position manager."""
        self.position: Optional[Position] = None

    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.position is not None and self.position.is_open

    def open_position(self, symbol: str, entry_date: datetime, entry_price: float,
                     quantity: float, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None, entry_reason: str = "",
                     commission_paid: float = 0.0,
                     entry_fx_rate: float = 1.0,
                     security_currency: str = "GBP",
                     entry_equity: float = 0.0) -> None:
        """
        Open a new position.

        Args:
            symbol: Security symbol
            entry_date: Entry date
            entry_price: Entry price
            quantity: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            entry_reason: Reason for entry
            commission_paid: Entry commission
            entry_fx_rate: FX rate at entry (security currency to base currency)
            security_currency: Currency the security is denominated in
            entry_equity: Total portfolio equity at time of entry
        """
        if self.has_position:
            raise ValueError(f"Cannot open new position - already in position for {self.position.symbol}")

        self.position = Position(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            initial_quantity=quantity,
            current_quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            total_commission_paid=commission_paid,
            entry_reason=entry_reason,
            entry_fx_rate=entry_fx_rate,
            security_currency=security_currency,
            entry_equity=entry_equity
        )

    def close_position(self) -> Position:
        """
        Close the current position.

        Returns:
            The closed position

        Raises:
            ValueError: If no position to close
        """
        if not self.has_position:
            raise ValueError("No position to close")

        closed_position = self.position
        self.position = None
        return closed_position

    def add_partial_exit(self, exit_date: datetime, quantity: float,
                        price: float, reason: str = "",
                        commission_paid: float = 0.0) -> None:
        """
        Record a partial exit from the position.

        Args:
            exit_date: Exit date
            quantity: Quantity to exit
            price: Exit price
            reason: Reason for exit
            commission_paid: Commission on this exit
        """
        if not self.has_position:
            raise ValueError("No position to partially exit")

        partial_exit = PartialExit(
            date=exit_date,
            quantity=quantity,
            price=price,
            reason=reason
        )

        self.position.add_partial_exit(partial_exit)
        self.position.total_commission_paid += commission_paid

    def adjust_stop_loss(self, new_stop_loss: float) -> None:
        """
        Adjust the stop loss price.

        Args:
            new_stop_loss: New stop loss price
        """
        if not self.has_position:
            raise ValueError("No position to adjust stop loss")

        self.position.update_stop_loss(new_stop_loss)

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss has been hit.

        For LONG positions: stop loss hit if price <= stop_loss

        Args:
            current_price: Current price

        Returns:
            True if stop loss hit
        """
        if not self.has_position or self.position.stop_loss is None:
            return False

        # For LONG: exit if price drops to or below stop loss
        return current_price <= self.position.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """
        Check if take profit has been hit.

        For LONG positions: take profit hit if price >= take_profit

        Args:
            current_price: Current price

        Returns:
            True if take profit hit
        """
        if not self.has_position or self.position.take_profit is None:
            return False

        # For LONG: exit if price rises to or above take profit
        return current_price >= self.position.take_profit

    def get_position_value(self, current_price: float) -> float:
        """
        Calculate current position value.

        Args:
            current_price: Current price

        Returns:
            Position value
        """
        if not self.has_position:
            return 0.0

        return self.position.calculate_value(current_price)

    def get_position(self) -> Optional[Position]:
        """
        Get current position.

        Returns:
            Current position or None
        """
        return self.position
