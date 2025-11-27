"""
Order model for trade execution.
"""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"  # Execute at current price


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """
    Order to be executed.

    Attributes:
        symbol: Security symbol
        side: Order side (BUY or SELL)
        quantity: Number of shares/units
        order_type: Order type (MARKET)
        price: Price at which order should execute
        timestamp: Order creation time
        stop_loss: Stop loss price (for new positions)
        take_profit: Take profit price (for new positions)
        reason: Reason for order
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""

    def total_value(self) -> float:
        """
        Calculate total order value (excluding commission).

        Returns:
            Order value (price * quantity)
        """
        return self.price * self.quantity

    def __str__(self) -> str:
        """String representation of order."""
        return (f"{self.side.value} {self.quantity:.4f} {self.symbol} "
                f"@ {self.price:.2f} ({self.reason})")
