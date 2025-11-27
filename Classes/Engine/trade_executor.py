"""
Trade executor for executing orders and tracking trades.
"""
from typing import List
from datetime import datetime
from ..Models.order import Order, OrderSide, OrderType
from ..Models.trade import Trade
from ..Models.position import Position
from ..Config.config import CommissionConfig


class TradeExecutor:
    """
    Executes orders and maintains trade history.

    Handles:
    - Order execution at close prices
    - Commission calculation
    - Trade record creation
    """

    def __init__(self, commission_config: CommissionConfig):
        """
        Initialize trade executor.

        Args:
            commission_config: Commission configuration
        """
        self.commission_config = commission_config
        self.trades: List[Trade] = []

    def execute_order(self, order: Order) -> float:
        """
        Execute an order and return commission paid.

        Args:
            order: Order to execute

        Returns:
            Commission paid on this order
        """
        trade_value = order.total_value()
        commission = self.commission_config.calculate(trade_value)
        return commission

    def create_trade(self, position: Position, exit_date: datetime,
                    exit_price: float, exit_reason: str,
                    exit_commission: float) -> Trade:
        """
        Create a completed trade from a closed position.

        Args:
            position: Closed position
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: Reason for exit
            exit_commission: Commission paid on exit

        Returns:
            Trade record
        """
        total_commission = position.total_commission_paid + exit_commission

        trade = Trade.from_position(
            position=position,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            commission_paid=total_commission
        )

        self.trades.append(trade)
        return trade

    def get_trades(self) -> List[Trade]:
        """
        Get all executed trades.

        Returns:
            List of trades
        """
        return self.trades.copy()

    def get_trade_count(self) -> int:
        """
        Get number of completed trades.

        Returns:
            Number of trades
        """
        return len(self.trades)

    def get_total_pl(self) -> float:
        """
        Calculate total P/L across all trades.

        Returns:
            Total P/L
        """
        return sum(trade.pl for trade in self.trades)

    def get_winning_trades(self) -> List[Trade]:
        """
        Get list of winning trades.

        Returns:
            List of winning trades
        """
        return [t for t in self.trades if t.is_winner]

    def get_losing_trades(self) -> List[Trade]:
        """
        Get list of losing trades.

        Returns:
            List of losing trades
        """
        return [t for t in self.trades if not t.is_winner]

    def reset(self) -> None:
        """Reset trade history."""
        self.trades = []
