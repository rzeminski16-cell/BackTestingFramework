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
                    exit_commission: float,
                    entry_fx_rate: float = 1.0,
                    exit_fx_rate: float = 1.0,
                    security_currency: str = "GBP") -> Trade:
        """
        Create a completed trade from a closed position.

        Args:
            position: Closed position
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: Reason for exit
            exit_commission: Commission paid on exit
            entry_fx_rate: FX rate at entry (security currency to base currency)
            exit_fx_rate: FX rate at exit (security currency to base currency)
            security_currency: Currency the security is denominated in

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

        # Add FX information
        trade.entry_fx_rate = entry_fx_rate
        trade.exit_fx_rate = exit_fx_rate
        trade.security_currency = security_currency

        # Calculate FX P&L breakdown
        # Security P/L in security currency (excluding partial exits for now)
        security_pl_in_sec_currency = (exit_price - position.entry_price) * position.initial_quantity

        # Convert security P/L at entry FX rate (what P/L would be if FX didn't change)
        trade.security_pl = security_pl_in_sec_currency * entry_fx_rate - total_commission

        # FX P/L is the difference between total P/L and security P/L
        # This captures the gain/loss from FX rate movements
        trade.fx_pl = trade.pl - trade.security_pl

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
