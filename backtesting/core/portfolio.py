"""Portfolio management for backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from backtesting.core.position import Position, PositionSide, Trade


@dataclass
class Portfolio:
    """Manages portfolio state during backtesting."""

    initial_capital: float
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_capital

    @property
    def total_equity(self) -> float:
        """Calculate total portfolio equity."""
        return self.cash

    def get_equity_at_price(self, prices: Dict[str, float]) -> float:
        """Calculate total equity including open positions at given prices."""
        equity = self.cash
        for symbol, position in self.positions.items():
            if symbol in prices:
                equity += position.unrealized_pnl(prices[symbol])
                equity += position.entry_price * position.quantity
        return equity

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        price: float,
        quantity: float,
        date: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Position]:
        """Open a new position."""
        cost = price * quantity

        if cost > self.cash:
            return None

        if symbol in self.positions:
            return None

        self.cash -= cost

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            entry_date=date,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.positions[symbol] = position
        return position

    def close_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
    ) -> Optional[Trade]:
        """Close an existing position."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        position.close(price, date)

        # Return cash + P&L
        self.cash += position.entry_price * position.quantity + position.pnl

        trade = Trade.from_position(position)
        self.closed_trades.append(trade)

        del self.positions[symbol]
        return trade

    def record_equity(self, date: datetime, prices: Dict[str, float]) -> None:
        """Record equity at a point in time."""
        equity = self.get_equity_at_price(prices)
        self.equity_curve.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "positions_value": equity - self.cash,
        })

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(columns=["date", "equity", "cash", "positions_value"])
        return pd.DataFrame(self.equity_curve)

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.equity_curve.clear()
