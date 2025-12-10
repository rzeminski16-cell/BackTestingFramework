"""
Trade model for completed trades.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid


# Global trade counter for generating sequential IDs
_trade_counter = 0


def generate_trade_id() -> str:
    """Generate a unique trade ID."""
    global _trade_counter
    _trade_counter += 1
    return f"T{_trade_counter:06d}"


def reset_trade_counter():
    """Reset the trade counter (call at start of each backtest)."""
    global _trade_counter
    _trade_counter = 0


@dataclass
class Trade:
    """
    Represents a completed trade.

    Attributes:
        trade_id: Unique identifier for the trade
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
        pl: Profit/Loss in base currency (GBP)
        pl_pct: Profit/Loss as percentage
        security_pl: P/L from security price movement (in base currency)
        fx_pl: P/L from FX rate changes (in base currency)
        entry_fx_rate: FX rate at entry (security currency to base currency)
        exit_fx_rate: FX rate at exit (security currency to base currency)
        security_currency: Currency the security is denominated in
        duration_days: Trade duration in days
        entry_equity: Total portfolio equity at time of entry
        entry_capital_available: Available capital at time of entry
        entry_capital_required: Capital required for this position
        concurrent_positions: Number of other positions open at entry
        competing_signals: List of other signals that were rejected due to this trade
        entry_reason: Reason for entry
        exit_reason: Reason for exit
        commission_paid: Total commission paid
        partial_exits: Number of partial exits before final exit
        metadata: Additional trade-specific data
    """
    trade_id: str = field(default_factory=generate_trade_id)
    symbol: str = ""
    entry_date: datetime = None
    entry_price: float = 0.0
    exit_date: datetime = None
    exit_price: float = 0.0
    quantity: float = 0.0
    side: str = "LONG"
    initial_stop_loss: Optional[float] = None
    final_stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pl: float = 0.0
    pl_pct: float = 0.0
    security_pl: float = 0.0  # P/L from price movement in base currency
    fx_pl: float = 0.0  # P/L from FX rate changes
    entry_fx_rate: float = 1.0  # FX rate at entry
    exit_fx_rate: float = 1.0  # FX rate at exit
    security_currency: str = "GBP"  # Currency security is denominated in
    duration_days: int = 0
    entry_equity: float = 0.0  # Total portfolio equity at time of entry
    entry_capital_available: float = 0.0  # Available capital at entry
    entry_capital_required: float = 0.0  # Capital required for position
    concurrent_positions: int = 0  # Other positions open at entry
    competing_signals: List[str] = field(default_factory=list)  # Symbols with rejected signals
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
            'trade_id': self.trade_id,
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
            'security_pl': self.security_pl,
            'fx_pl': self.fx_pl,
            'entry_fx_rate': self.entry_fx_rate,
            'exit_fx_rate': self.exit_fx_rate,
            'security_currency': self.security_currency,
            'duration_days': self.duration_days,
            'entry_equity': self.entry_equity,
            'entry_capital_available': self.entry_capital_available,
            'entry_capital_required': self.entry_capital_required,
            'concurrent_positions': self.concurrent_positions,
            'competing_signals': ','.join(self.competing_signals) if self.competing_signals else '',
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
            'commission_paid': self.commission_paid,
            'partial_exits': self.partial_exits
        }

    @classmethod
    def from_position(cls, position, exit_date: datetime, exit_price: float,
                     exit_reason: str, commission_paid: float,
                     entry_capital_available: float = 0.0,
                     entry_capital_required: float = 0.0,
                     concurrent_positions: int = 0,
                     competing_signals: List[str] = None) -> 'Trade':
        """
        Create a Trade from a closed Position.

        Args:
            position: Position object
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: Reason for exit
            commission_paid: Total commission paid (entry + exit)
            entry_capital_available: Capital available at entry
            entry_capital_required: Capital required for position
            concurrent_positions: Number of other open positions at entry
            competing_signals: Symbols of signals rejected due to this trade

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
            entry_equity=position.entry_equity,
            entry_capital_available=entry_capital_available,
            entry_capital_required=entry_capital_required,
            concurrent_positions=concurrent_positions,
            competing_signals=competing_signals or [],
            entry_reason=position.entry_reason,
            exit_reason=exit_reason,
            commission_paid=commission_paid,
            partial_exits=len(position.partial_exits)
        )

    def __str__(self) -> str:
        """String representation of trade."""
        fx_str = f", FX P/L: {self.fx_pl:.2f}" if self.fx_pl != 0 else ""
        return (f"[{self.trade_id}] {self.symbol} {self.side}: "
                f"{self.entry_date.strftime('%Y-%m-%d')} -> {self.exit_date.strftime('%Y-%m-%d')}, "
                f"P/L: {self.pl:.2f} ({self.pl_pct:.2f}%){fx_str}")
