"""
Configuration classes for the backtesting framework.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime

from .capital_contention import (
    CapitalContentionConfig,
    CapitalContentionMode,
    VulnerabilityScoreConfig
)


class CommissionMode(Enum):
    """Commission calculation mode."""
    PERCENTAGE = "percentage"  # Commission as percentage of trade value
    FIXED = "fixed"            # Fixed commission per trade


class ExecutionTiming(Enum):
    """
    When strategy signals are filled.

    SAME_BAR_CLOSE (default, TradingView-style): a signal computed on bar i's
    close fills at that same close. Optimistic but comparable with all
    historical results produced by this framework.

    NEXT_BAR_OPEN: a signal computed on bar i's close fills at bar i+1's open
    (falling back to that bar's close if no 'open' column exists). More
    realistic — you cannot act on a close you are still forming. Protective
    stop / take-profit checks are unaffected: they model resting orders and
    always evaluate on the current bar.

    Currently supported by the single-security engine; the portfolio engine
    rejects NEXT_BAR_OPEN at run start because its capital-contention
    decisions are tied to same-bar signal processing.
    """
    SAME_BAR_CLOSE = "same_bar_close"
    NEXT_BAR_OPEN = "next_bar_open"


@dataclass
class CommissionConfig:
    """
    Commission configuration.

    Attributes:
        mode: Commission calculation mode (percentage or fixed)
        value: Commission value (e.g., 0.001 for 0.1% or 3.0 for £3)
    """
    mode: CommissionMode = CommissionMode.PERCENTAGE
    value: float = 0.001  # Default: 0.1% commission

    def calculate(self, trade_value: float) -> float:
        """
        Calculate commission for a trade.

        Args:
            trade_value: Total value of the trade (price * quantity)

        Returns:
            Commission amount
        """
        if self.mode == CommissionMode.PERCENTAGE:
            return trade_value * self.value
        else:  # FIXED
            return self.value

    def __post_init__(self):
        """Validate commission configuration."""
        if self.value < 0:
            raise ValueError("Commission value must be non-negative")
        if self.mode == CommissionMode.PERCENTAGE and self.value > 1.0:
            raise ValueError("Percentage commission must be <= 1.0 (100%)")


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    Attributes:
        initial_capital: Starting capital for backtest
        commission: Commission configuration
        start_date: Optional start date for backtest (None = from beginning)
        end_date: Optional end date for backtest (None = to end)
        position_size_limit: Maximum position size as fraction of capital (default 1.0 = 100%)
        base_currency: Base currency for the account (default: GBP)
        slippage_percent: Slippage percentage applied to all trades (default: 0.1%)
        execution_timing: When signals fill (same-bar close, or next-bar open)
        intrabar_stops: If True, stop/take-profit triggers use the bar's
            high/low with gap-aware fills (gap through the level fills at the
            open). Default False preserves close-only evaluation.
    """
    initial_capital: float = 100000.0
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    position_size_limit: float = 1.0  # Max 100% of capital per position
    base_currency: str = "GBP"  # Base currency of account
    slippage_percent: float = 0.1  # Default 0.1% slippage
    execution_timing: ExecutionTiming = ExecutionTiming.SAME_BAR_CLOSE
    intrabar_stops: bool = False

    def __post_init__(self):
        """Validate backtest configuration."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.position_size_limit <= 0 or self.position_size_limit > 1.0:
            raise ValueError("Position size limit must be between 0 and 1.0")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.slippage_percent < 0:
            raise ValueError("Slippage percentage must be non-negative")
        if isinstance(self.execution_timing, str):
            self.execution_timing = ExecutionTiming(self.execution_timing)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'initial_capital': self.initial_capital,
            'commission': {
                'mode': self.commission.mode.value,
                'value': self.commission.value
            },
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'position_size_limit': self.position_size_limit,
            'base_currency': self.base_currency,
            'slippage_percent': self.slippage_percent,
            'execution_timing': self.execution_timing.value,
            'intrabar_stops': self.intrabar_stops
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """Create from dictionary, ignoring unknown keys (for migration)."""
        defaults = cls()
        commission_data = data.get('commission', {})
        commission = CommissionConfig(
            mode=CommissionMode(commission_data.get('mode', CommissionMode.PERCENTAGE.value)),
            value=float(commission_data.get('value', defaults.commission.value)),
        )
        return cls(
            initial_capital=float(data.get('initial_capital', defaults.initial_capital)),
            commission=commission,
            start_date=_parse_iso_datetime(data.get('start_date')),
            end_date=_parse_iso_datetime(data.get('end_date')),
            position_size_limit=float(data.get('position_size_limit', defaults.position_size_limit)),
            base_currency=data.get('base_currency', defaults.base_currency),
            slippage_percent=float(data.get('slippage_percent', defaults.slippage_percent)),
            execution_timing=ExecutionTiming(data.get('execution_timing', defaults.execution_timing.value)),
            intrabar_stops=bool(data.get('intrabar_stops', defaults.intrabar_stops)),
        )


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-format datetime string, passing through None/datetime."""
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio-level backtesting.

    Attributes:
        initial_capital: Total capital shared across all securities
        commission: Commission configuration
        start_date: Optional start date for backtest
        end_date: Optional end date for backtest
        capital_contention: Capital contention configuration (replaces max_positions, position limits)
        base_currency: Base currency for the account (default: GBP)
        slippage_percent: Slippage percentage applied to all trades (default: 0.1%)
        basket_name: Optional name of basket being used (for logging purposes)
        full_isolation: If True, simulate every trade in isolation. Every signal is
            taken (capital availability is ignored) and position size is always
            calculated from the starting equity (initial_capital), so it never
            compounds. Capital contention settings are not applied in this mode.
        randomize_signal_order: If True, same-day BUY signals are processed in a
            random order instead of the deterministic symbol order, so when there
            is not enough capital for all of them, which signals get entered is
            randomised rather than always favouring the first symbols.
        signal_seed: Seed for the signal-order shuffle. The seed is salted with
            each date, so one seed reproduces the entire run exactly while
            different days still draw different orders. None = a different
            shuffle every run.
    """
    initial_capital: float = 100000.0
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    capital_contention: CapitalContentionConfig = field(default_factory=CapitalContentionConfig)
    base_currency: str = "GBP"  # Base currency of account
    slippage_percent: float = 0.1  # Default 0.1% slippage
    basket_name: Optional[str] = None  # Optional basket name for logging
    full_isolation: bool = False  # Take every signal, fixed sizing equity
    execution_timing: ExecutionTiming = ExecutionTiming.SAME_BAR_CLOSE
    intrabar_stops: bool = False  # Stop/TP triggers on bar high/low with gap fills
    randomize_signal_order: bool = False  # Shuffle same-day BUY processing order
    signal_seed: Optional[int] = None  # Shuffle seed (date-salted); None = random

    def __post_init__(self):
        """Validate portfolio configuration."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.slippage_percent < 0:
            raise ValueError("Slippage percentage must be non-negative")
        if isinstance(self.execution_timing, str):
            self.execution_timing = ExecutionTiming(self.execution_timing)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'initial_capital': self.initial_capital,
            'commission': {
                'mode': self.commission.mode.value,
                'value': self.commission.value
            },
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'capital_contention': self.capital_contention.to_dict(),
            'base_currency': self.base_currency,
            'slippage_percent': self.slippage_percent,
            'basket_name': self.basket_name,
            'full_isolation': self.full_isolation,
            'execution_timing': self.execution_timing.value,
            'intrabar_stops': self.intrabar_stops,
            'randomize_signal_order': self.randomize_signal_order,
            'signal_seed': self.signal_seed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioConfig':
        """Create from dictionary, ignoring unknown keys (for migration)."""
        defaults = cls()
        commission_data = data.get('commission', {})
        commission = CommissionConfig(
            mode=CommissionMode(commission_data.get('mode', CommissionMode.PERCENTAGE.value)),
            value=float(commission_data.get('value', defaults.commission.value)),
        )
        return cls(
            initial_capital=float(data.get('initial_capital', defaults.initial_capital)),
            commission=commission,
            start_date=_parse_iso_datetime(data.get('start_date')),
            end_date=_parse_iso_datetime(data.get('end_date')),
            capital_contention=CapitalContentionConfig.from_dict(
                data.get('capital_contention', {})
            ),
            base_currency=data.get('base_currency', defaults.base_currency),
            slippage_percent=float(data.get('slippage_percent', defaults.slippage_percent)),
            basket_name=data.get('basket_name'),
            full_isolation=bool(data.get('full_isolation', defaults.full_isolation)),
            execution_timing=ExecutionTiming(data.get('execution_timing', defaults.execution_timing.value)),
            intrabar_stops=bool(data.get('intrabar_stops', defaults.intrabar_stops)),
            randomize_signal_order=bool(data.get('randomize_signal_order',
                                                 defaults.randomize_signal_order)),
            signal_seed=(int(data['signal_seed'])
                         if data.get('signal_seed') is not None else None),
        )


@dataclass
class OptimizationConfig:
    """
    Configuration for strategy optimization.

    Attributes:
        metric: Metric to optimize ('total_return', 'sharpe_ratio', 'profit_factor', etc.)
        per_security: If True, optimize parameters per security; if False, find global best
        min_trades: Minimum number of trades required for valid optimization
        maximize: If True, maximize metric; if False, minimize (e.g., for max_drawdown)
    """
    metric: str = 'total_return'
    per_security: bool = False
    min_trades: int = 1
    maximize: bool = True

    def __post_init__(self):
        """Validate optimization configuration."""
        valid_metrics = [
            'total_return', 'total_return_pct', 'sharpe_ratio', 'profit_factor',
            'win_rate', 'max_drawdown', 'avg_win', 'avg_loss', 'num_trades'
        ]
        if self.metric not in valid_metrics:
            raise ValueError(f"Metric must be one of: {valid_metrics}")
        if self.min_trades < 0:
            raise ValueError("Minimum trades must be non-negative")
