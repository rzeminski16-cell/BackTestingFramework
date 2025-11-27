"""
Configuration classes for the backtesting framework.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class CommissionMode(Enum):
    """Commission calculation mode."""
    PERCENTAGE = "percentage"  # Commission as percentage of trade value
    FIXED = "fixed"            # Fixed commission per trade


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
    """
    initial_capital: float = 100000.0
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    position_size_limit: float = 1.0  # Max 100% of capital per position

    def __post_init__(self):
        """Validate backtest configuration."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.position_size_limit <= 0 or self.position_size_limit > 1.0:
            raise ValueError("Position size limit must be between 0 and 1.0")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")


@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio-level backtesting.

    Attributes:
        initial_capital: Total capital shared across all securities
        commission: Commission configuration
        start_date: Optional start date for backtest
        end_date: Optional end date for backtest
        max_positions: Maximum number of concurrent positions (None = unlimited)
        position_size_limit: Maximum position size as fraction of capital per security
        total_allocation_limit: Maximum total allocation across all positions (default 1.0)
        rebalance_on_exit: Rebalance remaining positions when one closes
    """
    initial_capital: float = 100000.0
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_positions: Optional[int] = None
    position_size_limit: float = 0.3  # Max 30% per position by default
    total_allocation_limit: float = 1.0  # Max 100% total allocation
    rebalance_on_exit: bool = False

    def __post_init__(self):
        """Validate portfolio configuration."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.max_positions is not None and self.max_positions <= 0:
            raise ValueError("Max positions must be positive")
        if self.position_size_limit <= 0 or self.position_size_limit > 1.0:
            raise ValueError("Position size limit must be between 0 and 1.0")
        if self.total_allocation_limit <= 0 or self.total_allocation_limit > 1.0:
            raise ValueError("Total allocation limit must be between 0 and 1.0")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")


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
