"""
Backtest result container.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from ..Models.trade import Trade


@dataclass
class BacktestResult:
    """
    Container for backtest results.

    Attributes:
        symbol: Security symbol
        strategy_name: Strategy name
        trades: List of completed trades
        equity_curve: DataFrame with equity over time
        final_equity: Final equity value
        total_return: Total return in currency
        total_return_pct: Total return as percentage
        strategy_params: Dictionary of strategy parameters used in backtest
    """
    symbol: str
    strategy_name: str
    trades: List[Trade]
    equity_curve: pd.DataFrame
    final_equity: float
    total_return: float
    total_return_pct: float
    strategy_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    @property
    def num_trades(self) -> int:
        """Get number of trades."""
        return len(self.trades)

    @property
    def winning_trades(self) -> List[Trade]:
        """Get list of winning trades."""
        return [t for t in self.trades if t.is_winner]

    @property
    def losing_trades(self) -> List[Trade]:
        """Get list of losing trades."""
        return [t for t in self.trades if not t.is_winner]

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.num_trades == 0:
            return 0.0
        return len(self.winning_trades) / self.num_trades

    def __str__(self) -> str:
        """String representation."""
        return (f"BacktestResult({self.symbol}, {self.strategy_name}): "
                f"{self.num_trades} trades, "
                f"Return: {self.total_return:.2f} ({self.total_return_pct:.2f}%)")
