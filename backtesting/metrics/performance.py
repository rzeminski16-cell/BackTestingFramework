"""
Performance metrics calculation for backtest results.

This module provides a wrapper around the centralized performance metrics
to maintain backward compatibility with existing code while using the
standardized calculations from Classes.Core.performance_metrics.

All metric calculations are now delegated to CentralizedPerformanceMetrics
to ensure consistency across the entire framework.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
import pandas as pd
import numpy as np

# Import from centralized metrics module
from Classes.Core.performance_metrics import (
    CentralizedPerformanceMetrics,
    MetricDefinition,
    METRIC_DEFINITIONS as CENTRALIZED_METRICS,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    MAX_PROFIT_FACTOR,
)

if TYPE_CHECKING:
    from backtesting.core.engine import BacktestResult


# Re-export MetricDefinition for backward compatibility
# (already imported from centralized module)


# Map the centralized metrics to the old format for backward compatibility
AVAILABLE_METRICS: Dict[str, MetricDefinition] = {
    "total_return": CENTRALIZED_METRICS.get("total_return", MetricDefinition(
        name="Total Return",
        description="Total percentage return",
        higher_is_better=True,
        format_str="{:.2f}%",
    )),
    "annual_return": CENTRALIZED_METRICS.get("annual_return", MetricDefinition(
        name="Annual Return",
        description="Annualized percentage return",
        higher_is_better=True,
        format_str="{:.2f}%",
    )),
    "sharpe_ratio": CENTRALIZED_METRICS.get("sharpe_ratio", MetricDefinition(
        name="Sharpe Ratio",
        description=f"Risk-adjusted return (annualized, rf={DEFAULT_RISK_FREE_RATE*100:.1f}%)",
        higher_is_better=True,
        format_str="{:.3f}",
    )),
    "sortino_ratio": CENTRALIZED_METRICS.get("sortino_ratio", MetricDefinition(
        name="Sortino Ratio",
        description="Downside risk-adjusted return",
        higher_is_better=True,
        format_str="{:.3f}",
    )),
    "max_drawdown": CENTRALIZED_METRICS.get("max_drawdown_pct", MetricDefinition(
        name="Max Drawdown",
        description="Maximum peak-to-trough decline",
        higher_is_better=False,
        format_str="{:.2f}%",
    )),
    "calmar_ratio": CENTRALIZED_METRICS.get("calmar_ratio", MetricDefinition(
        name="Calmar Ratio",
        description="Annual return / Max drawdown",
        higher_is_better=True,
        format_str="{:.3f}",
    )),
    "win_rate": CENTRALIZED_METRICS.get("win_rate", MetricDefinition(
        name="Win Rate",
        description="Percentage of winning trades",
        higher_is_better=True,
        format_str="{:.2f}%",
    )),
    "profit_factor": CENTRALIZED_METRICS.get("profit_factor", MetricDefinition(
        name="Profit Factor",
        description="Gross profit / Gross loss",
        higher_is_better=True,
        format_str="{:.2f}",
    )),
    "total_trades": CENTRALIZED_METRICS.get("total_trades", MetricDefinition(
        name="Total Trades",
        description="Number of completed trades",
        higher_is_better=True,
        format_str="{:.0f}",
    )),
    "avg_trade_return": CENTRALIZED_METRICS.get("avg_trade_return", MetricDefinition(
        name="Avg Trade Return",
        description="Average return per trade",
        higher_is_better=True,
        format_str="{:.2f}%",
    )),
    "avg_win": CENTRALIZED_METRICS.get("avg_win", MetricDefinition(
        name="Avg Win",
        description="Average winning trade return",
        higher_is_better=True,
        format_str="{:.2f}%",
    )),
    "avg_loss": CENTRALIZED_METRICS.get("avg_loss", MetricDefinition(
        name="Avg Loss",
        description="Average losing trade return",
        higher_is_better=False,
        format_str="{:.2f}%",
    )),
    "max_consecutive_wins": CENTRALIZED_METRICS.get("max_consecutive_wins", MetricDefinition(
        name="Max Consecutive Wins",
        description="Longest winning streak",
        higher_is_better=True,
        format_str="{:.0f}",
    )),
    "max_consecutive_losses": CENTRALIZED_METRICS.get("max_consecutive_losses", MetricDefinition(
        name="Max Consecutive Losses",
        description="Longest losing streak",
        higher_is_better=False,
        format_str="{:.0f}",
    )),
    "volatility": CENTRALIZED_METRICS.get("volatility", MetricDefinition(
        name="Volatility",
        description="Annualized standard deviation of returns",
        higher_is_better=False,
        format_str="{:.2f}%",
    )),
    "expectancy": CENTRALIZED_METRICS.get("expectancy", MetricDefinition(
        name="Expectancy",
        description="Expected return per trade",
        higher_is_better=True,
        format_str="{:.2f}%",
    )),
    "recovery_factor": CENTRALIZED_METRICS.get("recovery_factor", MetricDefinition(
        name="Recovery Factor",
        description="Net profit / Max drawdown",
        higher_is_better=True,
        format_str="{:.2f}",
    )),
    "avg_trade_duration": CENTRALIZED_METRICS.get("avg_trade_duration", MetricDefinition(
        name="Avg Trade Duration",
        description="Average holding period in days",
        higher_is_better=False,
        format_str="{:.1f} days",
    )),
    "final_equity": CENTRALIZED_METRICS.get("final_equity", MetricDefinition(
        name="Final Equity",
        description="Final portfolio value",
        higher_is_better=True,
        format_str="${:,.2f}",
    )),
}


class PerformanceMetrics:
    """
    Calculate performance metrics from backtest results.

    This class now delegates all calculations to CentralizedPerformanceMetrics
    for consistency across the framework while maintaining backward compatibility.

    IMPORTANT: Risk-free rate is standardized at 3.5% (UK base rate approximation)
    across all systems for consistency. The risk_free_rate parameter is kept for
    backward compatibility but defaults to the standardized value.

    Example:
        metrics = PerformanceMetrics(result)
        print(metrics.get("sharpe_ratio"))
        print(metrics.calculate_all())
    """

    # Default metrics to calculate
    DEFAULT_METRICS = [
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
    ]

    def __init__(
        self,
        result: "BacktestResult",
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,  # Standardized at 3.5%
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ):
        """
        Initialize metrics calculator.

        Args:
            result: BacktestResult from backtest
            risk_free_rate: Annual risk-free rate (default: 3.5% - standardized)
            trading_days: Trading days per year (default: 252)
        """
        self.result = result
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

        # Cache computed values
        self._cache: Dict[str, float] = {}

        # Calculate daily returns
        equity = result.equity_curve["equity"]
        self.daily_returns = equity.pct_change().dropna()

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of all available metric names."""
        return list(AVAILABLE_METRICS.keys())

    @classmethod
    def get_metric_definition(cls, metric_name: str) -> MetricDefinition:
        """Get the definition for a metric."""
        if metric_name not in AVAILABLE_METRICS:
            raise ValueError(f"Unknown metric: {metric_name}")
        return AVAILABLE_METRICS[metric_name]

    def get(self, metric_name: str) -> float:
        """
        Calculate and return a specific metric.

        Args:
            metric_name: Name of the metric to calculate

        Returns:
            Calculated metric value
        """
        if metric_name in self._cache:
            return self._cache[metric_name]

        method_name = f"_calc_{metric_name}"
        if hasattr(self, method_name):
            value = getattr(self, method_name)()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        self._cache[metric_name] = value
        return value

    def calculate_selected(self, metrics: List[str]) -> Dict[str, float]:
        """
        Calculate selected metrics.

        Args:
            metrics: List of metric names to calculate

        Returns:
            Dictionary of metric name -> value
        """
        return {m: self.get(m) for m in metrics}

    def calculate_all(self) -> Dict[str, float]:
        """Calculate all available metrics."""
        return self.calculate_selected(list(AVAILABLE_METRICS.keys()))

    def calculate_default(self) -> Dict[str, float]:
        """Calculate default set of metrics."""
        return self.calculate_selected(self.DEFAULT_METRICS)

    def format_metric(self, metric_name: str, value: float) -> str:
        """Format a metric value for display."""
        defn = AVAILABLE_METRICS.get(metric_name)
        if defn:
            return defn.format(value)
        return f"{value:.2f}"

    def summary(self, metrics: Optional[List[str]] = None) -> str:
        """Generate a formatted summary of metrics."""
        if metrics is None:
            metrics = self.DEFAULT_METRICS

        lines = [f"Performance Summary for {self.result.strategy_name}", "=" * 50]

        for metric_name in metrics:
            value = self.get(metric_name)
            defn = AVAILABLE_METRICS[metric_name]
            formatted = defn.format(value)
            lines.append(f"{defn.name:25s}: {formatted}")

        return "\n".join(lines)

    # === Metric Calculation Methods ===
    # These methods now delegate to CentralizedPerformanceMetrics for consistency

    def _calc_total_return(self) -> float:
        """Calculate total return percentage using centralized method."""
        _, total_return_pct = CentralizedPerformanceMetrics.calculate_total_return(
            self.result.equity_curve
        )
        return total_return_pct

    def _calc_annual_return(self) -> float:
        """Calculate annualized return using centralized method."""
        return CentralizedPerformanceMetrics.calculate_annual_return(
            self.result.equity_curve
        )

    def _calc_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio using centralized method."""
        return CentralizedPerformanceMetrics.calculate_sharpe_ratio(
            self.result.equity_curve,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days
        )

    def _calc_sortino_ratio(self) -> float:
        """Calculate Sortino ratio using centralized method."""
        return CentralizedPerformanceMetrics.calculate_sortino_ratio(
            self.result.equity_curve,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days
        )

    def _calc_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage using centralized method."""
        _, max_dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(
            self.result.equity_curve
        )
        return max_dd_pct

    def _calc_calmar_ratio(self) -> float:
        """Calculate Calmar ratio using centralized method."""
        annual_return = self._calc_annual_return()
        max_dd = self._calc_max_drawdown()

        if max_dd == 0:
            return 0.0

        return annual_return / max_dd

    def _calc_win_rate(self) -> float:
        """Calculate win rate percentage using centralized method."""
        return CentralizedPerformanceMetrics.calculate_win_rate(self.result.trades)

    def _calc_profit_factor(self) -> float:
        """Calculate profit factor using centralized method."""
        return CentralizedPerformanceMetrics.calculate_profit_factor(self.result.trades)

    def _calc_total_trades(self) -> float:
        """Count total number of trades."""
        return float(len(self.result.trades))

    def _calc_avg_trade_return(self) -> float:
        """Calculate average return per trade using centralized method."""
        return CentralizedPerformanceMetrics.calculate_avg_trade_return(self.result.trades)

    def _calc_avg_win(self) -> float:
        """Calculate average winning trade return using centralized method."""
        return CentralizedPerformanceMetrics.calculate_avg_win(self.result.trades)

    def _calc_avg_loss(self) -> float:
        """Calculate average losing trade return using centralized method."""
        return CentralizedPerformanceMetrics.calculate_avg_loss(self.result.trades)

    def _calc_max_consecutive_wins(self) -> float:
        """Calculate longest winning streak using centralized method."""
        return float(CentralizedPerformanceMetrics.calculate_max_consecutive_wins(
            self.result.trades
        ))

    def _calc_max_consecutive_losses(self) -> float:
        """Calculate longest losing streak using centralized method."""
        return float(CentralizedPerformanceMetrics.calculate_max_consecutive_losses(
            self.result.trades
        ))

    def _calc_volatility(self) -> float:
        """Calculate annualized volatility using centralized method."""
        return CentralizedPerformanceMetrics.calculate_volatility(
            self.result.equity_curve,
            trading_days=self.trading_days
        )

    def _calc_expectancy(self) -> float:
        """Calculate trade expectancy using centralized method."""
        return CentralizedPerformanceMetrics.calculate_expectancy(self.result.trades)

    def _calc_recovery_factor(self) -> float:
        """Calculate recovery factor using centralized method."""
        total_return = self._calc_total_return()
        max_dd = self._calc_max_drawdown()

        if max_dd == 0:
            return 0.0

        return total_return / max_dd

    def _calc_avg_trade_duration(self) -> float:
        """Calculate average trade duration using centralized method."""
        return CentralizedPerformanceMetrics.calculate_avg_trade_duration(
            self.result.trades
        )

    def _calc_final_equity(self) -> float:
        """Get final portfolio equity."""
        equity = self.result.equity_curve["equity"]
        if len(equity) == 0:
            return 0.0
        return equity.iloc[-1]
