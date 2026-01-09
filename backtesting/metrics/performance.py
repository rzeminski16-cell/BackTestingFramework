"""Performance metrics calculation for backtest results."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from backtesting.core.engine import BacktestResult


@dataclass
class MetricDefinition:
    """Definition of a performance metric."""

    name: str
    description: str
    higher_is_better: bool = True
    format_str: str = "{:.2f}"

    def format(self, value: float) -> str:
        """Format the metric value for display."""
        if pd.isna(value):
            return "N/A"
        return self.format_str.format(value)


# Define all available metrics
AVAILABLE_METRICS: Dict[str, MetricDefinition] = {
    "total_return": MetricDefinition(
        name="Total Return",
        description="Total percentage return",
        higher_is_better=True,
        format_str="{:.2f}%",
    ),
    "annual_return": MetricDefinition(
        name="Annual Return",
        description="Annualized percentage return",
        higher_is_better=True,
        format_str="{:.2f}%",
    ),
    "sharpe_ratio": MetricDefinition(
        name="Sharpe Ratio",
        description="Risk-adjusted return (annualized)",
        higher_is_better=True,
        format_str="{:.3f}",
    ),
    "sortino_ratio": MetricDefinition(
        name="Sortino Ratio",
        description="Downside risk-adjusted return",
        higher_is_better=True,
        format_str="{:.3f}",
    ),
    "max_drawdown": MetricDefinition(
        name="Max Drawdown",
        description="Maximum peak-to-trough decline",
        higher_is_better=False,
        format_str="{:.2f}%",
    ),
    "calmar_ratio": MetricDefinition(
        name="Calmar Ratio",
        description="Annual return / Max drawdown",
        higher_is_better=True,
        format_str="{:.3f}",
    ),
    "win_rate": MetricDefinition(
        name="Win Rate",
        description="Percentage of winning trades",
        higher_is_better=True,
        format_str="{:.2f}%",
    ),
    "profit_factor": MetricDefinition(
        name="Profit Factor",
        description="Gross profit / Gross loss",
        higher_is_better=True,
        format_str="{:.2f}",
    ),
    "total_trades": MetricDefinition(
        name="Total Trades",
        description="Number of completed trades",
        higher_is_better=True,
        format_str="{:.0f}",
    ),
    "avg_trade_return": MetricDefinition(
        name="Avg Trade Return",
        description="Average return per trade",
        higher_is_better=True,
        format_str="{:.2f}%",
    ),
    "avg_win": MetricDefinition(
        name="Avg Win",
        description="Average winning trade return",
        higher_is_better=True,
        format_str="{:.2f}%",
    ),
    "avg_loss": MetricDefinition(
        name="Avg Loss",
        description="Average losing trade return",
        higher_is_better=False,
        format_str="{:.2f}%",
    ),
    "max_consecutive_wins": MetricDefinition(
        name="Max Consecutive Wins",
        description="Longest winning streak",
        higher_is_better=True,
        format_str="{:.0f}",
    ),
    "max_consecutive_losses": MetricDefinition(
        name="Max Consecutive Losses",
        description="Longest losing streak",
        higher_is_better=False,
        format_str="{:.0f}",
    ),
    "volatility": MetricDefinition(
        name="Volatility",
        description="Annualized standard deviation of returns",
        higher_is_better=False,
        format_str="{:.2f}%",
    ),
    "expectancy": MetricDefinition(
        name="Expectancy",
        description="Expected return per trade",
        higher_is_better=True,
        format_str="{:.2f}%",
    ),
    "recovery_factor": MetricDefinition(
        name="Recovery Factor",
        description="Net profit / Max drawdown",
        higher_is_better=True,
        format_str="{:.2f}",
    ),
    "avg_trade_duration": MetricDefinition(
        name="Avg Trade Duration",
        description="Average holding period in days",
        higher_is_better=False,  # Usually shorter is better
        format_str="{:.1f} days",
    ),
    "final_equity": MetricDefinition(
        name="Final Equity",
        description="Final portfolio value",
        higher_is_better=True,
        format_str="${:,.2f}",
    ),
}


class PerformanceMetrics:
    """
    Calculate performance metrics from backtest results.

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
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
    ):
        """
        Initialize metrics calculator.

        Args:
            result: BacktestResult from backtest
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days: Trading days per year (252 for stocks, 365 for crypto)
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

    def _calc_total_return(self) -> float:
        """Calculate total return percentage."""
        equity = self.result.equity_curve["equity"]
        if len(equity) < 2:
            return 0.0
        return ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100

    def _calc_annual_return(self) -> float:
        """Calculate annualized return."""
        equity = self.result.equity_curve["equity"]
        if len(equity) < 2:
            return 0.0

        total_return = equity.iloc[-1] / equity.iloc[0]
        days = (self.result.end_date - self.result.start_date).days
        if days <= 0:
            return 0.0

        years = days / 365.0
        annual_return = (total_return ** (1 / years)) - 1
        return annual_return * 100

    def _calc_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.daily_returns) < 2:
            return 0.0

        excess_returns = self.daily_returns - (self.risk_free_rate / self.trading_days)
        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std()
        return sharpe * np.sqrt(self.trading_days)

    def _calc_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(self.daily_returns) < 2:
            return 0.0

        excess_returns = self.daily_returns - (self.risk_free_rate / self.trading_days)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std()
        return sortino * np.sqrt(self.trading_days)

    def _calc_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        equity = self.result.equity_curve["equity"]
        if len(equity) < 2:
            return 0.0

        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        return abs(drawdowns.min())

    def _calc_calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        annual_return = self._calc_annual_return()
        max_dd = self._calc_max_drawdown()

        if max_dd == 0:
            return 0.0

        return annual_return / max_dd

    def _calc_win_rate(self) -> float:
        """Calculate win rate percentage."""
        trades = self.result.trades
        if not trades:
            return 0.0

        winning = sum(1 for t in trades if t.pnl > 0)
        return (winning / len(trades)) * 100

    def _calc_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        trades = self.result.trades
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calc_total_trades(self) -> float:
        """Count total number of trades."""
        return float(len(self.result.trades))

    def _calc_avg_trade_return(self) -> float:
        """Calculate average return per trade."""
        trades = self.result.trades
        if not trades:
            return 0.0

        returns = [t.pnl_percent for t in trades]
        return np.mean(returns)

    def _calc_avg_win(self) -> float:
        """Calculate average winning trade return."""
        trades = self.result.trades
        winning = [t.pnl_percent for t in trades if t.pnl > 0]

        if not winning:
            return 0.0

        return np.mean(winning)

    def _calc_avg_loss(self) -> float:
        """Calculate average losing trade return."""
        trades = self.result.trades
        losing = [t.pnl_percent for t in trades if t.pnl < 0]

        if not losing:
            return 0.0

        return np.mean(losing)

    def _calc_max_consecutive_wins(self) -> float:
        """Calculate longest winning streak."""
        trades = self.result.trades
        if not trades:
            return 0.0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade.pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return float(max_streak)

    def _calc_max_consecutive_losses(self) -> float:
        """Calculate longest losing streak."""
        trades = self.result.trades
        if not trades:
            return 0.0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade.pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return float(max_streak)

    def _calc_volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self.daily_returns) < 2:
            return 0.0

        return self.daily_returns.std() * np.sqrt(self.trading_days) * 100

    def _calc_expectancy(self) -> float:
        """Calculate trade expectancy."""
        win_rate = self._calc_win_rate() / 100
        avg_win = self._calc_avg_win()
        avg_loss = abs(self._calc_avg_loss())

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _calc_recovery_factor(self) -> float:
        """Calculate recovery factor."""
        total_return = self._calc_total_return()
        max_dd = self._calc_max_drawdown()

        if max_dd == 0:
            return 0.0

        return total_return / max_dd

    def _calc_avg_trade_duration(self) -> float:
        """Calculate average trade duration in days."""
        trades = self.result.trades
        if not trades:
            return 0.0

        durations = [(t.exit_date - t.entry_date).days for t in trades]
        return np.mean(durations)

    def _calc_final_equity(self) -> float:
        """Get final portfolio equity."""
        equity = self.result.equity_curve["equity"]
        if len(equity) == 0:
            return 0.0
        return equity.iloc[-1]
