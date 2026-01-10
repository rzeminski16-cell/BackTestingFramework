"""
Centralized Performance Metrics Module.

This module provides the single source of truth for all performance metric calculations
across the BackTestingFramework. All systems (Backtesting, Optimization, Analysis,
Vulnerability Scoring, etc.) should use these standardized calculations.

STANDARDIZED PARAMETERS:
    - Risk-Free Rate: 3.5% (UK base rate approximation)
    - Trading Days: 252 per year
    - Max Profit Factor: 999.99 (cap for all-winner scenarios)

CONSISTENCY GUARANTEES:
    - All Sharpe ratios use the same risk-free rate and annualization
    - All Sortino ratios use negative excess returns for downside deviation
    - All drawdown calculations use the same peak-to-trough methodology
    - All profit factor calculations handle edge cases identically

Usage:
    from Classes.Core import CentralizedPerformanceMetrics

    # From equity curve and trades
    metrics = CentralizedPerformanceMetrics.calculate_all_metrics(
        equity_curve=df,
        trades=trade_list,
        initial_capital=100000
    )

    # Static calculations for equity curves
    sharpe = CentralizedPerformanceMetrics.calculate_sharpe_ratio(equity_curve)
    sortino = CentralizedPerformanceMetrics.calculate_sortino_ratio(equity_curve)

    # Static calculations for trade lists
    win_rate = CentralizedPerformanceMetrics.calculate_win_rate(trades)
    profit_factor = CentralizedPerformanceMetrics.calculate_profit_factor(trades)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# STANDARDIZED CONSTANTS
# ============================================================================

# Risk-free rate: UK base rate approximation (standardized across all systems)
DEFAULT_RISK_FREE_RATE: float = 0.035  # 3.5% annual

# Trading days per year (standard for stocks)
TRADING_DAYS_PER_YEAR: int = 252

# Maximum profit factor (caps infinity for all-winner scenarios)
MAX_PROFIT_FACTOR: float = 999.99

# Minimum values to avoid division by zero
MIN_STD_DEV: float = 1e-10
MIN_DENOMINATOR: float = 1e-10


# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

@dataclass
class MetricDefinition:
    """
    Definition of a performance metric with formatting and comparison info.

    Attributes:
        name: Display name of the metric
        description: Description of what the metric measures
        higher_is_better: True if higher values indicate better performance
        format_str: Python format string for display
        category: Grouping category (return, risk, trade, ratio)
    """
    name: str
    description: str
    higher_is_better: bool = True
    format_str: str = "{:.2f}"
    category: str = "general"

    def format(self, value: float) -> str:
        """Format the metric value for display."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        try:
            return self.format_str.format(value)
        except (ValueError, TypeError):
            return str(value)


# Comprehensive metric definitions used across the framework
METRIC_DEFINITIONS: Dict[str, MetricDefinition] = {
    # Return Metrics
    "total_return": MetricDefinition(
        name="Total Return",
        description="Total percentage return over the period",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="return"
    ),
    "total_return_dollars": MetricDefinition(
        name="Total Return ($)",
        description="Total dollar return over the period",
        higher_is_better=True,
        format_str="${:,.2f}",
        category="return"
    ),
    "annual_return": MetricDefinition(
        name="Annual Return",
        description="Compound annual growth rate (CAGR)",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="return"
    ),
    "final_equity": MetricDefinition(
        name="Final Equity",
        description="Final portfolio value",
        higher_is_better=True,
        format_str="${:,.2f}",
        category="return"
    ),

    # Risk-Adjusted Return Metrics
    "sharpe_ratio": MetricDefinition(
        name="Sharpe Ratio",
        description=f"Risk-adjusted return (annualized, rf={DEFAULT_RISK_FREE_RATE*100:.1f}%)",
        higher_is_better=True,
        format_str="{:.3f}",
        category="ratio"
    ),
    "sortino_ratio": MetricDefinition(
        name="Sortino Ratio",
        description="Downside risk-adjusted return (annualized)",
        higher_is_better=True,
        format_str="{:.3f}",
        category="ratio"
    ),
    "calmar_ratio": MetricDefinition(
        name="Calmar Ratio",
        description="Annual return / Maximum drawdown",
        higher_is_better=True,
        format_str="{:.3f}",
        category="ratio"
    ),
    "recovery_factor": MetricDefinition(
        name="Recovery Factor",
        description="Net profit / Maximum drawdown",
        higher_is_better=True,
        format_str="{:.2f}",
        category="ratio"
    ),

    # Risk Metrics
    "max_drawdown": MetricDefinition(
        name="Max Drawdown ($)",
        description="Maximum peak-to-trough decline in dollars",
        higher_is_better=False,
        format_str="${:,.2f}",
        category="risk"
    ),
    "max_drawdown_pct": MetricDefinition(
        name="Max Drawdown (%)",
        description="Maximum peak-to-trough decline as percentage",
        higher_is_better=False,
        format_str="{:.2f}%",
        category="risk"
    ),
    "volatility": MetricDefinition(
        name="Volatility",
        description="Annualized standard deviation of returns",
        higher_is_better=False,
        format_str="{:.2f}%",
        category="risk"
    ),
    "downside_deviation": MetricDefinition(
        name="Downside Deviation",
        description="Annualized standard deviation of negative returns",
        higher_is_better=False,
        format_str="{:.2f}%",
        category="risk"
    ),

    # Trade Metrics
    "total_trades": MetricDefinition(
        name="Total Trades",
        description="Number of completed trades",
        higher_is_better=True,
        format_str="{:.0f}",
        category="trade"
    ),
    "win_rate": MetricDefinition(
        name="Win Rate",
        description="Percentage of winning trades",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="trade"
    ),
    "profit_factor": MetricDefinition(
        name="Profit Factor",
        description="Gross profit / Gross loss",
        higher_is_better=True,
        format_str="{:.2f}",
        category="trade"
    ),
    "avg_trade_return": MetricDefinition(
        name="Avg Trade Return",
        description="Average return per trade",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="trade"
    ),
    "avg_win": MetricDefinition(
        name="Avg Win",
        description="Average winning trade return",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="trade"
    ),
    "avg_loss": MetricDefinition(
        name="Avg Loss",
        description="Average losing trade return",
        higher_is_better=False,
        format_str="{:.2f}%",
        category="trade"
    ),
    "largest_win": MetricDefinition(
        name="Largest Win",
        description="Largest single winning trade",
        higher_is_better=True,
        format_str="${:,.2f}",
        category="trade"
    ),
    "largest_loss": MetricDefinition(
        name="Largest Loss",
        description="Largest single losing trade",
        higher_is_better=False,
        format_str="${:,.2f}",
        category="trade"
    ),
    "expectancy": MetricDefinition(
        name="Expectancy",
        description="Expected return per trade",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="trade"
    ),
    "max_consecutive_wins": MetricDefinition(
        name="Max Consecutive Wins",
        description="Longest winning streak",
        higher_is_better=True,
        format_str="{:.0f}",
        category="trade"
    ),
    "max_consecutive_losses": MetricDefinition(
        name="Max Consecutive Losses",
        description="Longest losing streak",
        higher_is_better=False,
        format_str="{:.0f}",
        category="trade"
    ),
    "avg_trade_duration": MetricDefinition(
        name="Avg Trade Duration",
        description="Average holding period in days",
        higher_is_better=False,
        format_str="{:.1f} days",
        category="trade"
    ),

    # Daily Metrics
    "best_day": MetricDefinition(
        name="Best Day",
        description="Best single day return",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="daily"
    ),
    "worst_day": MetricDefinition(
        name="Worst Day",
        description="Worst single day return",
        higher_is_better=False,
        format_str="{:.2f}%",
        category="daily"
    ),

    # Count Metrics
    "num_wins": MetricDefinition(
        name="Winning Trades",
        description="Number of winning trades",
        higher_is_better=True,
        format_str="{:.0f}",
        category="trade"
    ),
    "num_losses": MetricDefinition(
        name="Losing Trades",
        description="Number of losing trades",
        higher_is_better=False,
        format_str="{:.0f}",
        category="trade"
    ),
}


# ============================================================================
# CENTRALIZED PERFORMANCE METRICS CLASS
# ============================================================================

class CentralizedPerformanceMetrics:
    """
    Centralized performance metrics calculator.

    This class provides standardized metric calculations that should be used
    across all systems in the BackTestingFramework. Using this ensures
    consistency in:
        - Risk-free rate (3.5%)
        - Trading days (252)
        - Edge case handling (profit factor caps, zero divisions, etc.)
        - Annualization methodology

    The class provides both:
        - Static methods for individual metric calculations
        - A comprehensive calculate_all_metrics() method for full analysis
    """

    # Class-level constants
    RISK_FREE_RATE = DEFAULT_RISK_FREE_RATE
    TRADING_DAYS = TRADING_DAYS_PER_YEAR
    MAX_PROFIT_FACTOR = MAX_PROFIT_FACTOR

    # ========================================================================
    # EQUITY CURVE METRICS (Static Methods)
    # ========================================================================

    @staticmethod
    def calculate_sharpe_ratio(
        equity_curve: pd.DataFrame,
        risk_free_rate: Optional[float] = None,
        trading_days: Optional[int] = None,
        equity_column: str = 'equity'
    ) -> float:
        """
        Calculate annualized Sharpe ratio from equity curve.

        Formula: (Mean Excess Return / Std Dev of Excess Returns) * sqrt(trading_days)

        Args:
            equity_curve: DataFrame with equity values
            risk_free_rate: Annual risk-free rate (default: 3.5%)
            trading_days: Trading days per year (default: 252)
            equity_column: Name of equity column

        Returns:
            Annualized Sharpe ratio, or 0.0 if calculation not possible
        """
        if risk_free_rate is None:
            risk_free_rate = DEFAULT_RISK_FREE_RATE
        if trading_days is None:
            trading_days = TRADING_DAYS_PER_YEAR

        if len(equity_curve) < 2:
            return 0.0

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0]
        else:
            return 0.0

        # Calculate daily returns
        returns = equity.pct_change().dropna()

        if len(returns) == 0 or returns.std() < MIN_STD_DEV:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1

        # Calculate excess returns
        excess_returns = returns - daily_rf

        # Avoid division by zero
        if excess_returns.std() < MIN_STD_DEV:
            return 0.0

        # Annualized Sharpe ratio
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)

        return float(sharpe) if not np.isnan(sharpe) else 0.0

    @staticmethod
    def calculate_sortino_ratio(
        equity_curve: pd.DataFrame,
        risk_free_rate: Optional[float] = None,
        trading_days: Optional[int] = None,
        equity_column: str = 'equity'
    ) -> float:
        """
        Calculate annualized Sortino ratio from equity curve.

        The Sortino ratio uses downside deviation (standard deviation of negative
        excess returns) instead of total volatility, only penalizing downside risk.

        Formula: (Mean Excess Return / Downside Deviation) * sqrt(trading_days)

        Args:
            equity_curve: DataFrame with equity values
            risk_free_rate: Annual risk-free rate (default: 3.5%)
            trading_days: Trading days per year (default: 252)
            equity_column: Name of equity column

        Returns:
            Annualized Sortino ratio, or 0.0 if calculation not possible
        """
        if risk_free_rate is None:
            risk_free_rate = DEFAULT_RISK_FREE_RATE
        if trading_days is None:
            trading_days = TRADING_DAYS_PER_YEAR

        if len(equity_curve) < 2:
            return 0.0

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0]
        else:
            return 0.0

        # Calculate daily returns
        returns = equity.pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1

        # Calculate excess returns
        excess_returns = returns - daily_rf

        # Downside returns (only negative excess returns)
        downside_returns = excess_returns[excess_returns < 0]

        # Handle case with no negative returns (excellent performance)
        if len(downside_returns) == 0 or downside_returns.std() < MIN_STD_DEV:
            if excess_returns.mean() > 0:
                return 99.99  # Cap at high positive value
            return 0.0

        # Annualized Sortino ratio
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(trading_days)

        return float(sortino) if not np.isnan(sortino) else 0.0

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: pd.DataFrame,
        equity_column: str = 'equity'
    ) -> Tuple[float, float]:
        """
        Calculate maximum drawdown in dollars and percentage.

        Maximum drawdown is the largest peak-to-trough decline during the period.

        Args:
            equity_curve: DataFrame with equity values
            equity_column: Name of equity column

        Returns:
            Tuple of (max_drawdown_dollars, max_drawdown_percent)
        """
        if len(equity_curve) == 0:
            return 0.0, 0.0

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column].values
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0].values
        else:
            return 0.0, 0.0

        # Handle NaN and invalid values
        equity = np.nan_to_num(equity, nan=0.0, posinf=0.0, neginf=0.0)

        if len(equity) == 0 or np.all(equity <= 0):
            return 0.0, 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = running_max - equity
            drawdown_pct = np.where(running_max > 0, (drawdown / running_max) * 100, 0.0)

        # Clean up invalid values
        drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
        drawdown_pct = np.nan_to_num(drawdown_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Cap percentage at 100%
        drawdown_pct = np.clip(drawdown_pct, 0, 100)

        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        max_dd_pct = float(np.max(drawdown_pct)) if len(drawdown_pct) > 0 else 0.0

        return max_dd, max_dd_pct

    @staticmethod
    def calculate_volatility(
        equity_curve: pd.DataFrame,
        trading_days: Optional[int] = None,
        equity_column: str = 'equity'
    ) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).

        Args:
            equity_curve: DataFrame with equity values
            trading_days: Trading days per year (default: 252)
            equity_column: Name of equity column

        Returns:
            Annualized volatility as percentage
        """
        if trading_days is None:
            trading_days = TRADING_DAYS_PER_YEAR

        if len(equity_curve) < 2:
            return 0.0

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0]
        else:
            return 0.0

        returns = equity.pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        return float(returns.std() * np.sqrt(trading_days) * 100)

    @staticmethod
    def calculate_downside_deviation(
        equity_curve: pd.DataFrame,
        trading_days: Optional[int] = None,
        equity_column: str = 'equity'
    ) -> float:
        """
        Calculate annualized downside deviation (std of negative returns only).

        Args:
            equity_curve: DataFrame with equity values
            trading_days: Trading days per year (default: 252)
            equity_column: Name of equity column

        Returns:
            Annualized downside deviation as percentage
        """
        if trading_days is None:
            trading_days = TRADING_DAYS_PER_YEAR

        if len(equity_curve) < 2:
            return 0.0

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0]
        else:
            return 0.0

        returns = equity.pct_change().dropna()
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return 0.0

        return float(negative_returns.std() * np.sqrt(trading_days) * 100)

    @staticmethod
    def calculate_total_return(
        equity_curve: pd.DataFrame,
        equity_column: str = 'equity'
    ) -> Tuple[float, float]:
        """
        Calculate total return in dollars and percentage.

        Args:
            equity_curve: DataFrame with equity values
            equity_column: Name of equity column

        Returns:
            Tuple of (total_return_dollars, total_return_percent)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0]
        else:
            return 0.0, 0.0

        initial = equity.iloc[0]
        final = equity.iloc[-1]

        if initial <= 0:
            return 0.0, 0.0

        dollars = final - initial
        percent = ((final / initial) - 1) * 100

        return float(dollars), float(percent)

    @staticmethod
    def calculate_annual_return(
        equity_curve: pd.DataFrame,
        date_column: str = 'date',
        equity_column: str = 'equity'
    ) -> float:
        """
        Calculate compound annual growth rate (CAGR).

        Args:
            equity_curve: DataFrame with equity and date values
            date_column: Name of date column
            equity_column: Name of equity column

        Returns:
            CAGR as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        else:
            return 0.0

        initial = equity.iloc[0]
        final = equity.iloc[-1]

        if initial <= 0:
            return 0.0

        # Calculate time period
        if date_column in equity_curve.columns:
            start_date = pd.Timestamp(equity_curve[date_column].iloc[0])
            end_date = pd.Timestamp(equity_curve[date_column].iloc[-1])
            days = (end_date - start_date).days
        else:
            # Assume daily data
            days = len(equity_curve)

        if days <= 0:
            return 0.0

        years = days / 365.25

        if years <= 0:
            return 0.0

        cagr = (pow(final / initial, 1 / years) - 1) * 100

        return float(cagr) if not np.isnan(cagr) else 0.0

    @staticmethod
    def calculate_best_day(
        equity_curve: pd.DataFrame,
        equity_column: str = 'equity'
    ) -> float:
        """
        Calculate best single day return.

        Args:
            equity_curve: DataFrame with equity values
            equity_column: Name of equity column

        Returns:
            Best day return as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0]
        else:
            return 0.0

        returns = equity.pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        return float(returns.max() * 100)

    @staticmethod
    def calculate_worst_day(
        equity_curve: pd.DataFrame,
        equity_column: str = 'equity'
    ) -> float:
        """
        Calculate worst single day return.

        Args:
            equity_curve: DataFrame with equity values
            equity_column: Name of equity column

        Returns:
            Worst day return as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column]
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0]
        else:
            return 0.0

        returns = equity.pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        return float(returns.min() * 100)

    # ========================================================================
    # TRADE-BASED METRICS (Static Methods)
    # ========================================================================

    @staticmethod
    def calculate_win_rate(trades: List[Any]) -> float:
        """
        Calculate win rate from trade list.

        Args:
            trades: List of trade objects (must have 'pnl' or 'pl' attribute/key)

        Returns:
            Win rate as percentage (0-100)
        """
        if not trades:
            return 0.0

        # Handle both objects with attributes and dicts
        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        wins = sum(1 for t in trades if get_pnl(t) > 0)
        return (wins / len(trades)) * 100

    @staticmethod
    def calculate_profit_factor(trades: List[Any]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of trade objects

        Returns:
            Profit factor (capped at MAX_PROFIT_FACTOR for all-winner scenarios)
        """
        if not trades:
            return 0.0

        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        gross_profit = sum(get_pnl(t) for t in trades if get_pnl(t) > 0)
        gross_loss = abs(sum(get_pnl(t) for t in trades if get_pnl(t) < 0))

        if gross_loss < MIN_DENOMINATOR:
            return MAX_PROFIT_FACTOR if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_expectancy(trades: List[Any]) -> float:
        """
        Calculate trade expectancy (expected return per trade).

        Formula: (Win Rate * Avg Win) - ((1 - Win Rate) * Avg Loss)

        Args:
            trades: List of trade objects

        Returns:
            Expectancy as percentage
        """
        if not trades:
            return 0.0

        def get_pnl_pct(trade):
            if hasattr(trade, 'pnl_percent'):
                return trade.pnl_percent
            elif hasattr(trade, 'pl_pct'):
                return trade.pl_pct
            elif isinstance(trade, dict):
                return trade.get('pnl_percent', trade.get('pl_pct', 0))
            return 0

        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        wins = [t for t in trades if get_pnl(t) > 0]
        losses = [t for t in trades if get_pnl(t) < 0]

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([get_pnl_pct(t) for t in wins]) if wins else 0
        avg_loss = abs(np.mean([get_pnl_pct(t) for t in losses])) if losses else 0

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    @staticmethod
    def calculate_avg_trade_return(trades: List[Any]) -> float:
        """
        Calculate average trade return percentage.

        Args:
            trades: List of trade objects

        Returns:
            Average trade return as percentage
        """
        if not trades:
            return 0.0

        def get_pnl_pct(trade):
            if hasattr(trade, 'pnl_percent'):
                return trade.pnl_percent
            elif hasattr(trade, 'pl_pct'):
                return trade.pl_pct
            elif isinstance(trade, dict):
                return trade.get('pnl_percent', trade.get('pl_pct', 0))
            return 0

        return float(np.mean([get_pnl_pct(t) for t in trades]))

    @staticmethod
    def calculate_avg_win(trades: List[Any]) -> float:
        """
        Calculate average winning trade return percentage.

        Args:
            trades: List of trade objects

        Returns:
            Average win as percentage
        """
        if not trades:
            return 0.0

        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        def get_pnl_pct(trade):
            if hasattr(trade, 'pnl_percent'):
                return trade.pnl_percent
            elif hasattr(trade, 'pl_pct'):
                return trade.pl_pct
            elif isinstance(trade, dict):
                return trade.get('pnl_percent', trade.get('pl_pct', 0))
            return 0

        winners = [t for t in trades if get_pnl(t) > 0]

        if not winners:
            return 0.0

        return float(np.mean([get_pnl_pct(t) for t in winners]))

    @staticmethod
    def calculate_avg_loss(trades: List[Any]) -> float:
        """
        Calculate average losing trade return percentage.

        Args:
            trades: List of trade objects

        Returns:
            Average loss as percentage (negative value)
        """
        if not trades:
            return 0.0

        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        def get_pnl_pct(trade):
            if hasattr(trade, 'pnl_percent'):
                return trade.pnl_percent
            elif hasattr(trade, 'pl_pct'):
                return trade.pl_pct
            elif isinstance(trade, dict):
                return trade.get('pnl_percent', trade.get('pl_pct', 0))
            return 0

        losers = [t for t in trades if get_pnl(t) < 0]

        if not losers:
            return 0.0

        return float(np.mean([get_pnl_pct(t) for t in losers]))

    @staticmethod
    def calculate_max_consecutive_wins(trades: List[Any]) -> int:
        """
        Calculate longest winning streak.

        Args:
            trades: List of trade objects (in chronological order)

        Returns:
            Maximum consecutive winning trades
        """
        if not trades:
            return 0

        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if get_pnl(trade) > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def calculate_max_consecutive_losses(trades: List[Any]) -> int:
        """
        Calculate longest losing streak.

        Args:
            trades: List of trade objects (in chronological order)

        Returns:
            Maximum consecutive losing trades
        """
        if not trades:
            return 0

        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if get_pnl(trade) < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def calculate_avg_trade_duration(trades: List[Any]) -> float:
        """
        Calculate average trade duration in days.

        Args:
            trades: List of trade objects

        Returns:
            Average duration in days
        """
        if not trades:
            return 0.0

        durations = []

        for trade in trades:
            if hasattr(trade, 'duration_days'):
                durations.append(trade.duration_days)
            elif hasattr(trade, 'entry_date') and hasattr(trade, 'exit_date'):
                duration = (trade.exit_date - trade.entry_date).days
                durations.append(duration)
            elif isinstance(trade, dict):
                if 'duration_days' in trade:
                    durations.append(trade['duration_days'])
                elif 'entry_date' in trade and 'exit_date' in trade:
                    duration = (trade['exit_date'] - trade['entry_date']).days
                    durations.append(duration)

        return float(np.mean(durations)) if durations else 0.0

    # ========================================================================
    # COMPREHENSIVE METRIC CALCULATION
    # ========================================================================

    @classmethod
    def calculate_all_metrics(
        cls,
        equity_curve: Optional[pd.DataFrame] = None,
        trades: Optional[List[Any]] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: Optional[float] = None,
        trading_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available performance metrics.

        This is the main method for comprehensive metric calculation. It handles
        both equity curve-based and trade-based metrics, properly managing
        edge cases where one or both inputs may be missing.

        Args:
            equity_curve: DataFrame with 'equity' column (optional)
            trades: List of trade objects (optional)
            initial_capital: Initial capital for percentage calculations
            risk_free_rate: Annual risk-free rate (default: 3.5%)
            trading_days: Trading days per year (default: 252)

        Returns:
            Dictionary of all calculated metrics
        """
        if risk_free_rate is None:
            risk_free_rate = DEFAULT_RISK_FREE_RATE
        if trading_days is None:
            trading_days = TRADING_DAYS_PER_YEAR

        metrics = {}

        # ----------------------------------------------------------------
        # Equity Curve Metrics
        # ----------------------------------------------------------------
        if equity_curve is not None and len(equity_curve) > 0:
            # Sharpe and Sortino
            metrics['sharpe_ratio'] = cls.calculate_sharpe_ratio(
                equity_curve, risk_free_rate, trading_days
            )
            metrics['sortino_ratio'] = cls.calculate_sortino_ratio(
                equity_curve, risk_free_rate, trading_days
            )

            # Max Drawdown
            max_dd, max_dd_pct = cls.calculate_max_drawdown(equity_curve)
            metrics['max_drawdown'] = max_dd
            metrics['max_drawdown_pct'] = max_dd_pct

            # Volatility
            metrics['volatility'] = cls.calculate_volatility(equity_curve, trading_days)
            metrics['downside_deviation'] = cls.calculate_downside_deviation(
                equity_curve, trading_days
            )

            # Returns
            total_ret_dollars, total_ret_pct = cls.calculate_total_return(equity_curve)
            metrics['total_return_dollars'] = total_ret_dollars
            metrics['total_return'] = total_ret_pct
            metrics['annual_return'] = cls.calculate_annual_return(equity_curve)

            # Final equity
            if 'equity' in equity_curve.columns:
                metrics['final_equity'] = float(equity_curve['equity'].iloc[-1])

            # Calmar ratio
            if max_dd_pct > 0:
                metrics['calmar_ratio'] = metrics['annual_return'] / max_dd_pct
            else:
                metrics['calmar_ratio'] = 0.0

            # Recovery factor
            if max_dd_pct > 0:
                metrics['recovery_factor'] = total_ret_pct / max_dd_pct
            else:
                metrics['recovery_factor'] = 0.0

            # Daily extremes
            metrics['best_day'] = cls.calculate_best_day(equity_curve)
            metrics['worst_day'] = cls.calculate_worst_day(equity_curve)
        else:
            # Set defaults for equity-based metrics
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
            metrics['max_drawdown'] = 0.0
            metrics['max_drawdown_pct'] = 0.0
            metrics['volatility'] = 0.0
            metrics['downside_deviation'] = 0.0
            metrics['total_return_dollars'] = 0.0
            metrics['total_return'] = 0.0
            metrics['annual_return'] = 0.0
            metrics['final_equity'] = initial_capital
            metrics['calmar_ratio'] = 0.0
            metrics['recovery_factor'] = 0.0
            metrics['best_day'] = 0.0
            metrics['worst_day'] = 0.0

        # ----------------------------------------------------------------
        # Trade Metrics
        # ----------------------------------------------------------------
        if trades is not None and len(trades) > 0:
            metrics['total_trades'] = len(trades)
            metrics['win_rate'] = cls.calculate_win_rate(trades)
            metrics['profit_factor'] = cls.calculate_profit_factor(trades)
            metrics['expectancy'] = cls.calculate_expectancy(trades)
            metrics['avg_trade_return'] = cls.calculate_avg_trade_return(trades)
            metrics['avg_win'] = cls.calculate_avg_win(trades)
            metrics['avg_loss'] = cls.calculate_avg_loss(trades)
            metrics['max_consecutive_wins'] = cls.calculate_max_consecutive_wins(trades)
            metrics['max_consecutive_losses'] = cls.calculate_max_consecutive_losses(trades)
            metrics['avg_trade_duration'] = cls.calculate_avg_trade_duration(trades)

            # Count wins/losses
            def get_pnl(trade):
                if hasattr(trade, 'pnl'):
                    return trade.pnl
                elif hasattr(trade, 'pl'):
                    return trade.pl
                elif isinstance(trade, dict):
                    return trade.get('pnl', trade.get('pl', 0))
                return 0

            metrics['num_wins'] = sum(1 for t in trades if get_pnl(t) > 0)
            metrics['num_losses'] = sum(1 for t in trades if get_pnl(t) < 0)

            # Largest win/loss
            pnls = [get_pnl(t) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            metrics['largest_win'] = max(wins) if wins else 0.0
            metrics['largest_loss'] = min(losses) if losses else 0.0
        else:
            # Set defaults for trade-based metrics
            metrics['total_trades'] = 0
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['expectancy'] = 0.0
            metrics['avg_trade_return'] = 0.0
            metrics['avg_win'] = 0.0
            metrics['avg_loss'] = 0.0
            metrics['max_consecutive_wins'] = 0
            metrics['max_consecutive_losses'] = 0
            metrics['avg_trade_duration'] = 0.0
            metrics['num_wins'] = 0
            metrics['num_losses'] = 0
            metrics['largest_win'] = 0.0
            metrics['largest_loss'] = 0.0

        return metrics

    @classmethod
    def calculate_from_trades_only(
        cls,
        trades: List[Any],
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Calculate metrics from trades only (no equity curve).

        This method constructs an equity curve from trades and calculates
        all available metrics. Useful for systems that only have trade data.

        Args:
            trades: List of trade objects
            initial_capital: Initial capital

        Returns:
            Dictionary of all calculated metrics
        """
        if not trades:
            return cls.calculate_all_metrics(None, None, initial_capital)

        # Sort trades by exit date
        def get_exit_date(trade):
            if hasattr(trade, 'exit_date'):
                return trade.exit_date
            elif isinstance(trade, dict):
                return trade.get('exit_date', datetime.now())
            return datetime.now()

        def get_pnl(trade):
            if hasattr(trade, 'pnl'):
                return trade.pnl
            elif hasattr(trade, 'pl'):
                return trade.pl
            elif isinstance(trade, dict):
                return trade.get('pnl', trade.get('pl', 0))
            return 0

        sorted_trades = sorted(trades, key=get_exit_date)

        # Build equity curve from trades
        equity_points = [{'date': get_exit_date(sorted_trades[0]), 'equity': initial_capital}]
        cumulative_pnl = 0

        for trade in sorted_trades:
            cumulative_pnl += get_pnl(trade)
            equity_points.append({
                'date': get_exit_date(trade),
                'equity': initial_capital + cumulative_pnl
            })

        equity_curve = pd.DataFrame(equity_points)

        return cls.calculate_all_metrics(equity_curve, trades, initial_capital)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @staticmethod
    def get_metric_definition(metric_name: str) -> Optional[MetricDefinition]:
        """
        Get the definition for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            MetricDefinition or None if not found
        """
        return METRIC_DEFINITIONS.get(metric_name)

    @staticmethod
    def get_available_metrics() -> List[str]:
        """Get list of all available metric names."""
        return list(METRIC_DEFINITIONS.keys())

    @staticmethod
    def format_metric(metric_name: str, value: float) -> str:
        """
        Format a metric value for display.

        Args:
            metric_name: Name of the metric
            value: Value to format

        Returns:
            Formatted string
        """
        definition = METRIC_DEFINITIONS.get(metric_name)
        if definition:
            return definition.format(value)
        return f"{value:.2f}"

    @classmethod
    def get_summary_text(
        cls,
        metrics: Dict[str, Any],
        title: str = "Performance Summary"
    ) -> str:
        """
        Generate formatted summary text from metrics dictionary.

        Args:
            metrics: Dictionary of metric values
            title: Title for the summary

        Returns:
            Formatted multi-line string
        """
        lines = [
            title,
            "=" * 60,
            "",
            "Returns:",
            f"  Total Return:      {cls.format_metric('total_return', metrics.get('total_return', 0))}",
            f"  Annual Return:     {cls.format_metric('annual_return', metrics.get('annual_return', 0))}",
            f"  Final Equity:      {cls.format_metric('final_equity', metrics.get('final_equity', 0))}",
            "",
            "Risk Metrics:",
            f"  Max Drawdown:      {cls.format_metric('max_drawdown_pct', metrics.get('max_drawdown_pct', 0))}",
            f"  Volatility:        {cls.format_metric('volatility', metrics.get('volatility', 0))}",
            "",
            "Risk-Adjusted Returns:",
            f"  Sharpe Ratio:      {cls.format_metric('sharpe_ratio', metrics.get('sharpe_ratio', 0))}",
            f"  Sortino Ratio:     {cls.format_metric('sortino_ratio', metrics.get('sortino_ratio', 0))}",
            f"  Calmar Ratio:      {cls.format_metric('calmar_ratio', metrics.get('calmar_ratio', 0))}",
            "",
            "Trade Statistics:",
            f"  Total Trades:      {cls.format_metric('total_trades', metrics.get('total_trades', 0))}",
            f"  Win Rate:          {cls.format_metric('win_rate', metrics.get('win_rate', 0))}",
            f"  Profit Factor:     {cls.format_metric('profit_factor', metrics.get('profit_factor', 0))}",
            f"  Expectancy:        {cls.format_metric('expectancy', metrics.get('expectancy', 0))}",
            "=" * 60,
        ]

        return "\n".join(lines)
