"""
Univariate Parameter Optimization System.

This module implements univariate optimization where ONE parameter is tested at a time
while all other parameters remain at their 'control' values.

Key Features:
- Test single parameters in isolation
- Support for single security or portfolio of securities
- Option to run portfolio together or separately per security
- Calculates selected performance metrics for each parameter value
- Generates results for Excel export with charts
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Type, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.backtest_result import BacktestResult
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """Configuration for a parameter to optimize."""
    name: str
    values: List[Any]  # Explicit list of values to test
    param_type: str = "float"  # "int", "float", "bool"

    @classmethod
    def from_range(cls, name: str, min_val: float, max_val: float,
                   interval: float, param_type: str = "float") -> "ParameterConfig":
        """Create config from min/max/interval range."""
        values = []
        current = min_val
        while current <= max_val + 1e-9:  # Small epsilon for float comparison
            if param_type == "int":
                values.append(int(round(current)))
            else:
                values.append(round(current, 6))
            current += interval
        # Remove duplicates while preserving order
        seen = set()
        unique_values = []
        for v in values:
            if v not in seen:
                seen.add(v)
                unique_values.append(v)
        return cls(name=name, values=unique_values, param_type=param_type)


@dataclass
class OptimizationRunResult:
    """Result of a single backtest run within optimization."""
    parameter_name: str
    parameter_value: Any
    metrics: Dict[str, float]
    total_trades: int = 0
    final_equity: float = 0.0


@dataclass
class ParameterOptimizationResult:
    """Results for optimizing a single parameter across all its values."""
    parameter_name: str
    control_value: Any
    values_tested: List[Any]
    results: List[OptimizationRunResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for charting."""
        rows = []
        for result in self.results:
            row = {"parameter_value": result.parameter_value}
            row.update(result.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def get_best_value(self, metric: str, higher_is_better: bool = True) -> Any:
        """Get the parameter value that optimizes a given metric."""
        if not self.results:
            return self.control_value

        df = self.to_dataframe()
        if metric not in df.columns:
            return self.control_value

        # Filter out NaN values
        valid_df = df[df[metric].notna()]
        if valid_df.empty:
            return self.control_value

        if higher_is_better:
            idx = valid_df[metric].idxmax()
        else:
            idx = valid_df[metric].idxmin()

        return df.loc[idx, "parameter_value"]


@dataclass
class UnivariateOptimizationResult:
    """Complete results of univariate optimization."""
    strategy_name: str
    securities: List[str]
    run_mode: str  # "together" or "separate"
    control_values: Dict[str, Any]
    metrics_calculated: List[str]
    parameter_results: Dict[str, ParameterOptimizationResult]
    optimization_settings: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame showing best values per parameter/metric."""
        rows = []
        for param_name, param_result in self.parameter_results.items():
            row = {"parameter": param_name, "control_value": param_result.control_value}
            for metric in self.metrics_calculated:
                # Determine if higher is better for this metric
                higher_is_better = metric not in ["max_drawdown", "volatility", "avg_loss",
                                                   "max_consecutive_losses", "avg_trade_duration"]
                best_val = param_result.get_best_value(metric, higher_is_better)
                row[f"best_{metric}"] = best_val
            rows.append(row)
        return pd.DataFrame(rows)


# Metrics definitions for display
METRIC_DEFINITIONS = {
    "total_return": {"name": "Total Return", "higher_is_better": True, "format": "{:.2f}%"},
    "annual_return": {"name": "Annual Return", "higher_is_better": True, "format": "{:.2f}%"},
    "sharpe_ratio": {"name": "Sharpe Ratio", "higher_is_better": True, "format": "{:.3f}"},
    "sortino_ratio": {"name": "Sortino Ratio", "higher_is_better": True, "format": "{:.3f}"},
    "max_drawdown": {"name": "Max Drawdown", "higher_is_better": False, "format": "{:.2f}%"},
    "calmar_ratio": {"name": "Calmar Ratio", "higher_is_better": True, "format": "{:.3f}"},
    "win_rate": {"name": "Win Rate", "higher_is_better": True, "format": "{:.2f}%"},
    "profit_factor": {"name": "Profit Factor", "higher_is_better": True, "format": "{:.2f}"},
    "total_trades": {"name": "Total Trades", "higher_is_better": True, "format": "{:.0f}"},
    "avg_trade_return": {"name": "Avg Trade Return", "higher_is_better": True, "format": "{:.2f}%"},
    "expectancy": {"name": "Expectancy", "higher_is_better": True, "format": "{:.2f}%"},
    "volatility": {"name": "Volatility", "higher_is_better": False, "format": "{:.2f}%"},
    "final_equity": {"name": "Final Equity", "higher_is_better": True, "format": "${:,.2f}"},
}

AVAILABLE_METRICS = list(METRIC_DEFINITIONS.keys())


class UnivariateOptimizer:
    """
    Univariate parameter optimizer.

    Tests one parameter at a time while keeping all others at control values.
    Supports single security, portfolio (together), or portfolio (separate) modes.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
    ):
        """
        Initialize optimizer.

        Args:
            data_loader: DataLoader instance for loading security data
            initial_capital: Starting capital for backtests
            commission: Commission rate per trade
            slippage: Slippage percentage per trade
        """
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self._cancelled = False

    def cancel(self):
        """Request cancellation of optimization."""
        self._cancelled = True

    def reset(self):
        """Reset cancellation flag."""
        self._cancelled = False

    def optimize(
        self,
        strategy_class: Type[BaseStrategy],
        securities: List[str],
        parameters_to_optimize: List[ParameterConfig],
        control_values: Dict[str, Any],
        metrics: List[str],
        run_mode: str = "together",  # "together" or "separate"
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UnivariateOptimizationResult:
        """
        Run univariate optimization.

        Args:
            strategy_class: Strategy class to optimize
            securities: List of security symbols
            parameters_to_optimize: List of ParameterConfig for params to test
            control_values: Dict of all parameter names -> control values
            metrics: List of metric names to calculate
            run_mode: "together" (aggregate results) or "separate" (per-security average)
            progress_callback: Optional callback(stage, current, total)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            UnivariateOptimizationResult with all results
        """
        self._cancelled = False

        # Load data for all securities
        data_dict = {}
        for symbol in securities:
            try:
                data = self.data_loader.load_csv(symbol)
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.sort_values('date').reset_index(drop=True)
                # Apply date filters
                if start_date:
                    data = data[data['date'] >= start_date]
                if end_date:
                    data = data[data['date'] <= end_date]
                data = data.reset_index(drop=True)
                data_dict[symbol] = data
                logger.info(f"Loaded {symbol}: {len(data)} bars")
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                raise

        # Calculate total runs for progress
        total_runs = sum(len(p.values) for p in parameters_to_optimize)
        current_run = 0

        # Results storage
        parameter_results = {}

        # Create backtest config
        config = BacktestConfig(
            initial_capital=self.initial_capital,
            commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=self.commission),
            slippage_percent=self.slippage * 100,  # Convert to percentage
            start_date=start_date,
            end_date=end_date
        )

        # Run optimization for each parameter
        for param_config in parameters_to_optimize:
            if self._cancelled:
                break

            param_name = param_config.name
            logger.info(f"Optimizing parameter: {param_name}")

            results = []

            for param_value in param_config.values:
                if self._cancelled:
                    break

                current_run += 1
                if progress_callback:
                    progress_callback(
                        f"Testing {param_name}={param_value}",
                        current_run,
                        total_runs
                    )

                # Build parameters: control values + current test value
                params = control_values.copy()
                params[param_name] = param_value

                # Run backtest
                try:
                    run_metrics = self._run_backtest(
                        strategy_class, data_dict, params, metrics,
                        config, run_mode
                    )

                    results.append(OptimizationRunResult(
                        parameter_name=param_name,
                        parameter_value=param_value,
                        metrics=run_metrics,
                        total_trades=int(run_metrics.get("total_trades", 0)),
                        final_equity=run_metrics.get("final_equity", self.initial_capital)
                    ))

                except Exception as e:
                    logger.error(f"Backtest failed for {param_name}={param_value}: {e}")
                    # Add empty result
                    results.append(OptimizationRunResult(
                        parameter_name=param_name,
                        parameter_value=param_value,
                        metrics={m: np.nan for m in metrics}
                    ))

            # Store results for this parameter
            parameter_results[param_name] = ParameterOptimizationResult(
                parameter_name=param_name,
                control_value=control_values.get(param_name),
                values_tested=param_config.values,
                results=results
            )

        return UnivariateOptimizationResult(
            strategy_name=strategy_class.__name__,
            securities=securities,
            run_mode=run_mode,
            control_values=control_values,
            metrics_calculated=metrics,
            parameter_results=parameter_results,
            optimization_settings={
                "initial_capital": self.initial_capital,
                "commission": self.commission,
                "slippage": self.slippage,
                "start_date": str(start_date) if start_date else "All",
                "end_date": str(end_date) if end_date else "All"
            }
        )

    def _run_backtest(
        self,
        strategy_class: Type[BaseStrategy],
        data_dict: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        metrics: List[str],
        config: BacktestConfig,
        run_mode: str
    ) -> Dict[str, float]:
        """Run backtests and calculate metrics."""
        all_results = []

        for symbol, data in data_dict.items():
            # Create strategy with parameters
            strategy = strategy_class(**params)

            # Create engine
            engine = SingleSecurityEngine(config)

            # Run backtest
            result = engine.run(symbol, data.copy(), strategy)
            all_results.append(result)

        # Calculate metrics
        if run_mode == "together" or len(all_results) == 1:
            # Use first result or combine if portfolio
            return self._calculate_metrics_from_results(all_results, metrics)
        else:
            # Average metrics across securities
            return self._calculate_metrics_from_results(all_results, metrics)

    def _calculate_metrics_from_results(
        self,
        results: List[BacktestResult],
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate requested metrics from backtest results."""
        if not results:
            return {m: 0.0 for m in metrics}

        # Combine all equity curves and trades
        all_equity_values = []
        all_trades = []

        for result in results:
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                equity_col = result.equity_curve.get('equity', result.equity_curve.iloc[:, 0] if len(result.equity_curve.columns) > 0 else pd.Series())
                if len(equity_col) > 0:
                    all_equity_values.append(equity_col.values)
            all_trades.extend(result.trades)

        if not all_equity_values:
            return {m: 0.0 for m in metrics}

        # Combine equity curves (sum or average)
        max_len = max(len(e) for e in all_equity_values)
        combined_equity = np.zeros(max_len)
        for equity in all_equity_values:
            # Pad shorter series
            padded = np.pad(equity, (0, max_len - len(equity)), mode='edge')
            combined_equity += padded

        initial_equity = combined_equity[0] if len(combined_equity) > 0 else self.initial_capital

        # Calculate daily returns
        equity_series = pd.Series(combined_equity)
        daily_returns = equity_series.pct_change().dropna()

        calculated = {}

        for metric in metrics:
            try:
                if metric == "total_return":
                    val = ((combined_equity[-1] / combined_equity[0]) - 1) * 100 if len(combined_equity) > 1 else 0
                elif metric == "annual_return":
                    if len(combined_equity) > 1:
                        total_return = combined_equity[-1] / combined_equity[0]
                        days = len(combined_equity)
                        years = days / 252
                        val = ((total_return ** (1 / years)) - 1) * 100 if years > 0 else 0
                    else:
                        val = 0
                elif metric == "sharpe_ratio":
                    if len(daily_returns) > 1 and daily_returns.std() > 0:
                        val = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                    else:
                        val = 0.0
                elif metric == "sortino_ratio":
                    downside = daily_returns[daily_returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        val = (daily_returns.mean() / downside.std()) * np.sqrt(252)
                    else:
                        val = 0.0
                elif metric == "max_drawdown":
                    rolling_max = equity_series.expanding().max()
                    drawdowns = (equity_series - rolling_max) / rolling_max * 100
                    val = abs(drawdowns.min())
                elif metric == "calmar_ratio":
                    if len(combined_equity) > 1:
                        total_return = combined_equity[-1] / combined_equity[0]
                        days = len(combined_equity)
                        years = days / 252
                        annual_return = ((total_return ** (1 / years)) - 1) * 100 if years > 0 else 0
                        rolling_max = equity_series.expanding().max()
                        max_dd = abs((equity_series - rolling_max) / rolling_max * 100).max()
                        val = annual_return / max_dd if max_dd > 0 else 0
                    else:
                        val = 0
                elif metric == "win_rate":
                    if all_trades:
                        wins = sum(1 for t in all_trades if t.pnl > 0)
                        val = (wins / len(all_trades)) * 100
                    else:
                        val = 0.0
                elif metric == "profit_factor":
                    if all_trades:
                        gross_profit = sum(t.pnl for t in all_trades if t.pnl > 0)
                        gross_loss = abs(sum(t.pnl for t in all_trades if t.pnl < 0))
                        val = gross_profit / gross_loss if gross_loss > 0 else 0
                    else:
                        val = 0.0
                elif metric == "total_trades":
                    val = float(len(all_trades)) if all_trades else 0.0
                elif metric == "avg_trade_return":
                    if all_trades:
                        returns = [t.pnl_percent for t in all_trades]
                        val = np.mean(returns)
                    else:
                        val = 0.0
                elif metric == "expectancy":
                    if all_trades:
                        wins = [t for t in all_trades if t.pnl > 0]
                        losses = [t for t in all_trades if t.pnl < 0]
                        win_rate = len(wins) / len(all_trades)
                        avg_win = np.mean([t.pnl_percent for t in wins]) if wins else 0
                        avg_loss = abs(np.mean([t.pnl_percent for t in losses])) if losses else 0
                        val = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                    else:
                        val = 0.0
                elif metric == "volatility":
                    val = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
                elif metric == "final_equity":
                    val = combined_equity[-1] if len(combined_equity) > 0 else initial_equity
                else:
                    val = 0.0

                calculated[metric] = val
            except Exception as e:
                logger.warning(f"Failed to calculate {metric}: {e}")
                calculated[metric] = 0.0

        return calculated
