"""
Univariate Parameter Optimization System.

This module implements a sensitivity analysis approach to parameter optimization:
- One parameter is varied at a time while others stay at control values
- Results are collected for each parameter/value combination
- Excel reports with charts show how each parameter affects each metric
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union, Callable
import pandas as pd
import numpy as np
from copy import deepcopy
import warnings

from backtesting.core.engine import BacktestEngine, BacktestResult
from backtesting.strategies.base import Strategy, Parameter
from backtesting.metrics.performance import PerformanceMetrics, AVAILABLE_METRICS


@dataclass
class ParameterRange:
    """
    Configuration for a parameter to optimize.

    Args:
        name: Parameter name (must match strategy parameter)
        values: List of values to test, or None to auto-generate
        num_values: Number of values to generate if values is None
    """
    name: str
    values: Optional[List[Any]] = None
    num_values: Optional[int] = None

    def get_values(self, param_def: Parameter) -> List[Any]:
        """Get the list of values to test."""
        if self.values is not None:
            return self.values
        return param_def.generate_values(num_values=self.num_values)


@dataclass
class OptimizationConfig:
    """
    Configuration for optimization run.

    Args:
        strategy_class: Strategy class to optimize
        parameters_to_optimize: List of ParameterRange configs
        control_values: Control values for all parameters (defaults used if not specified)
        metrics: List of metric names to calculate
        initial_capital: Starting capital for backtests
        commission: Commission per trade (fraction)
        slippage: Slippage per trade (fraction)
    """
    strategy_class: Type[Strategy]
    parameters_to_optimize: List[ParameterRange]
    control_values: Optional[Dict[str, Any]] = None
    metrics: Optional[List[str]] = None
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005

    def __post_init__(self):
        """Validate and set defaults."""
        # Set default control values from strategy
        if self.control_values is None:
            self.control_values = self.strategy_class.get_control_values()
        else:
            # Merge with defaults for any missing values
            defaults = self.strategy_class.get_control_values()
            defaults.update(self.control_values)
            self.control_values = defaults

        # Set default metrics
        if self.metrics is None:
            self.metrics = PerformanceMetrics.DEFAULT_METRICS

        # Validate parameters exist
        param_defs = self.strategy_class.get_parameter_definitions()
        for param_range in self.parameters_to_optimize:
            if param_range.name not in param_defs:
                raise ValueError(
                    f"Parameter '{param_range.name}' not found in strategy. "
                    f"Available: {list(param_defs.keys())}"
                )

        # Validate metrics
        for metric in self.metrics:
            if metric not in AVAILABLE_METRICS:
                raise ValueError(
                    f"Unknown metric '{metric}'. "
                    f"Available: {list(AVAILABLE_METRICS.keys())}"
                )


@dataclass
class OptimizationResult:
    """
    Results from a single parameter value test.

    Attributes:
        parameter_name: Name of the parameter being tested
        parameter_value: Value used for this test
        metrics: Dictionary of metric name -> value
        backtest_result: Full backtest result (optional, for detailed analysis)
    """
    parameter_name: str
    parameter_value: Any
    metrics: Dict[str, float]
    backtest_result: Optional[BacktestResult] = None


@dataclass
class ParameterOptimizationResults:
    """
    Results for optimizing a single parameter.

    Attributes:
        parameter_name: Name of the parameter
        parameter_definition: Parameter metadata
        control_value: Control value used for other parameters
        values_tested: List of values that were tested
        results: List of OptimizationResult for each value
    """
    parameter_name: str
    parameter_definition: Parameter
    control_value: Any
    values_tested: List[Any]
    results: List[OptimizationResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        data = []
        for result in self.results:
            row = {"parameter_value": result.parameter_value}
            row.update(result.metrics)
            data.append(row)
        return pd.DataFrame(data)

    def get_best_value(self, metric: str) -> Any:
        """Get the parameter value that produced the best metric."""
        metric_def = AVAILABLE_METRICS.get(metric)
        if metric_def is None:
            raise ValueError(f"Unknown metric: {metric}")

        best_result = None
        best_value = None

        for result in self.results:
            value = result.metrics.get(metric)
            if value is None:
                continue

            if best_result is None:
                best_result = result
                best_value = value
            elif metric_def.higher_is_better:
                if value > best_value:
                    best_result = result
                    best_value = value
            else:
                if value < best_value:
                    best_result = result
                    best_value = value

        return best_result.parameter_value if best_result else None


@dataclass
class FullOptimizationResults:
    """
    Complete results from optimization run.

    Attributes:
        config: The optimization configuration used
        control_values: Control values for all parameters
        parameter_results: Dict of parameter name -> ParameterOptimizationResults
        timestamp: When the optimization was run
    """
    config: OptimizationConfig
    control_values: Dict[str, Any]
    parameter_results: Dict[str, ParameterOptimizationResults]
    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> pd.DataFrame:
        """Get summary of best values for each parameter/metric combination."""
        data = []
        for param_name, param_results in self.parameter_results.items():
            row = {"parameter": param_name, "control_value": param_results.control_value}
            for metric in self.config.metrics:
                best = param_results.get_best_value(metric)
                row[f"best_{metric}"] = best
            data.append(row)
        return pd.DataFrame(data)


class Optimizer:
    """
    Univariate parameter optimizer.

    Runs backtests varying one parameter at a time while keeping others
    at their control values. This approach enables sensitivity analysis
    to understand how each parameter affects performance.

    Example:
        config = OptimizationConfig(
            strategy_class=SMACrossover,
            parameters_to_optimize=[
                ParameterRange("fast_period", values=[5, 10, 15, 20, 25]),
                ParameterRange("slow_period", values=[20, 30, 40, 50, 60]),
            ],
            control_values={"fast_period": 10, "slow_period": 30},
            metrics=["total_return", "sharpe_ratio", "max_drawdown"],
        )

        optimizer = Optimizer(config)
        results = optimizer.run(data)
        optimizer.export_to_excel(results, "optimization_results.xlsx")
    """

    def __init__(
        self,
        config: OptimizationConfig,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
            progress_callback: Optional callback(param_name, current, total) for progress
        """
        self.config = config
        self.progress_callback = progress_callback

        # Create backtest engine
        self.engine = BacktestEngine(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
        )

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        store_backtest_results: bool = False,
    ) -> FullOptimizationResults:
        """
        Run the optimization.

        Args:
            data: Price data (single DataFrame or dict of DataFrames)
            start_date: Optional start date filter
            end_date: Optional end date filter
            store_backtest_results: Whether to store full backtest results

        Returns:
            FullOptimizationResults containing all optimization data
        """
        param_defs = self.config.strategy_class.get_parameter_definitions()
        parameter_results = {}

        # Calculate total runs for progress
        total_runs = sum(
            len(pr.get_values(param_defs[pr.name]))
            for pr in self.config.parameters_to_optimize
        )
        current_run = 0

        # Optimize each parameter independently
        for param_range in self.config.parameters_to_optimize:
            param_name = param_range.name
            param_def = param_defs[param_name]
            values_to_test = param_range.get_values(param_def)

            results = []

            for value in values_to_test:
                current_run += 1

                if self.progress_callback:
                    self.progress_callback(param_name, current_run, total_runs)

                # Create strategy with control values, except for the parameter being tested
                params = self.config.control_values.copy()
                params[param_name] = value

                try:
                    strategy = self.config.strategy_class(**params)

                    # Run backtest
                    backtest_result = self.engine.run(
                        strategy=strategy,
                        data=data,
                        start_date=start_date,
                        end_date=end_date,
                    )

                    # Calculate metrics
                    metrics_calc = PerformanceMetrics(backtest_result)
                    metrics = metrics_calc.calculate_selected(self.config.metrics)

                    result = OptimizationResult(
                        parameter_name=param_name,
                        parameter_value=value,
                        metrics=metrics,
                        backtest_result=backtest_result if store_backtest_results else None,
                    )

                except Exception as e:
                    warnings.warn(f"Backtest failed for {param_name}={value}: {e}")
                    # Store NaN metrics for failed runs
                    metrics = {m: float("nan") for m in self.config.metrics}
                    result = OptimizationResult(
                        parameter_name=param_name,
                        parameter_value=value,
                        metrics=metrics,
                    )

                results.append(result)

            parameter_results[param_name] = ParameterOptimizationResults(
                parameter_name=param_name,
                parameter_definition=param_def,
                control_value=self.config.control_values[param_name],
                values_tested=values_to_test,
                results=results,
            )

        return FullOptimizationResults(
            config=self.config,
            control_values=self.config.control_values,
            parameter_results=parameter_results,
        )

    def export_to_excel(
        self,
        results: FullOptimizationResults,
        filepath: str,
    ) -> None:
        """
        Export optimization results to Excel with charts.

        Creates an Excel file with:
        - Summary tab with control values and best values
        - One tab per optimized parameter with line charts

        Args:
            results: FullOptimizationResults from run()
            filepath: Path to save Excel file
        """
        from backtesting.optimization.report import ExcelReportGenerator

        generator = ExcelReportGenerator(results)
        generator.generate(filepath)
