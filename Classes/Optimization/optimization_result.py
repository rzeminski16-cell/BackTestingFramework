"""
Optimization result tracking.
"""
from dataclasses import dataclass
from typing import Dict, Any, List
from ..Engine.backtest_result import BacktestResult


@dataclass
class OptimizationResult:
    """
    Result from a single parameter combination.

    Attributes:
        parameters: Parameter combination used
        backtest_result: Backtest result
        metric_value: Value of the optimization metric
    """
    parameters: Dict[str, Any]
    backtest_result: BacktestResult
    metric_value: float

    def __str__(self) -> str:
        """String representation."""
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"OptimizationResult({param_str}): metric={self.metric_value:.4f}"


class OptimizationResults:
    """
    Collection of optimization results with ranking and selection.
    """

    def __init__(self, results: List[OptimizationResult], metric: str, maximize: bool):
        """
        Initialize optimization results collection.

        Args:
            results: List of optimization results
            metric: Metric name that was optimized
            maximize: Whether metric was maximized
        """
        self.results = results
        self.metric = metric
        self.maximize = maximize

        # Sort results
        self.results.sort(
            key=lambda x: x.metric_value,
            reverse=maximize
        )

    def get_best(self) -> OptimizationResult:
        """
        Get best result.

        Returns:
            Best optimization result
        """
        if not self.results:
            raise ValueError("No optimization results available")
        return self.results[0]

    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get best parameters.

        Returns:
            Best parameter combination
        """
        return self.get_best().parameters

    def get_top_n(self, n: int) -> List[OptimizationResult]:
        """
        Get top N results.

        Args:
            n: Number of results to return

        Returns:
            List of top N results
        """
        return self.results[:n]

    def get_all(self) -> List[OptimizationResult]:
        """
        Get all results.

        Returns:
            List of all results
        """
        return self.results

    def summary(self) -> str:
        """
        Get summary of optimization results.

        Returns:
            Summary string
        """
        if not self.results:
            return "No optimization results"

        best = self.get_best()
        worst = self.results[-1]

        summary = [
            f"Optimization Summary ({self.metric}):",
            f"  Total combinations tested: {len(self.results)}",
            f"  Best {self.metric}: {best.metric_value:.4f}",
            f"  Best parameters: {best.parameters}",
            f"  Worst {self.metric}: {worst.metric_value:.4f}",
            f"  Worst parameters: {worst.parameters}"
        ]

        return "\n".join(summary)
