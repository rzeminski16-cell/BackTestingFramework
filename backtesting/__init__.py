"""
Backtesting Framework with Parameter Optimization

A comprehensive framework for backtesting trading strategies with
univariate parameter optimization and Excel report generation.
"""

from backtesting.core.engine import BacktestEngine
from backtesting.strategies.base import Strategy, Parameter
from backtesting.optimization.optimizer import Optimizer
from backtesting.metrics.performance import PerformanceMetrics

__version__ = "1.0.0"
__all__ = [
    "BacktestEngine",
    "Strategy",
    "Parameter",
    "Optimizer",
    "PerformanceMetrics",
]
