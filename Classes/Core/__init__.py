"""
Core module containing centralized components used across the framework.
"""

from .performance_metrics import (
    CentralizedPerformanceMetrics,
    MetricDefinition,
    METRIC_DEFINITIONS,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    MAX_PROFIT_FACTOR,
)

__all__ = [
    'CentralizedPerformanceMetrics',
    'MetricDefinition',
    'METRIC_DEFINITIONS',
    'DEFAULT_RISK_FREE_RATE',
    'TRADING_DAYS_PER_YEAR',
    'MAX_PROFIT_FACTOR',
]
