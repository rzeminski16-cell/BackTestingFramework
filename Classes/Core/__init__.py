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

from .stable_metrics import (
    StableMetricsCalculator,
    StableMetricsResult,
    StableMetricDefinition,
    DrawdownDetail,
    STABLE_METRIC_DEFINITIONS,
    BARS_PER_YEAR,
)

__all__ = [
    # Standard performance metrics
    'CentralizedPerformanceMetrics',
    'MetricDefinition',
    'METRIC_DEFINITIONS',
    'DEFAULT_RISK_FREE_RATE',
    'TRADING_DAYS_PER_YEAR',
    'MAX_PROFIT_FACTOR',
    # Stable metrics
    'StableMetricsCalculator',
    'StableMetricsResult',
    'StableMetricDefinition',
    'DrawdownDetail',
    'STABLE_METRIC_DEFINITIONS',
    'BARS_PER_YEAR',
]
