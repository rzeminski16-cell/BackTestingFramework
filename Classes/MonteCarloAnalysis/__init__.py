"""
Monte Carlo Analysis package for trade log simulation.

Modules:
    trade_log_loader: Load trade returns / R-multiples from CSV files (canonical
        BackTestingFramework schema with auto-detection for foreign schemas).
    simulator: Vectorized Monte Carlo engine with simple and block bootstrap,
        compounding/fixed sizing, drawdown-based risk reduction, and costs.
    analyzer: Aggregate metrics from simulation output (percentiles, drawdowns,
        probability of loss, Sharpe, etc.).
    config: Dataclass containers for simulation configuration.
"""

from .config import (
    SamplingMethod,
    SizingMethod,
    SimulationConfig,
)
from .trade_log_loader import (
    LoadedTradeLog,
    TradeLogReturnSource,
    load_trade_logs,
)
from .simulator import (
    SimulationResult,
    MonteCarloSimulator,
)
from .analyzer import (
    SimulationMetrics,
    SimulationAnalyzer,
)

__all__ = [
    "SamplingMethod",
    "SizingMethod",
    "SimulationConfig",
    "LoadedTradeLog",
    "TradeLogReturnSource",
    "load_trade_logs",
    "SimulationResult",
    "MonteCarloSimulator",
    "SimulationMetrics",
    "SimulationAnalyzer",
]
