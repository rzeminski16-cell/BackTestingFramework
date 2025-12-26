"""
Backtest Analyzer Module

This module provides tools for analyzing backtesting results to understand
strategy performance across different market conditions.

Components:
- BacktestAnalyzer: Main orchestrator class
- AnalysisConfig: Configuration dataclass for all thresholds and parameters
- FundamentalFeaturesGenerator: Creates fundamental features with period flags
- TechnicalFeaturesGenerator: Creates technical features with trade flags
- WeeklyIndicatorCalculator: Calculates weekly technical indicators
"""

from .config import AnalysisConfig
from .analyzer import BacktestAnalyzer

__all__ = [
    'AnalysisConfig',
    'BacktestAnalyzer',
]
