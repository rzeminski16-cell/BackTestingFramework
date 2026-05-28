"""
Backtest Analyzer Module
========================

This module provides tools for analyzing backtesting results to understand
strategy performance across different market conditions.

Overview
--------
The Backtest Analyzer processes trade logs to generate:

1. **Fundamental Features CSVs** - Per-security files with quarterly rows and
   yearly performance classifications (period_GB_flag)

2. **Technical Features Master CSV** - All filtered trades with technical
   indicators at entry and classification flags (GB_Flag, Outcomes_Flag)

3. **Summary Reports** - Statistics and coverage reports

Point-in-time raw fundamentals are collected separately by
``Classes.DataCollection.FundamentalCollector`` (see ``raw_data/fundamentals/``);
derived metrics should be computed from that panel via the factor-analysis
``FundamentalLoader`` rather than re-fetched here.

Quick Start
-----------
Command line (backtest analysis)::

    python run_backtest_analysis.py logs/MyStrategy

Programmatic::

    from Classes.Analysis.backtest_analyzer import BacktestAnalyzer, AnalysisConfig
    from pathlib import Path

    config = AnalysisConfig(
        raw_data_directory=Path("raw_data"),
        gb_profit_good_threshold=5.0,  # 5% for "Good" trades
    )
    analyzer = BacktestAnalyzer(config)
    results = analyzer.analyze(Path("logs/MyStrategy"))

Components
----------
BacktestAnalyzer
    Main orchestrator class that runs the full analysis pipeline.

AnalysisConfig
    Configuration dataclass with all thresholds and parameters.
    All values are configurable to allow experimentation.

FundamentalFeaturesGenerator
    Creates per-security CSVs with quarterly rows and yearly period_GB_flag.

TechnicalFeaturesGenerator
    Creates master CSV with filtered trades, indicators, and flags.

WeeklyIndicatorCalculator
    Calculates weekly technical indicators from daily price data.

Alpha Vantage Integration
-------------------------
AlphaVantageConfig
    Configuration for Alpha Vantage API credentials and settings.

AlphaVantageClient
    API client with caching and rate limiting.

InteractiveHandler
    Handles user prompts for ambiguous data with answer memory.

Classification Flags
--------------------
period_GB_flag (yearly):
    - "good": Calmar ratio > 0.5 AND max drawdown <= 25%
    - "indeterminate": Calmar ratio > 0.5 AND max drawdown > 25%
    - "bad": Calmar ratio <= 0.5
    - "Unknown": No trades in that year

GB_Flag (per trade):
    - "G": Profit >= 5%
    - "I": Profit >= 0% and < 5%
    - "B": Profit < 0% (loss)

Outcomes_Flag (per trade):
    - "FullRideGood": Good profit, price didn't surge after exit
    - "EarlyExitGood": Breakeven+, no major extension after
    - "MissedOpportunity": Exited but price surged >10% after
    - "Neutral": Doesn't fit above categories

See Also
--------
docs/BACKTEST_ANALYSIS_GUIDE.md : Complete documentation
run_backtest_analysis.py : CLI tool for backtest analysis
"""

from .config import AnalysisConfig
from .analyzer import BacktestAnalyzer
from .alpha_vantage_config import AlphaVantageConfig, create_sample_config
from .alpha_vantage_client import AlphaVantageClient
from .interactive_handler import InteractiveHandler

__all__ = [
    'AnalysisConfig',
    'BacktestAnalyzer',
    'AlphaVantageConfig',
    'AlphaVantageClient',
    'InteractiveHandler',
    'create_sample_config',
]
