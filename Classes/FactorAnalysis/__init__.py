"""
Factor Analysis Module for Strategy Performance Analysis.

This module provides comprehensive factor-based analysis of trade performance,
identifying which market, technical, fundamental, insider, and options factors
correlate with trade outcomes while avoiding forward-looking bias.

Main Classes:
    - FactorAnalyzer: Main entry point for running analysis
    - FactorAnalysisConfig: Complete configuration for analysis
    - ProfileManager: Save/load analysis profiles
    - AnalysisResult: Container for analysis results

Example Usage:
    from Classes.FactorAnalysis import FactorAnalyzer, ProfileManager

    # Load configuration
    profile_manager = ProfileManager(config_dir="./configs")
    config = profile_manager.load("momentum_value_2025")

    # Initialize analyzer
    analyzer = FactorAnalyzer(config)

    # Run analysis
    result = analyzer.analyze_trades(
        trade_logs=["./data/trades_AAPL.csv"],
        data_sources={
            "price_indicators": "./data/price_daily.csv",
            "fundamentals": "./data/fundamentals.csv",
            "insider": "./data/insider.csv",
            "options": "./data/options.csv"
        }
    )

    # Export outputs
    analyzer.export_excel_report(result, "./output/factor_analysis.xlsx")
"""

from .config.factor_config import (
    FactorAnalysisConfig,
    TradeClassificationConfig,
    DataAlignmentConfig,
    FactorEngineeringConfig,
    FactorCategoryConfig,
    NullHandlingConfig,
    OutlierHandlingConfig,
    MultiLogAggregationConfig,
    StatisticalAnalysisConfig,
    Tier1Config,
    Tier2Config,
    Tier3Config,
    MultipleTestingConfig,
    ScenarioAnalysisConfig,
    OutputConfig,
    ExcelReportConfig,
    GuiPayloadsConfig,
    AuditLogConfig,
)

from .config.profile_manager import ProfileManager

from .logging.audit_logger import AuditLogger

# Main analyzer
from .analyzer import FactorAnalyzer, AnalysisOutput, AnalysisInput

# Data loaders
from .data import TradeLogLoader, PriceDataLoader, FundamentalLoader, InsiderLoader, OptionsLoader

# Preprocessing
from .preprocessing import TradeClassifier, TemporalAligner, DataEnricher, QualityScorer, MultiLogAggregator

# Factors
from .factors import (
    TechnicalFactors, FundamentalFactors, InsiderFactors,
    OptionsFactors, RegimeFactors, FactorNormalizer, OutlierHandler
)

# Analysis
from .analysis import Tier1Exploratory, Tier2Hypothesis, Tier3ML, MultipleTestingCorrector

# Scenarios
from .scenarios import ScenarioDetector, ScenarioValidator, InteractionAnalyzer

# Output
from .output import ExcelReportGenerator, JsonPayloadGenerator, ResultFormatter

__all__ = [
    # Main classes
    'FactorAnalyzer',
    'AnalysisOutput',
    'AnalysisInput',

    # Configuration
    'FactorAnalysisConfig',
    'TradeClassificationConfig',
    'DataAlignmentConfig',
    'FactorEngineeringConfig',
    'FactorCategoryConfig',
    'NullHandlingConfig',
    'OutlierHandlingConfig',
    'MultiLogAggregationConfig',
    'StatisticalAnalysisConfig',
    'Tier1Config',
    'Tier2Config',
    'Tier3Config',
    'MultipleTestingConfig',
    'ScenarioAnalysisConfig',
    'OutputConfig',
    'ExcelReportConfig',
    'GuiPayloadsConfig',
    'AuditLogConfig',

    # Profile management
    'ProfileManager',

    # Logging
    'AuditLogger',

    # Data loaders
    'TradeLogLoader',
    'PriceDataLoader',
    'FundamentalLoader',
    'InsiderLoader',
    'OptionsLoader',

    # Preprocessing
    'TradeClassifier',
    'TemporalAligner',
    'DataEnricher',
    'QualityScorer',
    'MultiLogAggregator',

    # Factors
    'TechnicalFactors',
    'FundamentalFactors',
    'InsiderFactors',
    'OptionsFactors',
    'RegimeFactors',
    'FactorNormalizer',
    'OutlierHandler',

    # Analysis
    'Tier1Exploratory',
    'Tier2Hypothesis',
    'Tier3ML',
    'MultipleTestingCorrector',

    # Scenarios
    'ScenarioDetector',
    'ScenarioValidator',
    'InteractionAnalyzer',

    # Output
    'ExcelReportGenerator',
    'JsonPayloadGenerator',
    'ResultFormatter',
]
