"""
Configuration module for Factor Analysis.

Provides immutable dataclass-based configuration with validation,
serialization, and profile management.
"""

from .factor_config import (
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

from .profile_manager import ProfileManager

__all__ = [
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
    'ProfileManager',
]
