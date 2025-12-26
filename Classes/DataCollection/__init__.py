"""
Data Collection Package

A comprehensive unified raw data collection system powered by the Alpha Vantage API.
Provides a professional-grade GUI that orchestrates data fetching, validation,
preparation, and logging for all data types required by the backtesting framework.
"""

from .config import (
    DataCollectionConfig,
    APIConfig,
    CacheConfig,
    ValidationConfig,
    TabConfig,
    DailyDataConfig,
    WeeklyDataConfig,
    FundamentalDataConfig,
    InsiderDataConfig,
    ForexDataConfig,
    OptionsDataConfig,
)
from .alpha_vantage_client import AlphaVantageClient
from .validation_engine import ValidationEngine, ValidationResult
from .file_manager import FileManager
from .logging_manager import LoggingManager, SessionLogger

__all__ = [
    # Configuration
    'DataCollectionConfig',
    'APIConfig',
    'CacheConfig',
    'ValidationConfig',
    'TabConfig',
    'DailyDataConfig',
    'WeeklyDataConfig',
    'FundamentalDataConfig',
    'InsiderDataConfig',
    'ForexDataConfig',
    'OptionsDataConfig',
    # Core components
    'AlphaVantageClient',
    'ValidationEngine',
    'ValidationResult',
    'FileManager',
    'LoggingManager',
    'SessionLogger',
]
