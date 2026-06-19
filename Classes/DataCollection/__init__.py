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
    CommodityDataConfig,
    MacroDataConfig,
    CorporateActionsDataConfig,
    # Constants
    AVAILABLE_INDICATORS,
    FUNDAMENTAL_CATEGORIES,
    COMMODITY_SERIES,
    CORE_COMMODITIES,
    MACRO_SERIES,
    DEFAULT_MACRO_SERIES,
    TREASURY_MATURITIES,
    # Enums
    MissingDataHandling,
    DateRangeType,
    MaxDTEForward,
    OptionsType,
)
from .alpha_vantage_client import AlphaVantageClient
from .validation_engine import ValidationEngine, ValidationResult
from .file_manager import FileManager
from .fundamental_collector import (
    FundamentalCollector,
    build_fundamental_panel,
    extract_overview_snapshot,
)
from .benchmark_collector import (
    BenchmarkCollector,
    transform_index_data,
    load_benchmark_registry,
    resolve_benchmark,
)
from .logging_manager import LoggingManager, SessionLogger
from .options_collector import OptionsCollector
from .series_transforms import transform_flat_series
from .commodity_collector import CommodityCollector, CommodityResult
from .macro_collector import MacroCollector, MacroResult
from .corporate_actions_collector import (
    CorporateActionsCollector,
    CorporateActionsResult,
    transform_dividends,
    transform_splits,
)
from .utilities_collector import (
    UtilitiesCollector,
    transform_market_status,
)

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
    'CommodityDataConfig',
    'MacroDataConfig',
    'CorporateActionsDataConfig',
    # Constants
    'AVAILABLE_INDICATORS',
    'FUNDAMENTAL_CATEGORIES',
    'COMMODITY_SERIES',
    'CORE_COMMODITIES',
    'MACRO_SERIES',
    'DEFAULT_MACRO_SERIES',
    'TREASURY_MATURITIES',
    # Enums
    'MissingDataHandling',
    'DateRangeType',
    'MaxDTEForward',
    'OptionsType',
    # Core components
    'AlphaVantageClient',
    'ValidationEngine',
    'ValidationResult',
    'FileManager',
    'FundamentalCollector',
    'build_fundamental_panel',
    'extract_overview_snapshot',
    'BenchmarkCollector',
    'transform_index_data',
    'load_benchmark_registry',
    'resolve_benchmark',
    'LoggingManager',
    'SessionLogger',
    'OptionsCollector',
    # New families (commodities / macro / corporate actions / utilities)
    'transform_flat_series',
    'CommodityCollector',
    'CommodityResult',
    'MacroCollector',
    'MacroResult',
    'CorporateActionsCollector',
    'CorporateActionsResult',
    'transform_dividends',
    'transform_splits',
    'UtilitiesCollector',
    'transform_market_status',
]
