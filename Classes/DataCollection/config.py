"""
Configuration dataclasses for the Data Collection System.

Provides structured configuration for API settings, caching, validation,
and individual data collection tabs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, date
from enum import Enum
import json


class MissingDataHandling(Enum):
    """How to handle missing data points."""
    SKIP = "skip"
    INTERPOLATE = "interpolate"
    FORWARD_FILL = "forward_fill"


class DateRangeType(Enum):
    """Type of date range selection."""
    ALL_AVAILABLE = "all_available"
    CUSTOM_RANGE = "custom_range"
    LAST_N_DAYS = "last_n_days"


class QuarterlyAnnual(Enum):
    """Fundamental data frequency."""
    BOTH = "both"
    QUARTERLY_ONLY = "quarterly_only"
    ANNUAL_ONLY = "annual_only"


class TransactionType(Enum):
    """Insider transaction types."""
    ALL = "all"
    BUYS_ONLY = "buys_only"
    SELLS_ONLY = "sells_only"


class WeekDefinition(Enum):
    """How to define end of week for forex."""
    FRIDAY_CLOSE = "friday_close"
    MONDAY_OPEN = "monday_open"


class OptionsType(Enum):
    """Options contract types."""
    CALLS_ONLY = "calls_only"
    PUTS_ONLY = "puts_only"
    BOTH = "both"


class OptionsSnapshotMode(Enum):
    """Options data snapshot mode."""
    SINGLE = "single"
    DAILY = "daily"
    WEEKLY = "weekly"


class MaxDTEForward(Enum):
    """Maximum days to expiration forward from trading date.

    Used to filter options chains - only include options where:
    expiration_date <= trading_date + max_dte_forward
    """
    DAYS_30 = 30      # ~1 month
    DAYS_90 = 90      # ~3 months
    DAYS_180 = 180    # ~6 months
    DAYS_365 = 365    # ~1 year
    DAYS_730 = 730    # ~2 years

    @classmethod
    def from_days(cls, days: int) -> 'MaxDTEForward':
        """Get the enum value from days."""
        for member in cls:
            if member.value == days:
                return member
        raise ValueError(f"No MaxDTEForward preset for {days} days")

    @classmethod
    def choices(cls) -> List[tuple]:
        """Return list of (label, value) tuples for UI."""
        return [
            ("30 days (~1 month)", cls.DAYS_30),
            ("90 days (~3 months)", cls.DAYS_90),
            ("180 days (~6 months)", cls.DAYS_180),
            ("365 days (~1 year)", cls.DAYS_365),
            ("730 days (~2 years)", cls.DAYS_730),
        ]


# All available technical indicators from Alpha Vantage
AVAILABLE_INDICATORS = [
    "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "VWAP", "T3",
    "MACD", "MACDEXT", "STOCH", "STOCHF", "RSI", "STOCHRSI", "WILLR",
    "ADX", "ADXR", "APO", "PPO", "MOM", "BOP", "CCI", "CMO", "ROC", "ROCR",
    "AROON", "AROONOSC", "MFI", "TRIX", "ULTOSC", "DX", "MINUS_DI", "PLUS_DI",
    "MINUS_DM", "PLUS_DM", "BBANDS", "MIDPOINT", "MIDPRICE", "SAR",
    "TRANGE", "ATR", "NATR", "AD", "ADOSC", "OBV",
    "HT_TRENDLINE", "HT_SINE", "HT_TRENDMODE", "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR"
]

# Indicators that require period parameter
INDICATORS_WITH_PERIOD = [
    "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3",
    "RSI", "STOCHRSI", "WILLR", "ADX", "ADXR", "APO", "PPO", "MOM",
    "CCI", "CMO", "ROC", "ROCR", "AROON", "AROONOSC", "MFI", "TRIX",
    "ULTOSC", "DX", "MINUS_DI", "PLUS_DI", "MINUS_DM", "PLUS_DM",
    "BBANDS", "ATR", "NATR", "ADOSC", "HT_TRENDLINE"
]

# Default indicator periods
DEFAULT_INDICATOR_PERIODS = {
    "SMA": [20, 50, 200],
    "EMA": [12, 26],
    "RSI": [14],
    "ATR": [14],
    "ADX": [14],
    "BBANDS": [20],
    "MACD": None,  # Uses default 12, 26, 9
    "STOCH": None,  # Uses default 5, 3, 3
}

# Fundamental data categories
FUNDAMENTAL_CATEGORIES = {
    "valuation_ratios": ["PERatio", "PriceToBookRatio", "PriceToSalesRatioTTM", "PEGRatio"],
    "profitability": ["EPS", "ProfitMargin", "OperatingMarginTTM", "ReturnOnEquityTTM", "ReturnOnAssetsTTM"],
    "growth_rates": ["QuarterlyRevenueGrowthYOY", "QuarterlyEarningsGrowthYOY"],
    "financial_health": ["DebtToEquity", "CurrentRatio", "QuickRatio"],
    "cash_flow": ["OperatingCashflow", "FreeCashflow"],
}


@dataclass
class APIConfig:
    """Alpha Vantage API configuration."""
    api_key: str = ""
    rate_limit_per_minute: int = 75
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # Exponential backoff base

    def __post_init__(self):
        if self.rate_limit_per_minute <= 0:
            raise ValueError("rate_limit_per_minute must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_base": self.retry_backoff_base,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheConfig:
    """Caching configuration."""
    cache_dir: Path = field(default_factory=lambda: Path("cache/alpha_vantage"))
    cache_expiry_hours: int = 24
    enabled: bool = True
    max_cache_size_mb: int = 1000

    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if self.cache_expiry_hours < 0:
            raise ValueError("cache_expiry_hours cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_dir": str(self.cache_dir),
            "cache_expiry_hours": self.cache_expiry_hours,
            "enabled": self.enabled,
            "max_cache_size_mb": self.max_cache_size_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheConfig':
        data = data.copy()
        if "cache_dir" in data:
            data["cache_dir"] = Path(data["cache_dir"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ValidationConfig:
    """Data validation configuration."""
    max_gap_trading_days: int = 5
    min_data_points: int = 50
    check_outliers: bool = True
    outlier_std_threshold: float = 3.0
    require_ohlcv_consistency: bool = True
    allow_zero_volume: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_gap_trading_days": self.max_gap_trading_days,
            "min_data_points": self.min_data_points,
            "check_outliers": self.check_outliers,
            "outlier_std_threshold": self.outlier_std_threshold,
            "require_ohlcv_consistency": self.require_ohlcv_consistency,
            "allow_zero_volume": self.allow_zero_volume,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TabConfig:
    """Base configuration for a data collection tab."""
    tickers: List[str] = field(default_factory=list)
    date_range_type: DateRangeType = DateRangeType.ALL_AVAILABLE
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    last_n_days: int = 365
    missing_data_handling: MissingDataHandling = MissingDataHandling.FORWARD_FILL

    def __post_init__(self):
        if isinstance(self.date_range_type, str):
            self.date_range_type = DateRangeType(self.date_range_type)
        if isinstance(self.missing_data_handling, str):
            self.missing_data_handling = MissingDataHandling(self.missing_data_handling)
        if isinstance(self.from_date, str):
            self.from_date = datetime.strptime(self.from_date, "%Y-%m-%d").date()
        if isinstance(self.to_date, str):
            self.to_date = datetime.strptime(self.to_date, "%Y-%m-%d").date()

    def get_effective_date_range(self) -> tuple:
        """Get the effective from/to dates based on range type."""
        if self.date_range_type == DateRangeType.ALL_AVAILABLE:
            return None, None
        elif self.date_range_type == DateRangeType.CUSTOM_RANGE:
            return self.from_date, self.to_date
        elif self.date_range_type == DateRangeType.LAST_N_DAYS:
            to_dt = date.today()
            from_dt = date.today()
            # Approximate trading days
            from datetime import timedelta
            from_dt = to_dt - timedelta(days=int(self.last_n_days * 1.5))
            return from_dt, to_dt
        return None, None


@dataclass
class DailyDataConfig(TabConfig):
    """Configuration for daily price & indicator collection."""
    indicators: List[str] = field(default_factory=list)
    indicator_periods: Dict[str, List[int]] = field(default_factory=dict)
    adjust_for_splits: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Validate indicators
        for ind in self.indicators:
            if ind not in AVAILABLE_INDICATORS and ind != "All":
                raise ValueError(f"Unknown indicator: {ind}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tickers": self.tickers,
            "date_range_type": self.date_range_type.value,
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "last_n_days": self.last_n_days,
            "missing_data_handling": self.missing_data_handling.value,
            "indicators": self.indicators,
            "indicator_periods": self.indicator_periods,
            "adjust_for_splits": self.adjust_for_splits,
        }


@dataclass
class WeeklyDataConfig(TabConfig):
    """Configuration for weekly price & indicator collection."""
    indicators: List[str] = field(default_factory=list)
    indicator_periods: Dict[str, List[int]] = field(default_factory=dict)
    adjust_for_splits: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tickers": self.tickers,
            "date_range_type": self.date_range_type.value,
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "last_n_days": self.last_n_days,
            "missing_data_handling": self.missing_data_handling.value,
            "indicators": self.indicators,
            "indicator_periods": self.indicator_periods,
            "adjust_for_splits": self.adjust_for_splits,
        }


@dataclass
class FundamentalDataConfig(TabConfig):
    """Configuration for fundamental data collection."""
    categories: List[str] = field(default_factory=list)
    include_analyst_estimates: bool = False
    quarterly_annual: QuarterlyAnnual = QuarterlyAnnual.BOTH

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.quarterly_annual, str):
            self.quarterly_annual = QuarterlyAnnual(self.quarterly_annual)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tickers": self.tickers,
            "date_range_type": self.date_range_type.value,
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "last_n_days": self.last_n_days,
            "missing_data_handling": self.missing_data_handling.value,
            "categories": self.categories,
            "include_analyst_estimates": self.include_analyst_estimates,
            "quarterly_annual": self.quarterly_annual.value,
        }


@dataclass
class InsiderDataConfig(TabConfig):
    """Configuration for insider transaction collection."""
    transaction_type: TransactionType = TransactionType.ALL
    relationship_types: List[str] = field(default_factory=list)
    min_transaction_value: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.transaction_type, str):
            self.transaction_type = TransactionType(self.transaction_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tickers": self.tickers,
            "date_range_type": self.date_range_type.value,
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "last_n_days": self.last_n_days,
            "missing_data_handling": self.missing_data_handling.value,
            "transaction_type": self.transaction_type.value,
            "relationship_types": self.relationship_types,
            "min_transaction_value": self.min_transaction_value,
        }


@dataclass
class ForexDataConfig:
    """Configuration for forex data collection."""
    pairs: List[str] = field(default_factory=lambda: ["GBPUSD", "GBPEUR"])
    date_range_type: DateRangeType = DateRangeType.ALL_AVAILABLE
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    last_n_days: int = 365
    week_definition: WeekDefinition = WeekDefinition.FRIDAY_CLOSE
    missing_data_handling: MissingDataHandling = MissingDataHandling.FORWARD_FILL

    def __post_init__(self):
        if isinstance(self.date_range_type, str):
            self.date_range_type = DateRangeType(self.date_range_type)
        if isinstance(self.week_definition, str):
            self.week_definition = WeekDefinition(self.week_definition)
        if isinstance(self.missing_data_handling, str):
            self.missing_data_handling = MissingDataHandling(self.missing_data_handling)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs": self.pairs,
            "date_range_type": self.date_range_type.value,
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "last_n_days": self.last_n_days,
            "week_definition": self.week_definition.value,
            "missing_data_handling": self.missing_data_handling.value,
        }


@dataclass
class OptionsDataConfig(TabConfig):
    """Configuration for options data collection.

    Trading Date Approach:
    - from_date/to_date define the TRADING DATE range (when option data was recorded)
    - max_dte_forward filters options to only include those expiring within N days
      from the trading date

    Example:
        trading_date = 2017-05-01, max_dte_forward = 365 days
        -> Only include options expiring before 2018-05-01

    Output:
        Files are organized by ticker and trading year:
        raw_data/options/{ticker}/{ticker}_options_{year}.csv
    """
    options_type: OptionsType = OptionsType.BOTH
    max_dte_forward: MaxDTEForward = MaxDTEForward.DAYS_365  # Max days to expiration
    strike_range: Optional[tuple] = None  # (from, to) or None for all
    atm_strikes: Optional[int] = None  # +/- N strikes from ATM
    data_fields: List[str] = field(default_factory=lambda: [
        "strike", "bid", "ask", "last_price", "volume",
        "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega"
    ])
    adjust_for_splits: bool = True

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.options_type, str):
            self.options_type = OptionsType(self.options_type)
        if isinstance(self.max_dte_forward, int):
            self.max_dte_forward = MaxDTEForward.from_days(self.max_dte_forward)
        elif isinstance(self.max_dte_forward, str):
            # Handle string like "DAYS_365" or just "365"
            try:
                self.max_dte_forward = MaxDTEForward[self.max_dte_forward]
            except KeyError:
                self.max_dte_forward = MaxDTEForward.from_days(int(self.max_dte_forward))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tickers": self.tickers,
            "date_range_type": self.date_range_type.value,
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "last_n_days": self.last_n_days,
            "missing_data_handling": self.missing_data_handling.value,
            "options_type": self.options_type.value,
            "max_dte_forward": self.max_dte_forward.value,
            "strike_range": self.strike_range,
            "atm_strikes": self.atm_strikes,
            "data_fields": self.data_fields,
            "adjust_for_splits": self.adjust_for_splits,
        }


@dataclass
class DataCollectionConfig:
    """Master configuration for the data collection system."""
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output_dir: Path = field(default_factory=lambda: Path("raw_data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs/data_collection"))

    # Tab configurations
    daily: Optional[DailyDataConfig] = None
    weekly: Optional[WeeklyDataConfig] = None
    fundamental: Optional[FundamentalDataConfig] = None
    insider: Optional[InsiderDataConfig] = None
    forex: Optional[ForexDataConfig] = None
    options: Optional[OptionsDataConfig] = None

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.logs_dir, str):
            self.logs_dir = Path(self.logs_dir)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api": self.api.to_dict(),
            "cache": self.cache.to_dict(),
            "validation": self.validation.to_dict(),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "daily": self.daily.to_dict() if self.daily else None,
            "weekly": self.weekly.to_dict() if self.weekly else None,
            "fundamental": self.fundamental.to_dict() if self.fundamental else None,
            "insider": self.insider.to_dict() if self.insider else None,
            "forex": self.forex.to_dict() if self.forex else None,
            "options": self.options.to_dict() if self.options else None,
        }

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'DataCollectionConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            api=APIConfig.from_dict(data.get("api", {})),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            validation=ValidationConfig.from_dict(data.get("validation", {})),
            output_dir=Path(data.get("output_dir", "raw_data")),
            logs_dir=Path(data.get("logs_dir", "logs/data_collection")),
        )
