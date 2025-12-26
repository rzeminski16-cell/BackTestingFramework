"""
Configuration dataclass for backtest analysis parameters.

All thresholds and parameters are configurable to allow experimentation
with different classification criteria.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisConfig:
    """
    Configuration for backtest analysis thresholds and parameters.

    All percentages are expressed as whole numbers (e.g., 5.0 means 5%).
    """

    # ==========================================================================
    # GB_Flag Thresholds (Trade-level Good/Indeterminate/Bad classification)
    # ==========================================================================

    # Profit threshold for "G" (Good) classification
    # Trades with profit >= this value are marked as Good
    gb_profit_good_threshold: float = 5.0  # 5%

    # Trades with profit >= 0% but < gb_profit_good_threshold are "I" (Indeterminate)
    # Trades with profit < 0% are "B" (Bad)

    # ==========================================================================
    # Outcomes_Flag Thresholds (Post-exit price behavior classification)
    # ==========================================================================

    # FullRideGood: Profit >= this AND price 21 days later <= exit price + extension
    full_ride_profit_min: float = 5.0  # 5%

    # Price extension tolerance for FullRideGood (% of entry price)
    # Price 21 days later must be <= exit_price + (entry_price * this/100)
    full_ride_price_extension: float = 10.0  # 10% of entry price

    # EarlyExitGood: Profit >= 0% AND no major extension after exit
    # Major extension is when price goes > exit_price + (exit_price * this/100)
    early_exit_major_extension: float = 5.0  # 5% on top of exit price

    # MissedOpportunity: Exit at small profit (<5%) or loss but price surges after
    # Price surge threshold (% above exit price)
    missed_opportunity_surge: float = 10.0  # 10%

    # Number of trading days after exit to check price behavior
    days_after_exit: int = 21

    # ==========================================================================
    # Period_GB_Flag Thresholds (Year-level classification for fundamentals)
    # ==========================================================================

    # Calmar ratio threshold for "good" classification
    calmar_good_threshold: float = 0.5

    # Max drawdown threshold for "good" classification (must be <= this)
    max_dd_good_threshold: float = 25.0  # 25%

    # Classification logic:
    # - "good": Calmar > threshold AND max_dd <= threshold
    # - "indeterminate": Calmar > threshold AND max_dd > threshold
    # - "bad": otherwise (Calmar <= threshold)

    # ==========================================================================
    # Trade Filtering Thresholds
    # ==========================================================================

    # Minimum trades per security per year to include in technical features
    min_trades_per_year: int = 4

    # ==========================================================================
    # Technical Indicator Parameters
    # ==========================================================================

    # RSI period (in weeks, will be converted to daily equivalent)
    rsi_period_weeks: int = 14

    # RSI SMA smoothing period (in weeks)
    rsi_sma_period_weeks: int = 14

    # 52-week high/low lookback period (in weeks)
    high_low_period_weeks: int = 52

    # Bollinger Bands parameters
    bb_period_weeks: int = 20
    bb_std_dev: float = 2.0

    # Volume trend lookback (in weeks)
    volume_ma_period_weeks: int = 50

    # ATR period (in weeks)
    atr_period_weeks: int = 14

    # Additional suggested indicators
    sma_200_period_days: int = 200  # For Price vs SMA200
    volume_ratio_period_days: int = 20  # For Volume Ratio at entry

    # ==========================================================================
    # Output Configuration
    # ==========================================================================

    # Output directory (relative to project root or absolute)
    output_directory: Optional[Path] = None

    # Whether to save weekly data for validation
    save_weekly_data: bool = True

    # ==========================================================================
    # Data Source Configuration
    # ==========================================================================

    # Raw data directory (for loading security price data)
    raw_data_directory: Optional[Path] = None

    # Trading days per week (for converting weekly to daily periods)
    trading_days_per_week: int = 5

    def __post_init__(self):
        """Convert path strings to Path objects if needed."""
        if self.output_directory is not None and isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        if self.raw_data_directory is not None and isinstance(self.raw_data_directory, str):
            self.raw_data_directory = Path(self.raw_data_directory)

    def get_daily_equivalent(self, weeks: int) -> int:
        """
        Convert weeks to daily trading period equivalent.

        Args:
            weeks: Number of weeks

        Returns:
            Number of trading days
        """
        return weeks * self.trading_days_per_week

    @property
    def rsi_period_days(self) -> int:
        """RSI period in trading days."""
        return self.get_daily_equivalent(self.rsi_period_weeks)

    @property
    def rsi_sma_period_days(self) -> int:
        """RSI SMA smoothing period in trading days."""
        return self.get_daily_equivalent(self.rsi_sma_period_weeks)

    @property
    def high_low_period_days(self) -> int:
        """52-week high/low period in trading days."""
        return self.get_daily_equivalent(self.high_low_period_weeks)

    @property
    def bb_period_days(self) -> int:
        """Bollinger Bands period in trading days."""
        return self.get_daily_equivalent(self.bb_period_weeks)

    @property
    def volume_ma_period_days(self) -> int:
        """Volume MA period in trading days."""
        return self.get_daily_equivalent(self.volume_ma_period_weeks)

    @property
    def atr_period_days(self) -> int:
        """ATR period in trading days."""
        return self.get_daily_equivalent(self.atr_period_weeks)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'gb_profit_good_threshold': self.gb_profit_good_threshold,
            'full_ride_profit_min': self.full_ride_profit_min,
            'full_ride_price_extension': self.full_ride_price_extension,
            'early_exit_major_extension': self.early_exit_major_extension,
            'missed_opportunity_surge': self.missed_opportunity_surge,
            'days_after_exit': self.days_after_exit,
            'calmar_good_threshold': self.calmar_good_threshold,
            'max_dd_good_threshold': self.max_dd_good_threshold,
            'min_trades_per_year': self.min_trades_per_year,
            'rsi_period_weeks': self.rsi_period_weeks,
            'rsi_sma_period_weeks': self.rsi_sma_period_weeks,
            'high_low_period_weeks': self.high_low_period_weeks,
            'bb_period_weeks': self.bb_period_weeks,
            'bb_std_dev': self.bb_std_dev,
            'volume_ma_period_weeks': self.volume_ma_period_weeks,
            'atr_period_weeks': self.atr_period_weeks,
            'sma_200_period_days': self.sma_200_period_days,
            'volume_ratio_period_days': self.volume_ratio_period_days,
            'trading_days_per_week': self.trading_days_per_week,
            'save_weekly_data': self.save_weekly_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AnalysisConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
