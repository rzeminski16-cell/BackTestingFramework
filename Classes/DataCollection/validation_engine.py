"""
Data Validation Engine for the Data Collection System.

Provides comprehensive validation for all data types:
- Type validation (datetime, float, int)
- Value range validation (prices > 0, high >= low, etc.)
- Temporal validation (chronological order, no duplicates, gap detection)
- Data completeness (required columns, null checks)
- Consistency checks (volume/price correlation, indicator verification)

Also identifies outliers and data quality issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .config import ValidationConfig
from .logging_manager import SessionLogger, ValidationLogEntry, DataIssueEntry


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    check_name: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.INFO
    column: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_log_entry(self, file_name: str, timestamp: datetime) -> ValidationLogEntry:
        """Convert to a log entry."""
        return ValidationLogEntry(
            timestamp=timestamp,
            file_name=file_name,
            check_name=self.check_name,
            passed=self.passed,
            message=self.message,
            column=self.column,
            details=self.details
        )


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    file_name: str
    timestamp: datetime
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    results: List[ValidationResult] = field(default_factory=list)
    data_quality_score: float = 100.0

    def get_errors(self) -> List[ValidationResult]:
        """Get all error-level results."""
        return [r for r in self.results if r.severity == ValidationSeverity.ERROR and not r.passed]

    def get_warnings(self) -> List[ValidationResult]:
        """Get all warning-level results."""
        return [r for r in self.results if r.severity == ValidationSeverity.WARNING and not r.passed]


class ValidationEngine:
    """
    Comprehensive data validation engine.

    Validates DataFrames for all data types collected by the system.
    """

    def __init__(self, config: ValidationConfig, logger: Optional[SessionLogger] = None):
        """
        Initialize validation engine.

        Args:
            config: Validation configuration
            logger: Optional session logger for logging validation results
        """
        self.config = config
        self.logger = logger

    def validate_daily_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        indicator_columns: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate daily price and indicator data.

        Args:
            df: DataFrame with daily data
            symbol: Stock symbol for logging
            indicator_columns: List of indicator column names

        Returns:
            ValidationReport with all check results
        """
        file_name = f"{symbol}_daily.csv"
        results = []

        # Required columns for daily data
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']

        # Run all checks
        results.extend(self._check_required_columns(df, required_cols))
        results.extend(self._check_column_types_ohlcv(df))
        results.extend(self._check_date_column(df))
        results.extend(self._check_price_ranges(df))
        results.extend(self._check_ohlc_consistency(df))
        results.extend(self._check_volume(df))
        results.extend(self._check_temporal_order(df))
        results.extend(self._check_duplicates(df, 'date'))
        results.extend(self._check_gaps(df, 'date', max_gap_days=self.config.max_gap_trading_days))
        results.extend(self._check_data_points(df))

        if self.config.check_outliers:
            results.extend(self._check_outliers(df, ['open', 'high', 'low', 'close']))

        # Validate indicator columns if present
        if indicator_columns:
            results.extend(self._check_indicator_columns(df, indicator_columns))

        return self._create_report(file_name, results)

    def validate_weekly_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        indicator_columns: Optional[List[str]] = None
    ) -> ValidationReport:
        """Validate weekly price and indicator data."""
        file_name = f"{symbol}_weekly.csv"
        results = []

        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']

        results.extend(self._check_required_columns(df, required_cols))
        results.extend(self._check_column_types_ohlcv(df))
        results.extend(self._check_date_column(df))
        results.extend(self._check_price_ranges(df))
        results.extend(self._check_ohlc_consistency(df))
        results.extend(self._check_volume(df))
        results.extend(self._check_temporal_order(df))
        results.extend(self._check_duplicates(df, 'date'))
        # Weekly data can have gaps of up to 14 days (2 weeks)
        results.extend(self._check_gaps(df, 'date', max_gap_days=14))
        results.extend(self._check_data_points(df))

        if self.config.check_outliers:
            results.extend(self._check_outliers(df, ['open', 'high', 'low', 'close']))

        if indicator_columns:
            results.extend(self._check_indicator_columns(df, indicator_columns))

        return self._create_report(file_name, results)

    def validate_fundamental_data(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> ValidationReport:
        """
        Validate fundamental data.

        Args:
            df: DataFrame with fundamental data
            symbol: Stock symbol for logging

        Returns:
            ValidationReport
        """
        file_name = f"{symbol}_fundamental.csv"
        results = []

        # Check for either date or quarter_end_date
        if 'date' not in df.columns and 'quarter_end_date' not in df.columns:
            results.append(ValidationResult(
                passed=False,
                check_name="Required date column",
                message="Missing 'date' or 'quarter_end_date' column",
                severity=ValidationSeverity.ERROR
            ))
        else:
            date_col = 'date' if 'date' in df.columns else 'quarter_end_date'
            results.extend(self._check_date_column(df, date_col))
            results.extend(self._check_temporal_order(df, date_col))
            results.extend(self._check_duplicates(df, date_col))

        # Check for reasonable quarter values if present
        if 'quarter' in df.columns:
            results.extend(self._check_quarter_values(df))

        # Check for placeholder values
        results.extend(self._check_placeholder_values(df))

        # Check numeric columns are actually numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            results.extend(self._check_reasonable_values(df, col))

        results.extend(self._check_data_points(df, min_points=4))  # At least 1 year of quarters

        return self._create_report(file_name, results)

    def validate_insider_data(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> ValidationReport:
        """Validate insider transaction data."""
        file_name = f"{symbol}_insider.csv"
        results = []

        required_cols = ['date', 'transaction_type', 'shares', 'price']

        results.extend(self._check_required_columns(df, required_cols))

        if 'date' in df.columns:
            results.extend(self._check_date_column(df))
            results.extend(self._check_future_dates(df))

        if 'transaction_type' in df.columns:
            results.extend(self._check_transaction_types(df))

        if 'shares' in df.columns:
            results.append(self._check_positive_values(df, 'shares', allow_zero=False))

        if 'price' in df.columns:
            results.append(self._check_positive_values(df, 'price', allow_zero=False))

        if 'value' in df.columns:
            results.extend(self._check_value_calculation(df))

        results.extend(self._check_duplicates_multi(df, ['date', 'insider_name', 'transaction_type', 'shares']))

        return self._create_report(file_name, results)

    def validate_forex_data(
        self,
        df: pd.DataFrame,
        pair: str
    ) -> ValidationReport:
        """Validate forex data."""
        file_name = f"{pair}_weekly.csv"
        results = []

        required_cols = ['date', 'close']

        results.extend(self._check_required_columns(df, required_cols))

        if 'date' in df.columns:
            results.extend(self._check_date_column(df))
            results.extend(self._check_temporal_order(df))
            results.extend(self._check_duplicates(df, 'date'))
            # Forex can have gaps due to holidays
            results.extend(self._check_gaps(df, 'date', max_gap_days=30))

        if 'close' in df.columns:
            results.append(self._check_positive_values(df, 'close'))
            results.extend(self._check_forex_range(df, pair))

        results.extend(self._check_data_points(df))

        return self._create_report(file_name, results)

    def validate_options_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        expiration: str
    ) -> ValidationReport:
        """Validate options data."""
        file_name = f"{symbol}_{expiration}_options.csv"
        results = []

        required_cols = ['strike', 'option_type']

        results.extend(self._check_required_columns(df, required_cols))

        if 'option_type' in df.columns:
            results.extend(self._check_option_types(df))

        if 'strike' in df.columns:
            results.append(self._check_positive_values(df, 'strike'))

        if 'expiration_date' in df.columns and 'snapshot_date' in df.columns:
            results.extend(self._check_expiration_dates(df))

        if 'bid' in df.columns and 'ask' in df.columns:
            results.extend(self._check_bid_ask_spread(df))

        if 'implied_volatility' in df.columns:
            results.append(self._check_positive_values(df, 'implied_volatility'))

        if 'delta' in df.columns:
            results.extend(self._check_delta_range(df))

        results.extend(self._check_duplicates_multi(df, ['strike', 'option_type', 'expiration_date']))

        return self._create_report(file_name, results)

    # === Core Validation Checks ===

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        required: List[str]
    ) -> List[ValidationResult]:
        """Check that all required columns are present."""
        results = []
        missing = [col for col in required if col not in df.columns]

        if missing:
            results.append(ValidationResult(
                passed=False,
                check_name="Required columns",
                message=f"Missing columns: {missing}",
                severity=ValidationSeverity.ERROR,
                details={"missing_columns": missing}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Required columns",
                message=f"All {len(required)} required columns present",
                details={"columns": required}
            ))

        return results

    def _check_column_types_ohlcv(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check OHLCV columns have correct types."""
        results = []
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    results.append(ValidationResult(
                        passed=False,
                        check_name=f"Column type: {col}",
                        message=f"Column '{col}' is not numeric (type: {df[col].dtype})",
                        severity=ValidationSeverity.ERROR,
                        column=col
                    ))
                else:
                    results.append(ValidationResult(
                        passed=True,
                        check_name=f"Column type: {col}",
                        message=f"Column '{col}' is numeric (type: {df[col].dtype})",
                        column=col
                    ))

        if 'volume' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['volume']):
                results.append(ValidationResult(
                    passed=False,
                    check_name="Column type: volume",
                    message=f"Column 'volume' is not numeric (type: {df['volume'].dtype})",
                    severity=ValidationSeverity.ERROR,
                    column='volume'
                ))

        return results

    def _check_date_column(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> List[ValidationResult]:
        """Check date column is valid datetime."""
        results = []

        if date_col not in df.columns:
            return results

        try:
            # Try to parse dates
            dates = pd.to_datetime(df[date_col])
            null_count = dates.isna().sum()

            if null_count > 0:
                results.append(ValidationResult(
                    passed=False,
                    check_name=f"Date validity: {date_col}",
                    message=f"{null_count} invalid/null dates found",
                    severity=ValidationSeverity.ERROR,
                    column=date_col,
                    details={"null_count": int(null_count)}
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    check_name=f"Date validity: {date_col}",
                    message=f"All {len(df)} dates are valid",
                    column=date_col
                ))

        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                check_name=f"Date parsing: {date_col}",
                message=f"Failed to parse dates: {str(e)}",
                severity=ValidationSeverity.ERROR,
                column=date_col
            ))

        return results

    def _check_price_ranges(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check price columns have valid ranges."""
        results = []
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            if col not in df.columns:
                continue

            # Check for negative values
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                results.append(ValidationResult(
                    passed=False,
                    check_name=f"Price range: {col}",
                    message=f"{neg_count} negative values found in '{col}'",
                    severity=ValidationSeverity.ERROR,
                    column=col,
                    details={"negative_count": int(neg_count)}
                ))
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                results.append(ValidationResult(
                    passed=True,
                    check_name=f"Price range: {col}",
                    message=f"Range: {min_val:.2f} - {max_val:.2f}",
                    column=col,
                    details={"min": float(min_val), "max": float(max_val)}
                ))

            # Check for zeros (suspicious for prices)
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                results.append(ValidationResult(
                    passed=False,
                    check_name=f"Zero prices: {col}",
                    message=f"{zero_count} zero values found in '{col}'",
                    severity=ValidationSeverity.WARNING,
                    column=col,
                    details={"zero_count": int(zero_count)}
                ))

        return results

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check OHLC price consistency (High >= Low, etc.)."""
        results = []

        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return results

        # High >= Low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="OHLC consistency: High >= Low",
                message=f"{invalid_hl} rows have High < Low",
                severity=ValidationSeverity.ERROR,
                details={"invalid_count": int(invalid_hl)}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="OHLC consistency: High >= Low",
                message="All rows have High >= Low"
            ))

        # High >= Open and Close
        invalid_ho = (df['high'] < df['open']).sum()
        invalid_hc = (df['high'] < df['close']).sum()
        total_invalid = invalid_ho + invalid_hc
        if total_invalid > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="OHLC consistency: High >= Open/Close",
                message=f"{invalid_ho} High < Open, {invalid_hc} High < Close",
                severity=ValidationSeverity.ERROR
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="OHLC consistency: High >= Open/Close",
                message="All rows have High >= Open and Close"
            ))

        # Low <= Open and Close
        invalid_lo = (df['low'] > df['open']).sum()
        invalid_lc = (df['low'] > df['close']).sum()
        total_invalid = invalid_lo + invalid_lc
        if total_invalid > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="OHLC consistency: Low <= Open/Close",
                message=f"{invalid_lo} Low > Open, {invalid_lc} Low > Close",
                severity=ValidationSeverity.ERROR
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="OHLC consistency: Low <= Open/Close",
                message="All rows have Low <= Open and Close"
            ))

        return results

    def _check_volume(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check volume column validity."""
        results = []

        if 'volume' not in df.columns:
            return results

        # Check for negative volume
        neg_count = (df['volume'] < 0).sum()
        if neg_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="Volume: non-negative",
                message=f"{neg_count} negative volume values",
                severity=ValidationSeverity.ERROR,
                column='volume'
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Volume: non-negative",
                message="All volume values are non-negative",
                column='volume'
            ))

        # Check for zero volume (warning)
        if not self.config.allow_zero_volume:
            zero_count = (df['volume'] == 0).sum()
            if zero_count > 0:
                results.append(ValidationResult(
                    passed=True,  # Warning, not failure
                    check_name="Volume: zero values",
                    message=f"{zero_count} zero volume values (may be valid for some dates)",
                    severity=ValidationSeverity.WARNING,
                    column='volume',
                    details={"zero_count": int(zero_count)}
                ))

        return results

    def _check_temporal_order(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> List[ValidationResult]:
        """Check data is in chronological order."""
        results = []

        if date_col not in df.columns:
            return results

        try:
            dates = pd.to_datetime(df[date_col])
            is_sorted = dates.is_monotonic_increasing

            if is_sorted:
                results.append(ValidationResult(
                    passed=True,
                    check_name="Chronological order",
                    message="Data is in ascending chronological order"
                ))
            else:
                # Check if descending
                if dates.is_monotonic_decreasing:
                    results.append(ValidationResult(
                        passed=False,
                        check_name="Chronological order",
                        message="Data is in descending order (should be ascending)",
                        severity=ValidationSeverity.WARNING
                    ))
                else:
                    results.append(ValidationResult(
                        passed=False,
                        check_name="Chronological order",
                        message="Data is not in chronological order",
                        severity=ValidationSeverity.ERROR
                    ))

        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                check_name="Chronological order",
                message=f"Could not check order: {str(e)}",
                severity=ValidationSeverity.WARNING
            ))

        return results

    def _check_duplicates(
        self,
        df: pd.DataFrame,
        column: str
    ) -> List[ValidationResult]:
        """Check for duplicate values in a column."""
        results = []

        if column not in df.columns:
            return results

        dup_count = df[column].duplicated().sum()

        if dup_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name=f"Duplicates: {column}",
                message=f"{dup_count} duplicate values found",
                severity=ValidationSeverity.ERROR,
                column=column,
                details={"duplicate_count": int(dup_count)}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name=f"Duplicates: {column}",
                message="No duplicate values"
            ))

        return results

    def _check_duplicates_multi(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> List[ValidationResult]:
        """Check for duplicate combinations of multiple columns."""
        results = []

        available_cols = [c for c in columns if c in df.columns]
        if len(available_cols) < 2:
            return results

        dup_count = df.duplicated(subset=available_cols, keep=False).sum()

        if dup_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name=f"Duplicates: {available_cols}",
                message=f"{dup_count} duplicate row combinations found",
                severity=ValidationSeverity.WARNING,
                details={"duplicate_count": int(dup_count), "columns": available_cols}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name=f"Duplicates: {available_cols}",
                message="No duplicate combinations"
            ))

        return results

    def _check_gaps(
        self,
        df: pd.DataFrame,
        date_col: str,
        max_gap_days: int
    ) -> List[ValidationResult]:
        """Check for gaps in time series data."""
        results = []

        if date_col not in df.columns or len(df) < 2:
            return results

        try:
            dates = pd.to_datetime(df[date_col]).sort_values()
            gaps = dates.diff().dt.days.dropna()

            # Filter for significant gaps (more than max_gap_days trading days ~ max_gap_days * 1.5 calendar days)
            threshold = max_gap_days * 1.5
            large_gaps = gaps[gaps > threshold]

            if len(large_gaps) > 0:
                max_gap = int(large_gaps.max())
                gap_count = len(large_gaps)
                results.append(ValidationResult(
                    passed=True,  # Gaps are warnings, not failures
                    check_name="Date gaps",
                    message=f"{gap_count} gaps > {max_gap_days} trading days found (max: {max_gap} days)",
                    severity=ValidationSeverity.WARNING,
                    details={
                        "gap_count": gap_count,
                        "max_gap_days": max_gap,
                        "threshold": max_gap_days
                    }
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    check_name="Date gaps",
                    message=f"No gaps > {max_gap_days} trading days"
                ))

        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                check_name="Date gaps",
                message=f"Could not check gaps: {str(e)}",
                severity=ValidationSeverity.WARNING
            ))

        return results

    def _check_data_points(
        self,
        df: pd.DataFrame,
        min_points: Optional[int] = None
    ) -> List[ValidationResult]:
        """Check minimum number of data points."""
        results = []
        min_required = min_points or self.config.min_data_points

        if len(df) < min_required:
            results.append(ValidationResult(
                passed=False,
                check_name="Data points",
                message=f"Only {len(df)} rows (minimum {min_required} recommended)",
                severity=ValidationSeverity.WARNING,
                details={"row_count": len(df), "minimum": min_required}
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Data points",
                message=f"{len(df)} rows (minimum {min_required})",
                details={"row_count": len(df)}
            ))

        return results

    def _check_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> List[ValidationResult]:
        """Check for statistical outliers using z-score."""
        results = []

        for col in columns:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) < 10:
                continue

            mean = values.mean()
            std = values.std()

            if std == 0:
                continue

            z_scores = np.abs((values - mean) / std)
            outlier_count = (z_scores > self.config.outlier_std_threshold).sum()

            if outlier_count > 0:
                outlier_pct = (outlier_count / len(values)) * 100
                results.append(ValidationResult(
                    passed=True,  # Outliers are informational
                    check_name=f"Outliers: {col}",
                    message=f"{outlier_count} outliers ({outlier_pct:.1f}%) beyond {self.config.outlier_std_threshold}σ",
                    severity=ValidationSeverity.INFO,
                    column=col,
                    details={
                        "outlier_count": int(outlier_count),
                        "threshold_sigma": self.config.outlier_std_threshold,
                        "mean": float(mean),
                        "std": float(std)
                    }
                ))

        return results

    def _check_indicator_columns(
        self,
        df: pd.DataFrame,
        indicator_columns: List[str]
    ) -> List[ValidationResult]:
        """Validate indicator columns."""
        results = []

        for col in indicator_columns:
            if col not in df.columns:
                results.append(ValidationResult(
                    passed=False,
                    check_name=f"Indicator present: {col}",
                    message=f"Indicator column '{col}' not found",
                    severity=ValidationSeverity.WARNING,
                    column=col
                ))
                continue

            # Check if numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                results.append(ValidationResult(
                    passed=False,
                    check_name=f"Indicator type: {col}",
                    message=f"Indicator '{col}' is not numeric",
                    severity=ValidationSeverity.ERROR,
                    column=col
                ))
                continue

            # Count NaN values (expected at start for some indicators)
            nan_count = df[col].isna().sum()
            nan_pct = (nan_count / len(df)) * 100

            # Most indicators have NaN at the start due to lookback period
            if nan_count > 0:
                results.append(ValidationResult(
                    passed=True,
                    check_name=f"Indicator NaN: {col}",
                    message=f"{nan_count} NaN values ({nan_pct:.1f}%) - expected for lookback period",
                    severity=ValidationSeverity.INFO,
                    column=col,
                    details={"nan_count": int(nan_count), "nan_pct": nan_pct}
                ))

            # Check for all NaN (indicator failed)
            if nan_count == len(df):
                results.append(ValidationResult(
                    passed=False,
                    check_name=f"Indicator data: {col}",
                    message=f"Indicator '{col}' has all NaN values",
                    severity=ValidationSeverity.ERROR,
                    column=col
                ))

        return results

    # === Specialized Checks ===

    def _check_positive_values(
        self,
        df: pd.DataFrame,
        column: str,
        allow_zero: bool = True
    ) -> ValidationResult:
        """Check column has positive (or non-negative) values."""
        if column not in df.columns:
            return ValidationResult(
                passed=False,
                check_name=f"Positive values: {column}",
                message=f"Column '{column}' not found",
                severity=ValidationSeverity.ERROR,
                column=column
            )

        if allow_zero:
            invalid_count = (df[column] < 0).sum()
            check_desc = "non-negative"
        else:
            invalid_count = (df[column] <= 0).sum()
            check_desc = "positive"

        if invalid_count > 0:
            return ValidationResult(
                passed=False,
                check_name=f"Positive values: {column}",
                message=f"{invalid_count} non-{check_desc} values in '{column}'",
                severity=ValidationSeverity.ERROR,
                column=column
            )

        return ValidationResult(
            passed=True,
            check_name=f"Positive values: {column}",
            message=f"All values in '{column}' are {check_desc}",
            column=column
        )

    def _check_quarter_values(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check quarter column has valid values (1-4)."""
        results = []

        if 'quarter' not in df.columns:
            return results

        invalid = df[(df['quarter'] < 1) | (df['quarter'] > 4)]
        if len(invalid) > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="Quarter values",
                message=f"{len(invalid)} invalid quarter values (must be 1-4)",
                severity=ValidationSeverity.ERROR,
                column='quarter'
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Quarter values",
                message="All quarter values are valid (1-4)",
                column='quarter'
            ))

        return results

    def _check_placeholder_values(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check for common placeholder values (-999, 999999, etc.)."""
        results = []
        placeholders = [-999, -9999, 999999, -1, 0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            for placeholder in placeholders:
                count = (df[col] == placeholder).sum()
                if count > 0 and count < len(df) * 0.5:  # Not if it's a valid common value
                    results.append(ValidationResult(
                        passed=True,
                        check_name=f"Placeholder check: {col}",
                        message=f"{count} possible placeholder values ({placeholder}) in '{col}'",
                        severity=ValidationSeverity.WARNING,
                        column=col
                    ))

        return results

    def _check_reasonable_values(
        self,
        df: pd.DataFrame,
        column: str
    ) -> List[ValidationResult]:
        """Check for unreasonable extreme values."""
        results = []

        if column not in df.columns:
            return results

        values = df[column].dropna()
        if len(values) == 0:
            return results

        # Check for infinity
        inf_count = np.isinf(values).sum()
        if inf_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name=f"Infinity values: {column}",
                message=f"{inf_count} infinite values in '{column}'",
                severity=ValidationSeverity.ERROR,
                column=column
            ))

        return results

    def _check_future_dates(self, df: pd.DataFrame, date_col: str = 'date') -> List[ValidationResult]:
        """Check for dates in the future."""
        results = []

        if date_col not in df.columns:
            return results

        today = pd.Timestamp.now().normalize()
        dates = pd.to_datetime(df[date_col])
        future_count = (dates > today).sum()

        if future_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="Future dates",
                message=f"{future_count} dates are in the future",
                severity=ValidationSeverity.ERROR,
                column=date_col
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Future dates",
                message="No future dates found"
            ))

        return results

    def _check_transaction_types(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check insider transaction types are valid."""
        results = []
        valid_types = {'BUY', 'SELL', 'P', 'S', 'Buy', 'Sell', 'Purchase', 'Sale'}

        if 'transaction_type' not in df.columns:
            return results

        invalid = ~df['transaction_type'].isin(valid_types)
        invalid_count = invalid.sum()

        if invalid_count > 0:
            invalid_values = df.loc[invalid, 'transaction_type'].unique().tolist()
            results.append(ValidationResult(
                passed=False,
                check_name="Transaction types",
                message=f"{invalid_count} invalid transaction types: {invalid_values[:5]}",
                severity=ValidationSeverity.ERROR,
                column='transaction_type'
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Transaction types",
                message="All transaction types are valid"
            ))

        return results

    def _check_value_calculation(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check value = shares * price."""
        results = []

        if not all(c in df.columns for c in ['shares', 'price', 'value']):
            return results

        calculated = df['shares'] * df['price']
        tolerance = 0.01  # 1% tolerance for rounding

        mismatch = abs(df['value'] - calculated) > (calculated * tolerance)
        mismatch_count = mismatch.sum()

        if mismatch_count > 0:
            results.append(ValidationResult(
                passed=True,  # Warning only
                check_name="Value calculation",
                message=f"{mismatch_count} rows where value != shares * price",
                severity=ValidationSeverity.WARNING
            ))

        return results

    def _check_forex_range(self, df: pd.DataFrame, pair: str) -> List[ValidationResult]:
        """Check forex rates are in reasonable ranges."""
        results = []

        # Define reasonable ranges for common pairs
        ranges = {
            "GBPUSD": (1.0, 2.5),
            "GBPEUR": (0.8, 1.5),
            "EURUSD": (0.8, 1.6),
        }

        pair_upper = pair.upper().replace("/", "")
        expected_range = ranges.get(pair_upper, (0.01, 1000))

        if 'close' in df.columns:
            out_of_range = (df['close'] < expected_range[0]) | (df['close'] > expected_range[1])
            oor_count = out_of_range.sum()

            if oor_count > 0:
                results.append(ValidationResult(
                    passed=True,
                    check_name=f"Forex range: {pair}",
                    message=f"{oor_count} values outside expected range {expected_range}",
                    severity=ValidationSeverity.WARNING,
                    column='close'
                ))

        return results

    def _check_option_types(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check option types are CALL or PUT."""
        results = []
        valid_types = {'CALL', 'PUT', 'C', 'P', 'Call', 'Put', 'call', 'put'}

        if 'option_type' not in df.columns:
            return results

        invalid = ~df['option_type'].isin(valid_types)
        invalid_count = invalid.sum()

        if invalid_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="Option types",
                message=f"{invalid_count} invalid option types (must be CALL or PUT)",
                severity=ValidationSeverity.ERROR,
                column='option_type'
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Option types",
                message="All option types are valid"
            ))

        return results

    def _check_expiration_dates(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check expiration date > snapshot date."""
        results = []

        exp_dates = pd.to_datetime(df['expiration_date'])
        snap_dates = pd.to_datetime(df['snapshot_date'])

        invalid = exp_dates < snap_dates
        invalid_count = invalid.sum()

        if invalid_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="Expiration dates",
                message=f"{invalid_count} rows have expiration < snapshot date",
                severity=ValidationSeverity.ERROR
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Expiration dates",
                message="All expiration dates are after snapshot dates"
            ))

        return results

    def _check_bid_ask_spread(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check bid < ask (with tolerance for zero-width spreads)."""
        results = []

        if not all(c in df.columns for c in ['bid', 'ask']):
            return results

        # Filter out rows where both are 0 or NaN
        valid_rows = (df['bid'] > 0) & (df['ask'] > 0)
        valid_df = df[valid_rows]

        if len(valid_df) == 0:
            return results

        invalid = valid_df['bid'] > valid_df['ask']
        invalid_count = invalid.sum()

        if invalid_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="Bid/Ask spread",
                message=f"{invalid_count} rows have bid > ask",
                severity=ValidationSeverity.ERROR
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Bid/Ask spread",
                message="All rows have bid <= ask"
            ))

        return results

    def _check_delta_range(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check delta is in range [-1, 1]."""
        results = []

        if 'delta' not in df.columns:
            return results

        out_of_range = (df['delta'] < -1) | (df['delta'] > 1)
        oor_count = out_of_range.sum()

        if oor_count > 0:
            results.append(ValidationResult(
                passed=False,
                check_name="Delta range",
                message=f"{oor_count} delta values outside [-1, 1]",
                severity=ValidationSeverity.ERROR,
                column='delta'
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="Delta range",
                message="All delta values in [-1, 1]",
                column='delta'
            ))

        return results

    # === Report Generation ===

    def _create_report(
        self,
        file_name: str,
        results: List[ValidationResult]
    ) -> ValidationReport:
        """Create a validation report from results."""
        timestamp = datetime.now()

        # Count results
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and r.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for r in results if not r.passed and r.severity == ValidationSeverity.WARNING)

        # Overall pass/fail
        overall_passed = failed == 0

        # Calculate quality score (100 - penalty for issues)
        total_checks = len(results)
        if total_checks > 0:
            # Errors have 10% penalty each, warnings 2%
            penalty = (failed * 10) + (warnings * 2)
            quality_score = max(0, 100 - penalty)
        else:
            quality_score = 100.0

        # Log results if logger available
        if self.logger:
            for result in results:
                self.logger.log_validation(result.to_log_entry(file_name, timestamp))

            self.logger.log_validation_summary(
                file_name=file_name,
                passed=overall_passed,
                total_checks=total_checks,
                failed_checks=failed,
                warnings=warnings
            )

        return ValidationReport(
            file_name=file_name,
            timestamp=timestamp,
            passed=overall_passed,
            total_checks=total_checks,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            results=results,
            data_quality_score=quality_score
        )
