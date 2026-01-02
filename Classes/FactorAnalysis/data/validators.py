"""
Data validation utilities for Factor Analysis.

Provides schema validation, data type checking, and cross-file consistency checks.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "column": self.column,
            "row_indices": self.row_indices[:10] if self.row_indices and len(self.row_indices) > 10 else self.row_indices,
            "affected_rows": len(self.row_indices) if self.row_indices else 0,
            "details": self.details
        }


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    valid_row_count: int = 0
    date_range: Optional[Tuple[str, str]] = None
    symbols: List[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "valid_row_count": self.valid_row_count,
            "date_range": self.date_range,
            "symbols": self.symbols,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues]
        }


class DataValidator:
    """
    Validates data for Factor Analysis.

    Provides:
    - Schema validation (required columns)
    - Data type validation
    - Value range validation
    - Cross-file consistency checks
    """

    # Required columns for each data type
    TRADE_LOG_REQUIRED = [
        'trade_id', 'symbol', 'entry_date', 'exit_date',
        'entry_price', 'exit_price', 'pl', 'pl_pct'
    ]

    TRADE_LOG_OPTIONAL = [
        'quantity', 'side', 'initial_stop_loss', 'final_stop_loss',
        'take_profit', 'security_pl', 'fx_pl', 'entry_fx_rate',
        'exit_fx_rate', 'security_currency', 'duration_days',
        'entry_equity', 'entry_capital_available', 'entry_reason',
        'exit_reason', 'commission_paid', 'partial_exits'
    ]

    PRICE_DATA_REQUIRED = ['date', 'close']
    PRICE_DATA_OPTIONAL = ['open', 'high', 'low', 'volume', 'symbol']

    FUNDAMENTAL_REQUIRED = ['date', 'symbol']
    FUNDAMENTAL_OPTIONAL = [
        'report_date', 'fiscaldateending', 'pe_ratio', 'price_to_book',
        'price_to_sales_ttm', 'dividend_yield', 'return_on_equity_ttm',
        'return_on_assets_ttm', 'revenue_growth_yoy', 'earnings_growth_yoy'
    ]

    INSIDER_REQUIRED = ['date', 'symbol', 'transaction_type']
    INSIDER_OPTIONAL = [
        'insider_title', 'shares', 'price', 'value',
        'executive', 'security_type'
    ]

    OPTIONS_REQUIRED = ['snapshot_date', 'symbol', 'option_type', 'strike']
    OPTIONS_OPTIONAL = [
        'expiration_date', 'days_to_expiration', 'bid', 'ask',
        'implied_volatility', 'delta', 'gamma', 'vega', 'theta',
        'volume', 'open_interest'
    ]

    def __init__(self):
        """Initialize DataValidator."""
        pass

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df = df.copy()
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
        return df

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        required: List[str],
        source_name: str
    ) -> List[ValidationIssue]:
        """Check for required columns."""
        issues = []
        df_columns = set(df.columns.str.lower())

        missing = [c for c in required if c.lower() not in df_columns]
        if missing:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Missing required columns in {source_name}",
                details={"missing": missing, "available": list(df.columns)}
            ))

        return issues

    def _check_null_values(
        self,
        df: pd.DataFrame,
        columns: List[str],
        source_name: str
    ) -> List[ValidationIssue]:
        """Check for null values in specified columns."""
        issues = []
        df_columns = df.columns.str.lower()

        for col in columns:
            col_lower = col.lower()
            if col_lower in df_columns.tolist():
                actual_col = df.columns[df_columns == col_lower][0]
                null_mask = df[actual_col].isna()
                null_count = null_mask.sum()

                if null_count > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Null values found in {source_name}",
                        column=actual_col,
                        row_indices=df.index[null_mask].tolist(),
                        details={"null_count": int(null_count), "percentage": f"{100*null_count/len(df):.1f}%"}
                    ))

        return issues

    def _check_date_format(
        self,
        df: pd.DataFrame,
        date_columns: List[str],
        source_name: str
    ) -> List[ValidationIssue]:
        """Check date columns can be parsed."""
        issues = []
        df_columns = df.columns.str.lower()

        for col in date_columns:
            col_lower = col.lower()
            if col_lower in df_columns.tolist():
                actual_col = df.columns[df_columns == col_lower][0]
                try:
                    # Try to parse dates
                    parsed = pd.to_datetime(df[actual_col], errors='coerce')
                    invalid_mask = parsed.isna() & df[actual_col].notna()
                    invalid_count = invalid_mask.sum()

                    if invalid_count > 0:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Invalid date format in {source_name}",
                            column=actual_col,
                            row_indices=df.index[invalid_mask].tolist(),
                            details={
                                "invalid_count": int(invalid_count),
                                "sample_values": df.loc[invalid_mask, actual_col].head(5).tolist()
                            }
                        ))
                except Exception as e:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Cannot parse date column {actual_col}: {e}",
                        column=actual_col
                    ))

        return issues

    def _check_numeric_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        source_name: str
    ) -> List[ValidationIssue]:
        """Check that specified columns are numeric."""
        issues = []
        df_columns = df.columns.str.lower()

        for col in columns:
            col_lower = col.lower()
            if col_lower in df_columns.tolist():
                actual_col = df.columns[df_columns == col_lower][0]
                if not pd.api.types.is_numeric_dtype(df[actual_col]):
                    # Try to convert
                    try:
                        converted = pd.to_numeric(df[actual_col], errors='coerce')
                        invalid_mask = converted.isna() & df[actual_col].notna()
                        invalid_count = invalid_mask.sum()

                        if invalid_count > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Non-numeric values in numeric column in {source_name}",
                                column=actual_col,
                                row_indices=df.index[invalid_mask].tolist(),
                                details={"invalid_count": int(invalid_count)}
                            ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Cannot convert {actual_col} to numeric: {e}",
                            column=actual_col
                        ))

        return issues

    def _check_value_ranges(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        source_name: str = ""
    ) -> List[ValidationIssue]:
        """Check value ranges for a column."""
        issues = []
        df_columns = df.columns.str.lower()
        col_lower = column.lower()

        if col_lower not in df_columns.tolist():
            return issues

        actual_col = df.columns[df_columns == col_lower][0]
        values = pd.to_numeric(df[actual_col], errors='coerce')

        if min_val is not None:
            below_min = values < min_val
            if below_min.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Values below minimum ({min_val}) in {source_name}",
                    column=actual_col,
                    row_indices=df.index[below_min].tolist(),
                    details={"min_found": float(values.min())}
                ))

        if max_val is not None:
            above_max = values > max_val
            if above_max.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Values above maximum ({max_val}) in {source_name}",
                    column=actual_col,
                    row_indices=df.index[above_max].tolist(),
                    details={"max_found": float(values.max())}
                ))

        return issues

    def _check_future_dates(
        self,
        df: pd.DataFrame,
        date_column: str,
        source_name: str
    ) -> List[ValidationIssue]:
        """Check for future dates."""
        issues = []
        df_columns = df.columns.str.lower()
        col_lower = date_column.lower()

        if col_lower not in df_columns.tolist():
            return issues

        actual_col = df.columns[df_columns == col_lower][0]
        dates = pd.to_datetime(df[actual_col], errors='coerce')
        today = pd.Timestamp.now().normalize()

        future_mask = dates > today
        if future_mask.any():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Future dates found in {source_name}",
                column=actual_col,
                row_indices=df.index[future_mask].tolist(),
                details={"future_count": int(future_mask.sum())}
            ))

        return issues

    def validate_trade_log(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a trade log DataFrame.

        Args:
            df: Trade log DataFrame

        Returns:
            ValidationResult with issues found
        """
        result = ValidationResult(
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns)
        )

        df = self._normalize_columns(df)

        # Check required columns
        for issue in self._check_required_columns(df, self.TRADE_LOG_REQUIRED, "trade_log"):
            result.add_issue(issue)

        if not result.is_valid:
            return result

        # Check null values in required columns
        for issue in self._check_null_values(df, self.TRADE_LOG_REQUIRED, "trade_log"):
            result.add_issue(issue)

        # Check date formats
        for issue in self._check_date_format(df, ['entry_date', 'exit_date'], "trade_log"):
            result.add_issue(issue)

        # Check numeric columns
        numeric_cols = ['entry_price', 'exit_price', 'pl', 'pl_pct', 'quantity']
        for issue in self._check_numeric_columns(df, numeric_cols, "trade_log"):
            result.add_issue(issue)

        # Check value ranges
        for issue in self._check_value_ranges(df, 'entry_price', min_val=0, source_name="trade_log"):
            result.add_issue(issue)
        for issue in self._check_value_ranges(df, 'exit_price', min_val=0, source_name="trade_log"):
            result.add_issue(issue)

        # Check for future dates
        for issue in self._check_future_dates(df, 'entry_date', "trade_log"):
            result.add_issue(issue)

        # Get date range and symbols
        try:
            entry_dates = pd.to_datetime(df['entry_date'], errors='coerce')
            result.date_range = (
                entry_dates.min().strftime('%Y-%m-%d'),
                entry_dates.max().strftime('%Y-%m-%d')
            )
        except:
            pass

        if 'symbol' in df.columns:
            result.symbols = df['symbol'].unique().tolist()

        # Count valid rows (no nulls in required columns)
        valid_mask = df[self.TRADE_LOG_REQUIRED].notna().all(axis=1)
        result.valid_row_count = int(valid_mask.sum())

        return result

    def validate_price_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate price/indicator data DataFrame."""
        result = ValidationResult(
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns)
        )

        df = self._normalize_columns(df)

        # Check required columns
        for issue in self._check_required_columns(df, self.PRICE_DATA_REQUIRED, "price_data"):
            result.add_issue(issue)

        if not result.is_valid:
            return result

        # Check date format
        for issue in self._check_date_format(df, ['date'], "price_data"):
            result.add_issue(issue)

        # Check numeric columns
        for issue in self._check_numeric_columns(df, ['close', 'open', 'high', 'low', 'volume'], "price_data"):
            result.add_issue(issue)

        # Check value ranges
        for issue in self._check_value_ranges(df, 'close', min_val=0, source_name="price_data"):
            result.add_issue(issue)

        # Get date range
        try:
            dates = pd.to_datetime(df['date'], errors='coerce')
            result.date_range = (
                dates.min().strftime('%Y-%m-%d'),
                dates.max().strftime('%Y-%m-%d')
            )
        except:
            pass

        if 'symbol' in df.columns:
            result.symbols = df['symbol'].unique().tolist()

        result.valid_row_count = int(df[self.PRICE_DATA_REQUIRED].notna().all(axis=1).sum())

        return result

    def validate_fundamental_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate fundamental data DataFrame."""
        result = ValidationResult(
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns)
        )

        df = self._normalize_columns(df)

        # Check required columns
        for issue in self._check_required_columns(df, self.FUNDAMENTAL_REQUIRED, "fundamentals"):
            result.add_issue(issue)

        if not result.is_valid:
            return result

        # Check date format
        date_cols = ['date', 'report_date', 'fiscaldateending']
        for issue in self._check_date_format(df, date_cols, "fundamentals"):
            result.add_issue(issue)

        # Get date range and symbols
        try:
            dates = pd.to_datetime(df['date'], errors='coerce')
            result.date_range = (
                dates.min().strftime('%Y-%m-%d'),
                dates.max().strftime('%Y-%m-%d')
            )
        except:
            pass

        result.symbols = df['symbol'].unique().tolist()
        result.valid_row_count = int(df[self.FUNDAMENTAL_REQUIRED].notna().all(axis=1).sum())

        return result

    def validate_insider_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate insider transaction data DataFrame."""
        result = ValidationResult(
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns)
        )

        df = self._normalize_columns(df)

        # Check required columns
        for issue in self._check_required_columns(df, self.INSIDER_REQUIRED, "insider"):
            result.add_issue(issue)

        if not result.is_valid:
            return result

        # Check date format
        for issue in self._check_date_format(df, ['date'], "insider"):
            result.add_issue(issue)

        # Get date range and symbols
        try:
            dates = pd.to_datetime(df['date'], errors='coerce')
            result.date_range = (
                dates.min().strftime('%Y-%m-%d'),
                dates.max().strftime('%Y-%m-%d')
            )
        except:
            pass

        result.symbols = df['symbol'].unique().tolist()
        result.valid_row_count = int(df[self.INSIDER_REQUIRED].notna().all(axis=1).sum())

        return result

    def validate_options_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate options data DataFrame."""
        result = ValidationResult(
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns)
        )

        df = self._normalize_columns(df)

        # Check required columns
        for issue in self._check_required_columns(df, self.OPTIONS_REQUIRED, "options"):
            result.add_issue(issue)

        if not result.is_valid:
            return result

        # Check date format
        for issue in self._check_date_format(df, ['snapshot_date', 'expiration_date'], "options"):
            result.add_issue(issue)

        # Check numeric columns
        for issue in self._check_numeric_columns(df, ['strike', 'implied_volatility', 'delta'], "options"):
            result.add_issue(issue)

        # Get date range and symbols
        try:
            dates = pd.to_datetime(df['snapshot_date'], errors='coerce')
            result.date_range = (
                dates.min().strftime('%Y-%m-%d'),
                dates.max().strftime('%Y-%m-%d')
            )
        except:
            pass

        result.symbols = df['symbol'].unique().tolist()
        result.valid_row_count = int(df[self.OPTIONS_REQUIRED].notna().all(axis=1).sum())

        return result

    def check_cross_file_consistency(
        self,
        trade_symbols: Set[str],
        price_symbols: Set[str],
        fundamental_symbols: Optional[Set[str]] = None,
        insider_symbols: Optional[Set[str]] = None,
        options_symbols: Optional[Set[str]] = None
    ) -> List[ValidationIssue]:
        """
        Check consistency across data files.

        Args:
            trade_symbols: Symbols in trade logs
            price_symbols: Symbols in price data
            fundamental_symbols: Symbols in fundamental data
            insider_symbols: Symbols in insider data
            options_symbols: Symbols in options data

        Returns:
            List of validation issues
        """
        issues = []

        # Check trade symbols exist in price data
        missing_price = trade_symbols - price_symbols
        if missing_price:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Trade symbols not found in price data",
                details={"missing_symbols": list(missing_price)}
            ))

        # Check fundamental data coverage
        if fundamental_symbols is not None:
            missing_fundamental = trade_symbols - fundamental_symbols
            if missing_fundamental:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Trade symbols not found in fundamental data",
                    details={"missing_symbols": list(missing_fundamental)}
                ))

        # Check insider data coverage
        if insider_symbols is not None:
            missing_insider = trade_symbols - insider_symbols
            if missing_insider:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Trade symbols not found in insider data",
                    details={"missing_symbols": list(missing_insider)}
                ))

        # Check options data coverage
        if options_symbols is not None:
            missing_options = trade_symbols - options_symbols
            if missing_options:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Trade symbols not found in options data",
                    details={"missing_symbols": list(missing_options)}
                ))

        return issues
