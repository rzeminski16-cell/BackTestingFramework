"""
Data validation for CSV files.
"""
import pandas as pd
from typing import List, Set, Optional
from pathlib import Path


class DataValidator:
    """
    Validates CSV data files for backtesting.
    """

    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_columns: List[str],
                                  symbol: str = "") -> None:
        """
        Validate that DataFrame contains all required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            symbol: Symbol name for error messages

        Raises:
            ValueError: If required columns are missing
        """
        df_columns = set(df.columns)
        required_set = set(required_columns)
        missing = required_set - df_columns

        if missing:
            symbol_str = f" for {symbol}" if symbol else ""
            raise ValueError(
                f"Missing required columns{symbol_str}: {sorted(missing)}\n"
                f"Available columns: {sorted(df_columns)}"
            )

    @staticmethod
    def validate_date_column(df: pd.DataFrame, date_column: str = 'date') -> None:
        """
        Validate date column exists and is in correct format.

        Args:
            df: DataFrame to validate
            date_column: Name of date column

        Raises:
            ValueError: If date column is invalid
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found")

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            raise ValueError(f"Column '{date_column}' must be datetime type")

    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> None:
        """
        Validate that specified columns are numeric.

        Args:
            df: DataFrame to validate
            numeric_columns: List of column names that should be numeric

        Raises:
            ValueError: If columns are not numeric
        """
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric type")

    @staticmethod
    def validate_no_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Validate that there are no missing values in specified columns.

        Args:
            df: DataFrame to validate
            columns: Columns to check (None = all columns)

        Raises:
            ValueError: If missing values are found
        """
        cols_to_check = columns if columns else df.columns.tolist()
        missing_counts = df[cols_to_check].isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]

        if len(columns_with_missing) > 0:
            raise ValueError(
                f"Missing values found:\n{columns_with_missing.to_dict()}"
            )

    @staticmethod
    def validate_file_exists(file_path: Path) -> None:
        """
        Validate that file exists.

        Args:
            file_path: Path to file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @staticmethod
    def validate_positive_values(df: pd.DataFrame, columns: List[str]) -> None:
        """
        Validate that specified columns contain only positive values.

        Args:
            df: DataFrame to validate
            columns: Columns to check

        Raises:
            ValueError: If non-positive values found
        """
        for col in columns:
            if col in df.columns:
                if (df[col] <= 0).any():
                    raise ValueError(f"Column '{col}' contains non-positive values")

    @staticmethod
    def validate_data_sorted(df: pd.DataFrame, date_column: str = 'date') -> None:
        """
        Validate that data is sorted by date.

        Args:
            df: DataFrame to validate
            date_column: Name of date column

        Raises:
            ValueError: If data is not sorted
        """
        if not df[date_column].is_monotonic_increasing:
            raise ValueError(f"Data must be sorted by '{date_column}' in ascending order")

    @classmethod
    def validate_data_frame(cls, df: pd.DataFrame, required_columns: List[str],
                           symbol: str = "", allow_missing_values: bool = False) -> None:
        """
        Perform complete validation of a DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: Required column names
            symbol: Symbol name for error messages
            allow_missing_values: If True, skip missing value check

        Raises:
            ValueError: If validation fails
        """
        # Check required columns exist
        cls.validate_required_columns(df, required_columns, symbol)

        # Validate date column if present
        if 'date' in df.columns:
            cls.validate_date_column(df, 'date')
            cls.validate_data_sorted(df, 'date')

        # Validate numeric columns (common ones)
        numeric_cols = [col for col in ['open', 'high', 'low', 'close', 'volume']
                       if col in df.columns]
        if numeric_cols:
            cls.validate_numeric_columns(df, numeric_cols)

        # Validate positive prices
        price_cols = [col for col in ['open', 'high', 'low', 'close']
                     if col in df.columns]
        if price_cols:
            cls.validate_positive_values(df, price_cols)

        # Check for missing values
        if not allow_missing_values:
            cls.validate_no_missing_values(df, required_columns)

        # Basic sanity check: ensure we have data
        if len(df) == 0:
            symbol_str = f" for {symbol}" if symbol else ""
            raise ValueError(f"DataFrame is empty{symbol_str}")
