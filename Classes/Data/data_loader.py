"""
Data loading functionality for CSV files.

All indicators and data columns are expected to be pre-calculated in the raw data.
This module enforces strict validation - if a required column is missing, a clear
error is raised to inform the user which columns need to be added to their data.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Set
from datetime import datetime


class MissingColumnError(ValueError):
    """
    Error raised when required columns are missing from raw data.

    This error provides clear feedback about which columns are missing
    and which columns are available, helping users understand what
    data they need to collect.
    """

    def __init__(self, symbol: str, missing_columns: List[str], available_columns: List[str]):
        self.symbol = symbol
        self.missing_columns = sorted(missing_columns)
        self.available_columns = sorted(available_columns)

        message = self._format_error_message()
        super().__init__(message)

    def _format_error_message(self) -> str:
        """Format a clear, actionable error message."""
        lines = [
            f"\n{'='*60}",
            f"MISSING REQUIRED COLUMNS FOR {self.symbol}",
            f"{'='*60}",
            "",
            "The following columns are required but not found in the raw data:",
        ]

        for col in self.missing_columns:
            lines.append(f"  ❌ {col}")

        lines.extend([
            "",
            "ACTION REQUIRED:",
            "  1. Re-run data collection with the missing indicators enabled",
            "  2. Or update your strategy to only use available columns",
            "",
            f"Available columns in {self.symbol}:",
        ])

        # Group available columns for readability
        for col in self.available_columns[:20]:  # Show first 20
            lines.append(f"  ✓ {col}")

        if len(self.available_columns) > 20:
            lines.append(f"  ... and {len(self.available_columns) - 20} more columns")

        lines.append(f"{'='*60}\n")

        return "\n".join(lines)


class DataLoader:
    """
    Loads and preprocesses CSV data files for backtesting.
    """

    def __init__(self, data_directory: Path):
        """
        Initialize data loader.

        Args:
            data_directory: Path to directory containing CSV files
        """
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {data_directory}")

    def load_csv(self, symbol: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load CSV file for a symbol.

        Args:
            symbol: Security symbol (e.g., 'AAPL', 'BTCUSD')
            required_columns: Optional list of required columns

        Returns:
            DataFrame with data

        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If required columns missing
        """
        file_path = self.data_directory / f"{symbol}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found for {symbol}: {file_path}")

        # Load CSV
        df = pd.read_csv(file_path)

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()

        # Convert 'time' column to 'date' if it exists
        if 'time' in df.columns and 'date' not in df.columns:
            df.rename(columns={'time': 'date'}, inplace=True)

        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Check for required columns - raise clear error if any are missing
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise MissingColumnError(
                    symbol=symbol,
                    missing_columns=list(missing),
                    available_columns=df.columns.tolist()
                )

        # Drop rows with NaN in required columns
        if required_columns:
            df.dropna(subset=required_columns, inplace=True)

        return df

    def load_multiple(self, symbols: List[str],
                     required_columns: Optional[List[str]] = None,
                     strict: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load CSV files for multiple symbols.

        Args:
            symbols: List of security symbols
            required_columns: Optional list of required columns
            strict: If True, raise error on first missing column. If False, skip and warn.

        Returns:
            Dictionary mapping symbol to DataFrame

        Raises:
            MissingColumnError: If strict=True and required columns are missing
            FileNotFoundError: If a symbol's CSV file is not found
        """
        data = {}
        errors = []
        missing_column_errors = []

        for symbol in symbols:
            try:
                df = self.load_csv(symbol, required_columns)
                data[symbol] = df
            except MissingColumnError as e:
                if strict:
                    # In strict mode, immediately raise the error
                    raise
                missing_column_errors.append(e)
                errors.append(f"{symbol}: Missing columns - {e.missing_columns}")
            except FileNotFoundError as e:
                errors.append(f"{symbol}: File not found")
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")

        if errors:
            print(f"\n⚠️  WARNING: Failed to load {len(errors)} symbols:")
            for error in errors:
                print(f"  - {error}")

            if missing_column_errors:
                print("\n📋 MISSING COLUMN SUMMARY:")
                all_missing = set()
                for e in missing_column_errors:
                    all_missing.update(e.missing_columns)
                print(f"  Columns needed: {sorted(all_missing)}")
                print("  Re-run data collection with these indicators enabled.\n")

        return data

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols (CSV files in data directory).

        Returns:
            List of symbol names
        """
        csv_files = list(self.data_directory.glob("*.csv"))
        symbols = [f.stem for f in csv_files if f.stem.upper() != 'SAMPLE']
        return sorted(symbols)

    def filter_by_date_range(self, df: pd.DataFrame,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Filter DataFrame by date range.

        Args:
            df: DataFrame with 'date' column
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            Filtered DataFrame
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column for filtering")

        filtered = df.copy()

        if start_date:
            filtered = filtered[filtered['date'] >= start_date]

        if end_date:
            filtered = filtered[filtered['date'] <= end_date]

        return filtered.reset_index(drop=True)

    @staticmethod
    def normalize_column_name(column: str) -> str:
        """
        Normalize column name to lowercase.

        Args:
            column: Column name

        Returns:
            Normalized column name
        """
        return column.lower().strip()
