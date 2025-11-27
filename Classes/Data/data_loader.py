"""
Data loading functionality for CSV files.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime


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

        # Check for required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Missing required columns for {symbol}: {sorted(missing)}\n"
                    f"Available: {sorted(df.columns.tolist())}"
                )

        # Drop rows with NaN in required columns
        if required_columns:
            df.dropna(subset=required_columns, inplace=True)

        return df

    def load_multiple(self, symbols: List[str],
                     required_columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load CSV files for multiple symbols.

        Args:
            symbols: List of security symbols
            required_columns: Optional list of required columns

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        errors = []

        for symbol in symbols:
            try:
                df = self.load_csv(symbol, required_columns)
                data[symbol] = df
            except (FileNotFoundError, ValueError) as e:
                errors.append(f"{symbol}: {str(e)}")

        if errors:
            print(f"Warning: Failed to load {len(errors)} symbols:")
            for error in errors:
                print(f"  - {error}")

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
