"""
Price and Indicator Data Loader for Factor Analysis.

Loads daily/weekly OHLCV data with pre-calculated technical indicators.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .validators import DataValidator, ValidationResult
from ..logging.audit_logger import AuditLogger


@dataclass
class PriceDataMetadata:
    """Metadata about loaded price data."""
    file_path: str
    symbol: str
    row_count: int
    date_range: Tuple[str, str]
    frequency: str  # 'daily' or 'weekly'
    indicator_columns: List[str]


class PriceDataLoader:
    """
    Loads and validates price/indicator data.

    Handles:
    - Daily and weekly OHLCV data
    - Pre-calculated technical indicators
    - Multi-symbol loading
    - Forward-fill for missing data
    """

    REQUIRED_COLUMNS = ['date', 'close']
    PRICE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    # Common indicator patterns to identify indicator columns
    INDICATOR_PATTERNS = [
        'sma', 'ema', 'rsi', 'macd', 'atr', 'adx', 'cci', 'bollinger',
        'stoch', 'williams', 'obv', 'roc', 'momentum', 'aroon',
        'kama', 'tema', 'dema', 'trix', 'mfi', 'ppo', 'bop'
    ]

    def __init__(
        self,
        data_directory: Optional[Union[str, Path]] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize PriceDataLoader.

        Args:
            data_directory: Base directory for price data files
            logger: Optional audit logger for tracking
        """
        self.data_directory = Path(data_directory) if data_directory else None
        self.logger = logger
        self.validator = DataValidator()
        self._cache: Dict[str, pd.DataFrame] = {}

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and data types."""
        df = df.copy()

        # Normalize column names
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

        # Handle 'time' column if 'date' is missing
        if 'date' not in df.columns and 'time' in df.columns:
            df = df.rename(columns={'time': 'date'})

        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

        # Ensure numeric columns
        for col in self.PRICE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def _identify_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that are technical indicators."""
        indicators = []
        for col in df.columns:
            col_lower = col.lower()
            # Skip standard price columns
            if col_lower in ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']:
                continue
            # Check if it matches indicator patterns
            for pattern in self.INDICATOR_PATTERNS:
                if pattern in col_lower:
                    indicators.append(col)
                    break
            else:
                # Also include columns with numeric suffixes (e.g., sma_50, rsi_14)
                if any(c.isdigit() for c in col_lower):
                    indicators.append(col)

        return indicators

    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect if data is daily or weekly."""
        if 'date' not in df.columns or len(df) < 2:
            return 'unknown'

        # Calculate average difference between consecutive dates
        date_diff = df['date'].diff().median()
        if date_diff is not None:
            if date_diff.days <= 2:
                return 'daily'
            elif date_diff.days <= 8:
                return 'weekly'
        return 'unknown'

    def load_single(
        self,
        file_path: Optional[Union[str, Path]] = None,
        symbol: Optional[str] = None,
        frequency: str = 'daily',
        validate: bool = True,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, ValidationResult, PriceDataMetadata]:
        """
        Load price data for a single symbol.

        Args:
            file_path: Direct path to CSV file (if provided, overrides symbol lookup)
            symbol: Symbol to load (used with data_directory)
            frequency: 'daily' or 'weekly'
            validate: Whether to validate the data
            use_cache: Whether to use cached data

        Returns:
            Tuple of (DataFrame, ValidationResult, PriceDataMetadata)
        """
        # Determine file path
        if file_path:
            path = Path(file_path)
        elif symbol and self.data_directory:
            path = self.data_directory / frequency / f"{symbol.upper()}.csv"
        else:
            raise ValueError("Either file_path or symbol with data_directory must be provided")

        # Check cache
        cache_key = str(path)
        if use_cache and cache_key in self._cache:
            df = self._cache[cache_key]
            validation = ValidationResult(is_valid=True, row_count=len(df))
            metadata = PriceDataMetadata(
                file_path=str(path),
                symbol=symbol or path.stem.upper(),
                row_count=len(df),
                date_range=(df['date'].min().strftime('%Y-%m-%d'), df['date'].max().strftime('%Y-%m-%d')),
                frequency=frequency,
                indicator_columns=self._identify_indicator_columns(df)
            )
            return df, validation, metadata

        if not path.exists():
            raise FileNotFoundError(f"Price data file not found: {path}")

        if self.logger:
            self.logger.info(f"Loading price data", {"file": str(path)})

        # Load CSV
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {path}: {e}")

        # Normalize
        df = self._normalize_dataframe(df)

        # Add symbol if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol or path.stem.upper()

        # Validate
        validation_result = ValidationResult(is_valid=True, row_count=len(df))
        if validate:
            validation_result = self.validator.validate_price_data(df)

        # Detect frequency
        detected_freq = self._detect_frequency(df)
        if detected_freq != 'unknown':
            frequency = detected_freq

        # Identify indicators
        indicator_columns = self._identify_indicator_columns(df)

        # Create metadata
        date_range = ("", "")
        if 'date' in df.columns and len(df) > 0:
            date_range = (
                df['date'].min().strftime('%Y-%m-%d'),
                df['date'].max().strftime('%Y-%m-%d')
            )

        metadata = PriceDataMetadata(
            file_path=str(path),
            symbol=symbol or path.stem.upper(),
            row_count=len(df),
            date_range=date_range,
            frequency=frequency,
            indicator_columns=indicator_columns
        )

        # Cache
        if use_cache:
            self._cache[cache_key] = df

        if self.logger:
            self.logger.log_data_summary(
                source_name=f"Price data ({metadata.symbol})",
                row_count=len(df),
                column_count=len(df.columns),
                date_range=date_range if date_range[0] else None,
                symbols=[metadata.symbol]
            )
            if indicator_columns:
                self.logger.info(f"Detected {len(indicator_columns)} indicator columns")

        return df, validation_result, metadata

    def load_multiple(
        self,
        symbols: List[str],
        frequency: str = 'daily',
        validate: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, ValidationResult]]:
        """
        Load price data for multiple symbols.

        Args:
            symbols: List of symbols to load
            frequency: 'daily' or 'weekly'
            validate: Whether to validate each file

        Returns:
            Tuple of (dict of DataFrames, dict of ValidationResults)
        """
        if not self.data_directory:
            raise ValueError("data_directory must be set to load by symbols")

        if self.logger:
            self.logger.info(f"Loading price data for multiple symbols", {
                "count": len(symbols),
                "frequency": frequency
            })

        data_dict = {}
        validation_dict = {}

        for symbol in symbols:
            try:
                df, validation, metadata = self.load_single(
                    symbol=symbol,
                    frequency=frequency,
                    validate=validate
                )
                data_dict[symbol] = df
                validation_dict[symbol] = validation

            except FileNotFoundError:
                if self.logger:
                    self.logger.warning(f"Price data not found for {symbol}")
                validation_dict[symbol] = ValidationResult(is_valid=False, row_count=0)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error loading price data", {
                        "symbol": symbol,
                        "error": str(e)
                    })
                validation_dict[symbol] = ValidationResult(is_valid=False, row_count=0)

        return data_dict, validation_dict

    def get_price_at_date(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        forward_fill: bool = True
    ) -> Optional[pd.Series]:
        """
        Get price/indicator data for a specific date.

        Args:
            df: Price data DataFrame
            date: Date to look up
            forward_fill: If True, use last available data for missing dates

        Returns:
            Series with price/indicator data, or None if not found
        """
        date = pd.to_datetime(date)

        # Exact match
        exact = df[df['date'] == date]
        if len(exact) > 0:
            return exact.iloc[0]

        # Forward fill - get last available before date
        if forward_fill:
            before = df[df['date'] < date]
            if len(before) > 0:
                return before.iloc[-1]

        return None

    def get_data_for_dates(
        self,
        df: pd.DataFrame,
        dates: List[pd.Timestamp],
        forward_fill: bool = True
    ) -> pd.DataFrame:
        """
        Get price/indicator data for multiple dates.

        Args:
            df: Price data DataFrame
            dates: List of dates to look up
            forward_fill: If True, use last available data for missing dates

        Returns:
            DataFrame with data for each date
        """
        results = []
        for date in dates:
            row = self.get_price_at_date(df, date, forward_fill=forward_fill)
            if row is not None:
                results.append(row)
            else:
                # Create a row with NaN values
                empty_row = pd.Series(index=df.columns)
                empty_row['date'] = date
                results.append(empty_row)

        return pd.DataFrame(results).reset_index(drop=True)

    def get_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of indicator column names."""
        return self._identify_indicator_columns(df)

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
