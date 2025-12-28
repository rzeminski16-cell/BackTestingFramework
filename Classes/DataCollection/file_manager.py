"""
File Manager for the Data Collection System.

Handles all file I/O operations including:
- CSV file writing with proper formatting
- Folder structure creation and management
- Metadata generation and storage
- Missing data handling (interpolation, forward-fill)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict

from .config import MissingDataHandling
from .logging_manager import SessionLogger


@dataclass
class FileMetadata:
    """Metadata for a single output file."""
    file_name: str
    file_path: str
    symbol: str
    data_type: str  # daily, weekly, fundamental, insider, forex, options
    rows: int
    columns: int
    column_names: List[str]
    date_range: Optional[List[str]] = None  # [start_date, end_date]
    size_kb: float = 0.0
    indicators: List[str] = field(default_factory=list)
    validation_status: str = "PENDING"
    data_quality_score: float = 100.0
    missing_data_handled: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionMetadata:
    """Metadata for a complete collection session."""
    session_id: str
    timestamp: str
    data_types: List[str]
    tickers: List[str]
    files_created: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    api_usage: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path):
        """Save metadata to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'SessionMetadata':
        """Load metadata from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class FileManager:
    """
    Manages file operations for the data collection system.

    Handles CSV writing, folder structure, and metadata generation.
    """

    # CSV formatting options
    CSV_ENCODING = 'utf-8'
    CSV_LINE_TERMINATOR = '\n'
    DATE_FORMAT = '%Y-%m-%d'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    PRICE_PRECISION = 2
    RATIO_PRECISION = 6
    VOLUME_PRECISION = 0

    def __init__(
        self,
        output_dir: Path,
        logger: Optional[SessionLogger] = None
    ):
        """
        Initialize file manager.

        Args:
            output_dir: Base output directory for all data files
            logger: Optional session logger
        """
        self.output_dir = Path(output_dir)
        self.logger = logger

        # Create base directory structure
        self._ensure_directories()

        # Track files created
        self._files_created: Dict[str, FileMetadata] = {}

    def _ensure_directories(self):
        """Create the required directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / 'daily',
            self.output_dir / 'weekly',
            self.output_dir / 'fundamentals',
            self.output_dir / 'insider_transactions',
            self.output_dir / 'forex',
            self.output_dir / 'options',
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, data_type: str, file_name: str) -> Path:
        """
        Get the full output path for a file.

        Args:
            data_type: Type of data (daily, weekly, etc.)
            file_name: Name of the file

        Returns:
            Full path to the file
        """
        type_dirs = {
            'daily': 'daily',
            'weekly': 'weekly',
            'fundamental': 'fundamentals',
            'insider': 'insider_transactions',
            'forex': 'forex',
            'options': 'options',
        }

        subdir = type_dirs.get(data_type, data_type)
        return self.output_dir / subdir / file_name

    def write_daily_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        indicators: Optional[List[str]] = None,
        missing_handling: MissingDataHandling = MissingDataHandling.FORWARD_FILL
    ) -> FileMetadata:
        """
        Write daily price and indicator data to CSV.

        Args:
            df: DataFrame with daily data
            symbol: Stock symbol
            indicators: List of indicator columns included
            missing_handling: How to handle missing data

        Returns:
            FileMetadata for the written file
        """
        file_name = f"{symbol}_daily.csv"
        file_path = self.get_output_path('daily', file_name)

        # Prepare data
        df = self._prepare_dataframe(df)
        df = self._handle_missing_data(df, missing_handling)
        df = self._order_columns_ohlcv(df, indicators or [])
        df = self._format_numeric_columns(df)

        # Write CSV
        missing_handled = df.isna().sum().sum()
        df.to_csv(
            file_path,
            index=False,
            encoding=self.CSV_ENCODING,
            lineterminator=self.CSV_LINE_TERMINATOR,
            date_format=self.DATE_FORMAT
        )

        # Create metadata
        metadata = self._create_file_metadata(
            df=df,
            file_name=file_name,
            file_path=file_path,
            symbol=symbol,
            data_type='daily',
            indicators=indicators or [],
            missing_handled=missing_handled
        )

        self._files_created[file_name] = metadata

        if self.logger:
            self.logger.log_session_info(
                f"SAVED: {file_name} ({metadata.rows} rows, {metadata.columns} columns)"
            )

        return metadata

    def write_weekly_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        indicators: Optional[List[str]] = None,
        missing_handling: MissingDataHandling = MissingDataHandling.FORWARD_FILL
    ) -> FileMetadata:
        """Write weekly price and indicator data to CSV."""
        file_name = f"{symbol}_weekly.csv"
        file_path = self.get_output_path('weekly', file_name)

        df = self._prepare_dataframe(df)
        df = self._handle_missing_data(df, missing_handling)
        df = self._order_columns_ohlcv(df, indicators or [])
        df = self._format_numeric_columns(df)

        missing_handled = df.isna().sum().sum()
        df.to_csv(
            file_path,
            index=False,
            encoding=self.CSV_ENCODING,
            lineterminator=self.CSV_LINE_TERMINATOR,
            date_format=self.DATE_FORMAT
        )

        metadata = self._create_file_metadata(
            df=df,
            file_name=file_name,
            file_path=file_path,
            symbol=symbol,
            data_type='weekly',
            indicators=indicators or [],
            missing_handled=missing_handled
        )

        self._files_created[file_name] = metadata
        return metadata

    def write_fundamental_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        missing_handling: MissingDataHandling = MissingDataHandling.SKIP
    ) -> FileMetadata:
        """Write fundamental data to CSV."""
        file_name = f"{symbol}_fundamental.csv"
        file_path = self.get_output_path('fundamental', file_name)

        df = self._prepare_dataframe(df)
        df = self._handle_missing_data(df, missing_handling)
        df = self._order_columns_fundamental(df)

        missing_handled = df.isna().sum().sum()
        df.to_csv(
            file_path,
            index=False,
            encoding=self.CSV_ENCODING,
            lineterminator=self.CSV_LINE_TERMINATOR,
            date_format=self.DATE_FORMAT
        )

        metadata = self._create_file_metadata(
            df=df,
            file_name=file_name,
            file_path=file_path,
            symbol=symbol,
            data_type='fundamental',
            missing_handled=missing_handled
        )

        self._files_created[file_name] = metadata
        return metadata

    def write_insider_data(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> FileMetadata:
        """Write insider transaction data to CSV."""
        file_name = f"{symbol}_insider.csv"
        file_path = self.get_output_path('insider', file_name)

        df = self._prepare_dataframe(df)
        df = self._order_columns_insider(df)

        df.to_csv(
            file_path,
            index=False,
            encoding=self.CSV_ENCODING,
            lineterminator=self.CSV_LINE_TERMINATOR,
            date_format=self.DATE_FORMAT
        )

        metadata = self._create_file_metadata(
            df=df,
            file_name=file_name,
            file_path=file_path,
            symbol=symbol,
            data_type='insider'
        )

        self._files_created[file_name] = metadata
        return metadata

    def write_forex_data(
        self,
        df: pd.DataFrame,
        pair: str,
        missing_handling: MissingDataHandling = MissingDataHandling.FORWARD_FILL
    ) -> FileMetadata:
        """Write forex data to CSV."""
        pair_clean = pair.replace('/', '')
        file_name = f"{pair_clean}_weekly.csv"
        file_path = self.get_output_path('forex', file_name)

        df = self._prepare_dataframe(df)
        df = self._handle_missing_data(df, missing_handling)
        df = self._order_columns_forex(df)

        missing_handled = df.isna().sum().sum()
        df.to_csv(
            file_path,
            index=False,
            encoding=self.CSV_ENCODING,
            lineterminator=self.CSV_LINE_TERMINATOR,
            date_format=self.DATE_FORMAT
        )

        metadata = self._create_file_metadata(
            df=df,
            file_name=file_name,
            file_path=file_path,
            symbol=pair,
            data_type='forex',
            missing_handled=missing_handled
        )

        self._files_created[file_name] = metadata
        return metadata

    def write_options_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        expiration: str,
        option_type: str
    ) -> FileMetadata:
        """
        Write options data to CSV.

        Args:
            df: DataFrame with options data
            symbol: Stock symbol
            expiration: Expiration date (YYYYMMDD format)
            option_type: 'calls' or 'puts'

        Returns:
            FileMetadata
        """
        file_name = f"{symbol}_{expiration}_{option_type}.csv"
        # Create symbol-specific subfolder for options
        symbol_dir = self.output_dir / 'options' / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        file_path = symbol_dir / file_name

        df = self._prepare_dataframe(df)
        df = self._order_columns_options(df)

        df.to_csv(
            file_path,
            index=False,
            encoding=self.CSV_ENCODING,
            lineterminator=self.CSV_LINE_TERMINATOR,
            date_format=self.DATETIME_FORMAT
        )

        metadata = self._create_file_metadata(
            df=df,
            file_name=file_name,
            file_path=file_path,
            symbol=symbol,
            data_type='options'
        )

        self._files_created[file_name] = metadata
        return metadata

    def write_options_yearly(
        self,
        df: pd.DataFrame,
        symbol: str,
        year: int,
        append: bool = True
    ) -> FileMetadata:
        """
        Write options data to a yearly CSV file organized by trading date.

        File structure: raw_data/options/{ticker}/{ticker}_options_{year}.csv

        Each row represents an option contract snapshot on a particular trading date.
        The file contains all trading dates within that year.

        Args:
            df: DataFrame with options data (must include 'snapshot_date' column)
            symbol: Stock ticker symbol
            year: Trading year (e.g., 2017, 2018)
            append: If True, append to existing file; if False, overwrite

        Returns:
            FileMetadata for the written file
        """
        file_name = f"{symbol.upper()}_options_{year}.csv"
        # Create symbol-specific subfolder for options
        symbol_dir = self.output_dir / 'options' / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        file_path = symbol_dir / file_name

        df = self._prepare_dataframe(df)
        df = self._order_columns_options(df)

        # Handle append mode
        if append and file_path.exists():
            existing_df = pd.read_csv(file_path)
            # Convert snapshot_date to datetime for comparison
            if 'snapshot_date' in existing_df.columns:
                existing_df['snapshot_date'] = pd.to_datetime(existing_df['snapshot_date'])
            if 'snapshot_date' in df.columns:
                df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # Define key columns for deduplication
            key_cols = ['snapshot_date', 'symbol', 'option_type', 'expiration_date', 'strike']
            key_cols = [c for c in key_cols if c in combined_df.columns]

            if key_cols:
                combined_df = combined_df.drop_duplicates(subset=key_cols, keep='last')

            # Sort by snapshot_date then expiration_date
            sort_cols = []
            if 'snapshot_date' in combined_df.columns:
                sort_cols.append('snapshot_date')
            if 'expiration_date' in combined_df.columns:
                sort_cols.append('expiration_date')
            if 'strike' in combined_df.columns:
                sort_cols.append('strike')
            if sort_cols:
                combined_df = combined_df.sort_values(sort_cols).reset_index(drop=True)

            df = combined_df

        df.to_csv(
            file_path,
            index=False,
            encoding=self.CSV_ENCODING,
            lineterminator=self.CSV_LINE_TERMINATOR,
            date_format=self.DATE_FORMAT
        )

        metadata = self._create_file_metadata(
            df=df,
            file_name=file_name,
            file_path=file_path,
            symbol=symbol,
            data_type='options'
        )

        self._files_created[file_name] = metadata
        return metadata

    # === Data Preparation Methods ===

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for writing.

        - Normalize column names to lowercase
        - Sort by date if present
        - Reset index
        """
        df = df.copy()

        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]

        # Sort by date if present
        date_cols = ['date', 'quarter_end_date', 'snapshot_date']
        for date_col in date_cols:
            if date_col in df.columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(date_col).reset_index(drop=True)
                    break
                except Exception:
                    pass

        return df

    def _handle_missing_data(
        self,
        df: pd.DataFrame,
        method: MissingDataHandling
    ) -> pd.DataFrame:
        """Apply missing data handling strategy."""
        if method == MissingDataHandling.SKIP:
            return df.dropna()

        elif method == MissingDataHandling.INTERPOLATE:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
            return df

        elif method == MissingDataHandling.FORWARD_FILL:
            return df.ffill()

        return df

    def _order_columns_ohlcv(
        self,
        df: pd.DataFrame,
        indicators: List[str]
    ) -> pd.DataFrame:
        """Order columns for OHLCV data: date, OHLCV, then indicators."""
        priority_order = ['date', 'open', 'high', 'low', 'close', 'volume']
        indicator_order = sorted([i.lower() for i in indicators])

        ordered = []
        # Add priority columns first
        for col in priority_order:
            if col in df.columns:
                ordered.append(col)

        # Add indicator columns
        for col in indicator_order:
            if col in df.columns and col not in ordered:
                ordered.append(col)

        # Add any remaining columns
        for col in df.columns:
            if col not in ordered:
                ordered.append(col)

        return df[ordered]

    def _order_columns_fundamental(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order columns for fundamental data."""
        priority_order = [
            'date', 'quarter_end_date', 'symbol', 'year', 'quarter',
            'eps', 'eps_ttm', 'eps_growth_yoy_pct',
            'revenue', 'revenue_ttm', 'revenue_growth_yoy_pct',
            'pe_ratio', 'pe_ratio_trailing', 'pe_ratio_forward'
        ]

        ordered = []
        for col in priority_order:
            if col in df.columns:
                ordered.append(col)

        for col in sorted(df.columns):
            if col not in ordered:
                ordered.append(col)

        return df[ordered]

    def _order_columns_insider(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order columns for insider data."""
        priority_order = [
            'date', 'symbol', 'insider_name', 'insider_title',
            'transaction_type', 'shares', 'price', 'value',
            'ownership_after', 'relationship'
        ]

        ordered = []
        for col in priority_order:
            if col in df.columns:
                ordered.append(col)

        for col in sorted(df.columns):
            if col not in ordered:
                ordered.append(col)

        return df[ordered]

    def _order_columns_forex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order columns for forex data."""
        priority_order = ['date', 'symbol', 'open', 'high', 'low', 'close']

        ordered = []
        for col in priority_order:
            if col in df.columns:
                ordered.append(col)

        for col in sorted(df.columns):
            if col not in ordered:
                ordered.append(col)

        return df[ordered]

    def _order_columns_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order columns for options data."""
        priority_order = [
            'snapshot_date', 'symbol', 'option_type', 'expiration_date',
            'strike', 'days_to_expiration',
            'bid', 'ask', 'last_price', 'volume', 'open_interest',
            'implied_volatility', 'delta', 'gamma', 'theta', 'vega',
            'prob_itm'
        ]

        ordered = []
        for col in priority_order:
            if col in df.columns:
                ordered.append(col)

        for col in sorted(df.columns):
            if col not in ordered:
                ordered.append(col)

        return df[ordered]

    def _format_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format numeric columns with appropriate precision."""
        df = df.copy()

        price_cols = ['open', 'high', 'low', 'close', 'price', 'strike', 'bid', 'ask', 'last_price']
        ratio_cols = ['pe_ratio', 'pb_ratio', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        volume_cols = ['volume', 'shares', 'open_interest']

        for col in df.columns:
            if col in price_cols and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(self.PRICE_PRECISION)
            elif col in ratio_cols and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(self.RATIO_PRECISION)
            elif col in volume_cols and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(self.VOLUME_PRECISION).astype('Int64')

        return df

    def _create_file_metadata(
        self,
        df: pd.DataFrame,
        file_name: str,
        file_path: Path,
        symbol: str,
        data_type: str,
        indicators: Optional[List[str]] = None,
        missing_handled: int = 0
    ) -> FileMetadata:
        """Create metadata for a written file."""
        # Get date range if date column exists
        date_range = None
        for date_col in ['date', 'quarter_end_date', 'snapshot_date']:
            if date_col in df.columns:
                dates = pd.to_datetime(df[date_col])
                date_range = [
                    dates.min().strftime(self.DATE_FORMAT),
                    dates.max().strftime(self.DATE_FORMAT)
                ]
                break

        # Get file size
        size_kb = file_path.stat().st_size / 1024 if file_path.exists() else 0

        return FileMetadata(
            file_name=file_name,
            file_path=str(file_path),
            symbol=symbol,
            data_type=data_type,
            rows=len(df),
            columns=len(df.columns),
            column_names=list(df.columns),
            date_range=date_range,
            size_kb=round(size_kb, 2),
            indicators=indicators or [],
            missing_data_handled=missing_handled,
            created_at=datetime.now().isoformat()
        )

    # === Session Metadata ===

    def create_session_metadata(
        self,
        session_id: str,
        data_types: List[str],
        tickers: List[str]
    ) -> SessionMetadata:
        """Create session metadata object."""
        return SessionMetadata(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            data_types=data_types,
            tickers=tickers
        )

    def save_session_metadata(
        self,
        metadata: SessionMetadata,
        api_stats: Optional[Dict[str, Any]] = None,
        processing_stats: Optional[Dict[str, Any]] = None,
        decisions: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Save session metadata to JSON file.

        Args:
            metadata: Session metadata object
            api_stats: API usage statistics
            processing_stats: Processing statistics
            decisions: List of decisions made during session
        """
        # Add file information
        metadata.files_created = {
            data_type: {} for data_type in ['daily', 'weekly', 'fundamental', 'insider', 'forex', 'options']
        }

        for file_name, file_meta in self._files_created.items():
            data_type = file_meta.data_type
            if data_type in metadata.files_created:
                metadata.files_created[data_type][file_name] = file_meta.to_dict()

        # Add statistics
        if api_stats:
            metadata.api_usage = api_stats
        if processing_stats:
            metadata.processing_stats = processing_stats
        if decisions:
            metadata.decisions_made = decisions

        # Save to file
        metadata_path = self.output_dir / 'metadata.json'
        metadata.save(metadata_path)

        if self.logger:
            self.logger.log_session_info(f"METADATA: Saved to {metadata_path}")

    def get_files_created(self) -> Dict[str, FileMetadata]:
        """Get all files created in this session."""
        return self._files_created

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of files created."""
        summary = {
            'total_files': len(self._files_created),
            'total_rows': sum(m.rows for m in self._files_created.values()),
            'total_size_kb': sum(m.size_kb for m in self._files_created.values()),
            'by_type': {}
        }

        for file_meta in self._files_created.values():
            data_type = file_meta.data_type
            if data_type not in summary['by_type']:
                summary['by_type'][data_type] = {
                    'count': 0,
                    'rows': 0,
                    'size_kb': 0
                }
            summary['by_type'][data_type]['count'] += 1
            summary['by_type'][data_type]['rows'] += file_meta.rows
            summary['by_type'][data_type]['size_kb'] += file_meta.size_kb

        return summary


class DataTransformer:
    """
    Transforms raw API responses into clean DataFrames.

    Handles the conversion from Alpha Vantage JSON responses
    to pandas DataFrames ready for validation and storage.
    """

    @staticmethod
    def transform_daily_prices(response_data: Dict[str, Any], adjusted: bool = True) -> pd.DataFrame:
        """
        Transform daily time series response to DataFrame.

        Args:
            response_data: Raw API response data
            adjusted: Whether this is adjusted data

        Returns:
            DataFrame with OHLCV columns
        """
        # Find the time series key
        ts_key = None
        for key in response_data.keys():
            if 'Time Series' in key:
                ts_key = key
                break

        if not ts_key:
            return pd.DataFrame()

        time_series = response_data[ts_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        column_map = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adjusted_close',
            '5. volume': 'volume',
            '6. volume': 'volume',
            '7. dividend amount': 'dividend',
            '8. split coefficient': 'split_coefficient',
        }

        df = df.rename(columns=column_map)

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Use adjusted close if available and requested
        if adjusted and 'adjusted_close' in df.columns:
            df['close'] = df['adjusted_close']

        # Add date column from index
        df['date'] = df.index
        df = df.reset_index(drop=True)

        # Select and order columns
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        available = [c for c in columns if c in df.columns]

        return df[available]

    @staticmethod
    def transform_weekly_prices(response_data: Dict[str, Any], adjusted: bool = True) -> pd.DataFrame:
        """Transform weekly time series response to DataFrame."""
        # Same logic as daily
        return DataTransformer.transform_daily_prices(response_data, adjusted)

    @staticmethod
    def transform_indicator(
        response_data: Dict[str, Any],
        indicator_name: str
    ) -> pd.DataFrame:
        """
        Transform technical indicator response to DataFrame.

        Args:
            response_data: Raw API response data
            indicator_name: Name of the indicator

        Returns:
            DataFrame with date and indicator value columns
        """
        # Find the technical analysis key
        ta_key = None
        for key in response_data.keys():
            if 'Technical Analysis' in key:
                ta_key = key
                break

        if not ta_key:
            return pd.DataFrame()

        ta_data = response_data[ta_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(ta_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename columns with indicator prefix
        new_columns = {}
        for col in df.columns:
            # Clean column name
            clean_name = col.lower().strip()
            new_columns[col] = f"{indicator_name.lower()}_{clean_name}" if clean_name != indicator_name.lower() else clean_name

        df = df.rename(columns=new_columns)

        # Add date column
        df['date'] = df.index
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def transform_fundamental_overview(response_data: Dict[str, Any]) -> pd.DataFrame:
        """Transform company overview to DataFrame row."""
        if not response_data or 'Symbol' not in response_data:
            return pd.DataFrame()

        # Select relevant fields
        fields = {
            'Symbol': 'symbol',
            'PERatio': 'pe_ratio',
            'PEGRatio': 'peg_ratio',
            'BookValue': 'book_value',
            'DividendPerShare': 'dividend_per_share',
            'DividendYield': 'dividend_yield',
            'EPS': 'eps',
            'RevenuePerShareTTM': 'revenue_per_share_ttm',
            'ProfitMargin': 'profit_margin',
            'OperatingMarginTTM': 'operating_margin_ttm',
            'ReturnOnAssetsTTM': 'return_on_assets_ttm',
            'ReturnOnEquityTTM': 'return_on_equity_ttm',
            'RevenueTTM': 'revenue_ttm',
            'GrossProfitTTM': 'gross_profit_ttm',
            'QuarterlyEarningsGrowthYOY': 'earnings_growth_yoy',
            'QuarterlyRevenueGrowthYOY': 'revenue_growth_yoy',
            'TrailingPE': 'trailing_pe',
            'ForwardPE': 'forward_pe',
            'PriceToSalesRatioTTM': 'price_to_sales_ttm',
            'PriceToBookRatio': 'price_to_book',
            'Beta': 'beta',
            '52WeekHigh': 'week_52_high',
            '52WeekLow': 'week_52_low',
            '50DayMovingAverage': 'sma_50',
            '200DayMovingAverage': 'sma_200',
        }

        row = {}
        for api_field, our_field in fields.items():
            value = response_data.get(api_field, None)
            if value and value != 'None' and value != '-':
                try:
                    row[our_field] = float(value)
                except (ValueError, TypeError):
                    row[our_field] = value

        row['date'] = datetime.now().strftime('%Y-%m-%d')

        return pd.DataFrame([row])

    @staticmethod
    def transform_earnings(response_data: Dict[str, Any]) -> pd.DataFrame:
        """Transform earnings data to DataFrame."""
        quarterly = response_data.get('quarterlyEarnings', [])
        if not quarterly:
            return pd.DataFrame()

        df = pd.DataFrame(quarterly)

        # Rename columns
        column_map = {
            'fiscalDateEnding': 'date',
            'reportedDate': 'report_date',
            'reportedEPS': 'reported_eps',
            'estimatedEPS': 'estimated_eps',
            'surprise': 'earnings_surprise',
            'surprisePercentage': 'surprise_pct',
        }

        df = df.rename(columns=column_map)

        # Convert to numeric
        numeric_cols = ['reported_eps', 'estimated_eps', 'earnings_surprise', 'surprise_pct']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    @staticmethod
    def transform_insider_transactions(response_data: Dict[str, Any]) -> pd.DataFrame:
        """Transform insider transactions to DataFrame."""
        transactions = response_data.get('data', [])
        if not transactions:
            return pd.DataFrame()

        df = pd.DataFrame(transactions)

        # Rename columns to match our schema
        column_map = {
            'transaction_date': 'date',
            'ticker': 'symbol',
            'executive_name': 'insider_name',
            'executive_title': 'insider_title',
            'acquisition_or_disposal': 'transaction_type',
            'shares': 'shares',
            'share_price': 'price',
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Normalize transaction type
        if 'transaction_type' in df.columns:
            df['transaction_type'] = df['transaction_type'].map({
                'A': 'BUY', 'D': 'SELL',
                'P': 'BUY', 'S': 'SELL',
            }).fillna(df['transaction_type'])

        # Calculate value
        if 'shares' in df.columns and 'price' in df.columns:
            df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['value'] = df['shares'] * df['price']

        return df

    @staticmethod
    def transform_forex_weekly(response_data: Dict[str, Any], pair: str) -> pd.DataFrame:
        """Transform forex weekly data to DataFrame."""
        # Find the time series key
        ts_key = None
        for key in response_data.keys():
            if 'Time Series' in key:
                ts_key = key
                break

        if not ts_key:
            return pd.DataFrame()

        time_series = response_data[ts_key]

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        column_map = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
        }

        df = df.rename(columns=column_map)

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add metadata
        df['date'] = df.index
        df['symbol'] = pair.replace('/', '')
        df = df.reset_index(drop=True)

        return df[['date', 'symbol', 'open', 'high', 'low', 'close']]

    @staticmethod
    def transform_options(response_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """Transform historical options data to DataFrame."""
        data = response_data.get('data', [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Rename columns to match our schema
        column_map = {
            'contractID': 'contract_id',
            'symbol': 'symbol',
            'expiration': 'expiration_date',
            'strike': 'strike',
            'type': 'option_type',
            'last': 'last_price',
            'bid': 'bid',
            'ask': 'ask',
            'volume': 'volume',
            'open_interest': 'open_interest',
            'implied_volatility': 'implied_volatility',
            'delta': 'delta',
            'gamma': 'gamma',
            'theta': 'theta',
            'vega': 'vega',
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Normalize option type
        if 'option_type' in df.columns:
            df['option_type'] = df['option_type'].str.upper()

        # Add snapshot date
        df['snapshot_date'] = datetime.now().strftime('%Y-%m-%d')

        # Convert numeric columns
        numeric_cols = ['strike', 'last_price', 'bid', 'ask', 'volume', 'open_interest',
                        'implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    @staticmethod
    def merge_price_and_indicators(
        price_df: pd.DataFrame,
        indicator_dfs: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge price data with indicator data.

        Args:
            price_df: DataFrame with OHLCV data
            indicator_dfs: List of DataFrames with indicator data

        Returns:
            Merged DataFrame
        """
        if price_df.empty:
            return price_df

        result = price_df.copy()

        for ind_df in indicator_dfs:
            if ind_df.empty:
                continue

            # Merge on date
            if 'date' in ind_df.columns:
                # Get indicator columns (everything except date)
                ind_cols = [c for c in ind_df.columns if c != 'date']

                result = result.merge(
                    ind_df[['date'] + ind_cols],
                    on='date',
                    how='left'
                )

        return result
