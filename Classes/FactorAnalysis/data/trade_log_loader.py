"""
Trade Log Loader for Factor Analysis.

Loads and validates trade log CSV files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .validators import DataValidator, ValidationResult
from ..logging.audit_logger import AuditLogger


@dataclass
class TradeLogMetadata:
    """Metadata about a loaded trade log."""
    file_path: str
    row_count: int
    symbol: Optional[str]
    date_range: Tuple[str, str]
    trade_count: int
    good_trades: int = 0
    bad_trades: int = 0
    indeterminate_trades: int = 0


class TradeLogLoader:
    """
    Loads and validates trade log CSV files.

    Handles:
    - Single or multiple trade log files
    - Column normalization
    - Date parsing
    - Validation and error reporting
    - Metadata tagging for multi-log aggregation
    """

    REQUIRED_COLUMNS = [
        'trade_id', 'symbol', 'entry_date', 'exit_date',
        'entry_price', 'exit_price', 'pl', 'pl_pct'
    ]

    def __init__(self, logger: Optional[AuditLogger] = None):
        """
        Initialize TradeLogLoader.

        Args:
            logger: Optional audit logger for tracking
        """
        self.logger = logger
        self.validator = DataValidator()

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and data types."""
        df = df.copy()

        # Normalize column names
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

        # Handle 'time' column if 'date' is missing
        if 'date' not in df.columns and 'time' in df.columns:
            df = df.rename(columns={'time': 'date'})

        # Parse date columns
        date_columns = ['entry_date', 'exit_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Ensure numeric columns
        numeric_columns = ['entry_price', 'exit_price', 'pl', 'pl_pct', 'quantity',
                          'initial_stop_loss', 'final_stop_loss', 'take_profit',
                          'duration_days', 'entry_equity', 'entry_capital_available',
                          'commission_paid']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate duration_days if not present
        if 'duration_days' not in df.columns and 'entry_date' in df.columns and 'exit_date' in df.columns:
            df['duration_days'] = (df['exit_date'] - df['entry_date']).dt.days

        return df

    def load_single(
        self,
        file_path: Union[str, Path],
        validate: bool = True,
        log_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, ValidationResult, TradeLogMetadata]:
        """
        Load a single trade log file.

        Args:
            file_path: Path to CSV file
            validate: Whether to validate the data
            log_id: Optional identifier for this log

        Returns:
            Tuple of (DataFrame, ValidationResult, TradeLogMetadata)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If critical validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Trade log file not found: {file_path}")

        if self.logger:
            self.logger.info(f"Loading trade log", {"file": str(file_path)})

        # Load CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {e}")

        # Normalize
        df = self._normalize_dataframe(df)

        # Add log metadata
        if log_id:
            df['_log_id'] = log_id
        else:
            df['_log_id'] = file_path.stem

        # Validate
        validation_result = ValidationResult(is_valid=True, row_count=len(df))
        if validate:
            validation_result = self.validator.validate_trade_log(df)

            if self.logger:
                if not validation_result.is_valid:
                    self.logger.error("Validation failed", {
                        "file": str(file_path),
                        "errors": validation_result.error_count
                    })
                elif validation_result.warning_count > 0:
                    self.logger.warning("Validation warnings", {
                        "file": str(file_path),
                        "warnings": validation_result.warning_count
                    })

        # Create metadata
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns and len(df) > 0 else None
        date_range = validation_result.date_range or ("", "")

        metadata = TradeLogMetadata(
            file_path=str(file_path),
            row_count=len(df),
            symbol=symbol,
            date_range=date_range,
            trade_count=len(df)
        )

        if self.logger:
            self.logger.log_data_summary(
                source_name=f"Trade log ({file_path.name})",
                row_count=len(df),
                column_count=len(df.columns),
                date_range=date_range if date_range[0] else None,
                symbols=[symbol] if symbol else None
            )

        return df, validation_result, metadata

    def load_multiple(
        self,
        file_paths: List[Union[str, Path]],
        validate: bool = True,
        combine: bool = True
    ) -> Tuple[pd.DataFrame, List[ValidationResult], List[TradeLogMetadata]]:
        """
        Load multiple trade log files.

        Args:
            file_paths: List of paths to CSV files
            validate: Whether to validate each file
            combine: Whether to combine into single DataFrame

        Returns:
            Tuple of (combined DataFrame, list of ValidationResults, list of Metadata)
        """
        if self.logger:
            self.logger.info(f"Loading multiple trade logs", {"count": len(file_paths)})

        dfs = []
        validation_results = []
        metadata_list = []

        for i, file_path in enumerate(file_paths):
            try:
                df, validation, metadata = self.load_single(
                    file_path,
                    validate=validate,
                    log_id=f"log_{i}"
                )
                dfs.append(df)
                validation_results.append(validation)
                metadata_list.append(metadata)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load trade log", {
                        "file": str(file_path),
                        "error": str(e)
                    })
                # Create error validation result
                validation_results.append(ValidationResult(
                    is_valid=False,
                    row_count=0
                ))

        # Combine DataFrames
        if combine and dfs:
            combined_df = pd.concat(dfs, ignore_index=True)

            # Ensure trade_id is unique across logs
            combined_df['_original_trade_id'] = combined_df['trade_id']
            combined_df['trade_id'] = combined_df['_log_id'] + '_' + combined_df['trade_id'].astype(str)

            if self.logger:
                self.logger.info("Combined trade logs", {
                    "total_trades": len(combined_df),
                    "logs_loaded": len(dfs)
                })

            return combined_df, validation_results, metadata_list
        elif dfs:
            return dfs[0], validation_results, metadata_list
        else:
            return pd.DataFrame(), validation_results, metadata_list

    def load_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*_trades.csv",
        validate: bool = True
    ) -> Tuple[pd.DataFrame, List[ValidationResult], List[TradeLogMetadata]]:
        """
        Load all trade logs from a directory.

        Args:
            directory: Directory containing trade log files
            pattern: Glob pattern to match files
            validate: Whether to validate each file

        Returns:
            Tuple of (combined DataFrame, list of ValidationResults, list of Metadata)
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = list(directory.glob(pattern))

        if not files:
            if self.logger:
                self.logger.warning(f"No trade logs found", {
                    "directory": str(directory),
                    "pattern": pattern
                })
            return pd.DataFrame(), [], []

        if self.logger:
            self.logger.info(f"Found trade logs in directory", {
                "directory": str(directory),
                "count": len(files)
            })

        return self.load_multiple(files, validate=validate)

    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter trades by date range.

        Args:
            df: Trade log DataFrame
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered DataFrame
        """
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df['entry_date'] >= start]

        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df['entry_date'] <= end]

        return df

    def filter_by_symbols(
        self,
        df: pd.DataFrame,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Filter trades by symbols.

        Args:
            df: Trade log DataFrame
            symbols: List of symbols to include

        Returns:
            Filtered DataFrame
        """
        symbols_upper = [s.upper() for s in symbols]
        return df[df['symbol'].str.upper().isin(symbols_upper)]

    def get_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for trade log.

        Args:
            df: Trade log DataFrame

        Returns:
            Dictionary with summary statistics
        """
        if len(df) == 0:
            return {"total_trades": 0}

        return {
            "total_trades": len(df),
            "symbols": df['symbol'].unique().tolist() if 'symbol' in df.columns else [],
            "date_range": {
                "start": df['entry_date'].min().strftime('%Y-%m-%d') if 'entry_date' in df.columns else None,
                "end": df['entry_date'].max().strftime('%Y-%m-%d') if 'entry_date' in df.columns else None
            },
            "total_pl": float(df['pl'].sum()) if 'pl' in df.columns else 0,
            "avg_pl_pct": float(df['pl_pct'].mean()) if 'pl_pct' in df.columns else 0,
            "win_rate": float((df['pl'] > 0).mean()) if 'pl' in df.columns else 0,
            "avg_duration": float(df['duration_days'].mean()) if 'duration_days' in df.columns else 0,
            "unique_logs": df['_log_id'].nunique() if '_log_id' in df.columns else 1
        }
