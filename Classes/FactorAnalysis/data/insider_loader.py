"""
Insider Transaction Data Loader for Factor Analysis.

Loads insider trading activity data with proper handling of filing dates
to prevent forward-looking bias.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .validators import DataValidator, ValidationResult
from ..logging.audit_logger import AuditLogger


@dataclass
class InsiderMetadata:
    """Metadata about loaded insider data."""
    file_path: str
    row_count: int
    symbols: List[str]
    date_range: Tuple[str, str]
    transaction_types: List[str]
    total_buy_transactions: int
    total_sell_transactions: int


class InsiderLoader:
    """
    Loads and validates insider transaction data.

    Handles:
    - Filing date vs transaction date distinction
    - Forward-looking bias prevention via filing delay
    - Aggregation within time windows
    """

    REQUIRED_COLUMNS = ['date', 'symbol', 'transaction_type']

    OPTIONAL_COLUMNS = [
        'insider_title', 'shares', 'price', 'value',
        'executive', 'security_type', 'filing_date'
    ]

    def __init__(
        self,
        data_directory: Optional[Union[str, Path]] = None,
        logger: Optional[AuditLogger] = None,
        default_filing_delay_days: int = 3
    ):
        """
        Initialize InsiderLoader.

        Args:
            data_directory: Base directory for insider data files
            logger: Optional audit logger for tracking
            default_filing_delay_days: Default delay between transaction and filing
        """
        self.data_directory = Path(data_directory) if data_directory else None
        self.logger = logger
        self.validator = DataValidator()
        self.default_filing_delay_days = default_filing_delay_days

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and data types."""
        df = df.copy()

        # Normalize column names
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

        # Parse date columns
        date_columns = ['date', 'filing_date', 'transaction_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        # Normalize transaction_type
        if 'transaction_type' in df.columns:
            df['transaction_type'] = df['transaction_type'].str.lower().str.strip()
            # Standardize variations
            df['transaction_type'] = df['transaction_type'].replace({
                'purchase': 'buy',
                'sale': 'sell',
                'p': 'buy',
                's': 'sell',
                'acquisition': 'buy',
                'disposition': 'sell'
            })

        # Ensure numeric columns
        for col in ['shares', 'price', 'value']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure boolean executive column
        if 'executive' in df.columns:
            df['executive'] = df['executive'].astype(bool)

        return df

    def _compute_available_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute when insider data became publicly available.

        Uses filing_date if available, otherwise date + default_filing_delay_days.
        """
        df = df.copy()

        if 'filing_date' in df.columns:
            df['_available_date'] = df['filing_date']
            # Fill missing filing_date with date + delay
            missing = df['_available_date'].isna()
            df.loc[missing, '_available_date'] = (
                df.loc[missing, 'date'] + pd.Timedelta(days=self.default_filing_delay_days)
            )
        else:
            # Use date + delay
            df['_available_date'] = df['date'] + pd.Timedelta(days=self.default_filing_delay_days)

        return df

    def load(
        self,
        file_path: Optional[Union[str, Path]] = None,
        symbol: Optional[str] = None,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, ValidationResult, InsiderMetadata]:
        """
        Load insider transaction data from a CSV file.

        Args:
            file_path: Path to CSV file
            symbol: Symbol to filter for
            validate: Whether to validate the data

        Returns:
            Tuple of (DataFrame, ValidationResult, InsiderMetadata)
        """
        if file_path:
            path = Path(file_path)
        elif symbol and self.data_directory:
            possible_paths = [
                self.data_directory / f"{symbol.upper()}.csv",
                self.data_directory / f"{symbol.upper()}_insider.csv",
                self.data_directory / "insider" / f"{symbol.upper()}.csv"
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break
            if path is None:
                raise FileNotFoundError(f"Insider data not found for {symbol}")
        else:
            raise ValueError("Either file_path or symbol with data_directory must be provided")

        if not path.exists():
            raise FileNotFoundError(f"Insider data file not found: {path}")

        if self.logger:
            self.logger.info(f"Loading insider data", {"file": str(path)})

        # Load CSV
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {path}: {e}")

        # Normalize
        df = self._normalize_dataframe(df)

        # Filter by symbol if specified
        if symbol and 'symbol' in df.columns:
            df = df[df['symbol'].str.upper() == symbol.upper()].copy()

        # Compute available date
        df = self._compute_available_date(df)

        # Sort by available date
        df = df.sort_values('_available_date').reset_index(drop=True)

        # Validate
        validation_result = ValidationResult(is_valid=True, row_count=len(df))
        if validate:
            validation_result = self.validator.validate_insider_data(df)

        # Count transactions
        buy_count = (df['transaction_type'] == 'buy').sum() if 'transaction_type' in df.columns else 0
        sell_count = (df['transaction_type'] == 'sell').sum() if 'transaction_type' in df.columns else 0

        # Create metadata
        date_range = ("", "")
        if '_available_date' in df.columns and len(df) > 0:
            date_range = (
                df['_available_date'].min().strftime('%Y-%m-%d'),
                df['_available_date'].max().strftime('%Y-%m-%d')
            )

        symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else []
        transaction_types = df['transaction_type'].unique().tolist() if 'transaction_type' in df.columns else []

        metadata = InsiderMetadata(
            file_path=str(path),
            row_count=len(df),
            symbols=symbols,
            date_range=date_range,
            transaction_types=transaction_types,
            total_buy_transactions=int(buy_count),
            total_sell_transactions=int(sell_count)
        )

        if self.logger:
            self.logger.log_data_summary(
                source_name="Insider data",
                row_count=len(df),
                column_count=len(df.columns),
                date_range=date_range if date_range[0] else None,
                symbols=symbols[:5] if symbols else None
            )
            self.logger.info(f"Transaction breakdown", {
                "buys": buy_count,
                "sells": sell_count
            })

        return df, validation_result, metadata

    def get_insider_activity_in_window(
        self,
        df: pd.DataFrame,
        symbol: str,
        end_date: pd.Timestamp,
        window_days: int = 30,
        delay_days: int = 3
    ) -> Dict:
        """
        Get aggregated insider activity within a time window.

        Args:
            df: Insider data DataFrame
            symbol: Symbol to look up
            end_date: End date of window (trade entry date)
            window_days: Number of days to look back
            delay_days: Filing delay to account for

        Returns:
            Dictionary with aggregated insider metrics
        """
        end_date = pd.to_datetime(end_date)
        effective_end = end_date - pd.Timedelta(days=delay_days)
        start_date = effective_end - pd.Timedelta(days=window_days)

        # Filter by symbol and date range
        symbol_data = df[df['symbol'].str.upper() == symbol.upper()].copy()
        window_data = symbol_data[
            (symbol_data['_available_date'] >= start_date) &
            (symbol_data['_available_date'] <= effective_end)
        ]

        if len(window_data) == 0:
            return {
                'insider_buy_count': 0,
                'insider_sell_count': 0,
                'insider_total_count': 0,
                'insider_net_shares': 0,
                'insider_total_shares': 0,
                'insider_buy_value': 0.0,
                'insider_sell_value': 0.0,
                'insider_net_value': 0.0,
                'insider_buy_sell_ratio': 0.0,
                'insider_score': 0.0,
                'insider_executive_buys': 0,
                'insider_executive_sells': 0,
                '_insider_window_start': start_date,
                '_insider_window_end': effective_end,
                '_insider_transactions_found': 0
            }

        # Calculate counts
        buys = window_data[window_data['transaction_type'] == 'buy']
        sells = window_data[window_data['transaction_type'] == 'sell']

        buy_count = len(buys)
        sell_count = len(sells)
        total_count = buy_count + sell_count

        # Calculate shares
        buy_shares = buys['shares'].sum() if 'shares' in window_data.columns else 0
        sell_shares = sells['shares'].sum() if 'shares' in window_data.columns else 0
        net_shares = buy_shares - sell_shares
        total_shares = buy_shares + sell_shares

        # Calculate values
        buy_value = buys['value'].sum() if 'value' in window_data.columns else 0.0
        sell_value = sells['value'].sum() if 'value' in window_data.columns else 0.0
        net_value = buy_value - sell_value

        # Calculate ratios
        buy_sell_ratio = buy_count / sell_count if sell_count > 0 else (float('inf') if buy_count > 0 else 0.0)

        # Executive transactions (weighted more heavily)
        exec_buys = 0
        exec_sells = 0
        if 'executive' in window_data.columns:
            exec_buys = len(buys[buys['executive'] == True])
            exec_sells = len(sells[sells['executive'] == True])

        # Calculate insider score
        # Higher score = more bullish insider activity
        # Weights: executive transactions count double
        weighted_buys = buy_count + exec_buys  # executives counted twice
        weighted_sells = sell_count + exec_sells

        if weighted_buys + weighted_sells > 0:
            insider_score = (weighted_buys - weighted_sells) / (weighted_buys + weighted_sells) * 100
        else:
            insider_score = 0.0

        return {
            'insider_buy_count': int(buy_count),
            'insider_sell_count': int(sell_count),
            'insider_total_count': int(total_count),
            'insider_net_shares': float(net_shares),
            'insider_total_shares': float(total_shares),
            'insider_buy_value': float(buy_value),
            'insider_sell_value': float(sell_value),
            'insider_net_value': float(net_value),
            'insider_buy_sell_ratio': float(buy_sell_ratio) if buy_sell_ratio != float('inf') else 999.0,
            'insider_score': float(insider_score),
            'insider_executive_buys': int(exec_buys),
            'insider_executive_sells': int(exec_sells),
            '_insider_window_start': start_date,
            '_insider_window_end': effective_end,
            '_insider_transactions_found': int(total_count)
        }

    def get_insider_activity_for_trades(
        self,
        df: pd.DataFrame,
        trades_df: pd.DataFrame,
        window_days: int = 30,
        delay_days: int = 3
    ) -> pd.DataFrame:
        """
        Get insider activity metrics aligned to trade entry dates.

        Args:
            df: Insider data DataFrame
            trades_df: Trade log DataFrame with 'symbol' and 'entry_date'
            window_days: Lookback window in days
            delay_days: Filing delay days

        Returns:
            DataFrame with insider metrics for each trade
        """
        results = []

        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            entry_date = trade['entry_date']

            activity = self.get_insider_activity_in_window(
                df, symbol, entry_date, window_days, delay_days
            )
            activity['_trade_idx'] = idx
            results.append(activity)

        result_df = pd.DataFrame(results)

        if self.logger:
            trades_with_activity = (result_df['insider_total_count'] > 0).sum()
            self.logger.log_temporal_alignment(
                source_name="Insider data",
                trades_matched=int(trades_with_activity),
                trades_missing=len(trades_df) - int(trades_with_activity),
                method=f"{window_days}-day window with {delay_days}-day delay"
            )

        return result_df

    def classify_insider_sentiment(self, metrics: Dict) -> str:
        """
        Classify insider sentiment based on activity metrics.

        Args:
            metrics: Dictionary of insider metrics

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        score = metrics.get('insider_score', 0)

        if score > 25:
            return 'bullish'
        elif score < -25:
            return 'bearish'
        else:
            return 'neutral'
