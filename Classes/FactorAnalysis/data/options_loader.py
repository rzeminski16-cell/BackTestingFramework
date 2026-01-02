"""
Options Data Loader for Factor Analysis.

Loads options snapshot data with proper handling of snapshot dates
to prevent forward-looking bias.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .validators import DataValidator, ValidationResult
from ..logging.audit_logger import AuditLogger


@dataclass
class OptionsMetadata:
    """Metadata about loaded options data."""
    file_path: str
    row_count: int
    symbols: List[str]
    date_range: Tuple[str, str]
    has_implied_volatility: bool
    has_greeks: bool
    option_types: List[str]
    avg_contracts_per_snapshot: float


class OptionsLoader:
    """
    Loads and validates options snapshot data.

    Handles:
    - Snapshot date alignment
    - Options chain aggregation
    - IV and Greeks extraction
    - Forward-looking bias prevention
    """

    REQUIRED_COLUMNS = ['snapshot_date', 'symbol', 'option_type', 'strike']

    OPTIONAL_COLUMNS = [
        'expiration_date', 'days_to_expiration', 'bid', 'ask', 'last_price',
        'mark', 'volume', 'open_interest', 'implied_volatility',
        'delta', 'gamma', 'vega', 'theta', 'rho', 'bid_size', 'ask_size'
    ]

    def __init__(
        self,
        data_directory: Optional[Union[str, Path]] = None,
        logger: Optional[AuditLogger] = None,
        max_lookback_days: int = 60
    ):
        """
        Initialize OptionsLoader.

        Args:
            data_directory: Base directory for options data files
            logger: Optional audit logger for tracking
            max_lookback_days: Maximum days to look back for options snapshots
        """
        self.data_directory = Path(data_directory) if data_directory else None
        self.logger = logger
        self.validator = DataValidator()
        self.max_lookback_days = max_lookback_days

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and data types."""
        df = df.copy()

        # Normalize column names
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

        # Handle date column variations
        if 'snapshot_date' not in df.columns:
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'snapshot_date'})

        # Parse date columns
        date_columns = ['snapshot_date', 'expiration_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        # Normalize option_type
        if 'option_type' in df.columns:
            df['option_type'] = df['option_type'].str.lower().str.strip()
            df['option_type'] = df['option_type'].replace({
                'c': 'call',
                'p': 'put'
            })

        # Ensure numeric columns
        numeric_cols = [
            'strike', 'bid', 'ask', 'last_price', 'mark', 'volume',
            'open_interest', 'implied_volatility', 'delta', 'gamma',
            'vega', 'theta', 'rho', 'days_to_expiration'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def load(
        self,
        file_path: Optional[Union[str, Path]] = None,
        symbol: Optional[str] = None,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, ValidationResult, OptionsMetadata]:
        """
        Load options data from a CSV file.

        Args:
            file_path: Path to CSV file
            symbol: Symbol to filter for
            validate: Whether to validate the data

        Returns:
            Tuple of (DataFrame, ValidationResult, OptionsMetadata)
        """
        if file_path:
            path = Path(file_path)
        elif symbol and self.data_directory:
            possible_paths = [
                self.data_directory / f"{symbol.upper()}.csv",
                self.data_directory / f"{symbol.upper()}_options.csv",
                self.data_directory / "options" / f"{symbol.upper()}.csv"
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break
            if path is None:
                raise FileNotFoundError(f"Options data not found for {symbol}")
        else:
            raise ValueError("Either file_path or symbol with data_directory must be provided")

        if not path.exists():
            raise FileNotFoundError(f"Options data file not found: {path}")

        if self.logger:
            self.logger.info(f"Loading options data", {"file": str(path)})

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

        # Sort by snapshot_date
        if 'snapshot_date' in df.columns:
            df = df.sort_values('snapshot_date').reset_index(drop=True)

        # Validate
        validation_result = ValidationResult(is_valid=True, row_count=len(df))
        if validate:
            validation_result = self.validator.validate_options_data(df)

        # Create metadata
        date_range = ("", "")
        if 'snapshot_date' in df.columns and len(df) > 0:
            date_range = (
                df['snapshot_date'].min().strftime('%Y-%m-%d'),
                df['snapshot_date'].max().strftime('%Y-%m-%d')
            )

        symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else []
        option_types = df['option_type'].unique().tolist() if 'option_type' in df.columns else []

        # Calculate average contracts per snapshot
        avg_contracts = 0.0
        if 'snapshot_date' in df.columns:
            contracts_per_date = df.groupby('snapshot_date').size()
            avg_contracts = contracts_per_date.mean() if len(contracts_per_date) > 0 else 0.0

        metadata = OptionsMetadata(
            file_path=str(path),
            row_count=len(df),
            symbols=symbols,
            date_range=date_range,
            has_implied_volatility='implied_volatility' in df.columns,
            has_greeks=any(col in df.columns for col in ['delta', 'gamma', 'vega', 'theta']),
            option_types=option_types,
            avg_contracts_per_snapshot=float(avg_contracts)
        )

        if self.logger:
            self.logger.log_data_summary(
                source_name="Options data",
                row_count=len(df),
                column_count=len(df.columns),
                date_range=date_range if date_range[0] else None,
                symbols=symbols[:5] if symbols else None
            )

        return df, validation_result, metadata

    def get_options_snapshot_as_of(
        self,
        df: pd.DataFrame,
        symbol: str,
        as_of_date: pd.Timestamp,
        max_lookback_days: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent options snapshot as of a given date.

        Args:
            df: Options data DataFrame
            symbol: Symbol to look up
            as_of_date: Date to get snapshot as of
            max_lookback_days: Maximum days to look back

        Returns:
            DataFrame with options chain, or None if not found
        """
        as_of_date = pd.to_datetime(as_of_date)
        lookback = max_lookback_days or self.max_lookback_days
        min_date = as_of_date - pd.Timedelta(days=lookback)

        # Filter by symbol
        symbol_data = df[df['symbol'].str.upper() == symbol.upper()].copy()

        if len(symbol_data) == 0:
            return None

        # Get snapshots within lookback window
        available = symbol_data[
            (symbol_data['snapshot_date'] <= as_of_date) &
            (symbol_data['snapshot_date'] >= min_date)
        ]

        if len(available) == 0:
            return None

        # Get most recent snapshot date
        latest_snapshot_date = available['snapshot_date'].max()

        # Return all contracts from that snapshot
        return available[available['snapshot_date'] == latest_snapshot_date].copy()

    def aggregate_options_metrics(
        self,
        snapshot_df: pd.DataFrame,
        underlying_price: Optional[float] = None
    ) -> Dict:
        """
        Aggregate options chain into summary metrics.

        Args:
            snapshot_df: DataFrame with options snapshot
            underlying_price: Current underlying price (for ATM detection)

        Returns:
            Dictionary with aggregated options metrics
        """
        if snapshot_df is None or len(snapshot_df) == 0:
            return {
                'options_iv_median': None,
                'options_iv_mean': None,
                'options_iv_min': None,
                'options_iv_max': None,
                'options_put_call_volume_ratio': None,
                'options_put_call_oi_ratio': None,
                'options_total_volume': 0,
                'options_total_oi': 0,
                'options_avg_bid_ask_spread': None,
                'options_contracts_count': 0,
                '_options_snapshot_date': None,
                '_options_data_found': False
            }

        result = {
            '_options_data_found': True,
            'options_contracts_count': len(snapshot_df)
        }

        # Snapshot date
        if 'snapshot_date' in snapshot_df.columns:
            result['_options_snapshot_date'] = snapshot_df['snapshot_date'].iloc[0]

        # Implied Volatility metrics
        if 'implied_volatility' in snapshot_df.columns:
            iv = snapshot_df['implied_volatility'].dropna()
            if len(iv) > 0:
                result['options_iv_median'] = float(iv.median())
                result['options_iv_mean'] = float(iv.mean())
                result['options_iv_min'] = float(iv.min())
                result['options_iv_max'] = float(iv.max())

                # IV percentile (if we had historical data, we'd compare)
                # For now, just compute relative to current chain
                result['options_iv_spread'] = float(iv.max() - iv.min())
            else:
                result['options_iv_median'] = None
                result['options_iv_mean'] = None
                result['options_iv_min'] = None
                result['options_iv_max'] = None

        # Put/Call ratios
        if 'option_type' in snapshot_df.columns:
            calls = snapshot_df[snapshot_df['option_type'] == 'call']
            puts = snapshot_df[snapshot_df['option_type'] == 'put']

            # Volume ratio
            if 'volume' in snapshot_df.columns:
                call_volume = calls['volume'].sum()
                put_volume = puts['volume'].sum()
                result['options_total_volume'] = float(call_volume + put_volume)
                if call_volume > 0:
                    result['options_put_call_volume_ratio'] = float(put_volume / call_volume)
                else:
                    result['options_put_call_volume_ratio'] = float('inf') if put_volume > 0 else 0.0

            # Open Interest ratio
            if 'open_interest' in snapshot_df.columns:
                call_oi = calls['open_interest'].sum()
                put_oi = puts['open_interest'].sum()
                result['options_total_oi'] = float(call_oi + put_oi)
                if call_oi > 0:
                    result['options_put_call_oi_ratio'] = float(put_oi / call_oi)
                else:
                    result['options_put_call_oi_ratio'] = float('inf') if put_oi > 0 else 0.0

        # Bid-Ask spread
        if 'bid' in snapshot_df.columns and 'ask' in snapshot_df.columns:
            mid = (snapshot_df['bid'] + snapshot_df['ask']) / 2
            spread = (snapshot_df['ask'] - snapshot_df['bid']) / mid.replace(0, np.nan)
            result['options_avg_bid_ask_spread'] = float(spread.mean()) if spread.notna().any() else None

        # ATM metrics (if underlying price provided)
        if underlying_price and 'strike' in snapshot_df.columns:
            # Find ATM options (closest strike to underlying)
            snapshot_df = snapshot_df.copy()
            snapshot_df['_strike_diff'] = abs(snapshot_df['strike'] - underlying_price)
            atm_threshold = underlying_price * 0.05  # Within 5% of underlying

            atm_options = snapshot_df[snapshot_df['_strike_diff'] <= atm_threshold]
            if len(atm_options) > 0 and 'implied_volatility' in atm_options.columns:
                atm_iv = atm_options['implied_volatility'].dropna()
                if len(atm_iv) > 0:
                    result['options_atm_iv'] = float(atm_iv.mean())

        # Greeks aggregation
        for greek in ['delta', 'gamma', 'vega', 'theta']:
            if greek in snapshot_df.columns:
                values = snapshot_df[greek].dropna()
                if len(values) > 0:
                    result[f'options_{greek}_mean'] = float(values.mean())

        return result

    def get_options_metrics_for_trades(
        self,
        df: pd.DataFrame,
        trades_df: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
        max_lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get options metrics aligned to trade entry dates.

        Args:
            df: Options data DataFrame
            trades_df: Trade log DataFrame with 'symbol' and 'entry_date'
            price_data: Optional price data for underlying prices
            max_lookback_days: Maximum lookback for snapshots

        Returns:
            DataFrame with options metrics for each trade
        """
        results = []

        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            entry_date = trade['entry_date']

            # Get underlying price if available
            underlying_price = None
            if price_data is not None and 'close' in price_data.columns:
                price_row = price_data[
                    (price_data['symbol'].str.upper() == symbol.upper()) &
                    (price_data['date'] == entry_date)
                ]
                if len(price_row) > 0:
                    underlying_price = price_row.iloc[0]['close']
                elif 'entry_price' in trade:
                    underlying_price = trade['entry_price']

            # Get options snapshot
            snapshot = self.get_options_snapshot_as_of(
                df, symbol, entry_date, max_lookback_days
            )

            # Aggregate metrics
            metrics = self.aggregate_options_metrics(snapshot, underlying_price)
            metrics['_trade_idx'] = idx

            # Calculate days before entry
            if snapshot is not None and '_options_snapshot_date' in metrics and metrics['_options_snapshot_date']:
                metrics['_options_days_before_entry'] = (
                    entry_date - pd.to_datetime(metrics['_options_snapshot_date'])
                ).days
            else:
                metrics['_options_days_before_entry'] = None

            results.append(metrics)

        result_df = pd.DataFrame(results)

        if self.logger:
            trades_with_data = result_df['_options_data_found'].sum()
            self.logger.log_temporal_alignment(
                source_name="Options data",
                trades_matched=int(trades_with_data),
                trades_missing=len(trades_df) - int(trades_with_data),
                method=f"{max_lookback_days or self.max_lookback_days}-day lookback"
            )

        return result_df

    def compute_iv_percentile(
        self,
        df: pd.DataFrame,
        symbol: str,
        as_of_date: pd.Timestamp,
        lookback_days: int = 252
    ) -> Optional[float]:
        """
        Compute IV percentile rank compared to historical IV.

        Args:
            df: Options data DataFrame
            symbol: Symbol to analyze
            as_of_date: Date to compute percentile as of
            lookback_days: Days of history to use

        Returns:
            IV percentile (0-100), or None if insufficient data
        """
        as_of_date = pd.to_datetime(as_of_date)
        min_date = as_of_date - pd.Timedelta(days=lookback_days)

        # Filter by symbol and date range
        symbol_data = df[
            (df['symbol'].str.upper() == symbol.upper()) &
            (df['snapshot_date'] >= min_date) &
            (df['snapshot_date'] <= as_of_date)
        ]

        if 'implied_volatility' not in symbol_data.columns:
            return None

        # Get daily median IV
        daily_iv = symbol_data.groupby('snapshot_date')['implied_volatility'].median()

        if len(daily_iv) < 20:  # Need minimum history
            return None

        # Current IV (most recent)
        current_iv = daily_iv.iloc[-1]

        # Compute percentile
        percentile = (daily_iv < current_iv).sum() / len(daily_iv) * 100

        return float(percentile)
