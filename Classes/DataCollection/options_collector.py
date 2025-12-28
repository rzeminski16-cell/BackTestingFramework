"""
Options Data Collector

Collects historical options data organized by trading date with expiration filtering.

Key Features:
- Collects options for a specified trading date range
- Filters options by max days-to-expiration (DTE) from trading date
- Organizes output by trading year: {ticker}_options_{year}.csv
- Handles rate limiting and caching through AlphaVantageClient

Example:
    collector = OptionsCollector(client, file_manager, logger)
    collector.collect(
        symbol="AAPL",
        from_date=date(2017, 1, 1),
        to_date=date(2018, 12, 31),
        max_dte_forward=365,
        options_type=OptionsType.BOTH,
        progress_callback=callback
    )

Output:
    raw_data/options/AAPL/AAPL_options_2017.csv
    raw_data/options/AAPL/AAPL_options_2018.csv
"""

import logging
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
from collections import defaultdict

import pandas as pd

from .alpha_vantage_client import AlphaVantageClient
from .config import OptionsType, MaxDTEForward
from .file_manager import FileManager
from .logging_manager import SessionLogger

logger = logging.getLogger(__name__)


class OptionsCollector:
    """
    Collects historical options data organized by trading date.

    Filters options based on max days-to-expiration from the trading date
    and saves data organized by trading year.
    """

    def __init__(
        self,
        client: AlphaVantageClient,
        file_manager: FileManager,
        session_logger: Optional[SessionLogger] = None
    ):
        """
        Initialize the options collector.

        Args:
            client: Alpha Vantage API client
            file_manager: File manager for writing output
            session_logger: Optional session logger for tracking
        """
        self.client = client
        self.file_manager = file_manager
        self.session_logger = session_logger

    def collect(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        max_dte_forward: int = 365,
        options_type: OptionsType = OptionsType.BOTH,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Collect options data for a symbol over a trading date range.

        Args:
            symbol: Stock ticker symbol
            from_date: Start of trading date range
            to_date: End of trading date range
            max_dte_forward: Maximum days to expiration from trading date
            options_type: CALLS_ONLY, PUTS_ONLY, or BOTH
            progress_callback: Optional callback(message, progress_pct)

        Returns:
            Dictionary mapping year -> DataFrame of options data
        """
        logger.info(
            f"Collecting options for {symbol} from {from_date} to {to_date} "
            f"with max DTE {max_dte_forward} days"
        )

        # Generate list of trading dates (weekdays)
        trading_dates = self._get_trading_dates(from_date, to_date)
        total_dates = len(trading_dates)

        if total_dates == 0:
            logger.warning(f"No trading dates found between {from_date} and {to_date}")
            return {}

        # Collect data grouped by year
        yearly_data: Dict[int, List[pd.DataFrame]] = defaultdict(list)
        processed = 0
        errors = 0

        for trading_date in trading_dates:
            try:
                if progress_callback:
                    pct = (processed / total_dates) * 100
                    progress_callback(
                        f"Fetching {symbol} options for {trading_date.strftime('%Y-%m-%d')}",
                        pct
                    )

                # Fetch options for this trading date
                df = self._fetch_options_for_date(
                    symbol=symbol,
                    trading_date=trading_date,
                    max_dte_forward=max_dte_forward,
                    options_type=options_type
                )

                if df is not None and not df.empty:
                    year = trading_date.year
                    yearly_data[year].append(df)

                processed += 1

            except Exception as e:
                logger.error(f"Error fetching options for {symbol} on {trading_date}: {e}")
                errors += 1
                processed += 1

        # Combine data by year and write to files
        result = {}
        for year, dfs in yearly_data.items():
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)

                # Write to yearly file
                self.file_manager.write_options_yearly(
                    df=combined_df,
                    symbol=symbol,
                    year=year,
                    append=False  # We're collecting all at once
                )

                result[year] = combined_df
                logger.info(
                    f"Wrote {len(combined_df)} options records for {symbol} in {year}"
                )

        if progress_callback:
            progress_callback(
                f"Completed {symbol}: {processed} dates, {errors} errors",
                100.0
            )

        logger.info(
            f"Completed options collection for {symbol}: "
            f"{processed} dates processed, {errors} errors, "
            f"{sum(len(df) for df in result.values())} total records"
        )

        return result

    def _get_trading_dates(self, from_date: date, to_date: date) -> List[date]:
        """
        Generate list of trading dates (weekdays only).

        Note: This is a simple approximation. For more accuracy,
        you could use a proper trading calendar.
        """
        dates = []
        current = from_date

        while current <= to_date:
            # Weekdays only (0=Monday, 4=Friday)
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)

        return dates

    def _fetch_options_for_date(
        self,
        symbol: str,
        trading_date: date,
        max_dte_forward: int,
        options_type: OptionsType
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and filter options data for a specific trading date.

        Args:
            symbol: Stock ticker symbol
            trading_date: The trading date to fetch options for
            max_dte_forward: Maximum days to expiration
            options_type: CALLS_ONLY, PUTS_ONLY, or BOTH

        Returns:
            Filtered DataFrame or None if no data
        """
        date_str = trading_date.strftime('%Y-%m-%d')

        # Fetch from Alpha Vantage
        response = self.client.get_historical_options(symbol, date=date_str)

        if not response.success or not response.data:
            return None

        # Transform to DataFrame
        df = self._transform_options_response(response.data, symbol, trading_date)

        if df.empty:
            return None

        # Filter by expiration date
        max_expiration = trading_date + timedelta(days=max_dte_forward)
        df = self._filter_by_expiration(df, max_expiration)

        if df.empty:
            return None

        # Filter by option type
        df = self._filter_by_option_type(df, options_type)

        return df if not df.empty else None

    def _transform_options_response(
        self,
        response_data: Dict[str, Any],
        symbol: str,
        trading_date: date
    ) -> pd.DataFrame:
        """
        Transform Alpha Vantage options response to DataFrame.

        Adds snapshot_date column for the trading date.
        """
        data = response_data.get('data', [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Add snapshot_date (trading date)
        df['snapshot_date'] = trading_date

        # Ensure symbol column exists
        if 'symbol' not in df.columns:
            df['symbol'] = symbol.upper()

        # Parse expiration_date if present
        if 'expiration' in df.columns:
            df['expiration_date'] = pd.to_datetime(df['expiration']).dt.date
            df.drop(columns=['expiration'], inplace=True, errors='ignore')
        elif 'expiration_date' in df.columns:
            df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.date

        # Calculate days to expiration
        if 'expiration_date' in df.columns:
            df['days_to_expiration'] = df['expiration_date'].apply(
                lambda x: (x - trading_date).days if pd.notna(x) else None
            )

        # Normalize option_type column
        if 'type' in df.columns:
            df['option_type'] = df['type'].str.lower()
            df.drop(columns=['type'], inplace=True, errors='ignore')
        elif 'option_type' in df.columns:
            df['option_type'] = df['option_type'].str.lower()

        # Convert numeric columns
        numeric_cols = [
            'strike', 'bid', 'ask', 'last', 'last_price',
            'volume', 'open_interest',
            'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename last to last_price if needed
        if 'last' in df.columns and 'last_price' not in df.columns:
            df['last_price'] = df['last']
            df.drop(columns=['last'], inplace=True, errors='ignore')

        return df

    def _filter_by_expiration(
        self,
        df: pd.DataFrame,
        max_expiration: date
    ) -> pd.DataFrame:
        """Filter options to only include those expiring before max_expiration."""
        if 'expiration_date' not in df.columns:
            return df

        # Convert max_expiration to same format
        mask = df['expiration_date'] <= max_expiration
        return df[mask].copy()

    def _filter_by_option_type(
        self,
        df: pd.DataFrame,
        options_type: OptionsType
    ) -> pd.DataFrame:
        """Filter options by call/put type."""
        if options_type == OptionsType.BOTH:
            return df

        if 'option_type' not in df.columns:
            return df

        if options_type == OptionsType.CALLS_ONLY:
            mask = df['option_type'].str.lower() == 'call'
        elif options_type == OptionsType.PUTS_ONLY:
            mask = df['option_type'].str.lower() == 'put'
        else:
            return df

        return df[mask].copy()

    def collect_multiple(
        self,
        symbols: List[str],
        from_date: date,
        to_date: date,
        max_dte_forward: int = 365,
        options_type: OptionsType = OptionsType.BOTH,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Collect options data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            from_date: Start of trading date range
            to_date: End of trading date range
            max_dte_forward: Maximum days to expiration
            options_type: CALLS_ONLY, PUTS_ONLY, or BOTH
            progress_callback: Optional callback(message, progress_pct)

        Returns:
            Dictionary mapping symbol -> {year -> DataFrame}
        """
        results = {}
        total_symbols = len(symbols)

        for i, symbol in enumerate(symbols):
            if progress_callback:
                base_pct = (i / total_symbols) * 100
                progress_callback(f"Processing {symbol} ({i+1}/{total_symbols})", base_pct)

            # Create a wrapped callback that adjusts percentage
            def symbol_callback(msg: str, pct: float):
                if progress_callback:
                    # Scale pct to be within this symbol's range
                    adjusted_pct = base_pct + (pct / total_symbols)
                    progress_callback(msg, adjusted_pct)

            try:
                results[symbol] = self.collect(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    max_dte_forward=max_dte_forward,
                    options_type=options_type,
                    progress_callback=symbol_callback
                )
            except Exception as e:
                logger.error(f"Failed to collect options for {symbol}: {e}")
                results[symbol] = {}

        if progress_callback:
            progress_callback("Completed all symbols", 100.0)

        return results
