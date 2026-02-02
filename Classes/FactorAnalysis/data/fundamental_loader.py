"""
Fundamental Data Loader for Factor Analysis.

Loads fundamental financial data with proper handling of report dates
to prevent forward-looking bias.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .validators import DataValidator, ValidationResult
from ..logging.audit_logger import AuditLogger


@dataclass
class FundamentalMetadata:
    """Metadata about loaded fundamental data."""
    file_path: str
    row_count: int
    symbols: List[str]
    date_range: Tuple[str, str]
    available_metrics: List[str]
    has_report_date: bool
    has_fiscal_date: bool


class FundamentalLoader:
    """
    Loads and validates fundamental financial data.

    Handles:
    - Report date vs fiscal date distinction
    - Forward-looking bias prevention
    - Multiple metric categories (valuation, profitability, growth, quality)
    """

    REQUIRED_COLUMNS = ['date', 'symbol']

    # Categorized fundamental metrics
    VALUATION_METRICS = [
        'pe_ratio', 'forward_pe', 'trailing_pe', 'peg_ratio',
        'price_to_book', 'price_to_sales_ttm', 'dividend_yield',
        'enterprise_value', 'ev_to_ebitda', 'ev_to_revenue'
    ]

    PROFITABILITY_METRICS = [
        'profit_margin', 'operating_margin_ttm', 'gross_profit_ttm',
        'ebit', 'ebitda', 'netincome', 'revenue_ttm', 'reported_eps'
    ]

    GROWTH_METRICS = [
        'revenue_growth_yoy', 'earnings_surprise', 'estimated_eps'
    ]

    QUALITY_METRICS = [
        'return_on_equity_ttm', 'return_on_assets_ttm', 'book_value',
        'total_assets', 'totalliabilities', 'totalshareholderequity',
        'currentratio', 'debt_to_equity'
    ]

    CASH_FLOW_METRICS = [
        'operatingcashflow', 'cashflowfromfinancing', 'cashflowfrominvestment',
        'capitalexpenditures', 'freecashflow'
    ]

    OTHER_METRICS = [
        'beta', 'dividend_per_share', 'dividendpayout',
        'common_stock_shares_outstanding', 'market_cap'
    ]

    def __init__(
        self,
        data_directory: Optional[Union[str, Path]] = None,
        logger: Optional[AuditLogger] = None,
        default_reporting_delay_days: int = 45
    ):
        """
        Initialize FundamentalLoader.

        Args:
            data_directory: Base directory for fundamental data files
            logger: Optional audit logger for tracking
            default_reporting_delay_days: Default delay to add to fiscal date if report_date missing
        """
        self.data_directory = Path(data_directory) if data_directory else None
        self.logger = logger
        self.validator = DataValidator()
        self.default_reporting_delay_days = default_reporting_delay_days

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and data types."""
        df = df.copy()

        # Normalize column names
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

        # Parse date columns
        date_columns = ['date', 'report_date', 'fiscaldateending']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        # Ensure numeric columns
        all_metrics = (
            self.VALUATION_METRICS + self.PROFITABILITY_METRICS +
            self.GROWTH_METRICS + self.QUALITY_METRICS +
            self.CASH_FLOW_METRICS + self.OTHER_METRICS
        )
        for col in all_metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _compute_as_of_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the as-of date for each row.

        The as-of date is when the data became publicly known:
        - If report_date exists: use report_date
        - Else if fiscaldateending exists: use fiscaldateending + default_reporting_delay_days
        - Else: use date column

        This is critical for avoiding forward-looking bias.
        """
        df = df.copy()

        if 'report_date' in df.columns:
            # Use report_date where available
            df['_as_of_date'] = df['report_date']

            # Fill missing report_date with fiscal date + delay
            if 'fiscaldateending' in df.columns:
                missing_report = df['_as_of_date'].isna()
                df.loc[missing_report, '_as_of_date'] = (
                    df.loc[missing_report, 'fiscaldateending'] +
                    pd.Timedelta(days=self.default_reporting_delay_days)
                )
        elif 'fiscaldateending' in df.columns:
            # Use fiscal date + delay
            df['_as_of_date'] = df['fiscaldateending'] + pd.Timedelta(days=self.default_reporting_delay_days)
        else:
            # Fall back to date column
            df['_as_of_date'] = df['date']

        return df

    def _identify_available_metrics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify which metrics are available in the data."""
        available = {
            'valuation': [],
            'profitability': [],
            'growth': [],
            'quality': [],
            'cash_flow': [],
            'other': []
        }

        for metric in self.VALUATION_METRICS:
            if metric in df.columns and df[metric].notna().any():
                available['valuation'].append(metric)

        for metric in self.PROFITABILITY_METRICS:
            if metric in df.columns and df[metric].notna().any():
                available['profitability'].append(metric)

        for metric in self.GROWTH_METRICS:
            if metric in df.columns and df[metric].notna().any():
                available['growth'].append(metric)

        for metric in self.QUALITY_METRICS:
            if metric in df.columns and df[metric].notna().any():
                available['quality'].append(metric)

        for metric in self.CASH_FLOW_METRICS:
            if metric in df.columns and df[metric].notna().any():
                available['cash_flow'].append(metric)

        for metric in self.OTHER_METRICS:
            if metric in df.columns and df[metric].notna().any():
                available['other'].append(metric)

        return available

    def load(
        self,
        file_path: Optional[Union[str, Path]] = None,
        symbol: Optional[str] = None,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, ValidationResult, FundamentalMetadata]:
        """
        Load fundamental data from a CSV file.

        Args:
            file_path: Path to CSV file
            symbol: Symbol to filter for (if file contains multiple symbols)
            validate: Whether to validate the data

        Returns:
            Tuple of (DataFrame, ValidationResult, FundamentalMetadata)
        """
        if file_path:
            path = Path(file_path)
        elif symbol and self.data_directory:
            # Try common naming patterns
            possible_paths = [
                self.data_directory / f"{symbol.upper()}.csv",
                self.data_directory / f"{symbol.upper()}_fundamentals.csv",
                self.data_directory / "fundamentals" / f"{symbol.upper()}.csv"
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break
            if path is None:
                raise FileNotFoundError(f"Fundamental data not found for {symbol}")
        else:
            raise ValueError("Either file_path or symbol with data_directory must be provided")

        if not path.exists():
            raise FileNotFoundError(f"Fundamental data file not found: {path}")

        if self.logger:
            self.logger.info(f"Loading fundamental data", {"file": str(path)})

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

        # Compute as-of date
        df = self._compute_as_of_date(df)

        # Sort by as-of date
        df = df.sort_values('_as_of_date').reset_index(drop=True)

        # Validate
        validation_result = ValidationResult(is_valid=True, row_count=len(df))
        if validate:
            validation_result = self.validator.validate_fundamental_data(df)

        # Identify available metrics
        available_metrics = self._identify_available_metrics(df)
        all_available = [m for metrics in available_metrics.values() for m in metrics]

        # Create metadata
        date_range = ("", "")
        if '_as_of_date' in df.columns and len(df) > 0:
            date_range = (
                df['_as_of_date'].min().strftime('%Y-%m-%d'),
                df['_as_of_date'].max().strftime('%Y-%m-%d')
            )

        symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else []

        metadata = FundamentalMetadata(
            file_path=str(path),
            row_count=len(df),
            symbols=symbols,
            date_range=date_range,
            available_metrics=all_available,
            has_report_date='report_date' in df.columns,
            has_fiscal_date='fiscaldateending' in df.columns
        )

        if self.logger:
            self.logger.log_data_summary(
                source_name="Fundamental data",
                row_count=len(df),
                column_count=len(df.columns),
                date_range=date_range if date_range[0] else None,
                symbols=symbols[:5] if symbols else None
            )
            self.logger.info(f"Available metrics by category", available_metrics)

        return df, validation_result, metadata

    def get_fundamentals_as_of(
        self,
        df: pd.DataFrame,
        symbol: str,
        as_of_date: pd.Timestamp,
        delay_days: int = 0
    ) -> Optional[pd.Series]:
        """
        Get the most recent fundamental data as of a given date.

        This is the critical method for avoiding forward-looking bias.

        Args:
            df: Fundamental data DataFrame
            symbol: Symbol to look up
            as_of_date: Date to get data as of
            delay_days: Additional delay days beyond report_date

        Returns:
            Series with fundamental data, or None if not found
        """
        as_of_date = pd.to_datetime(as_of_date)
        effective_date = as_of_date - pd.Timedelta(days=delay_days)

        # Filter by symbol
        symbol_data = df[df['symbol'].str.upper() == symbol.upper()].copy()

        if len(symbol_data) == 0:
            return None

        # Get most recent data before effective date
        available = symbol_data[symbol_data['_as_of_date'] <= effective_date]

        if len(available) == 0:
            return None

        # Return most recent
        return available.iloc[-1]

    def get_fundamentals_for_trades(
        self,
        df: pd.DataFrame,
        trades_df: pd.DataFrame,
        delay_days: int = 0
    ) -> pd.DataFrame:
        """
        Get fundamental data aligned to trade entry dates.

        Args:
            df: Fundamental data DataFrame
            trades_df: Trade log DataFrame with 'symbol' and 'entry_date'
            delay_days: Reporting delay days

        Returns:
            DataFrame with fundamental data for each trade
        """
        results = []

        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            entry_date = trade['entry_date']

            fundamentals = self.get_fundamentals_as_of(df, symbol, entry_date, delay_days)

            if fundamentals is not None:
                result = {
                    '_trade_idx': idx,
                    '_fundamental_date': fundamentals['_as_of_date'],
                    '_days_before_entry': (entry_date - fundamentals['_as_of_date']).days
                }
                # Add all fundamental columns
                for col in fundamentals.index:
                    if col not in ['_as_of_date', 'date', 'symbol']:
                        result[f'fund_{col}'] = fundamentals[col]
                results.append(result)
            else:
                results.append({
                    '_trade_idx': idx,
                    '_fundamental_date': None,
                    '_days_before_entry': None
                })

        return pd.DataFrame(results)

    def compute_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived fundamental metrics.

        Args:
            df: Fundamental data DataFrame

        Returns:
            DataFrame with additional derived metrics
        """
        df = df.copy()

        # Debt-to-equity ratio
        if 'totalliabilities' in df.columns and 'totalshareholderequity' in df.columns:
            df['debt_to_equity'] = df['totalliabilities'] / df['totalshareholderequity'].replace(0, float('nan'))

        # Free cash flow
        if 'operatingcashflow' in df.columns and 'capitalexpenditures' in df.columns:
            df['freecashflow'] = df['operatingcashflow'] - df['capitalexpenditures'].abs()

        # Gross margin
        if 'gross_profit_ttm' in df.columns and 'revenue_ttm' in df.columns:
            df['gross_margin'] = df['gross_profit_ttm'] / df['revenue_ttm'].replace(0, float('nan'))

        # FCF yield (needs price data - placeholder)
        # Would need market cap or share price to compute

        return df

    def get_metric_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get coverage statistics for each metric.

        Args:
            df: Fundamental data DataFrame

        Returns:
            Dictionary of metric name to coverage percentage
        """
        coverage = {}
        all_metrics = (
            self.VALUATION_METRICS + self.PROFITABILITY_METRICS +
            self.GROWTH_METRICS + self.QUALITY_METRICS +
            self.CASH_FLOW_METRICS + self.OTHER_METRICS
        )

        total_rows = len(df)
        for metric in all_metrics:
            if metric in df.columns:
                non_null = df[metric].notna().sum()
                coverage[metric] = non_null / total_rows if total_rows > 0 else 0.0

        return coverage
