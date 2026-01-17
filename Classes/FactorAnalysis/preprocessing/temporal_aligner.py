"""
Temporal Aligner for Factor Analysis.

Ensures proper as-of alignment of all data to prevent forward-looking bias.
This is a critical component for maintaining data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..config.factor_config import DataAlignmentConfig
from ..logging.audit_logger import AuditLogger


@dataclass
class AlignmentResult:
    """Result of temporal alignment for a data source."""
    source_name: str
    trades_matched: int
    trades_missing: int
    coverage_pct: float
    avg_days_before_entry: float
    max_days_before_entry: int
    method_used: str
    issues: List[str] = field(default_factory=list)


@dataclass
class AlignmentSummary:
    """Summary of all temporal alignments."""
    total_trades: int
    price_data: Optional[AlignmentResult] = None
    fundamental_data: Optional[AlignmentResult] = None
    insider_data: Optional[AlignmentResult] = None
    options_data: Optional[AlignmentResult] = None
    fully_aligned_trades: int = 0
    partially_aligned_trades: int = 0
    unaligned_trades: int = 0


class TemporalAligner:
    """
    Aligns factor data to trade entry dates with proper temporal constraints.

    Critical Design Principle:
    - NO data that was not publicly available at trade entry time may be used
    - All lookups use as-of date logic with configurable delays
    - Every alignment decision is logged for audit trail
    """

    def __init__(
        self,
        config: Optional[DataAlignmentConfig] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize TemporalAligner.

        Args:
            config: Alignment configuration with delays
            logger: Optional audit logger
        """
        self.config = config or DataAlignmentConfig()
        self.logger = logger

    def _log_alignment(self, source: str, trade_id: Any, entry_date: Any,
                       data_date: Any, success: bool, message: str = "") -> None:
        """Log individual alignment decision."""
        if self.logger:
            if success:
                self.logger.debug(f"Aligned {source} for trade {trade_id}", {
                    'entry_date': str(entry_date),
                    'data_date': str(data_date),
                    'days_before': (entry_date - data_date).days if data_date else None
                })
            else:
                self.logger.warning(f"Missing {source} for trade {trade_id}", {
                    'entry_date': str(entry_date),
                    'message': message
                })

    def align_price_data(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        forward_fill: bool = True
    ) -> Tuple[pd.DataFrame, AlignmentResult]:
        """
        Align price/indicator data to trade entry dates.

        For each trade, get the price data from entry_date.
        If missing, optionally forward-fill from last available date.

        Args:
            trades_df: Trade log DataFrame
            price_df: Price data DataFrame with 'date' column
            forward_fill: Whether to use last available price for missing dates

        Returns:
            Tuple of (aligned DataFrame, AlignmentResult)
        """
        result_rows = []
        matched = 0
        missing = 0
        days_before_list = []
        issues = []

        # Ensure date columns are datetime
        trades_df = trades_df.copy()
        price_df = price_df.copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        price_df['date'] = pd.to_datetime(price_df['date'])

        # Sort price data by date
        price_df = price_df.sort_values('date')

        for idx, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            symbol = trade.get('symbol', '')

            # Filter price data for symbol if available
            if 'symbol' in price_df.columns and symbol:
                symbol_prices = price_df[price_df['symbol'].str.upper() == symbol.upper()]
            else:
                symbol_prices = price_df

            # Exact match
            exact = symbol_prices[symbol_prices['date'] == entry_date]

            if len(exact) > 0:
                row = exact.iloc[0].to_dict()
                row['_price_date'] = entry_date
                row['_price_days_before_entry'] = 0
                row['_price_forward_filled'] = False
                result_rows.append(row)
                matched += 1
                days_before_list.append(0)
            elif forward_fill and self.config.price_forward_fill_allowed:
                # Get last available before entry date
                before = symbol_prices[symbol_prices['date'] < entry_date]
                if len(before) > 0:
                    row = before.iloc[-1].to_dict()
                    actual_date = before.iloc[-1]['date']
                    days_before = (entry_date - actual_date).days
                    row['_price_date'] = actual_date
                    row['_price_days_before_entry'] = days_before
                    row['_price_forward_filled'] = True
                    result_rows.append(row)
                    matched += 1
                    days_before_list.append(days_before)

                    if self.config.flag_price_gaps:
                        issues.append(f"Trade {trade.get('trade_id', idx)}: Price gap of {days_before} days")
                else:
                    result_rows.append({
                        '_price_date': None,
                        '_price_days_before_entry': None,
                        '_price_forward_filled': None
                    })
                    missing += 1
                    self._log_alignment('price', trade.get('trade_id', idx),
                                       entry_date, None, False, "No price data before entry")
            else:
                result_rows.append({
                    '_price_date': None,
                    '_price_days_before_entry': None,
                    '_price_forward_filled': None
                })
                missing += 1

        aligned_df = pd.DataFrame(result_rows)

        total = matched + missing
        alignment_result = AlignmentResult(
            source_name='price_data',
            trades_matched=matched,
            trades_missing=missing,
            coverage_pct=matched / total * 100 if total > 0 else 0,
            avg_days_before_entry=np.mean(days_before_list) if days_before_list else 0,
            max_days_before_entry=max(days_before_list) if days_before_list else 0,
            method_used=f"exact match + forward_fill={forward_fill}",
            issues=issues[:10]  # Limit issues logged
        )

        if self.logger:
            self.logger.log_temporal_alignment(
                source_name='Price data',
                trades_matched=matched,
                trades_missing=missing,
                method=alignment_result.method_used
            )

        return aligned_df, alignment_result

    def align_fundamental_data(
        self,
        trades_df: pd.DataFrame,
        fundamental_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, AlignmentResult]:
        """
        Align fundamental data to trade entry dates using report_date.

        Uses the most recent fundamental record where:
        report_date <= (entry_date - fundamentals_reporting_delay_days)

        Args:
            trades_df: Trade log DataFrame
            fundamental_df: Fundamental data with '_as_of_date' column

        Returns:
            Tuple of (aligned DataFrame, AlignmentResult)
        """
        result_rows = []
        matched = 0
        missing = 0
        days_before_list = []
        delay = self.config.fundamentals_reporting_delay_days

        trades_df = trades_df.copy()
        fundamental_df = fundamental_df.copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])

        # Ensure _as_of_date exists
        if '_as_of_date' not in fundamental_df.columns:
            if 'report_date' in fundamental_df.columns:
                fundamental_df['_as_of_date'] = pd.to_datetime(fundamental_df['report_date'])
            elif 'date' in fundamental_df.columns:
                fundamental_df['_as_of_date'] = pd.to_datetime(fundamental_df['date'])
            else:
                raise ValueError("Fundamental data must have '_as_of_date', 'report_date', or 'date' column")

        fundamental_df = fundamental_df.sort_values('_as_of_date')

        for idx, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            effective_date = entry_date - pd.Timedelta(days=delay)
            symbol = trade.get('symbol', '')

            # Filter for symbol
            if 'symbol' in fundamental_df.columns and symbol:
                symbol_data = fundamental_df[fundamental_df['symbol'].str.upper() == symbol.upper()]
            else:
                symbol_data = fundamental_df

            # Get most recent before effective date
            available = symbol_data[symbol_data['_as_of_date'] <= effective_date]

            if len(available) > 0:
                row = available.iloc[-1].to_dict()
                data_date = available.iloc[-1]['_as_of_date']
                days_before = (entry_date - data_date).days
                row['_fundamental_date'] = data_date
                row['_fundamental_days_before_entry'] = days_before
                result_rows.append(row)
                matched += 1
                days_before_list.append(days_before)
            else:
                result_rows.append({
                    '_fundamental_date': None,
                    '_fundamental_days_before_entry': None
                })
                missing += 1

        aligned_df = pd.DataFrame(result_rows)

        total = matched + missing
        alignment_result = AlignmentResult(
            source_name='fundamental_data',
            trades_matched=matched,
            trades_missing=missing,
            coverage_pct=matched / total * 100 if total > 0 else 0,
            avg_days_before_entry=np.mean(days_before_list) if days_before_list else 0,
            max_days_before_entry=max(days_before_list) if days_before_list else 0,
            method_used=f"as-of lookup with {delay}-day delay"
        )

        if self.logger:
            self.logger.log_temporal_alignment(
                source_name='Fundamental data',
                trades_matched=matched,
                trades_missing=missing,
                method=alignment_result.method_used
            )

        return aligned_df, alignment_result

    # Alias for backward compatibility
    def align_fundamentals(
        self,
        trades_df: pd.DataFrame,
        fundamental_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Alias for align_fundamental_data that returns just the DataFrame.

        Args:
            trades_df: Trade log DataFrame
            fundamental_df: Fundamental data with '_as_of_date' column

        Returns:
            Aligned DataFrame
        """
        aligned_df, _ = self.align_fundamental_data(trades_df, fundamental_df)
        return aligned_df

    def align_insider_data(
        self,
        trades_df: pd.DataFrame,
        insider_df: pd.DataFrame,
        window_days: int = 30
    ) -> Tuple[pd.DataFrame, AlignmentResult]:
        """
        Align insider activity data using aggregation window.

        Aggregates insider transactions in the window:
        (entry_date - window_days) to (entry_date - insiders_reporting_delay_days)

        Args:
            trades_df: Trade log DataFrame
            insider_df: Insider data with '_available_date' column
            window_days: Aggregation window in days

        Returns:
            Tuple of (aligned DataFrame, AlignmentResult)
        """
        result_rows = []
        matched = 0
        missing = 0
        delay = self.config.insiders_reporting_delay_days

        trades_df = trades_df.copy()
        insider_df = insider_df.copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])

        if '_available_date' not in insider_df.columns:
            if 'date' in insider_df.columns:
                insider_df['_available_date'] = pd.to_datetime(insider_df['date']) + pd.Timedelta(days=delay)
            else:
                raise ValueError("Insider data must have '_available_date' or 'date' column")

        for idx, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            window_end = entry_date - pd.Timedelta(days=delay)
            window_start = window_end - pd.Timedelta(days=window_days)
            symbol = trade.get('symbol', '')

            # Filter for symbol and window
            if 'symbol' in insider_df.columns and symbol:
                symbol_data = insider_df[insider_df['symbol'].str.upper() == symbol.upper()]
            else:
                symbol_data = insider_df

            window_data = symbol_data[
                (symbol_data['_available_date'] >= window_start) &
                (symbol_data['_available_date'] <= window_end)
            ]

            row = {
                '_insider_window_start': window_start,
                '_insider_window_end': window_end,
                '_insider_transactions_found': len(window_data)
            }

            if len(window_data) > 0:
                matched += 1
            else:
                # No insider activity is valid (count as 0)
                matched += 1  # Still counts as "aligned" - just zero activity

            result_rows.append(row)

        aligned_df = pd.DataFrame(result_rows)

        total = len(trades_df)
        alignment_result = AlignmentResult(
            source_name='insider_data',
            trades_matched=total,  # All trades get insider alignment (may be 0 activity)
            trades_missing=0,
            coverage_pct=100.0,
            avg_days_before_entry=window_days / 2,  # Approximate
            max_days_before_entry=window_days,
            method_used=f"{window_days}-day window with {delay}-day delay"
        )

        if self.logger:
            trades_with_activity = (aligned_df['_insider_transactions_found'] > 0).sum()
            self.logger.log_temporal_alignment(
                source_name='Insider data',
                trades_matched=int(trades_with_activity),
                trades_missing=total - int(trades_with_activity),
                method=alignment_result.method_used
            )

        return aligned_df, alignment_result

    def align_options_data(
        self,
        trades_df: pd.DataFrame,
        options_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, AlignmentResult]:
        """
        Align options snapshot data to trade entry dates.

        Uses the most recent snapshot where:
        snapshot_date <= entry_date AND
        (entry_date - snapshot_date) <= options_lookback_days

        Args:
            trades_df: Trade log DataFrame
            options_df: Options data with 'snapshot_date' column

        Returns:
            Tuple of (aligned DataFrame, AlignmentResult)
        """
        result_rows = []
        matched = 0
        missing = 0
        days_before_list = []
        lookback = self.config.options_lookback_days

        trades_df = trades_df.copy()
        options_df = options_df.copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        options_df['snapshot_date'] = pd.to_datetime(options_df['snapshot_date'])

        for idx, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            min_date = entry_date - pd.Timedelta(days=lookback)
            symbol = trade.get('symbol', '')

            # Filter for symbol
            if 'symbol' in options_df.columns and symbol:
                symbol_data = options_df[options_df['symbol'].str.upper() == symbol.upper()]
            else:
                symbol_data = options_df

            # Get snapshots within lookback window
            available = symbol_data[
                (symbol_data['snapshot_date'] <= entry_date) &
                (symbol_data['snapshot_date'] >= min_date)
            ]

            if len(available) > 0:
                # Get most recent snapshot date
                latest_date = available['snapshot_date'].max()
                days_before = (entry_date - latest_date).days
                row = {
                    '_options_snapshot_date': latest_date,
                    '_options_days_before_entry': days_before,
                    '_options_contracts_in_snapshot': len(available[available['snapshot_date'] == latest_date])
                }
                result_rows.append(row)
                matched += 1
                days_before_list.append(days_before)
            else:
                result_rows.append({
                    '_options_snapshot_date': None,
                    '_options_days_before_entry': None,
                    '_options_contracts_in_snapshot': 0
                })
                missing += 1

        aligned_df = pd.DataFrame(result_rows)

        total = matched + missing
        alignment_result = AlignmentResult(
            source_name='options_data',
            trades_matched=matched,
            trades_missing=missing,
            coverage_pct=matched / total * 100 if total > 0 else 0,
            avg_days_before_entry=np.mean(days_before_list) if days_before_list else 0,
            max_days_before_entry=max(days_before_list) if days_before_list else 0,
            method_used=f"{lookback}-day lookback window"
        )

        if self.logger:
            self.logger.log_temporal_alignment(
                source_name='Options data',
                trades_matched=matched,
                trades_missing=missing,
                method=alignment_result.method_used
            )

        return aligned_df, alignment_result

    def create_alignment_summary(
        self,
        trades_df: pd.DataFrame,
        price_result: Optional[AlignmentResult] = None,
        fundamental_result: Optional[AlignmentResult] = None,
        insider_result: Optional[AlignmentResult] = None,
        options_result: Optional[AlignmentResult] = None
    ) -> AlignmentSummary:
        """
        Create summary of all temporal alignments.

        Args:
            trades_df: Trade log DataFrame
            price_result: Price alignment result
            fundamental_result: Fundamental alignment result
            insider_result: Insider alignment result
            options_result: Options alignment result

        Returns:
            AlignmentSummary with overall statistics
        """
        total = len(trades_df)

        # Count trades with different levels of alignment
        # Full alignment = price + fundamental + at least one of insider/options
        # Partial = some but not all required
        # None = missing critical data

        summary = AlignmentSummary(
            total_trades=total,
            price_data=price_result,
            fundamental_data=fundamental_result,
            insider_data=insider_result,
            options_data=options_result
        )

        # Estimate alignment levels
        if price_result and fundamental_result:
            both_matched = min(price_result.trades_matched, fundamental_result.trades_matched)
            summary.fully_aligned_trades = both_matched
            summary.partially_aligned_trades = max(
                price_result.trades_matched,
                fundamental_result.trades_matched
            ) - both_matched
            summary.unaligned_trades = total - summary.fully_aligned_trades - summary.partially_aligned_trades
        elif price_result:
            summary.fully_aligned_trades = price_result.trades_matched
            summary.unaligned_trades = price_result.trades_missing

        if self.logger:
            self.logger.info("Temporal alignment summary", {
                'total_trades': total,
                'fully_aligned': summary.fully_aligned_trades,
                'partially_aligned': summary.partially_aligned_trades,
                'unaligned': summary.unaligned_trades
            })

        return summary
