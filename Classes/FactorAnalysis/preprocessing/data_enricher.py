"""
Data Enricher for Factor Analysis.

Merges aligned factor data into trade records, creating the enriched dataset
for factor analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..logging.audit_logger import AuditLogger


@dataclass
class EnrichmentResult:
    """Result of data enrichment."""
    total_trades: int
    factor_columns_added: int
    trades_with_all_factors: int
    trades_missing_factors: int
    factor_coverage: Dict[str, float]


class DataEnricher:
    """
    Merges aligned factor data into trade records.

    Combines:
    - Trade classification
    - Price/technical indicator data
    - Fundamental metrics
    - Insider activity metrics
    - Options metrics
    - Computed regime metrics

    Creates a single enriched DataFrame ready for factor analysis.
    """

    def __init__(self, logger: Optional[AuditLogger] = None):
        """
        Initialize DataEnricher.

        Args:
            logger: Optional audit logger
        """
        self.logger = logger

    def enrich_trades(
        self,
        trades_df: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
        fundamental_data: Optional[pd.DataFrame] = None,
        insider_data: Optional[pd.DataFrame] = None,
        options_data: Optional[pd.DataFrame] = None,
        exclude_internal_columns: bool = True
    ) -> tuple[pd.DataFrame, EnrichmentResult]:
        """
        Merge all factor data into trade records.

        Args:
            trades_df: Trade log DataFrame (already classified)
            price_data: Aligned price/indicator DataFrame (indexed same as trades)
            fundamental_data: Aligned fundamental DataFrame
            insider_data: Aligned insider DataFrame
            options_data: Aligned options DataFrame
            exclude_internal_columns: Whether to exclude internal columns (starting with '_')

        Returns:
            Tuple of (enriched DataFrame, EnrichmentResult)
        """
        if self.logger:
            self.logger.start_section("DATA_ENRICHMENT")

        enriched = trades_df.copy()
        initial_cols = len(enriched.columns)
        factor_coverage = {}

        # Merge price/indicator data
        if price_data is not None and len(price_data) > 0:
            enriched = self._merge_data(
                enriched, price_data, prefix='price_', exclude_cols=['symbol', 'date']
            )
            price_cols = [c for c in enriched.columns if c.startswith('price_') and not c.startswith('_')]
            factor_coverage['price'] = self._calculate_coverage(enriched, price_cols)
            if self.logger:
                self.logger.info(f"Merged price data", {
                    'columns_added': len(price_cols),
                    'coverage': f"{factor_coverage['price']:.1%}"
                })

        # Merge fundamental data
        if fundamental_data is not None and len(fundamental_data) > 0:
            enriched = self._merge_data(
                enriched, fundamental_data, prefix='fund_', exclude_cols=['symbol', 'date']
            )
            fund_cols = [c for c in enriched.columns if c.startswith('fund_') and not c.startswith('_')]
            factor_coverage['fundamental'] = self._calculate_coverage(enriched, fund_cols)
            if self.logger:
                self.logger.info(f"Merged fundamental data", {
                    'columns_added': len(fund_cols),
                    'coverage': f"{factor_coverage['fundamental']:.1%}"
                })

        # Merge insider data
        if insider_data is not None and len(insider_data) > 0:
            enriched = self._merge_data(
                enriched, insider_data, prefix='', exclude_cols=['symbol']
            )
            insider_cols = [c for c in enriched.columns if c.startswith('insider_')]
            factor_coverage['insider'] = self._calculate_coverage(enriched, insider_cols)
            if self.logger:
                self.logger.info(f"Merged insider data", {
                    'columns_added': len(insider_cols),
                    'coverage': f"{factor_coverage['insider']:.1%}"
                })

        # Merge options data
        if options_data is not None and len(options_data) > 0:
            enriched = self._merge_data(
                enriched, options_data, prefix='', exclude_cols=['symbol']
            )
            options_cols = [c for c in enriched.columns if c.startswith('options_')]
            factor_coverage['options'] = self._calculate_coverage(enriched, options_cols)
            if self.logger:
                self.logger.info(f"Merged options data", {
                    'columns_added': len(options_cols),
                    'coverage': f"{factor_coverage['options']:.1%}"
                })

        # Optionally remove internal columns
        if exclude_internal_columns:
            internal_cols = [c for c in enriched.columns if c.startswith('_') and c != '_log_id']
            enriched = enriched.drop(columns=internal_cols, errors='ignore')

        # Calculate result statistics
        factor_cols = [c for c in enriched.columns if not c.startswith('_')
                      and c not in trades_df.columns]
        trades_with_all = self._count_complete_rows(enriched, factor_cols)

        result = EnrichmentResult(
            total_trades=len(enriched),
            factor_columns_added=len(enriched.columns) - initial_cols,
            trades_with_all_factors=trades_with_all,
            trades_missing_factors=len(enriched) - trades_with_all,
            factor_coverage=factor_coverage
        )

        if self.logger:
            self.logger.info("Enrichment complete", {
                'total_trades': result.total_trades,
                'columns_added': result.factor_columns_added,
                'complete_trades': result.trades_with_all_factors
            })
            self.logger.end_section()

        return enriched, result

    def _merge_data(
        self,
        base_df: pd.DataFrame,
        data_df: pd.DataFrame,
        prefix: str = '',
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Merge data into base DataFrame by index.

        Args:
            base_df: Base DataFrame
            data_df: Data to merge
            prefix: Prefix to add to merged columns
            exclude_cols: Columns to exclude from merge

        Returns:
            Merged DataFrame
        """
        exclude = set(exclude_cols or [])
        exclude.add('_trade_idx')  # Internal index column

        # Get columns to merge
        merge_cols = [c for c in data_df.columns if c not in exclude]

        # Reset index if needed for proper alignment
        if len(data_df) == len(base_df):
            # Same length - assume aligned by position
            data_subset = data_df[merge_cols].reset_index(drop=True)
            base_df = base_df.reset_index(drop=True)

            # Add prefix to columns
            if prefix:
                rename_map = {c: f"{prefix}{c}" if not c.startswith('_') else c
                             for c in merge_cols}
                data_subset = data_subset.rename(columns=rename_map)

            return pd.concat([base_df, data_subset], axis=1)
        else:
            # Different lengths - use _trade_idx if available
            if '_trade_idx' in data_df.columns:
                data_subset = data_df.set_index('_trade_idx')[merge_cols]
                if prefix:
                    rename_map = {c: f"{prefix}{c}" if not c.startswith('_') else c
                                 for c in merge_cols}
                    data_subset = data_subset.rename(columns=rename_map)
                return base_df.join(data_subset, how='left')
            else:
                # Can't align - just return base
                return base_df

    def _calculate_coverage(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> float:
        """Calculate coverage (non-null ratio) for columns."""
        if not columns:
            return 0.0

        available_cols = [c for c in columns if c in df.columns]
        if not available_cols:
            return 0.0

        non_null_counts = df[available_cols].notna().sum(axis=1)
        # Trade has data if at least one factor is non-null
        trades_with_data = (non_null_counts > 0).sum()
        return trades_with_data / len(df) if len(df) > 0 else 0.0

    def _count_complete_rows(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> int:
        """Count rows with all specified columns non-null."""
        if not columns:
            return len(df)

        available_cols = [c for c in columns if c in df.columns]
        if not available_cols:
            return len(df)

        return df[available_cols].notna().all(axis=1).sum()

    def get_factor_summary(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all factor columns.

        Args:
            enriched_df: Enriched trade DataFrame

        Returns:
            DataFrame with factor statistics
        """
        # Identify factor columns (exclude trade metadata)
        metadata_cols = [
            'trade_id', 'symbol', 'entry_date', 'exit_date', 'entry_price',
            'exit_price', 'pl', 'pl_pct', 'quantity', 'side', 'duration_days',
            'trade_class', 'trade_class_numeric', '_log_id'
        ]

        factor_cols = [c for c in enriched_df.columns
                      if c not in metadata_cols and not c.startswith('_')]

        stats = []
        for col in factor_cols:
            if pd.api.types.is_numeric_dtype(enriched_df[col]):
                stats.append({
                    'factor': col,
                    'type': 'numeric',
                    'count': enriched_df[col].notna().sum(),
                    'missing': enriched_df[col].isna().sum(),
                    'coverage': f"{enriched_df[col].notna().mean():.1%}",
                    'mean': enriched_df[col].mean(),
                    'std': enriched_df[col].std(),
                    'min': enriched_df[col].min(),
                    'max': enriched_df[col].max()
                })
            else:
                stats.append({
                    'factor': col,
                    'type': 'categorical',
                    'count': enriched_df[col].notna().sum(),
                    'missing': enriched_df[col].isna().sum(),
                    'coverage': f"{enriched_df[col].notna().mean():.1%}",
                    'unique': enriched_df[col].nunique(),
                    'top_value': enriched_df[col].mode().iloc[0] if len(enriched_df[col].dropna()) > 0 else None
                })

        return pd.DataFrame(stats)

    def filter_by_coverage(
        self,
        enriched_df: pd.DataFrame,
        min_coverage: float = 0.5,
        factor_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filter to trades with minimum factor coverage.

        Args:
            enriched_df: Enriched trade DataFrame
            min_coverage: Minimum proportion of factors that must be non-null
            factor_categories: Categories to consider (e.g., ['price', 'fund', 'insider'])

        Returns:
            Filtered DataFrame
        """
        if factor_categories is None:
            factor_categories = ['price', 'fund', 'insider', 'options']

        factor_cols = []
        for cat in factor_categories:
            factor_cols.extend([c for c in enriched_df.columns
                              if c.startswith(f'{cat}_') or c.startswith(cat)])

        if not factor_cols:
            return enriched_df

        # Calculate coverage per trade
        coverage = enriched_df[factor_cols].notna().mean(axis=1)

        return enriched_df[coverage >= min_coverage].copy()
