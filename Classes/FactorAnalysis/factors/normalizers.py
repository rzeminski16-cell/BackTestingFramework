"""
Factor Normalizers for Factor Analysis.

Provides normalization methods for factor values including z-score and percentile rank.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..config.factor_config import NormalizationType
from ..logging.audit_logger import AuditLogger


@dataclass
class NormalizationResult:
    """Result of factor normalization."""
    factors_normalized: int
    normalization_type: str
    factors_skipped: int
    skipped_reasons: Dict[str, str]


class FactorNormalizer:
    """
    Normalizes factor values for comparability.

    Normalization Methods:
    - Z-Score: (x - mean) / std
    - Percentile Rank: rank / n * 100
    - Min-Max: (x - min) / (max - min)
    - Robust: (x - median) / IQR
    """

    def __init__(
        self,
        default_method: NormalizationType = NormalizationType.ZSCORE,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize FactorNormalizer.

        Args:
            default_method: Default normalization method
            logger: Optional audit logger
        """
        self.default_method = default_method
        self.logger = logger
        self._stats: Dict[str, Dict] = {}

    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Optional[NormalizationType] = None,
        exclude_columns: Optional[List[str]] = None,
        suffix: str = '_norm'
    ) -> Tuple[pd.DataFrame, NormalizationResult]:
        """
        Normalize factor columns.

        Args:
            df: DataFrame with factor columns
            columns: Specific columns to normalize (None = all numeric)
            method: Normalization method to use
            exclude_columns: Columns to exclude from normalization
            suffix: Suffix for normalized column names

        Returns:
            Tuple of (DataFrame with normalized columns, NormalizationResult)
        """
        method = method or self.default_method
        result_df = df.copy()

        # Determine columns to normalize
        if columns is None:
            columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        # Apply exclusions
        exclude = set(exclude_columns or [])
        exclude.update(['_trade_idx', 'trade_id', 'trade_class_numeric'])
        columns = [c for c in columns if c not in exclude and not c.startswith('_')]

        normalized_count = 0
        skipped = {}

        for col in columns:
            try:
                values = df[col].dropna()

                if len(values) < 2:
                    skipped[col] = "Insufficient data points"
                    continue

                if method == NormalizationType.ZSCORE:
                    normalized = self._zscore_normalize(df[col])
                elif method == NormalizationType.PERCENTILE_RANK:
                    normalized = self._percentile_normalize(df[col])
                else:  # NONE
                    normalized = df[col].copy()

                result_df[f'{col}{suffix}'] = normalized

                # Store stats for later reference
                self._stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'method': method.value
                }

                normalized_count += 1

            except Exception as e:
                skipped[col] = str(e)

        result = NormalizationResult(
            factors_normalized=normalized_count,
            normalization_type=method.value,
            factors_skipped=len(skipped),
            skipped_reasons=skipped
        )

        if self.logger:
            self.logger.info("Factor normalization complete", {
                'normalized': normalized_count,
                'skipped': len(skipped),
                'method': method.value
            })

        return result_df, result

    def _zscore_normalize(
        self,
        series: pd.Series,
        min_std: float = 1e-10
    ) -> pd.Series:
        """
        Z-score normalization: (x - mean) / std

        Args:
            series: Series to normalize
            min_std: Minimum std to avoid division by zero

        Returns:
            Normalized series
        """
        mean = series.mean()
        std = max(series.std(), min_std)
        return (series - mean) / std

    def _percentile_normalize(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Percentile rank normalization: rank / n * 100

        Args:
            series: Series to normalize

        Returns:
            Normalized series (0-100)
        """
        return series.rank(pct=True) * 100

    def _minmax_normalize(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Min-max normalization: (x - min) / (max - min)

        Args:
            series: Series to normalize

        Returns:
            Normalized series (0-1)
        """
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val

        if range_val == 0:
            return pd.Series(0.5, index=series.index)

        return (series - min_val) / range_val

    def _robust_normalize(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Robust normalization: (x - median) / IQR

        More robust to outliers than z-score.

        Args:
            series: Series to normalize

        Returns:
            Normalized series
        """
        median = series.median()
        q75, q25 = series.quantile([0.75, 0.25])
        iqr = q75 - q25

        if iqr == 0:
            return series - median

        return (series - median) / iqr

    def normalize_by_category(
        self,
        df: pd.DataFrame,
        category_methods: Dict[str, NormalizationType]
    ) -> Tuple[pd.DataFrame, Dict[str, NormalizationResult]]:
        """
        Normalize factors by category with different methods.

        Args:
            df: DataFrame with factor columns
            category_methods: Dict mapping column prefix to normalization method
                             e.g., {'value_': ZSCORE, 'insider_': PERCENTILE_RANK}

        Returns:
            Tuple of (normalized DataFrame, dict of results by category)
        """
        result_df = df.copy()
        results = {}

        for prefix, method in category_methods.items():
            cols = [c for c in df.columns if c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c])]

            if cols:
                partial_df, result = self.normalize(
                    df[cols],
                    method=method,
                    suffix='_norm'
                )

                # Add normalized columns to result
                for col in partial_df.columns:
                    if col.endswith('_norm'):
                        result_df[col] = partial_df[col]

                results[prefix] = result

        return result_df, results

    def inverse_normalize(
        self,
        normalized_values: pd.Series,
        column_name: str
    ) -> pd.Series:
        """
        Convert normalized values back to original scale.

        Args:
            normalized_values: Normalized values
            column_name: Original column name (to look up stats)

        Returns:
            Values in original scale
        """
        if column_name not in self._stats:
            raise ValueError(f"No stats stored for column {column_name}")

        stats = self._stats[column_name]

        if stats['method'] == 'zscore':
            return normalized_values * stats['std'] + stats['mean']
        else:
            raise ValueError(f"Inverse not supported for method {stats['method']}")

    def get_normalization_stats(self) -> Dict[str, Dict]:
        """Get stored normalization statistics."""
        return self._stats.copy()

    def create_normalized_copy(
        self,
        df: pd.DataFrame,
        method: NormalizationType = NormalizationType.ZSCORE
    ) -> pd.DataFrame:
        """
        Create a copy with all numeric columns normalized (in place).

        Args:
            df: DataFrame with factor columns
            method: Normalization method

        Returns:
            DataFrame with normalized values (same column names)
        """
        result_df = df.copy()

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and not col.startswith('_'):
                if df[col].std() > 0:
                    if method == NormalizationType.ZSCORE:
                        result_df[col] = self._zscore_normalize(df[col])
                    elif method == NormalizationType.PERCENTILE_RANK:
                        result_df[col] = self._percentile_normalize(df[col])

        return result_df
