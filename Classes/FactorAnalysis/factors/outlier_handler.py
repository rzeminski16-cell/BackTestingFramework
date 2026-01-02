"""
Outlier Handler for Factor Analysis.

Detects and handles outliers in factor values using configurable methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..config.factor_config import OutlierHandlingConfig, OutlierMethod
from ..logging.audit_logger import AuditLogger


@dataclass
class OutlierInfo:
    """Information about a detected outlier."""
    column: str
    row_index: int
    value: float
    zscore: float
    action_taken: str


@dataclass
class OutlierResult:
    """Result of outlier detection and handling."""
    total_outliers: int
    outliers_by_column: Dict[str, int]
    trades_with_outliers: int
    total_trades: int
    method_used: str
    threshold: float
    outlier_details: List[OutlierInfo] = field(default_factory=list)


class OutlierHandler:
    """
    Detects and handles outliers in factor values.

    Detection Methods:
    - Z-Score: Values > threshold standard deviations from mean
    - IQR: Values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
    - Percentile: Values outside specified percentile range

    Handling Methods:
    - Flag and report: Mark outliers but keep values
    - Winsorize: Cap at specified percentile
    - Exclude: Set to NaN
    """

    def __init__(
        self,
        config: Optional[OutlierHandlingConfig] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize OutlierHandler.

        Args:
            config: Outlier handling configuration
            logger: Optional audit logger
        """
        self.config = config or OutlierHandlingConfig()
        self.logger = logger

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, OutlierResult]:
        """
        Detect outliers in factor columns.

        Args:
            df: DataFrame with factor columns
            columns: Specific columns to check (None = all numeric)
            threshold: Z-score threshold (default from config)

        Returns:
            Tuple of (DataFrame with outlier flags, OutlierResult)
        """
        if not self.config.enabled:
            return df, OutlierResult(
                total_outliers=0,
                outliers_by_column={},
                trades_with_outliers=0,
                total_trades=len(df),
                method_used='disabled',
                threshold=0
            )

        threshold = threshold or self.config.threshold_zscore
        result_df = df.copy()

        # Determine columns to check
        if columns is None:
            columns = [c for c in df.columns
                      if pd.api.types.is_numeric_dtype(df[c])
                      and not c.startswith('_')]

        outliers_by_column = {}
        outlier_details = []
        trades_with_outliers = set()

        for col in columns:
            values = df[col].dropna()

            if len(values) < 3:
                continue

            # Calculate z-scores
            mean = values.mean()
            std = values.std()

            if std == 0:
                continue

            zscores = (df[col] - mean) / std
            outlier_mask = zscores.abs() > threshold

            # Create outlier flag column
            result_df[f'{col}_outlier'] = outlier_mask
            result_df[f'{col}_zscore'] = zscores

            # Count and log outliers
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                outliers_by_column[col] = int(n_outliers)

                # Get outlier indices
                outlier_indices = df.index[outlier_mask].tolist()
                trades_with_outliers.update(outlier_indices)

                # Record details (limit to first 10 per column)
                for idx in outlier_indices[:10]:
                    outlier_details.append(OutlierInfo(
                        column=col,
                        row_index=idx,
                        value=float(df.loc[idx, col]),
                        zscore=float(zscores.loc[idx]),
                        action_taken='flagged'
                    ))

                if self.logger:
                    self.logger.log_outliers(
                        factor_name=col,
                        outlier_count=n_outliers,
                        total_values=len(values),
                        threshold=threshold,
                        action='flagged'
                    )

        # Create overall outlier count per trade
        outlier_flag_cols = [c for c in result_df.columns if c.endswith('_outlier')]
        if outlier_flag_cols:
            result_df['_outlier_count'] = result_df[outlier_flag_cols].sum(axis=1)

        result = OutlierResult(
            total_outliers=sum(outliers_by_column.values()),
            outliers_by_column=outliers_by_column,
            trades_with_outliers=len(trades_with_outliers),
            total_trades=len(df),
            method_used='zscore',
            threshold=threshold,
            outlier_details=outlier_details
        )

        return result_df, result

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Optional[OutlierMethod] = None
    ) -> Tuple[pd.DataFrame, OutlierResult]:
        """
        Detect and handle outliers according to configuration.

        Args:
            df: DataFrame with factor columns
            columns: Specific columns to handle
            method: Handling method (default from config)

        Returns:
            Tuple of (handled DataFrame, OutlierResult)
        """
        method = method or self.config.method

        if self.logger:
            self.logger.start_section("OUTLIER_HANDLING")

        # First detect outliers
        flagged_df, detection_result = self.detect_outliers(df, columns)

        # Apply handling method
        if method == OutlierMethod.FLAG_AND_REPORT:
            result_df = flagged_df
            # Update action in details
            for detail in detection_result.outlier_details:
                detail.action_taken = 'flagged_only'

        elif method == OutlierMethod.WINSORIZE:
            result_df = self._winsorize(flagged_df, columns)
            for detail in detection_result.outlier_details:
                detail.action_taken = 'winsorized'

        elif method == OutlierMethod.EXCLUDE:
            result_df = self._exclude(flagged_df, columns)
            for detail in detection_result.outlier_details:
                detail.action_taken = 'excluded'

        else:
            result_df = flagged_df

        # Update method in result
        detection_result.method_used = method.value

        if self.logger:
            self.logger.info("Outlier handling complete", {
                'method': method.value,
                'total_outliers': detection_result.total_outliers,
                'trades_affected': detection_result.trades_with_outliers
            })
            self.logger.end_section()

        return result_df, detection_result

    def _winsorize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Winsorize outliers by capping at percentile.

        Args:
            df: DataFrame with outlier flags
            columns: Columns to winsorize

        Returns:
            Winsorized DataFrame
        """
        result_df = df.copy()
        percentile = self.config.winsorize_percentile

        if columns is None:
            columns = [c for c in df.columns
                      if pd.api.types.is_numeric_dtype(df[c])
                      and not c.startswith('_')
                      and not c.endswith('_outlier')
                      and not c.endswith('_zscore')]

        for col in columns:
            values = df[col].dropna()
            if len(values) < 3:
                continue

            lower = np.percentile(values, 100 - percentile)
            upper = np.percentile(values, percentile)

            result_df[col] = df[col].clip(lower=lower, upper=upper)

        return result_df

    def _exclude(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Exclude outliers by setting to NaN.

        Args:
            df: DataFrame with outlier flags
            columns: Columns to process

        Returns:
            DataFrame with outliers set to NaN
        """
        result_df = df.copy()

        if columns is None:
            columns = [c for c in df.columns
                      if pd.api.types.is_numeric_dtype(df[c])
                      and not c.startswith('_')
                      and not c.endswith('_outlier')
                      and not c.endswith('_zscore')]

        for col in columns:
            outlier_col = f'{col}_outlier'
            if outlier_col in df.columns:
                result_df.loc[df[outlier_col], col] = np.nan

        return result_df

    def detect_iqr_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers using IQR method.

        Args:
            df: DataFrame with factor columns
            columns: Columns to check
            multiplier: IQR multiplier (default 1.5)

        Returns:
            DataFrame with outlier flags
        """
        result_df = df.copy()

        if columns is None:
            columns = [c for c in df.columns
                      if pd.api.types.is_numeric_dtype(df[c])
                      and not c.startswith('_')]

        for col in columns:
            values = df[col].dropna()
            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            result_df[f'{col}_iqr_outlier'] = outlier_mask

        return result_df

    def get_outlier_summary(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Get summary of outlier flags in DataFrame.

        Args:
            df: DataFrame with outlier flag columns

        Returns:
            Summary dictionary
        """
        outlier_cols = [c for c in df.columns if c.endswith('_outlier')]

        summary = {
            'columns_checked': len(outlier_cols),
            'by_column': {}
        }

        for col in outlier_cols:
            base_col = col.replace('_outlier', '')
            summary['by_column'][base_col] = {
                'outliers': int(df[col].sum()),
                'percentage': f"{df[col].mean() * 100:.2f}%"
            }

        if '_outlier_count' in df.columns:
            summary['trades_by_outlier_count'] = df['_outlier_count'].value_counts().to_dict()
            summary['trades_with_any_outlier'] = int((df['_outlier_count'] > 0).sum())

        return summary

    def filter_clean_trades(
        self,
        df: pd.DataFrame,
        max_outliers: int = 0
    ) -> pd.DataFrame:
        """
        Filter to trades with at most max_outliers flagged factors.

        Args:
            df: DataFrame with _outlier_count column
            max_outliers: Maximum allowed outliers per trade

        Returns:
            Filtered DataFrame
        """
        if '_outlier_count' not in df.columns:
            raise ValueError("DataFrame must have _outlier_count column. Run detect_outliers first.")

        return df[df['_outlier_count'] <= max_outliers].copy()
