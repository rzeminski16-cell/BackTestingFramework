"""
Trade Classifier for Factor Analysis.

Classifies trades as good, bad, or indeterminate based on configurable thresholds.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ..config.factor_config import TradeClassificationConfig, ThresholdType
from ..logging.audit_logger import AuditLogger


class TradeClass(Enum):
    """Trade classification categories."""
    GOOD = "good"
    BAD = "bad"
    INDETERMINATE = "indeterminate"


@dataclass
class ClassificationResult:
    """Result of trade classification."""
    total_trades: int
    good_count: int
    bad_count: int
    indeterminate_count: int
    good_pct: float
    bad_pct: float
    indeterminate_pct: float
    thresholds_used: Dict


class TradeClassifier:
    """
    Classifies trades based on performance thresholds.

    Classification Logic (absolute thresholds):
    - GOOD: pl_pct > good_threshold_pct
    - BAD: pl_pct < bad_threshold_pct AND duration_days >= bad_min_days
    - INDETERMINATE: Everything else (middle range or short duration bad trades)

    Classification Logic (percentile thresholds):
    - GOOD: pl_pct > percentile(good_threshold_pct)
    - BAD: pl_pct < percentile(bad_threshold_pct)
    - INDETERMINATE: Everything in between
    """

    def __init__(
        self,
        config: Optional[TradeClassificationConfig] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize TradeClassifier.

        Args:
            config: Classification configuration
            logger: Optional audit logger
        """
        self.config = config or TradeClassificationConfig()
        self.logger = logger

    def classify_trade(
        self,
        pl_pct: float,
        duration_days: int,
        thresholds: Optional[Dict] = None
    ) -> TradeClass:
        """
        Classify a single trade.

        Args:
            pl_pct: Trade return percentage
            duration_days: Trade holding period in days
            thresholds: Optional override thresholds

        Returns:
            TradeClass enum value
        """
        good_thresh = thresholds.get('good') if thresholds else self.config.good_threshold_pct
        bad_thresh = thresholds.get('bad') if thresholds else self.config.bad_threshold_pct
        indet_max_days = self.config.indeterminate_max_days
        bad_min_days = self.config.bad_min_days

        # Check for good trade
        if pl_pct > good_thresh:
            return TradeClass.GOOD

        # Check for bad trade
        if pl_pct < bad_thresh:
            # Must also meet duration requirement
            if duration_days >= bad_min_days:
                return TradeClass.BAD
            else:
                # Short-duration losing trade is indeterminate
                return TradeClass.INDETERMINATE

        # Middle range: pl_pct between thresholds
        if duration_days <= indet_max_days:
            return TradeClass.INDETERMINATE
        else:
            # Long duration but mediocre return = bad
            return TradeClass.BAD

    def _compute_percentile_thresholds(
        self,
        pl_pct_series: pd.Series
    ) -> Tuple[float, float]:
        """Compute thresholds based on percentiles."""
        good_percentile = 100 - self.config.good_threshold_pct  # e.g., 2.0 -> top 2%
        bad_percentile = abs(self.config.bad_threshold_pct)  # e.g., -1.0 -> bottom 1%

        good_thresh = np.percentile(pl_pct_series.dropna(), good_percentile)
        bad_thresh = np.percentile(pl_pct_series.dropna(), bad_percentile)

        return good_thresh, bad_thresh

    def classify_trades(
        self,
        trades_df: pd.DataFrame,
        pl_column: str = 'pl_pct',
        duration_column: str = 'duration_days'
    ) -> Tuple[pd.DataFrame, ClassificationResult]:
        """
        Classify all trades in a DataFrame.

        Args:
            trades_df: Trade log DataFrame
            pl_column: Column name for P&L percentage
            duration_column: Column name for duration days

        Returns:
            Tuple of (DataFrame with classification column, ClassificationResult)
        """
        df = trades_df.copy()

        # Ensure required columns exist
        if pl_column not in df.columns:
            raise ValueError(f"P&L column '{pl_column}' not found in DataFrame")

        # Calculate duration if not present
        if duration_column not in df.columns:
            if 'entry_date' in df.columns and 'exit_date' in df.columns:
                df[duration_column] = (
                    pd.to_datetime(df['exit_date']) - pd.to_datetime(df['entry_date'])
                ).dt.days
            else:
                df[duration_column] = 0

        # Compute thresholds
        if self.config.threshold_type == ThresholdType.PERCENTILE:
            good_thresh, bad_thresh = self._compute_percentile_thresholds(df[pl_column])
            thresholds = {'good': good_thresh, 'bad': bad_thresh}
        else:
            good_thresh = self.config.good_threshold_pct
            bad_thresh = self.config.bad_threshold_pct
            thresholds = {'good': good_thresh, 'bad': bad_thresh}

        # Classify each trade
        classifications = []
        for _, row in df.iterrows():
            pl = row[pl_column]
            duration = row[duration_column]

            if pd.isna(pl):
                classifications.append(TradeClass.INDETERMINATE.value)
            else:
                trade_class = self.classify_trade(pl, duration, thresholds)
                classifications.append(trade_class.value)

        df['trade_class'] = classifications

        # Also add numeric encoding for analysis
        class_to_num = {
            TradeClass.GOOD.value: 1,
            TradeClass.BAD.value: -1,
            TradeClass.INDETERMINATE.value: 0
        }
        df['trade_class_numeric'] = df['trade_class'].map(class_to_num)

        # Compute summary
        good_count = (df['trade_class'] == TradeClass.GOOD.value).sum()
        bad_count = (df['trade_class'] == TradeClass.BAD.value).sum()
        indet_count = (df['trade_class'] == TradeClass.INDETERMINATE.value).sum()
        total = len(df)

        result = ClassificationResult(
            total_trades=total,
            good_count=int(good_count),
            bad_count=int(bad_count),
            indeterminate_count=int(indet_count),
            good_pct=good_count / total * 100 if total > 0 else 0,
            bad_pct=bad_count / total * 100 if total > 0 else 0,
            indeterminate_pct=indet_count / total * 100 if total > 0 else 0,
            thresholds_used={
                'good_threshold_pct': good_thresh,
                'bad_threshold_pct': bad_thresh,
                'indeterminate_max_days': self.config.indeterminate_max_days,
                'bad_min_days': self.config.bad_min_days,
                'threshold_type': self.config.threshold_type.value
            }
        )

        if self.logger:
            self.logger.log_trade_classification(
                good_count=result.good_count,
                bad_count=result.bad_count,
                indeterminate_count=result.indeterminate_count,
                config_used=result.thresholds_used
            )

        return df, result

    def get_trades_by_class(
        self,
        trades_df: pd.DataFrame,
        trade_class: TradeClass
    ) -> pd.DataFrame:
        """
        Filter trades by classification.

        Args:
            trades_df: DataFrame with 'trade_class' column
            trade_class: Class to filter for

        Returns:
            Filtered DataFrame
        """
        if 'trade_class' not in trades_df.columns:
            raise ValueError("DataFrame must have 'trade_class' column. Run classify_trades first.")

        return trades_df[trades_df['trade_class'] == trade_class.value].copy()

    def get_class_statistics(
        self,
        trades_df: pd.DataFrame,
        metrics: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Get statistics for each trade class.

        Args:
            trades_df: DataFrame with 'trade_class' column
            metrics: List of columns to compute stats for

        Returns:
            DataFrame with statistics per class
        """
        if 'trade_class' not in trades_df.columns:
            raise ValueError("DataFrame must have 'trade_class' column")

        if metrics is None:
            metrics = ['pl_pct', 'pl', 'duration_days']

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in trades_df.columns]

        stats = trades_df.groupby('trade_class')[available_metrics].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ])

        return stats
