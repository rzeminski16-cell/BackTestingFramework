"""
Data Quality Scorer for Factor Analysis.

Computes quality scores for each trade based on data completeness and reliability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..logging.audit_logger import AuditLogger


@dataclass
class QualityScore:
    """Quality score for a single trade."""
    trade_id: str
    overall_score: float
    fundamental_score: float
    insider_score: float
    options_score: float
    price_score: float
    outlier_penalty: float
    flags: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Overall data quality report."""
    total_trades: int
    avg_quality_score: float
    median_quality_score: float
    std_quality_score: float
    trades_excellent: int  # > 80%
    trades_good: int  # 60-80%
    trades_fair: int  # 40-60%
    trades_poor: int  # < 40%
    coverage_by_source: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class QualityScorer:
    """
    Computes data quality scores for trades.

    Quality Score Formula:
    score = (
        fundamental_weight * fundamental_available +
        insider_weight * insider_available +
        options_weight * options_available +
        price_weight * price_available
    ) * (1 - outlier_penalty_factor * outlier_count)

    Default Weights:
    - Fundamental: 25%
    - Insider: 20%
    - Options: 15%
    - Price: 30%
    - Outlier penalty: 5% per outlier

    Remaining 10% is reserved for data recency/quality factors.
    """

    DEFAULT_WEIGHTS = {
        'fundamental': 0.25,
        'insider': 0.20,
        'options': 0.15,
        'price': 0.30,
        'recency': 0.10
    }

    OUTLIER_PENALTY = 0.05  # 5% per outlier

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize QualityScorer.

        Args:
            weights: Custom weights for data sources
            logger: Optional audit logger
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.logger = logger

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def score_trade(
        self,
        trade_row: pd.Series,
        fundamental_cols: Optional[List[str]] = None,
        insider_cols: Optional[List[str]] = None,
        options_cols: Optional[List[str]] = None,
        price_cols: Optional[List[str]] = None,
        outlier_cols: Optional[List[str]] = None
    ) -> QualityScore:
        """
        Compute quality score for a single trade.

        Args:
            trade_row: Series with trade data
            fundamental_cols: List of fundamental factor columns
            insider_cols: List of insider factor columns
            options_cols: List of options factor columns
            price_cols: List of price/indicator columns
            outlier_cols: List of columns flagged as outliers

        Returns:
            QualityScore for the trade
        """
        flags = []

        # Calculate availability for each source
        def calc_availability(cols: Optional[List[str]]) -> float:
            if not cols:
                return 0.0
            available = [c for c in cols if c in trade_row.index]
            if not available:
                return 0.0
            non_null = sum(1 for c in available if pd.notna(trade_row[c]))
            return non_null / len(available)

        fundamental_avail = calc_availability(fundamental_cols)
        insider_avail = calc_availability(insider_cols)
        options_avail = calc_availability(options_cols)
        price_avail = calc_availability(price_cols)

        # Calculate recency score (based on days before entry)
        recency_score = 1.0
        for col in ['_fundamental_days_before_entry', '_options_days_before_entry']:
            if col in trade_row.index and pd.notna(trade_row[col]):
                days = trade_row[col]
                if days > 60:
                    recency_score *= 0.8
                    flags.append(f"Stale data: {col} = {days} days")
                elif days > 30:
                    recency_score *= 0.9

        # Count outliers
        outlier_count = 0
        if outlier_cols:
            for col in outlier_cols:
                if col in trade_row.index and trade_row[col]:
                    outlier_count += 1
                    flags.append(f"Outlier: {col}")

        outlier_penalty = min(outlier_count * self.OUTLIER_PENALTY, 0.5)  # Cap at 50%

        # Add flags for missing data
        if fundamental_avail < 0.5:
            flags.append("Low fundamental data coverage")
        if options_avail < 0.3:
            flags.append("Limited options data")
        if price_avail < 0.8:
            flags.append("Missing price/indicator data")

        # Calculate weighted score
        base_score = (
            self.weights.get('fundamental', 0.25) * fundamental_avail +
            self.weights.get('insider', 0.20) * insider_avail +
            self.weights.get('options', 0.15) * options_avail +
            self.weights.get('price', 0.30) * price_avail +
            self.weights.get('recency', 0.10) * recency_score
        )

        overall_score = base_score * (1 - outlier_penalty) * 100  # Scale to 0-100

        return QualityScore(
            trade_id=str(trade_row.get('trade_id', '')),
            overall_score=round(overall_score, 1),
            fundamental_score=round(fundamental_avail * 100, 1),
            insider_score=round(insider_avail * 100, 1),
            options_score=round(options_avail * 100, 1),
            price_score=round(price_avail * 100, 1),
            outlier_penalty=round(outlier_penalty * 100, 1),
            flags=flags
        )

    def score_all_trades(
        self,
        enriched_df: pd.DataFrame,
        outlier_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, QualityReport]:
        """
        Compute quality scores for all trades.

        Args:
            enriched_df: Enriched trade DataFrame
            outlier_df: Optional DataFrame with outlier flags

        Returns:
            Tuple of (DataFrame with quality columns, QualityReport)
        """
        if self.logger:
            self.logger.start_section("QUALITY_SCORING")

        # Identify columns by category
        fundamental_cols = [c for c in enriched_df.columns
                          if c.startswith('fund_') and not c.startswith('_')]
        insider_cols = [c for c in enriched_df.columns
                       if c.startswith('insider_') and not c.startswith('_')]
        options_cols = [c for c in enriched_df.columns
                       if c.startswith('options_') and not c.startswith('_')]
        price_cols = [c for c in enriched_df.columns
                     if c.startswith('price_') and not c.startswith('_')]

        # Get outlier columns if available
        outlier_cols = []
        if outlier_df is not None:
            outlier_cols = [c for c in outlier_df.columns if c.endswith('_outlier')]

        # Score each trade
        scores = []
        for idx, row in enriched_df.iterrows():
            # Merge outlier info if available
            if outlier_df is not None and idx in outlier_df.index:
                combined_row = pd.concat([row, outlier_df.loc[idx]])
            else:
                combined_row = row

            score = self.score_trade(
                combined_row,
                fundamental_cols=fundamental_cols,
                insider_cols=insider_cols,
                options_cols=options_cols,
                price_cols=price_cols,
                outlier_cols=outlier_cols
            )
            scores.append(score)

        # Add scores to DataFrame
        df = enriched_df.copy()
        df['quality_score'] = [s.overall_score for s in scores]
        df['quality_fundamental'] = [s.fundamental_score for s in scores]
        df['quality_insider'] = [s.insider_score for s in scores]
        df['quality_options'] = [s.options_score for s in scores]
        df['quality_price'] = [s.price_score for s in scores]
        df['quality_outlier_penalty'] = [s.outlier_penalty for s in scores]
        df['quality_flags'] = ['; '.join(s.flags) for s in scores]

        # Generate report
        all_scores = df['quality_score']
        report = QualityReport(
            total_trades=len(df),
            avg_quality_score=round(all_scores.mean(), 1),
            median_quality_score=round(all_scores.median(), 1),
            std_quality_score=round(all_scores.std(), 1),
            trades_excellent=int((all_scores > 80).sum()),
            trades_good=int(((all_scores > 60) & (all_scores <= 80)).sum()),
            trades_fair=int(((all_scores > 40) & (all_scores <= 60)).sum()),
            trades_poor=int((all_scores <= 40).sum()),
            coverage_by_source={
                'fundamental': round(df['quality_fundamental'].mean(), 1),
                'insider': round(df['quality_insider'].mean(), 1),
                'options': round(df['quality_options'].mean(), 1),
                'price': round(df['quality_price'].mean(), 1)
            }
        )

        # Generate recommendations
        if report.trades_poor > len(df) * 0.2:
            report.recommendations.append(
                f"High proportion of poor quality trades ({report.trades_poor}). "
                "Consider filtering to quality_score > 40%."
            )
        if report.coverage_by_source.get('fundamental', 0) < 50:
            report.recommendations.append(
                "Low fundamental data coverage. Check data alignment and availability."
            )
        if report.coverage_by_source.get('options', 0) < 30:
            report.recommendations.append(
                "Limited options data. Consider excluding options factors from analysis."
            )

        if self.logger:
            self.logger.info("Quality scoring complete", {
                'avg_score': report.avg_quality_score,
                'excellent': report.trades_excellent,
                'good': report.trades_good,
                'fair': report.trades_fair,
                'poor': report.trades_poor
            })
            for rec in report.recommendations:
                self.logger.warning("Quality recommendation", {'message': rec})
            self.logger.end_section()

        return df, report

    def filter_by_quality(
        self,
        df: pd.DataFrame,
        min_score: float = 40.0,
        require_fundamental: bool = False,
        require_price: bool = True
    ) -> pd.DataFrame:
        """
        Filter trades by quality score.

        Args:
            df: DataFrame with quality scores
            min_score: Minimum overall quality score
            require_fundamental: Whether to require fundamental data
            require_price: Whether to require price data

        Returns:
            Filtered DataFrame
        """
        if 'quality_score' not in df.columns:
            raise ValueError("DataFrame must have quality_score column. Run score_all_trades first.")

        mask = df['quality_score'] >= min_score

        if require_fundamental and 'quality_fundamental' in df.columns:
            mask &= df['quality_fundamental'] > 0

        if require_price and 'quality_price' in df.columns:
            mask &= df['quality_price'] > 50

        return df[mask].copy()

    def get_quality_distribution(
        self,
        df: pd.DataFrame,
        by_class: bool = False
    ) -> pd.DataFrame:
        """
        Get distribution of quality scores.

        Args:
            df: DataFrame with quality scores
            by_class: Whether to group by trade_class

        Returns:
            DataFrame with quality distribution
        """
        if 'quality_score' not in df.columns:
            raise ValueError("DataFrame must have quality_score column")

        # Create quality bins
        bins = [0, 40, 60, 80, 100]
        labels = ['Poor', 'Fair', 'Good', 'Excellent']
        df = df.copy()
        df['quality_category'] = pd.cut(df['quality_score'], bins=bins, labels=labels)

        if by_class and 'trade_class' in df.columns:
            return df.groupby(['trade_class', 'quality_category']).size().unstack(fill_value=0)
        else:
            return df['quality_category'].value_counts().sort_index()
