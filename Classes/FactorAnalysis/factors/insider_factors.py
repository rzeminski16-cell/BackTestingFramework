"""
Insider Factors for Factor Analysis.

Processes insider transaction data into factor metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..logging.audit_logger import AuditLogger


@dataclass
class InsiderFactorResult:
    """Result of insider factor computation."""
    factors_computed: int
    trades_with_activity: int
    total_trades: int
    avg_transactions_per_trade: float
    factor_names: List[str]


class InsiderFactors:
    """
    Computes insider activity factors.

    Factors include:
    - Transaction counts (buys, sells, net)
    - Share volumes
    - Value metrics
    - Sentiment scores
    - Executive vs non-executive activity
    """

    FACTOR_NAMES = [
        'insider_buy_count',
        'insider_sell_count',
        'insider_net_count',
        'insider_total_count',
        'insider_buy_ratio',
        'insider_net_shares',
        'insider_net_value',
        'insider_score',
        'insider_executive_ratio',
        'insider_sentiment'
    ]

    def __init__(self, logger: Optional[AuditLogger] = None):
        """
        Initialize InsiderFactors.

        Args:
            logger: Optional audit logger
        """
        self.logger = logger

    def compute_factors(
        self,
        insider_data_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, InsiderFactorResult]:
        """
        Process pre-aligned insider data into factors.

        The input DataFrame should already be aligned to trades
        (one row per trade with aggregated insider metrics).

        Args:
            insider_data_df: DataFrame with aligned insider metrics

        Returns:
            Tuple of (DataFrame with factors, InsiderFactorResult)
        """
        if self.logger:
            self.logger.start_section("INSIDER_FACTORS")

        df = insider_data_df.copy()

        # Standardize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Compute derived factors
        results = []
        trades_with_activity = 0
        total_transactions = 0

        for idx, row in df.iterrows():
            factors = {'_trade_idx': row.get('_trade_idx', idx)}

            # Extract base metrics
            buy_count = row.get('insider_buy_count', 0) or 0
            sell_count = row.get('insider_sell_count', 0) or 0
            total_count = buy_count + sell_count

            # Basic counts
            factors['insider_buy_count'] = int(buy_count)
            factors['insider_sell_count'] = int(sell_count)
            factors['insider_net_count'] = int(buy_count - sell_count)
            factors['insider_total_count'] = int(total_count)

            # Buy ratio
            if total_count > 0:
                factors['insider_buy_ratio'] = buy_count / total_count
                trades_with_activity += 1
            else:
                factors['insider_buy_ratio'] = 0.5  # Neutral when no activity

            total_transactions += total_count

            # Share and value metrics
            factors['insider_net_shares'] = row.get('insider_net_shares', 0) or 0
            factors['insider_net_value'] = row.get('insider_net_value', 0) or 0

            # Insider score (from data loader or compute here)
            if 'insider_score' in row and pd.notna(row['insider_score']):
                factors['insider_score'] = float(row['insider_score'])
            else:
                # Compute score: weighted by count
                if total_count > 0:
                    factors['insider_score'] = (buy_count - sell_count) / total_count * 100
                else:
                    factors['insider_score'] = 0.0

            # Executive ratio
            exec_buys = row.get('insider_executive_buys', 0) or 0
            exec_sells = row.get('insider_executive_sells', 0) or 0
            exec_total = exec_buys + exec_sells

            if total_count > 0:
                factors['insider_executive_ratio'] = exec_total / total_count
            else:
                factors['insider_executive_ratio'] = 0.0

            # Executive net (weighted more heavily)
            factors['insider_executive_net'] = int(exec_buys - exec_sells)

            # Sentiment classification
            if factors['insider_score'] > 25:
                factors['insider_sentiment'] = 'bullish'
                factors['insider_sentiment_numeric'] = 1
            elif factors['insider_score'] < -25:
                factors['insider_sentiment'] = 'bearish'
                factors['insider_sentiment_numeric'] = -1
            else:
                factors['insider_sentiment'] = 'neutral'
                factors['insider_sentiment_numeric'] = 0

            # Activity level classification
            if total_count == 0:
                factors['insider_activity_level'] = 'none'
            elif total_count <= 2:
                factors['insider_activity_level'] = 'low'
            elif total_count <= 5:
                factors['insider_activity_level'] = 'moderate'
            else:
                factors['insider_activity_level'] = 'high'

            results.append(factors)

        factors_df = pd.DataFrame(results)

        # Compute result summary
        avg_transactions = total_transactions / len(df) if len(df) > 0 else 0

        result = InsiderFactorResult(
            factors_computed=len([c for c in factors_df.columns if c.startswith('insider_')]),
            trades_with_activity=trades_with_activity,
            total_trades=len(df),
            avg_transactions_per_trade=round(avg_transactions, 2),
            factor_names=self.FACTOR_NAMES
        )

        if self.logger:
            self.logger.log_factor_engineering(
                category='Insider',
                factor_count=result.factors_computed,
                trades_with_data=result.trades_with_activity,
                total_trades=result.total_trades
            )
            self.logger.info("Insider activity summary", {
                'trades_with_activity': trades_with_activity,
                'avg_transactions': avg_transactions
            })
            self.logger.end_section()

        return factors_df, result

    def compute_relative_factors(
        self,
        factors_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute factors relative to cross-sectional distribution.

        Args:
            factors_df: DataFrame with insider factors

        Returns:
            DataFrame with added relative factors
        """
        df = factors_df.copy()

        # Percentile ranks for key metrics
        for col in ['insider_total_count', 'insider_score', 'insider_net_value']:
            if col in df.columns:
                # Percentile rank (0-100)
                df[f'{col}_percentile'] = df[col].rank(pct=True) * 100

        # Z-scores
        for col in ['insider_score', 'insider_net_shares']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_zscore'] = (df[col] - mean) / std
                else:
                    df[f'{col}_zscore'] = 0

        return df

    def get_activity_summary(
        self,
        factors_df: pd.DataFrame
    ) -> Dict:
        """
        Get summary of insider activity across all trades.

        Args:
            factors_df: DataFrame with insider factors

        Returns:
            Summary dictionary
        """
        if 'insider_activity_level' in factors_df.columns:
            activity_dist = factors_df['insider_activity_level'].value_counts().to_dict()
        else:
            activity_dist = {}

        if 'insider_sentiment' in factors_df.columns:
            sentiment_dist = factors_df['insider_sentiment'].value_counts().to_dict()
        else:
            sentiment_dist = {}

        return {
            'total_trades': len(factors_df),
            'trades_with_activity': (factors_df['insider_total_count'] > 0).sum() if 'insider_total_count' in factors_df.columns else 0,
            'activity_distribution': activity_dist,
            'sentiment_distribution': sentiment_dist,
            'avg_score': factors_df['insider_score'].mean() if 'insider_score' in factors_df.columns else 0,
            'avg_buy_ratio': factors_df['insider_buy_ratio'].mean() if 'insider_buy_ratio' in factors_df.columns else 0.5
        }
