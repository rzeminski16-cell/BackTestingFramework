"""
Options Factors for Factor Analysis.

Processes options data into factor metrics including IV, skew, and sentiment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..logging.audit_logger import AuditLogger


@dataclass
class OptionsFactorResult:
    """Result of options factor computation."""
    factors_computed: int
    trades_with_data: int
    total_trades: int
    avg_iv: Optional[float]
    factor_names: List[str]


class OptionsFactors:
    """
    Computes options-derived factors.

    Factors include:
    - Implied Volatility (IV) metrics
    - Put/Call ratios
    - IV percentile and skew
    - Options sentiment indicators
    """

    FACTOR_NAMES = [
        'options_iv_median',
        'options_iv_mean',
        'options_iv_percentile',
        'options_put_call_ratio',
        'options_put_call_oi_ratio',
        'options_total_volume',
        'options_bid_ask_spread',
        'options_iv_skew',
        'options_sentiment'
    ]

    def __init__(self, logger: Optional[AuditLogger] = None):
        """
        Initialize OptionsFactors.

        Args:
            logger: Optional audit logger
        """
        self.logger = logger

    def compute_factors(
        self,
        options_data_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, OptionsFactorResult]:
        """
        Process pre-aligned options data into factors.

        The input DataFrame should already be aligned to trades
        (one row per trade with aggregated options metrics).

        Args:
            options_data_df: DataFrame with aligned options metrics

        Returns:
            Tuple of (DataFrame with factors, OptionsFactorResult)
        """
        if self.logger:
            self.logger.start_section("OPTIONS_FACTORS")

        df = options_data_df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        results = []
        trades_with_data = 0
        iv_values = []

        for idx, row in df.iterrows():
            factors = {'_trade_idx': row.get('_trade_idx', idx)}

            has_data = row.get('_options_data_found', False)

            if has_data:
                trades_with_data += 1

                # IV metrics
                iv_median = row.get('options_iv_median')
                iv_mean = row.get('options_iv_mean')

                factors['options_iv_median'] = float(iv_median) if pd.notna(iv_median) else np.nan
                factors['options_iv_mean'] = float(iv_mean) if pd.notna(iv_mean) else np.nan

                if pd.notna(iv_median):
                    iv_values.append(iv_median)

                # IV range/spread
                iv_min = row.get('options_iv_min')
                iv_max = row.get('options_iv_max')
                if pd.notna(iv_min) and pd.notna(iv_max):
                    factors['options_iv_range'] = float(iv_max - iv_min)
                else:
                    factors['options_iv_range'] = np.nan

                # Put/Call ratios
                pcr_volume = row.get('options_put_call_volume_ratio')
                pcr_oi = row.get('options_put_call_oi_ratio')

                factors['options_put_call_ratio'] = float(pcr_volume) if pd.notna(pcr_volume) and pcr_volume != float('inf') else np.nan
                factors['options_put_call_oi_ratio'] = float(pcr_oi) if pd.notna(pcr_oi) and pcr_oi != float('inf') else np.nan

                # Volume metrics
                factors['options_total_volume'] = row.get('options_total_volume', 0) or 0
                factors['options_total_oi'] = row.get('options_total_oi', 0) or 0

                # Bid-ask spread
                spread = row.get('options_avg_bid_ask_spread')
                factors['options_bid_ask_spread'] = float(spread) if pd.notna(spread) else np.nan

                # ATM IV if available
                atm_iv = row.get('options_atm_iv')
                if pd.notna(atm_iv):
                    factors['options_atm_iv'] = float(atm_iv)

                # Greeks if available
                for greek in ['delta', 'gamma', 'vega', 'theta']:
                    greek_val = row.get(f'options_{greek}_mean')
                    if pd.notna(greek_val):
                        factors[f'options_{greek}_mean'] = float(greek_val)

            else:
                # No options data
                for name in self.FACTOR_NAMES:
                    factors[name] = np.nan

            results.append(factors)

        factors_df = pd.DataFrame(results)

        # Compute IV percentile (relative to dataset)
        if iv_values:
            avg_iv = np.mean(iv_values)
            factors_df['options_iv_percentile'] = factors_df['options_iv_median'].rank(pct=True) * 100
        else:
            avg_iv = None
            factors_df['options_iv_percentile'] = np.nan

        # Compute derived factors
        factors_df = self._compute_derived_factors(factors_df)

        result = OptionsFactorResult(
            factors_computed=len([c for c in factors_df.columns if c.startswith('options_')]),
            trades_with_data=trades_with_data,
            total_trades=len(df),
            avg_iv=round(avg_iv, 4) if avg_iv else None,
            factor_names=self.FACTOR_NAMES
        )

        if self.logger:
            self.logger.log_factor_engineering(
                category='Options',
                factor_count=result.factors_computed,
                trades_with_data=result.trades_with_data,
                total_trades=result.total_trades
            )
            self.logger.info("Options data summary", {
                'avg_iv': result.avg_iv,
                'coverage': f"{trades_with_data / len(df) * 100:.1f}%" if len(df) > 0 else "0%"
            })
            self.logger.end_section()

        return factors_df, result

    def _compute_derived_factors(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute derived options factors.

        Args:
            df: DataFrame with options factors

        Returns:
            DataFrame with added derived factors
        """
        df = df.copy()

        # IV regime classification
        if 'options_iv_percentile' in df.columns:
            df['options_iv_regime'] = pd.cut(
                df['options_iv_percentile'],
                bins=[0, 25, 75, 100],
                labels=['low', 'normal', 'high']
            )

        # Put/Call sentiment
        if 'options_put_call_ratio' in df.columns:
            pcr = df['options_put_call_ratio']
            conditions = [
                pcr < 0.7,
                (pcr >= 0.7) & (pcr <= 1.3),
                pcr > 1.3
            ]
            choices = ['bullish', 'neutral', 'bearish']
            df['options_pcr_sentiment'] = np.select(conditions, choices, default='neutral')

            # Numeric encoding
            sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            df['options_pcr_sentiment_numeric'] = df['options_pcr_sentiment'].map(sentiment_map)

        # Combined options sentiment
        if 'options_iv_regime' in df.columns and 'options_pcr_sentiment' in df.columns:
            # High IV + bearish PCR = very bearish
            # Low IV + bullish PCR = very bullish
            def combined_sentiment(row):
                iv = row.get('options_iv_regime')
                pcr = row.get('options_pcr_sentiment')

                if pd.isna(iv) or pd.isna(pcr):
                    return 'neutral'

                if iv == 'high' and pcr == 'bearish':
                    return 'very_bearish'
                elif iv == 'low' and pcr == 'bullish':
                    return 'very_bullish'
                elif iv == 'high' or pcr == 'bearish':
                    return 'bearish'
                elif iv == 'low' or pcr == 'bullish':
                    return 'bullish'
                else:
                    return 'neutral'

            df['options_combined_sentiment'] = df.apply(combined_sentiment, axis=1)

        return df

    def compute_relative_factors(
        self,
        factors_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute factors relative to cross-sectional distribution.

        Args:
            factors_df: DataFrame with options factors

        Returns:
            DataFrame with added relative factors
        """
        df = factors_df.copy()

        # Z-scores for key metrics
        for col in ['options_iv_median', 'options_put_call_ratio']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_zscore'] = (df[col] - mean) / std
                else:
                    df[f'{col}_zscore'] = 0

        return df

    def get_options_summary(
        self,
        factors_df: pd.DataFrame
    ) -> Dict:
        """
        Get summary of options data across all trades.

        Args:
            factors_df: DataFrame with options factors

        Returns:
            Summary dictionary
        """
        summary = {
            'total_trades': len(factors_df),
            'trades_with_data': factors_df['options_iv_median'].notna().sum() if 'options_iv_median' in factors_df.columns else 0
        }

        if 'options_iv_median' in factors_df.columns:
            iv = factors_df['options_iv_median'].dropna()
            if len(iv) > 0:
                summary['iv_stats'] = {
                    'mean': round(iv.mean(), 4),
                    'median': round(iv.median(), 4),
                    'min': round(iv.min(), 4),
                    'max': round(iv.max(), 4)
                }

        if 'options_iv_regime' in factors_df.columns:
            summary['iv_regime_distribution'] = factors_df['options_iv_regime'].value_counts().to_dict()

        if 'options_pcr_sentiment' in factors_df.columns:
            summary['pcr_sentiment_distribution'] = factors_df['options_pcr_sentiment'].value_counts().to_dict()

        return summary
