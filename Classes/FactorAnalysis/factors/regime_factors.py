"""
Regime Factors for Factor Analysis.

Classifies market conditions into regimes based on volatility, trend, and momentum.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..logging.audit_logger import AuditLogger


@dataclass
class RegimeFactorResult:
    """Result of regime factor computation."""
    factors_computed: int
    trades_analyzed: int
    regime_distribution: Dict[str, Dict[str, int]]
    factor_names: List[str]


class RegimeFactors:
    """
    Computes market regime classification factors.

    Regimes:
    - Volatility: low, normal, high (based on ATR percentile)
    - Trend: bullish, neutral, bearish (based on SMA relationships)
    - Momentum: oversold, neutral, overbought (based on RSI)
    - Combined market regime
    """

    FACTOR_NAMES = [
        'regime_volatility',
        'regime_trend',
        'regime_momentum',
        'regime_combined',
        'regime_score'
    ]

    def __init__(
        self,
        volatility_lookback: int = 252,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize RegimeFactors.

        Args:
            volatility_lookback: Days for volatility percentile calculation
            logger: Optional audit logger
        """
        self.volatility_lookback = volatility_lookback
        self.logger = logger

    def compute_factors(
        self,
        trades_df: pd.DataFrame,
        technical_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, RegimeFactorResult]:
        """
        Compute regime factors for each trade.

        Args:
            trades_df: Trade log DataFrame
            technical_df: Technical factors DataFrame (aligned to trades)

        Returns:
            Tuple of (DataFrame with regime factors, RegimeFactorResult)
        """
        if self.logger:
            self.logger.start_section("REGIME_FACTORS")

        results = []
        regime_counts = {
            'volatility': {'low': 0, 'normal': 0, 'high': 0},
            'trend': {'bullish': 0, 'neutral': 0, 'bearish': 0},
            'momentum': {'oversold': 0, 'neutral': 0, 'overbought': 0}
        }

        for idx in range(len(trades_df)):
            factors = {'_trade_idx': idx}

            # Get technical data for this trade
            if idx < len(technical_df):
                tech_row = technical_df.iloc[idx]
            else:
                tech_row = pd.Series()

            # Volatility regime
            vol_regime = self._classify_volatility(tech_row)
            factors['regime_volatility'] = vol_regime
            factors['regime_volatility_numeric'] = {'low': -1, 'normal': 0, 'high': 1}.get(vol_regime, 0)
            if vol_regime in regime_counts['volatility']:
                regime_counts['volatility'][vol_regime] += 1

            # Trend regime
            trend_regime = self._classify_trend(tech_row)
            factors['regime_trend'] = trend_regime
            factors['regime_trend_numeric'] = {'bearish': -1, 'neutral': 0, 'bullish': 1}.get(trend_regime, 0)
            if trend_regime in regime_counts['trend']:
                regime_counts['trend'][trend_regime] += 1

            # Momentum regime
            momentum_regime = self._classify_momentum(tech_row)
            factors['regime_momentum'] = momentum_regime
            factors['regime_momentum_numeric'] = {'oversold': -1, 'neutral': 0, 'overbought': 1}.get(momentum_regime, 0)
            if momentum_regime in regime_counts['momentum']:
                regime_counts['momentum'][momentum_regime] += 1

            # Combined regime
            combined = self._compute_combined_regime(vol_regime, trend_regime, momentum_regime)
            factors['regime_combined'] = combined

            # Regime score (bullish = positive, bearish = negative)
            factors['regime_score'] = (
                factors['regime_trend_numeric'] +
                factors['regime_momentum_numeric'] * 0.5 -
                factors['regime_volatility_numeric'] * 0.3
            )

            results.append(factors)

        factors_df = pd.DataFrame(results)

        result = RegimeFactorResult(
            factors_computed=len(self.FACTOR_NAMES),
            trades_analyzed=len(trades_df),
            regime_distribution=regime_counts,
            factor_names=self.FACTOR_NAMES
        )

        if self.logger:
            self.logger.log_factor_engineering(
                category='Regime',
                factor_count=result.factors_computed,
                trades_with_data=result.trades_analyzed,
                total_trades=result.trades_analyzed
            )
            self.logger.info("Regime distribution", regime_counts)
            self.logger.end_section()

        return factors_df, result

    def _classify_volatility(self, tech_row: pd.Series) -> str:
        """Classify volatility regime."""
        # Look for ATR or NATR columns
        atr_cols = [c for c in tech_row.index if 'atr' in c.lower()]

        if not atr_cols:
            return 'normal'

        # Use first ATR column found
        atr_col = atr_cols[0]
        atr_value = tech_row.get(atr_col)

        if pd.isna(atr_value):
            return 'normal'

        # Look for ATR percentile column
        percentile_col = f'{atr_col}_percentile'
        if percentile_col in tech_row.index:
            percentile = tech_row[percentile_col]
            if pd.notna(percentile):
                if percentile < 25:
                    return 'low'
                elif percentile > 75:
                    return 'high'
                else:
                    return 'normal'

        # Fallback: assume normal
        return 'normal'

    def _classify_trend(self, tech_row: pd.Series) -> str:
        """Classify trend regime based on moving averages."""
        # Look for SMA/EMA columns
        sma_50 = None
        sma_200 = None

        for col in tech_row.index:
            col_lower = col.lower()
            if ('sma' in col_lower or 'ema' in col_lower):
                if '50' in col_lower:
                    sma_50 = tech_row.get(col)
                elif '200' in col_lower:
                    sma_200 = tech_row.get(col)

        # Look for close price
        close = tech_row.get('close', tech_row.get('price_close'))

        if pd.notna(sma_50) and pd.notna(sma_200):
            # Golden cross / Death cross logic
            if sma_50 > sma_200 * 1.02:
                return 'bullish'
            elif sma_50 < sma_200 * 0.98:
                return 'bearish'
            else:
                return 'neutral'
        elif pd.notna(close) and pd.notna(sma_50):
            # Price relative to 50 SMA
            if close > sma_50 * 1.03:
                return 'bullish'
            elif close < sma_50 * 0.97:
                return 'bearish'
            else:
                return 'neutral'

        # Look for ADX for trend strength
        adx_cols = [c for c in tech_row.index if 'adx' in c.lower()]
        if adx_cols:
            adx = tech_row.get(adx_cols[0])
            if pd.notna(adx):
                if adx < 20:
                    return 'neutral'  # No clear trend

        return 'neutral'

    def _classify_momentum(self, tech_row: pd.Series) -> str:
        """Classify momentum regime based on RSI."""
        # Look for RSI column
        rsi_cols = [c for c in tech_row.index if 'rsi' in c.lower()]

        if not rsi_cols:
            return 'neutral'

        rsi = tech_row.get(rsi_cols[0])

        if pd.isna(rsi):
            return 'neutral'

        if rsi < 30:
            return 'oversold'
        elif rsi > 70:
            return 'overbought'
        else:
            return 'neutral'

    def _compute_combined_regime(
        self,
        volatility: str,
        trend: str,
        momentum: str
    ) -> str:
        """Compute combined market regime."""
        # Favorable conditions
        if trend == 'bullish' and momentum != 'overbought' and volatility != 'high':
            return 'favorable_long'

        if trend == 'bearish' and momentum != 'oversold' and volatility != 'high':
            return 'favorable_short'

        # Risky conditions
        if volatility == 'high':
            if trend == 'bearish':
                return 'high_risk_bearish'
            else:
                return 'high_risk'

        # Contrarian opportunities
        if momentum == 'oversold' and trend != 'bearish':
            return 'potential_reversal_up'
        if momentum == 'overbought' and trend != 'bullish':
            return 'potential_reversal_down'

        return 'neutral'

    def compute_regime_features(
        self,
        factors_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create one-hot encoded regime features for ML models.

        Args:
            factors_df: DataFrame with regime factors

        Returns:
            DataFrame with added one-hot features
        """
        df = factors_df.copy()

        # One-hot encode regimes
        for col in ['regime_volatility', 'regime_trend', 'regime_momentum']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)

        return df

    def get_regime_summary(
        self,
        factors_df: pd.DataFrame
    ) -> Dict:
        """
        Get summary of regime classifications.

        Args:
            factors_df: DataFrame with regime factors

        Returns:
            Summary dictionary
        """
        summary = {'total_trades': len(factors_df)}

        for col in ['regime_volatility', 'regime_trend', 'regime_momentum', 'regime_combined']:
            if col in factors_df.columns:
                summary[col] = factors_df[col].value_counts().to_dict()

        if 'regime_score' in factors_df.columns:
            summary['regime_score_stats'] = {
                'mean': round(factors_df['regime_score'].mean(), 3),
                'std': round(factors_df['regime_score'].std(), 3),
                'min': round(factors_df['regime_score'].min(), 3),
                'max': round(factors_df['regime_score'].max(), 3)
            }

        return summary
