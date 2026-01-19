"""
Tier 1 Exploratory Analysis for Factor Analysis.

Provides descriptive statistics, correlations, and distribution analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats

from ..config.factor_config import Tier1Config
from ..logging.audit_logger import AuditLogger
from .analysis_result import CorrelationResult, FactorResult


class Tier1Exploratory:
    """
    Tier 1: Exploratory Descriptive Statistics.

    Computes:
    - Summary statistics per trade class
    - Pearson and Spearman correlations
    - Distribution characteristics
    """

    def __init__(
        self,
        config: Optional[Tier1Config] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize Tier1Exploratory.

        Args:
            config: Tier 1 configuration
            logger: Optional audit logger
        """
        self.config = config or Tier1Config()
        self.logger = logger

    def analyze(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str = 'trade_class_numeric',
        class_column: str = 'trade_class'
    ) -> Dict:
        """
        Run complete Tier 1 analysis.

        Args:
            df: Enriched trade DataFrame
            factor_columns: List of factor column names
            outcome_column: Numeric outcome column
            class_column: Categorical class column

        Returns:
            Dictionary with all Tier 1 results
        """
        if not self.config.enabled:
            return {'enabled': False}

        if self.logger:
            self.logger.start_section("TIER1_EXPLORATORY")

        results = {'enabled': True}

        # Descriptive statistics
        if self.config.descriptive_stats:
            results['descriptive_stats'] = self.compute_descriptive_stats(
                df, factor_columns, class_column
            )

        # Correlations
        if self.config.correlations:
            results['correlations_pearson'] = self.compute_correlations(
                df, factor_columns, outcome_column, method='pearson'
            )
            results['correlations_spearman'] = self.compute_correlations(
                df, factor_columns, outcome_column, method='spearman'
            )

        # Distribution analysis
        if self.config.distributions:
            results['distributions'] = self.analyze_distributions(
                df, factor_columns
            )

        if self.logger:
            n_sig_pearson = len([c for c in results.get('correlations_pearson', [])
                                if c.p_value and c.p_value < 0.05])
            self.logger.info("Tier 1 analysis complete", {
                'factors_analyzed': len(factor_columns),
                'significant_correlations_pearson': n_sig_pearson
            })
            self.logger.end_section()

        return results

    def compute_descriptive_stats(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        class_column: str = 'trade_class'
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics per trade class.

        Args:
            df: DataFrame with factors
            factor_columns: Columns to analyze
            class_column: Class column for grouping

        Returns:
            DataFrame with statistics
        """
        # Filter to valid factor columns
        valid_cols = [c for c in factor_columns
                     if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not valid_cols:
            return pd.DataFrame()

        stats_list = []

        for col in valid_cols:
            for class_val in df[class_column].unique():
                class_data = df[df[class_column] == class_val][col].dropna()

                # Convert boolean to int for numeric operations
                if class_data.dtype == bool:
                    class_data = class_data.astype(int)

                if len(class_data) > 0:
                    stats_list.append({
                        'factor': col,
                        'class': class_val,
                        'count': len(class_data),
                        'mean': class_data.mean(),
                        'std': class_data.std(),
                        'median': class_data.median(),
                        'min': class_data.min(),
                        'max': class_data.max(),
                        'q25': class_data.quantile(0.25),
                        'q75': class_data.quantile(0.75),
                        'skewness': class_data.skew(),
                        'kurtosis': class_data.kurtosis()
                    })

        return pd.DataFrame(stats_list)

    def compute_correlations(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str,
        method: str = 'pearson'
    ) -> List[CorrelationResult]:
        """
        Compute correlations between factors and outcome.

        Args:
            df: DataFrame with factors and outcome
            factor_columns: Factor columns to analyze
            outcome_column: Outcome column
            method: 'pearson' or 'spearman'

        Returns:
            List of CorrelationResult objects
        """
        results = []

        if outcome_column not in df.columns:
            return results

        outcome = df[outcome_column].dropna()

        for col in factor_columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Get paired non-null values
            mask = df[col].notna() & df[outcome_column].notna()
            x = df.loc[mask, col]
            y = df.loc[mask, outcome_column]

            if len(x) < 3:
                continue

            try:
                # Convert boolean to int to avoid numpy boolean subtract error
                if x.dtype == bool:
                    x = x.astype(int)
                if y.dtype == bool:
                    y = y.astype(int)

                if method == 'pearson':
                    r, p = stats.pearsonr(x, y)
                else:
                    r, p = stats.spearmanr(x, y)

                results.append(CorrelationResult(
                    factor=col,
                    correlation=float(r),
                    p_value=float(p),
                    method=method,
                    significant=p < 0.05,
                    n_samples=len(x)
                ))

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Correlation failed for {col}", {'error': str(e)})

        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x.correlation), reverse=True)

        return results

    def analyze_distributions(
        self,
        df: pd.DataFrame,
        factor_columns: List[str]
    ) -> Dict[str, Dict]:
        """
        Analyze factor distributions.

        Args:
            df: DataFrame with factors
            factor_columns: Columns to analyze

        Returns:
            Dictionary with distribution info per factor
        """
        distributions = {}

        for col in factor_columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            values = df[col].dropna()
            if len(values) < 10:
                continue

            # Convert boolean to int for numeric operations
            if values.dtype == bool:
                values = values.astype(int)

            # Basic stats
            dist_info = {
                'n': len(values),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'skewness': float(values.skew()),
                'kurtosis': float(values.kurtosis())
            }

            # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
            try:
                if len(values) < 5000:
                    if len(values) >= 20:
                        stat, p = stats.shapiro(values.sample(min(5000, len(values))))
                        dist_info['normality_test'] = 'shapiro'
                        dist_info['normality_stat'] = float(stat)
                        dist_info['normality_p'] = float(p)
                        dist_info['is_normal'] = p > 0.05
                else:
                    stat, p = stats.normaltest(values)
                    dist_info['normality_test'] = 'dagostino'
                    dist_info['normality_stat'] = float(stat)
                    dist_info['normality_p'] = float(p)
                    dist_info['is_normal'] = p > 0.05
            except Exception:
                dist_info['is_normal'] = None

            # Percentiles for histogram bins
            dist_info['percentiles'] = {
                'p5': float(values.quantile(0.05)),
                'p25': float(values.quantile(0.25)),
                'p50': float(values.quantile(0.50)),
                'p75': float(values.quantile(0.75)),
                'p95': float(values.quantile(0.95))
            }

            distributions[col] = dist_info

        return distributions

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Compute correlation matrix between factors.

        Args:
            df: DataFrame with factors
            factor_columns: Columns for matrix
            method: 'pearson' or 'spearman'

        Returns:
            Correlation matrix DataFrame
        """
        valid_cols = [c for c in factor_columns
                     if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not valid_cols:
            return pd.DataFrame()

        if method == 'pearson':
            return df[valid_cols].corr(method='pearson')
        else:
            return df[valid_cols].corr(method='spearman')

    def identify_highly_correlated(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of highly correlated factors.

        Args:
            df: DataFrame with factors
            factor_columns: Columns to check
            threshold: Correlation threshold

        Returns:
            List of (factor1, factor2, correlation) tuples
        """
        corr_matrix = self.compute_correlation_matrix(df, factor_columns)

        if corr_matrix.empty:
            return []

        pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) >= threshold:
                        pairs.append((col1, col2, corr))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    def get_summary(self, results: Dict) -> Dict:
        """
        Get summary of Tier 1 results.

        Args:
            results: Full Tier 1 results

        Returns:
            Summary dictionary
        """
        summary = {'tier': 1, 'name': 'Exploratory Analysis'}

        if 'correlations_pearson' in results:
            pearson = results['correlations_pearson']
            significant = [c for c in pearson if c.significant]
            summary['pearson_correlations'] = {
                'total': len(pearson),
                'significant': len(significant),
                'top_5': [{'factor': c.factor, 'r': round(c.correlation, 3)}
                         for c in pearson[:5]]
            }

        if 'correlations_spearman' in results:
            spearman = results['correlations_spearman']
            significant = [c for c in spearman if c.significant]
            summary['spearman_correlations'] = {
                'total': len(spearman),
                'significant': len(significant),
                'top_5': [{'factor': c.factor, 'r': round(c.correlation, 3)}
                         for c in spearman[:5]]
            }

        return summary
