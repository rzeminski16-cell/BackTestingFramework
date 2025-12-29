"""
Tier 2 Hypothesis Testing for Factor Analysis.

Provides regression analysis, ANOVA, chi-square, and other statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings

from ..config.factor_config import Tier2Config
from ..logging.audit_logger import AuditLogger
from .analysis_result import FactorResult, RegressionResult, ANOVAResult

# Try to import statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class Tier2Hypothesis:
    """
    Tier 2: Hypothesis Testing and Regression Analysis.

    Provides:
    - Logistic regression for P(good trade)
    - ANOVA / Kruskal-Wallis tests
    - Chi-square tests for categorical factors
    - Mann-Whitney U tests for pairwise comparisons
    """

    def __init__(
        self,
        config: Optional[Tier2Config] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize Tier2Hypothesis.

        Args:
            config: Tier 2 configuration
            logger: Optional audit logger
        """
        self.config = config or Tier2Config()
        self.logger = logger

    def analyze(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str = 'trade_class_numeric',
        class_column: str = 'trade_class'
    ) -> Dict:
        """
        Run complete Tier 2 analysis.

        Args:
            df: Enriched trade DataFrame
            factor_columns: List of factor column names
            outcome_column: Numeric outcome column (1=good, -1=bad)
            class_column: Categorical class column

        Returns:
            Dictionary with all Tier 2 results
        """
        if not self.config.enabled:
            return {'enabled': False}

        if self.logger:
            self.logger.start_section("TIER2_HYPOTHESIS")

        results = {'enabled': True}

        # Logistic regression
        if self.config.logistic_regression:
            results['logistic_regression'] = self.run_logistic_regression(
                df, factor_columns, outcome_column
            )

        # ANOVA
        if self.config.anova:
            results['anova'] = self.run_anova(
                df, factor_columns, class_column
            )

        # Kruskal-Wallis
        if self.config.kruskal_wallis:
            results['kruskal_wallis'] = self.run_kruskal_wallis(
                df, factor_columns, class_column
            )

        # Chi-square for categorical
        if self.config.chi_square:
            categorical_cols = [c for c in factor_columns
                               if c in df.columns and df[c].dtype == 'object']
            results['chi_square'] = self.run_chi_square(
                df, categorical_cols, class_column
            )

        # Mann-Whitney for pairwise
        if self.config.wilcoxon_mannwhitney:
            results['mann_whitney'] = self.run_mann_whitney(
                df, factor_columns, class_column
            )

        if self.logger:
            n_sig = 0
            if results.get('logistic_regression'):
                n_sig = len(results['logistic_regression'].get_significant_factors())
            self.logger.info("Tier 2 analysis complete", {
                'significant_factors_regression': n_sig
            })
            self.logger.end_section()

        return results

    def run_logistic_regression(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str
    ) -> Optional[RegressionResult]:
        """
        Run logistic regression.

        Args:
            df: DataFrame with factors and outcome
            factor_columns: Predictor columns
            outcome_column: Outcome column (should be 0/1 or convertible)

        Returns:
            RegressionResult or None
        """
        if not STATSMODELS_AVAILABLE:
            if self.logger:
                self.logger.warning("Statsmodels not available, skipping logistic regression")
            return None

        # Prepare data
        valid_cols = [c for c in factor_columns
                     if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not valid_cols:
            return None

        # Create binary outcome (good vs not good)
        df_clean = df.dropna(subset=[outcome_column] + valid_cols).copy()

        if len(df_clean) < len(valid_cols) + 10:
            if self.logger:
                self.logger.warning("Insufficient data for logistic regression")
            return None

        # Convert outcome to binary
        y = (df_clean[outcome_column] > 0).astype(int)  # 1 = good, 0 = not good
        X = df_clean[valid_cols]

        # Standardize predictors for numerical stability
        X_standardized = (X - X.mean()) / X.std()
        X_standardized = X_standardized.fillna(0)

        # Add constant
        X_const = sm.add_constant(X_standardized)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Logit(y, X_const)
                result = model.fit(disp=0, maxiter=200)

            factor_results = []
            for i, col in enumerate(valid_cols):
                # Get coefficient (skip constant at index 0)
                idx = i + 1
                coef = result.params.iloc[idx]
                se = result.bse.iloc[idx]
                p_val = result.pvalues.iloc[idx]

                # Confidence interval
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se

                factor_results.append(FactorResult(
                    factor_name=col,
                    coefficient=float(coef),
                    odds_ratio=float(np.exp(coef)),
                    std_error=float(se),
                    p_value=float(p_val),
                    ci_lower=float(ci_lower),
                    ci_upper=float(ci_upper)
                ))

            return RegressionResult(
                model_type='logistic',
                n_observations=len(y),
                n_factors=len(valid_cols),
                aic=float(result.aic),
                bic=float(result.bic),
                pseudo_r2=float(result.prsquared),
                converged=result.mle_retvals.get('converged', True),
                factor_results=factor_results
            )

        except Exception as e:
            if self.logger:
                self.logger.error("Logistic regression failed", {'error': str(e)})
            return None

    def run_anova(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        class_column: str
    ) -> List[ANOVAResult]:
        """
        Run ANOVA for each factor.

        Args:
            df: DataFrame with factors
            factor_columns: Factor columns to test
            class_column: Grouping column

        Returns:
            List of ANOVAResult objects
        """
        results = []

        for col in factor_columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            groups = []
            group_means = {}

            for class_val in df[class_column].dropna().unique():
                group_data = df[df[class_column] == class_val][col].dropna()
                if len(group_data) >= 2:
                    groups.append(group_data)
                    group_means[str(class_val)] = float(group_data.mean())

            if len(groups) < 2:
                continue

            try:
                f_stat, p_val = stats.f_oneway(*groups)

                # Effect size (eta-squared)
                total_mean = df[col].dropna().mean()
                ss_between = sum(len(g) * (g.mean() - total_mean) ** 2 for g in groups)
                ss_total = ((df[col].dropna() - total_mean) ** 2).sum()
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                results.append(ANOVAResult(
                    factor=col,
                    test_type='anova',
                    statistic=float(f_stat),
                    p_value=float(p_val),
                    effect_size=float(eta_squared),
                    group_means=group_means,
                    significant=p_val < 0.05
                ))

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"ANOVA failed for {col}", {'error': str(e)})

        return results

    def run_kruskal_wallis(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        class_column: str
    ) -> List[ANOVAResult]:
        """
        Run Kruskal-Wallis test (non-parametric ANOVA).

        Args:
            df: DataFrame with factors
            factor_columns: Factor columns to test
            class_column: Grouping column

        Returns:
            List of ANOVAResult objects
        """
        results = []

        for col in factor_columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            groups = []
            for class_val in df[class_column].dropna().unique():
                group_data = df[df[class_column] == class_val][col].dropna()
                if len(group_data) >= 2:
                    groups.append(group_data)

            if len(groups) < 2:
                continue

            try:
                h_stat, p_val = stats.kruskal(*groups)

                results.append(ANOVAResult(
                    factor=col,
                    test_type='kruskal_wallis',
                    statistic=float(h_stat),
                    p_value=float(p_val),
                    significant=p_val < 0.05
                ))

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Kruskal-Wallis failed for {col}", {'error': str(e)})

        return results

    def run_chi_square(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        class_column: str
    ) -> Dict[str, Dict]:
        """
        Run chi-square test for categorical factors.

        Args:
            df: DataFrame with factors
            categorical_columns: Categorical columns to test
            class_column: Outcome class column

        Returns:
            Dictionary with chi-square results per column
        """
        results = {}

        for col in categorical_columns:
            if col not in df.columns:
                continue

            try:
                contingency = pd.crosstab(df[col], df[class_column])

                if contingency.size < 4:
                    continue

                chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

                # Cramér's V for effect size
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                results[col] = {
                    'chi2': float(chi2),
                    'p_value': float(p_val),
                    'dof': int(dof),
                    'cramers_v': float(cramers_v),
                    'significant': p_val < 0.05,
                    'contingency_table': contingency.to_dict()
                }

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Chi-square failed for {col}", {'error': str(e)})

        return results

    def run_mann_whitney(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        class_column: str
    ) -> Dict[str, Dict]:
        """
        Run Mann-Whitney U test for pairwise comparisons (good vs bad).

        Args:
            df: DataFrame with factors
            factor_columns: Factor columns to test
            class_column: Class column

        Returns:
            Dictionary with Mann-Whitney results per factor
        """
        results = {}

        # Get good and bad trades
        good_mask = df[class_column] == 'good'
        bad_mask = df[class_column] == 'bad'

        for col in factor_columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            good_vals = df.loc[good_mask, col].dropna()
            bad_vals = df.loc[bad_mask, col].dropna()

            if len(good_vals) < 2 or len(bad_vals) < 2:
                continue

            try:
                u_stat, p_val = stats.mannwhitneyu(
                    good_vals, bad_vals, alternative='two-sided'
                )

                # Effect size (rank-biserial correlation)
                n1, n2 = len(good_vals), len(bad_vals)
                r = 1 - (2 * u_stat) / (n1 * n2)

                results[col] = {
                    'u_statistic': float(u_stat),
                    'p_value': float(p_val),
                    'effect_size': float(r),
                    'good_median': float(good_vals.median()),
                    'bad_median': float(bad_vals.median()),
                    'significant': p_val < 0.05
                }

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Mann-Whitney failed for {col}", {'error': str(e)})

        return results

    def get_summary(self, results: Dict) -> Dict:
        """
        Get summary of Tier 2 results.

        Args:
            results: Full Tier 2 results

        Returns:
            Summary dictionary
        """
        summary = {'tier': 2, 'name': 'Hypothesis Testing'}

        if results.get('logistic_regression'):
            reg = results['logistic_regression']
            sig_factors = reg.get_significant_factors()
            summary['logistic_regression'] = {
                'n_observations': reg.n_observations,
                'n_factors': reg.n_factors,
                'significant_factors': len(sig_factors),
                'top_significant': sig_factors[:5]
            }

        if results.get('anova'):
            anova = results['anova']
            sig = [a for a in anova if a.significant]
            summary['anova'] = {
                'total_tests': len(anova),
                'significant': len(sig)
            }

        return summary
