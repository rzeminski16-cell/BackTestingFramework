"""
Multiple Testing Correction for Factor Analysis.

Provides FDR (Benjamini-Hochberg) and Bonferroni corrections
to control for multiple hypothesis testing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..logging.audit_logger import AuditLogger


class CorrectionMethod(Enum):
    """Multiple testing correction methods."""
    NONE = "none"
    BONFERRONI = "bonferroni"
    FDR = "fdr"  # Benjamini-Hochberg
    HOLM = "holm"  # Holm-Bonferroni step-down


@dataclass
class CorrectedPValue:
    """A p-value with correction information."""
    factor_name: str
    original_p: float
    corrected_p: float
    significant_original: bool
    significant_corrected: bool
    correction_method: str


@dataclass
class MultipleTestingResult:
    """Result of multiple testing correction."""
    n_tests: int
    correction_method: str
    alpha: float
    n_significant_original: int
    n_significant_corrected: int
    corrected_pvalues: List[CorrectedPValue] = field(default_factory=list)

    def get_significant_factors(self, use_corrected: bool = True) -> List[str]:
        """Get list of significant factors."""
        if use_corrected:
            return [p.factor_name for p in self.corrected_pvalues
                   if p.significant_corrected]
        else:
            return [p.factor_name for p in self.corrected_pvalues
                   if p.significant_original]

    def get_false_discovery_estimate(self) -> int:
        """
        Estimate number of false discoveries if no correction applied.

        Under the null hypothesis, expected false positives = n_tests * alpha
        """
        return int(self.n_tests * self.alpha)


class MultipleTestingCorrector:
    """
    Applies multiple testing corrections to p-values.

    Supports:
    - Bonferroni correction (conservative)
    - Benjamini-Hochberg FDR (less conservative)
    - Holm-Bonferroni step-down (balanced)
    """

    def __init__(
        self,
        method: CorrectionMethod = CorrectionMethod.FDR,
        alpha: float = 0.05,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize MultipleTestingCorrector.

        Args:
            method: Correction method to use
            alpha: Significance level
            logger: Optional audit logger
        """
        self.method = method
        self.alpha = alpha
        self.logger = logger

    def correct(
        self,
        pvalues: Dict[str, float]
    ) -> MultipleTestingResult:
        """
        Apply multiple testing correction.

        Args:
            pvalues: Dictionary mapping factor names to p-values

        Returns:
            MultipleTestingResult with corrected p-values
        """
        if not pvalues:
            return MultipleTestingResult(
                n_tests=0,
                correction_method=self.method.value,
                alpha=self.alpha,
                n_significant_original=0,
                n_significant_corrected=0
            )

        n_tests = len(pvalues)
        factors = list(pvalues.keys())
        original_pvals = np.array([pvalues[f] for f in factors])

        # Apply correction based on method
        if self.method == CorrectionMethod.BONFERRONI:
            corrected_pvals = self._bonferroni(original_pvals)
        elif self.method == CorrectionMethod.FDR:
            corrected_pvals = self._fdr_bh(original_pvals)
        elif self.method == CorrectionMethod.HOLM:
            corrected_pvals = self._holm(original_pvals)
        else:
            corrected_pvals = original_pvals.copy()

        # Create results
        corrected_results = []
        for i, factor in enumerate(factors):
            corrected_results.append(CorrectedPValue(
                factor_name=factor,
                original_p=float(original_pvals[i]),
                corrected_p=float(corrected_pvals[i]),
                significant_original=original_pvals[i] < self.alpha,
                significant_corrected=corrected_pvals[i] < self.alpha,
                correction_method=self.method.value
            ))

        # Sort by corrected p-value
        corrected_results.sort(key=lambda x: x.corrected_p)

        n_sig_original = sum(1 for p in original_pvals if p < self.alpha)
        n_sig_corrected = sum(1 for p in corrected_pvals if p < self.alpha)

        result = MultipleTestingResult(
            n_tests=n_tests,
            correction_method=self.method.value,
            alpha=self.alpha,
            n_significant_original=n_sig_original,
            n_significant_corrected=n_sig_corrected,
            corrected_pvalues=corrected_results
        )

        if self.logger:
            self.logger.info("Multiple testing correction applied", {
                'method': self.method.value,
                'n_tests': n_tests,
                'significant_before': n_sig_original,
                'significant_after': n_sig_corrected
            })

        return result

    def _bonferroni(self, pvalues: np.ndarray) -> np.ndarray:
        """
        Apply Bonferroni correction.

        Most conservative: p_corrected = p * n_tests
        """
        n = len(pvalues)
        corrected = pvalues * n
        return np.minimum(corrected, 1.0)

    def _fdr_bh(self, pvalues: np.ndarray) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR correction.

        Controls false discovery rate rather than family-wise error rate.
        Less conservative than Bonferroni.
        """
        n = len(pvalues)

        # Sort p-values
        sorted_indices = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_indices]

        # Calculate BH critical values
        # For rank i, threshold is (i/n) * alpha
        # Adjusted p-value: p * n / rank
        adjusted = np.zeros(n)
        for i in range(n):
            rank = i + 1
            adjusted[i] = sorted_pvals[i] * n / rank

        # Ensure monotonicity (cumulative minimum from end)
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)

        # Unsort to original order
        result = np.zeros(n)
        result[sorted_indices] = adjusted

        return result

    def _holm(self, pvalues: np.ndarray) -> np.ndarray:
        """
        Apply Holm-Bonferroni step-down correction.

        Less conservative than Bonferroni but controls FWER.
        """
        n = len(pvalues)

        # Sort p-values
        sorted_indices = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_indices]

        # Apply Holm correction
        # For rank i (0-indexed), multiply by (n - i)
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[i] = sorted_pvals[i] * (n - i)

        # Ensure monotonicity (cumulative maximum from start)
        for i in range(1, n):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])

        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)

        # Unsort to original order
        result = np.zeros(n)
        result[sorted_indices] = adjusted

        return result

    def correct_from_tier1(
        self,
        tier1_results: Dict
    ) -> MultipleTestingResult:
        """
        Apply correction to Tier 1 correlation results.

        Args:
            tier1_results: Results from Tier1Exploratory.analyze()

        Returns:
            MultipleTestingResult with corrected p-values
        """
        pvalues = {}

        # Extract p-values from correlations
        if 'point_biserial' in tier1_results:
            for corr in tier1_results['point_biserial']:
                pvalues[f"pb_{corr.factor}"] = corr.p_value

        if 'spearman' in tier1_results:
            for factor, corr in tier1_results['spearman'].items():
                pvalues[f"sp_{factor}"] = corr.p_value

        return self.correct(pvalues)

    def correct_from_tier2(
        self,
        tier2_results: Dict
    ) -> MultipleTestingResult:
        """
        Apply correction to Tier 2 hypothesis test results.

        Args:
            tier2_results: Results from Tier2Hypothesis.analyze()

        Returns:
            MultipleTestingResult with corrected p-values
        """
        pvalues = {}

        # Extract p-values from logistic regression
        if tier2_results.get('logistic_regression'):
            reg = tier2_results['logistic_regression']
            for factor_result in reg.factor_results:
                pvalues[f"logit_{factor_result.factor_name}"] = factor_result.p_value

        # Extract from ANOVA
        if tier2_results.get('anova'):
            for anova_result in tier2_results['anova']:
                pvalues[f"anova_{anova_result.factor}"] = anova_result.p_value

        # Extract from Kruskal-Wallis
        if tier2_results.get('kruskal_wallis'):
            for kw_result in tier2_results['kruskal_wallis']:
                pvalues[f"kw_{kw_result.factor}"] = kw_result.p_value

        # Extract from chi-square
        if tier2_results.get('chi_square'):
            for factor, chi_result in tier2_results['chi_square'].items():
                pvalues[f"chi2_{factor}"] = chi_result['p_value']

        # Extract from Mann-Whitney
        if tier2_results.get('mann_whitney'):
            for factor, mw_result in tier2_results['mann_whitney'].items():
                pvalues[f"mw_{factor}"] = mw_result['p_value']

        return self.correct(pvalues)

    def get_summary_table(
        self,
        result: MultipleTestingResult
    ) -> List[Dict]:
        """
        Get summary table of corrected p-values.

        Args:
            result: MultipleTestingResult from correct()

        Returns:
            List of dictionaries for tabular display
        """
        return [
            {
                'factor': p.factor_name,
                'original_p': f"{p.original_p:.4f}",
                'corrected_p': f"{p.corrected_p:.4f}",
                'sig_original': '***' if p.significant_original else '',
                'sig_corrected': '***' if p.significant_corrected else ''
            }
            for p in result.corrected_pvalues
        ]

    def compare_methods(
        self,
        pvalues: Dict[str, float]
    ) -> Dict[str, MultipleTestingResult]:
        """
        Compare different correction methods.

        Args:
            pvalues: Dictionary mapping factor names to p-values

        Returns:
            Dictionary mapping method name to result
        """
        results = {}

        for method in [CorrectionMethod.NONE, CorrectionMethod.BONFERRONI,
                      CorrectionMethod.FDR, CorrectionMethod.HOLM]:
            corrector = MultipleTestingCorrector(
                method=method,
                alpha=self.alpha,
                logger=self.logger
            )
            results[method.value] = corrector.correct(pvalues)

        if self.logger:
            comparison = {
                method: res.n_significant_corrected
                for method, res in results.items()
            }
            self.logger.info("Multiple testing method comparison", comparison)

        return results


def calculate_required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n_tests: int = 1
) -> int:
    """
    Calculate required sample size accounting for multiple testing.

    Args:
        effect_size: Expected effect size (Cohen's d or similar)
        alpha: Target significance level
        power: Desired statistical power
        n_tests: Number of tests being conducted

    Returns:
        Required sample size per group
    """
    from scipy import stats

    # Adjust alpha for multiple testing (Bonferroni)
    adjusted_alpha = alpha / n_tests

    # Z-scores for adjusted alpha and power
    z_alpha = stats.norm.ppf(1 - adjusted_alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Sample size formula for two-sample t-test
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


def suggest_correction_method(
    n_tests: int,
    study_type: str = 'exploratory'
) -> CorrectionMethod:
    """
    Suggest appropriate correction method.

    Args:
        n_tests: Number of tests being conducted
        study_type: 'exploratory' or 'confirmatory'

    Returns:
        Recommended CorrectionMethod
    """
    if study_type == 'confirmatory':
        # Confirmatory studies need strict FWER control
        if n_tests <= 5:
            return CorrectionMethod.BONFERRONI
        else:
            return CorrectionMethod.HOLM
    else:
        # Exploratory studies can use less conservative FDR
        if n_tests <= 3:
            return CorrectionMethod.NONE
        elif n_tests <= 20:
            return CorrectionMethod.FDR
        else:
            return CorrectionMethod.FDR
