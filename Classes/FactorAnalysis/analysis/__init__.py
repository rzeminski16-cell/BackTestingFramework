"""
Statistical analysis module for Factor Analysis.

Provides three tiers of analysis:
- Tier 1: Exploratory (descriptive stats, correlations, distributions)
- Tier 2: Hypothesis testing (regression, ANOVA, chi-square)
- Tier 3: ML-based (Random Forest, SHAP, Mutual Information)
"""

from .tier1_exploratory import Tier1Exploratory
from .tier2_hypothesis import Tier2Hypothesis
from .tier3_ml import Tier3ML
from .multiple_testing import MultipleTestingCorrector
from .analysis_result import AnalysisResult, FactorResult, CorrelationResult

__all__ = [
    'Tier1Exploratory',
    'Tier2Hypothesis',
    'Tier3ML',
    'MultipleTestingCorrector',
    'AnalysisResult',
    'FactorResult',
    'CorrelationResult',
]
