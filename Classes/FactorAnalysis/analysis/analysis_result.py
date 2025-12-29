"""
Analysis Result containers for Factor Analysis.

Provides structured containers for analysis outputs.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    factor: str
    correlation: float
    p_value: float
    p_value_corrected: Optional[float] = None
    method: str = 'pearson'
    significant: bool = False
    n_samples: int = 0


@dataclass
class FactorResult:
    """Comprehensive result for a single factor."""
    factor_name: str

    # Descriptive stats
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    n_valid: int = 0
    n_missing: int = 0

    # Correlation
    pearson_r: Optional[float] = None
    pearson_p: Optional[float] = None
    spearman_r: Optional[float] = None
    spearman_p: Optional[float] = None

    # Regression (logistic)
    coefficient: Optional[float] = None
    odds_ratio: Optional[float] = None
    std_error: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    # ANOVA
    f_statistic: Optional[float] = None
    anova_p: Optional[float] = None

    # ML importance
    rf_importance: Optional[float] = None
    shap_importance: Optional[float] = None
    mutual_info: Optional[float] = None

    # Significance after correction
    significant_after_correction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    model_type: str
    n_observations: int
    n_factors: int
    aic: Optional[float] = None
    bic: Optional[float] = None
    pseudo_r2: Optional[float] = None
    converged: bool = True
    factor_results: List[FactorResult] = field(default_factory=list)

    def get_significant_factors(self, alpha: float = 0.05) -> List[str]:
        """Get list of significant factor names."""
        return [f.factor_name for f in self.factor_results
                if f.p_value is not None and f.p_value < alpha]


@dataclass
class ANOVAResult:
    """Result of ANOVA analysis."""
    factor: str
    test_type: str  # 'anova' or 'kruskal_wallis'
    statistic: float
    p_value: float
    p_value_corrected: Optional[float] = None
    effect_size: Optional[float] = None  # eta-squared
    group_means: Dict[str, float] = field(default_factory=dict)
    significant: bool = False


@dataclass
class FeatureImportance:
    """Feature importance from ML model."""
    factor: str
    importance: float
    rank: int
    method: str  # 'random_forest', 'shap', 'mutual_info'


@dataclass
class AnalysisResult:
    """
    Complete analysis result container.

    Contains results from all three tiers of analysis.
    """
    # Metadata
    analysis_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    profile_name: str = ""
    config_hash: str = ""

    # Data summary
    total_trades: int = 0
    good_trades: int = 0
    bad_trades: int = 0
    indeterminate_trades: int = 0
    total_factors: int = 0
    factors_analyzed: int = 0

    # Tier 1: Exploratory
    descriptive_stats: Optional[pd.DataFrame] = None
    correlations_pearson: List[CorrelationResult] = field(default_factory=list)
    correlations_spearman: List[CorrelationResult] = field(default_factory=list)

    # Tier 2: Hypothesis Testing
    logistic_regression: Optional[RegressionResult] = None
    anova_results: List[ANOVAResult] = field(default_factory=list)
    chi_square_results: Dict[str, Any] = field(default_factory=dict)

    # Tier 3: ML Analysis
    feature_importances: List[FeatureImportance] = field(default_factory=list)
    shap_values: Optional[pd.DataFrame] = None

    # Combined factor results
    factor_results: Dict[str, FactorResult] = field(default_factory=dict)

    # Key findings
    significant_factors: List[str] = field(default_factory=list)
    top_predictors: List[str] = field(default_factory=list)

    def add_factor_result(self, result: FactorResult) -> None:
        """Add or update a factor result."""
        self.factor_results[result.factor_name] = result

    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all factor results."""
        records = [f.to_dict() for f in self.factor_results.values()]
        return pd.DataFrame(records)

    def get_top_factors(self, n: int = 10, by: str = 'rf_importance') -> List[str]:
        """Get top N factors by specified metric."""
        factors = [(name, getattr(result, by, 0) or 0)
                  for name, result in self.factor_results.items()]
        factors.sort(key=lambda x: abs(x[1]), reverse=True)
        return [f[0] for f in factors[:n]]

    def get_significant_correlations(
        self,
        method: str = 'pearson',
        alpha: float = 0.05
    ) -> List[CorrelationResult]:
        """Get significant correlations."""
        correlations = self.correlations_pearson if method == 'pearson' else self.correlations_spearman
        return [c for c in correlations if c.p_value is not None and c.p_value < alpha]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': {
                'analysis_date': self.analysis_date,
                'profile_name': self.profile_name,
                'config_hash': self.config_hash
            },
            'summary': {
                'total_trades': self.total_trades,
                'good_trades': self.good_trades,
                'bad_trades': self.bad_trades,
                'indeterminate_trades': self.indeterminate_trades,
                'total_factors': self.total_factors,
                'factors_analyzed': self.factors_analyzed
            },
            'correlations': {
                'pearson': [{'factor': c.factor, 'r': c.correlation, 'p': c.p_value}
                           for c in self.correlations_pearson],
                'spearman': [{'factor': c.factor, 'r': c.correlation, 'p': c.p_value}
                            for c in self.correlations_spearman]
            },
            'regression': {
                'model_type': self.logistic_regression.model_type if self.logistic_regression else None,
                'n_obs': self.logistic_regression.n_observations if self.logistic_regression else None,
                'significant_factors': self.logistic_regression.get_significant_factors() if self.logistic_regression else []
            } if self.logistic_regression else None,
            'feature_importance': [
                {'factor': f.factor, 'importance': f.importance, 'method': f.method}
                for f in self.feature_importances
            ],
            'key_findings': {
                'significant_factors': self.significant_factors,
                'top_predictors': self.top_predictors
            },
            'factor_details': {name: result.to_dict()
                             for name, result in self.factor_results.items()}
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create from dictionary."""
        result = cls()
        result.analysis_date = data.get('metadata', {}).get('analysis_date', '')
        result.profile_name = data.get('metadata', {}).get('profile_name', '')

        summary = data.get('summary', {})
        result.total_trades = summary.get('total_trades', 0)
        result.good_trades = summary.get('good_trades', 0)
        result.bad_trades = summary.get('bad_trades', 0)
        result.indeterminate_trades = summary.get('indeterminate_trades', 0)

        return result
