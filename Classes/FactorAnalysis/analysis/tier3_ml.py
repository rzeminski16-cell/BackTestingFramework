"""
Tier 3 Machine Learning Analysis for Factor Analysis.

Provides Random Forest feature importance, SHAP values, and Mutual Information.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings

from ..config.factor_config import Tier3Config
from ..logging.audit_logger import AuditLogger

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class FeatureImportance:
    """Feature importance result."""
    feature_name: str
    importance: float
    importance_std: float = 0.0
    rank: int = 0


@dataclass
class SHAPResult:
    """SHAP analysis result."""
    feature_name: str
    mean_abs_shap: float
    direction: str  # 'positive', 'negative', 'mixed'
    rank: int = 0


@dataclass
class MutualInfoResult:
    """Mutual Information result."""
    feature_name: str
    mi_score: float
    normalized_mi: float = 0.0
    rank: int = 0


@dataclass
class Tier3Result:
    """Complete Tier 3 analysis result."""
    enabled: bool = True
    n_observations: int = 0
    n_features: int = 0

    # Random Forest results
    rf_accuracy: Optional[float] = None
    rf_cv_accuracy: Optional[float] = None
    rf_cv_std: Optional[float] = None
    rf_feature_importances: List[FeatureImportance] = field(default_factory=list)

    # SHAP results
    shap_available: bool = False
    shap_results: List[SHAPResult] = field(default_factory=list)

    # Mutual Information results
    mi_results: List[MutualInfoResult] = field(default_factory=list)

    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=dict)

    def get_top_features(self, n: int = 10, method: str = 'rf') -> List[str]:
        """Get top N features by specified method."""
        if method == 'rf' and self.rf_feature_importances:
            sorted_feats = sorted(
                self.rf_feature_importances,
                key=lambda x: x.importance,
                reverse=True
            )
            return [f.feature_name for f in sorted_feats[:n]]
        elif method == 'shap' and self.shap_results:
            sorted_feats = sorted(
                self.shap_results,
                key=lambda x: x.mean_abs_shap,
                reverse=True
            )
            return [f.feature_name for f in sorted_feats[:n]]
        elif method == 'mi' and self.mi_results:
            sorted_feats = sorted(
                self.mi_results,
                key=lambda x: x.mi_score,
                reverse=True
            )
            return [f.feature_name for f in sorted_feats[:n]]
        return []

    def get_consensus_top_features(self, n: int = 10) -> List[str]:
        """Get top features that appear in multiple methods."""
        rf_top = set(self.get_top_features(n * 2, 'rf'))
        shap_top = set(self.get_top_features(n * 2, 'shap'))
        mi_top = set(self.get_top_features(n * 2, 'mi'))

        # Score by how many methods include the feature
        scores = {}
        for f in rf_top | shap_top | mi_top:
            scores[f] = sum([
                f in rf_top,
                f in shap_top,
                f in mi_top
            ])

        sorted_features = sorted(scores.items(), key=lambda x: -x[1])
        return [f for f, _ in sorted_features[:n]]


class Tier3ML:
    """
    Tier 3: Machine Learning-Based Analysis.

    Provides:
    - Random Forest feature importance
    - SHAP (SHapley Additive exPlanations) values
    - Mutual Information scores
    """

    def __init__(
        self,
        config: Optional[Tier3Config] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize Tier3ML.

        Args:
            config: Tier 3 configuration
            logger: Optional audit logger
        """
        self.config = config or Tier3Config()
        self.logger = logger
        self._model = None
        self._scaler = None

    def analyze(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str = 'trade_class_numeric'
    ) -> Tier3Result:
        """
        Run complete Tier 3 analysis.

        Args:
            df: Enriched trade DataFrame
            factor_columns: List of factor column names
            outcome_column: Numeric outcome column (1=good, -1=bad)

        Returns:
            Tier3Result with all ML analysis results
        """
        if not self.config.enabled:
            return Tier3Result(enabled=False)

        if not SKLEARN_AVAILABLE:
            if self.logger:
                self.logger.warning("scikit-learn not available, skipping Tier 3 analysis")
            return Tier3Result(enabled=False)

        if self.logger:
            self.logger.start_section("TIER3_ML")

        result = Tier3Result(enabled=True)

        # Prepare data
        valid_cols = [c for c in factor_columns
                     if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not valid_cols:
            if self.logger:
                self.logger.warning("No valid numeric columns for ML analysis")
            return result

        # Clean data
        df_clean = df.dropna(subset=[outcome_column] + valid_cols).copy()

        if len(df_clean) < self.config.min_samples:
            if self.logger:
                self.logger.warning(
                    f"Insufficient samples for ML analysis: {len(df_clean)} < {self.config.min_samples}"
                )
            return result

        # Convert outcome to binary
        y = (df_clean[outcome_column] > 0).astype(int)
        X = df_clean[valid_cols]

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self._scaler.fit_transform(X),
            columns=valid_cols,
            index=X.index
        )

        result.n_observations = len(y)
        result.n_features = len(valid_cols)

        # Random Forest feature importance
        if self.config.random_forest:
            rf_result = self._run_random_forest(X_scaled, y, valid_cols)
            result.rf_accuracy = rf_result.get('accuracy')
            result.rf_cv_accuracy = rf_result.get('cv_accuracy')
            result.rf_cv_std = rf_result.get('cv_std')
            result.rf_feature_importances = rf_result.get('importances', [])
            result.model_params = rf_result.get('params', {})

        # SHAP analysis
        if self.config.shap and SHAP_AVAILABLE and self._model is not None:
            result.shap_available = True
            result.shap_results = self._run_shap_analysis(X_scaled, valid_cols)
        elif self.config.shap and not SHAP_AVAILABLE:
            if self.logger:
                self.logger.warning("SHAP library not available")

        # Mutual Information
        if self.config.mutual_information:
            result.mi_results = self._run_mutual_information(X_scaled, y, valid_cols)

        if self.logger:
            self.logger.info("Tier 3 ML analysis complete", {
                'n_observations': result.n_observations,
                'n_features': result.n_features,
                'rf_accuracy': result.rf_accuracy,
                'shap_computed': result.shap_available
            })
            self.logger.end_section()

        return result

    def _run_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> Dict:
        """Run Random Forest analysis."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Train Random Forest
                self._model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )

                self._model.fit(X, y)

                # Get accuracy
                accuracy = self._model.score(X, y)

                # Cross-validation
                cv_scores = cross_val_score(
                    self._model, X, y,
                    cv=min(5, len(y) // 10),
                    scoring='accuracy'
                )

                # Feature importances
                importances = self._model.feature_importances_

                # Get importance std from tree estimators
                importance_std = np.std([
                    tree.feature_importances_
                    for tree in self._model.estimators_
                ], axis=0)

                # Create sorted results
                importance_results = []
                for i, (feat, imp, std) in enumerate(
                    sorted(
                        zip(feature_names, importances, importance_std),
                        key=lambda x: -x[1]
                    )
                ):
                    importance_results.append(FeatureImportance(
                        feature_name=feat,
                        importance=float(imp),
                        importance_std=float(std),
                        rank=i + 1
                    ))

                return {
                    'accuracy': float(accuracy),
                    'cv_accuracy': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'importances': importance_results,
                    'params': {
                        'n_estimators': self.config.n_estimators,
                        'max_depth': self.config.max_depth,
                        'min_samples_leaf': self.config.min_samples_leaf
                    }
                }

        except Exception as e:
            if self.logger:
                self.logger.error("Random Forest analysis failed", {'error': str(e)})
            return {}

    def _run_shap_analysis(
        self,
        X: pd.DataFrame,
        feature_names: List[str]
    ) -> List[SHAPResult]:
        """Run SHAP analysis on trained model."""
        if self._model is None:
            return []

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Use TreeExplainer for Random Forest
                explainer = shap.TreeExplainer(self._model)

                # Calculate SHAP values (sample if too large)
                if len(X) > 1000:
                    X_sample = X.sample(n=1000, random_state=42)
                else:
                    X_sample = X

                shap_values = explainer.shap_values(X_sample)

                # For binary classification, shap_values is a list [class0, class1]
                # Use class 1 (positive/good trade) SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                # Compute mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

                # Determine direction (positive = higher value -> higher P(good))
                mean_shap = shap_values.mean(axis=0)

                results = []
                for i, (feat, mas, ms) in enumerate(
                    sorted(
                        zip(feature_names, mean_abs_shap, mean_shap),
                        key=lambda x: -x[1]
                    )
                ):
                    if ms > 0.01:
                        direction = 'positive'
                    elif ms < -0.01:
                        direction = 'negative'
                    else:
                        direction = 'mixed'

                    results.append(SHAPResult(
                        feature_name=feat,
                        mean_abs_shap=float(mas),
                        direction=direction,
                        rank=i + 1
                    ))

                return results

        except Exception as e:
            if self.logger:
                self.logger.error("SHAP analysis failed", {'error': str(e)})
            return []

    def _run_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> List[MutualInfoResult]:
        """Run Mutual Information analysis."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Calculate mutual information
                mi_scores = mutual_info_classif(
                    X, y,
                    discrete_features=False,
                    random_state=42
                )

                # Normalize by max MI
                max_mi = mi_scores.max() if mi_scores.max() > 0 else 1
                normalized_mi = mi_scores / max_mi

                results = []
                for i, (feat, mi, norm_mi) in enumerate(
                    sorted(
                        zip(feature_names, mi_scores, normalized_mi),
                        key=lambda x: -x[1]
                    )
                ):
                    results.append(MutualInfoResult(
                        feature_name=feat,
                        mi_score=float(mi),
                        normalized_mi=float(norm_mi),
                        rank=i + 1
                    ))

                return results

        except Exception as e:
            if self.logger:
                self.logger.error("Mutual Information analysis failed", {'error': str(e)})
            return []

    def get_feature_importance_df(self, result: Tier3Result) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.

        Args:
            result: Tier3Result from analyze()

        Returns:
            DataFrame with feature importance from all methods
        """
        data = []

        # RF importances
        rf_dict = {f.feature_name: f.importance for f in result.rf_feature_importances}

        # SHAP values
        shap_dict = {f.feature_name: f.mean_abs_shap for f in result.shap_results}

        # MI scores
        mi_dict = {f.feature_name: f.mi_score for f in result.mi_results}

        # Combine all features
        all_features = set(rf_dict.keys()) | set(shap_dict.keys()) | set(mi_dict.keys())

        for feat in all_features:
            data.append({
                'feature': feat,
                'rf_importance': rf_dict.get(feat, np.nan),
                'shap_importance': shap_dict.get(feat, np.nan),
                'mi_score': mi_dict.get(feat, np.nan)
            })

        df = pd.DataFrame(data)

        # Add composite rank
        for col in ['rf_importance', 'shap_importance', 'mi_score']:
            if col in df.columns:
                df[f'{col}_rank'] = df[col].rank(ascending=False, method='min')

        rank_cols = [c for c in df.columns if c.endswith('_rank')]
        if rank_cols:
            df['avg_rank'] = df[rank_cols].mean(axis=1)
            df = df.sort_values('avg_rank')

        return df

    def get_summary(self, result: Tier3Result) -> Dict:
        """
        Get summary of Tier 3 results.

        Args:
            result: Tier3Result from analyze()

        Returns:
            Summary dictionary
        """
        summary = {
            'tier': 3,
            'name': 'Machine Learning Analysis',
            'enabled': result.enabled,
            'n_observations': result.n_observations,
            'n_features': result.n_features
        }

        if result.rf_accuracy is not None:
            summary['random_forest'] = {
                'accuracy': result.rf_accuracy,
                'cv_accuracy': result.rf_cv_accuracy,
                'cv_std': result.rf_cv_std,
                'top_5_features': [
                    f.feature_name for f in result.rf_feature_importances[:5]
                ]
            }

        if result.shap_results:
            summary['shap'] = {
                'computed': True,
                'top_5_features': [
                    f.feature_name for f in result.shap_results[:5]
                ]
            }

        if result.mi_results:
            summary['mutual_information'] = {
                'top_5_features': [
                    f.feature_name for f in result.mi_results[:5]
                ]
            }

        # Consensus features
        consensus = result.get_consensus_top_features(5)
        if consensus:
            summary['consensus_top_features'] = consensus

        return summary
