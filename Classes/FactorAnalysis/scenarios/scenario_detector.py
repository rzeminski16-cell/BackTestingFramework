"""
Scenario Detector for Factor Analysis.

Identifies best and worst trading scenarios using:
- Binary mode: Above/below threshold conditions
- Clustering mode: Automatic discovery via k-means/hierarchical
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ..config.factor_config import ScenarioAnalysisConfig, ScenarioMode
from ..logging.audit_logger import AuditLogger

# Try to import sklearn for clustering
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ScenarioType(Enum):
    """Type of scenario."""
    BEST = "best"
    WORST = "worst"
    NEUTRAL = "neutral"


@dataclass
class ScenarioCondition:
    """A single condition defining a scenario."""
    factor: str
    operator: str  # '>', '<', '>=', '<=', '==', 'between'
    threshold: float
    upper_threshold: Optional[float] = None  # For 'between' operator

    def evaluate(self, value: float) -> bool:
        """Check if a value meets this condition."""
        if pd.isna(value):
            return False

        if self.operator == '>':
            return value > self.threshold
        elif self.operator == '<':
            return value < self.threshold
        elif self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '<=':
            return value <= self.threshold
        elif self.operator == '==':
            return value == self.threshold
        elif self.operator == 'between':
            return self.threshold <= value <= (self.upper_threshold or self.threshold)
        return False

    def __str__(self) -> str:
        if self.operator == 'between':
            return f"{self.factor} between [{self.threshold:.2f}, {self.upper_threshold:.2f}]"
        return f"{self.factor} {self.operator} {self.threshold:.2f}"


@dataclass
class Scenario:
    """A trading scenario defined by factor conditions."""
    name: str
    scenario_type: ScenarioType
    conditions: List[ScenarioCondition]
    n_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    good_trade_rate: float = 0.0
    confidence: float = 0.0
    support: float = 0.0  # Fraction of total trades
    lift: float = 1.0  # Improvement over baseline

    def matches(self, row: pd.Series) -> bool:
        """Check if a trade matches all conditions."""
        return all(cond.evaluate(row.get(cond.factor, np.nan))
                  for cond in self.conditions)

    def get_condition_string(self) -> str:
        """Get human-readable condition string."""
        return " AND ".join(str(c) for c in self.conditions)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.scenario_type.value,
            'conditions': [
                {
                    'factor': c.factor,
                    'operator': c.operator,
                    'threshold': c.threshold,
                    'upper_threshold': c.upper_threshold
                }
                for c in self.conditions
            ],
            'n_trades': self.n_trades,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return,
            'good_trade_rate': self.good_trade_rate,
            'confidence': self.confidence,
            'support': self.support,
            'lift': self.lift
        }


@dataclass
class ScenarioResult:
    """Result of scenario detection."""
    mode: str
    n_scenarios: int
    n_best: int
    n_worst: int
    best_scenarios: List[Scenario] = field(default_factory=list)
    worst_scenarios: List[Scenario] = field(default_factory=list)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    factor_importance: Dict[str, float] = field(default_factory=dict)

    def get_top_scenarios(
        self,
        scenario_type: ScenarioType,
        n: int = 5
    ) -> List[Scenario]:
        """Get top N scenarios by lift."""
        if scenario_type == ScenarioType.BEST:
            sorted_scenarios = sorted(
                self.best_scenarios,
                key=lambda s: s.lift,
                reverse=True
            )
        else:
            sorted_scenarios = sorted(
                self.worst_scenarios,
                key=lambda s: s.lift,
                reverse=False
            )
        return sorted_scenarios[:n]


class ScenarioDetector:
    """
    Detects best and worst trading scenarios.

    Modes:
    - Binary: Uses threshold-based conditions (e.g., RSI > 70)
    - Clustering: Uses unsupervised clustering to find natural groupings
    """

    def __init__(
        self,
        config: Optional[ScenarioAnalysisConfig] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize ScenarioDetector.

        Args:
            config: Scenario analysis configuration
            logger: Optional audit logger
        """
        self.config = config or ScenarioAnalysisConfig()
        self.logger = logger

    def detect(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str = 'trade_class',
        return_column: str = 'pl_pct'
    ) -> ScenarioResult:
        """
        Detect trading scenarios.

        Args:
            df: Enriched trade DataFrame
            factor_columns: List of factor column names
            outcome_column: Trade class column
            return_column: Return percentage column

        Returns:
            ScenarioResult with detected scenarios
        """
        # Check if scenario analysis is enabled (default to True if not specified)
        if hasattr(self.config, 'enabled') and not self.config.enabled:
            return ScenarioResult(mode='disabled', n_scenarios=0, n_best=0, n_worst=0)

        if self.logger:
            self.logger.start_section("SCENARIO_DETECTION")

        # Calculate baseline metrics
        baseline = self._calculate_baseline(df, outcome_column, return_column)

        # Filter to valid factor columns
        valid_cols = [c for c in factor_columns
                     if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not valid_cols:
            if self.logger:
                self.logger.warning("No valid factor columns for scenario detection")
            return ScenarioResult(
                mode=self.config.scenario_mode.value,
                n_scenarios=0,
                n_best=0,
                n_worst=0,
                baseline_metrics=baseline
            )

        # Detect scenarios based on mode
        if self.config.scenario_mode == ScenarioMode.BINARY:
            result = self._detect_binary_scenarios(
                df, valid_cols, outcome_column, return_column, baseline
            )
        elif self.config.scenario_mode == ScenarioMode.CLUSTERING:
            result = self._detect_clustering_scenarios(
                df, valid_cols, outcome_column, return_column, baseline
            )
        else:
            # Auto mode: try binary first, then clustering
            result = self._detect_binary_scenarios(
                df, valid_cols, outcome_column, return_column, baseline
            )
            if len(result.best_scenarios) + len(result.worst_scenarios) < 3:
                result = self._detect_clustering_scenarios(
                    df, valid_cols, outcome_column, return_column, baseline
                )

        if self.logger:
            self.logger.info("Scenario detection complete", {
                'mode': result.mode,
                'best_scenarios': result.n_best,
                'worst_scenarios': result.n_worst
            })
            self.logger.end_section()

        return result

    def _calculate_baseline(
        self,
        df: pd.DataFrame,
        outcome_column: str,
        return_column: str
    ) -> Dict[str, float]:
        """Calculate baseline metrics for comparison."""
        baseline = {
            'total_trades': len(df),
            'good_trade_rate': (df[outcome_column] == 'good').mean() if outcome_column in df.columns else 0.0,
            'win_rate': (df[return_column] > 0).mean() if return_column in df.columns else 0.0,
            'avg_return': df[return_column].mean() if return_column in df.columns else 0.0
        }
        return baseline

    def _detect_binary_scenarios(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str,
        return_column: str,
        baseline: Dict[str, float]
    ) -> ScenarioResult:
        """Detect scenarios using binary threshold conditions."""
        best_scenarios = []
        worst_scenarios = []

        # Single-factor scenarios
        for col in factor_columns:
            scenarios = self._find_single_factor_scenarios(
                df, col, outcome_column, return_column, baseline
            )
            for scenario in scenarios:
                if scenario.scenario_type == ScenarioType.BEST:
                    best_scenarios.append(scenario)
                elif scenario.scenario_type == ScenarioType.WORST:
                    worst_scenarios.append(scenario)

        # Two-factor combinations (if enabled)
        if self.config.max_factors >= 2 and len(factor_columns) >= 2:
            combo_scenarios = self._find_two_factor_scenarios(
                df, factor_columns, outcome_column, return_column, baseline
            )
            for scenario in combo_scenarios:
                if scenario.scenario_type == ScenarioType.BEST:
                    best_scenarios.append(scenario)
                elif scenario.scenario_type == ScenarioType.WORST:
                    worst_scenarios.append(scenario)

        # Sort by lift
        best_scenarios = sorted(best_scenarios, key=lambda s: s.lift, reverse=True)
        worst_scenarios = sorted(worst_scenarios, key=lambda s: s.lift)

        # Limit to top N
        best_scenarios = best_scenarios[:self.config.max_scenarios]
        worst_scenarios = worst_scenarios[:self.config.max_scenarios]

        return ScenarioResult(
            mode='binary',
            n_scenarios=len(best_scenarios) + len(worst_scenarios),
            n_best=len(best_scenarios),
            n_worst=len(worst_scenarios),
            best_scenarios=best_scenarios,
            worst_scenarios=worst_scenarios,
            baseline_metrics=baseline
        )

    def _find_single_factor_scenarios(
        self,
        df: pd.DataFrame,
        factor: str,
        outcome_column: str,
        return_column: str,
        baseline: Dict[str, float]
    ) -> List[Scenario]:
        """Find scenarios based on a single factor."""
        scenarios = []
        values = df[factor].dropna()

        if len(values) < 10:
            return scenarios

        # Try different thresholds
        percentiles = [10, 25, 50, 75, 90]
        thresholds = [np.percentile(values, p) for p in percentiles]

        for thresh in thresholds:
            # High values
            high_mask = df[factor] >= thresh
            if high_mask.sum() >= self.config.min_trades_per_scenario:
                scenario = self._evaluate_scenario(
                    df, high_mask,
                    [ScenarioCondition(factor, '>=', thresh)],
                    outcome_column, return_column, baseline,
                    f"{factor}_high_{thresh:.2f}"
                )
                if scenario:
                    scenarios.append(scenario)

            # Low values
            low_mask = df[factor] <= thresh
            if low_mask.sum() >= self.config.min_trades_per_scenario:
                scenario = self._evaluate_scenario(
                    df, low_mask,
                    [ScenarioCondition(factor, '<=', thresh)],
                    outcome_column, return_column, baseline,
                    f"{factor}_low_{thresh:.2f}"
                )
                if scenario:
                    scenarios.append(scenario)

        return scenarios

    def _find_two_factor_scenarios(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str,
        return_column: str,
        baseline: Dict[str, float]
    ) -> List[Scenario]:
        """Find scenarios based on two-factor combinations."""
        scenarios = []

        # Limit combinations to avoid explosion
        max_pairs = min(len(factor_columns) * (len(factor_columns) - 1) // 2, 50)
        pairs_checked = 0

        for i, f1 in enumerate(factor_columns):
            if pairs_checked >= max_pairs:
                break

            for f2 in factor_columns[i+1:]:
                if pairs_checked >= max_pairs:
                    break

                pairs_checked += 1

                # Try median splits
                m1 = df[f1].median()
                m2 = df[f2].median()

                # High-High
                mask_hh = (df[f1] >= m1) & (df[f2] >= m2)
                if mask_hh.sum() >= self.config.min_trades_per_scenario:
                    scenario = self._evaluate_scenario(
                        df, mask_hh,
                        [
                            ScenarioCondition(f1, '>=', m1),
                            ScenarioCondition(f2, '>=', m2)
                        ],
                        outcome_column, return_column, baseline,
                        f"{f1}_high_AND_{f2}_high"
                    )
                    if scenario:
                        scenarios.append(scenario)

                # High-Low
                mask_hl = (df[f1] >= m1) & (df[f2] < m2)
                if mask_hl.sum() >= self.config.min_trades_per_scenario:
                    scenario = self._evaluate_scenario(
                        df, mask_hl,
                        [
                            ScenarioCondition(f1, '>=', m1),
                            ScenarioCondition(f2, '<', m2)
                        ],
                        outcome_column, return_column, baseline,
                        f"{f1}_high_AND_{f2}_low"
                    )
                    if scenario:
                        scenarios.append(scenario)

                # Low-High
                mask_lh = (df[f1] < m1) & (df[f2] >= m2)
                if mask_lh.sum() >= self.config.min_trades_per_scenario:
                    scenario = self._evaluate_scenario(
                        df, mask_lh,
                        [
                            ScenarioCondition(f1, '<', m1),
                            ScenarioCondition(f2, '>=', m2)
                        ],
                        outcome_column, return_column, baseline,
                        f"{f1}_low_AND_{f2}_high"
                    )
                    if scenario:
                        scenarios.append(scenario)

                # Low-Low
                mask_ll = (df[f1] < m1) & (df[f2] < m2)
                if mask_ll.sum() >= self.config.min_trades_per_scenario:
                    scenario = self._evaluate_scenario(
                        df, mask_ll,
                        [
                            ScenarioCondition(f1, '<', m1),
                            ScenarioCondition(f2, '<', m2)
                        ],
                        outcome_column, return_column, baseline,
                        f"{f1}_low_AND_{f2}_low"
                    )
                    if scenario:
                        scenarios.append(scenario)

        return scenarios

    def _evaluate_scenario(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        conditions: List[ScenarioCondition],
        outcome_column: str,
        return_column: str,
        baseline: Dict[str, float],
        name: str
    ) -> Optional[Scenario]:
        """Evaluate a potential scenario."""
        subset = df[mask]
        n_trades = len(subset)

        if n_trades < self.config.min_trades_per_scenario:
            return None

        # Calculate metrics
        good_rate = (subset[outcome_column] == 'good').mean() if outcome_column in subset.columns else 0
        win_rate = (subset[return_column] > 0).mean() if return_column in subset.columns else 0
        avg_return = subset[return_column].mean() if return_column in subset.columns else 0

        # Calculate support and lift
        support = n_trades / len(df)
        baseline_good_rate = baseline.get('good_trade_rate', 0.5)
        lift = good_rate / baseline_good_rate if baseline_good_rate > 0 else 1.0

        # Determine scenario type
        if lift >= self.config.min_lift:
            scenario_type = ScenarioType.BEST
        elif lift <= (1 / self.config.min_lift):
            scenario_type = ScenarioType.WORST
        else:
            return None  # Neutral scenario, skip

        # Calculate confidence (statistical significance proxy)
        from scipy import stats
        n_good = (subset[outcome_column] == 'good').sum() if outcome_column in subset.columns else 0
        n_total = n_trades

        if n_total > 0 and baseline_good_rate > 0:
            # Binomial test for significant difference from baseline
            p_value = stats.binom_test(n_good, n_total, baseline_good_rate)
            confidence = 1 - p_value
        else:
            confidence = 0

        return Scenario(
            name=name,
            scenario_type=scenario_type,
            conditions=conditions,
            n_trades=n_trades,
            win_rate=float(win_rate),
            avg_return=float(avg_return),
            good_trade_rate=float(good_rate),
            confidence=float(confidence),
            support=float(support),
            lift=float(lift)
        )

    def _detect_clustering_scenarios(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str,
        return_column: str,
        baseline: Dict[str, float]
    ) -> ScenarioResult:
        """Detect scenarios using clustering."""
        if not SKLEARN_AVAILABLE:
            if self.logger:
                self.logger.warning("scikit-learn not available for clustering")
            return ScenarioResult(
                mode='clustering',
                n_scenarios=0,
                n_best=0,
                n_worst=0,
                baseline_metrics=baseline
            )

        # Prepare data
        df_clean = df.dropna(subset=factor_columns).copy()
        if len(df_clean) < self.config.n_clusters * 10:
            if self.logger:
                self.logger.warning("Insufficient data for clustering")
            return ScenarioResult(
                mode='clustering',
                n_scenarios=0,
                n_best=0,
                n_worst=0,
                baseline_metrics=baseline
            )

        X = df_clean[factor_columns].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run k-means clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=42,
                n_init=10
            )
            clusters = kmeans.fit_predict(X_scaled)

        df_clean['_cluster'] = clusters

        best_scenarios = []
        worst_scenarios = []

        baseline_good_rate = baseline.get('good_trade_rate', 0.5)

        for cluster_id in range(self.config.n_clusters):
            cluster_mask = df_clean['_cluster'] == cluster_id
            cluster_df = df_clean[cluster_mask]

            if len(cluster_df) < self.config.min_trades_per_scenario:
                continue

            # Calculate cluster metrics
            good_rate = (cluster_df[outcome_column] == 'good').mean()
            win_rate = (cluster_df[return_column] > 0).mean()
            avg_return = cluster_df[return_column].mean()
            support = len(cluster_df) / len(df_clean)
            lift = good_rate / baseline_good_rate if baseline_good_rate > 0 else 1.0

            # Generate conditions from cluster center
            conditions = self._cluster_to_conditions(
                kmeans.cluster_centers_[cluster_id],
                factor_columns,
                scaler,
                df_clean[factor_columns]
            )

            if lift >= self.config.min_lift:
                scenario_type = ScenarioType.BEST
            elif lift <= (1 / self.config.min_lift):
                scenario_type = ScenarioType.WORST
            else:
                continue

            scenario = Scenario(
                name=f"cluster_{cluster_id}",
                scenario_type=scenario_type,
                conditions=conditions,
                n_trades=len(cluster_df),
                win_rate=float(win_rate),
                avg_return=float(avg_return),
                good_trade_rate=float(good_rate),
                confidence=0.0,  # Would need additional validation
                support=float(support),
                lift=float(lift)
            )

            if scenario_type == ScenarioType.BEST:
                best_scenarios.append(scenario)
            else:
                worst_scenarios.append(scenario)

        # Calculate factor importance from cluster centers
        factor_importance = self._calculate_factor_importance_from_clusters(
            kmeans.cluster_centers_,
            factor_columns
        )

        return ScenarioResult(
            mode='clustering',
            n_scenarios=len(best_scenarios) + len(worst_scenarios),
            n_best=len(best_scenarios),
            n_worst=len(worst_scenarios),
            best_scenarios=sorted(best_scenarios, key=lambda s: s.lift, reverse=True),
            worst_scenarios=sorted(worst_scenarios, key=lambda s: s.lift),
            baseline_metrics=baseline,
            factor_importance=factor_importance
        )

    def _cluster_to_conditions(
        self,
        center: np.ndarray,
        factor_columns: List[str],
        scaler: 'StandardScaler',
        original_data: pd.DataFrame
    ) -> List[ScenarioCondition]:
        """Convert cluster center to interpretable conditions."""
        conditions = []

        # Inverse transform to original scale
        center_original = scaler.inverse_transform([center])[0]

        # Find most extreme factors (> 1 std from mean in scaled space)
        for i, (col, val, scaled_val) in enumerate(
            zip(factor_columns, center_original, center)
        ):
            if abs(scaled_val) > 0.5:  # Factor deviates from mean
                if scaled_val > 0:
                    conditions.append(ScenarioCondition(
                        factor=col,
                        operator='>=',
                        threshold=float(original_data[col].median())
                    ))
                else:
                    conditions.append(ScenarioCondition(
                        factor=col,
                        operator='<',
                        threshold=float(original_data[col].median())
                    ))

        # Limit to top conditions
        return conditions[:self.config.max_factors]

    def _calculate_factor_importance_from_clusters(
        self,
        centers: np.ndarray,
        factor_columns: List[str]
    ) -> Dict[str, float]:
        """Calculate factor importance based on cluster center variance."""
        # Variance of each factor across cluster centers
        variances = np.var(centers, axis=0)
        total_var = variances.sum()

        if total_var == 0:
            return {col: 0.0 for col in factor_columns}

        importance = variances / total_var

        return {
            col: float(imp)
            for col, imp in zip(factor_columns, importance)
        }

    def find_optimal_thresholds(
        self,
        df: pd.DataFrame,
        factor: str,
        outcome_column: str = 'trade_class',
        n_splits: int = 10
    ) -> Dict:
        """
        Find optimal threshold splits for a factor.

        Args:
            df: Trade DataFrame
            factor: Factor column to analyze
            outcome_column: Outcome column
            n_splits: Number of threshold splits to try

        Returns:
            Dictionary with optimal thresholds and metrics
        """
        values = df[factor].dropna()
        if len(values) < 20:
            return {'error': 'Insufficient data'}

        percentiles = np.linspace(10, 90, n_splits)
        thresholds = [np.percentile(values, p) for p in percentiles]

        results = []
        baseline_good_rate = (df[outcome_column] == 'good').mean()

        for thresh in thresholds:
            high_mask = df[factor] >= thresh
            low_mask = df[factor] < thresh

            high_good_rate = (df.loc[high_mask, outcome_column] == 'good').mean()
            low_good_rate = (df.loc[low_mask, outcome_column] == 'good').mean()

            separation = abs(high_good_rate - low_good_rate)

            results.append({
                'threshold': float(thresh),
                'high_count': int(high_mask.sum()),
                'low_count': int(low_mask.sum()),
                'high_good_rate': float(high_good_rate),
                'low_good_rate': float(low_good_rate),
                'separation': float(separation),
                'high_lift': float(high_good_rate / baseline_good_rate) if baseline_good_rate > 0 else 1.0,
                'low_lift': float(low_good_rate / baseline_good_rate) if baseline_good_rate > 0 else 1.0
            })

        # Find optimal threshold (max separation)
        best = max(results, key=lambda x: x['separation'])

        return {
            'factor': factor,
            'optimal_threshold': best['threshold'],
            'max_separation': best['separation'],
            'all_thresholds': results
        }
