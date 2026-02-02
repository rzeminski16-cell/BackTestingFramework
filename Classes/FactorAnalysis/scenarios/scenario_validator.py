"""
Scenario Validator for Factor Analysis.

Validates discovered scenarios using:
- Statistical significance testing
- Bootstrap confidence intervals
- Out-of-sample validation
- Stability analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats

from .scenario_detector import Scenario, ScenarioType
from ..logging.audit_logger import AuditLogger


@dataclass
class ValidationResult:
    """Result of scenario validation."""
    scenario_name: str
    is_valid: bool
    confidence_level: float
    p_value: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    stability_score: float
    out_of_sample_lift: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict:
        """Get validation summary."""
        return {
            'scenario': self.scenario_name,
            'valid': self.is_valid,
            'confidence': f"{self.confidence_level:.1%}",
            'p_value': f"{self.p_value:.4f}",
            'ci': f"[{self.bootstrap_ci_lower:.3f}, {self.bootstrap_ci_upper:.3f}]",
            'stability': f"{self.stability_score:.2f}",
            'warnings': self.warnings
        }


class ScenarioValidator:
    """
    Validates trading scenarios for statistical robustness.

    Validation Methods:
    - Binomial test for significance
    - Bootstrap for confidence intervals
    - Time-based split for out-of-sample testing
    - Subgroup stability analysis
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        min_oos_trades: int = 20,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize ScenarioValidator.

        Args:
            alpha: Significance level
            n_bootstrap: Number of bootstrap samples
            min_oos_trades: Minimum trades for out-of-sample validation
            logger: Optional audit logger
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.min_oos_trades = min_oos_trades
        self.logger = logger

    def validate_scenario(
        self,
        df: pd.DataFrame,
        scenario: Scenario,
        outcome_column: str = 'trade_class',
        date_column: str = 'entry_date'
    ) -> ValidationResult:
        """
        Validate a single scenario.

        Args:
            df: Trade DataFrame
            scenario: Scenario to validate
            outcome_column: Outcome column
            date_column: Date column for time-based split

        Returns:
            ValidationResult with validation metrics
        """
        warnings = []

        # Get scenario trades
        scenario_mask = pd.Series([scenario.matches(row) for _, row in df.iterrows()], index=df.index)
        scenario_df = df[scenario_mask]
        n_scenario = len(scenario_df)

        if n_scenario < 10:
            return ValidationResult(
                scenario_name=scenario.name,
                is_valid=False,
                confidence_level=0.0,
                p_value=1.0,
                bootstrap_ci_lower=0.0,
                bootstrap_ci_upper=0.0,
                stability_score=0.0,
                warnings=["Insufficient trades for validation"]
            )

        # Calculate observed good trade rate
        observed_good_rate = (scenario_df[outcome_column] == 'good').mean()

        # Baseline good rate
        baseline_rate = (df[outcome_column] == 'good').mean()

        # 1. Statistical significance test (binomial)
        n_good = (scenario_df[outcome_column] == 'good').sum()

        if scenario.scenario_type == ScenarioType.BEST:
            # Test if good rate is significantly higher than baseline
            result = stats.binomtest(n_good, n_scenario, baseline_rate, alternative='greater')
            p_value = result.pvalue
        else:
            # Test if good rate is significantly lower than baseline
            result = stats.binomtest(n_good, n_scenario, baseline_rate, alternative='less')
            p_value = result.pvalue

        is_significant = p_value < self.alpha

        # 2. Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(
            scenario_df[outcome_column].values,
            baseline_rate
        )

        # 3. Stability analysis
        stability_score = self._calculate_stability(
            df, scenario, outcome_column
        )

        # 4. Out-of-sample validation (if date column available)
        oos_lift = None
        if date_column in df.columns:
            oos_lift = self._out_of_sample_validation(
                df, scenario, outcome_column, date_column
            )
            if oos_lift is not None and scenario.scenario_type == ScenarioType.BEST:
                if oos_lift < 1.0:
                    warnings.append("Out-of-sample lift below 1.0")

        # Determine overall validity
        is_valid = (
            is_significant and
            stability_score >= 0.5 and
            (ci_lower > baseline_rate if scenario.scenario_type == ScenarioType.BEST
             else ci_upper < baseline_rate)
        )

        # Add warnings
        if not is_significant:
            warnings.append(f"Not statistically significant (p={p_value:.4f})")
        if stability_score < 0.5:
            warnings.append(f"Low stability score ({stability_score:.2f})")

        confidence_level = 1 - p_value if is_valid else p_value

        return ValidationResult(
            scenario_name=scenario.name,
            is_valid=is_valid,
            confidence_level=float(confidence_level),
            p_value=float(p_value),
            bootstrap_ci_lower=float(ci_lower),
            bootstrap_ci_upper=float(ci_upper),
            stability_score=float(stability_score),
            out_of_sample_lift=oos_lift,
            warnings=warnings
        )

    def validate_all(
        self,
        df: pd.DataFrame,
        scenarios: List[Scenario],
        outcome_column: str = 'trade_class',
        date_column: str = 'entry_date'
    ) -> List[ValidationResult]:
        """
        Validate multiple scenarios.

        Args:
            df: Trade DataFrame
            scenarios: List of scenarios to validate
            outcome_column: Outcome column
            date_column: Date column

        Returns:
            List of ValidationResult objects
        """
        if self.logger:
            self.logger.start_section("SCENARIO_VALIDATION")

        results = []
        for scenario in scenarios:
            result = self.validate_scenario(df, scenario, outcome_column, date_column)
            results.append(result)

        valid_count = sum(1 for r in results if r.is_valid)

        if self.logger:
            self.logger.info("Scenario validation complete", {
                'total_scenarios': len(scenarios),
                'valid_scenarios': valid_count,
                'invalid_scenarios': len(scenarios) - valid_count
            })
            self.logger.end_section()

        return results

    def _bootstrap_ci(
        self,
        outcomes: np.ndarray,
        baseline_rate: float,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for good trade rate.

        Args:
            outcomes: Array of outcomes ('good' or other)
            baseline_rate: Baseline good rate
            confidence: Confidence level

        Returns:
            Tuple of (lower, upper) confidence bounds
        """
        n = len(outcomes)
        if n == 0:
            return 0.0, 0.0

        # Convert to binary
        is_good = (outcomes == 'good').astype(int)

        # Bootstrap samples
        boot_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(is_good, size=n, replace=True)
            boot_means.append(sample.mean())

        boot_means = np.array(boot_means)

        # Calculate percentiles
        alpha = 1 - confidence
        lower = np.percentile(boot_means, alpha / 2 * 100)
        upper = np.percentile(boot_means, (1 - alpha / 2) * 100)

        return lower, upper

    def _calculate_stability(
        self,
        df: pd.DataFrame,
        scenario: Scenario,
        outcome_column: str,
        n_splits: int = 5
    ) -> float:
        """
        Calculate scenario stability across random subsamples.

        Args:
            df: Trade DataFrame
            scenario: Scenario to check
            outcome_column: Outcome column
            n_splits: Number of random splits

        Returns:
            Stability score (0-1, higher is more stable)
        """
        # Get scenario trades
        scenario_mask = pd.Series([scenario.matches(row) for _, row in df.iterrows()], index=df.index)
        scenario_indices = df.index[scenario_mask].tolist()

        if len(scenario_indices) < n_splits * 2:
            return 0.0

        # Calculate baseline metrics
        baseline_good_rate = (df[outcome_column] == 'good').mean()

        # Calculate good rate in random subsamples
        good_rates = []
        sample_size = len(scenario_indices) // 2

        for _ in range(n_splits):
            sample_indices = np.random.choice(scenario_indices, size=sample_size, replace=False)
            sample_df = df.loc[sample_indices]
            good_rate = (sample_df[outcome_column] == 'good').mean()
            good_rates.append(good_rate)

        good_rates = np.array(good_rates)

        # Stability: low variance relative to effect size
        if scenario.scenario_type == ScenarioType.BEST:
            # For best scenarios, all samples should be above baseline
            above_baseline = (good_rates > baseline_good_rate).mean()
            consistency = above_baseline
        else:
            # For worst scenarios, all samples should be below baseline
            below_baseline = (good_rates < baseline_good_rate).mean()
            consistency = below_baseline

        # Also factor in coefficient of variation
        cv = good_rates.std() / good_rates.mean() if good_rates.mean() > 0 else 1.0
        cv_score = max(0, 1 - cv)  # Lower CV = higher score

        return float((consistency + cv_score) / 2)

    def _out_of_sample_validation(
        self,
        df: pd.DataFrame,
        scenario: Scenario,
        outcome_column: str,
        date_column: str,
        train_ratio: float = 0.7
    ) -> Optional[float]:
        """
        Validate scenario out-of-sample using time-based split.

        Args:
            df: Trade DataFrame
            scenario: Scenario to validate
            outcome_column: Outcome column
            date_column: Date column
            train_ratio: Fraction of data for training

        Returns:
            Out-of-sample lift, or None if insufficient data
        """
        # Sort by date
        df_sorted = df.sort_values(date_column).copy()

        # Split
        split_idx = int(len(df_sorted) * train_ratio)
        test_df = df_sorted.iloc[split_idx:]

        # Get scenario trades in test set
        test_mask = pd.Series([scenario.matches(row) for _, row in test_df.iterrows()], index=test_df.index)
        test_scenario_df = test_df[test_mask]

        if len(test_scenario_df) < self.min_oos_trades:
            return None

        # Calculate out-of-sample metrics
        oos_good_rate = (test_scenario_df[outcome_column] == 'good').mean()
        test_baseline = (test_df[outcome_column] == 'good').mean()

        if test_baseline == 0:
            return None

        return float(oos_good_rate / test_baseline)

    def cross_validate_scenario(
        self,
        df: pd.DataFrame,
        scenario: Scenario,
        outcome_column: str = 'trade_class',
        n_folds: int = 5
    ) -> Dict:
        """
        Cross-validate scenario across time folds.

        Args:
            df: Trade DataFrame
            scenario: Scenario to validate
            outcome_column: Outcome column
            n_folds: Number of cross-validation folds

        Returns:
            Cross-validation results
        """
        # Get scenario trades
        scenario_mask = pd.Series([scenario.matches(row) for _, row in df.iterrows()], index=df.index)
        scenario_indices = df.index[scenario_mask].tolist()

        if len(scenario_indices) < n_folds * 2:
            return {'error': 'Insufficient trades for cross-validation'}

        # Shuffle indices
        np.random.shuffle(scenario_indices)

        # Split into folds
        fold_size = len(scenario_indices) // n_folds
        fold_results = []

        baseline_rate = (df[outcome_column] == 'good').mean()

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(scenario_indices)
            fold_indices = scenario_indices[start_idx:end_idx]

            fold_df = df.loc[fold_indices]
            fold_good_rate = (fold_df[outcome_column] == 'good').mean()
            fold_lift = fold_good_rate / baseline_rate if baseline_rate > 0 else 1.0

            fold_results.append({
                'fold': i + 1,
                'n_trades': len(fold_df),
                'good_rate': float(fold_good_rate),
                'lift': float(fold_lift)
            })

        # Calculate summary statistics
        lifts = [f['lift'] for f in fold_results]

        return {
            'scenario': scenario.name,
            'n_folds': n_folds,
            'fold_results': fold_results,
            'mean_lift': float(np.mean(lifts)),
            'std_lift': float(np.std(lifts)),
            'min_lift': float(np.min(lifts)),
            'max_lift': float(np.max(lifts)),
            'all_positive': all(l > 1 for l in lifts) if scenario.scenario_type == ScenarioType.BEST else all(l < 1 for l in lifts)
        }

    def get_validation_summary(
        self,
        results: List[ValidationResult]
    ) -> pd.DataFrame:
        """
        Get summary DataFrame of validation results.

        Args:
            results: List of ValidationResult objects

        Returns:
            Summary DataFrame
        """
        data = []
        for r in results:
            data.append({
                'scenario': r.scenario_name,
                'valid': r.is_valid,
                'confidence': r.confidence_level,
                'p_value': r.p_value,
                'ci_lower': r.bootstrap_ci_lower,
                'ci_upper': r.bootstrap_ci_upper,
                'stability': r.stability_score,
                'oos_lift': r.out_of_sample_lift,
                'n_warnings': len(r.warnings)
            })

        return pd.DataFrame(data)
