"""
Interaction Analyzer for Factor Analysis.

Analyzes factor interactions and synergies to identify:
- Two-way and higher-order interactions
- Synergistic and antagonistic factor combinations
- Conditional effects
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
from itertools import combinations

from ..logging.audit_logger import AuditLogger


@dataclass
class Interaction:
    """Represents an interaction between factors."""
    factors: Tuple[str, ...]
    interaction_strength: float
    interaction_type: str  # 'synergistic', 'antagonistic', 'additive'
    p_value: float
    effect_size: float
    sample_size: int

    def __str__(self) -> str:
        return f"{' x '.join(self.factors)}: {self.interaction_type} ({self.interaction_strength:.3f})"


@dataclass
class InteractionResult:
    """Result of interaction analysis."""
    n_factors: int
    n_interactions_tested: int
    n_significant_interactions: int
    interactions: List[Interaction] = field(default_factory=list)
    interaction_matrix: Optional[pd.DataFrame] = None

    def get_top_interactions(self, n: int = 10) -> List[Interaction]:
        """Get top N interactions by strength."""
        sorted_interactions = sorted(
            self.interactions,
            key=lambda x: abs(x.interaction_strength),
            reverse=True
        )
        return sorted_interactions[:n]

    def get_synergistic(self) -> List[Interaction]:
        """Get synergistic interactions."""
        return [i for i in self.interactions if i.interaction_type == 'synergistic']

    def get_antagonistic(self) -> List[Interaction]:
        """Get antagonistic interactions."""
        return [i for i in self.interactions if i.interaction_type == 'antagonistic']


class InteractionAnalyzer:
    """
    Analyzes interactions between factors.

    Methods:
    - Two-way interaction detection
    - Synergy/antagonism classification
    - Conditional effect analysis
    - Interaction visualization data
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_cell_size: int = 10,
        max_interactions: int = 100,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize InteractionAnalyzer.

        Args:
            alpha: Significance level
            min_cell_size: Minimum observations per cell
            max_interactions: Maximum interactions to test
            logger: Optional audit logger
        """
        self.alpha = alpha
        self.min_cell_size = min_cell_size
        self.max_interactions = max_interactions
        self.logger = logger

    def analyze_interactions(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str = 'trade_class'
    ) -> InteractionResult:
        """
        Analyze two-way interactions between factors.

        Args:
            df: Trade DataFrame
            factor_columns: List of factor columns
            outcome_column: Outcome column

        Returns:
            InteractionResult with detected interactions
        """
        if self.logger:
            self.logger.start_section("INTERACTION_ANALYSIS")

        # Filter to numeric factors
        valid_factors = [
            c for c in factor_columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if len(valid_factors) < 2:
            if self.logger:
                self.logger.warning("Insufficient factors for interaction analysis")
            return InteractionResult(
                n_factors=len(valid_factors),
                n_interactions_tested=0,
                n_significant_interactions=0
            )

        interactions = []
        n_tested = 0

        # Test pairwise interactions
        pairs = list(combinations(valid_factors, 2))

        # Limit number of pairs
        if len(pairs) > self.max_interactions:
            pairs = pairs[:self.max_interactions]

        for f1, f2 in pairs:
            interaction = self._test_interaction(df, f1, f2, outcome_column)
            if interaction is not None:
                interactions.append(interaction)
            n_tested += 1

        # Count significant
        n_significant = sum(1 for i in interactions if i.p_value < self.alpha)

        # Create interaction matrix
        matrix = self._create_interaction_matrix(valid_factors, interactions)

        result = InteractionResult(
            n_factors=len(valid_factors),
            n_interactions_tested=n_tested,
            n_significant_interactions=n_significant,
            interactions=sorted(interactions, key=lambda x: abs(x.interaction_strength), reverse=True),
            interaction_matrix=matrix
        )

        if self.logger:
            self.logger.info("Interaction analysis complete", {
                'factors': len(valid_factors),
                'tested': n_tested,
                'significant': n_significant
            })
            self.logger.end_section()

        return result

    def _test_interaction(
        self,
        df: pd.DataFrame,
        factor1: str,
        factor2: str,
        outcome_column: str
    ) -> Optional[Interaction]:
        """Test interaction between two factors."""
        # Create binary splits at median
        f1_median = df[factor1].median()
        f2_median = df[factor2].median()

        df_temp = df.copy()
        df_temp['_f1_high'] = df_temp[factor1] >= f1_median
        df_temp['_f2_high'] = df_temp[factor2] >= f2_median

        # Create 2x2 contingency
        cells = {}
        for f1_h in [True, False]:
            for f2_h in [True, False]:
                mask = (df_temp['_f1_high'] == f1_h) & (df_temp['_f2_high'] == f2_h)
                cell_df = df_temp[mask]

                if len(cell_df) < self.min_cell_size:
                    return None

                good_rate = (cell_df[outcome_column] == 'good').mean()
                cells[(f1_h, f2_h)] = {
                    'count': len(cell_df),
                    'good_rate': good_rate
                }

        # Calculate interaction effect
        # Interaction = (HH + LL) - (HL + LH)
        # Positive = synergistic (both high/both low is better)
        # Negative = antagonistic (mixed is better)

        hh = cells[(True, True)]['good_rate']
        ll = cells[(False, False)]['good_rate']
        hl = cells[(True, False)]['good_rate']
        lh = cells[(False, True)]['good_rate']

        # Main effects
        f1_effect = ((hh + hl) / 2) - ((lh + ll) / 2)
        f2_effect = ((hh + lh) / 2) - ((hl + ll) / 2)

        # Interaction effect
        # Expected under additivity: hh_expected = base + f1_effect + f2_effect
        base = ll
        expected_hh = base + f1_effect + f2_effect
        interaction_effect = hh - expected_hh

        # Statistical test (permutation-based)
        p_value = self._permutation_test_interaction(
            df_temp, factor1, factor2, outcome_column, interaction_effect
        )

        # Classify interaction type
        if abs(interaction_effect) < 0.02:
            interaction_type = 'additive'
        elif interaction_effect > 0:
            interaction_type = 'synergistic'
        else:
            interaction_type = 'antagonistic'

        # Effect size (standardized)
        pooled_std = df[outcome_column].map({'good': 1, 'bad': 0}).std()
        effect_size = interaction_effect / pooled_std if pooled_std > 0 else 0

        return Interaction(
            factors=(factor1, factor2),
            interaction_strength=float(interaction_effect),
            interaction_type=interaction_type,
            p_value=float(p_value),
            effect_size=float(effect_size),
            sample_size=len(df_temp)
        )

    def _permutation_test_interaction(
        self,
        df: pd.DataFrame,
        factor1: str,
        factor2: str,
        outcome_column: str,
        observed_effect: float,
        n_permutations: int = 1000
    ) -> float:
        """Permutation test for interaction significance."""
        # Simplify by permuting outcome labels
        outcomes = df[outcome_column].values.copy()
        f1_median = df[factor1].median()
        f2_median = df[factor2].median()

        f1_high = (df[factor1] >= f1_median).values
        f2_high = (df[factor2] >= f2_median).values

        perm_effects = []

        for _ in range(n_permutations):
            np.random.shuffle(outcomes)

            # Calculate interaction effect
            hh_rate = (outcomes[(f1_high) & (f2_high)] == 'good').mean()
            ll_rate = (outcomes[(~f1_high) & (~f2_high)] == 'good').mean()
            hl_rate = (outcomes[(f1_high) & (~f2_high)] == 'good').mean()
            lh_rate = (outcomes[(~f1_high) & (f2_high)] == 'good').mean()

            f1_effect = ((hh_rate + hl_rate) / 2) - ((lh_rate + ll_rate) / 2)
            f2_effect = ((hh_rate + lh_rate) / 2) - ((hl_rate + ll_rate) / 2)

            expected_hh = ll_rate + f1_effect + f2_effect
            perm_effect = hh_rate - expected_hh
            perm_effects.append(perm_effect)

        # Two-tailed p-value
        perm_effects = np.array(perm_effects)
        p_value = (np.abs(perm_effects) >= np.abs(observed_effect)).mean()

        return max(p_value, 1 / n_permutations)  # Prevent p=0

    def _create_interaction_matrix(
        self,
        factors: List[str],
        interactions: List[Interaction]
    ) -> pd.DataFrame:
        """Create interaction strength matrix."""
        n = len(factors)
        matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=factors,
            columns=factors
        )

        for inter in interactions:
            f1, f2 = inter.factors
            matrix.loc[f1, f2] = inter.interaction_strength
            matrix.loc[f2, f1] = inter.interaction_strength

        return matrix

    def analyze_conditional_effects(
        self,
        df: pd.DataFrame,
        target_factor: str,
        conditioning_factor: str,
        outcome_column: str = 'trade_class',
        n_quantiles: int = 4
    ) -> Dict:
        """
        Analyze effect of target factor conditioned on another factor.

        Args:
            df: Trade DataFrame
            target_factor: Factor to analyze
            conditioning_factor: Factor to condition on
            outcome_column: Outcome column
            n_quantiles: Number of quantiles for conditioning

        Returns:
            Conditional effect analysis results
        """
        # Create quantiles of conditioning factor
        df_temp = df.copy()
        df_temp['_cond_quantile'] = pd.qcut(
            df_temp[conditioning_factor],
            q=n_quantiles,
            labels=[f"Q{i+1}" for i in range(n_quantiles)],
            duplicates='drop'
        )

        results = {
            'target_factor': target_factor,
            'conditioning_factor': conditioning_factor,
            'n_quantiles': n_quantiles,
            'conditional_effects': []
        }

        baseline_rate = (df[outcome_column] == 'good').mean()

        for q in df_temp['_cond_quantile'].unique():
            q_df = df_temp[df_temp['_cond_quantile'] == q]

            if len(q_df) < 20:
                continue

            # Correlate target factor with outcome within this quantile
            target_vals = q_df[target_factor].values
            outcome_vals = (q_df[outcome_column] == 'good').astype(int).values

            # Point-biserial correlation
            corr, p_value = stats.pointbiserialr(outcome_vals, target_vals)

            # Split by target factor median
            target_median = q_df[target_factor].median()
            high_mask = q_df[target_factor] >= target_median
            low_mask = q_df[target_factor] < target_median

            high_good_rate = (q_df.loc[high_mask, outcome_column] == 'good').mean()
            low_good_rate = (q_df.loc[low_mask, outcome_column] == 'good').mean()

            effect = high_good_rate - low_good_rate

            results['conditional_effects'].append({
                'quantile': str(q),
                'n_trades': len(q_df),
                'cond_factor_range': (
                    float(q_df[conditioning_factor].min()),
                    float(q_df[conditioning_factor].max())
                ),
                'correlation': float(corr),
                'p_value': float(p_value),
                'effect': float(effect),
                'high_good_rate': float(high_good_rate),
                'low_good_rate': float(low_good_rate)
            })

        # Check for effect heterogeneity
        effects = [e['effect'] for e in results['conditional_effects']]
        if len(effects) >= 2:
            results['effect_heterogeneity'] = float(np.std(effects))
            results['effect_range'] = (float(min(effects)), float(max(effects)))

            # Sign consistency
            positive_effects = sum(1 for e in effects if e > 0.01)
            negative_effects = sum(1 for e in effects if e < -0.01)
            results['effect_consistent'] = (
                positive_effects == len(effects) or
                negative_effects == len(effects)
            )

        return results

    def find_synergistic_pairs(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str = 'trade_class',
        min_synergy: float = 0.05
    ) -> List[Dict]:
        """
        Find synergistic factor pairs.

        Args:
            df: Trade DataFrame
            factor_columns: Factor columns to analyze
            outcome_column: Outcome column
            min_synergy: Minimum synergy threshold

        Returns:
            List of synergistic pair information
        """
        result = self.analyze_interactions(df, factor_columns, outcome_column)

        synergistic_pairs = []
        for inter in result.interactions:
            if inter.interaction_type == 'synergistic' and inter.interaction_strength >= min_synergy:
                synergistic_pairs.append({
                    'factors': inter.factors,
                    'synergy': inter.interaction_strength,
                    'p_value': inter.p_value,
                    'effect_size': inter.effect_size,
                    'description': (
                        f"When both {inter.factors[0]} and {inter.factors[1]} are "
                        f"{'high' if inter.interaction_strength > 0 else 'low'} together, "
                        f"good trade rate increases by {abs(inter.interaction_strength):.1%} "
                        "beyond additive expectation."
                    )
                })

        return sorted(synergistic_pairs, key=lambda x: x['synergy'], reverse=True)

    def find_antagonistic_pairs(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        outcome_column: str = 'trade_class',
        min_antagonism: float = 0.05
    ) -> List[Dict]:
        """
        Find antagonistic factor pairs.

        Args:
            df: Trade DataFrame
            factor_columns: Factor columns to analyze
            outcome_column: Outcome column
            min_antagonism: Minimum antagonism threshold

        Returns:
            List of antagonistic pair information
        """
        result = self.analyze_interactions(df, factor_columns, outcome_column)

        antagonistic_pairs = []
        for inter in result.interactions:
            if inter.interaction_type == 'antagonistic' and abs(inter.interaction_strength) >= min_antagonism:
                antagonistic_pairs.append({
                    'factors': inter.factors,
                    'antagonism': abs(inter.interaction_strength),
                    'p_value': inter.p_value,
                    'effect_size': inter.effect_size,
                    'description': (
                        f"When {inter.factors[0]} and {inter.factors[1]} are in "
                        f"opposite states (one high, one low), good trade rate is better "
                        f"than when both are high or both are low."
                    )
                })

        return sorted(antagonistic_pairs, key=lambda x: x['antagonism'], reverse=True)

    def get_interaction_summary(
        self,
        result: InteractionResult
    ) -> pd.DataFrame:
        """
        Get summary DataFrame of interactions.

        Args:
            result: InteractionResult from analyze_interactions

        Returns:
            Summary DataFrame
        """
        data = []
        for inter in result.interactions:
            data.append({
                'factor_1': inter.factors[0],
                'factor_2': inter.factors[1],
                'strength': inter.interaction_strength,
                'type': inter.interaction_type,
                'p_value': inter.p_value,
                'effect_size': inter.effect_size,
                'significant': inter.p_value < self.alpha
            })

        df = pd.DataFrame(data)
        if len(df) > 0:
            df = df.sort_values('strength', key=abs, ascending=False)

        return df
