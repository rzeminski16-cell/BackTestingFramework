"""
The model ladder — conservative, interpretable-first, leakage-safe.

Models run in order: descriptive baseline → regularised linear/logistic →
shallow tree → constrained gradient-boosting (opt-in). Every model is a
scikit-learn ``Pipeline`` whose preprocessing (impute, scale, one-hot) is fit
*inside* each fold, so transforms never see the test split. Tuning uses small,
explicit, model-specific grids — not giant free-form sweeps — and boosted trees
use shrinkage + early stopping rather than unconstrained growth.

LogisticRegression is regularised by default (smaller ``C`` = stronger). The
HistGradientBoosting tier exposes ``learning_rate``, ``max_leaf_nodes``,
``l2_regularization`` and early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import LadderConfig


def _onehot() -> OneHotEncoder:
    # Compat across sklearn versions for the sparse argument name.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: List[str],
                       categorical_features: List[str]) -> ColumnTransformer:
    """Impute+scale numerics, impute+one-hot categoricals. Fit per fold."""
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", _onehot()),
        ]), categorical_features))
    return ColumnTransformer(transformers, remainder="drop")


@dataclass
class ModelSpec:
    """One rung of the ladder."""
    name: str
    tier: str
    pipeline: Pipeline
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    is_classification: bool = True
    calibratable: bool = False
    interpretable_linear: bool = False
    interpretable_tree: bool = False


class ModelLadder:
    """Builds the ordered set of model specs for a run."""

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 is_classification: bool):
        self.numeric = numeric_features
        self.categorical = categorical_features
        self.is_classification = is_classification

    def _pre(self) -> ColumnTransformer:
        return build_preprocessor(self.numeric, self.categorical)

    def _pipe(self, model) -> Pipeline:
        return Pipeline([("pre", self._pre()), ("model", model)])

    def build(self, ladder: LadderConfig) -> List[ModelSpec]:
        specs: List[ModelSpec] = []
        clf = self.is_classification

        if ladder.run_baseline:
            base = (DummyClassifier(strategy="prior") if clf
                    else DummyRegressor(strategy="mean"))
            specs.append(ModelSpec("Baseline", "baseline", self._pipe(base),
                                   {}, clf, calibratable=False))

        if ladder.run_linear:
            if clf:
                model = LogisticRegression(max_iter=2000, class_weight=None)
                grid = {"model__C": [0.05, 0.25, 1.0, 4.0]} if ladder.tune else {}
                specs.append(ModelSpec("Logistic (regularised)", "linear",
                                       self._pipe(model), grid, clf,
                                       calibratable=True, interpretable_linear=True))
            else:
                model = ElasticNet(max_iter=5000)
                grid = {"model__alpha": [0.01, 0.1, 1.0],
                        "model__l1_ratio": [0.1, 0.5, 0.9]} if ladder.tune else {}
                specs.append(ModelSpec("Elastic-net linear", "linear",
                                       self._pipe(model), grid, clf,
                                       interpretable_linear=True))

        if ladder.run_tree:
            if clf:
                model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                               class_weight=None, random_state=0)
            else:
                model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20,
                                              random_state=0)
            grid = {"model__max_depth": [2, 3, 4],
                    "model__min_samples_leaf": [10, 20, 40]} if ladder.tune else {}
            specs.append(ModelSpec("Shallow tree", "tree", self._pipe(model), grid,
                                   clf, calibratable=clf, interpretable_tree=True))

        if ladder.run_ensemble:
            if clf:
                model = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_leaf_nodes=15, max_depth=3,
                    min_samples_leaf=20, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=10, random_state=0)
            else:
                model = HistGradientBoostingRegressor(
                    learning_rate=0.05, max_leaf_nodes=15, max_depth=3,
                    min_samples_leaf=20, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=10, random_state=0)
            grid = {"model__learning_rate": [0.03, 0.05, 0.1],
                    "model__max_leaf_nodes": [7, 15, 31]} if ladder.tune else {}
            specs.append(ModelSpec("Gradient boosting (constrained)", "ensemble",
                                   self._pipe(model), grid, clf, calibratable=clf))

        return specs
