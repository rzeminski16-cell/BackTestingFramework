"""
Exportable scoring function — a research artefact with a clear contract.

It bundles the selected feature list, the preprocessing recipe + fitted model
parameters (the sklearn ``Pipeline``), the probability-calibration behaviour, the
allow / reduce / block decision thresholds, an optional size-reduction mapping,
the schema the input must satisfy, and provenance tied to the source run package.

Deployment is out of scope, so this needs only to be reproducible, inspectable,
and easy to test against held-out examples — which :meth:`ScoringFunction.apply`
supports directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ScoringFunction:
    """Self-contained, inspectable trade-scoring contract."""
    model_run_id: str
    source_run_id: str
    target_kind: str
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    invert_score: bool                  # True for tail-loss targets (high == bad)
    allow_threshold: float
    block_threshold: float
    reduce_size_factor: float
    calibrated: bool
    model_repr: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _pipeline: Any = None               # fitted sklearn pipeline (not JSON-serialised)

    # -- scoring ------------------------------------------------------------ #
    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Model output for each row (probability for classifiers)."""
        if self._pipeline is None:
            raise RuntimeError("ScoringFunction has no fitted pipeline loaded.")
        model = self._pipeline
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X[self.feature_columns])
            classes = list(getattr(model, "classes_", [0, 1]))
            pos = classes.index(1) if 1 in classes else len(classes) - 1
            return proba[:, pos]
        return np.asarray(model.predict(X[self.feature_columns]), dtype=float)

    def good_score(self, X: pd.DataFrame) -> np.ndarray:
        s = self.raw_score(X)
        return (1.0 - s) if self.invert_score else s

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score rows and assign allow / reduce / block + a size multiplier."""
        score = self.good_score(X)
        decision = np.where(score >= self.allow_threshold, "allow",
                            np.where(score < self.block_threshold, "block", "reduce"))
        size = np.where(decision == "allow", 1.0,
                        np.where(decision == "block", 0.0, self.reduce_size_factor))
        return pd.DataFrame({"good_score": score, "decision": decision,
                             "size_multiplier": size}, index=X.index)

    # -- persistence -------------------------------------------------------- #
    def contract(self) -> Dict[str, Any]:
        """JSON-serialisable description (everything but the fitted weights)."""
        return {
            "artefact": "scoring_function",
            "version": "1.0",
            "model_run_id": self.model_run_id,
            "source_run_id": self.source_run_id,
            "target_kind": self.target_kind,
            "feature_columns": list(self.feature_columns),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "invert_score": self.invert_score,
            "thresholds": {
                "allow": self.allow_threshold,
                "block": self.block_threshold,
                "reduce_size_factor": self.reduce_size_factor,
            },
            "calibrated": self.calibrated,
            "model_repr": self.model_repr,
            "best_params": self.best_params,
            "provenance": self.provenance,
            "created_at": self.created_at,
            "model_file": "scoring_model.joblib",
        }

    def save(self, directory: str) -> Dict[str, str]:
        os.makedirs(directory, exist_ok=True)
        contract_path = os.path.join(directory, "scoring_function.json")
        with open(contract_path, "w", encoding="utf-8") as fh:
            json.dump(self.contract(), fh, indent=2, default=str)
        model_path = os.path.join(directory, "scoring_model.joblib")
        if self._pipeline is not None:
            try:
                import joblib
                joblib.dump(self._pipeline, model_path)
            except Exception:
                model_path = ""
        return {"scoring_function": contract_path, "scoring_model": model_path}

    @classmethod
    def load(cls, directory: str) -> "ScoringFunction":
        with open(os.path.join(directory, "scoring_function.json"), "r", encoding="utf-8") as fh:
            c = json.load(fh)
        thresholds = c.get("thresholds", {})
        sf = cls(
            model_run_id=c["model_run_id"], source_run_id=c["source_run_id"],
            target_kind=c["target_kind"], feature_columns=c["feature_columns"],
            numeric_features=c["numeric_features"], categorical_features=c["categorical_features"],
            invert_score=c["invert_score"], allow_threshold=thresholds.get("allow", 0.5),
            block_threshold=thresholds.get("block", 0.0),
            reduce_size_factor=thresholds.get("reduce_size_factor", 0.5),
            calibrated=c.get("calibrated", False), model_repr=c.get("model_repr", ""),
            best_params=c.get("best_params", {}), provenance=c.get("provenance", {}),
            created_at=c.get("created_at", ""),
        )
        model_path = os.path.join(directory, c.get("model_file", "scoring_model.joblib"))
        if os.path.isfile(model_path):
            try:
                import joblib
                sf._pipeline = joblib.load(model_path)
            except Exception:
                sf._pipeline = None
        return sf


def build_scoring_function(evaluation, config, feature_matrix,
                           train_scores: Optional[pd.Series] = None) -> ScoringFunction:
    """Construct a scoring function from a finalist evaluation + run config."""
    from .config import TargetKind

    invert = (config.target.primary == TargetKind.BINARY_TAIL_LOSS)
    scores = train_scores if train_scores is not None else evaluation.oos_predictions
    good = (1.0 - scores) if invert else scores
    allow_thr = float(np.quantile(good.values, config.top_quantile)) if len(good) else 0.5
    block_thr = float(np.quantile(good.values, max(0.0, config.top_quantile - 0.5))) if len(good) else 0.0

    pipeline = evaluation.fitted_full
    return ScoringFunction(
        model_run_id=config.model_run_id,
        source_run_id=config.source_run_id,
        target_kind=config.target.primary.value,
        feature_columns=list(feature_matrix.feature_names),
        numeric_features=list(feature_matrix.numeric_features),
        categorical_features=list(feature_matrix.categorical_features),
        invert_score=invert,
        allow_threshold=allow_thr,
        block_threshold=block_thr,
        reduce_size_factor=config.reduce_size_factor,
        calibrated=evaluation.calibrated,
        model_repr=repr(pipeline.named_steps.get("model")) if pipeline is not None else "",
        best_params=evaluation.best_params,
        provenance={
            "source_run_id": config.source_run_id,
            "model_run_id": config.model_run_id,
            "model_name": evaluation.name,
            "view": config.view.value,
            "adjusted_rar": config.adjusted_rar.to_dict(),
        },
        _pipeline=pipeline,
    )
