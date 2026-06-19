"""
Interpretation — staged, not monolithic.

Coefficients (sign/monotonicity) and shallow-tree rules (thresholds) come first.
Permutation importance is computed **on held-out data only** (importance on a poor
model is meaningless). SHAP is the preferred layer for ensemble finalists but is
optional — if it is not installed the tool falls back gracefully. PDP/ICE-style
reasoning is flagged when features are highly correlated, because correlated
market features can make those plots misleading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _model_step(pipeline):
    return pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline


def _pre_step(pipeline):
    return pipeline.named_steps.get("pre") if hasattr(pipeline, "named_steps") else None


def _feature_names_out(pipeline) -> List[str]:
    pre = _pre_step(pipeline)
    if pre is not None:
        try:
            return list(pre.get_feature_names_out())
        except Exception:
            pass
    return []


def linear_coefficients(pipeline) -> Optional[pd.DataFrame]:
    """Signed coefficients for a linear/logistic finalist (transformed features)."""
    model = _model_step(pipeline)
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    coef = np.ravel(coef)
    names = _feature_names_out(pipeline)
    if len(names) != len(coef):
        names = [f"f{i}" for i in range(len(coef))]
    df = pd.DataFrame({"feature": names, "coefficient": coef})
    df["abs"] = df["coefficient"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def tree_rules(pipeline, max_depth: int = 4) -> Optional[str]:
    """Human-readable rules for a shallow-tree finalist."""
    model = _model_step(pipeline)
    if not hasattr(model, "tree_"):
        return None
    from sklearn.tree import export_text
    names = _feature_names_out(pipeline)
    try:
        return export_text(model, feature_names=list(names) if names else None,
                           max_depth=max_depth)
    except Exception:
        try:
            return export_text(model, max_depth=max_depth)
        except Exception:
            return None


def permutation_importance_holdout(estimator, X_holdout: pd.DataFrame,
                                   y_holdout: pd.Series, is_classification: bool,
                                   n_repeats: int = 10,
                                   random_state: int = 0) -> Optional[pd.DataFrame]:
    """Permutation importance on the held-out fold (input-feature level)."""
    if estimator is None or X_holdout is None or y_holdout is None or len(X_holdout) < 5:
        return None
    from sklearn.inspection import permutation_importance
    scoring = "average_precision" if is_classification else "neg_mean_absolute_error"
    try:
        result = permutation_importance(estimator, X_holdout, y_holdout,
                                        scoring=scoring, n_repeats=n_repeats,
                                        random_state=random_state, n_jobs=1)
    except Exception:
        return None
    df = pd.DataFrame({
        "feature": list(X_holdout.columns),
        "importance": result.importances_mean,
        "std": result.importances_std,
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def shap_summary(pipeline, X_sample: pd.DataFrame,
                 max_samples: int = 200) -> Optional[pd.DataFrame]:
    """Mean |SHAP| per transformed feature for an ensemble/tree finalist.

    Returns ``None`` (not an error) when SHAP is unavailable, so the rest of the
    interpretation report still renders.
    """
    if not shap_available() or pipeline is None:
        return None
    try:
        import shap
        pre = _pre_step(pipeline)
        model = _model_step(pipeline)
        Xs = X_sample.iloc[:max_samples]
        Xt = pre.transform(Xs) if pre is not None else Xs.values
        names = _feature_names_out(pipeline) or [f"f{i}" for i in range(Xt.shape[1])]
        if hasattr(model, "tree_") or model.__class__.__name__.startswith("HistGradient") \
                or "Forest" in model.__class__.__name__:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(Xt)
        else:
            explainer = shap.Explainer(model, Xt)
            values = explainer(Xt).values
        if isinstance(values, list):       # binary classifier -> list per class
            values = values[-1]
        mean_abs = np.abs(np.asarray(values)).mean(axis=0).ravel()
        df = pd.DataFrame({"feature": names[:len(mean_abs)], "mean_abs_shap": mean_abs})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception:
        return None


def calibration_curve_data(y_true: pd.Series, proba: pd.Series,
                           n_bins: int = 10) -> Optional[Dict[str, List[float]]]:
    """Reliability-diagram points for a probabilistic classifier."""
    try:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(y_true.values, proba.values,
                                                 n_bins=n_bins, strategy="quantile")
        return {"mean_predicted": list(map(float, mean_pred)),
                "fraction_positive": list(map(float, frac_pos))}
    except Exception:
        return None


def correlation_caution(X: pd.DataFrame, numeric_features: List[str],
                        threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Flag highly correlated numeric features (PDP/ICE interpretation caution)."""
    cols = [c for c in numeric_features if c in X.columns]
    if len(cols) < 2:
        return []
    corr = X[cols].corr().abs()
    flagged: List[Dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= threshold:
                flagged.append({"a": a, "b": b, "abs_corr": round(float(r), 3)})
    return sorted(flagged, key=lambda d: d["abs_corr"], reverse=True)[:25]


def regime_timeline(period_frame: Optional[pd.DataFrame],
                    regime_series: Optional[pd.Series]) -> List[Dict[str, Any]]:
    """Compact favourable/neutral/hostile timeline for the report."""
    if regime_series is None or regime_series.empty:
        return []
    out: List[Dict[str, Any]] = []
    cur, start = None, None
    for ts, val in regime_series.items():
        if val != cur:
            if cur is not None:
                out.append({"regime": str(cur), "start": str(pd.Timestamp(start).date()),
                            "end": str(pd.Timestamp(ts).date())})
            cur, start = val, ts
    if cur is not None:
        out.append({"regime": str(cur), "start": str(pd.Timestamp(start).date()),
                    "end": str(pd.Timestamp(regime_series.index[-1]).date())})
    return out
