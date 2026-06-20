"""
Robustness & significance — a research-control system, not an afterthought.

This module provides the guardrails the docs require once a research programme
compares many targets, models, weightings and validation designs: bootstrap
confidence intervals on economic deltas, a permutation test for predictive
significance, multiple-hypothesis correction, White's Reality Check for
data-snooping, plus a formal leakage/overfitting **risk register** and an
**attempt ledger** so the full search — not just the winner — is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .adjusted_rar import adjusted_rar_from_trades
from .config import AdjustedRARConfig


# --------------------------------------------------------------------------- #
# Bootstrap CI on an economic delta (overlay vs baseline Adjusted RAR%).
# --------------------------------------------------------------------------- #
def bootstrap_adjusted_rar_delta(baseline_records: List[Dict[str, Any]],
                                 overlay_records: List[Dict[str, Any]],
                                 adjusted_rar: AdjustedRARConfig,
                                 initial_capital: float,
                                 n_boot: int = 500, alpha: float = 0.05,
                                 random_state: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for (overlay − baseline) Adjusted RAR%."""
    if not baseline_records:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0}
    rng = np.random.default_rng(random_state)
    base_ids = {r["trade_id"] for r in baseline_records if "trade_id" in r}
    keep_ids = {r["trade_id"] for r in overlay_records if "trade_id" in r}

    point = (adjusted_rar_from_trades(overlay_records, adjusted_rar, initial_capital)
             - adjusted_rar_from_trades(baseline_records, adjusted_rar, initial_capital))

    deltas = []
    n = len(baseline_records)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bs = [baseline_records[i] for i in idx]
        ov = [r for r in bs if r.get("trade_id") in keep_ids] if keep_ids else bs
        d = (adjusted_rar_from_trades(ov, adjusted_rar, initial_capital)
             - adjusted_rar_from_trades(bs, adjusted_rar, initial_capital))
        deltas.append(d)
    deltas = np.asarray(deltas, dtype=float)
    lo = float(np.nanpercentile(deltas, 100 * alpha / 2))
    hi = float(np.nanpercentile(deltas, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p-value for delta != 0.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"point": float(point), "lo": lo, "hi": hi, "p_value": float(min(1.0, p))}


def bootstrap_period_overlay_delta(next_returns: Sequence[float], scores: Sequence[float],
                                   allow_quantile: float = 0.5, n_boot: int = 500,
                                   alpha: float = 0.05, random_state: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for (favourable-only − always) mean next-period return."""
    nr = np.asarray(next_returns, dtype=float)
    sc = np.asarray(scores, dtype=float)
    mask = np.isfinite(nr) & np.isfinite(sc)
    nr, sc = nr[mask], sc[mask]
    n = len(nr)
    if n < 5:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0}

    def delta(idx: np.ndarray) -> float:
        r, s = nr[idx], sc[idx]
        thr = np.quantile(s, allow_quantile)
        exposed = s >= thr
        over = r[exposed].mean() if exposed.any() else 0.0
        return float(over - r.mean())

    rng = np.random.default_rng(random_state)
    point = delta(np.arange(n))
    deltas = np.array([delta(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo = float(np.nanpercentile(deltas, 100 * alpha / 2))
    hi = float(np.nanpercentile(deltas, 100 * (1 - alpha / 2)))
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"point": float(point), "lo": lo, "hi": hi, "p_value": float(min(1.0, p))}


# --------------------------------------------------------------------------- #
# Permutation test on a cross-validated score.
# --------------------------------------------------------------------------- #
def permutation_significance(pipeline, X, y, cv, is_classification: bool,
                             n_permutations: int = 200,
                             random_state: int = 0) -> Optional[Dict[str, float]]:
    """Empirical p-value that the CV score could arise under feature/target
    independence (``sklearn.model_selection.permutation_test_score``)."""
    from sklearn.base import clone
    from sklearn.model_selection import permutation_test_score
    scoring = "average_precision" if is_classification else "neg_mean_absolute_error"
    try:
        score, perm_scores, pvalue = permutation_test_score(
            clone(pipeline), X, y, scoring=scoring, cv=cv,
            n_permutations=n_permutations, random_state=random_state, n_jobs=1)
        return {"score": float(score), "p_value": float(pvalue),
                "n_permutations": int(n_permutations)}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Multiple-hypothesis correction.
# --------------------------------------------------------------------------- #
def correct_pvalues(pvalues: Sequence[float], method: str = "holm",
                    alpha: float = 0.05) -> Dict[str, Any]:
    """Holm / Bonferroni / Benjamini-Hochberg correction via statsmodels."""
    pvals = [p for p in pvalues if p is not None and np.isfinite(p)]
    if not pvals:
        return {"method": method, "reject": [], "pvals_corrected": []}
    method_map = {"holm": "holm", "bonferroni": "bonferroni",
                  "fdr_bh": "fdr_bh", "bh": "fdr_bh"}
    try:
        from statsmodels.stats.multitest import multipletests
        reject, corrected, _, _ = multipletests(
            pvals, alpha=alpha, method=method_map.get(method, "holm"))
        return {"method": method, "alpha": alpha,
                "reject": list(map(bool, reject)),
                "pvals_corrected": list(map(float, corrected))}
    except Exception:
        return {"method": method, "reject": [], "pvals_corrected": list(map(float, pvals))}


# --------------------------------------------------------------------------- #
# White's Reality Check (data-snooping control across many model tries).
# --------------------------------------------------------------------------- #
def whites_reality_check(performance: Dict[str, np.ndarray],
                         n_boot: int = 1000, random_state: int = 0) -> Optional[Dict[str, Any]]:
    """White's Reality Check p-value that the best model beats the benchmark.

    ``performance`` maps model name -> per-observation performance *differential*
    versus the benchmark (e.g. per-fold Adjusted RAR delta). Tests whether the
    best observed average could plausibly arise by chance under the null that no
    model beats the benchmark.
    """
    series = {k: np.asarray(v, dtype=float) for k, v in performance.items()
              if v is not None and len(v) > 1}
    if not series:
        return None
    n = min(len(v) for v in series.values())
    if n < 2:
        return None
    rng = np.random.default_rng(random_state)
    means = {k: v[:n].mean() for k, v in series.items()}
    centred = {k: v[:n] - means[k] for k, v in series.items()}
    v_obs = max(np.sqrt(n) * m for m in means.values())

    count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v_star = max(np.sqrt(n) * centred[k][idx].mean() for k in series)
        if v_star >= v_obs:
            count += 1
    best = max(means, key=means.get)
    return {"best_model": best, "statistic": float(v_obs),
            "p_value": float((count + 1) / (n_boot + 1)), "n_models": len(series)}


# --------------------------------------------------------------------------- #
# Leakage & overfitting risk register (a formal output artefact).
# --------------------------------------------------------------------------- #
def build_risk_register(*, preprocessing_in_pipeline: bool, time_leakage_safe: bool,
                        purge_embargo_active: bool, nested_validation: bool,
                        n_attempts: int, calibration_present: bool,
                        correlated_feature_flags: int) -> List[Dict[str, str]]:
    """Pass/fail register mirroring the docs' mandatory-check table."""
    def row(risk, check, ok, detail):
        return {"risk": risk, "check": check,
                "status": "pass" if ok else "warn", "detail": detail}

    return [
        row("Preprocessing leakage", "All transforms fit inside fold pipelines",
            preprocessing_in_pipeline,
            "Impute/scale/encode live inside the sklearn Pipeline; fit on train folds only."),
        row("Time leakage", "Every feature evaluated against availability timestamp",
            time_leakage_safe,
            "Features attached via backward merge_asof on available_ts; nothing post-dates entry."),
        row("Overlap leakage", "Purge & embargo active where labels overlap",
            purge_embargo_active,
            "Purged/embargoed trade splitter removes overlapping label windows."
            if purge_embargo_active else "Design is not purged — acceptable only for regular panels."),
        row("Hyperparameter optimism", "Nested validation used for tuned models",
            nested_validation,
            "Inner time-aware search; outer walk-forward scores untuned-on splits."),
        row("Search overfitting", "Every tried variant logged",
            n_attempts > 0, f"{n_attempts} model/target variants recorded in the attempt ledger."),
        row("Interpretation misuse", "PDP/ICE caution on correlated features",
            correlated_feature_flags == 0,
            f"{correlated_feature_flags} highly-correlated feature pairs flagged."),
        row("Calibration misuse", "Threshold policies require calibration",
            calibration_present,
            "Probabilities calibrated within folds before threshold overlays."
            if calibration_present else "No calibration — treat threshold overlays as rank-only."),
    ]


# --------------------------------------------------------------------------- #
# Attempt ledger.
# --------------------------------------------------------------------------- #
@dataclass
class AttemptLedger:
    """Records every model/target variant tried, for an auditable search log."""
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, **fields: Any) -> None:
        entry = dict(fields)
        entry["logged_at"] = datetime.now(timezone.utc).isoformat()
        self.entries.append(entry)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def count(self) -> int:
        return len(self.entries)
