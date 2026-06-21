"""
Dashboard data layer — load a model-run export and answer the diagnostic
questions, with **no Streamlit dependency** so it is fully unit-testable.

The interactive dashboard (``apps/modelling_dashboard.py``) is a thin view over
this module. Everything here is pure data/compute and reuses the engine's
Adjusted RAR% and economic-metric functions, so the dashboard never diverges from
how the leaderboard was scored.

The questions it answers (straight from the design docs):
- *Does the strategy work?* — leaderboard, OOS Adjusted RAR% vs baseline, guardrails.
- *What regimes are favourable / hostile?* — economics sliced by any feature bucket.
- *Which factors explain it?* — coefficients / permutation / SHAP summaries.
- *Is a reduce-size / filter overlay justified?* — live allow/reduce/block economics.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .adjusted_rar import (adjusted_rar_from_equity, adjusted_rar_from_trades,
                           build_daily_equity_curve)
from .config import AdjustedRARConfig
from .evaluation import economic_metrics


# --------------------------------------------------------------------------- #
# Discovery + loading.
# --------------------------------------------------------------------------- #
def discover_model_runs(runs_root: str = "processed_data/runs") -> List[Dict[str, str]]:
    """Every exported model run found under ``runs_root/<run>/modelling/<id>/``."""
    out: List[Dict[str, str]] = []
    if not os.path.isdir(runs_root):
        return out
    for src in sorted(os.listdir(runs_root)):
        mdir = os.path.join(runs_root, src, "modelling")
        if not os.path.isdir(mdir):
            continue
        for mid in sorted(os.listdir(mdir)):
            path = os.path.join(mdir, mid)
            if os.path.isfile(os.path.join(path, "dashboard_manifest.json")) or \
                    os.path.isfile(os.path.join(path, "model_leaderboard.csv")):
                out.append({"source_run": src, "model_run": mid, "path": path})
    return out


def _read_json(path: str) -> Any:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()


@dataclass
class DashboardData:
    """Everything loaded from one model-run export directory."""
    path: str
    manifest: Dict[str, Any] = field(default_factory=dict)
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    economics: pd.DataFrame = field(default_factory=pd.DataFrame)
    walk_forward: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness: Dict[str, Any] = field(default_factory=dict)
    risk_register: List[Dict[str, str]] = field(default_factory=list)
    regime_timeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    attempt_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysis: Dict[str, pd.DataFrame] = field(default_factory=dict)   # view -> frame
    interpretation: Dict[str, Any] = field(default_factory=dict)      # filename -> object

    # -- convenience -------------------------------------------------------- #
    @property
    def views(self) -> List[str]:
        return list(self.manifest.get("views", {}).keys()) or list(self.analysis.keys())

    def view_meta(self, view: str) -> Dict[str, Any]:
        return self.manifest.get("views", {}).get(view, {})

    @property
    def adjusted_rar(self) -> AdjustedRARConfig:
        return AdjustedRARConfig.from_dict(self.manifest.get("adjusted_rar", {}))

    @property
    def initial_capital(self) -> float:
        return float(self.manifest.get("initial_capital", 100_000.0))

    def numeric_features(self, view: str) -> List[str]:
        feats = self.view_meta(view).get("numeric_features")
        if feats:
            return feats
        df = self.analysis.get(view, pd.DataFrame())
        ignore = {"target", "good_score", "pl", "pl_pct", "open_trades",
                  "closed_trades", "period_realised_pl"}
        return [c for c in df.columns
                if c not in ignore and not c.startswith("oos__")
                and pd.api.types.is_numeric_dtype(df[c])]

    def categorical_features(self, view: str) -> List[str]:
        return self.view_meta(view).get("categorical_features", [])

    def feature_family(self, view: str) -> Dict[str, str]:
        return self.view_meta(view).get("feature_family", {})


def load_dashboard_data(run_dir: str) -> DashboardData:
    """Load a model-run export directory into a :class:`DashboardData`."""
    dd = DashboardData(path=run_dir)
    dd.manifest = _read_json(os.path.join(run_dir, "dashboard_manifest.json")) or {}
    dd.leaderboard = _read_csv(os.path.join(run_dir, "model_leaderboard.csv"))
    dd.economics = _read_csv(os.path.join(run_dir, "economic_backtest.csv"))
    dd.walk_forward = _read_csv(os.path.join(run_dir, "walk_forward_folds.csv"))
    dd.robustness = _read_json(os.path.join(run_dir, "robustness.json")) or {}
    dd.risk_register = _read_json(os.path.join(run_dir, "risk_register.json")) or []
    dd.regime_timeline = _read_csv(os.path.join(run_dir, "regime_timeline.csv"))
    dd.attempt_ledger = _read_csv(os.path.join(run_dir, "attempt_ledger.csv"))

    for parquet in glob.glob(os.path.join(run_dir, "analysis_frame_*.parquet")):
        view = os.path.basename(parquet)[len("analysis_frame_"):-len(".parquet")]
        df = pd.read_parquet(parquet)
        for col in ("entry_date", "exit_date", "period_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        dd.analysis[view] = df

    for pattern in ("coefficients_*.csv", "permutation_importance_*.csv",
                    "shap_summary_*.csv"):
        for f in glob.glob(os.path.join(run_dir, pattern)):
            dd.interpretation[os.path.basename(f)] = _read_csv(f)
    for f in glob.glob(os.path.join(run_dir, "calibration_*.json")):
        dd.interpretation[os.path.basename(f)] = _read_json(f)
    for f in glob.glob(os.path.join(run_dir, "tree_rules_*.txt")):
        with open(f, "r", encoding="utf-8") as fh:
            dd.interpretation[os.path.basename(f)] = fh.read()
    return dd


# --------------------------------------------------------------------------- #
# Compute helpers (pure; reuse the engine's economics).
# --------------------------------------------------------------------------- #
def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Trade records (pl, pl_pct, dates) for the economics functions."""
    if df.empty or "pl" not in df.columns:
        return []
    recs = []
    for _, r in df.iterrows():
        recs.append({
            "trade_id": r.get("trade_id", ""),
            "pl": float(r.get("pl", 0.0) or 0.0),
            "pl_pct": float(r.get("pl_pct", 0.0) or 0.0),
            "entry_date": r.get("entry_date"),
            "exit_date": r.get("exit_date"),
        })
    return recs


def regime_table(df: pd.DataFrame, feature: str, adjusted_rar: AdjustedRARConfig,
                 initial_capital: float, n_buckets: int = 5,
                 categorical: bool = False) -> pd.DataFrame:
    """Per-bucket strategy economics for one feature — the regime answer.

    Returns a row per bucket with count, Adjusted RAR%, hit rate, avg return,
    total P/L, Sharpe and max drawdown, so favourable vs hostile feature ranges
    are immediately visible.
    """
    need = [feature] + [c for c in ("pl", "pl_pct") if c in df.columns]
    sub = df.dropna(subset=[feature]).copy()
    if sub.empty or "pl" not in sub.columns:
        return pd.DataFrame()

    if categorical or not pd.api.types.is_numeric_dtype(sub[feature]):
        sub["_bucket"] = sub[feature].astype(str)
        order = sorted(sub["_bucket"].unique())
    else:
        try:
            sub["_bucket"] = pd.qcut(sub[feature], q=min(n_buckets, sub[feature].nunique()),
                                     duplicates="drop")
        except (ValueError, IndexError):
            return pd.DataFrame()
        order = list(sub["_bucket"].cat.categories)

    rows = []
    for b in order:
        g = sub[sub["_bucket"] == b]
        if g.empty:
            continue
        m = economic_metrics(_records(g), adjusted_rar, initial_capital)
        rows.append({
            "bucket": str(b),
            "count": int(len(g)),
            "adjusted_rar": round(m["adjusted_rar"], 3),
            "hit_rate": round(m["hit_rate"], 1),
            "avg_return_pct": round(m["avg_return_pct"], 3),
            "total_return_pct": round(m["total_return"], 2),
            "sharpe": round(m["sharpe"], 2),
            "max_drawdown_pct": round(m["max_drawdown_pct"], 2),
        })
    return pd.DataFrame(rows)


def two_feature_heatmap(df: pd.DataFrame, fx: str, fy: str,
                        adjusted_rar: AdjustedRARConfig, initial_capital: float,
                        n_bins: int = 4, metric: str = "adjusted_rar") -> pd.DataFrame:
    """Matrix of ``metric`` over a quantile grid of two features (regime map)."""
    sub = df.dropna(subset=[fx, fy]).copy()
    if sub.empty or "pl" not in sub.columns:
        return pd.DataFrame()
    try:
        sub["_x"] = pd.qcut(sub[fx], q=min(n_bins, sub[fx].nunique()), duplicates="drop")
        sub["_y"] = pd.qcut(sub[fy], q=min(n_bins, sub[fy].nunique()), duplicates="drop")
    except (ValueError, IndexError):
        return pd.DataFrame()
    out = {}
    for yb in sub["_y"].cat.categories:
        row = {}
        for xb in sub["_x"].cat.categories:
            g = sub[(sub["_x"] == xb) & (sub["_y"] == yb)]
            if g.empty:
                row[str(xb)] = np.nan
            else:
                m = economic_metrics(_records(g), adjusted_rar, initial_capital)
                row[str(xb)] = round(m.get(metric, np.nan), 3)
        out[str(yb)] = row
    return pd.DataFrame(out).T


def favourable_unfavourable(df: pd.DataFrame, features: List[str],
                            adjusted_rar: AdjustedRARConfig, initial_capital: float,
                            n_buckets: int = 5, top: int = 8) -> pd.DataFrame:
    """Rank every feature bucket by Adjusted RAR% → the favourable/hostile shortlist."""
    rows = []
    for feat in features:
        tbl = regime_table(df, feat, adjusted_rar, initial_capital, n_buckets)
        for _, r in tbl.iterrows():
            rows.append({"feature": feat, "bucket": r["bucket"], "count": r["count"],
                         "adjusted_rar": r["adjusted_rar"], "hit_rate": r["hit_rate"]})
    if not rows:
        return pd.DataFrame()
    allr = pd.DataFrame(rows).sort_values("adjusted_rar", ascending=False)
    fav = allr.head(top).assign(regime="favourable")
    hostile = allr.tail(top).assign(regime="hostile")
    return pd.concat([fav, hostile], ignore_index=True)


def overlay_economics(df: pd.DataFrame, adjusted_rar: AdjustedRARConfig,
                      initial_capital: float, allow_quantile: float = 0.70,
                      reduce_factor: float = 0.50) -> pd.DataFrame:
    """Baseline vs allow/reduce overlays on the *same* OOS trades (live recompute)."""
    if df.empty or "good_score" not in df.columns or "pl" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["good_score", "pl"]).copy()
    recs = _records(work)
    baseline = economic_metrics(recs, adjusted_rar, initial_capital)

    thr = float(np.quantile(work["good_score"].values, allow_quantile))
    keep_mask = work["good_score"].values >= thr
    top = economic_metrics([r for r, k in zip(recs, keep_mask) if k],
                           adjusted_rar, initial_capital)

    med = float(np.median(work["good_score"].values))
    scaled = []
    for r, s in zip(recs, work["good_score"].values):
        rr = dict(r)
        if s < med:
            rr["pl"] *= reduce_factor
            rr["pl_pct"] *= reduce_factor
        scaled.append(rr)
    reduced = economic_metrics(scaled, adjusted_rar, initial_capital)

    rows = []
    for policy, m in (("baseline", baseline), ("top_quantile_only", top),
                      ("reduce_size_in_hostile", reduced)):
        row = {"policy": policy}
        row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def overlay_equity_curves(df: pd.DataFrame, initial_capital: float,
                          allow_quantile: float = 0.70) -> pd.DataFrame:
    """Daily equity curves for baseline vs top-quantile overlay (long format)."""
    if df.empty or "good_score" not in df.columns or "pl" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["good_score", "pl", "exit_date"]).copy()
    recs = _records(work)
    thr = float(np.quantile(work["good_score"].values, allow_quantile))
    kept = [r for r, s in zip(recs, work["good_score"].values) if s >= thr]
    frames = []
    for name, rs in (("baseline", recs), ("top_quantile_only", kept)):
        curve = build_daily_equity_curve(rs, initial_capital=initial_capital)
        if not curve.empty:
            curve["policy"] = name
            frames.append(curve)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def year_stability(df: pd.DataFrame, adjusted_rar: AdjustedRARConfig,
                   initial_capital: float) -> pd.DataFrame:
    """Per-year Adjusted RAR% (is the edge consistent through time?)."""
    if df.empty or "entry_date" not in df.columns or "pl" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["entry_date", "pl"]).copy()
    work["year"] = pd.to_datetime(work["entry_date"]).dt.year
    rows = []
    for yr, g in work.groupby("year"):
        m = economic_metrics(_records(g), adjusted_rar, initial_capital)
        rows.append({"year": int(yr), "count": int(len(g)),
                     "adjusted_rar": round(m["adjusted_rar"], 3),
                     "hit_rate": round(m["hit_rate"], 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-period variants (regime view has no per-trade P/L; slice the period
# outcome instead — next-period return / realised period P/L).
# --------------------------------------------------------------------------- #
def is_per_trade(df: pd.DataFrame) -> bool:
    return "pl" in df.columns


def period_value_column(df: pd.DataFrame) -> Optional[str]:
    """The numeric outcome to slice a per-period panel by."""
    if "target" in df.columns and pd.api.types.is_numeric_dtype(df["target"]):
        return "target"
    if "period_realised_pl" in df.columns:
        return "period_realised_pl"
    return None


def _bucketise(sub: pd.DataFrame, feature: str, n_buckets: int, categorical: bool):
    if categorical or not pd.api.types.is_numeric_dtype(sub[feature]):
        sub["_bucket"] = sub[feature].astype(str)
        return sub, sorted(sub["_bucket"].unique())
    try:
        sub["_bucket"] = pd.qcut(sub[feature], q=min(n_buckets, sub[feature].nunique()),
                                 duplicates="drop")
    except (ValueError, IndexError):
        return sub, []
    return sub, list(sub["_bucket"].cat.categories)


def period_regime_table(df: pd.DataFrame, feature: str, n_buckets: int = 5,
                        categorical: bool = False,
                        value_col: Optional[str] = None) -> pd.DataFrame:
    """Per-bucket mean outcome for a per-period panel (favourable vs hostile)."""
    value_col = value_col or period_value_column(df)
    if value_col is None or feature not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[feature, value_col]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub, order = _bucketise(sub, feature, n_buckets, categorical)
    rows = []
    for b in order:
        g = sub[sub["_bucket"] == b]
        if g.empty:
            continue
        vals = pd.to_numeric(g[value_col], errors="coerce").dropna()
        rows.append({
            "bucket": str(b),
            "count": int(len(g)),
            "mean_outcome": round(float(vals.mean()), 4) if len(vals) else 0.0,
            "pct_positive": round(float((vals > 0).mean() * 100), 1) if len(vals) else 0.0,
            "total_outcome": round(float(vals.sum()), 3),
        })
    return pd.DataFrame(rows)


def period_two_feature_heatmap(df: pd.DataFrame, fx: str, fy: str, n_bins: int = 4,
                               value_col: Optional[str] = None) -> pd.DataFrame:
    """Matrix of mean outcome over a quantile grid of two features (per-period)."""
    value_col = value_col or period_value_column(df)
    if value_col is None:
        return pd.DataFrame()
    sub = df.dropna(subset=[fx, fy, value_col]).copy()
    if sub.empty:
        return pd.DataFrame()
    try:
        sub["_x"] = pd.qcut(sub[fx], q=min(n_bins, sub[fx].nunique()), duplicates="drop")
        sub["_y"] = pd.qcut(sub[fy], q=min(n_bins, sub[fy].nunique()), duplicates="drop")
    except (ValueError, IndexError):
        return pd.DataFrame()
    out = {}
    for yb in sub["_y"].cat.categories:
        row = {}
        for xb in sub["_x"].cat.categories:
            g = sub[(sub["_x"] == xb) & (sub["_y"] == yb)]
            row[str(xb)] = round(float(pd.to_numeric(g[value_col], errors="coerce").mean()), 4) \
                if not g.empty else np.nan
        out[str(yb)] = row
    return pd.DataFrame(out).T


def period_favourable_unfavourable(df: pd.DataFrame, features: List[str],
                                   n_buckets: int = 5, top: int = 8,
                                   value_col: Optional[str] = None) -> pd.DataFrame:
    """Rank every per-period feature bucket by mean outcome (favourable/hostile)."""
    rows = []
    for feat in features:
        tbl = period_regime_table(df, feat, n_buckets=n_buckets, value_col=value_col)
        for _, r in tbl.iterrows():
            rows.append({"feature": feat, "bucket": r["bucket"], "count": r["count"],
                         "mean_outcome": r["mean_outcome"], "pct_positive": r["pct_positive"]})
    if not rows:
        return pd.DataFrame()
    allr = pd.DataFrame(rows).sort_values("mean_outcome", ascending=False)
    return pd.concat([allr.head(top).assign(regime="favourable"),
                      allr.tail(top).assign(regime="hostile")], ignore_index=True)


def period_year_stability(df: pd.DataFrame, value_col: Optional[str] = None) -> pd.DataFrame:
    """Per-year mean outcome for a per-period panel."""
    value_col = value_col or period_value_column(df)
    if value_col is None or "period_ts" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["period_ts", value_col]).copy()
    work["year"] = pd.to_datetime(work["period_ts"]).dt.year
    rows = []
    for yr, g in work.groupby("year"):
        vals = pd.to_numeric(g[value_col], errors="coerce").dropna()
        rows.append({"year": int(yr), "count": int(len(g)),
                     "mean_outcome": round(float(vals.mean()), 4) if len(vals) else 0.0,
                     "pct_positive": round(float((vals > 0).mean() * 100), 1) if len(vals) else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Performance-driver screen — "what factors suggest better trade performance?"
# A univariate, model-agnostic factor screen with FDR control + cross-year
# consistency. Descriptive (can be confounded), to be cross-checked against the
# model-based held-out importance on the Factors page.
# --------------------------------------------------------------------------- #
def performance_column(df: pd.DataFrame) -> Optional[str]:
    """The column that best represents realised performance for this view."""
    if "pl_pct" in df.columns:
        return "pl_pct"
    if "target" in df.columns and pd.api.types.is_numeric_dtype(df["target"]):
        return "target"
    if "period_realised_pl" in df.columns:
        return "period_realised_pl"
    return None


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r, p = stats.spearmanr(x, y, nan_policy="omit")
            return (float(r) if r == r else 0.0, float(p) if p == p else 1.0)
        except Exception:
            return 0.0, 1.0


def _year_sign_consistency(sub: pd.DataFrame, feature: str, vcol: str,
                           overall_sign: float) -> float:
    tcol = "entry_date" if "entry_date" in sub.columns else (
        "period_ts" if "period_ts" in sub.columns else None)
    if tcol is None:
        return float("nan")
    s = sub.dropna(subset=[tcol]).copy()
    s["_year"] = pd.to_datetime(s[tcol]).dt.year
    signs = []
    for _, g in s.groupby("_year"):
        if len(g) < 8:
            continue
        r, _ = _spearman(g[feature].values, pd.to_numeric(g[vcol], errors="coerce").values)
        if r != 0:
            signs.append(1.0 if np.sign(r) == np.sign(overall_sign) else 0.0)
    return round(float(np.mean(signs)), 2) if signs else float("nan")


def factor_screen(df: pd.DataFrame, numeric_features: List[str],
                  value_col: Optional[str] = None, n_buckets: int = 5,
                  min_count: int = 20, fdr_alpha: float = 0.1) -> pd.DataFrame:
    """Rank numeric factors by their association with realised performance.

    For each factor: Spearman rank correlation with the outcome (robust + a
    p-value), the top-minus-bottom bucket outcome spread (economic effect size),
    a single-factor AUC for ranking above/below-median trades, coverage, and the
    fraction of years whose effect keeps the same sign. P-values are
    Benjamini-Hochberg corrected across factors (many factors -> some look good by
    chance).
    """
    vcol = value_col or performance_column(df)
    if vcol is None:
        return pd.DataFrame()
    base = df.dropna(subset=[vcol]).copy()
    base[vcol] = pd.to_numeric(base[vcol], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for feat in numeric_features:
        if feat not in base.columns:
            continue
        sub = base.dropna(subset=[feat])
        if len(sub) < max(min_count, 20) or sub[feat].nunique() < 3:
            continue
        x = sub[feat].astype(float).values
        y = sub[vcol].astype(float).values
        if np.nanstd(x) == 0:
            continue
        r, p = _spearman(x, y)
        try:
            q = pd.qcut(sub[feat], q=min(n_buckets, sub[feat].nunique()), duplicates="drop")
        except (ValueError, IndexError):
            continue
        means = sub.groupby(q, observed=False)[vcol].mean()
        if len(means) < 2:
            continue
        top_mean, bottom_mean = float(means.iloc[-1]), float(means.iloc[0])
        auc = float("nan")
        good = (y > np.nanmedian(y)).astype(int)
        if len(np.unique(good)) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                a = roc_auc_score(good, x)
                auc = round(float(max(a, 1 - a)), 3)
            except Exception:
                pass
        rows.append({
            "feature": feat,
            "direction": "higher → better" if r >= 0 else "higher → worse",
            "rank_corr": round(r, 3),
            "abs_rank_corr": round(abs(r), 3),
            "p_value": p,
            "effect_top_minus_bottom": round(top_mean - bottom_mean, 4),
            "top_bucket_mean": round(top_mean, 4),
            "bottom_bucket_mean": round(bottom_mean, 4),
            "auc": auc,
            "n": int(len(sub)),
            "year_consistency": _year_sign_consistency(sub, feat, vcol, np.sign(r) or 1.0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    try:
        from statsmodels.stats.multitest import multipletests
        reject, corrected, _, _ = multipletests(out["p_value"].fillna(1.0).values,
                                                alpha=fdr_alpha, method="fdr_bh")
        out["p_value_bh"] = np.round(corrected, 4)
        out["significant"] = reject
    except Exception:
        out["p_value_bh"] = out["p_value"]
        out["significant"] = out["p_value"] < 0.05
    out["p_value"] = out["p_value"].round(4)
    return out.sort_values("abs_rank_corr", ascending=False).reset_index(drop=True)


def factor_decile_profile(df: pd.DataFrame, feature: str, value_col: Optional[str] = None,
                          n_buckets: int = 10) -> pd.DataFrame:
    """Mean performance + count per feature decile (the relationship shape)."""
    vcol = value_col or performance_column(df)
    if vcol is None or feature not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[feature, vcol]).copy()
    sub[vcol] = pd.to_numeric(sub[vcol], errors="coerce")
    if sub[feature].nunique() < 2:
        return pd.DataFrame()
    try:
        sub["_b"] = pd.qcut(sub[feature], q=min(n_buckets, sub[feature].nunique()),
                            duplicates="drop")
    except (ValueError, IndexError):
        return pd.DataFrame()
    rows = []
    for b, g in sub.groupby("_b", observed=False):
        v = g[vcol].dropna()
        rows.append({"bucket": str(b), "feature_mid": round(float(g[feature].median()), 4),
                     "count": int(len(g)), "mean_performance": round(float(v.mean()), 4),
                     "pct_positive": round(float((v > 0).mean() * 100), 1)})
    return pd.DataFrame(rows)


def categorical_factor_table(df: pd.DataFrame, feature: str, value_col: Optional[str] = None,
                             min_count: int = 5) -> pd.DataFrame:
    """Mean performance by category (which symbols / sides / reasons do best)."""
    vcol = value_col or performance_column(df)
    if vcol is None or feature not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[feature, vcol]).copy()
    sub[vcol] = pd.to_numeric(sub[vcol], errors="coerce")
    rows = []
    for cat, g in sub.groupby(sub[feature].astype(str)):
        if len(g) < min_count:
            continue
        v = g[vcol].dropna()
        rows.append({"category": str(cat), "count": int(len(g)),
                     "mean_performance": round(float(v.mean()), 4),
                     "pct_positive": round(float((v > 0).mean() * 100), 1)})
    out = pd.DataFrame(rows)
    return out.sort_values("mean_performance", ascending=False).reset_index(drop=True) \
        if not out.empty else out


def period_overlay_curves(df: pd.DataFrame, allow_quantile: float = 0.5,
                          value_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative outcome: always-exposed baseline vs score-gated exposure.

    Returns ``(long_curves, summary)``. The overlay takes exposure only when the
    finalist's score is in the top ``1 − allow_quantile``.
    """
    value_col = value_col or period_value_column(df)
    if value_col is None or "good_score" not in df.columns or "period_ts" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = df.dropna(subset=["good_score", value_col, "period_ts"]).sort_values("period_ts").copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    val = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0).values
    thr = float(np.quantile(work["good_score"].values, allow_quantile))
    exposed = (work["good_score"].values >= thr).astype(float)
    base = pd.DataFrame({"period": work["period_ts"].values,
                         "cumulative": np.cumsum(val), "policy": "always exposed"})
    over = pd.DataFrame({"period": work["period_ts"].values,
                         "cumulative": np.cumsum(val * exposed),
                         "policy": "exposed in favourable only"})
    summary = pd.DataFrame([
        {"policy": "always exposed", "periods": int(len(work)),
         "total_outcome": round(float(val.sum()), 3),
         "mean_outcome": round(float(val.mean()), 4)},
        {"policy": "exposed in favourable only", "periods": int(exposed.sum()),
         "total_outcome": round(float((val * exposed).sum()), 3),
         "mean_outcome": round(float(val[exposed > 0].mean()), 4) if exposed.sum() else 0.0},
    ])
    return pd.concat([base, over], ignore_index=True), summary

