"""
Evaluation — a two-layer system.

Layer one asks whether a model predicts anything useful (balanced accuracy,
average precision, Brier / calibration, or regression error). Layer two asks
whether those predictions improve the *economics* of the existing strategy. The
final leaderboard is therefore led by out-of-sample **Adjusted RAR%** with
minimum trade-frequency and maximum-drawdown guardrails; statistical scores are
diagnostics, not the crown.

Predictions are produced by nested, chronological walk-forward (preprocessing fit
inside each fold; tuning on an inner time-aware split). Classifier probabilities
are calibrated within each fold before being turned into allow / reduce / block
overlays, because threshold-based overlays need probabilities that read as
confidence, not just ranks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

from Classes.Core.performance_metrics import CentralizedPerformanceMetrics

from .adjusted_rar import adjusted_rar_from_equity, build_daily_equity_curve
from .config import (AdjustedRARConfig, ModellingView, TargetKind,
                     ValidationConfig)
from .pipeline import ModelSpec
from .targets import TargetSet
from .validation_split import (make_inner_splitter, make_splitter,
                               sort_by_label_time)


@dataclass
class ModelEvaluation:
    """OOS results for one model on one target."""
    name: str
    tier: str
    is_classification: bool
    oos_predictions: pd.Series                       # index = trade_id / period
    primary_metric: float = 0.0                      # OOS Adjusted RAR% (best overlay)
    baseline_adjusted_rar: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    economics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # policy -> metrics
    guardrails: Dict[str, Any] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    calibrated: bool = False
    fold_summary: List[Dict[str, Any]] = field(default_factory=list)
    fold_deltas: List[float] = field(default_factory=list)  # per-fold overlay−baseline Adj RAR
    fitted_full: Any = None                          # pipeline fit on all rows
    stability: Dict[str, Any] = field(default_factory=dict)
    passes_guardrails: bool = True
    # Last fold's fitted estimator + its (genuinely held-out) test set, so
    # permutation importance is computed where the model has not seen the data.
    holdout_estimator: Any = None
    holdout_X: Any = None
    holdout_y: Any = None


def economic_metrics(trade_records: List[Dict[str, Any]],
                     adjusted_rar: AdjustedRARConfig,
                     initial_capital: float) -> Dict[str, float]:
    """Economic summary for a set of realised trades (reuses the framework metrics)."""
    if not trade_records:
        return {"adjusted_rar": 0.0, "n_trades": 0, "sharpe": 0.0, "sortino": 0.0,
                "max_drawdown_pct": 0.0, "hit_rate": 0.0, "profit_factor": 0.0,
                "total_return": 0.0, "avg_return_pct": 0.0}
    curve = build_daily_equity_curve(trade_records, initial_capital=initial_capital)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = CentralizedPerformanceMetrics.calculate_all_metrics(
            equity_curve=curve.rename(columns={"equity": "equity"}) if not curve.empty else None,
            trades=trade_records, initial_capital=initial_capital,
        )
    span_days = max(1, (curve["date"].max() - curve["date"].min()).days) if not curve.empty else 1
    years = span_days / 365.25
    returns = [tr.get("pl_pct", 0.0) for tr in trade_records]
    return {
        "adjusted_rar": adjusted_rar_from_equity(curve, adjusted_rar),
        "n_trades": len(trade_records),
        "trade_frequency": len(trade_records) / years if years > 0 else float(len(trade_records)),
        "sharpe": float(m.get("sharpe_ratio", 0.0) or 0.0),
        "sortino": float(m.get("sortino_ratio", 0.0) or 0.0),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0) or 0.0),
        "hit_rate": float(m.get("win_rate", 0.0) or 0.0),
        "profit_factor": float(m.get("profit_factor", 0.0) or 0.0),
        "total_return": float(m.get("total_return", 0.0) or 0.0),
        "avg_return_pct": float(np.nanmean(returns)) if returns else 0.0,
    }


class Evaluator:
    """Runs nested walk-forward and economic backtests for the model ladder."""

    def __init__(self, view: ModellingView, target: TargetSet, trades: pd.DataFrame,
                 validation: ValidationConfig, adjusted_rar: AdjustedRARConfig,
                 initial_capital: float = 100_000.0, top_quantile: float = 0.70,
                 reduce_size_factor: float = 0.50, calibrate: bool = True,
                 period_frame: Optional[pd.DataFrame] = None):
        self.view = view
        self.target = target
        self.trades = trades
        self.validation = validation
        self.adjusted_rar = adjusted_rar
        self.initial_capital = initial_capital
        self.top_quantile = top_quantile
        self.reduce_size_factor = reduce_size_factor
        self.calibrate = calibrate
        self.period_frame = period_frame

    # -- public ------------------------------------------------------------- #
    def evaluate(self, spec: ModelSpec, X: pd.DataFrame) -> ModelEvaluation:
        y = self.target.y
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]
        ls = self.target.label_start.loc[common]
        le = self.target.label_end.loc[common]
        w = (self.target.sample_weight.loc[common]
             if self.target.sample_weight is not None else None)

        # Time order everything.
        order = sort_by_label_time(ls)
        Xo, yo = X.iloc[order], y.iloc[order]
        lso, leo = ls.iloc[order], le.iloc[order]
        wo = w.iloc[order] if w is not None else None

        splitter = make_splitter(self.validation, lso, leo)
        is_clf = spec.is_classification

        oos = pd.Series(index=Xo.index, dtype=float)
        fold_summary: List[Dict[str, Any]] = []
        fold_deltas: List[float] = []
        best_params_seen: List[Dict[str, Any]] = []
        calibrated_any = False
        last_est = last_Xte = last_yte = None

        for k, (tr, te) in enumerate(splitter.split(Xo), start=1):
            X_tr, X_te = Xo.iloc[tr], Xo.iloc[te]
            y_tr, y_te = yo.iloc[tr], yo.iloc[te]
            w_tr = wo.iloc[tr].values if wo is not None else None
            if is_clf and y_tr.nunique() < 2:
                continue  # cannot train a classifier on one class

            est, params = self._fit_fold(spec, X_tr, y_tr, w_tr, lso.iloc[tr], leo.iloc[tr])
            best_params_seen.append(params)
            preds, did_cal = self._predict_fold(spec, est, X_tr, y_tr, X_te, lso.iloc[tr], leo.iloc[tr])
            calibrated_any = calibrated_any or did_cal
            oos.iloc[te] = preds
            last_est, last_Xte, last_yte = est, X_te, y_te
            fold_summary.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te))})
            fold_deltas.append(self._fold_delta(X_te.index, preds))

        ev = ModelEvaluation(name=spec.name, tier=spec.tier, is_classification=is_clf,
                             oos_predictions=oos.dropna(), calibrated=calibrated_any,
                             fold_summary=fold_summary, fold_deltas=fold_deltas)
        ev.holdout_estimator, ev.holdout_X, ev.holdout_y = last_est, last_Xte, last_yte
        ev.best_params = self._modal_params(best_params_seen)

        predicted = oos.dropna()
        if not predicted.empty:
            y_eval = yo.loc[predicted.index]
            ev.quality_metrics = self._quality_metrics(y_eval, predicted, is_clf)
            self._economic_backtest(ev, predicted, yo)
            ev.stability = self._stability(predicted, y_eval, lso)

        # Refit on ALL rows for interpretation / export.
        ev.fitted_full = self._fit_full(spec, Xo, yo, wo)
        return ev

    # -- fold internals ----------------------------------------------------- #
    def _fit_fold(self, spec: ModelSpec, X_tr, y_tr, w_tr, ls_tr, le_tr):
        if spec.param_grid and self.validation.nested:
            inner = make_inner_splitter(self.validation, ls_tr, le_tr)
            scoring = self._grid_scoring(spec.is_classification)
            try:
                gs = GridSearchCV(clone(spec.pipeline), spec.param_grid, scoring=scoring,
                                  cv=inner, n_jobs=1, error_score="raise")
                gs.fit(X_tr, y_tr)
                return gs.best_estimator_, gs.best_params_
            except Exception:
                pass  # fall through to a plain fit on any tuning failure
        est = clone(spec.pipeline)
        self._safe_fit(est, X_tr, y_tr, w_tr)
        return est, {}

    def _predict_fold(self, spec, est, X_tr, y_tr, X_te, ls_tr, le_tr):
        if spec.is_classification:
            if spec.calibratable and self.calibrate and y_tr.nunique() >= 2:
                try:
                    inner = make_inner_splitter(self.validation, ls_tr, le_tr)
                    cal = CalibratedClassifierCV(clone(est), method="sigmoid", cv=inner)
                    cal.fit(X_tr, y_tr)
                    return self._proba_pos(cal, X_te), True
                except Exception:
                    pass
            return self._proba_pos(est, X_te), False
        return np.asarray(est.predict(X_te), dtype=float), False

    @staticmethod
    def _proba_pos(model, X):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = list(getattr(model, "classes_", [0, 1]))
            pos = classes.index(1) if 1 in classes else len(classes) - 1
            return proba[:, pos]
        if hasattr(model, "decision_function"):
            from scipy.special import expit
            return expit(model.decision_function(X))
        return np.asarray(model.predict(X), dtype=float)

    @staticmethod
    def _safe_fit(est, X, y, w):
        if w is not None:
            try:
                est.fit(X, y, model__sample_weight=w)
                return
            except (TypeError, ValueError):
                pass
        est.fit(X, y)

    def _fit_full(self, spec, X, y, w):
        est = clone(spec.pipeline)
        if spec.param_grid and self.validation.nested and len(X) > self.validation.min_train_size:
            try:
                inner = make_inner_splitter(self.validation, self.target.label_start.loc[X.index],
                                            self.target.label_end.loc[X.index])
                gs = GridSearchCV(est, spec.param_grid,
                                  scoring=self._grid_scoring(spec.is_classification),
                                  cv=inner, n_jobs=1)
                gs.fit(X, y)
                return gs.best_estimator_
            except Exception:
                est = clone(spec.pipeline)
        try:
            self._safe_fit(est, X, y, w.values if w is not None else None)
        except Exception:
            return None
        return est

    @staticmethod
    def _grid_scoring(is_clf: bool) -> str:
        return "average_precision" if is_clf else "neg_mean_absolute_error"

    @staticmethod
    def _modal_params(seen: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not seen:
            return {}
        keys = set().union(*[set(d.keys()) for d in seen]) if seen else set()
        out = {}
        for key in keys:
            vals = [d.get(key) for d in seen if key in d]
            if vals:
                out[key] = max(set(map(str, vals)), key=lambda s: list(map(str, vals)).count(s))
        return out

    # -- metrics ------------------------------------------------------------ #
    def _quality_metrics(self, y_true: pd.Series, pred: pd.Series, is_clf: bool) -> Dict[str, float]:
        from sklearn import metrics as skm
        out: Dict[str, float] = {}
        yt = y_true.values
        pr = pred.values
        if is_clf:
            yhat = (pr >= 0.5).astype(int)
            try:
                out["balanced_accuracy"] = float(skm.balanced_accuracy_score(yt, yhat))
            except Exception:
                pass
            try:
                out["average_precision"] = float(skm.average_precision_score(yt, pr))
            except Exception:
                pass
            try:
                out["roc_auc"] = float(skm.roc_auc_score(yt, pr))
            except Exception:
                pass
            try:
                out["brier"] = float(skm.brier_score_loss(yt, pr))
            except Exception:
                pass
        else:
            try:
                out["mae"] = float(skm.mean_absolute_error(yt, pr))
                out["rmse"] = float(np.sqrt(skm.mean_squared_error(yt, pr)))
                out["r2"] = float(skm.r2_score(yt, pr))
            except Exception:
                pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out["spearman"] = float(pd.Series(yt).corr(pd.Series(pr), method="spearman"))
            except Exception:
                pass
        return out

    def _good_score(self, pred: pd.Series) -> pd.Series:
        """Higher == more attractive, regardless of target polarity."""
        if self.target.primary == TargetKind.BINARY_TAIL_LOSS:
            return 1.0 - pred  # high tail-prob == less attractive
        return pred

    def _economic_backtest(self, ev: ModelEvaluation, pred: pd.Series, y_all: pd.Series) -> None:
        if self.view == ModellingView.PER_PERIOD:
            self._economic_backtest_period(ev, pred)
            return
        # Per-trade: map predictions back to realised trades.
        trades = self.trades.set_index("trade_id")
        common = pred.index.intersection(trades.index)
        if common.empty:
            return
        recs = self._trade_records(trades.loc[common])
        score = self._good_score(pred.loc[common])

        baseline = economic_metrics(recs, self.adjusted_rar, self.initial_capital)
        ev.economics["baseline"] = baseline
        ev.baseline_adjusted_rar = baseline["adjusted_rar"]

        # Top-quantile overlay: keep the most attractive (1 - top_quantile) share.
        thr = float(np.quantile(score.values, self.top_quantile))
        keep = score >= thr
        top = economic_metrics([r for r, k in zip(recs, keep.values) if k],
                               self.adjusted_rar, self.initial_capital)
        ev.economics["top_quantile_only"] = top

        # Reduce-size overlay: scale P/L of the less attractive half.
        med = float(np.median(score.values))
        scaled = []
        for r, s in zip(recs, score.values):
            rr = dict(r)
            if s < med:
                rr["pl"] = rr.get("pl", 0.0) * self.reduce_size_factor
                rr["pl_pct"] = rr.get("pl_pct", 0.0) * self.reduce_size_factor
            scaled.append(rr)
        ev.economics["reduce_size_in_hostile"] = economic_metrics(
            scaled, self.adjusted_rar, self.initial_capital)

        # Primary metric = best overlay's OOS Adjusted RAR%.
        overlay_rars = {p: m["adjusted_rar"] for p, m in ev.economics.items()}
        best_policy = max(overlay_rars, key=overlay_rars.get)
        ev.primary_metric = overlay_rars[best_policy]
        ev.guardrails = self._guardrails(ev.economics[best_policy], baseline)
        ev.guardrails["best_policy"] = best_policy
        ev.passes_guardrails = ev.guardrails.get("passes", True)

    def _economic_backtest_period(self, ev: ModelEvaluation, pred: pd.Series) -> None:
        if self.period_frame is None:
            ev.primary_metric = float(ev.quality_metrics.get("r2", 0.0))
            return
        realised = pd.to_numeric(self.period_frame["period_realised_pl"], errors="coerce")
        nxt = realised.shift(-1).reindex(pred.index).fillna(0.0)
        score = self._good_score(pred)
        # Baseline: always exposed. Overlay: exposed only in predicted-favourable periods.
        base_equity = self.initial_capital + nxt.cumsum()
        thr = float(np.quantile(score.values, 0.5))
        exposed = (score >= thr).astype(float)
        ov_equity = self.initial_capital + (nxt * exposed).cumsum()
        base_curve = pd.DataFrame({"date": pred.index, "equity": base_equity.values})
        ov_curve = pd.DataFrame({"date": pred.index, "equity": ov_equity.values})
        ev.baseline_adjusted_rar = adjusted_rar_from_equity(base_curve, self.adjusted_rar)
        ev.economics["baseline"] = {"adjusted_rar": ev.baseline_adjusted_rar,
                                    "n_periods": int(len(pred))}
        ov_rar = adjusted_rar_from_equity(ov_curve, self.adjusted_rar)
        ev.economics["regime_exposure"] = {"adjusted_rar": ov_rar,
                                           "periods_exposed": int(exposed.sum())}
        ev.primary_metric = max(ev.baseline_adjusted_rar, ov_rar)
        ev.guardrails = {"passes": True, "best_policy":
                         "regime_exposure" if ov_rar >= ev.baseline_adjusted_rar else "baseline"}

    def _fold_delta(self, test_index: pd.Index, preds: np.ndarray) -> float:
        """Per-fold overlay−baseline Adjusted RAR% (top-quantile policy)."""
        if self.view == ModellingView.PER_PERIOD:
            return 0.0
        trades = self.trades.set_index("trade_id")
        common = test_index.intersection(trades.index)
        if len(common) < 3:
            return 0.0
        recs = self._trade_records(trades.loc[common])
        pred_s = pd.Series(preds, index=test_index).loc[common]
        score = self._good_score(pred_s).values
        thr = float(np.quantile(score, self.top_quantile))
        kept = [r for r, s in zip(recs, score) if s >= thr]
        base = adjusted_rar_from_equity(
            build_daily_equity_curve(recs, self.initial_capital), self.adjusted_rar)
        over = adjusted_rar_from_equity(
            build_daily_equity_curve(kept, self.initial_capital), self.adjusted_rar)
        return float(over - base)

    def _trade_records(self, trades: pd.DataFrame) -> List[Dict[str, Any]]:
        recs = []
        for tid, row in trades.iterrows():
            recs.append({
                "trade_id": tid,
                "pl": float(row.get("pl", 0.0) or 0.0),
                "pl_pct": float(row.get("pl_pct", 0.0) or 0.0),
                "entry_date": row.get("entry_date"),
                "exit_date": row.get("exit_date"),
            })
        return recs

    def _guardrails(self, overlay: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, Any]:
        min_freq = max(1.0, 0.25 * baseline.get("trade_frequency", 0.0))
        dd_cap = abs(baseline.get("max_drawdown_pct", 0.0)) * 1.5 + 1e-9
        passes = (overlay.get("trade_frequency", 0.0) >= min_freq
                  and abs(overlay.get("max_drawdown_pct", 0.0)) <= dd_cap)
        return {
            "min_trade_frequency": round(min_freq, 2),
            "overlay_trade_frequency": round(overlay.get("trade_frequency", 0.0), 2),
            "max_drawdown_cap_pct": round(dd_cap, 2),
            "overlay_drawdown_pct": round(overlay.get("max_drawdown_pct", 0.0), 2),
            "adjusted_rar_delta": round(overlay.get("adjusted_rar", 0.0)
                                        - baseline.get("adjusted_rar", 0.0), 3),
            "passes": bool(passes),
        }

    def _stability(self, pred: pd.Series, y_true: pd.Series, label_start: pd.Series) -> Dict[str, Any]:
        """Year-by-year primary-metric stability (is the edge consistent?)."""
        years = pd.to_datetime(label_start.loc[pred.index]).dt.year
        out: Dict[str, Any] = {}
        for yr, idx in pred.groupby(years).groups.items():
            sub = pred.loc[idx]
            yt = y_true.loc[idx]
            if self.target.is_classification and yt.nunique() >= 2:
                from sklearn.metrics import balanced_accuracy_score
                out[str(int(yr))] = round(float(balanced_accuracy_score(
                    yt.values, (sub.values >= 0.5).astype(int))), 3)
            elif not self.target.is_classification:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr = pd.Series(yt.values).corr(pd.Series(sub.values), method="spearman")
                out[str(int(yr))] = round(float(corr or 0.0), 3)
        return {"by_year": out}


def build_leaderboard(evaluations: List[ModelEvaluation]) -> List[ModelEvaluation]:
    """Rank by OOS Adjusted RAR% (guardrail-passing models first)."""
    return sorted(
        evaluations,
        key=lambda e: (e.passes_guardrails, e.primary_metric),
        reverse=True,
    )
