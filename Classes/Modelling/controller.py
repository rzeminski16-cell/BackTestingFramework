"""
ModellingController — headless orchestration the GUI (or a CLI/tests) drives.

Holds the model-run state and sequences the stage: load the prepared run package,
build leakage-safe features, construct economic targets, preview folds, run the
model ladder under nested chronological validation, evaluate economics, interpret
finalists, apply robustness controls, and export the artefact set. Keeping this
GUI-free makes the whole flow unit-testable without a display.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (ModellingConfig, ModellingView, TargetKind)
from .run_package import PackageReadiness, RunPackage, RunPackageLoader
from .features import FeatureBuilder, FeatureMatrix
from .targets import TargetBuilder, TargetSet
from .validation_split import (ChronologicalSplitter, fold_preview, make_splitter,
                               sort_by_label_time)
from .pipeline import ModelLadder
from .evaluation import (Evaluator, ModelEvaluation, build_leaderboard,
                         economic_metrics)
from . import interpretation as interp
from . import robustness as rob
from .scoring_function import ScoringFunction, build_scoring_function
from .report import write_artifacts

ProgressCb = Optional[Callable[[float, str], None]]


# Per-trade vs per-period default primary targets.
_PER_TRADE_TARGETS = {TargetKind.BINARY_GOOD_TRADE, TargetKind.CONTINUOUS_NET_RETURN,
                      TargetKind.BINARY_TAIL_LOSS}
_PER_PERIOD_TARGETS = {TargetKind.NEXT_PERIOD_RETURN, TargetKind.NEXT_PERIOD_ADJ_RAR,
                       TargetKind.REGIME_LABEL}


@dataclass
class ViewResults:
    """Evaluation outputs for a single analytical view."""
    view: ModellingView
    feature_matrix: FeatureMatrix
    target_set: TargetSet
    leaderboard: List[ModelEvaluation] = field(default_factory=list)
    period_frame: Optional[pd.DataFrame] = None


@dataclass
class ModelRunResults:
    """Everything a finished model run produces."""
    config: ModellingConfig
    views: List[ViewResults] = field(default_factory=list)
    interpretations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    robustness: Dict[str, Any] = field(default_factory=dict)
    risk_register: List[Dict[str, str]] = field(default_factory=list)
    regime_timeline: List[Dict[str, Any]] = field(default_factory=list)
    scoring_function: Optional[ScoringFunction] = None
    attempt_ledger: rob.AttemptLedger = field(default_factory=rob.AttemptLedger)

    @property
    def leaderboard(self) -> List[ModelEvaluation]:
        """Combined leaderboard across views (primary-metric ranked)."""
        evals = [e for v in self.views for e in v.leaderboard]
        return build_leaderboard(evals)


class ModellingController:
    """Stateful façade over the modelling/evaluation pipeline for UI use."""

    def __init__(self, config: ModellingConfig):
        self.config = config
        self.loader = RunPackageLoader(config.runs_root)
        self.package: Optional[RunPackage] = None
        self.readiness: Optional[PackageReadiness] = None
        self._prepared: Dict[ModellingView, Tuple[FeatureMatrix, TargetSet, Optional[pd.DataFrame]]] = {}

    # -- screen 1: choose run package -------------------------------------- #
    def discover_packages(self) -> List[Dict[str, Any]]:
        return self.loader.discover()

    def load_package(self, run_id: str) -> PackageReadiness:
        self.package = self.loader.load(run_id)
        self.config.source_run_id = self.package.run_id
        self.readiness = self.loader.check_readiness(self.package)
        self._prepared.clear()
        return self.readiness

    # -- view preparation (features + targets) ----------------------------- #
    def _views_to_run(self) -> List[ModellingView]:
        if self.config.view == ModellingView.DUAL:
            return [ModellingView.PER_TRADE, ModellingView.PER_PERIOD]
        return [self.config.view]

    def _coerce_primary(self, view: ModellingView) -> None:
        primary = self.config.target.primary
        if view == ModellingView.PER_TRADE and primary not in _PER_TRADE_TARGETS:
            self.config.target.primary = TargetKind.BINARY_GOOD_TRADE
        if view == ModellingView.PER_PERIOD and primary not in _PER_PERIOD_TARGETS:
            self.config.target.primary = TargetKind.NEXT_PERIOD_RETURN

    def prepare(self, view: ModellingView) -> Tuple[FeatureMatrix, TargetSet, Optional[pd.DataFrame]]:
        if self.package is None:
            raise RuntimeError("Load a run package first.")
        if view in self._prepared:
            return self._prepared[view]
        self._coerce_primary(view)
        builder = FeatureBuilder(self.package)
        tbuilder = TargetBuilder(self.config.target, self.config.weight_mode,
                                 self.config.adjusted_rar, self.config.initial_capital)
        period_frame = None
        if view == ModellingView.PER_PERIOD:
            fm, period_frame = builder.build_per_period(
                self.config.target.period_freq, self.config.feature_families,
                self.config.feature_columns)
            ts = tbuilder.build_per_period(period_frame)
        else:
            fm = builder.build_per_trade(self.config.feature_families,
                                         self.config.feature_columns)
            ts = tbuilder.build_per_trade(self.package.trades)
        self._prepared[view] = (fm, ts, period_frame)
        return self._prepared[view]

    # -- screen 2: target preview ------------------------------------------ #
    def target_preview(self) -> Dict[str, Any]:
        view = self._views_to_run()[0]
        fm, ts, _ = self.prepare(view)
        return {"view": view.value, "primary": ts.primary.value,
                "is_classification": ts.is_classification,
                "class_balance": ts.class_balance, "n_rows": int(len(ts.y)),
                "n_features": len(fm.feature_names),
                "feature_warnings": list(fm.warnings)}

    # -- screen 3: validation fold preview --------------------------------- #
    def fold_preview(self) -> List[Dict[str, Any]]:
        view = self._views_to_run()[0]
        fm, ts, _ = self.prepare(view)
        order = sort_by_label_time(ts.label_start)
        ls = ts.label_start.iloc[order]
        le = ts.label_end.iloc[order]
        splitter = make_splitter(self.config.validation, ls, le)
        return fold_preview(splitter, len(fm.X), ls)

    def leakage_warnings(self) -> List[str]:
        warns: List[str] = []
        if self.package is None:
            return warns
        if self.config.validation.design.value != "purged_embargoed" \
                and self.config.view in (ModellingView.PER_TRADE, ModellingView.DUAL):
            warns.append("Per-trade data with overlapping trades should use the "
                         "purged/embargoed design to avoid overlap leakage.")
        for fam in self.package.available_families():
            df = self.package.panels[fam]
            if "available_ts" in df.columns and df["available_ts"].isna().mean() > 0.2:
                warns.append(f"Family '{fam}' has >20% null available_ts; those rows "
                             "cannot be joined point-in-time.")
        return warns

    # -- screen 4: run the model ladder ------------------------------------ #
    def run(self, progress: ProgressCb = None) -> ModelRunResults:
        results = ModelRunResults(config=self.config)
        views = self._views_to_run()
        for vi, view in enumerate(views):
            fm, ts, period_frame = self.prepare(view)
            ladder = ModelLadder(fm.numeric_features, fm.categorical_features,
                                 ts.is_classification)
            specs = ladder.build(self.config.ladder)
            evaluator = Evaluator(
                view=view, target=ts, trades=self.package.trades,
                validation=self.config.validation, adjusted_rar=self.config.adjusted_rar,
                initial_capital=self.config.initial_capital,
                top_quantile=self.config.top_quantile,
                reduce_size_factor=self.config.reduce_size_factor,
                calibrate=self.config.ladder.calibrate, period_frame=period_frame)

            evals: List[ModelEvaluation] = []
            for si, spec in enumerate(specs):
                if progress:
                    frac = (vi + (si + 1) / max(1, len(specs))) / len(views)
                    progress(min(0.95, frac), f"{view.value}: {spec.name}")
                results.attempt_ledger.record(
                    view=view.value, model=spec.name, tier=spec.tier,
                    target=ts.primary.value, validation=self.config.validation.design.value,
                    weighting=self.config.weight_mode.value, tuned=bool(spec.param_grid))
                evals.append(evaluator.evaluate(spec, fm.X))

            vr = ViewResults(view=view, feature_matrix=fm, target_set=ts,
                             leaderboard=build_leaderboard(evals), period_frame=period_frame)
            results.views.append(vr)

        if progress:
            progress(0.97, "Interpreting & testing robustness")
        self._post_process(results)
        if progress:
            progress(1.0, "Done")
        return results

    # -- interpretation + robustness + scoring ----------------------------- #
    def _post_process(self, results: ModelRunResults) -> None:
        leaderboard = results.leaderboard
        if not leaderboard:
            results.risk_register = self._risk_register(results, finalist=None,
                                                        correlated=0)
            return
        finalist = leaderboard[0]
        # Locate the view that produced the finalist.
        vr = next((v for v in results.views if finalist in v.leaderboard), results.views[0])

        # Interpretation (finalist + best linear/tree if present).
        correlated_flags = self._interpret_finalists(results, vr)

        # Robustness.
        results.robustness = self._robustness(results, vr, finalist)

        # Regime timeline (per-period regime target if available).
        results.regime_timeline = self._regime_timeline(vr)

        # Scoring function from the finalist.
        try:
            results.scoring_function = build_scoring_function(
                finalist, self.config, vr.feature_matrix)
        except Exception:
            results.scoring_function = None

        results.risk_register = self._risk_register(results, finalist, correlated_flags)

    def _interpret_finalists(self, results: ModelRunResults, vr: ViewResults) -> int:
        to_explain = []
        seen = set()
        for ev in vr.leaderboard[:3]:
            if ev.name not in seen and ev.tier != "baseline":
                to_explain.append(ev)
                seen.add(ev.name)
        correlated = interp.correlation_caution(vr.feature_matrix.X,
                                                vr.feature_matrix.numeric_features)
        for ev in to_explain:
            entry: Dict[str, Any] = {}
            if ev.fitted_full is not None:
                entry["coefficients"] = interp.linear_coefficients(ev.fitted_full)
                entry["tree_rules"] = interp.tree_rules(ev.fitted_full)
                if ev.tier == "ensemble":
                    entry["shap_summary"] = interp.shap_summary(
                        ev.fitted_full, vr.feature_matrix.X)
            entry["permutation_importance"] = interp.permutation_importance_holdout(
                ev.holdout_estimator, ev.holdout_X, ev.holdout_y, ev.is_classification)
            if ev.is_classification and not ev.oos_predictions.empty:
                yt = vr.target_set.y.loc[ev.oos_predictions.index]
                entry["calibration"] = interp.calibration_curve_data(yt, ev.oos_predictions)
            entry["correlated_features"] = correlated
            results.interpretations[ev.name] = entry
        return len(correlated)

    def _robustness(self, results: ModelRunResults, vr: ViewResults,
                    finalist: ModelEvaluation) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Per-model bootstrap p-values of the overlay−baseline delta → multiple testing.
        model_pvals: List[float] = []
        model_names: List[str] = []
        if vr.view == ModellingView.PER_TRADE:
            trades = self.package.trades.set_index("trade_id")
            for ev in vr.leaderboard:
                if ev.tier == "baseline" or ev.oos_predictions.empty:
                    continue
                common = ev.oos_predictions.index.intersection(trades.index)
                if len(common) < 5:
                    continue
                recs = [{"trade_id": tid, "pl": float(trades.loc[tid].get("pl", 0.0) or 0.0),
                         "pl_pct": float(trades.loc[tid].get("pl_pct", 0.0) or 0.0),
                         "entry_date": trades.loc[tid].get("entry_date"),
                         "exit_date": trades.loc[tid].get("exit_date")} for tid in common]
                pred = ev.oos_predictions.loc[common]
                good = (1.0 - pred) if ev.is_classification and \
                    vr.target_set.primary == TargetKind.BINARY_TAIL_LOSS else pred
                thr = float(np.quantile(good.values, self.config.top_quantile))
                keep_ids = {tid for tid, s in zip(common, good.values) if s >= thr}
                overlay = [r for r in recs if r["trade_id"] in keep_ids]
                bd = rob.bootstrap_adjusted_rar_delta(
                    recs, overlay, self.config.adjusted_rar, self.config.initial_capital,
                    n_boot=300)
                model_pvals.append(bd["p_value"])
                model_names.append(ev.name)
                if ev is finalist:
                    out["bootstrap_delta"] = bd

        if model_pvals:
            mt = rob.correct_pvalues(model_pvals, method="holm")
            mt["models"] = model_names
            out["multiple_testing"] = mt

        # White's Reality Check across models using per-fold Adjusted RAR deltas.
        perf = {ev.name: np.asarray(ev.fold_deltas, dtype=float)
                for ev in vr.leaderboard if ev.tier != "baseline" and len(ev.fold_deltas) > 1}
        wrc = rob.whites_reality_check(perf) if perf else None
        if wrc:
            out["whites_reality_check"] = wrc

        # Permutation test on the finalist (outer chronological CV).
        try:
            spec_pipeline = finalist.fitted_full
            ts = vr.target_set
            X = vr.feature_matrix.X.loc[ts.y.index]
            order = sort_by_label_time(ts.label_start)
            cv = make_splitter(self.config.validation,
                               ts.label_start.iloc[order], ts.label_end.iloc[order])
            pt = rob.permutation_significance(
                spec_pipeline, X.iloc[order], ts.y.iloc[order], cv,
                finalist.is_classification, n_permutations=100)
            if pt:
                out["permutation_test"] = pt
        except Exception:
            pass
        return out

    def _regime_timeline(self, vr: ViewResults) -> List[Dict[str, Any]]:
        ts = vr.target_set
        regime = ts.targets.get(TargetKind.REGIME_LABEL.value)
        if regime is None or vr.period_frame is None:
            return []
        return interp.regime_timeline(vr.period_frame, regime.dropna())

    def _risk_register(self, results: ModelRunResults,
                       finalist: Optional[ModelEvaluation], correlated: int) -> List[Dict[str, str]]:
        purged = self.config.validation.design.value == "purged_embargoed"
        period_only = self.config.view == ModellingView.PER_PERIOD
        return rob.build_risk_register(
            preprocessing_in_pipeline=True,
            time_leakage_safe=True,
            purge_embargo_active=purged or period_only,
            nested_validation=self.config.validation.nested,
            n_attempts=results.attempt_ledger.count(),
            calibration_present=bool(finalist.calibrated) if finalist else False,
            correlated_feature_flags=correlated,
        )

    # -- screen 6: export -------------------------------------------------- #
    def output_dir(self) -> str:
        return os.path.join(self.config.runs_root, self.config.source_run_id,
                            "modelling", self.config.model_run_id)

    def export(self, results: ModelRunResults) -> Dict[str, str]:
        out_dir = self.output_dir()
        readiness_info = self.readiness.info if self.readiness else {}
        return write_artifacts(
            out_dir, config=self.config, leaderboard=results.leaderboard,
            interpretations=results.interpretations, risk_register=results.risk_register,
            attempt_ledger_df=results.attempt_ledger.to_frame(),
            robustness=results.robustness, regime_timeline_rows=results.regime_timeline,
            scoring_function=results.scoring_function, readiness_info=readiness_info)
