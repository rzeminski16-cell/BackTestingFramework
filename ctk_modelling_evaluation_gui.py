"""
Modelling & Evaluation GUI -- diagnose when a strategy works.

A six-step progressive-disclosure wizard (choose run package -> define targets ->
choose validation design -> run the model ladder -> review evaluation ->
interpret & export) that consumes the point-in-time run package produced by the
Data Preparation stage and answers *when does this strategy work or fail, and
when is a reduce-size / filter overlay justified?*

It is a thin view over ``Classes.Modelling.controller.ModellingController``; all
modelling, validation, evaluation, interpretation and export logic lives there.

Run standalone:  python ctk_modelling_evaluation_gui.py
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import customtkinter as ctk

# Ensure the project root is importable when launched directly.
sys.path.insert(0, str(Path(__file__).parent))

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes
from Classes.GUI.ctk_wizard_base import CTkWizardBase, CTkWizardStep

from Classes.Modelling.config import (AdjustedRARConfig, ModellingConfig,
                                      ModellingView, TargetKind, TargetSpec,
                                      ValidationConfig, ValidationDesign, WeightMode)
from Classes.Modelling.controller import ModellingController, ModelRunResults

_VIEW_LABELS = {ModellingView.PER_TRADE: "Per-trade (trust this trade?)",
                ModellingView.PER_PERIOD: "Per-period (regime / is the strategy in a good state?)",
                ModellingView.DUAL: "Dual (both, reported together)"}
_PER_TRADE_PRIMARY = [TargetKind.BINARY_GOOD_TRADE, TargetKind.CONTINUOUS_NET_RETURN,
                      TargetKind.BINARY_TAIL_LOSS]
_PER_PERIOD_PRIMARY = [TargetKind.NEXT_PERIOD_RETURN, TargetKind.NEXT_PERIOD_ADJ_RAR,
                       TargetKind.REGIME_LABEL]
_DESIGN_LABELS = {
    ValidationDesign.EXPANDING_WALK_FORWARD: "Expanding walk-forward (regular period panel)",
    ValidationDesign.ROLLING_WALK_FORWARD: "Rolling walk-forward (regime-drift check)",
    ValidationDesign.PURGED_EMBARGOED: "Purged & embargoed (overlapping trades)",
}


def _label_for(mapping, value):
    return mapping.get(value, str(value))


def _value_for(mapping, label):
    for k, v in mapping.items():
        if v == label:
            return k
    return list(mapping.keys())[0]


# ===========================================================================
# Step 1 - Choose run package
# ===========================================================================
class ChooseRunPackageStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        Theme.create_hint(parent, "Pick a prepared run package to analyse. The "
                          "modelling stage consumes it as the single source of truth "
                          "and never re-does data preparation.").pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Model run name *", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.name_var = ctk.StringVar(value=self.wizard.config.model_run_name)
        Theme.create_entry(parent, placeholder="e.g. regime_diag_v1",
                           textvariable=self.name_var).pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Runs directory", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.root_var = ctk.StringVar(value=self.wizard.config.runs_root)
        row = Theme.create_frame(parent); row.pack(fill="x", pady=(0, Sizes.PAD_M))
        Theme.create_entry(row, textvariable=self.root_var).pack(side="left", fill="x", expand=True)
        Theme.create_button(row, "Refresh", command=self._refresh, style="secondary",
                            width=90).pack(side="left", padx=(Sizes.PAD_S, 0))

        Theme.create_label(parent, "Run package", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.pkg_var = ctk.StringVar(value="")
        self.pkg_menu = Theme.create_optionmenu(parent, ["(refresh to list)"], variable=self.pkg_var)
        self.pkg_menu.pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_button(parent, "Load & check", command=self._load,
                            style="primary", width=130).pack(anchor="w")
        self.readiness_box = Theme.create_label(parent, "", font=Fonts.BODY_S,
                                               text_color=Colors.TEXT_SECONDARY, justify="left")
        self.readiness_box.pack(anchor="w", pady=(Sizes.PAD_M, 0))
        self._loaded = False

    def on_enter(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        self.wizard.config.runs_root = self.root_var.get().strip() or "processed_data/runs"
        self.wizard.controller = ModellingController(self.wizard.config)
        packages = self.wizard.controller.discover_packages()
        values = [p["run_id"] for p in packages] or ["(no packages found)"]
        self.pkg_menu.configure(values=values)
        if values:
            self.pkg_var.set(values[0])

    def _load(self) -> None:
        run_id = self.pkg_var.get().strip()
        if not run_id or run_id.startswith("("):
            self.readiness_box.configure(text="No run package selected.", text_color=Colors.ERROR)
            return
        try:
            readiness = self.wizard.controller.load_package(run_id)
        except Exception as exc:  # pragma: no cover - UI feedback
            self.readiness_box.configure(text=f"Failed to load: {exc}", text_color=Colors.ERROR)
            return
        info = readiness.info
        lines = [f"Trades: {info.get('n_trades', 0)}  |  Symbols: {info.get('n_symbols', '?')}",
                 f"Date range: {info.get('date_range', ['?', '?'])}",
                 f"Families present: {', '.join(info.get('families_present', [])) or 'none'}"]
        for w in readiness.warnings:
            lines.append(f"⚠ {w}")
        for e in readiness.errors:
            lines.append(f"✖ {e}")
        self._loaded = readiness.ok
        self.readiness_box.configure(
            text="\n".join(lines),
            text_color=Colors.SUCCESS if readiness.ok else Colors.ERROR)

    def validate(self) -> bool:
        self.validation_errors = []
        if not self.name_var.get().strip():
            self.validation_errors.append("A model run name is required.")
        if not self._loaded:
            self.validation_errors.append("Load a run package that passes the readiness check.")
        if self.validation_errors:
            return False
        self.wizard.config.model_run_name = self.name_var.get().strip()
        return True

    def get_summary(self) -> Dict[str, str]:
        if not hasattr(self, "name_var"):
            return {}
        return {"Model run": self.name_var.get().strip() or "(unset)",
                "Package": self.pkg_var.get()}


# ===========================================================================
# Step 2 - Define targets
# ===========================================================================
class DefineTargetsStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        cfg = self.wizard.config
        Theme.create_hint(parent, "Define what \"works\" means. Thresholds are economic, "
                          "not arbitrary: a trade is good only if it beats costs plus a "
                          "buffer.").pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Analytical view", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.view_var = ctk.StringVar(value=_label_for(_VIEW_LABELS, cfg.view))
        Theme.create_optionmenu(parent, list(_VIEW_LABELS.values()), variable=self.view_var,
                                command=lambda *_: self._sync_targets()).pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Primary target", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.target_var = ctk.StringVar(value=cfg.target.primary.value)
        self.target_menu = Theme.create_optionmenu(parent, [k.value for k in _PER_TRADE_PRIMARY],
                                                   variable=self.target_var)
        self.target_menu.pack(anchor="w", pady=(0, Sizes.PAD_M))

        grid = Theme.create_frame(parent); grid.pack(fill="x", pady=(0, Sizes.PAD_M))
        self.cost_var = ctk.StringVar(value=str(cfg.target.cost_buffer_pct))
        self.tail_var = ctk.StringVar(value=str(cfg.target.tail_loss_pct))
        self.clip_var = ctk.StringVar(value=str(cfg.target.return_clip_pct))
        self.period_var = ctk.StringVar(value=cfg.target.period_freq)
        self._labelled_entry(grid, "Cost+buffer threshold (%)", self.cost_var)
        self._labelled_entry(grid, "Tail-loss threshold (%)", self.tail_var)
        self._labelled_entry(grid, "Return clip (±%)", self.clip_var)
        self._labelled_entry(grid, "Period frequency (D/W/M)", self.period_var)

        Theme.create_label(parent, "Sample weighting", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.weight_var = ctk.StringVar(value=cfg.weight_mode.value)
        Theme.create_optionmenu(parent, [w.value for w in WeightMode], variable=self.weight_var
                               ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Adjusted RAR% (the configurable house metric).
        card = Theme.create_card(parent); card.pack(fill="x", pady=(0, Sizes.PAD_M))
        Theme.create_header(card, "Adjusted RAR% (primary selection metric)", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))
        Theme.create_label(card, "Default = the framework house metric: RAR% (log-equity "
                           "regression) × R². Editable below.", font=Fonts.BODY_S,
                           text_color=Colors.TEXT_SECONDARY).pack(anchor="w", padx=Sizes.PAD_M)
        inner = Theme.create_frame(card); inner.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)
        self.bpy_var = ctk.StringVar(value=str(cfg.adjusted_rar.bars_per_year))
        self._labelled_entry(inner, "Bars per year", self.bpy_var)
        self.r2_var = ctk.BooleanVar(value=cfg.adjusted_rar.weight_by_r_squared)
        Theme.create_checkbox(inner, "Weight by R² (penalise noisy curves)",
                             variable=self.r2_var).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_button(parent, "Preview targets & class balance", command=self._preview,
                            style="secondary").pack(anchor="w")
        self.preview_box = Theme.create_label(parent, "", font=Fonts.BODY_S,
                                              text_color=Colors.TEXT_SECONDARY, justify="left")
        self.preview_box.pack(anchor="w", pady=(Sizes.PAD_S, 0))
        self._sync_targets()

    def _labelled_entry(self, parent, label, var) -> None:
        row = Theme.create_frame(parent); row.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(row, label, font=Fonts.BODY_S).pack(side="left")
        Theme.create_entry(row, textvariable=var, width=110).pack(side="right")

    def _current_view(self) -> ModellingView:
        return _value_for(_VIEW_LABELS, self.view_var.get())

    def _sync_targets(self) -> None:
        view = self._current_view()
        options = _PER_PERIOD_PRIMARY if view == ModellingView.PER_PERIOD else _PER_TRADE_PRIMARY
        vals = [k.value for k in options]
        self.target_menu.configure(values=vals)
        if self.target_var.get() not in vals:
            self.target_var.set(vals[0])

    def _commit(self) -> None:
        cfg = self.wizard.config
        cfg.view = self._current_view()
        cfg.weight_mode = WeightMode(self.weight_var.get())
        cfg.target = TargetSpec(
            kinds=[TargetKind(self.target_var.get())],
            primary=TargetKind(self.target_var.get()),
            cost_buffer_pct=_to_float(self.cost_var.get(), 0.2),
            tail_loss_pct=_to_float(self.tail_var.get(), -5.0),
            return_clip_pct=_to_float(self.clip_var.get(), 25.0),
            period_freq=self.period_var.get().strip() or "W",
        )
        cfg.adjusted_rar = AdjustedRARConfig(
            bars_per_year=int(_to_float(self.bpy_var.get(), 365)),
            weight_by_r_squared=bool(self.r2_var.get()))

    def _preview(self) -> None:
        self._commit()
        try:
            prev = self.wizard.controller.target_preview()
        except Exception as exc:  # pragma: no cover - UI feedback
            self.preview_box.configure(text=f"Preview failed: {exc}", text_color=Colors.ERROR)
            return
        cb = prev.get("class_balance", {})
        primary = prev.get("primary")
        detail = cb.get(primary, {})
        lines = [f"View: {prev['view']}  |  rows: {prev['n_rows']}  |  "
                 f"features: {prev.get('n_features', '?')}  |  primary: {primary}",
                 f"Class balance: {detail}"]
        for w in prev.get("feature_warnings", []):
            lines.append(f"⚠ {w}")
        self.preview_box.configure(text="\n".join(lines), text_color=Colors.TEXT_SECONDARY)

    def validate(self) -> bool:
        self.validation_errors = []
        self._commit()
        return True

    def get_summary(self) -> Dict[str, str]:
        if not hasattr(self, "view_var"):
            return {}
        return {"View": self._current_view().value, "Target": self.target_var.get(),
                "Weighting": self.weight_var.get()}


# ===========================================================================
# Step 3 - Choose validation design
# ===========================================================================
class ChooseValidationStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        cfg = self.wizard.config
        Theme.create_hint(parent, "All validation is chronological — never shuffled. "
                          "Overlapping trades need purging & embargo to avoid leakage.").pack(
            anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Validation design", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.design_var = ctk.StringVar(value=_label_for(_DESIGN_LABELS, cfg.validation.design))
        Theme.create_optionmenu(parent, list(_DESIGN_LABELS.values()), variable=self.design_var
                               ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        grid = Theme.create_frame(parent); grid.pack(fill="x", pady=(0, Sizes.PAD_M))
        self.splits_var = ctk.StringVar(value=str(cfg.validation.n_splits))
        self.embargo_var = ctk.StringVar(value=str(cfg.validation.embargo_days))
        self.inner_var = ctk.StringVar(value=str(cfg.validation.inner_splits))
        self._labelled_entry(grid, "Outer folds", self.splits_var)
        self._labelled_entry(grid, "Embargo (days)", self.embargo_var)
        self._labelled_entry(grid, "Inner folds (tuning)", self.inner_var)
        self.nested_var = ctk.BooleanVar(value=cfg.validation.nested)
        Theme.create_checkbox(parent, "Nested validation for tuned models (recommended)",
                             variable=self.nested_var).pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_button(parent, "Preview folds", command=self._preview,
                            style="secondary").pack(anchor="w")
        self.preview_box = Theme.create_label(parent, "", font=Fonts.BODY_S,
                                              text_color=Colors.TEXT_SECONDARY, justify="left")
        self.preview_box.pack(anchor="w", pady=(Sizes.PAD_S, 0))

    def _labelled_entry(self, parent, label, var) -> None:
        row = Theme.create_frame(parent); row.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(row, label, font=Fonts.BODY_S).pack(side="left")
        Theme.create_entry(row, textvariable=var, width=110).pack(side="right")

    def _commit(self) -> None:
        cfg = self.wizard.config
        cfg.validation = ValidationConfig(
            design=_value_for(_DESIGN_LABELS, self.design_var.get()),
            n_splits=int(_to_float(self.splits_var.get(), 5)),
            embargo_days=int(_to_float(self.embargo_var.get(), 5)),
            inner_splits=int(_to_float(self.inner_var.get(), 3)),
            nested=bool(self.nested_var.get()))

    def _preview(self) -> None:
        self._commit()
        try:
            folds = self.wizard.controller.fold_preview()
            warns = self.wizard.controller.leakage_warnings()
        except Exception as exc:  # pragma: no cover - UI feedback
            self.preview_box.configure(text=f"Preview failed: {exc}", text_color=Colors.ERROR)
            return
        lines = [f"Fold {f['fold']}: train={f['n_train']} test={f['n_test']} "
                 f"[{f.get('test_start','?')}→{f.get('test_end','?')}]" for f in folds]
        for w in warns:
            lines.append(f"⚠ {w}")
        self.preview_box.configure(text="\n".join(lines) or "No folds produced.",
                                  text_color=Colors.TEXT_SECONDARY)

    def validate(self) -> bool:
        self.validation_errors = []
        self._commit()
        return True

    def get_summary(self) -> Dict[str, str]:
        if not hasattr(self, "design_var"):
            return {}
        return {"Design": _value_for(_DESIGN_LABELS, self.design_var.get()).value,
                "Folds": self.splits_var.get(), "Embargo": self.embargo_var.get()}


# ===========================================================================
# Step 4 - Run the model ladder
# ===========================================================================
class RunModelLadderStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        cfg = self.wizard.config
        Theme.create_hint(parent, "Run the conservative ladder: baselines → regularised "
                          "linear/logistic → shallow tree → constrained gradient boosting. "
                          "Simpler models are preferred unless complexity earns its keep.").pack(
            anchor="w", pady=(0, Sizes.PAD_M))

        self.baseline_var = ctk.BooleanVar(value=cfg.ladder.run_baseline)
        self.linear_var = ctk.BooleanVar(value=cfg.ladder.run_linear)
        self.tree_var = ctk.BooleanVar(value=cfg.ladder.run_tree)
        self.ensemble_var = ctk.BooleanVar(value=cfg.ladder.run_ensemble)
        self.tune_var = ctk.BooleanVar(value=cfg.ladder.tune)
        self.cal_var = ctk.BooleanVar(value=cfg.ladder.calibrate)
        for text, var in (("Descriptive baseline", self.baseline_var),
                          ("Regularised linear / logistic", self.linear_var),
                          ("Shallow decision tree", self.tree_var),
                          ("Constrained gradient boosting (opt-in)", self.ensemble_var),
                          ("Tune (small, nested search)", self.tune_var),
                          ("Calibrate probabilities", self.cal_var)):
            Theme.create_checkbox(parent, text, variable=var).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_button(parent, "Run model ladder", command=self._run,
                            style="primary", width=170).pack(anchor="w", pady=(Sizes.PAD_M, 0))
        self.status = Theme.create_label(parent, "Not run yet.", font=Fonts.BODY_S,
                                        text_color=Colors.TEXT_SECONDARY, justify="left")
        self.status.pack(anchor="w", pady=(Sizes.PAD_S, 0))

    def _commit(self) -> None:
        cfg = self.wizard.config
        cfg.ladder.run_baseline = bool(self.baseline_var.get())
        cfg.ladder.run_linear = bool(self.linear_var.get())
        cfg.ladder.run_tree = bool(self.tree_var.get())
        cfg.ladder.run_ensemble = bool(self.ensemble_var.get())
        cfg.ladder.tune = bool(self.tune_var.get())
        cfg.ladder.calibrate = bool(self.cal_var.get())

    def _run(self) -> None:
        self._commit()
        self.wizard.results = None
        self.status.configure(text="Running… (training across walk-forward folds)",
                             text_color=Colors.TEXT_SECONDARY)
        q: "queue.Queue" = queue.Queue()

        def worker():
            try:
                res = self.wizard.controller.run(
                    progress=lambda frac, msg: q.put(("progress", (frac, msg))))
                q.put(("done", res))
            except Exception as exc:
                q.put(("error", f"{exc}\n{traceback.format_exc()}"))

        threading.Thread(target=worker, daemon=True).start()
        self._poll(q)

    def _poll(self, q: "queue.Queue") -> None:
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "progress":
                    frac, msg = payload
                    self.status.configure(text=f"[{int(frac*100)}%] {msg}",
                                         text_color=Colors.TEXT_SECONDARY)
                elif kind == "done":
                    self.wizard.results = payload
                    n = len(payload.leaderboard)
                    self.status.configure(text=f"Done — {n} models evaluated. Click Next to review.",
                                         text_color=Colors.SUCCESS)
                    return
                elif kind == "error":
                    self.status.configure(text=f"Run failed: {payload}", text_color=Colors.ERROR)
                    return
        except queue.Empty:
            pass
        self.wizard.root.after(120, lambda: self._poll(q))

    def validate(self) -> bool:
        self.validation_errors = []
        if self.wizard.results is None:
            self.validation_errors.append("Run the model ladder before continuing.")
            return False
        return True

    def get_summary(self) -> Dict[str, str]:
        if not hasattr(self, "linear_var"):
            return {}
        tiers = [t for t, v in (("baseline", self.baseline_var), ("linear", self.linear_var),
                                ("tree", self.tree_var), ("ensemble", self.ensemble_var))
                 if v.get()]
        return {"Ladder": ", ".join(tiers)}


# ===========================================================================
# Step 5 - Review evaluation
# ===========================================================================
class ReviewEvaluationStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        Theme.create_hint(parent, "Models are ranked by out-of-sample Adjusted RAR% with "
                          "trade-frequency and drawdown guardrails. Statistical scores are "
                          "diagnostics, not the crown.").pack(anchor="w", pady=(0, Sizes.PAD_M))
        self.box = Theme.create_textbox(parent, height=380)
        self.box.pack(fill="both", expand=True)

    def on_enter(self) -> None:
        self.box.delete("1.0", "end")
        results: Optional[ModelRunResults] = self.wizard.results
        if not results or not results.leaderboard:
            self.box.insert("1.0", "No results to show.")
            return
        lines = ["LEADERBOARD (out-of-sample Adjusted RAR%)", "=" * 52]
        for rank, ev in enumerate(results.leaderboard, start=1):
            flag = "pass" if ev.passes_guardrails else "review"
            lines.append(f"{rank}. {ev.name} [{ev.tier}]  AdjRAR={ev.primary_metric:.3f} "
                         f"(baseline {ev.baseline_adjusted_rar:.3f})  guardrails={flag}")
            if ev.quality_metrics:
                qm = "  ".join(f"{k}={v:.3f}" for k, v in ev.quality_metrics.items())
                lines.append(f"     diagnostics: {qm}")
            best = ev.economics.get(ev.guardrails.get("best_policy", ""), {})
            if best:
                lines.append(f"     best policy: {ev.guardrails.get('best_policy','')}  "
                             f"trades={best.get('n_trades','?')} "
                             f"sharpe={best.get('sharpe',0):.2f} "
                             f"maxDD={best.get('max_drawdown_pct',0):.1f}%")
        rob = results.robustness
        if rob:
            lines += ["", "ROBUSTNESS", "-" * 52]
            if rob.get("whites_reality_check"):
                w = rob["whites_reality_check"]
                lines.append(f"White's Reality Check: best={w.get('best_model')} p={w.get('p_value')}")
            if rob.get("bootstrap_delta"):
                b = rob["bootstrap_delta"]
                lines.append(f"Bootstrap AdjRAR delta: {b.get('point')} "
                             f"[{b.get('lo')}, {b.get('hi')}] p={b.get('p_value')}")
            if rob.get("permutation_test"):
                lines.append(f"Permutation test p={rob['permutation_test'].get('p_value')}")
        self.box.insert("1.0", "\n".join(lines))

    def get_summary(self) -> Dict[str, str]:
        results: Optional[ModelRunResults] = getattr(self.wizard, "results", None)
        if not results or not results.leaderboard:
            return {}
        top = results.leaderboard[0]
        return {"Finalist": top.name, "OOS Adj RAR%": f"{top.primary_metric:.3f}"}


# ===========================================================================
# Step 6 - Interpret & export
# ===========================================================================
class InterpretExportStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        cfg = self.wizard.config
        Theme.create_hint(parent, "Interpretation first (coefficients / rules / held-out "
                          "permutation importance / SHAP). Set the overlay thresholds, then "
                          "export the full artefact set including the scoring function.").pack(
            anchor="w", pady=(0, Sizes.PAD_M))

        grid = Theme.create_frame(parent); grid.pack(fill="x", pady=(0, Sizes.PAD_M))
        self.topq_var = ctk.StringVar(value=str(cfg.top_quantile))
        self.reduce_var = ctk.StringVar(value=str(cfg.reduce_size_factor))
        for label, var in (("Allow top quantile (0-1)", self.topq_var),
                           ("Reduce-size factor (0-1)", self.reduce_var)):
            row = Theme.create_frame(grid); row.pack(fill="x", pady=Sizes.PAD_XS)
            Theme.create_label(row, label, font=Fonts.BODY_S).pack(side="left")
            Theme.create_entry(row, textvariable=var, width=110).pack(side="right")

        self.box = Theme.create_textbox(parent, height=300)
        self.box.pack(fill="both", expand=True, pady=(Sizes.PAD_S, 0))

    def on_enter(self) -> None:
        self.box.delete("1.0", "end")
        results: Optional[ModelRunResults] = self.wizard.results
        if not results:
            self.box.insert("1.0", "No results.")
            return
        lines = []
        for name, interp in results.interpretations.items():
            lines.append(f"== {name} ==")
            coefs = interp.get("coefficients")
            if coefs is not None and not coefs.empty:
                top = coefs.head(6)
                lines.append("  Top coefficients: " +
                             ", ".join(f"{r.feature}={r.coefficient:+.2f}" for r in top.itertuples()))
            perm = interp.get("permutation_importance")
            if perm is not None and not perm.empty:
                top = perm.head(6)
                lines.append("  Permutation importance (held-out): " +
                             ", ".join(f"{r.feature}={r.importance:.3f}" for r in top.itertuples()))
            if interp.get("shap_summary") is not None:
                lines.append("  SHAP summary available.")
            if interp.get("tree_rules"):
                lines.append("  Tree rules captured.")
            cf = interp.get("correlated_features") or []
            if cf:
                lines.append(f"  ⚠ {len(cf)} highly-correlated feature pairs (PDP/ICE caution).")
        lines.append("")
        lines.append("Risk register:")
        for r in results.risk_register:
            lines.append(f"  [{r['status']}] {r['risk']}")
        self.box.insert("1.0", "\n".join(lines) or "No interpretation available.")

    def _commit(self) -> None:
        cfg = self.wizard.config
        cfg.top_quantile = _to_float(self.topq_var.get(), 0.7)
        cfg.reduce_size_factor = _to_float(self.reduce_var.get(), 0.5)

    def validate(self) -> bool:
        self.validation_errors = []
        self._commit()
        return True

    def get_summary(self) -> Dict[str, str]:
        return {"Output": str(Path(self.wizard.config.runs_root) /
                              self.wizard.config.source_run_id / "modelling" /
                              self.wizard.config.model_run_id)}


# ===========================================================================
# Wizard
# ===========================================================================
class ModellingWizard(CTkWizardBase):
    def __init__(self):
        super().__init__(title="Modelling & Evaluation — Strategy Diagnostics",
                         width=1180, height=860)
        self.config = ModellingConfig(model_run_name="")
        self.controller: Optional[ModellingController] = None
        self.results: Optional[ModelRunResults] = None

        self.add_step(ChooseRunPackageStep(self, "Choose run package"))
        self.add_step(DefineTargetsStep(self, "Define targets"))
        self.add_step(ChooseValidationStep(self, "Choose validation design"))
        self.add_step(RunModelLadderStep(self, "Run model ladder"))
        self.add_step(ReviewEvaluationStep(self, "Review evaluation"))
        self.add_step(InterpretExportStep(self, "Interpret & export"))

        self.on_complete = self._export
        self.start()

    def _get_final_button_text(self) -> str:
        return "Export"

    def _export(self) -> None:
        if self.controller is None or self.results is None:
            self._dialog("Nothing to export", "Run the model ladder first.", error=True)
            return
        try:
            written = self.controller.export(self.results)
        except Exception as exc:  # pragma: no cover - UI feedback
            self._dialog("Export failed", str(exc), error=True)
            traceback.print_exc()
            return
        out = self.controller.output_dir()
        self._export_dialog(out, len(written))

    def _launch_dashboard(self, out_dir: str) -> None:
        """Open the interactive Streamlit dashboard pointed at this run."""
        script = Path(__file__).parent / "apps" / "modelling_dashboard.py"
        env = dict(os.environ, MODEL_RUN_DIR=out_dir)
        try:
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(script)], env=env)
        except Exception as exc:  # pragma: no cover - UI feedback
            self._dialog("Could not launch dashboard",
                         f"{exc}\n\nRun manually:\n  streamlit run apps/modelling_dashboard.py",
                         error=True)

    def _export_dialog(self, out_dir: str, n_files: int) -> None:
        dlg = ctk.CTkToplevel(self.root)
        dlg.title("Export complete")
        dlg.geometry("600x280")
        dlg.transient(self.root)
        dlg.grab_set()
        Theme.create_label(
            dlg,
            f"Artefacts written to:\n{out_dir}\n\n{n_files} files incl. leaderboard, "
            f"research report, risk register, the exportable scoring function, and the "
            f"interactive dashboard data.\n\nExplore the results interactively below.",
            font=Fonts.BODY_M, text_color=Colors.TEXT_PRIMARY, wraplength=550,
            justify="left").pack(expand=True, padx=20, pady=20)
        row = Theme.create_frame(dlg); row.pack(pady=(0, 20))
        Theme.create_button(row, "Open dashboard",
                            command=lambda: self._launch_dashboard(out_dir),
                            style="primary", width=160).pack(side="left", padx=Sizes.PAD_S)
        Theme.create_button(row, "Close", command=dlg.destroy, style="secondary",
                            width=100).pack(side="left", padx=Sizes.PAD_S)

    def _dialog(self, title: str, message: str, error: bool = False) -> None:
        dlg = ctk.CTkToplevel(self.root)
        dlg.title(title)
        dlg.geometry("560x260")
        dlg.transient(self.root)
        dlg.grab_set()
        Theme.create_label(dlg, message, font=Fonts.BODY_M,
                           text_color=Colors.ERROR if error else Colors.TEXT_PRIMARY,
                           wraplength=510, justify="left").pack(expand=True, padx=20, pady=20)
        Theme.create_button(dlg, "OK", command=dlg.destroy, width=100).pack(pady=(0, 20))


def _to_float(text: str, default: float) -> float:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


def main():
    ModellingWizard().run()


if __name__ == "__main__":
    main()
