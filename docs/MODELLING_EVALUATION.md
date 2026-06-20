# Modelling & Evaluation Tool

The Modelling & Evaluation tool runs the **modelling, validation, evaluation, and
interpretation** stage of the strategy-diagnostics pipeline. Its purpose is **not**
to forecast market direction. It learns **when the existing strategy tends to work,
when it fails, and when a research overlay** (trade filtering or size reduction)
**is justified**.

It consumes the named, self-contained **run package** produced by the
[Data Preparation](DATA_PREPARATION.md) stage and **does not rebuild preparation
logic**. The package is the contract between the two stages.

> Launch: `python ctk_modelling_evaluation_gui.py` (or the **Modelling &
> Evaluation** card in `python ctk_main_gui.py`).

> **Detailed guides:** [User Guide](MODELLING_USER_GUIDE.md) (the process,
> step by step) · [Dashboard Guide](DASHBOARD_GUIDE.md) (reading the results) ·
> [Data Dictionary](MODELLING_DATA_DICTIONARY.md) (every feature, target, metric).

---

## Core principles

- **Chronological, never shuffled.** All validation respects time order.
- **Interpretable before complex.** A conservative model ladder; ensembles only
  if they improve held-out *economics* and stay explainable.
- **Economic before cosmetic.** The leaderboard is led by out-of-sample
  **Adjusted RAR%**, with trade-frequency and drawdown guardrails. Statistical
  scores are diagnostics, not the crown.
- **Leakage prevention is first-class.** Preprocessing is fit inside folds;
  features join strictly on `available_ts`; overlapping per-trade labels are
  purged & embargoed; tuned models use nested validation.
- **Auditable search.** Every variant tried is logged; a leakage/overfitting risk
  register is a formal output.

---

## Two native views

| View | Row | Answers | Default targets |
|------|-----|---------|-----------------|
| **Per-trade** | one trade at entry | *Should I trust this next trade?* | cost-aware good-trade (binary), clipped net-return (continuous), tail-loss (binary) |
| **Per-period** | one fixed period | *Is the strategy in a favourable / hostile regime?* | next-period return, next-period Adjusted RAR%, regime label |
| **Dual** | both | both, reported together | runs both views |

---

## The six-step flow (`ctk_modelling_evaluation_gui.py`)

1. **Choose run package** – pick a prepared run from `processed_data/runs/`;
   readiness check (trades, families present, leakage-safe `available_ts`).
2. **Define targets** – view, primary target, **cost-aware** thresholds, sample
   weighting, and the configurable **Adjusted RAR%** formula; preview class balance.
3. **Choose validation design** – expanding / rolling / purged-embargoed walk-forward,
   folds, embargo, nested tuning; preview folds + leakage warnings.
4. **Run model ladder** – baseline → linear/logistic → shallow tree → constrained
   gradient boosting; runs in a background thread with live progress.
5. **Review evaluation** – leaderboard by OOS Adjusted RAR%, economics per overlay,
   classification/calibration diagnostics, significance summary.
6. **Interpret & export** – coefficients / rules / held-out permutation importance /
   SHAP, overlay thresholds, then export the artefact set + scoring function. The
   export dialog has an **Open dashboard** button.

---

## Interactive results dashboard

A static report is the wrong medium for deep exploration, so the export also
powers an interactive **Streamlit** dashboard organised around the diagnostic
questions:

> Launch: `streamlit run apps/modelling_dashboard.py` (or the **Results Dashboard**
> card in `ctk_main_gui.py`, or **Open dashboard** after exporting).

| Page | Question | What you can do |
|------|----------|-----------------|
| Does it work? | overall verdict | leaderboard, OOS Adjusted RAR% vs baseline, guardrails, significance (bootstrap CI / permutation / White's Reality Check), risk register |
| Regimes | *what regimes are favourable?* | slice realised economics by any feature bucket; favourable/hostile shortlist across all features; 2-feature regime heatmap; regime timeline |
| Factors | *which factors explain it?* | signed coefficients, held-out permutation importance, SHAP summary, year-by-year stability |
| Scoring & overlays | *is an overlay justified?* | live allow/reduce/block threshold sliders that recompute economics on the same OOS trades; calibration curve; best/worst-scored trades |
| Walk-forward | consistency | per-fold overlay−baseline Adjusted RAR% |
| Robustness & search | did it survive scrutiny? | multiple-testing correction, White's Reality Check, full attempt ledger |

The dashboard reads a model-run export directory; the compute lives in
`Classes/Modelling/dashboard_data.py` (Streamlit-free and unit-tested) and reuses
the engine's Adjusted RAR% / economic-metric functions, so it never diverges from
how the leaderboard was scored.

---

## Architecture

The pipeline is GUI-free and unit-testable; the GUI is a thin view over the controller.

| Layer | Where | Role |
|-------|-------|------|
| Contract | `Classes/Modelling/run_package.py` | Load manifest/contract/trades/panels; readiness check. |
| Config | `Classes/Modelling/config.py` | `ModellingConfig` + target / validation / ladder / Adjusted-RAR sub-configs. |
| Features | `Classes/Modelling/features.py` | Leakage-safe as-of joins → per-trade / per-period matrices, with a calendar-day derived suite (multi-window % change, rolling volatility, z-score, MA-distance, range position). See the Data Dictionary §3a. |
| Targets | `Classes/Modelling/targets.py` | Cost-aware economic targets + sample weighting + class balance. |
| Metric | `Classes/Modelling/adjusted_rar.py` | Configurable Adjusted RAR% (reuses `Classes/Core/stable_metrics.py`). |
| Validation | `Classes/Modelling/validation_split.py` | Chronological + purged/embargoed splitters. |
| Models | `Classes/Modelling/pipeline.py` | The ladder as sklearn pipelines (preprocessing fit in-fold). |
| Evaluation | `Classes/Modelling/evaluation.py` | Walk-forward OOS predictions, in-fold calibration, overlay backtests, leaderboard. |
| Interpretation | `Classes/Modelling/interpretation.py` | Coefficients, tree rules, permutation importance, optional SHAP, calibration. |
| Robustness | `Classes/Modelling/robustness.py` | Bootstrap CIs, permutation test, multiple-testing, White's Reality Check, risk register, attempt ledger. |
| Scoring | `Classes/Modelling/scoring_function.py` | Exportable allow/reduce/block scoring function. |
| Output | `Classes/Modelling/report.py` | Writes the artefact set. |
| Orchestration | `Classes/Modelling/controller.py` | `ModellingController` (the GUI-free façade). |
| GUI | `ctk_modelling_evaluation_gui.py` | Six-step wizard. |

It reuses the framework's existing metrics engine
(`Classes/Core/performance_metrics.py`, `Classes/Core/stable_metrics.py`) and GUI
framework (`Classes/GUI/ctk_wizard_base.py`, `ctk_theme.py`).

---

## Consuming the run package

Features are attached with a **backward `merge_asof` on `available_ts`**, bounded
by each family's `carry_forward_tolerance_days` — never calendar equality — so
nothing the model sees post-dates the trade's entry. Outcome-derived trade
columns (`pl`, `exit_*`, `duration_days`, …) are blocklisted from features. For
per-trade folds, each trade's `[entry_date, exit_date]` label interval drives the
purge & embargo, which is essential because trades may all be open at once.

## Adjusted RAR% (configurable house metric)

The default reproduces the framework's house metric — **RAR%** (annualised return
from a log-equity regression) **× R²** — from `Classes/Core/stable_metrics.py`.
The GUI exposes bars/year, the R²-weighting toggle, and optional clipping, because
the exact formula is house-specific and must remain the user's choice.

---

## Output package

Written to `processed_data/runs/<run_id>/modelling/<model_run_id>/`:

| Artefact | Purpose |
|----------|---------|
| `model_leaderboard.csv` | Candidate models on primary + supporting metrics. |
| `walk_forward_predictions.csv` | Out-of-sample fold predictions. |
| `economic_backtest.csv` | Overlay backtests (baseline / top-quantile / reduce-size). |
| `regime_timeline.csv` | Favourable / neutral / hostile timeline (per-period). |
| `coefficients_*.csv`, `tree_rules_*.txt`, `permutation_importance_*.csv`, `shap_summary_*.csv` | Interpretation outputs. |
| `calibration_*.json` / `.png` | Reliability curves for threshold overlays. |
| `risk_register.json` | Leakage & overfitting risk register. |
| `robustness.json` | Bootstrap CIs, permutation test, multiple-testing, White's Reality Check. |
| `attempt_ledger.csv` | Every model/target variant tried. |
| `research_report.md` | Human-readable operating statement. |
| `scoring_function.json` + `scoring_model.joblib` | Exportable trade-by-trade allow/reduce/block scorer. |
| `analysis_frame_<view>.parquet` | Tidy per-row table (features + target + outcome + OOS score + regime) that powers the dashboard. |
| `dashboard_manifest.json`, `walk_forward_folds.csv` | Dashboard metadata + per-fold deltas. |
| `model_config.json` | The full, reproducible run configuration. |

---

## Dependencies

Adds `scikit-learn` and `statsmodels` (and `joblib`), plus `streamlit` + `plotly`
for the dashboard. **SHAP is optional**: if it is not installed, the tool falls
back to permutation importance + coefficients / tree rules without error.

Tests: `tests/test_modelling.py` (synthetic run-package fixture covering loading,
point-in-time features, the purged/embargoed splitter, Adjusted RAR% reuse, an
end-to-end controller run + export, and the scoring-function round-trip).

## Extending

- **New target**: add a `TargetKind` + construction in `targets.py`.
- **New model tier**: add a `ModelSpec` in `pipeline.py` (keep grids small/explicit).
- **New validation design**: add a `ValidationDesign` + splitter in `validation_split.py`.
- **New robustness control / export**: add to `robustness.py` / `report.py`.
