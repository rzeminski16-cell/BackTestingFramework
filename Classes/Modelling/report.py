"""
Export the artefact set — treated as part of the product, not an afterthought.

Writes the model leaderboard, walk-forward predictions, regime/economic tables,
feature-importance / SHAP summaries, calibration outputs, the leakage &
overfitting risk register, the attempt ledger, a human-readable research report,
and the exportable scoring function — all under
``processed_data/runs/<run_id>/modelling/<model_run_id>/``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_csv(df: Optional[pd.DataFrame], path: str, written: Dict[str, str], key: str) -> None:
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        written[key] = path


def _save_calibration_plot(curve: Dict[str, List[float]], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
        ax.plot(curve["mean_predicted"], curve["fraction_positive"], "o-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration (reliability) curve")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True
    except Exception:
        return False


def _leaderboard_frame(leaderboard: List[Any]) -> pd.DataFrame:
    rows = []
    for rank, ev in enumerate(leaderboard, start=1):
        row = {
            "rank": rank, "model": ev.name, "tier": ev.tier,
            "oos_adjusted_rar": round(ev.primary_metric, 3),
            "baseline_adjusted_rar": round(ev.baseline_adjusted_rar, 3),
            "passes_guardrails": ev.passes_guardrails,
            "best_policy": ev.guardrails.get("best_policy", ""),
            "calibrated": ev.calibrated,
        }
        row.update({f"q_{k}": round(v, 4) for k, v in ev.quality_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _economics_frame(leaderboard: List[Any]) -> pd.DataFrame:
    rows = []
    for ev in leaderboard:
        for policy, m in ev.economics.items():
            row = {"model": ev.name, "policy": policy}
            row.update({k: round(v, 4) if isinstance(v, (int, float)) else v
                        for k, v in m.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _predictions_frame(leaderboard: List[Any]) -> pd.DataFrame:
    frames = []
    for ev in leaderboard:
        s = ev.oos_predictions
        if s is not None and not s.empty:
            frames.append(pd.DataFrame({"id": s.index, "model": ev.name,
                                        "oos_prediction": s.values}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_artifacts(output_dir: str, *, config: Any, leaderboard: List[Any],
                    interpretations: Dict[str, Dict[str, Any]],
                    risk_register: List[Dict[str, str]],
                    attempt_ledger_df: pd.DataFrame,
                    robustness: Dict[str, Any],
                    regime_timeline_rows: List[Dict[str, Any]],
                    scoring_function: Any,
                    readiness_info: Dict[str, Any]) -> Dict[str, str]:
    """Write the full export set; returns ``{artefact_key: path}``."""
    os.makedirs(output_dir, exist_ok=True)
    written: Dict[str, str] = {}

    # Config + leaderboard + economics + predictions.
    cfg_path = os.path.join(output_dir, "model_config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2, default=str)
    written["model_config"] = cfg_path

    _safe_csv(_leaderboard_frame(leaderboard),
              os.path.join(output_dir, "model_leaderboard.csv"), written, "model_leaderboard")
    _safe_csv(_economics_frame(leaderboard),
              os.path.join(output_dir, "economic_backtest.csv"), written, "economic_backtest")
    _safe_csv(_predictions_frame(leaderboard),
              os.path.join(output_dir, "walk_forward_predictions.csv"),
              written, "walk_forward_predictions")

    # Interpretation outputs for the finalist(s).
    for name, interp in interpretations.items():
        stem = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
        _safe_csv(interp.get("coefficients"),
                  os.path.join(output_dir, f"coefficients_{stem}.csv"),
                  written, f"coefficients_{stem}")
        _safe_csv(interp.get("permutation_importance"),
                  os.path.join(output_dir, f"permutation_importance_{stem}.csv"),
                  written, f"permutation_importance_{stem}")
        _safe_csv(interp.get("shap_summary"),
                  os.path.join(output_dir, f"shap_summary_{stem}.csv"),
                  written, f"shap_summary_{stem}")
        if interp.get("tree_rules"):
            p = os.path.join(output_dir, f"tree_rules_{stem}.txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(interp["tree_rules"])
            written[f"tree_rules_{stem}"] = p
        if interp.get("calibration"):
            p = os.path.join(output_dir, f"calibration_{stem}.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(interp["calibration"], fh, indent=2)
            written[f"calibration_{stem}"] = p
            if _save_calibration_plot(interp["calibration"],
                                      os.path.join(output_dir, f"calibration_{stem}.png")):
                written[f"calibration_plot_{stem}"] = os.path.join(
                    output_dir, f"calibration_{stem}.png")

    # Regime timeline + risk register + robustness + attempt ledger.
    if regime_timeline_rows:
        _safe_csv(pd.DataFrame(regime_timeline_rows),
                  os.path.join(output_dir, "regime_timeline.csv"), written, "regime_timeline")
    rr_path = os.path.join(output_dir, "risk_register.json")
    with open(rr_path, "w", encoding="utf-8") as fh:
        json.dump(risk_register, fh, indent=2)
    written["risk_register"] = rr_path

    rob_path = os.path.join(output_dir, "robustness.json")
    with open(rob_path, "w", encoding="utf-8") as fh:
        json.dump(robustness, fh, indent=2, default=str)
    written["robustness"] = rob_path

    _safe_csv(attempt_ledger_df, os.path.join(output_dir, "attempt_ledger.csv"),
              written, "attempt_ledger")

    # Exportable scoring function (+ fitted model).
    if scoring_function is not None:
        written.update(scoring_function.save(output_dir))

    # Human-readable research report.
    report_path = os.path.join(output_dir, "research_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(config, leaderboard, risk_register, robustness,
                                  regime_timeline_rows, readiness_info))
    written["research_report"] = report_path

    return written


def _render_markdown(config: Any, leaderboard: List[Any],
                     risk_register: List[Dict[str, str]], robustness: Dict[str, Any],
                     regime_timeline_rows: List[Dict[str, Any]],
                     readiness_info: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Research report — {config.model_run_name}\n")
    lines.append(f"- Source run package: `{config.source_run_id}`")
    lines.append(f"- View: **{config.view.value}**")
    lines.append(f"- Primary target: **{config.target.primary.value}**")
    lines.append(f"- Validation: **{config.validation.design.value}** "
                 f"({config.validation.n_splits} folds, embargo "
                 f"{config.validation.embargo_days}d, nested={config.validation.nested})")
    lines.append(f"- Weighting: **{config.weight_mode.value}**")
    rar = config.adjusted_rar
    lines.append(f"- Adjusted RAR%%: bars/year={rar.bars_per_year}, "
                 f"R²-weighted={rar.weight_by_r_squared}\n")

    lines.append("## What this answers\n")
    lines.append("Not market direction — *when does this strategy work or fail, and "
                 "when is a reduce-size / filter overlay justified?* The leaderboard is "
                 "led by out-of-sample Adjusted RAR%, with trade-frequency and drawdown "
                 "guardrails; statistical scores are diagnostics, not the crown.\n")

    lines.append("## Model leaderboard (out-of-sample)\n")
    lines.append("| Rank | Model | Tier | OOS Adj RAR% | Baseline Adj RAR% | Best policy | Guardrails |")
    lines.append("|---|---|---|---|---|---|---|")
    for rank, ev in enumerate(leaderboard, start=1):
        lines.append(f"| {rank} | {ev.name} | {ev.tier} | {ev.primary_metric:.3f} | "
                     f"{ev.baseline_adjusted_rar:.3f} | {ev.guardrails.get('best_policy','')} | "
                     f"{'pass' if ev.passes_guardrails else 'review'} |")
    lines.append("")

    if leaderboard:
        best = leaderboard[0]
        lines.append("## Finalist\n")
        lines.append(f"**{best.name}** — OOS Adjusted RAR% {best.primary_metric:.3f} "
                     f"vs baseline {best.baseline_adjusted_rar:.3f} "
                     f"(Δ {best.guardrails.get('adjusted_rar_delta', 0)}).")
        if best.quality_metrics:
            qm = ", ".join(f"{k}={v:.3f}" for k, v in best.quality_metrics.items())
            lines.append(f"Quality diagnostics: {qm}.")
        if best.stability.get("by_year"):
            lines.append(f"Year-by-year stability: {best.stability['by_year']}.")
        lines.append("")

    if regime_timeline_rows:
        lines.append("## Regime timeline\n")
        for r in regime_timeline_rows[:30]:
            lines.append(f"- {r['start']} → {r['end']}: **{r['regime']}**")
        lines.append("")

    if robustness:
        lines.append("## Significance & robustness\n")
        wrc = robustness.get("whites_reality_check")
        if wrc:
            lines.append(f"- White's Reality Check: best=`{wrc.get('best_model')}`, "
                         f"p={wrc.get('p_value')}")
        if robustness.get("bootstrap_delta"):
            bd = robustness["bootstrap_delta"]
            lines.append(f"- Bootstrap Adjusted RAR% delta (finalist vs baseline): "
                         f"{bd.get('point')} [{bd.get('lo')}, {bd.get('hi')}], p={bd.get('p_value')}")
        if robustness.get("permutation_test"):
            pt = robustness["permutation_test"]
            lines.append(f"- Permutation test p-value: {pt.get('p_value')}")
        if robustness.get("multiple_testing"):
            lines.append(f"- Multiple-testing correction: {robustness['multiple_testing'].get('method')}")
        lines.append("")

    lines.append("## Leakage & overfitting risk register\n")
    lines.append("| Risk | Status | Detail |")
    lines.append("|---|---|---|")
    for r in risk_register:
        lines.append(f"| {r['risk']} | {r['status']} | {r['detail']} |")
    lines.append("")

    lines.append("## Assumptions & caveats\n")
    lines.append("- All trades may be open simultaneously; per-trade folds purge & "
                 "embargo overlapping label windows.")
    lines.append("- Adjusted RAR% uses the configured house formula; per-period RAR "
                 "inherits the configured bars/year.")
    lines.append("- PDP/ICE-style reasoning is unreliable where features are highly "
                 "correlated (see correlation flags).")
    lines.append("- Threshold overlays rely on calibrated probabilities; uncalibrated "
                 "models are rank-useful only.\n")
    return "\n".join(lines)
