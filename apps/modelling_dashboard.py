"""
Modelling & Evaluation — interactive results dashboard (Streamlit).

A browser dashboard for exploring a model-run export in depth, organised around
the diagnostic questions from the design docs:

    Does the strategy work?  ·  What regimes are favourable / hostile?
    Which factors explain it?  ·  Is a reduce-size / filter overlay justified?
    Does it survive leakage / nested / multiple-testing checks?

Run:  streamlit run apps/modelling_dashboard.py
Optionally point it straight at a run:  MODEL_RUN_DIR=/path/to/run streamlit run apps/modelling_dashboard.py

All computation lives in ``Classes.Modelling.dashboard_data`` (Streamlit-free and
unit-tested); this file is layout only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Classes.Modelling.dashboard_data import (  # noqa: E402
    DashboardData, discover_model_runs, favourable_unfavourable, load_dashboard_data,
    overlay_economics, overlay_equity_curves, regime_table, two_feature_heatmap,
    year_stability,
)

st.set_page_config(page_title="Strategy Diagnostics Dashboard", layout="wide")
_POS, _NEG = "#27AE60", "#E74C3C"


@st.cache_data(show_spinner=False)
def _load(path: str) -> DashboardData:
    return load_dashboard_data(path)


def _select_run() -> str:
    env_dir = os.environ.get("MODEL_RUN_DIR")
    if env_dir and os.path.isdir(env_dir):
        st.sidebar.success(f"Loaded from MODEL_RUN_DIR:\n{env_dir}")
        return env_dir
    runs_root = st.sidebar.text_input("Runs directory", value="processed_data/runs")
    runs = discover_model_runs(runs_root)
    if not runs:
        st.sidebar.warning("No exported model runs found. Run the Modelling & "
                           "Evaluation tool and export first.")
        st.stop()
    labels = [f"{r['source_run']} / {r['model_run']}" for r in runs]
    choice = st.sidebar.selectbox("Model run", labels, index=len(labels) - 1)
    return runs[labels.index(choice)]["path"]


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    """Optional symbol / side / year filters shared across pages."""
    with st.sidebar.expander("Filters", expanded=False):
        out = df.copy()
        if "symbol" in out.columns:
            syms = sorted(out["symbol"].dropna().astype(str).unique())
            pick = st.multiselect("Symbol", syms, default=[])
            if pick:
                out = out[out["symbol"].astype(str).isin(pick)]
        if "side" in out.columns:
            sides = sorted(out["side"].dropna().astype(str).unique())
            pick = st.multiselect("Side", sides, default=[])
            if pick:
                out = out[out["side"].astype(str).isin(pick)]
        if "entry_date" in out.columns and out["entry_date"].notna().any():
            yrs = sorted(pd.to_datetime(out["entry_date"]).dt.year.dropna().unique())
            if len(yrs) > 1:
                lo, hi = st.select_slider("Year range", options=yrs,
                                          value=(yrs[0], yrs[-1]))
                yy = pd.to_datetime(out["entry_date"]).dt.year
                out = out[(yy >= lo) & (yy <= hi)]
        st.caption(f"{len(out)} of {len(df)} rows after filters")
    return out


# --------------------------------------------------------------------------- #
# Pages.
# --------------------------------------------------------------------------- #
def page_overview(dd: DashboardData, view: str) -> None:
    st.header("Does the strategy work?")
    lb = dd.leaderboard
    if lb.empty:
        st.info("No leaderboard found.")
        return
    top = lb.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Finalist", str(top.get("model", "?")))
    c2.metric("OOS Adjusted RAR%", f"{top.get('oos_adjusted_rar', 0):.2f}")
    c3.metric("Baseline Adjusted RAR%", f"{top.get('baseline_adjusted_rar', 0):.2f}",
              delta=f"{top.get('oos_adjusted_rar', 0) - top.get('baseline_adjusted_rar', 0):.2f}")
    c4.metric("Guardrails", "PASS" if top.get("passes_guardrails", False) else "REVIEW")

    st.subheader("Model leaderboard")
    st.caption("Ranked by out-of-sample Adjusted RAR% with trade-frequency and "
               "drawdown guardrails; statistical scores are diagnostics, not the crown.")
    st.dataframe(lb, use_container_width=True, hide_index=True)

    rob = dd.robustness
    if rob:
        st.subheader("Significance & robustness")
        cols = st.columns(3)
        wrc = rob.get("whites_reality_check") or {}
        cols[0].metric("White's Reality Check p", f"{wrc.get('p_value', float('nan')):.3f}"
                       if wrc else "—")
        bd = rob.get("bootstrap_delta") or {}
        cols[1].metric("Bootstrap ΔAdjRAR p", f"{bd.get('p_value', float('nan')):.3f}"
                       if bd else "—")
        pt = rob.get("permutation_test") or {}
        cols[2].metric("Permutation test p", f"{pt.get('p_value', float('nan')):.3f}"
                       if pt else "—")
        if bd:
            st.caption(f"Bootstrap Adjusted RAR% delta (finalist vs baseline): "
                       f"{bd.get('point')} CI [{bd.get('lo')}, {bd.get('hi')}]")

    if dd.risk_register:
        st.subheader("Leakage & overfitting risk register")
        rr = pd.DataFrame(dd.risk_register)
        rr["status"] = rr["status"].map({"pass": "✅ pass", "warn": "⚠️ warn"}).fillna(rr["status"])
        st.dataframe(rr, use_container_width=True, hide_index=True)


def page_regimes(dd: DashboardData, view: str, df: pd.DataFrame) -> None:
    st.header("What regimes are favourable?")
    st.caption("Slice the strategy's realised economics by any feature. Green bars "
               "mark favourable feature ranges; red mark hostile ones.")
    if "pl" not in df.columns:
        st.info("Regime economics need per-trade P/L (use the per-trade view).")
        if not dd.regime_timeline.empty:
            st.subheader("Regime timeline")
            st.dataframe(dd.regime_timeline, use_container_width=True, hide_index=True)
        return

    numeric = [f for f in dd.numeric_features(view) if f in df.columns]
    categorical = [f for f in dd.categorical_features(view) if f in df.columns]
    feats = numeric + categorical
    if not feats:
        st.info("No features available to slice.")
        return

    c1, c2 = st.columns([2, 1])
    feature = c1.selectbox("Feature to slice by", feats, index=0)
    n_buckets = c2.slider("Buckets", 2, 10, 5)
    tbl = regime_table(df, feature, dd.adjusted_rar, dd.initial_capital,
                       n_buckets=n_buckets, categorical=feature in categorical)
    if tbl.empty:
        st.info("Not enough data to bucket this feature.")
    else:
        fig = px.bar(tbl, x="bucket", y="adjusted_rar",
                     color=tbl["adjusted_rar"] >= 0,
                     color_discrete_map={True: _POS, False: _NEG},
                     hover_data=["count", "hit_rate", "avg_return_pct", "max_drawdown_pct"],
                     labels={"adjusted_rar": "Adjusted RAR%", "bucket": feature})
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.subheader("Favourable vs hostile shortlist (all features)")
    st.caption("Every feature bucket ranked by Adjusted RAR% — the strategy's "
               "best and worst conditions at a glance.")
    fav = favourable_unfavourable(df, numeric, dd.adjusted_rar, dd.initial_capital,
                                  n_buckets=n_buckets)
    if not fav.empty:
        fav_disp = fav.copy()
        fav_disp["regime"] = fav_disp["regime"].map({"favourable": "🟢 favourable",
                                                     "hostile": "🔴 hostile"})
        st.dataframe(fav_disp, use_container_width=True, hide_index=True)

    if len(numeric) >= 2:
        st.subheader("Two-feature regime map")
        cc1, cc2, cc3 = st.columns(3)
        fx = cc1.selectbox("X feature", numeric, index=0)
        fy = cc2.selectbox("Y feature", numeric, index=min(1, len(numeric) - 1))
        nb = cc3.slider("Grid bins", 2, 6, 4)
        heat = two_feature_heatmap(df, fx, fy, dd.adjusted_rar, dd.initial_capital, n_bins=nb)
        if not heat.empty:
            fig = px.imshow(heat, color_continuous_scale="RdYlGn", aspect="auto",
                            labels={"x": fx, "y": fy, "color": "Adjusted RAR%"},
                            text_auto=True)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    if not dd.regime_timeline.empty:
        st.subheader("Regime timeline")
        st.dataframe(dd.regime_timeline, use_container_width=True, hide_index=True)


def page_factors(dd: DashboardData, view: str) -> None:
    st.header("Which factors explain it?")
    st.caption("Interpretation is staged: coefficients (sign), held-out permutation "
               "importance (impact), and SHAP for ensemble finalists.")
    interp = dd.interpretation
    shown = False

    def _bar(frame: pd.DataFrame, value_col: str, title: str, diverging=False):
        f = frame.head(20).iloc[::-1]
        if diverging:
            fig = px.bar(f, x=value_col, y="feature", orientation="h",
                         color=f[value_col] >= 0,
                         color_discrete_map={True: _POS, False: _NEG})
            fig.update_layout(showlegend=False)
        else:
            fig = px.bar(f, x=value_col, y="feature", orientation="h")
        fig.update_layout(title=title, height=520)
        st.plotly_chart(fig, use_container_width=True)

    for name, obj in interp.items():
        if name.startswith("coefficients_") and isinstance(obj, pd.DataFrame) and not obj.empty:
            _bar(obj, "coefficient", "Linear coefficients (signed)", diverging=True)
            shown = True
    for name, obj in interp.items():
        if name.startswith("permutation_importance_") and isinstance(obj, pd.DataFrame) and not obj.empty:
            _bar(obj, "importance", "Permutation importance (held-out)")
            shown = True
    for name, obj in interp.items():
        if name.startswith("shap_summary_") and isinstance(obj, pd.DataFrame) and not obj.empty:
            _bar(obj, "mean_abs_shap", "SHAP summary (mean |value|)")
            shown = True
    for name, obj in interp.items():
        if name.startswith("tree_rules_") and isinstance(obj, str):
            with st.expander("Shallow-tree rules"):
                st.code(obj)
            shown = True
    if not shown:
        st.info("No interpretation artefacts found for this run.")

    df = dd.analysis.get(view, pd.DataFrame())
    if "pl" in df.columns and "entry_date" in df.columns:
        st.subheader("Stability through time")
        ys = year_stability(df, dd.adjusted_rar, dd.initial_capital)
        if not ys.empty:
            fig = px.bar(ys, x="year", y="adjusted_rar",
                         color=ys["adjusted_rar"] >= 0,
                         color_discrete_map={True: _POS, False: _NEG},
                         hover_data=["count", "hit_rate"],
                         labels={"adjusted_rar": "Adjusted RAR%"})
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)


def page_scoring(dd: DashboardData, view: str, df: pd.DataFrame) -> None:
    st.header("Is a reduce-size / filter overlay justified?")
    if "good_score" not in df.columns or "pl" not in df.columns:
        st.info("Overlay analysis needs per-trade scores and P/L (use the per-trade view).")
        return

    st.caption("Translate the finalist's calibrated score into allow / reduce / "
               "block overlays and recompute economics live on the same OOS trades.")
    c1, c2 = st.columns(2)
    allow_q = c1.slider("Allow: keep top quantile", 0.0, 0.95, 0.70, 0.05)
    reduce_f = c2.slider("Reduce-size factor (hostile half)", 0.0, 1.0, 0.50, 0.05)

    econ = overlay_economics(df, dd.adjusted_rar, dd.initial_capital,
                             allow_quantile=allow_q, reduce_factor=reduce_f)
    if not econ.empty:
        base = econ[econ["policy"] == "baseline"].iloc[0]
        cols = st.columns(3)
        for col, policy in zip(cols, ["baseline", "top_quantile_only", "reduce_size_in_hostile"]):
            row = econ[econ["policy"] == policy]
            if row.empty:
                continue
            r = row.iloc[0]
            delta = r["adjusted_rar"] - base["adjusted_rar"]
            col.metric(policy.replace("_", " "), f"AdjRAR {r['adjusted_rar']:.2f}",
                       delta=None if policy == "baseline" else f"{delta:+.2f}")
            col.caption(f"trades={int(r.get('n_trades', 0))}  sharpe={r.get('sharpe', 0):.2f}  "
                        f"maxDD={r.get('max_drawdown_pct', 0):.1f}%  hit={r.get('hit_rate', 0):.0f}%")
        st.dataframe(econ, use_container_width=True, hide_index=True)

    curves = overlay_equity_curves(df, dd.initial_capital, allow_quantile=allow_q)
    if not curves.empty:
        fig = px.line(curves, x="date", y="equity", color="policy",
                      title="Out-of-sample equity: baseline vs top-quantile overlay")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score distribution")
    work = df.dropna(subset=["good_score"]).copy()
    color = "target" if "target" in work.columns else None
    fig = px.histogram(work, x="good_score", color=color, nbins=40, barmode="overlay",
                       opacity=0.7)
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    # Calibration (reliability) curve.
    for name, obj in dd.interpretation.items():
        if name.startswith("calibration_") and isinstance(obj, dict) and obj.get("mean_predicted"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="grey"),
                                     name="Perfectly calibrated"))
            fig.add_trace(go.Scatter(x=obj["mean_predicted"], y=obj["fraction_positive"],
                                     mode="lines+markers", name="Model"))
            fig.update_layout(title="Calibration (reliability) curve", height=360,
                              xaxis_title="Mean predicted probability",
                              yaxis_title="Observed frequency")
            st.plotly_chart(fig, use_container_width=True)
            break

    st.subheader("Best & worst scored trades")
    cols = [c for c in ("trade_id", "symbol", "entry_date", "good_score", "pl_pct", "pl", "target")
            if c in work.columns]
    ranked = work.sort_values("good_score", ascending=False)[cols]
    cc1, cc2 = st.columns(2)
    cc1.caption("Top scored"); cc1.dataframe(ranked.head(12), use_container_width=True, hide_index=True)
    cc2.caption("Bottom scored"); cc2.dataframe(ranked.tail(12), use_container_width=True, hide_index=True)


def page_walkforward(dd: DashboardData, view: str) -> None:
    st.header("Walk-forward")
    wf = dd.walk_forward
    if wf.empty:
        st.info("No walk-forward fold data found.")
        return
    wfv = wf[wf["view"] == view] if "view" in wf.columns else wf
    models = sorted(wfv["model"].unique()) if "model" in wfv.columns else []
    pick = st.multiselect("Models", models, default=models[:4])
    show = wfv[wfv["model"].isin(pick)] if pick else wfv
    if "adjusted_rar_delta" in show.columns:
        fig = px.bar(show, x="fold", y="adjusted_rar_delta", color="model", barmode="group",
                     title="Per-fold overlay − baseline Adjusted RAR%")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(show, use_container_width=True, hide_index=True)


def page_search(dd: DashboardData, view: str) -> None:
    st.header("Robustness & research search")
    rob = dd.robustness
    if rob.get("multiple_testing"):
        st.subheader("Multiple-hypothesis correction")
        mt = rob["multiple_testing"]
        frame = pd.DataFrame({"model": mt.get("models", []),
                              "p_corrected": mt.get("pvals_corrected", []),
                              "reject_null": mt.get("reject", [])})
        st.caption(f"Method: {mt.get('method')} (α={mt.get('alpha', 0.05)})")
        st.dataframe(frame, use_container_width=True, hide_index=True)
    if rob.get("whites_reality_check"):
        st.subheader("White's Reality Check")
        st.json(rob["whites_reality_check"])
    if not dd.attempt_ledger.empty:
        st.subheader("Attempt ledger (every variant tried)")
        st.dataframe(dd.attempt_ledger, use_container_width=True, hide_index=True)
    with st.expander("Raw robustness payload"):
        st.json(rob or {})


# --------------------------------------------------------------------------- #
# App.
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("Strategy Diagnostics — Modelling & Evaluation")
    run_dir = _select_run()
    dd = _load(run_dir)

    views = dd.views
    if not views:
        st.error("This export has no analysis frames. Re-export with the latest tool.")
        st.stop()
    view = st.sidebar.selectbox("View", views, index=0)
    meta = dd.view_meta(view)
    st.sidebar.caption(f"Primary target: {meta.get('primary_target', '?')}  ·  "
                       f"finalist: {meta.get('finalist', '?')}")
    for w in meta.get("warnings", []):
        st.sidebar.warning(w)

    base_df = dd.analysis.get(view, pd.DataFrame())
    df = _filters(base_df) if not base_df.empty else base_df

    page = st.sidebar.radio("Page", [
        "Does it work?", "Regimes", "Factors", "Scoring & overlays",
        "Walk-forward", "Robustness & search",
    ])
    if page == "Does it work?":
        page_overview(dd, view)
    elif page == "Regimes":
        page_regimes(dd, view, df)
    elif page == "Factors":
        page_factors(dd, view)
    elif page == "Scoring & overlays":
        page_scoring(dd, view, df)
    elif page == "Walk-forward":
        page_walkforward(dd, view)
    else:
        page_search(dd, view)


if __name__ == "__main__":
    main()
