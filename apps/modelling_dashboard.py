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
    DashboardData, categorical_factor_table, discover_model_runs, factor_decile_profile,
    factor_screen, favourable_unfavourable, is_per_trade, load_dashboard_data,
    overlay_economics, overlay_equity_curves, performance_column,
    period_favourable_unfavourable, period_overlay_curves, period_regime_table,
    period_two_feature_heatmap, period_value_column, period_year_stability,
    regime_table, two_feature_heatmap, year_stability,
)

st.set_page_config(page_title="Strategy Diagnostics Dashboard", layout="wide")
_POS, _NEG = "#27AE60", "#E74C3C"


@st.cache_data(show_spinner="Loading model run…")
def _load(path: str) -> DashboardData:
    return load_dashboard_data(path)


# --------------------------------------------------------------------------- #
# Cached compute wrappers. Streamlit reruns the whole script on every widget
# change, so these memoise the heavy slices keyed by (data, params). The
# Adjusted RAR config is passed as ``_rar`` (underscore = not hashed) with its
# values folded into ``rar_key`` so the cache still invalidates if the formula
# changes. Spinners show only on an actual (cache-miss) recompute.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Computing regime buckets…")
def c_regime_table(df, feature, n_buckets, categorical, per_trade, cap, rar_key, _rar):
    if per_trade:
        return regime_table(df, feature, _rar, cap, n_buckets=n_buckets, categorical=categorical)
    return period_regime_table(df, feature, n_buckets=n_buckets, categorical=categorical)


@st.cache_data(show_spinner="Ranking favourable vs hostile conditions…")
def c_favourable(df, numeric, n_buckets, per_trade, cap, rar_key, _rar):
    if per_trade:
        return favourable_unfavourable(df, list(numeric), _rar, cap, n_buckets=n_buckets)
    return period_favourable_unfavourable(df, list(numeric), n_buckets=n_buckets)


@st.cache_data(show_spinner="Building two-feature regime map…")
def c_two_feature(df, fx, fy, n_bins, per_trade, cap, rar_key, _rar):
    if per_trade:
        return two_feature_heatmap(df, fx, fy, _rar, cap, n_bins=n_bins)
    return period_two_feature_heatmap(df, fx, fy, n_bins=n_bins)


@st.cache_data(show_spinner="Recomputing overlay economics…")
def c_overlay_economics(df, allow_q, reduce_f, cap, rar_key, _rar):
    return overlay_economics(df, _rar, cap, allow_quantile=allow_q, reduce_factor=reduce_f)


@st.cache_data(show_spinner=False)
def c_overlay_curves(df, allow_q, cap):
    return overlay_equity_curves(df, cap, allow_quantile=allow_q)


@st.cache_data(show_spinner=False)
def c_period_overlay(df, allow_q):
    return period_overlay_curves(df, allow_quantile=allow_q)


@st.cache_data(show_spinner="Computing year-by-year stability…")
def c_year_stability(df, per_trade, cap, rar_key, _rar):
    if per_trade:
        return year_stability(df, _rar, cap)
    return period_year_stability(df)


@st.cache_data(show_spinner="Screening factors for performance drivers…")
def c_factor_screen(df, numeric, n_buckets):
    return factor_screen(df, list(numeric), n_buckets=n_buckets)


@st.cache_data(show_spinner=False)
def c_factor_decile(df, feature, n_buckets):
    return factor_decile_profile(df, feature, n_buckets=n_buckets)


@st.cache_data(show_spinner=False)
def c_categorical_factor(df, feature):
    return categorical_factor_table(df, feature)


def _rar_key(dd: DashboardData):
    r = dd.adjusted_rar
    return (r.bars_per_year, r.weight_by_r_squared, r.clip_min, r.clip_max)


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

    # Honest banner when no model improved on simply running the strategy as-is.
    no_edge = ("best_policy" in lb.columns and (lb["best_policy"] == "baseline").all()) \
        or str(top.get("model", "")).lower().startswith("baseline")
    if no_edge:
        st.warning("**No model beat the baseline.** Every model's best policy is just "
                   "*take every trade/period*, so the models found no exploitable edge with "
                   "these features and target. The headline RAR% is the strategy's own "
                   "baseline — identical across rows by construction. Try a different target "
                   "(e.g. `next_period_return`), the per-trade view, or richer features.")

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
    if df.empty:
        st.info("No analysis frame for this view.")
        return
    numeric = [f for f in dd.numeric_features(view) if f in df.columns]
    categorical = [f for f in dd.categorical_features(view) if f in df.columns]
    feats = numeric + categorical
    if not feats:
        st.info("No features available to slice.")
        return
    per_trade = is_per_trade(df)
    vcol = None if per_trade else period_value_column(df)

    if per_trade:
        st.caption("Slice the strategy's realised **trade economics** by any feature. "
                   "Green bars mark favourable feature ranges; red mark hostile ones.")
    else:
        st.caption(f"Slice the **mean per-period outcome** (`{vcol}`) by any feature. "
                   "Green bars mark favourable feature ranges; red mark hostile ones.")

    c1, c2 = st.columns([2, 1])
    feature = c1.selectbox("Feature to slice by", feats, index=0)
    n_buckets = c2.slider("Buckets", 2, 10, 5)

    # --- single-feature bucket bar ---
    if per_trade:
        tbl = c_regime_table(df, feature, n_buckets, feature in categorical, True,
                             dd.initial_capital, _rar_key(dd), dd.adjusted_rar)
        ycol, ylabel = "adjusted_rar", "Adjusted RAR%"
        hover = ["count", "hit_rate", "avg_return_pct", "max_drawdown_pct"]
    else:
        tbl = c_regime_table(df, feature, n_buckets, feature in categorical, False,
                             dd.initial_capital, _rar_key(dd), dd.adjusted_rar)
        ycol, ylabel = "mean_outcome", f"Mean outcome ({vcol})"
        hover = ["count", "pct_positive", "total_outcome"]

    if tbl.empty:
        st.info("Not enough data to bucket this feature.")
    else:
        fig = px.bar(tbl, x="bucket", y=ycol, color=tbl[ycol] >= 0,
                     color_discrete_map={True: _POS, False: _NEG},
                     hover_data=[h for h in hover if h in tbl.columns],
                     labels={ycol: ylabel, "bucket": feature})
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # --- favourable vs hostile shortlist ---
    st.subheader("Favourable vs hostile shortlist (all features)")
    st.caption("Every feature bucket ranked — the strategy's best and worst conditions at a glance.")
    fav = c_favourable(df, tuple(numeric), n_buckets, per_trade, dd.initial_capital,
                       _rar_key(dd), dd.adjusted_rar)
    if fav.empty:
        st.caption("Need at least one numeric feature with enough data to rank.")
    else:
        fav_disp = fav.copy()
        fav_disp["regime"] = fav_disp["regime"].map({"favourable": "🟢 favourable",
                                                     "hostile": "🔴 hostile"})
        st.dataframe(fav_disp, use_container_width=True, hide_index=True)

    # --- two-feature regime map ---
    st.subheader("Two-feature regime map")
    if len(numeric) < 2:
        st.caption("Add at least two numeric features (include more data families) to "
                   "see the two-feature regime map.")
    else:
        cc1, cc2, cc3 = st.columns(3)
        fx = cc1.selectbox("X feature", numeric, index=0)
        fy = cc2.selectbox("Y feature", numeric, index=min(1, len(numeric) - 1))
        nb = cc3.slider("Grid bins", 2, 6, 4)
        heat = c_two_feature(df, fx, fy, nb, per_trade, dd.initial_capital,
                             _rar_key(dd), dd.adjusted_rar)
        clabel = "Adjusted RAR%" if per_trade else f"Mean outcome ({vcol})"
        if heat.empty:
            st.caption("Not enough overlapping data in the grid for these two features.")
        else:
            fig = px.imshow(heat, color_continuous_scale="RdYlGn", aspect="auto",
                            labels={"x": fx, "y": fy, "color": clabel}, text_auto=True)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    # --- regime timeline ---
    st.subheader("Regime timeline")
    if dd.regime_timeline.empty:
        st.caption("A favourable/neutral/hostile timeline is produced for **per-period** "
                   "runs that use a regime-label target.")
    else:
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
    if not df.empty:
        st.subheader("Stability through time")
        pt = is_per_trade(df) and "entry_date" in df.columns
        ys = c_year_stability(df, pt, dd.initial_capital, _rar_key(dd), dd.adjusted_rar)
        if pt:
            ycol, ylabel, hover = "adjusted_rar", "Adjusted RAR%", ["count", "hit_rate"]
        else:
            ycol, ylabel, hover = "mean_outcome", "Mean outcome", ["count", "pct_positive"]
        if ys.empty:
            st.caption("Not enough dated rows to assess year-by-year stability.")
        else:
            fig = px.bar(ys, x="year", y=ycol, color=ys[ycol] >= 0,
                         color_discrete_map={True: _POS, False: _NEG},
                         hover_data=[h for h in hover if h in ys.columns],
                         labels={ycol: ylabel})
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)


def page_drivers(dd: DashboardData, view: str, df: pd.DataFrame) -> None:
    st.header("What factors drive better performance?")
    st.caption("A model-agnostic screen of which factors are associated with **better "
               "realised performance**. Univariate and *descriptive* (correlated factors "
               "can echo one another, and association is not causation) — cross-check "
               "against the held-out model importance on the **Factors** page.")
    if df.empty:
        st.info("No analysis frame for this view.")
        return
    vcol = performance_column(df)
    if vcol is None:
        st.info("No performance column available to screen against.")
        return
    numeric = [f for f in dd.numeric_features(view) if f in df.columns]
    categorical = [f for f in dd.categorical_features(view) if f in df.columns]
    if not numeric and not categorical:
        st.info("No features available to screen.")
        return

    n_buckets = st.slider("Buckets (top vs bottom comparison)", 3, 10, 5)
    st.caption(f"Performance measured by `{vcol}`. p-values are Benjamini-Hochberg "
               "corrected across factors; ✅ marks factors that survive FDR control.")

    screen = c_factor_screen(df, tuple(numeric), n_buckets) if numeric else pd.DataFrame()
    if screen.empty:
        st.info("Not enough data to screen numeric factors.")
    else:
        top = screen.head(15).iloc[::-1]
        fig = px.bar(top, x="rank_corr", y="feature", orientation="h",
                     color=top["rank_corr"] >= 0,
                     color_discrete_map={True: _POS, False: _NEG},
                     hover_data=["effect_top_minus_bottom", "auc", "n",
                                 "p_value_bh", "year_consistency"],
                     labels={"rank_corr": "Rank correlation with performance"})
        fig.update_layout(showlegend=False, height=520,
                          title="Factor → performance association (Spearman rank correlation)")
        st.plotly_chart(fig, use_container_width=True)

        disp = screen.copy()
        disp.insert(0, "FDR", np.where(disp["significant"], "✅", ""))
        disp = disp.drop(columns=["significant", "abs_rank_corr"])
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption("**effect_top_minus_bottom** = mean performance of the top feature "
                   "bucket minus the bottom bucket (economic effect size). **auc** = how "
                   "well the factor alone separates above/below-median trades. "
                   "**year_consistency** = fraction of years the effect keeps the same sign.")

        st.subheader("Factor detail")
        sel = st.selectbox("Inspect a factor", list(screen["feature"]))
        prof = c_factor_decile(df, sel, min(10, max(4, n_buckets * 2)))
        if not prof.empty:
            fig = px.bar(prof, x="bucket", y="mean_performance",
                         color=prof["mean_performance"] >= 0,
                         color_discrete_map={True: _POS, False: _NEG},
                         hover_data=["count", "pct_positive", "feature_mid"],
                         labels={"mean_performance": f"Mean {vcol}",
                                 "bucket": f"{sel} (low → high)"})
            fig.update_layout(showlegend=False, height=340,
                              title=f"Mean performance across {sel} buckets")
            st.plotly_chart(fig, use_container_width=True)
            samp = df.dropna(subset=[sel, vcol])
            if len(samp) > 1500:
                samp = samp.sample(1500, random_state=0)
            fig2 = px.scatter(samp, x=sel, y=vcol, opacity=0.4,
                              labels={vcol: f"performance ({vcol})"})
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)

    if categorical:
        st.subheader("Categorical factors")
        cat = st.selectbox("Category to compare", categorical)
        ctab = c_categorical_factor(df, cat)
        if ctab.empty:
            st.caption("Not enough data per category to compare.")
        else:
            head = ctab.head(25)
            fig = px.bar(head, x="category", y="mean_performance",
                         color=head["mean_performance"] >= 0,
                         color_discrete_map={True: _POS, False: _NEG},
                         hover_data=["count", "pct_positive"],
                         labels={"mean_performance": f"Mean {vcol}"})
            fig.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ctab, use_container_width=True, hide_index=True)

    st.info("These are univariate, in-sample associations. A factor that ranks high "
            "here **and** shows held-out permutation importance on the Factors page "
            "**and** stays consistent across years is the most trustworthy driver.")


def page_scoring(dd: DashboardData, view: str, df: pd.DataFrame) -> None:
    st.header("Is a reduce-size / filter overlay justified?")
    if df.empty or "good_score" not in df.columns:
        st.info("This run has no finalist scores to build overlays from.")
        return
    per_trade = is_per_trade(df)

    if per_trade:
        st.caption("Translate the finalist's calibrated score into allow / reduce / "
                   "block overlays and recompute economics live on the same OOS trades.")
        c1, c2 = st.columns(2)
        allow_q = c1.slider("Allow: keep top quantile", 0.0, 0.95, 0.70, 0.05)
        reduce_f = c2.slider("Reduce-size factor (hostile half)", 0.0, 1.0, 0.50, 0.05)
        econ = c_overlay_economics(df, allow_q, reduce_f, dd.initial_capital,
                                   _rar_key(dd), dd.adjusted_rar)
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
        curves = c_overlay_curves(df, allow_q, dd.initial_capital)
        if not curves.empty:
            fig = px.line(curves, x="date", y="equity", color="policy",
                          title="Out-of-sample equity: baseline vs top-quantile overlay")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
    else:
        vcol = period_value_column(df)
        st.caption(f"Per-period overlay: take exposure only when the finalist score is "
                   f"favourable, then compare cumulative outcome (`{vcol}`) to always-exposed.")
        allow_q = st.slider("Expose when score is in the top quantile", 0.0, 0.95, 0.50, 0.05)
        curves, summary = c_period_overlay(df, allow_q)
        if not summary.empty:
            st.dataframe(summary, use_container_width=True, hide_index=True)
        if not curves.empty:
            fig = px.line(curves, x="period", y="cumulative", color="policy",
                          title=f"Cumulative {vcol}: always exposed vs favourable-only")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # --- score distribution (both views) ---
    st.subheader("Score distribution")
    work = df.dropna(subset=["good_score"]).copy()
    color = "target" if "target" in work.columns and work["target"].nunique() <= 10 else None
    fig = px.histogram(work, x="good_score", color=color, nbins=40, barmode="overlay",
                       opacity=0.7)
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    # --- calibration (reliability) curve, if present ---
    cal = next((obj for name, obj in dd.interpretation.items()
                if name.startswith("calibration_") and isinstance(obj, dict)
                and obj.get("mean_predicted")), None)
    if cal:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="grey"),
                                 name="Perfectly calibrated"))
        fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"],
                                 mode="lines+markers", name="Model"))
        fig.update_layout(title="Calibration (reliability) curve", height=360,
                          xaxis_title="Mean predicted probability",
                          yaxis_title="Observed frequency")
        st.plotly_chart(fig, use_container_width=True)

    # --- best & worst scored rows (columns adapt to the view) ---
    st.subheader("Best & worst scored rows")
    cols = [c for c in ("trade_id", "period", "symbol", "entry_date", "period_ts",
                        "good_score", "pl_pct", "pl", "target") if c in work.columns]
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
        "Does it work?", "Regimes", "Performance drivers", "Factors",
        "Scoring & overlays", "Walk-forward", "Robustness & search",
    ])
    if page == "Does it work?":
        page_overview(dd, view)
    elif page == "Regimes":
        page_regimes(dd, view, df)
    elif page == "Performance drivers":
        page_drivers(dd, view, df)
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
