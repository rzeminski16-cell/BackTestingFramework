# Modelling & Evaluation — User Guide

A step-by-step guide to running the **Modelling & Evaluation** tool: the second
half of the trading-strategy diagnostics pipeline. This stage takes the
point-in-time **run package** built by the Data Preparation tool and helps you
answer, with defensible evidence:

- **When does this strategy work, and when does it fail?**
- **Which features or regimes explain that behaviour?**
- **When should you trust the next trade less?**
- **Is a model-driven overlay (filter trades / reduce size) actually justified?**

It is a *diagnostics* layer, not a market-direction forecaster. The goal is a
credible **operating statement** about favourable and hostile conditions — not a
better AUC.

> **Companion documents**
> - [Dashboard Guide](DASHBOARD_GUIDE.md) — how to read every chart and table.
> - [Data Dictionary](MODELLING_DATA_DICTIONARY.md) — every feature, target, and metric.
> - [Architecture & contract](MODELLING_EVALUATION.md) — how the code is organised.

---

## 1. What you need first (prerequisites)

The tool **consumes a prepared run package** and never re-does data preparation.
Before you start you need a run package on disk at
`processed_data/runs/<run_id>/`, produced by the Data Preparation tool
(`python ctk_data_prep_gui.py`). See [Data Preparation](DATA_PREPARATION.md).

A usable package must contain at least:

- `selected_trades.parquet` — the trades to diagnose, each with `trade_id`,
  `symbol`, `entry_date`, `exit_date`, and a P/L column (`pl` and/or `pl_pct`).
- `run_manifest.json` and `data_contract.json` — timing policies and join rules.
- Zero or more **family panels** (equity prices, index, FX, commodities, macro,
  fundamentals, …). More families = richer regime/factor analysis, but the tool
  still runs on trades alone (using trade-intrinsic features).

If a package is missing P/L or trade keys, Step 1 will block with a clear error.

---

## 2. The mental model (read this once)

Four ideas shape every choice in the tool:

1. **Two native views.** The tool can model the same strategy two ways:
   - **Per-trade** — one row per trade, scored at *entry time*. Answers *"should I
     trust this next trade?"* Good for trade filtering and size-reduction overlays.
   - **Per-period** — one row per fixed calendar period (e.g. weekly). Answers
     *"is the strategy currently in a favourable or hostile regime?"* Good for
     top-down throttling.
   - **Dual** runs both and reports them together.

2. **Chronological, never shuffled.** All validation respects time order. The
   tool walks forward through history; it never trains on the future. For
   overlapping trades it additionally *purges and embargoes* (see §6).

3. **Economic before cosmetic.** Models are ranked by **out-of-sample Adjusted
   RAR%** (a regression-based, drawdown-aware return metric — see the Data
   Dictionary), with trade-frequency and drawdown **guardrails**. Statistical
   scores (accuracy, AUC, Brier) are *diagnostics*, not the winner-picker.

4. **Interpretable before complex.** A conservative model ladder runs simple,
   explainable models first; black-box models only earn their place if they
   improve held-out economics.

**Leakage prevention is first-class.** Every feature is attached to a trade with
a *backward as-of join* on the value's availability timestamp, so nothing the
model sees was unknowable at decision time. Outcome columns (`pl`, `exit_*`, …)
are never used as features.

---

## 3. Launching

```bash
python ctk_modelling_evaluation_gui.py
# or the "Modelling & Evaluation" card in:  python ctk_main_gui.py
```

You'll see a six-step wizard with a progress stepper, a live "Current Selections"
sidebar, and Back / Next navigation. Each step validates before letting you
continue.

---

## 4. The six steps in detail

### Step 1 — Choose run package

**Decision:** which prepared run to analyse, and what to name this model run.

- **Model run name** — becomes the stable id for the output folder
  (`processed_data/runs/<source_run>/modelling/<model_run_id>/`). Use something
  descriptive, e.g. `regime_diag_v1`.
- **Runs directory** — defaults to `processed_data/runs`. Change it if your
  packages live elsewhere, then **Refresh**.
- **Run package** — pick from the discovered packages.
- **Load & check** — runs a readiness check and prints a summary:
  - Trade count, symbol count, date range, families present.
  - ⚠ warnings (e.g. a family with all-null availability timestamps — it will be
    skipped) and ✖ errors (e.g. missing P/L — you cannot proceed).

You can only continue once a package **loads and passes** the readiness check.

### Step 2 — Define targets

**Decision:** what "works" means, economically.

- **Analytical view** — per-trade, per-period, or dual (see §2).
- **Primary target** — what the models predict. Options depend on the view:
  - *Per-trade:* `binary_good_trade` (cost-aware), `continuous_net_return`
    (clipped), `binary_tail_loss`.
  - *Per-period:* `next_period_return`, `next_period_adjusted_rar`, `regime_label`
    (the 3-class favourable/neutral/hostile label is modelled as a binary
    *favourable vs rest* target; the full label still drives the regime timeline).
- **Cost+buffer threshold (%)** — a trade is labelled "good" only if its net
  return beats this. Set it to your realistic round-trip cost plus a safety
  buffer (e.g. `0.20`). **Do not** leave it at zero — "positive return" is
  usually an economically meaningless threshold.
- **Tail-loss threshold (%)** — defines a "tail loss" trade (e.g. `-5.0`).
- **Return clip (±%)** — caps the continuous target so a few outliers don't
  dominate the regression target.
- **Period frequency (D/W/M)** — only for per-period; the calendar bucket size.
- **Sample weighting:**
  - `equal` — pure diagnosis (recommended default).
  - `class` — inverse-frequency weighting to counter class imbalance.
  - `economic` — weight by |P/L| so large trades matter more. Use deliberately;
    it can make the model "learn which trades are big" rather than which are good.
- **Adjusted RAR%** — the primary selection metric, configurable:
  - *Bars per year* (default `365`) and *Weight by R²* (default on). The default
    reproduces the framework house metric (RAR% × R²). Leave it unless you have a
    house formula to match.
- **Preview targets & class balance** — shows the row count, feature count, the
  class balance (so you can see imbalance *before* training), and any
  feature-build warnings (e.g. a wide family that was capped).

### Step 3 — Choose validation design

**Decision:** how to test honestly.

- **Validation design:**
  - *Expanding walk-forward* — training window grows over time. Default for
    regular **per-period** panels.
  - *Rolling walk-forward* — fixed recent window; a sensitivity check when you
    suspect regime drift.
  - *Purged & embargoed* — the default for **per-trade** data with overlapping
    trades. Removes training trades whose outcome window overlaps a test window,
    then embargoes a buffer around each test block (see §6).
- **Outer folds** — number of walk-forward test blocks (e.g. 5).
- **Embargo (days)** — buffer around test blocks for purged designs.
- **Inner folds (tuning)** — folds for the nested hyper-parameter search.
- **Nested validation** — keep on for tuned models so tuning never inflates the
  reported score.
- **Preview folds** — shows each fold's train/test sizes and date ranges, plus
  ⚠ leakage warnings (e.g. "per-trade data should use the purged design").

### Step 4 — Run the model ladder

**Decision:** which models to compare, then run them.

Toggle the tiers (run in this order):

1. **Descriptive baseline** — the trivial reference (majority class / mean).
2. **Regularised linear / logistic** — the first explainable production candidate.
3. **Shallow decision tree** — human-readable threshold rules.
4. **Constrained gradient boosting** — opt-in; only worth it if it beats simpler
   models on held-out economics.

Plus **Tune** (small nested search) and **Calibrate probabilities** (needed for
trustworthy threshold overlays).

Click **Run model ladder**. Training runs in the background with a live progress
line (`[42%] per_trade: Logistic (regularised)`). When it finishes you'll see
"Done — N models evaluated." Only then does **Next** unlock.

> **Runtime:** scales with trades × features × folds × models, and tuning adds an
> inner loop. A few hundred trades with the default ladder is typically seconds to
> a couple of minutes; enabling the ensemble + tuning is the slowest.

### Step 5 — Review evaluation

A scrollable summary, ranked by **out-of-sample Adjusted RAR%**:

- Each model with its OOS Adjusted RAR% vs the baseline, guardrail status, and
  best overlay policy; classification/regression diagnostics; and the best
  policy's trade count, Sharpe and max drawdown.
- A **Robustness** block: White's Reality Check p-value, the bootstrap Adjusted
  RAR% delta with confidence interval, and the permutation-test p-value.

This screen is a quick read; the **dashboard** (next step) is where you explore
in depth.

### Step 6 — Interpret & export

- **Interpretation summary** for the finalist(s): top coefficients (sign), top
  held-out permutation-importance features, whether SHAP/tree-rules were captured,
  and a ⚠ count of highly-correlated feature pairs (a PDP/ICE caution).
- The **risk register** (pass/warn per leakage & overfitting check).
- **Overlay thresholds:**
  - *Allow top quantile (0–1)* — keep trades scoring in the top `1 − q` (e.g.
    `0.70` keeps the top 30%).
  - *Reduce-size factor (0–1)* — size multiplier applied to the less attractive
    half.
- Click **Export**. The tool writes the full artefact set and shows an **Open
  dashboard** button (or run `streamlit run apps/modelling_dashboard.py`).

---

## 5. The model ladder, and why it's conservative

More complex models must never displace simpler ones unless they improve the
**primary economic objective** *and* stay interpretable. The promotion logic:

| Tier | Purpose | Promote when |
|------|---------|--------------|
| Baseline | Is anything worth learning at all? | Always run |
| Linear / logistic | Explainable first candidate (signs, sparsity) | Always run |
| Shallow tree | Simple non-linearity / interactions as rules | Always run |
| Gradient boosting | Ranking lift, regime-boundary refinement | Only if it beats simpler models on **held-out economics** |

Tuning uses **small, explicit, model-specific grids** (e.g. regularisation
strength for logistic; depth/leaf size for trees; learning rate/leaves for
boosting), never giant sweeps. Boosted trees use shrinkage + early stopping
rather than unconstrained growth.

---

## 6. Validation & leakage controls (in depth)

- **Preprocessing inside folds.** Imputation, scaling and encoding live inside a
  scikit-learn `Pipeline` fit on the training fold only, so the test fold never
  influences the transforms.
- **As-of feature joins.** Each feature is attached with a *backward* `merge_asof`
  on the value's `available_ts`, bounded by the family's
  `carry_forward_tolerance_days`. A value can only be attached if it was known on
  or before the trade's entry — structural leakage protection.
- **Purge & embargo (per-trade).** Because trades overlap, a training trade whose
  `[entry_date, exit_date]` label window overlaps a test window is **removed**
  from training, and an **embargo** of *N* days around each test block drops
  near-neighbours that could leak through shared/serial outcomes.
- **Nested validation.** Tuned models are scored on an **outer** walk-forward,
  with hyper-parameters chosen on an **inner** time-aware split — so the search
  cannot inflate the final number.
- **Calibration.** Classifier probabilities are calibrated **within each fold**
  before being turned into thresholds, because allow/reduce/block overlays rely on
  probabilities reading as confidence, not just ranks.

---

## 7. Evaluation philosophy

1. **Layer one — does the model predict anything useful?** Balanced accuracy,
   average precision, ROC-AUC and Brier (classification) or MAE/RMSE/R²/Spearman
   (regression).
2. **Layer two — does it improve the economics?** Predictions are turned into a
   few interpretable **overlay policies** (always trade, top-quantile only,
   reduce-size in hostile) and backtested on Adjusted RAR%, Sharpe/Sortino,
   drawdown, trade count and hit rate.

**Guardrails.** A model only "wins" if its best overlay keeps enough trade
frequency and doesn't blow out drawdown versus the baseline. A model with
slightly weaker accuracy but materially better Adjusted RAR%, lower drawdown and
clearer regime separation is the better diagnostic asset.

---

## 8. Robustness controls (guarding against false discoveries)

Because the tool encourages comparing many targets, models, weightings and
designs, it ships data-snooping guardrails:

- **Bootstrap confidence intervals** on the finalist's overlay-vs-baseline
  Adjusted RAR% delta.
- **Permutation test** — is the cross-validated score better than chance under
  feature/target independence?
- **Multiple-hypothesis correction** (Holm) across the model comparison family.
- **White's Reality Check** — once you've tried many models, is *the best one*
  genuinely superior to the benchmark, or just the luckiest of many tries?
- **Leakage & overfitting risk register** — a formal pass/warn table.
- **Attempt ledger** — every model/target variant tried, for an auditable search.

Treat a great-looking model with a weak White's Reality Check / permutation
p-value with caution: it may be a search artefact.

---

## 9. What gets exported

Under `processed_data/runs/<source_run>/modelling/<model_run_id>/`:

- `model_leaderboard.csv`, `economic_backtest.csv`, `walk_forward_predictions.csv`,
  `walk_forward_folds.csv`
- `coefficients_*.csv`, `permutation_importance_*.csv`, `shap_summary_*.csv`,
  `tree_rules_*.txt`, `calibration_*.json` / `.png`
- `regime_timeline.csv`, `risk_register.json`, `robustness.json`,
  `attempt_ledger.csv`
- `research_report.md` — the human-readable operating statement
- `scoring_function.json` + `scoring_model.joblib` — the exportable allow/reduce/
  block trade scorer
- `analysis_frame_<view>.parquet`, `dashboard_manifest.json` — power the dashboard
- `model_config.json` — the full, reproducible configuration

Field-level descriptions are in the [Data Dictionary](MODELLING_DATA_DICTIONARY.md).

---

## 10. Reproducibility & re-running

Everything is tied to the named model run. To reproduce or compare:

- The `model_config.json` records every choice (view, targets, validation,
  ladder, weighting, Adjusted RAR% formula, thresholds).
- Re-run with a **new model run name** to compare variants side by side in the
  dashboard (it lists all runs under a package).
- The exported `scoring_function` can score new trades offline:
  `ScoringFunction.load(dir).apply(feature_matrix)`.

---

## 11. Good-practice checklist

- [ ] Cost+buffer threshold reflects **real** costs, not zero.
- [ ] Per-trade data uses the **purged & embargoed** design.
- [ ] Tuned models use **nested** validation.
- [ ] Probabilities are **calibrated** before you trust threshold overlays.
- [ ] You compared the finalist to the **baseline** on Adjusted RAR%, not just AUC.
- [ ] You checked **stability** across years/regimes (dashboard).
- [ ] You read the **risk register** and **White's Reality Check** before believing a win.
- [ ] You treated `economic` weighting and high-cardinality categoricals carefully.

---

## 12. Troubleshooting

- **"Load a run package that passes the readiness check."** The package is
  missing trade keys or P/L. Re-export it from the Data Preparation tool.
- **A family is skipped / capped.** A "kept the N most-populated series" warning
  means a family had many entities; provide an explicit feature allow-list if you
  need specific series. All-null-availability families are skipped (can't join).
- **No model passes guardrails.** The overlay isn't improving economics without
  starving the strategy or worsening drawdown — a legitimate, useful finding.
  Check the baseline: maybe there's no edge to gate.
- **Tiny sample / unstable folds.** With few trades, reduce outer folds, and read
  the stability and significance panels sceptically.
- **Dashboard won't launch.** Ensure `pip install -r requirements.txt`
  (adds `streamlit`, `plotly`), then `streamlit run apps/modelling_dashboard.py`.
