# Reading the Results Dashboard

A guide to every page, chart and table in the interactive **Modelling &
Evaluation** dashboard, with heuristics for what "good" and "bad" look like and
worked interpretation examples.

The dashboard is the primary way to explore a model run in depth. It is organised
around the diagnostic questions from the design docs and **recomputes economics
live** using the same Adjusted RAR% / metric functions the leaderboard used, so
what you see can never silently diverge from how models were scored.

> **See also:** [User Guide](MODELLING_USER_GUIDE.md) (the process) and
> [Data Dictionary](MODELLING_DATA_DICTIONARY.md) (every field and metric).

---

## Launching & loading a run

Three ways to open it:

```bash
# 1. Directly
streamlit run apps/modelling_dashboard.py

# 2. Pointed at a specific run (skips the picker)
MODEL_RUN_DIR=processed_data/runs/<src>/modelling/<id> streamlit run apps/modelling_dashboard.py

# 3. From the GUI — the "Results Dashboard" launcher card, or the
#    "Open dashboard" button on the modelling tool's export dialog.
```

It opens in your browser (usually `http://localhost:8501`).

**Loading & speed.** Streamlit re-runs the script on every interaction, so the run
load and the heavier slices (the favourable/hostile shortlist, regime maps, overlay
economics, stability) are **cached** and only recompute when their inputs actually
change. A small spinner (e.g. *"Ranking favourable vs hostile conditions…"*) appears
only during a real recompute; switching pages or nudging an unrelated control reuses
the cached result. A change to the **filters**, the selected **feature/buckets**, or
the **overlay sliders** invalidates just the affected computation.

### Sidebar (always visible)

- **Runs directory** — where exported model runs live (`processed_data/runs`).
- **Model run** — pick `<source_run> / <model_run>`. You can switch runs anytime
  to compare variants.
- **View** — `per_trade`, `per_period`, or both (for a dual run). The caption
  shows the primary target and the finalist model.
- **Filters** (expander) — restrict every page to a subset:
  - *Symbol* / *Side* multi-selects, and a *Year range* slider.
  - The caption shows "X of Y rows after filters". Use this to ask *"does the edge
    hold for just US tech?"* or *"...only in 2022?"*
- **Page** — the seven pages below.

> Filters apply to the **trade-level** pages (Regimes, Scoring & overlays). The
> Overview/Factors/Walk-forward/Robustness pages reflect the run as exported.

---

## Page 1 — "Does it work?" (the verdict)

**Question:** is there a credible, out-of-sample edge worth acting on?

- **Headline metrics:** the finalist model, its **OOS Adjusted RAR%**, the
  **Baseline Adjusted RAR%** (with the delta), and **Guardrails** (PASS/REVIEW).
  - *Read:* a positive delta means the model-driven overlay improves the
    strategy's risk-adjusted return out of sample. PASS means it does so without
    starving trade frequency or blowing out drawdown.
- **Model leaderboard:** every model, ranked by OOS Adjusted RAR%. Columns
  include the baseline, guardrail flag, best overlay policy, whether the model is
  calibrated, and `q_*` quality diagnostics (see Data Dictionary).
  - *Read:* prefer the **simplest** model near the top that passes guardrails. A
    boosting model that barely beats logistic is usually not worth the opacity.
- **Significance & robustness:** White's Reality Check p, bootstrap ΔAdjRAR p,
  permutation-test p, and the bootstrap confidence interval for the delta.
  - *Read:* small p-values (e.g. < 0.05) and a confidence interval that **excludes
    zero** support a real effect. A great Adjusted RAR% with a weak White's
    Reality Check p is a red flag for a search artefact.
- **Leakage & overfitting risk register:** ✅/⚠️ per mandatory check.
  - *Read:* any ⚠️ tells you which assumption to scrutinise (e.g. calibration
    absent → treat threshold overlays as rank-only).
- **"No model beat the baseline" banner.** If every model's best policy is just
  *take every trade/period*, the page shows a warning and the headline RAR% is the
  strategy's own baseline (identical across rows by construction). That's an honest
  *no exploitable edge* result, not a glitch — try a different target (e.g.
  `next_period_return`), the per-trade view, or richer features. Significance is
  reported for the best **non-baseline** model, never the tie-broken dummy.

**Worked example.** Finalist OOS AdjRAR 14.2 vs baseline 9.1 (Δ +5.1), guardrails
PASS, White's RC p = 0.03, bootstrap CI [1.2, 8.9] excludes zero → a defensible
edge. If instead the CI were [−2.0, 11.0], the point estimate is encouraging but
not yet significant.

---

## Page 2 — "Regimes" (what conditions are favourable?)

**Question:** in which feature ranges does the strategy thrive or struggle? This
is the core diagnostic page.

- **Feature to slice by + Buckets:** pick any feature; the tool buckets it (equal
  count quantiles for numerics, categories for categoricals) and computes the
  strategy's realised economics **per bucket**.
- **Bar chart:** Adjusted RAR% per bucket. **Green = favourable, red = hostile.**
  Hover for count, hit rate, average return and max drawdown.
  - *Read:* a clean monotonic gradient (e.g. Adjusted RAR% rising with the
    feature) is a strong, interpretable signal. A single green spike on a
    low-count bucket is likely noise — check the `count` column.
- **Bucket table:** the numbers behind the bars (count, Adjusted RAR%, hit rate,
  avg/total return, Sharpe, max drawdown).
- **Favourable vs hostile shortlist (all features):** every feature bucket across
  all numeric features, ranked by Adjusted RAR%; the top are 🟢 favourable, the
  bottom 🔴 hostile.
  - *Read:* this is your one-glance "operating statement". Watch `count` — a
    favourable bucket backed by 4 trades is not actionable.
- **Two-feature regime map:** a heatmap of Adjusted RAR% over a quantile grid of
  two features (green good, red bad).
  - *Read:* reveals *interactions* — e.g. the strategy only works when momentum is
    high **and** volatility is low (one green corner).
- **Regime timeline** (per-period runs): the favourable/neutral/hostile sequence
  over time.

**Worked example.** Slicing by `index_panel_VIX__close` shows green in the lowest
two VIX buckets (AdjRAR > 12, ~60 trades each) and deep red in the top bucket
(AdjRAR −20, 45 trades). Interpretation: *the strategy is a low-volatility
phenomenon; in high-VIX regimes it bleeds.* The two-feature map vs
`macro_panel_*__value` can then show whether that's modulated by the macro
backdrop.

> **Per-trade vs per-period.** On a **per-trade** run, buckets show realised trade
> **economics** (Adjusted RAR%, hit rate, drawdown). On a **per-period** run there is
> no per-trade P/L, so buckets show the **mean per-period outcome** (next-period
> return / realised period P/L) and `% positive` instead — same favourable/hostile
> reading, different metric. The shortlist and two-feature map adapt automatically.

> **Caveat:** these are realised-economics slices, not causal claims. Highly
> correlated features can tell the same story twice — see the Factors page's
> correlation caution.

---

## Page 3 — "Performance drivers" (what factors suggest better trades?)

**Question:** which factors are associated with *better realised performance* — a
direct, model-agnostic screen, complementary to the model-based Factors page.

- **Factor → performance bars:** every numeric factor ranked by its **Spearman
  rank correlation** with performance (green = higher value → better, red =
  higher → worse).
  - *Read:* the longest green/red bars are the strongest linear-in-rank drivers.
- **Factor table:** for each factor — `direction`, `rank_corr`, FDR-corrected
  `p_value_bh` (✅ = survives Benjamini-Hochberg control across all factors),
  `effect_top_minus_bottom` (mean performance of the top bucket minus the bottom —
  the economic effect size), `auc` (how well the factor *alone* separates
  above/below-median trades), `n`, and `year_consistency` (fraction of years the
  effect keeps the same sign).
  - *Read:* trust a factor most when it has a meaningful effect, **survives FDR**,
    a high AUC, *and* high year-consistency. A big effect with low consistency or a
    non-significant p is likely noise (many factors → some look good by chance —
    that's exactly why FDR control is applied).
- **Factor detail:** pick a factor to see **mean performance across its deciles**
  (is the relationship monotonic, or U-shaped?) and a raw scatter of the factor vs
  performance.
- **Categorical factors:** mean performance per category (which symbols / sides /
  entry reasons perform best/worst).

**Worked example.** `index_panel__close__ret5` tops the list with rank-corr +0.34
(✅ FDR), effect +1.8% top-vs-bottom, AUC 0.66, year-consistency 1.0 → recent
benchmark strength is a robust driver of trade quality. A trade-intrinsic factor
with effect +2.0% but p_value_bh 0.4 and consistency 0.4 is *not* trustworthy.

> **Caveat:** these are **univariate, in-sample associations** — descriptive, not
> causal, and correlated factors echo one another. A driver you can rely on ranks
> high here **and** shows held-out permutation importance on the next page **and**
> stays consistent across years.

## Page 4 — "Factors" (which variables drive it?)

**Question:** what did the finalist actually learn?

- **Linear coefficients (signed):** horizontal bars, green positive / red
  negative, for a linear/logistic finalist.
  - *Read:* sign and magnitude on the *standardised* features. A large positive
    coefficient on `index_panel__close__ret5` means recent benchmark strength
    raises the predicted trade quality.
- **Permutation importance (held-out):** how much each input feature matters,
  measured by score drop when it's shuffled — computed on **held-out** data, so
  it reflects genuine predictive value, not in-sample fit.
  - *Read:* this is the most trustworthy global importance. If a feature tops
    coefficients but is low here, the model isn't really relying on it out of
    sample.
- **SHAP summary** (ensemble finalists, if `shap` is installed): mean absolute
  contribution per feature.
- **Stability through time:** per-year Adjusted RAR% bars.
  - *Read:* a green-every-year pattern is a robust edge; alternating green/red
    suggests the edge is regime-specific (cross-reference the Regimes page).

> If features are highly correlated, single-feature interpretations (and
> PDP/ICE-style reasoning) can mislead. The export's risk register flags the count
> of correlated pairs.

---

## Page 5 — "Scoring & overlays" (is an overlay justified?)

**Question:** does turning the model into an allow/reduce/block rule actually help,
and at what threshold?

- **Allow: keep top quantile** and **Reduce-size factor** sliders.
  - *Allow 0.70* keeps the top 30% of scored trades (block the rest).
  - *Reduce factor 0.50* halves size on the less-attractive half.
- **Live policy cards:** baseline vs `top_quantile_only` vs
  `reduce_size_in_hostile`, each showing Adjusted RAR% (with delta vs baseline),
  trade count, Sharpe, max drawdown and hit rate — **recomputed instantly** as you
  move the sliders.
  - *Read:* find the sweet spot where Adjusted RAR% improves **without** cutting
    trade count or hit rate to ribbons. If aggressive filtering only helps by
    discarding most trades, that's fragile.
- **Out-of-sample equity curve:** baseline vs the top-quantile overlay over time.
  - *Read:* you want the overlay line above the baseline *consistently*, not from
    one lucky stretch.
- **Score distribution:** histogram of the finalist's `good_score`, coloured by
  the realised target.
  - *Read:* good separation (positive outcomes shifted right) means the score
    ranks trades well.
- **Calibration (reliability) curve:** predicted probability vs observed
  frequency, against the diagonal.
  - *Read:* points on the diagonal = trustworthy probabilities (safe for
    thresholds). A bowed curve means the raw scores are mis-scaled and thresholds
    should be treated cautiously.
- **Best & worst scored trades:** the highest- and lowest-scoring trades with
  their features and outcomes — useful for sanity-checking individual calls.

> **Per-period runs** show a single *exposure* slider and compare the cumulative
> period outcome of *always exposed* vs *exposed only in favourable periods*, plus
> the score distribution and (where available) the calibration curve.

**Worked example.** At Allow 0.70 the `top_quantile_only` card shows AdjRAR 16.0
(Δ +6.9), trades 84 (of 280), Sharpe 1.3, maxDD −8% vs baseline −14%. The equity
curve sits cleanly above baseline. The calibration curve hugs the diagonal →
this is a credible, deployable filter. Pushing Allow to 0.90 spikes AdjRAR but
leaves only 28 trades — too thin to trust.

---

## Page 6 — "Walk-forward" (is it consistent across folds?)

**Question:** does the edge appear in *every* out-of-sample block, or just one?

- **Per-fold overlay − baseline Adjusted RAR%:** grouped bars per model.
  - *Read:* mostly-positive bars across folds = a stable edge. One huge positive
    fold among negatives = the headline number is carried by a single period;
    discount it.
- **Fold table:** train/test sizes and the per-fold delta.

---

## Page 7 — "Robustness & search" (did it survive scrutiny?)

**Question:** could the result be a by-product of trying many things?

- **Multiple-hypothesis correction:** Holm-corrected p-values across the compared
  models, with reject-null flags.
  - *Read:* a model still significant *after* correction is far more credible.
- **White's Reality Check:** the data-snooping test for "is the best of many
  models genuinely superior?" with its p-value.
- **Attempt ledger:** every model/target variant tried — the audit trail of your
  search. The more you tried, the more multiple-testing discipline matters.
- **Raw robustness payload:** the full JSON for completeness.

---

## A suggested reading order

1. **Does it work?** — get the verdict and check significance.
2. **Regimes** — build the favourable/hostile operating statement.
3. **Performance drivers** — see which factors most separate good from bad trades
   (with FDR control and year-consistency).
4. **Scoring & overlays** — confirm an overlay actually improves economics and
   find a sensible threshold.
5. **Factors** — explain *why* in terms of the model's learned variables.
6. **Walk-forward** + **Robustness** — stress-test before you believe it.

If steps 1, 3 and 5 disagree with step 2's nice story, trust the out-of-sample
economics and significance over the in-sample narrative.
