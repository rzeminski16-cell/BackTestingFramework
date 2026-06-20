# Modelling & Evaluation — Data Dictionary

Field-level reference for everything the Modelling & Evaluation tool produces:
**features**, **targets**, **outcome columns**, **model outputs**, and every
**metric** in the leaderboard, economics tables and dashboard.

> See also: [User Guide](MODELLING_USER_GUIDE.md) and
> [Dashboard Guide](DASHBOARD_GUIDE.md).

---

## 1. Naming conventions

Feature columns use a `namespace__name` scheme (double underscore separates
parts):

```
trade__concurrent_positions            # trade-intrinsic feature
equity_prices__close                   # symbol-specific family value
equity_prices__close__ret5             # 5-step trailing return of that value
index_panel_SPX__close                 # market-wide family, per entity (SPX)
index_panel_SPX__close__ret1           # 1-step trailing return
macro_panel_US_GDP__value__age_days    # freshness of that value at trade entry
```

| Element | Meaning |
|---------|---------|
| `trade__…` | Computed from the trade record itself (known at entry). |
| `<family>__…` | **Symbol-specific** family (joined to the trade's own symbol): `equity_prices`, `corporate_actions`, `fundamentals_pit`. No entity suffix. |
| `<family>_<entity>__…` | **Market-wide** family (one column set per series): `index_panel`, `fx_panel`, `commodities_panel`, `macro_panel`, `utilities_panel`. The entity is the series id (e.g. `SPX`, `VIX`, `US_GDP`, `WTI`). |
| `…__<value_col>` | The underlying panel value (e.g. `close`, `rate`, `value`, a fundamentals field). |
| `…__ret1`, `…__ret5` | Trailing 1-step and 5-step percentage change of the value, computed **within each entity's own history** (so no cross-series leakage). |
| `…__age_days` | Freshness: days between the trade's `entry_date` and the attached value's availability timestamp. **Always ≥ 0** (a structural no-leakage guarantee). |

**Leakage rule.** Every family value is attached with a *backward as-of join* on
the value's `available_ts`, bounded by the family's
`carry_forward_tolerance_days`. Nothing attached post-dates the trade entry.
Outcome columns (P/L, exit fields, etc.) are **never** features.

---

## 2. Trade-intrinsic features (`trade__*`)

Known at or before entry; present in the per-trade view.

| Feature | Type | Description |
|---------|------|-------------|
| `trade__concurrent_positions` | numeric | Other positions open at entry (portfolio crowding). |
| `trade__entry_capital_required` | numeric | Capital this position needed. |
| `trade__entry_capital_available` | numeric | Capital available at entry. |
| `trade__entry_equity` | numeric | Total portfolio equity at entry. |
| `trade__entry_price` | numeric | Entry price per unit. |
| `trade__quantity` | numeric | Units/shares traded. |
| `trade__commission_paid` | numeric | Total commission on the trade. |
| `trade__capital_pressure` | numeric | `entry_capital_required / entry_capital_available` — capital-budget tightness at entry. |
| `trade__entry_month` | numeric (1–12) | Calendar month of entry (seasonality). |
| `trade__entry_dayofweek` | numeric (0–6) | Day of week of entry. |
| `trade__entry_quarter` | numeric (1–4) | Calendar quarter of entry. |
| `trade__symbol` | categorical | Security symbol (one-hot; rare levels bucketed). |
| `trade__side` | categorical | Trade side (e.g. LONG). |
| `trade__currency` | categorical | Security currency. |
| `trade__entry_reason` | categorical | Strategy's entry reason/tag. |

*(A `trade__*` feature only appears if the source column exists and is numeric.)*

---

## 3. Family as-of features

Each included family contributes, **per value column**, three numeric features
plus a freshness feature. The value columns depend on the family:

| Family | Kind | Default value column(s) |
|--------|------|-------------------------|
| `equity_prices` | symbol-specific | `close` (primary; OHLCV reduced to close by default) |
| `index_panel` | market-wide | `close` per index (e.g. benchmark, `VIX`) |
| `fx_panel` | market-wide | `rate` per pair |
| `commodities_panel` | market-wide | `value` per series (e.g. `WTI`, `BRENT`) |
| `macro_panel` | market-wide | `value` per series (e.g. `US_GDP`, `YIELD_10Y`) |
| `fundamentals_pit` | symbol-specific | every numeric fundamentals field (e.g. `reported_eps`, …) |
| `corporate_actions` | symbol-specific | numeric event fields (e.g. `amount`, `split_factor`) |
| `utilities_panel` | market-wide | numeric status fields |

For each `<value_col>` you get:

| Suffix | Description |
|--------|-------------|
| `__<value_col>` | The as-of **level** at entry (e.g. the benchmark close known at the trade's entry). |
| `__<value_col>__ret1` | Trailing 1-observation % change (short-term momentum/direction). |
| `__<value_col>__ret5` | Trailing 5-observation % change (medium-term trend). |
| `__age_days` | Days since that value became available (staleness; one per family/entity). |

> **Capping.** A market-wide family with more than 40 distinct series keeps the
> most-populated 40 (recorded as a feature-build warning) to bound width. All-NaN
> derived columns (e.g. `__ret5` with too little history) are dropped. ±inf from
> divide-by-zero in returns is mapped to NaN and imputed.

### Per-period portfolio-state features

In the per-period view, alongside market-wide family features:

| Feature | Description |
|---------|-------------|
| `open_trades` | Number of strategy trades open during the period. |
| `closed_trades` | Trades that closed in the period. |
| `period_realised_pl` | Realised P/L booked in the period (the portfolio process). |

---

## 4. Targets

What the models predict (the `target` column holds the **primary** one).

### Per-trade

| Target | Type | Definition |
|--------|------|-----------|
| `binary_good_trade` | binary | `1` if `pl_pct > cost_buffer_pct` (cost-aware "good" trade), else `0`. |
| `continuous_net_return` | continuous | `pl_pct` clipped to `±return_clip_pct` (economic granularity without outlier domination). |
| `binary_tail_loss` | binary | `1` if `pl_pct ≤ tail_loss_pct` (a tail-loss event), else `0`. |

### Per-period

| Target | Type | Definition |
|--------|------|-----------|
| `next_period_return` | continuous | Next period's realised P/L ÷ initial capital. |
| `next_period_adjusted_rar` | continuous | Adjusted RAR% over a short **forward** window of the period equity path. |
| `regime_label` | categorical | `favourable` / `neutral` / `hostile`, from next-period return vs the configured thresholds. |

---

## 5. The `analysis_frame_<view>.parquet` (dashboard backbone)

One tidy row per trade (or period): all features above **plus**:

| Column | View | Description |
|--------|------|-------------|
| `trade_id` / `period` | both | Row identifier (the id column named in the manifest). |
| `target` | both | The primary target value for this row. |
| `entry_date`, `exit_date` | per-trade | Trade timestamps. |
| `symbol`, `side` | per-trade | From the trade record (also filterable in the dashboard). |
| `pl` | per-trade | Realised profit/loss in base currency. |
| `pl_pct` | per-trade | Realised profit/loss as a percentage. |
| `period_ts` | per-period | The period timestamp. |
| `open_trades`, `closed_trades`, `period_realised_pl` | per-period | Portfolio-state aggregates. |
| `regime` | per-period | The favourable/neutral/hostile label (if produced). |
| `oos__<model>` | both | Out-of-sample prediction from each model (probability for classifiers; value for regressors). |
| `good_score` | both | The finalist's polarity-adjusted attractiveness score (higher = more attractive; inverted for tail-loss targets). Drives overlays. |
| `finalist_model` | both | Name of the finalist the `good_score` came from. |

---

## 6. Metric glossary

All economic metrics are computed on a **calendar-daily, forward-filled equity
curve** built from the relevant trades, reusing the framework's centralised
metric engine.

### Primary selection metric

| Metric | Description |
|--------|-------------|
| **Adjusted RAR%** (`adjusted_rar`, `oos_adjusted_rar`) | The house metric: **RAR%** (annualised return from a log-equity regression) **× R²** (curve straightness), so noisy equity curves are penalised. Configurable (bars/year, R²-weighting, clipping). The leaderboard's primary ranking key. |
| **Baseline Adjusted RAR%** (`baseline_adjusted_rar`) | Adjusted RAR% of the strategy with **no** overlay (all OOS trades), the reference the overlay is judged against. |

### Economic metrics (per overlay policy)

| Metric | Description |
|--------|-------------|
| `n_trades` | Trades retained after the policy. |
| `trade_frequency` | Trades per year (guards against "wins" that just starve the strategy). |
| `sharpe` | Return per unit of total volatility (risk-free 3.5%, 252 trading days). |
| `sortino` | Like Sharpe but penalising only downside deviation. |
| `max_drawdown_pct` | Largest peak-to-trough equity decline (%). |
| `hit_rate` | Percentage of winning trades. |
| `profit_factor` | Gross wins ÷ gross losses (>1 = profitable; capped at 999.99). |
| `total_return` | Total percentage return over the period. |
| `avg_return_pct` | Mean trade return (%). |

### Overlay policies (the `policy` column in economics tables)

| Policy | Meaning |
|--------|---------|
| `baseline` | Take every OOS trade (no model gating). |
| `top_quantile_only` | Keep only trades scoring in the top `1 − allow_quantile`. |
| `reduce_size_in_hostile` | Scale P/L of the less-attractive half by the reduce-size factor. |
| `regime_exposure` | (per-period) Take exposure only in predicted-favourable periods. |

### Model-quality diagnostics (leaderboard `q_*` columns)

Classification:

| Field | Description |
|-------|-------------|
| `q_balanced_accuracy` | Accuracy averaged over classes (robust to imbalance). |
| `q_average_precision` | Area under precision-recall — ranking of rare positives. |
| `q_roc_auc` | Area under ROC — overall ranking quality. |
| `q_brier` | Mean squared error of probabilities (**lower = better**; calibration). |

Regression:

| Field | Description |
|-------|-------------|
| `q_mae` | Mean absolute error (lower better). |
| `q_rmse` | Root mean squared error (lower better). |
| `q_r2` | Coefficient of determination. |
| `q_spearman` | Rank correlation between prediction and outcome. |

---

## 7. Leaderboard columns (`model_leaderboard.csv`)

| Column | Description |
|--------|-------------|
| `rank` | Position (1 = finalist). |
| `model` | Model name. |
| `tier` | `baseline` / `linear` / `tree` / `ensemble`. |
| `oos_adjusted_rar` | Best overlay's OOS Adjusted RAR% (the ranking key). |
| `baseline_adjusted_rar` | No-overlay Adjusted RAR%. |
| `passes_guardrails` | Whether the best overlay kept enough trade frequency and contained drawdown. |
| `best_policy` | The overlay that achieved `oos_adjusted_rar`. |
| `calibrated` | Whether probabilities were calibrated in-fold. |
| `q_*` | Model-quality diagnostics (above). |

---

## 8. Regime-table columns (dashboard "Regimes" page)

Per feature bucket:

| Column | Description |
|--------|-------------|
| `bucket` | The feature range (quantile interval) or category. |
| `count` | Trades in the bucket (**watch this — low counts are noisy**). |
| `adjusted_rar` | Adjusted RAR% of trades in the bucket (favourable vs hostile). |
| `hit_rate` | Win rate (%) in the bucket. |
| `avg_return_pct` | Mean trade return (%). |
| `total_return_pct` | Total return (%) of the bucket. |
| `sharpe` | Sharpe of the bucket. |
| `max_drawdown_pct` | Max drawdown (%) within the bucket. |

The **favourable/hostile shortlist** adds a `feature` column and a `regime`
label (`favourable` / `hostile`) ranking all buckets by Adjusted RAR%.

---

## 9. Guardrail & robustness fields

Guardrails (per model):

| Field | Description |
|-------|-------------|
| `min_trade_frequency` | Minimum trades/year the overlay must keep (vs baseline). |
| `overlay_trade_frequency` | The overlay's actual trades/year. |
| `max_drawdown_cap_pct` | Drawdown ceiling the overlay must respect. |
| `overlay_drawdown_pct` | The overlay's actual max drawdown. |
| `adjusted_rar_delta` | Overlay − baseline Adjusted RAR%. |
| `passes` | Whether all guardrails held. |

Robustness (`robustness.json`):

| Field | Description |
|-------|-------------|
| `bootstrap_delta` | `{point, lo, hi, p_value}` — bootstrap CI for the finalist's Adjusted RAR% delta vs baseline. |
| `permutation_test` | `{score, p_value, n_permutations}` — significance vs feature/target independence. |
| `multiple_testing` | `{method, alpha, reject[], pvals_corrected[], models[]}` — Holm correction across models. |
| `whites_reality_check` | `{best_model, statistic, p_value, n_models}` — data-snooping test for the best-of-many. |

Risk register (`risk_register.json`) — each row: `risk`, `check`, `status`
(`pass`/`warn`), `detail`. Covers preprocessing leakage, time leakage, overlap
leakage, hyper-parameter optimism, search overfitting, interpretation misuse
(correlated features), and calibration misuse.

---

## 10. Configuration parameters (`model_config.json`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `view` | `per_trade` | `per_trade` / `per_period` / `dual`. |
| `target.primary` | `binary_good_trade` | The predicted target. |
| `target.cost_buffer_pct` | `0.20` | Cost+buffer threshold for "good" trades (%). |
| `target.tail_loss_pct` | `-5.0` | Tail-loss threshold (%). |
| `target.return_clip_pct` | `25.0` | Clip for the continuous target (±%). |
| `target.period_freq` | `W` | Period bucket for per-period (`D`/`W`/`M`). |
| `adjusted_rar.bars_per_year` | `365` | Annualisation basis for Adjusted RAR%. |
| `adjusted_rar.weight_by_r_squared` | `true` | Multiply RAR% by R² (penalise noisy curves). |
| `validation.design` | `purged_embargoed` | Walk-forward design. |
| `validation.n_splits` | `5` | Outer folds. |
| `validation.embargo_days` | `5` | Embargo around test blocks (days). |
| `validation.nested` | `true` | Nested inner search for tuned models. |
| `validation.inner_splits` | `3` | Inner folds for tuning. |
| `weight_mode` | `equal` | `equal` / `class` / `economic` sample weighting. |
| `initial_capital` | `100000` | Capital for equity-curve construction. |
| `top_quantile` | `0.70` | Allow overlay: keep the top `1 − q`. |
| `reduce_size_factor` | `0.50` | Size multiplier for the less-attractive half. |

---

## 11. Upstream provenance columns (from the run package)

These live on the family panels (not in the feature matrix) but govern how
features are joined and are worth understanding:

| Column | Description |
|--------|-------------|
| `observation_date` | The period/date a value describes. |
| `available_ts` | **The as-of join key** — earliest moment the value could have been known. |
| `report_date` | Explicit release date (fundamentals) where available. |
| `retrieved_at` | When the value was ingested from the vendor. |
| `native_frequency` | `daily` / `weekly` / `monthly` / `quarterly` / `snapshot`. |
| `quality_flag` | `ok` / `carried_forward` / `stale` / `inferred_timestamp` / `revision_risk` / `snapshot`. |
| `carry_forward_tolerance_days` | Max age a value may be carried forward in the as-of join (per family, in the manifest). |
| `revision_risk_flag` / `geo_scope` | Macro-specific: latest-history (not vintage) and US-centric markers. |

See [Data Preparation](DATA_PREPARATION.md) for the full upstream contract.
