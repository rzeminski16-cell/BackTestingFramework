# System Evaluation & Improvement Roadmap

**Date:** 2026-07-13
**Scope:** Full review of the framework — correctness of the analytics, UX of every surface (GUIs, dashboard, reports, docs), documentation freshness, and a prioritised roadmap to professional grade.
**Test baseline:** `pytest tests/` — **829 passed, 0 failed** (2 min 24 s, Python 3.11).

---

## 1. Executive summary

This is a genuinely capable research platform: a bar-by-bar engine with strict
look-ahead protection, portfolio backtesting with capital contention, walk-forward
and univariate optimisation, Monte Carlo analysis, a leakage-aware ML
modelling/evaluation stage with real statistical controls (purged & embargoed
splits, nested tuning, calibration, bootstrap CIs, permutation tests, White's
Reality Check, an attempt ledger), and an 11-module desktop launcher plus a
Streamlit results dashboard. The 829-test suite passes clean.

The gap to "professional grade" is not capability — it is **consistency and
hygiene**. The audit found one leaked credential, one genuine accounting bug
(short trades), one metric-definition inconsistency (RAR% annualisation) that
makes numbers non-comparable across tools, a 6.5 GB repository bloated by
committed API caches, and a README that had drifted badly from reality (fixed
alongside this report). Each is fixable without redesign.

**Verified strengths**

- **No look-ahead in the engine path.** `HistoricalDataView` hard-limits strategy
  access to bars ≤ current index; `prepare_data()` runs a NaN-pattern heuristic to
  catch `.shift(-n)` leaks in custom indicators.
- **Strict FX handling.** Missing currencies or FX series fail fast with
  actionable messages; trade records carry a security-P/L vs FX-P/L decomposition.
- **The Modelling & Evaluation stage is the strongest part of the system** —
  chronological purged/embargoed CV keyed on true label windows, preprocessing fit
  inside each fold, calibrated probabilities, an economics-first leaderboard with
  guardrails, and a robustness layer most commercial tools lack.
- **Monte Carlo** is properly vectorised, supports block bootstrap and
  drawdown-responsive sizing, and clamps wiped-out paths at zero.
- **Docs for the newer subsystems are current and good** (Data Preparation,
  Modelling guides, Dashboard guide, Data Dictionary) — the dashboard guide matches
  the live page set exactly.

---

## 2. Critical findings (fix before trusting results)

### 2.1 SECURITY — live Alpha Vantage API key committed to git

`config/data_collection/settings.json` contains a real API key and has been
tracked in git since the "New Data" commits, despite a
`settings.json.template` existing precisely to prevent this.

**Action taken in this branch:** the file is untracked (`git rm --cached`) and
added to `.gitignore`.
**Action required by you:**

1. **Rotate the key** at alphavantage.co — it is in git history, so untracking
   alone does not protect it.
2. After pulling this branch, recreate your local file:
   `cp config/data_collection/settings.json.template config/data_collection/settings.json`
   and paste in the *new* key. (Your working copy is removed by the pull because
   the file left version control.)
3. Optionally scrub history (`git filter-repo`) if the repo will ever be shared.

### 2.2 CORRECTNESS — short-trade cash ledger and equity curve are inverted

`Position.calculate_value()` returns `quantity × price` and both engines book a
short entry as a cash **outflow** and the cover as an **inflow** — long
semantics. Trade records, meanwhile, are correctly direction-aware.

Empirical repro (12-bar series falling 100 → 80, one SHORT via
`ShortOnlyBaseStrategy`-style signal, zero costs):

| Source | Reported result |
|---|---|
| Trade record (`trade.pl`) | **+£1,000 profit** ✓ |
| `final_equity` / `total_return` | **−£1,000 loss** ✗ |
| Equity curve during the trade | falls as price falls ✗ |

Every equity-derived metric (total return, Sharpe, Sortino, drawdown, CAGR,
RAR%) is therefore **inverted for short strategies**. Long-only strategies
(the AlphaTrend family) are unaffected.

**Recommended fix:** make the ledger direction-aware — book short-entry
proceeds as cash-in with a liability marked at
`entry_value + (entry − price) × qty` (or model margin explicitly), and give
`calculate_value()` a direction branch. Add an engine-level invariant test:
`sum(trade.pl) − costs == final_equity − initial_capital` for both directions.
Until then, treat `ShortOnlyBaseStrategy` results as unreliable beyond the
trade log.

### 2.3 CORRECTNESS — RAR% / Adjusted RAR% is annualised inconsistently across tools

`StableMetricsCalculator` annualises the log-equity regression slope with
`BARS_PER_YEAR = 365`, i.e. it assumes **one row = one calendar day**. Three
different cadences are actually fed in:

| Caller | Rows on the curve | Effect on RAR% |
|---|---|---|
| Modelling stage (`adjusted_rar.py`) | calendar-daily (correct — it forward-fills a daily grid) | ✓ correct |
| Excel reports, univariate optimiser | one row per **trading day** (~252/yr) | overstated ~45% in log space (`exp(slope·365)` vs `exp(slope·252)`) |
| Rule Tester (`metrics_calculator._build_equity_curve`) | one row per **trade** | wildly overstated — treats the strategy as if it traded 365×/year |

The same metric name shows non-comparable numbers in different tools, and the
Rule Tester's RAR%/R-cubed are not meaningful as annual figures.

**Recommended fix:** resample every curve to a calendar-daily grid before the
regression (reuse `build_daily_equity_curve()` from the Modelling stage — it
exists and is tested), or derive `bars_per_year = n_rows / years_elapsed` from
the date column. Add a cross-tool consistency test on a shared fixture.

### 2.4 INSTALL — `pip install -r requirements.txt` fails on a clean machine

The `ta>=0.10.0` package is **imported nowhere in the codebase** but its
sdist build requires numpy to be pre-installed, so a fresh install dies before
numpy is set up (verified on Python 3.11). **Fixed in this branch** — `ta`
removed. `setup.py` remains stale (wrong dependency list, placeholder author);
see roadmap P1.

### 2.5 HYGIENE — 6.4 GB of data artifacts committed to git

- `cache/alpha_vantage/` — **33,955 JSON API responses, 5.2 GB**, tracked.
- `raw_data/` — 220 tracked files (~93 MB) under a partially-negated ignore rule.
- `.git` itself is 1.3 GB; a fresh clone downloads all of it.
- Committed generated artifacts: `pattern_analysis_report.xlsx` (repo root),
  workbooks under `reports/`/`exports/`, and an Excel **lock file**
  (`reports/vulnerability/Base_Set_Up/~$trade_analysis_Defaults.xlsx`, removed
  in this branch).

**Recommended fix:** ignore `cache/` entirely (it is a *cache* — the collector
rebuilds it), decide deliberately whether versioned market data belongs in the
repo (if yes, use Git LFS or a data submodule; if no, distribute via the
collection tool), and stop committing generated reports. History rewrite
(`git filter-repo --path cache --invert-paths`) would shrink the repo ~80%.

---

## 3. Analytics review (methodology-level findings)

These are defensible-but-questionable choices worth a deliberate decision,
in rough priority order:

1. **Same-bar close execution.** A signal computed on bar *i*'s close fills at
   bar *i*'s close. This is the documented TradingView convention, but it is
   optimistic (you cannot act on a close you are still forming). Offer
   **next-bar-open execution** as a config option and make the choice visible in
   reports.
2. **Stops/TP evaluated on close only.** A stop at 95 with a bar low of 90 and
   close of 100 does not trigger; gaps through stops fill at the close rather
   than the stop/open. Since `high`/`low` are already recommended columns, add
   intrabar stop checks (`low <= stop` for longs) with gap-aware fills
   (`fill = min(stop, open)`), behind a config flag for backwards comparability.
3. **Sortino ratio** divides by the standard deviation of negative excess
   returns (which subtracts *their own mean*), not the root-mean-square of
   downside deviations from the target over **all** periods (the standard
   definition). `downside_deviation` separately uses raw (not excess) negative
   returns — the two disagree with each other. Pick the textbook definition
   (`sqrt(mean(min(r − target, 0)²))` over all N) and document it in the
   Metrics Glossary.
4. **Sentinel caps pollute rankings.** All-winner runs return
   `sortino = 99.99`, `profit_factor = 999.99`, `r_cubed = 999.99`. Fine for
   display ("N/A — no losses"), but these flow into optimisation scoring and
   sensitivity analysis where a capped 999.99 dominates real values. Score with
   `NaN`-aware handling instead; cap only at the formatting layer.
5. **Max drawdown NaN handling** (`Classes/Core/performance_metrics.py:519`):
   `np.nan_to_num(equity, nan=0.0)` turns any NaN equity row into a fabricated
   100% drawdown. Drop NaN rows instead.
6. **`calculate_annual_return` fallback** treats row count as calendar days
   when no date column exists (252 trading rows ≈ 0.69 "years" → CAGR
   overstated). Prefer requiring dates, or scale by 252.
7. **Win-rate units differ by layer**: `BacktestResult.win_rate` and
   `PerformanceMetrics.calculate_metrics()['win_rate']` are fractions (0–1),
   `CentralizedPerformanceMetrics.calculate_win_rate()` returns percent
   (0–100). One unit everywhere, or rename (`win_rate_pct`).
8. **Robust Sharpe** annualises overlapping 21-day rolling returns with
   `sqrt(12)`; overlapping windows autocorrelate and the 365/21 ≈ 17.4 vs 12
   periods choice is undocumented outside a code comment. Document or switch to
   non-overlapping monthly returns.
9. **Per-period Adjusted-RAR target understates its label window**
   (`targets.py`): the forward RAR is computed over up to ~6 future periods but
   `label_end` is set to t+1, so the purge in the splitter cannot fully protect
   that target. Set `label_end` to the true end of the forward window when
   `NEXT_PERIOD_ADJ_RAR` is primary.
10. **Overlay thresholds are pooled ex-post** (`evaluation.py`): the
    top-quantile cut is computed over the *entire* OOS score series, then applied
    to those same trades — a mild look-ahead in overlay economics. Compute the
    threshold per fold from training scores.
11. **HistGradientBoosting early stopping** uses sklearn's internal *random*
    validation split inside otherwise chronological folds. Acceptable, but worth
    a note — or disable early stopping and rely on the tuned iteration cap.
12. **Silent exception swallowing** in `evaluation._fit_fold`/`_predict_fold`
    (`except Exception: pass`) can hide systematic tuning/calibration failures.
    Log the exception into the run manifest at minimum.

---

## 4. UX review

### Desktop GUIs (CustomTkinter)

- **Good:** consistent dark theme via a shared `Theme`/`Colors` module, wizard
  pattern for multi-step flows, background threads with progress callbacks,
  presets for strategies and Monte Carlo/vulnerability settings.
- **The launcher is missing Walk-Forward Optimization.** `ctk_optimization_gui.py`
  works and `_launch_optimization()` exists, but no card referenced it — the
  README advertised it while the launcher couldn't reach it. **Fixed in this
  branch** (card added).
- Launch-as-subprocess means a module crash is invisible (no stderr surfaced).
  Consider capturing child stderr and showing a "module crashed" dialog.
- Three generations of GUI coexist (see §6): `ctk_*` (current), `apps/*_gui.py`
  (older tkinter), `backtesting/gui/` (legacy package). Only one is the product.
- Hard-coded `v1.0.0` in the launcher footer; no single version source.

### Streamlit dashboard

- Clean separation (all computation in unit-tested `dashboard_data.py`, layout
  in the app), diagnostic-question page structure, caching with spinners, warning
  banners tied to robustness results. This is the best UX in the system.
- Run discovery defaults to relative `processed_data/runs` — breaks if launched
  from another CWD; resolve against the repo root as the GUIs do.

### Reports

- Excel reports are rich (50+ metrics, charts, stable-metrics sheet) but RAR%
  figures are affected by §2.3, and `win_rate` unit drift (§3.7) makes
  cross-report comparison risky until standardised.

### Installation & first-run

- **Fresh install was broken** (§2.4, fixed). Remaining friction: no
  `pyproject.toml`, stale `setup.py`, no pinned/lock versions, no CI to catch
  install rot, giant clone (§2.5). The README previously pointed at eight
  non-existent docs and a deleted script (fixed alongside this report).

---

## 5. Documentation review

| Area | Verdict |
|---|---|
| README.md | **Was badly stale** — 8 dead doc links, deleted script referenced, launcher description 3 modules out of date, "Long positions only" contradicted by the short-direction code. Rewritten in this branch. |
| Obsidian vault (`docs/Home.md`, Usability/, Implementation/) | Well-structured and mostly accurate. Quick Start's Python example called a non-existent `result.sharpe_ratio` (fixed). Wiki-links (`[[...]]`) only render in Obsidian — acceptable, now noted in the README. |
| Newer guides (DATA_PREPARATION, MODELLING_*, DASHBOARD_GUIDE) | Current, detailed, user-centric. Dashboard guide verified against the live page list. |
| `tests/TEST_REPORT.md` | Frozen at Dec 2025 (39 tests vs today's 829). Refreshed in this branch. |
| Coverage gaps | No user guides for: Monte Carlo GUI, Rule Tester, Pattern Analysis, Vulnerability Visualizer, Data Collection GUI. The launcher offers 12 modules; the Usability vault documents 6. |
| Cruft | Empty `docs/2026-03-14.md`, two empty `Untitled*.canvas` files (removed in this branch). |

---

## 6. Improvement roadmap

### P0 — Trust the numbers (do first)

1. **Rotate the leaked Alpha Vantage key** and recreate the local settings file
   (§2.1 — untracking already done here).
2. **Fix short-trade accounting** in both engines + `Position.calculate_value`
   (§2.2), with a ledger-vs-trades invariant test in both directions.
3. **Unify RAR% annualisation on a calendar-daily curve** everywhere (§2.3).
4. **De-bloat the repository** (§2.5): ignore `cache/`, decide the raw-data
   policy (LFS or external), drop committed generated reports, then rewrite
   history once.

### P1 — Professional-grade engineering platform

5. **One packaging story:** replace `setup.py` with `pyproject.toml` (single
   version source, extras: `gui`, `dashboard`, `ml`, `dev`), pin a lock file,
   delete the dead `ta` entry's cousins by auditing imports (e.g. verify
   `yfinance`, `seaborn` usage).
6. **CI (GitHub Actions):** fresh-venv install + `pytest` on 3.10/3.11/3.12 +
   `ruff` lint. The install breakage (§2.4) would have been caught on day one.
7. **Delete or archive the legacy frameworks:** `backtesting/` package,
   `run_gui.py`, `examples/`, and the tkinter `apps/backtest_gui.py` /
   `apps/optimization_gui.py` duplicate the `Classes/` + `ctk_*` system with
   *different* metric implementations. Two sources of truth for Sharpe is how
   silent discrepancies are born. Keep one; `git tag legacy` the rest away.
8. **Consolidate metrics to the centralized module only:**
   `Classes/Analysis/performance_metrics.py` still re-implements Calmar/CAGR/
   profit-factor logic locally; make it a pure delegate, fix win-rate units
   (§3.7), and add golden-value tests for every metric.
9. **Structured logging** instead of `print()` in engines/collectors, with a
   per-run log file the GUIs can open.

### P2 — Analytics depth

10. **Execution realism options:** next-bar-open fills (§3.1), intrabar
    stop/TP triggers with gap handling (§3.2), per-security commission/slippage
    profiles, optional volume-participation cap.
11. **Benchmark-relative analytics:** alpha/beta, information ratio, and
    benchmark-relative drawdown in reports (benchmark data collection already
    exists — `raw_data/benchmarks` is unused by the report layer).
12. **Risk metrics:** exposure %, time-in-market, VaR/CVaR on daily returns,
    rolling Sharpe/vol charts, MAE/MFE per trade (feeds the E-ratio work you
    already have), trade-level heat (risk at stop × concurrent positions).
13. **Deflated Sharpe / SPA test** in the optimizer leaderboards, reusing the
    Reality-Check machinery already in `Classes/Modelling/robustness.py` —
    walk-forward currently reports raw OOS metrics with no multiplicity control.
14. **Cross-engine parity test:** run one strategy through single-security and
    portfolio engines on one symbol and assert near-identical trades/equity —
    cheap insurance against drift between the two code paths.
15. **Monte Carlo:** add per-path metric distributions for Calmar/Sharpe and
    an option to bootstrap *daily returns* (not just per-trade) so duration risk
    is represented.

### P3 — Product polish

16. **Unified CLI** (`python -m btf backtest|optimize|prep|model|dashboard ...`)
    so every GUI flow is scriptable/automatable; today only ad-hoc `tools/`
    scripts exist.
17. **Launcher upgrades:** surface module crashes (child stderr), disable cards
    whose prerequisites are missing (e.g. dashboard without streamlit) with a
    tooltip, single version constant.
18. **Dashboard:** absolute default run path (§4), a "compare two runs"
    page, and CSV export buttons on every table.
19. **Docs completion:** guides for the six undocumented GUI modules (§5),
    a metrics glossary entry for RAR%/R-cubed/Robust Sharpe with the exact
    formulas and the annualisation convention chosen in P0-3.
20. **Windows-path hygiene:** `cache_dir: "cache\\alpha_vantage"` in settings —
    normalise with `pathlib` so configs are portable.

### P4 — Nice to have

21. Parquet-first data store with an ingest CLI (CSV → typed, validated
    Parquet) — faster loads, schema enforcement, and smaller than CSV.
22. Plugin-style strategy discovery (entry points) instead of editing a
    registry, plus a strategy scaffold generator (`btf new-strategy`).
23. Live-paper bridge: emit signals from the newest bar of collected data on a
    schedule, so research and (simulated) production share one code path.
24. Property-based tests (hypothesis) for the engine invariants: cash + value
    conservation, no negative quantities, stop monotonicity per direction.

---

## 7. Changes made on this branch

- `README.md` rewritten — accurate module list, real doc links, honest feature
  claims, correct data layout, install notes.
- `requirements.txt` — removed unused `ta` (was breaking fresh installs).
- `ctk_main_gui.py` — added the missing Walk-Forward Optimization card; fixed
  stale docstring/comments.
- `docs/Usability/Getting Started/Quick Start.md` — fixed the API example
  (`result.sharpe_ratio` does not exist; now computes metrics properly).
- `tests/TEST_REPORT.md` — refreshed to the 2026-07-13 run (829 passed).
- `config/data_collection/settings.json` — **untracked** (key leak, §2.1) and
  ignored; template remains tracked.
- Removed cruft: empty `docs/2026-03-14.md`, `docs/Untitled.canvas`,
  `docs/Untitled 1.canvas`, committed Excel lock file under `reports/`.
- Added this report.

No engine/analytics behaviour was changed: the P0 correctness fixes (§2.2,
§2.3) alter results and deserve your sign-off on the intended semantics before
implementation.
