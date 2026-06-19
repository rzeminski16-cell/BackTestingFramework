# Data Preparation Tool

The Data Preparation tool builds a **named, point-in-time, trade-centred research
run package** that a separate modelling stage consumes to diagnose *when* a
trading strategy works or fails (regime performance and factor contribution).

It only prepares and exports inputs. It does **not** train, tune, or evaluate
models — that is the downstream stage's job. The package is the contract between
the two.

> Launch: `python ctk_data_prep_gui.py` (or the **Data Preparation** card in
> `python ctk_main_gui.py`).

---

## Core principles

- **The run is the primary object.** Everything is tied to a named run; the run
  name becomes the stable `run_id` used for folder naming and the manifest.
- **The dataset is trade-centred**, not price-panel-centred. Backtest trades
  (from `logs/`) anchor every feature.
- **Every feature row preserves two timestamps**: `observation_date` (the period
  a value describes) and `available_ts` (the earliest moment it could have been
  known in live trading). Downstream joins must be **as-of** joins on
  `available_ts`, never calendar-date equality.
- The exported package is **reproducible, auditable, and self-describing**.

---

## The six-step flow (`ctk_data_prep_gui.py`)

1. **Run setup** – run name, base currency, intended modelling frequency, notes/tags.
2. **Trade selection** – pick one or more trade logs from `logs/`; preview count,
   date range, instruments and asset-class mix; fail early on missing trade keys.
3. **Data families** – toggle families to include and pick series for commodities
   and macro.
4. **Mappings & timing** – the expert screen: benchmark mapping and per-family
   timing policy (availability rule, publication lag, carry-forward tolerance,
   same-day-close usability).
5. **Validation & preview** – assemble panels and run the pre-flight checklist
   (coverage, missingness, unmapped trades, base-currency feasibility, frequency
   mismatches, leakage). Errors block export; warnings are recorded as
   acknowledged.
6. **Export** – write the self-describing run package.

---

## Architecture

| Layer | Where | Role |
|-------|-------|------|
| Collection | `Classes/DataCollection` | Alpha Vantage client + collectors (equities, fundamentals, FX, index, **commodities, macro, corporate actions, utilities**). |
| Contract | `Classes/DataPrep/schema.py`, `run.py`, `timing.py` | Families, package inventory, provenance columns, run/manifest model, availability & missing-data policies. |
| Assembly | `Classes/DataPrep/sources.py`, `families.py`, `entity_mapping.py` | Build normalised, point-in-time family panels and the trade→entity mapping. |
| Validation | `Classes/DataPrep/validation.py` | Pre-flight checklist with severities. |
| Output | `Classes/DataPrep/package_writer.py` | Parquet tables + manifest + data contract + validation report/summary. |
| Orchestration | `Classes/DataPrep/builder.py`, `controller.py` | `RunBuilder` (export blocks on errors) and the GUI-free `DataPrepController`. |
| GUI | `ctk_data_prep_gui.py` | Six-step wizard (thin view over the controller). |

The pipeline is **standalone** — independent of `Classes/FactorAnalysis`.

---

## The run package (output contract)

Written to `processed_data/runs/<run_id>/`:

| Artefact | Purpose |
|----------|---------|
| `run_manifest.json` | Run name, timestamps, family toggles, timing policies, benchmark/currency/sector maps, base currency, missing-data policies, acknowledged warnings, output inventory, row counts. |
| `family_config.json` | Per-family settings (so downstream jobs can rebuild/verify assumptions). |
| `data_contract.json` | Mandatory columns, timestamp semantics, leakage-sensitive fields, join semantics, per-family native frequencies/units. |
| `selected_trades.parquet` | The exact trade subset, with stable `trade_id`. |
| `entity_mapping.parquet` | trade → symbol / benchmark(s) / currency / sector. |
| `equity_prices.parquet`, `corporate_actions.parquet`, `fundamentals_pit.parquet`, `index_panel.parquet`, `fx_panel.parquet`, `commodities_panel.parquet`, `macro_panel.parquet`, `utilities_panel.parquet` | Normalised per-family panels (only included families). |
| `validation_report.json` / `validation_summary.html` | Machine- and human-readable quality outputs. |

### Provenance columns (every family panel)

```
run_id, family, entity_id, observation_date, available_ts,
native_frequency, source_function, source_vendor, retrieved_at, quality_flag
```

Macro panels additionally carry `geo_scope` and `revision_risk_flag` (Alpha
Vantage macro is US-centric and latest-history, not revision-vintage).

### Timestamp semantics

- `observation_date` — the period/observation the value describes.
- `available_ts` — earliest moment the value could have been known; **the as-of
  join key**.
- `report_date` — explicit fundamentals release date where available.
- `retrieved_at` — vendor ingestion time.

### Consuming the package (modelling stage)

Join features to trades with a **backward `merge_asof`** on `available_ts`,
bounded by each family's `carry_forward_tolerance_days` — never on calendar
equality. Treat `quality_flag == revision_risk` and any `inferred_timestamp`
rows with caution, and respect the leakage-sensitive fields listed in
`data_contract.json`.

---

## Collecting the source data

The data-prep stage reads from a local store first and only falls back to a live
Alpha Vantage call if a family is absent (so runs stay reproducible). Populate
the store with the **Data Collection** tool (`python apps/data_collection_gui.py`),
which now has tabs for every family the data-prep stage consumes:

| Collection tab | Writes to | Feeds data-prep family |
|----------------|-----------|------------------------|
| Daily | `raw_data/daily/` | Equity prices |
| Benchmarks | `raw_data/benchmarks/` | Index / regime (incl. VIX) |
| Forex (weekly **or daily**) | `raw_data/forex/` | FX |
| Fundamental | `raw_data/fundamentals/` | Fundamentals (PIT) |
| **Commodities** | `raw_data/commodities/` | Commodities |
| **Macro** | `raw_data/macro/` | Macro |
| **Corporate Actions** | `raw_data/corporate_actions/` | Corporate actions |

Commodities, macro, and corporate actions require an Alpha Vantage key in
`config/data_collection/settings.json`. Collect them once; the data-prep stage
then builds from the stored CSVs. Daily FX is available in the Forex tab for
exact daily base-currency conversion (the default GBP pairs are weekly, which the
validator flags for daily runs).

## Extending

- **New Alpha Vantage family**: add an endpoint + method in
  `Classes/DataCollection/alpha_vantage_client.py`, a collector that normalises
  to a tidy `observation_date`/`value` panel, a `write_*` method + directory in
  `Classes/DataCollection/file_manager.py`, a tab in
  `apps/data_collection_gui.py`, then a builder in `Classes/DataPrep/sources.py`
  (local-store-first, live fallback).
- **New canonical family**: add a `Family` member (its value is the table stem),
  a default `TimingPolicy` in `DEFAULT_FAMILY_TIMING`, and a source builder.
- **New validation rule**: add a check to `Classes/DataPrep/validation.py`.

Tests: `tests/test_data_collection_expansion.py`, `tests/test_dataprep.py`,
`tests/test_dataprep_sources.py`.
