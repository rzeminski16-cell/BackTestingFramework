"""
Tests for the standalone DataPrep run-package pipeline.

Covers timing/availability rules, run/manifest config round-trips, the trade
source loader, entity mapping, family normalisation (point-in-time stamping),
the validation checklist, and an end-to-end package export written to a temp dir.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Classes.DataPrep.schema import Family, PROVENANCE_COLUMNS, QualityFlag
from Classes.DataPrep.timing import (
    TimingPolicy, AvailabilityRule, MissingDataPolicy, compute_available_ts,
    default_timing_for,
)
from Classes.DataPrep.run import RunConfig, FamilyConfig, ModellingFrequency, slugify
from Classes.DataPrep.trade_source import TradeSource, _infer_asset_class
from Classes.DataPrep.entity_mapping import EntityMapper
from Classes.DataPrep.families import normalise_family_panel
from Classes.DataPrep.validation import Validator, Severity
from Classes.DataPrep.builder import RunBuilder, ExportBlocked


def _trades():
    return pd.DataFrame({
        "trade_id": ["T1", "T2", "T3"],
        "symbol": ["AAPL", "MSFT", "AAPL"],
        "entry_date": ["2022-01-10", "2022-03-15", "2022-06-01"],
        "exit_date": ["2022-01-20", "2022-03-25", "2022-06-10"],
        "pl": [100.0, -50.0, 25.0],
        "pl_pct": [1.0, -0.5, 0.25],
        "currency": ["USD", "USD", "USD"],
    })


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
class TestTiming(unittest.TestCase):
    def test_same_day(self):
        obs = pd.Series(pd.to_datetime(["2022-01-03", "2022-01-04"]))
        out = compute_available_ts(obs, TimingPolicy(availability_rule=AvailabilityRule.SAME_DAY))
        self.assertTrue((out == obs).all())

    def test_publication_lag(self):
        obs = pd.Series(pd.to_datetime(["2022-01-01"]))
        out = compute_available_ts(
            obs, TimingPolicy(availability_rule=AvailabilityRule.PUBLICATION_LAG, publication_lag_days=30))
        self.assertEqual(out.iloc[0], pd.Timestamp("2022-01-31"))

    def test_same_day_hardens_to_next_session(self):
        # 2022-01-07 is a Friday; next business day is Monday 2022-01-10.
        obs = pd.Series(pd.to_datetime(["2022-01-07"]))
        policy = TimingPolicy(availability_rule=AvailabilityRule.SAME_DAY, allow_same_day_close=False)
        self.assertEqual(policy.effective_rule(), AvailabilityRule.NEXT_SESSION)
        out = compute_available_ts(obs, policy)
        self.assertEqual(out.iloc[0], pd.Timestamp("2022-01-10"))

    def test_report_date_with_fallback(self):
        obs = pd.Series(pd.to_datetime(["2022-01-01", "2022-04-01"]))
        rd = pd.Series([pd.NaT, pd.Timestamp("2022-04-20")])
        policy = TimingPolicy(availability_rule=AvailabilityRule.REPORT_DATE, publication_lag_days=45)
        out = compute_available_ts(obs, policy, rd)
        self.assertEqual(out.iloc[0], pd.Timestamp("2022-02-15"))  # fallback obs+45
        self.assertEqual(out.iloc[1], pd.Timestamp("2022-04-20"))  # explicit report date


# --------------------------------------------------------------------------- #
# run / manifest
# --------------------------------------------------------------------------- #
class TestRunConfig(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("My Run #1 / test"), "My_Run_1_test")
        self.assertEqual(slugify(""), "unnamed_run")

    def test_default_families_all_excluded(self):
        cfg = RunConfig(run_name="r")
        self.assertEqual(len(cfg.families), len(Family.ordered()))
        self.assertEqual(cfg.included_families(), [])

    def test_roundtrip(self):
        cfg = RunConfig(run_name="Run A", base_currency="GBP")
        cfg.families[Family.COMMODITIES].include = True
        cfg.families[Family.COMMODITIES].series = ["WTI", "BRENT"]
        cfg.benchmark_map = {"equity": ["SPX"]}
        rt = RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        self.assertEqual(rt.run_id, "Run_A")
        self.assertTrue(rt.families[Family.COMMODITIES].include)
        self.assertEqual(rt.families[Family.COMMODITIES].series, ["WTI", "BRENT"])
        self.assertEqual(rt.benchmark_map, {"equity": ["SPX"]})


# --------------------------------------------------------------------------- #
# trade source
# --------------------------------------------------------------------------- #
class TestTradeSource(unittest.TestCase):
    def test_infer_asset_class(self):
        self.assertEqual(_infer_asset_class("GBPUSD"), "fx")
        self.assertEqual(_infer_asset_class("AAPL"), "equity")

    def test_load_and_summarise(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "strat_AAPL_trades.csv"
            _trades().to_csv(p, index=False)
            ts = TradeSource(tmp)
            df, issues = ts.load([p])
            self.assertEqual(issues, [])
            summary = ts.summarise(df)
            self.assertEqual(summary.trade_count, 3)
            self.assertEqual(summary.n_symbols, 2)
            self.assertEqual(summary.date_range[0], "2022-01-10")

    def test_validate_missing_keys(self):
        bad = pd.DataFrame({"symbol": ["AAPL"], "entry_date": ["2022-01-01"]})
        issues = TradeSource.validate(bad)
        self.assertTrue(any("trade_id" in i for i in issues))

    def test_discover_prefers_portfolio(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "portfolio" / "run1"
            d.mkdir(parents=True)
            _trades().to_csv(d / "portfolio_trades.csv", index=False)
            _trades().to_csv(d / "AAPL_trades.csv", index=False)
            found = TradeSource(tmp).discover()
            # The per-symbol file in the same dir is suppressed.
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["kind"], "portfolio")


# --------------------------------------------------------------------------- #
# entity mapping
# --------------------------------------------------------------------------- #
class TestEntityMapping(unittest.TestCase):
    def test_benchmark_resolution_order(self):
        trades = _trades()
        trades["asset_class"] = "equity"
        mapper = EntityMapper(
            benchmark_map={"AAPL": ["NDX"], "equity": ["SPX"]},
            base_currency="GBP",
        )
        m = mapper.build(trades)
        aapl = m.table[m.table["symbol"] == "AAPL"].iloc[0]
        msft = m.table[m.table["symbol"] == "MSFT"].iloc[0]
        self.assertEqual(aapl["benchmarks"], "NDX")   # symbol-specific wins
        self.assertEqual(msft["benchmarks"], "SPX")   # falls back to asset_class
        self.assertTrue(m.table["needs_fx_conversion"].all())  # USD vs GBP base

    def test_unmapped_benchmarks_reported(self):
        m = EntityMapper(benchmark_map={}).build(_trades())
        self.assertEqual(len(m.unmapped_benchmarks), 3)


# --------------------------------------------------------------------------- #
# family normalisation
# --------------------------------------------------------------------------- #
class TestNormalisation(unittest.TestCase):
    def test_commodity_panel_stamped(self):
        raw = pd.DataFrame({
            "series_id": ["WTI", "WTI"],
            "observation_date": pd.to_datetime(["2022-01-01", "2022-02-01"]),
            "value": [80.0, None],
            "native_frequency": ["monthly", "monthly"],
            "source_function": ["WTI", "WTI"],
        })
        panel = normalise_family_panel(
            raw, family=Family.COMMODITIES, run_id="r",
            timing=default_timing_for(Family.COMMODITIES),
            entity_id_col="series_id", value_col="value",
        )
        self.assertEqual(list(panel.columns[:len(PROVENANCE_COLUMNS)]), PROVENANCE_COLUMNS)
        self.assertTrue((panel["run_id"] == "r").all())
        self.assertEqual(panel["family"].iloc[0], Family.COMMODITIES.value)
        self.assertIn("available_ts", panel.columns)
        # second row had null value -> flagged missing
        self.assertEqual(panel["quality_flag"].iloc[1], QualityFlag.MISSING.value)

    def test_macro_panel_revision_risk(self):
        raw = pd.DataFrame({
            "series_id": ["CPI"],
            "observation_date": pd.to_datetime(["2022-01-01"]),
            "value": [300.0],
            "revision_risk_flag": [True],
        })
        panel = normalise_family_panel(
            raw, family=Family.MACRO, run_id="r",
            timing=default_timing_for(Family.MACRO),
            entity_id_col="series_id", value_col="value",
        )
        self.assertEqual(panel["quality_flag"].iloc[0], QualityFlag.REVISION_RISK.value)
        # macro publication lag of 30 days applied
        self.assertEqual(panel["available_ts"].iloc[0], pd.Timestamp("2022-01-31"))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
class TestValidation(unittest.TestCase):
    def _config_with(self, *families):
        cfg = RunConfig(run_name="v", base_currency="GBP")
        for fam in families:
            cfg.families[fam].include = True
        return cfg

    def test_currency_infeasible_without_fx(self):
        cfg = self._config_with(Family.COMMODITIES)
        mapper = EntityMapper(base_currency="GBP")
        mapping = mapper.build(_trades())  # trades are USD
        report = Validator().validate(cfg, _trades(), mapping, {})
        codes = {f.code for f in report.findings}
        self.assertIn("fx_conversion_infeasible", codes)

    def test_macro_not_lagged_is_warning(self):
        cfg = self._config_with(Family.MACRO)
        cfg.families[Family.MACRO].timing.availability_rule = AvailabilityRule.SAME_DAY
        report = Validator().validate(cfg, _trades(), EntityMapper().build(_trades()),
                                      {Family.MACRO: pd.DataFrame()})
        codes = {f.code for f in report.findings}
        self.assertIn("macro_not_lagged", codes)

    def test_missing_trade_keys_blocks(self):
        cfg = RunConfig(run_name="v")
        bad = pd.DataFrame({"symbol": ["AAPL"]})
        report = Validator().validate(cfg, bad, None, {})
        self.assertTrue(report.is_blocking)


# --------------------------------------------------------------------------- #
# end-to-end build
# --------------------------------------------------------------------------- #
class TestEndToEnd(unittest.TestCase):
    def test_export_writes_package(self):
        cfg = RunConfig(run_name="E2E Run", base_currency="USD",
                        modelling_frequency=ModellingFrequency.DAILY)
        cfg.families[Family.COMMODITIES].include = True
        cfg.families[Family.COMMODITIES].series = ["WTI"]
        cfg.benchmark_map = {"equity": ["SPX"]}

        raw_commodity = pd.DataFrame({
            "series_id": ["WTI"] * 6,
            "observation_date": pd.to_datetime(
                ["2022-01-01", "2022-02-01", "2022-03-01",
                 "2022-04-01", "2022-05-01", "2022-06-01"]),
            "value": [78, 88, 100, 105, 110, 115],
            "native_frequency": ["monthly"] * 6,
            "source_function": ["WTI"] * 6,
            "unit": ["USD per barrel"] * 6,
        })

        with tempfile.TemporaryDirectory() as tmp:
            builder = RunBuilder(cfg, output_root=tmp)
            builder.set_trades(_trades())
            builder.add_panel(
                Family.COMMODITIES, raw_commodity,
                entity_id_col="series_id", value_col="value",
            )
            report = builder.validate()
            self.assertFalse(report.is_blocking)
            package = builder.export(report)

            run_dir = Path(package.run_dir)
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "selected_trades.parquet").exists())
            self.assertTrue((run_dir / "commodities_panel.parquet").exists())
            self.assertTrue((run_dir / "entity_mapping.parquet").exists())
            self.assertTrue((run_dir / "data_contract.json").exists())
            self.assertTrue((run_dir / "validation_summary.html").exists())

            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "E2E_Run")
            self.assertEqual(manifest["base_currency"], "USD")
            self.assertTrue(manifest["family_toggles"]["commodities_panel"])
            self.assertEqual(manifest["table_row_counts"]["selected_trades"], 3)

            # Parquet round-trips and keeps the provenance contract.
            panel = pd.read_parquet(run_dir / "commodities_panel.parquet")
            for col in PROVENANCE_COLUMNS:
                self.assertIn(col, panel.columns)

            contract = json.loads((run_dir / "data_contract.json").read_text())
            self.assertIn("commodities_panel", contract["families"])
            self.assertIn("available_ts", contract["timestamp_semantics"])

    def test_export_blocked_on_errors(self):
        cfg = RunConfig(run_name="bad")
        with tempfile.TemporaryDirectory() as tmp:
            builder = RunBuilder(cfg, output_root=tmp)
            builder.set_trades(pd.DataFrame({"symbol": ["AAPL"]}))  # missing keys
            with self.assertRaises(ExportBlocked):
                builder.export()


if __name__ == "__main__":
    unittest.main()
