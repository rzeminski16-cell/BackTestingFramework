"""
Tests for DataPrep panel sources and the controller (headless, no GUI).

Builds a small temp data tree (local CSVs) plus a fake Alpha Vantage client and
drives the full controller flow: load trades -> assemble panels -> validate ->
export package.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Classes.DataCollection.alpha_vantage_client import APIResponse
from Classes.DataPrep.schema import Family, PROVENANCE_COLUMNS
from Classes.DataPrep.run import RunConfig, ModellingFrequency
from Classes.DataPrep.sources import PanelSourceBuilder
from Classes.DataPrep.controller import DataPrepController


class FakeAV:
    """Minimal Alpha Vantage client returning canned payloads for sources."""

    def get_commodity(self, function, interval="monthly"):
        return APIResponse(success=True, data={
            "name": function, "interval": interval, "unit": "USD",
            "data": [{"date": "2022-01-01", "value": "80"},
                     {"date": "2022-02-01", "value": "85"}],
        })

    def get_economic_indicator(self, function, interval=None, maturity=None):
        return APIResponse(success=True, data={
            "name": function, "interval": interval or "monthly", "unit": "pct",
            "data": [{"date": "2021-12-01", "value": "3.1"},
                     {"date": "2022-01-01", "value": "3.4"}],
        })

    def get_dividends(self, symbol):
        return APIResponse(success=True, data={
            "symbol": symbol,
            "data": [{"ex_dividend_date": "2022-02-09", "amount": "0.22"}],
        })

    def get_splits(self, symbol):
        return APIResponse(success=True, data={"symbol": symbol, "data": []})


def _make_data_tree(root: Path):
    (root / "raw_data" / "daily").mkdir(parents=True)
    (root / "raw_data" / "forex").mkdir(parents=True)
    (root / "raw_data" / "benchmarks").mkdir(parents=True)
    (root / "processed_data" / "fundamentals").mkdir(parents=True)
    (root / "logs" / "single_security" / "run1").mkdir(parents=True)

    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    px = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": [100 + i * 0.1 for i in range(len(dates))],
        "volume": 1_000_000,
    })
    px.to_csv(root / "raw_data" / "daily" / "AAPL_daily.csv", index=False)
    spx = px.copy(); spx["symbol"] = "SPX"
    spx.to_csv(root / "raw_data" / "benchmarks" / "SPX_daily.csv", index=False)

    fx = pd.DataFrame({
        "date": ["2022-01-07", "2022-01-14", "2022-01-21"],
        "symbol": "GBPUSD", "open": 1.35, "high": 1.36, "low": 1.34, "close": 1.355,
    })
    fx.to_csv(root / "raw_data" / "forex" / "GBPUSD_weekly.csv", index=False)

    fund = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "frequency": ["quarterly", "quarterly"],
        "fiscaldateending": ["2021-12-31", "2022-03-31"],
        "report_date": ["2022-01-27", "2022-04-28"],
        "reporttime": ["post-market", "post-market"],
        "reported_eps": [2.10, 1.52],
    })
    fund.to_csv(root / "processed_data" / "fundamentals" / "AAPL_fundamental.csv", index=False)

    trades = pd.DataFrame({
        "trade_id": ["T1", "T2"],
        "symbol": ["AAPL", "AAPL"],
        "entry_date": ["2022-02-01", "2022-03-01"],
        "exit_date": ["2022-02-10", "2022-03-10"],
        "pl": [10.0, -5.0], "pl_pct": [0.1, -0.05],
        "currency": ["USD", "USD"],
    })
    tp = root / "logs" / "single_security" / "run1" / "strat_AAPL_trades.csv"
    trades.to_csv(tp, index=False)
    return str(tp)


class TestSourcesAndController(unittest.TestCase):
    def test_full_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trade_path = _make_data_tree(root)

            cfg = RunConfig(run_name="Source Test", base_currency="GBP",
                            modelling_frequency=ModellingFrequency.DAILY)
            for fam in (Family.EQUITY_PRICES, Family.FX, Family.FUNDAMENTALS,
                        Family.INDEX, Family.COMMODITIES, Family.MACRO,
                        Family.CORPORATE_ACTIONS):
                cfg.families[fam].include = True
            cfg.families[Family.COMMODITIES].series = ["WTI", "BRENT"]
            cfg.families[Family.MACRO].series = ["CPI", "FEDERAL_FUNDS_RATE"]
            cfg.benchmark_map = {"equity": ["SPX"]}

            controller = DataPrepController(
                cfg,
                logs_dir=str(root / "logs"),
                raw_data_dir=str(root / "raw_data"),
                processed_dir=str(root / "processed_data"),
                av_client=FakeAV(),
                output_root=str(root / "out"),
            )

            summary, issues = controller.load_trades([trade_path])
            self.assertEqual(issues, [])
            self.assertEqual(summary.trade_count, 2)

            warnings = controller.assemble()
            # All seven families should have produced data.
            self.assertIn(Family.EQUITY_PRICES, controller.panels)
            self.assertIn(Family.FX, controller.panels)
            self.assertIn(Family.FUNDAMENTALS, controller.panels)
            self.assertIn(Family.INDEX, controller.panels)
            self.assertIn(Family.COMMODITIES, controller.panels)
            self.assertIn(Family.MACRO, controller.panels)
            self.assertIn(Family.CORPORATE_ACTIONS, controller.panels)

            # Every panel carries the provenance contract.
            for fam, panel in controller.panels.items():
                for col in PROVENANCE_COLUMNS:
                    self.assertIn(col, panel.columns, f"{fam} missing {col}")

            report = controller.validate()
            codes = {f.code for f in report.findings}
            # weekly FX on a daily run should be flagged.
            self.assertIn("weekly_fx_daily_model", codes)
            # GBP base with USD trades + FX included -> conversion feasible (no error).
            self.assertFalse(report.is_blocking)

            package = controller.export(report)
            run_dir = Path(package.run_dir)
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "commodities_panel.parquet").exists())
            self.assertTrue((run_dir / "macro_panel.parquet").exists())
            self.assertTrue((run_dir / "fundamentals_pit.parquet").exists())

            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "Source_Test")
            self.assertGreaterEqual(manifest["table_row_counts"]["commodities_panel"], 4)

            # Fundamentals available_ts uses the explicit report_date.
            fund = pd.read_parquet(run_dir / "fundamentals_pit.parquet")
            row = fund[fund["observation_date"] == pd.Timestamp("2021-12-31")].iloc[0]
            self.assertEqual(pd.Timestamp(row["available_ts"]), pd.Timestamp("2022-01-27"))

    def test_no_av_client_skips_av_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_data_tree(root)
            cfg = RunConfig(run_name="No AV")
            cfg.families[Family.COMMODITIES].include = True
            cfg.families[Family.COMMODITIES].series = ["WTI"]
            builder = PanelSourceBuilder(
                raw_data_dir=str(root / "raw_data"),
                processed_dir=str(root / "processed_data"),
                av_client=None,
            )
            panels, warnings = builder.build_all(cfg, ["AAPL"], ["USD"])
            self.assertNotIn(Family.COMMODITIES, panels)
            self.assertTrue(any("Commodities" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
