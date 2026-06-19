"""
Integration test: collect (Alpha Vantage) -> persist (file_manager) -> read from
the local store (DataPrep sources) -> normalise.

Verifies the full "land to a local store, then build from it" workflow for the
new families (commodities, macro, corporate actions, utilities), without a
network or a live API key.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Classes.DataCollection.alpha_vantage_client import APIResponse
from Classes.DataCollection.file_manager import FileManager
from Classes.DataCollection.commodity_collector import CommodityCollector
from Classes.DataCollection.macro_collector import MacroCollector
from Classes.DataCollection.corporate_actions_collector import CorporateActionsCollector

from Classes.DataPrep.schema import Family, PROVENANCE_COLUMNS
from Classes.DataPrep.run import RunConfig
from Classes.DataPrep.sources import PanelSourceBuilder


class FakeAV:
    def get_commodity(self, function, interval="monthly"):
        return APIResponse(success=True, data={
            "name": function, "interval": interval, "unit": "USD",
            "data": [{"date": "2022-01-01", "value": "80"},
                     {"date": "2022-02-01", "value": "85"}]})

    def get_economic_indicator(self, function, interval=None, maturity=None):
        return APIResponse(success=True, data={
            "name": function, "interval": interval or "monthly", "unit": "pct",
            "data": [{"date": "2022-01-01", "value": "3.4"}]})

    def get_dividends(self, symbol):
        return APIResponse(success=True, data={
            "symbol": symbol, "data": [{"ex_dividend_date": "2022-02-09", "amount": "0.22"}]})

    def get_splits(self, symbol):
        return APIResponse(success=True, data={
            "symbol": symbol, "data": [{"effective_date": "2020-08-31", "split_factor": "4.0"}]})


class TestCollectPersistRead(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_data"
            fm = FileManager(output_dir=raw)
            client = FakeAV()

            # --- collect + persist commodities ---
            for key, res in CommodityCollector(client).collect_many(["WTI", "BRENT"]).items():
                self.assertFalse(res.empty)
                fm.write_commodity_data(res.df, res.series_key, res.meta["interval"])
            self.assertTrue((raw / "commodities" / "WTI_monthly.csv").exists())
            self.assertTrue((raw / "commodities" / "BRENT_monthly.csv").exists())

            # --- collect + persist macro (incl. treasury maturity) ---
            macro = MacroCollector(client).collect_many(
                ["CPI", "TREASURY_YIELD"], treasury_maturities=["10year"])
            for sid, res in macro.items():
                interval = res.meta["interval"]
                fm.write_macro_data(res.df, res.series_id, interval)
            self.assertTrue(any((raw / "macro").glob("CPI_*.csv")))
            self.assertTrue(any((raw / "macro").glob("TREASURY_YIELD_10year_*.csv")))

            # --- collect + persist corporate actions ---
            ca = CorporateActionsCollector(client).collect("AAPL")
            fm.write_corporate_actions_data("AAPL", ca.dividends, ca.splits)
            self.assertTrue((raw / "corporate_actions" / "AAPL_dividends.csv").exists())
            self.assertTrue((raw / "corporate_actions" / "AAPL_splits.csv").exists())

            # --- read back from the local store WITHOUT a client ---
            cfg = RunConfig(run_name="store test")
            cfg.families[Family.COMMODITIES].include = True
            cfg.families[Family.COMMODITIES].series = ["WTI", "BRENT"]
            cfg.families[Family.MACRO].include = True
            cfg.families[Family.MACRO].series = ["CPI", "TREASURY_YIELD"]
            cfg.families[Family.CORPORATE_ACTIONS].include = True

            builder = PanelSourceBuilder(raw_data_dir=str(raw), av_client=None)
            panels, warnings = builder.build_all(cfg, ["AAPL"], ["USD"])

            self.assertIn(Family.COMMODITIES, panels)
            self.assertIn(Family.MACRO, panels)
            self.assertIn(Family.CORPORATE_ACTIONS, panels)

            com = panels[Family.COMMODITIES]
            self.assertEqual(set(com["entity_id"].unique()), {"WTI", "BRENT"})
            for col in PROVENANCE_COLUMNS:
                self.assertIn(col, com.columns)

            macro_panel = panels[Family.MACRO]
            self.assertTrue((macro_panel["entity_id"] == "TREASURY_YIELD_10year").any())

            ca_panel = panels[Family.CORPORATE_ACTIONS]
            self.assertEqual(set(ca_panel["event_type"].unique()), {"dividend", "split"})

    def test_market_status_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw_data"
            fm = FileManager(output_dir=raw)
            status = pd.DataFrame([
                {"market_type": "Equity", "region": "United States", "current_status": "open"},
            ])
            fm.write_market_status_data(status)
            self.assertTrue((raw / "utilities" / "market_status.csv").exists())


if __name__ == "__main__":
    unittest.main()
