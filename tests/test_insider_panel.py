"""
Tests for the insider-data upgrade:
- DataTransformer.transform_insider_transactions: correct AV field mapping,
  is_executive derivation, buy/sell normalization, value, cleaning/dedup.
- insider_panel: clean_transactions + point-in-time rolling features (no
  look-ahead, executive net, cluster-buy flag).
- InsiderLoader: executive metrics now populate via is_executive.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Classes.DataCollection.file_manager import DataTransformer
from Classes.FactorAnalysis.data import insider_panel as ip
from Classes.FactorAnalysis.data.insider_loader import InsiderLoader


def _av_response():
    """Alpha Vantage INSIDER_TRANSACTIONS-shaped payload."""
    return {"data": [
        {"transaction_date": "2021-01-10", "ticker": "AAA", "executive": "Jane Smith",
         "executive_title": "Chief Executive Officer", "security_type": "Common Stock",
         "acquisition_or_disposal": "A", "shares": "1000", "share_price": "50"},
        {"transaction_date": "2021-01-12", "ticker": "AAA", "executive": "Bob Jones",
         "executive_title": "Director", "security_type": "Common Stock",
         "acquisition_or_disposal": "D", "shares": "500", "share_price": "52"},
        {"transaction_date": "2021-01-12", "ticker": "AAA", "executive": "Pat Lee",
         "executive_title": "10% Owner", "security_type": "Common Stock",
         "acquisition_or_disposal": "A", "shares": "200", "share_price": "51"},
        # zero shares -> dropped
        {"transaction_date": "2021-01-15", "ticker": "AAA", "executive": "X",
         "executive_title": "Officer", "security_type": "Common Stock",
         "acquisition_or_disposal": "A", "shares": "0", "share_price": "50"},
        # exact duplicate of row 1 -> deduped
        {"transaction_date": "2021-01-10", "ticker": "AAA", "executive": "Jane Smith",
         "executive_title": "Chief Executive Officer", "security_type": "Common Stock",
         "acquisition_or_disposal": "A", "shares": "1000", "share_price": "50"},
    ]}


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.df = DataTransformer.transform_insider_transactions(_av_response())

    def test_field_mapping_and_cleaning(self):
        self.assertIn("insider_name", self.df.columns)
        self.assertIn("security_type", self.df.columns)
        self.assertEqual(self.df["insider_name"].iloc[0], "Jane Smith")  # from 'executive'
        # zero-share dropped + duplicate removed -> 3 rows
        self.assertEqual(len(self.df), 3)

    def test_buy_sell_and_value(self):
        self.assertEqual(set(self.df["transaction_type"]), {"buy", "sell"})
        jane = self.df[self.df.insider_name == "Jane Smith"].iloc[0]
        self.assertEqual(jane["value"], 1000 * 50)

    def test_is_executive(self):
        flags = dict(zip(self.df["insider_name"], self.df["is_executive"]))
        self.assertTrue(flags["Jane Smith"])     # CEO
        self.assertTrue(flags["Bob Jones"])       # Director
        self.assertFalse(flags["Pat Lee"])        # 10% Owner -> not executive

    def test_empty(self):
        self.assertTrue(DataTransformer.transform_insider_transactions({"data": []}).empty)


class TestPanelFeatures(unittest.TestCase):
    def _txns(self):
        rows = [
            ("2021-01-05", "AAA", "CEO A", "CEO", "buy", 1000, 50),
            ("2021-01-20", "AAA", "CFO B", "CFO", "buy", 800, 51),
            ("2021-02-01", "AAA", "Dir C", "Director", "buy", 300, 52),
            ("2021-02-10", "AAA", "Dir D", "Director", "sell", 400, 55),
            ("2021-09-01", "AAA", "CEO A", "CEO", "buy", 200, 60),  # far later
        ]
        df = pd.DataFrame(rows, columns=["date", "symbol", "insider_name", "insider_title",
                                         "transaction_type", "shares", "price"])
        df["value"] = df["shares"] * df["price"]
        return df

    def test_point_in_time_no_lookahead(self):
        panel = ip.add_pit_features(self._txns(), filing_delay_days=3)
        # earliest available_date row only sees the first buy in its windows
        first = panel.sort_values("available_date").iloc[0]
        self.assertEqual(first["ins_90d_buy_count"], 1)
        self.assertEqual(first["ins_90d_sell_count"], 0)
        # the Sep buy is >180d after Feb -> not counted in Feb rows
        feb_row = panel[panel["available_date"] == pd.Timestamp("2021-02-10") + pd.Timedelta(days=3)].iloc[0]
        self.assertEqual(feb_row["ins_180d_net_count"], 3 - 1)  # 3 buys, 1 sell to date

    def test_cluster_and_exec_net(self):
        panel = ip.add_pit_features(self._txns(), filing_delay_days=3, cluster_min_buyers=3)
        feb_row = panel[panel["available_date"] == pd.Timestamp("2021-02-10") + pd.Timedelta(days=3)].iloc[0]
        # 3 distinct buyers within 90d (CEO A, CFO B, Dir C) -> cluster flag set
        self.assertEqual(feb_row["cluster_buy_flag"], 1)
        # exec net within 90d: buys CEO,CFO,Dir(exec) = 3 exec buys; sell Dir D exec = 1 -> net 2
        self.assertEqual(feb_row["ins_90d_exec_net"], 2)

    def test_build_aggregate_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._txns().to_csv(tmp / "AAA_insider.csv", index=False)
            out = tmp / "insider_data.csv"
            panel = ip.build_aggregate(tmp, out_path=out)
            self.assertTrue(out.exists())
            self.assertIn("ins_90d_net_value", panel.columns)
            self.assertIn("cluster_buy_flag", panel.columns)


class TestLoaderExecWiring(unittest.TestCase):
    def test_executive_metrics_populate(self):
        df = pd.DataFrame([
            {"date": "2021-01-05", "symbol": "AAA", "insider_title": "Chief Executive Officer",
             "transaction_type": "buy", "shares": 1000, "price": 50, "value": 50000},
            {"date": "2021-01-06", "symbol": "AAA", "insider_title": "Director",
             "transaction_type": "sell", "shares": 500, "price": 52, "value": 26000},
            {"date": "2021-01-07", "symbol": "AAA", "insider_title": "10% Owner",
             "transaction_type": "buy", "shares": 200, "price": 51, "value": 10200},
        ])
        loader = InsiderLoader()
        df = loader._normalize_dataframe(df)
        df = loader._compute_available_date(df)
        self.assertIn("is_executive", df.columns)
        act = loader.get_insider_activity_in_window(df, "AAA", pd.Timestamp("2021-03-01"),
                                                    window_days=90, delay_days=3)
        # CEO buy + Director sell are executives; 10% Owner is not
        self.assertEqual(act["insider_executive_buys"], 1)
        self.assertEqual(act["insider_executive_sells"], 1)


if __name__ == "__main__":
    unittest.main()
