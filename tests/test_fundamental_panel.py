"""
Tests for Classes/FactorAnalysis/data/fundamental_panel.py — point-in-time
derivation of fundamental ratios from the raw per-symbol panels.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Classes.FactorAnalysis.data import fundamental_panel as fp


def _panel(symbol="AAA", n=8, start_rev=1000.0, start_ni=100.0):
    """n quarterly rows with steadily growing revenue/income."""
    dates = pd.date_range("2020-03-31", periods=n, freq="QE")
    rows = []
    for i in range(n):
        rev = start_rev + i * 100
        ni = start_ni + i * 10
        rows.append({
            "symbol": symbol, "frequency": "quarterly",
            "fiscaldateending": dates[i], "report_date": dates[i] + pd.Timedelta(days=40),
            "reported_eps": round(ni / 100, 2),
            "totalrevenue": rev, "netincome": ni, "operatingincome": ni * 1.2,
            "grossprofit": rev * 0.6, "ebitda": ni * 1.5,
            "operatingcashflow": ni * 1.1, "capitalexpenditures": -20.0,
            "totalassets": 5000.0, "totalliabilities": 3000.0,
            "totalshareholderequity": 2000.0,
            "totalcurrentassets": 1500.0, "totalcurrentliabilities": 1000.0,
        })
    return pd.DataFrame(rows)


class TestDerivedFactors(unittest.TestCase):
    def setUp(self):
        self.df = fp.add_pit_derived_factors(_panel())

    def test_ttm_needs_four_quarters(self):
        # first 3 rows have NaN TTM (point-in-time: no look-ahead/backfill)
        rev_ttm = self.df.sort_values("fiscaldateending")["revenue_ttm"].tolist()
        self.assertTrue(all(pd.isna(v) for v in rev_ttm[:3]))
        self.assertFalse(pd.isna(rev_ttm[3]))

    def test_ttm_sum_value(self):
        d = self.df.sort_values("fiscaldateending").reset_index(drop=True)
        # Q4 revenue_ttm == sum of first 4 quarterly revenues (1000+1100+1200+1300)
        self.assertAlmostEqual(d.loc[3, "revenue_ttm"], 4600.0, places=6)

    def test_margin_and_leverage(self):
        d = self.df.sort_values("fiscaldateending").reset_index(drop=True)
        row = d.loc[3]
        self.assertAlmostEqual(row["profit_margin"], row["net_income_ttm"] / row["revenue_ttm"] * 100, places=6)
        self.assertAlmostEqual(row["debt_to_equity"], 3000.0 / 2000.0, places=6)
        self.assertAlmostEqual(row["currentratio"], 1500.0 / 1000.0, places=6)
        self.assertAlmostEqual(row["return_on_equity_ttm"], row["net_income_ttm"] / 2000.0 * 100, places=6)

    def test_yoy_growth(self):
        d = self.df.sort_values("fiscaldateending").reset_index(drop=True)
        # row 7 (Q8): yoy = ttm[7]/ttm[3]-1
        expected = (d.loc[7, "revenue_ttm"] / d.loc[3, "revenue_ttm"] - 1) * 100
        self.assertAlmostEqual(d.loc[7, "revenue_growth_yoy"], expected, places=6)

    def test_no_cross_symbol_leakage(self):
        two = pd.concat([_panel("AAA"), _panel("BBB", start_rev=5000)], ignore_index=True)
        out = fp.add_pit_derived_factors(two)
        # BBB's first 3 quarters must still be NaN (windows don't bleed across symbols)
        bbb = out[out.symbol == "BBB"].sort_values("fiscaldateending")
        self.assertTrue(pd.isna(bbb["revenue_ttm"].iloc[0]))
        self.assertAlmostEqual(bbb["revenue_ttm"].iloc[3], 5000 + 5100 + 5200 + 5300, places=6)


class TestLoadAndAggregate(unittest.TestCase):
    def test_load_and_build_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _panel("AAA").to_csv(tmp / "AAA_fundamental.csv", index=False)
            _panel("BBB", start_rev=2000).to_csv(tmp / "BBB_fundamental.csv", index=False)
            loaded = fp.load_panels(tmp)
            self.assertEqual(set(loaded["symbol"].unique()), {"AAA", "BBB"})

            out_path = tmp / "fundamental_data.csv"
            agg = fp.build_aggregate(tmp, out_path=out_path)
            self.assertTrue(out_path.exists())
            self.assertIn("profit_margin", agg.columns)
            self.assertIn("return_on_equity_ttm", agg.columns)
            self.assertIn("revenue_growth_yoy", agg.columns)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(fp.load_panels(Path(tmp)).empty)


if __name__ == "__main__":
    unittest.main()
