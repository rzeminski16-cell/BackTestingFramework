"""
Tests for Classes/DataCollection/fundamental_collector.py

Covers the point-in-time fundamental panel assembly that replaced the previous
concat + forward-fill pipeline:
- statements joined into one row per fiscal period (not stacked across rows)
- a real publication date (report_date) recovered from EARNINGS, with a fiscal
  + lag fallback when earnings has no match
- the point-in-time guarantee report_date >= fiscal_date_ending
- no smearing of values across periods
- fields shared across statements (netincome, reportedcurrency) coalesced
- earnings-only history (older than statements) preserved
- frequency filtering, annual reported_eps, incremental merge dedup
- OVERVIEW snapshot extraction
"""

import unittest

import numpy as np
import pandas as pd

from Classes.DataCollection.fundamental_collector import (
    DEFAULT_REPORTING_LAG_DAYS,
    build_fundamental_panel,
    extract_overview_snapshot,
    merge_panels,
)


def _raw_fixture():
    """A small but representative set of raw Alpha Vantage responses."""
    income = {
        "symbol": "TEST",
        "quarterlyReports": [
            {"fiscalDateEnding": "2023-12-31", "reportedCurrency": "USD",
             "totalRevenue": "1000", "netIncome": "100", "ebitda": "200"},
            {"fiscalDateEnding": "2023-09-30", "reportedCurrency": "USD",
             "totalRevenue": "900", "netIncome": "90", "ebitda": "180"},
            # A statement period with NO matching earnings -> fallback report_date.
            {"fiscalDateEnding": "2023-06-30", "reportedCurrency": "USD",
             "totalRevenue": "800", "netIncome": "80", "ebitda": "160"},
        ],
        "annualReports": [
            {"fiscalDateEnding": "2023-12-31", "reportedCurrency": "USD",
             "totalRevenue": "3800", "netIncome": "370"},
        ],
    }
    balance = {
        "symbol": "TEST",
        "quarterlyReports": [
            {"fiscalDateEnding": "2023-12-31", "reportedCurrency": "USD",
             "totalAssets": "5000", "totalLiabilities": "3000", "totalShareholderEquity": "2000"},
            {"fiscalDateEnding": "2023-09-30", "reportedCurrency": "USD",
             "totalAssets": "4800", "totalLiabilities": "2900", "totalShareholderEquity": "1900"},
            {"fiscalDateEnding": "2023-06-30", "reportedCurrency": "USD",
             "totalAssets": "4600", "totalLiabilities": "2800", "totalShareholderEquity": "1800"},
        ],
        "annualReports": [
            {"fiscalDateEnding": "2023-12-31", "reportedCurrency": "USD",
             "totalAssets": "5000", "totalLiabilities": "3000", "totalShareholderEquity": "2000"},
        ],
    }
    cash = {
        "symbol": "TEST",
        "quarterlyReports": [
            {"fiscalDateEnding": "2023-12-31", "reportedCurrency": "USD",
             "operatingCashflow": "300", "capitalExpenditures": "50", "netIncome": "100"},
            {"fiscalDateEnding": "2023-09-30", "reportedCurrency": "USD",
             "operatingCashflow": "280", "capitalExpenditures": "45", "netIncome": "90"},
            {"fiscalDateEnding": "2023-06-30", "reportedCurrency": "USD",
             "operatingCashflow": "260", "capitalExpenditures": "40", "netIncome": "80"},
        ],
        "annualReports": [
            {"fiscalDateEnding": "2023-12-31", "reportedCurrency": "USD",
             "operatingCashflow": "1140", "capitalExpenditures": "180", "netIncome": "370"},
        ],
    }
    earnings = {
        "symbol": "TEST",
        "quarterlyEarnings": [
            {"fiscalDateEnding": "2023-12-31", "reportedDate": "2024-02-01", "reportedEPS": "1.0",
             "estimatedEPS": "0.9", "surprise": "0.1", "surprisePercentage": "11.1",
             "reportTime": "post-market"},
            {"fiscalDateEnding": "2023-09-30", "reportedDate": "2023-11-01", "reportedEPS": "0.9",
             "estimatedEPS": "0.85", "surprise": "0.05", "surprisePercentage": "5.9",
             "reportTime": "pre-market"},
            # Earnings-only period, older than any statement.
            {"fiscalDateEnding": "2010-12-31", "reportedDate": "2011-02-01", "reportedEPS": "0.5",
             "estimatedEPS": "0.45", "surprise": "0.05", "surprisePercentage": "11.1",
             "reportTime": "post-market"},
        ],
        "annualEarnings": [
            {"fiscalDateEnding": "2023-12-31", "reportedEPS": "3.5"},
        ],
    }
    return {
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow": cash,
        "earnings": earnings,
    }


class TestBuildFundamentalPanel(unittest.TestCase):
    def setUp(self):
        self.raw = _raw_fixture()
        self.panel = build_fundamental_panel(self.raw, "TEST", frequency="both")

    def test_one_row_per_fiscal_period(self):
        """Statements are joined: each (frequency, fiscal date) is a single row."""
        q = self.panel[self.panel.frequency == "quarterly"]
        self.assertTrue(q["fiscaldateending"].is_unique)
        # 2023-12-31 has income + balance + cash all present on the SAME row.
        row = q[q.fiscaldateending == pd.Timestamp("2023-12-31")].iloc[0]
        self.assertEqual(row["totalrevenue"], 1000)
        self.assertEqual(row["totalassets"], 5000)
        self.assertEqual(row["operatingcashflow"], 300)

    def test_report_date_from_earnings(self):
        q = self.panel[self.panel.frequency == "quarterly"]
        row = q[q.fiscaldateending == pd.Timestamp("2023-12-31")].iloc[0]
        self.assertEqual(row["report_date"], pd.Timestamp("2024-02-01"))
        self.assertEqual(row["reporttime"], "post-market")
        self.assertEqual(row["reported_eps"], 1.0)

    def test_report_date_fallback_when_no_earnings(self):
        q = self.panel[self.panel.frequency == "quarterly"]
        row = q[q.fiscaldateending == pd.Timestamp("2023-06-30")].iloc[0]
        expected = pd.Timestamp("2023-06-30") + pd.Timedelta(days=DEFAULT_REPORTING_LAG_DAYS)
        self.assertEqual(row["report_date"], expected)

    def test_point_in_time_guarantee(self):
        """report_date must never precede fiscal_date_ending."""
        both = self.panel.dropna(subset=["report_date", "fiscaldateending"])
        self.assertTrue((both["report_date"] >= both["fiscaldateending"]).all())

    def test_no_smearing(self):
        """Line items vary across periods (the old pipeline collapsed them)."""
        q = self.panel[self.panel.frequency == "quarterly"]
        self.assertEqual(q["totalrevenue"].dropna().nunique(), 3)

    def test_shared_fields_coalesced(self):
        """netincome/reportedcurrency appear once; income statement wins."""
        cols = list(self.panel.columns)
        self.assertEqual(cols.count("netincome"), 1)
        self.assertEqual(cols.count("reportedcurrency"), 1)
        row = self.panel[
            (self.panel.frequency == "quarterly")
            & (self.panel.fiscaldateending == pd.Timestamp("2023-12-31"))
        ].iloc[0]
        self.assertEqual(row["netincome"], 100)  # income value, not duplicated/clobbered
        self.assertEqual(row["reportedcurrency"], "USD")

    def test_earnings_only_history_preserved(self):
        """Earnings periods older than statements survive with NaN statement fields."""
        q = self.panel[self.panel.frequency == "quarterly"]
        old = q[q.fiscaldateending == pd.Timestamp("2010-12-31")]
        self.assertEqual(len(old), 1)
        self.assertEqual(old.iloc[0]["reported_eps"], 0.5)
        self.assertTrue(pd.isna(old.iloc[0]["totalrevenue"]))

    def test_annual_rows(self):
        annual = self.panel[self.panel.frequency == "annual"]
        row = annual[annual.fiscaldateending == pd.Timestamp("2023-12-31")].iloc[0]
        self.assertEqual(row["reported_eps"], 3.5)  # from annualEarnings
        self.assertEqual(row["report_date"], pd.Timestamp("2024-02-01"))  # from Q4 earnings
        self.assertEqual(row["totalrevenue"], 3800)

    def test_frequency_filter(self):
        q_only = build_fundamental_panel(self.raw, "TEST", frequency="quarterly")
        self.assertEqual(set(q_only["frequency"].unique()), {"quarterly"})

    def test_key_columns_lead(self):
        self.assertEqual(
            list(self.panel.columns[:6]),
            ["symbol", "frequency", "fiscaldateending", "report_date", "reporttime", "reportedcurrency"],
        )

    def test_empty_inputs(self):
        self.assertTrue(build_fundamental_panel({}, "TEST").empty)


class TestMergePanels(unittest.TestCase):
    def test_merge_dedups_keep_last(self):
        raw = _raw_fixture()
        existing = build_fundamental_panel(raw, "TEST", frequency="quarterly")

        # New collection revises 2023-12-31 revenue and adds nothing else new.
        revised = existing.copy()
        mask = revised.fiscaldateending == pd.Timestamp("2023-12-31")
        revised.loc[mask, "totalrevenue"] = 1234

        merged = merge_panels(existing, revised)
        # Same number of periods (dedup), revised value kept.
        self.assertEqual(len(merged), len(existing))
        row = merged[merged.fiscaldateending == pd.Timestamp("2023-12-31")].iloc[0]
        self.assertEqual(row["totalrevenue"], 1234)

    def test_merge_with_empty(self):
        raw = _raw_fixture()
        panel = build_fundamental_panel(raw, "TEST", frequency="quarterly")
        self.assertTrue(merge_panels(pd.DataFrame(), panel).equals(panel))


class TestOverviewSnapshot(unittest.TestCase):
    def test_snapshot_single_row(self):
        overview = {"Symbol": "TEST", "Beta": "1.5", "PERatio": "20"}
        snap = extract_overview_snapshot(overview, "TEST", collected_at=pd.Timestamp("2024-01-01"))
        self.assertEqual(len(snap), 1)
        self.assertIn("as_of_date", snap.columns)
        self.assertEqual(snap.iloc[0]["beta"], "1.5")
        self.assertEqual(snap.iloc[0]["symbol"], "TEST")

    def test_empty_overview(self):
        self.assertTrue(extract_overview_snapshot({}, "TEST").empty)


if __name__ == "__main__":
    unittest.main()
