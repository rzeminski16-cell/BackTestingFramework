"""
Tests for the factor-analysis upgrade:
- enrich_dataframe: normalize a mixed-case/string fundamental frame + derive ratios
- FundamentalFactors.create_composite_scores robustness (all-NaN category must not crash)
- html_generator: standalone HTML report renders with findings + correlation heatmap
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Classes.FactorAnalysis.data.fundamental_panel import enrich_dataframe
from Classes.FactorAnalysis.factors.fundamental_factors import FundamentalFactors
from Classes.FactorAnalysis.output.html_generator import generate_html_report
from Classes.FactorAnalysis.preprocessing.trade_classifier import TradeClassifier
from Classes.FactorAnalysis.config.factor_config import TradeClassificationConfig, ThresholdType


class TestEnrichDataframe(unittest.TestCase):
    def test_mixed_case_string_input_derives_ratios(self):
        df = pd.DataFrame([
            dict(Symbol="AAA", frequency="quarterly",
                 fiscalDateEnding=f"2020-0{q}-01", report_date=f"2020-0{q}-15",
                 totalrevenue=str(1000 + q * 100), netincome=str(100 + q * 10),
                 totalshareholderequity="2000", totalassets="5000",
                 totalcurrentassets="1500", totalcurrentliabilities="1000")
            for q in range(1, 6)
        ])
        out = enrich_dataframe(df)
        for col in ("profit_margin", "return_on_equity_ttm", "revenue_ttm", "debt_to_equity", "currentratio"):
            self.assertIn(col, out.columns)
        # ratios numeric after coercion
        self.assertTrue(pd.api.types.is_numeric_dtype(out["debt_to_equity"]))

    def test_empty_passthrough(self):
        self.assertTrue(enrich_dataframe(pd.DataFrame()).empty)


class TestCompositeRobustness(unittest.TestCase):
    def test_all_nan_category_does_not_crash(self):
        # value_* columns all NaN (no price ratios), quality populated -> must not raise
        ff = FundamentalFactors()
        df = pd.DataFrame({
            "value_pe_ratio": [np.nan] * 5,
            "value_price_to_book": [np.nan] * 5,
            "quality_return_on_equity": [10.0, 12.0, 8.0, 15.0, 9.0],
            "quality_debt_to_equity": [1.0, 1.2, 0.8, 2.0, 1.1],
            "growth_revenue_growth": [5.0, 7.0, -2.0, 10.0, 3.0],
        })
        out = ff.create_composite_scores(df)
        # quality/growth composites created; value composite absent (no data)
        self.assertIn("composite_quality", out.columns)
        self.assertIn("composite_growth", out.columns)
        self.assertNotIn("composite_value", out.columns)
        self.assertIn("composite_fundamental", out.columns)
        self.assertTrue(out["composite_quality"].notna().any())


class TestHtmlReport(unittest.TestCase):
    def test_renders_findings_and_heatmap(self):
        result = SimpleNamespace(
            timestamp="2026-01-01T00:00:00",
            data_summary={"trades": 120, "symbols": 4, "good": 50, "bad": 40},
            quality_score={"overall": 88.5},
            key_findings=["Quality factors strongest", "ROE correlates with good trades"],
            warnings=["small sample in 2019"],
            tier1={"correlations_pearson": {
                "quality_return_on_equity": {"correlation": 0.31, "p_value": 0.01},
                "growth_revenue_growth": {"correlation": -0.12, "p_value": 0.2},
            }},
            tier2={"significant_factors": [{"factor": "quality_profit_margin", "p_value": 0.02}]},
            scenarios=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, Path(tmp) / "r.html")
            self.assertTrue(path.exists())
            html = path.read_text()
            self.assertIn("Factor Analysis Report", html)
            self.assertIn("Quality factors strongest", html)
            self.assertIn("quality_return_on_equity", html)
            self.assertIn("background:rgb", html)  # correlation heatmap coloring

    def test_minimal_result(self):
        result = SimpleNamespace(timestamp="2026-01-01", data_summary={}, key_findings=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, Path(tmp) / "r.html")
            self.assertTrue(path.exists())


class TestClassifierNumericCoercion(unittest.TestCase):
    """Trade return/duration columns loaded as strings must not crash classification
    ("'>' not supported between instances of 'float' and 'str'")."""

    def _df(self):
        return pd.DataFrame({
            "pl_pct": ["5.2", "-3.1", "0.4", "8.0", "-6.5", "bad", "12.0", "-9.0", "1.0", "-0.5"],
            "duration_days": ["10", "30", "5", "12", "40", "7", "25", "35", "3", "22"],
            "entry_date": pd.date_range("2021-01-01", periods=10),
            "exit_date": pd.date_range("2021-02-01", periods=10),
        })

    def test_absolute_thresholds_with_string_columns(self):
        cfg = TradeClassificationConfig(good_threshold_pct=2.0, bad_threshold_pct=-1.0,
                                        threshold_type=ThresholdType.ABSOLUTE)
        out, _ = TradeClassifier(config=cfg).classify_trades(self._df())
        self.assertIn("trade_class", out.columns)
        self.assertEqual(len(out), 10)
        # unparseable 'bad' -> NaN -> indeterminate (not a crash)
        self.assertIn("indeterminate", out["trade_class"].unique())

    def test_percentile_thresholds_with_string_columns(self):
        cfg = TradeClassificationConfig(good_threshold_pct=20.0, bad_threshold_pct=-20.0,
                                        threshold_type=ThresholdType.PERCENTILE)
        out, _ = TradeClassifier(config=cfg).classify_trades(self._df())
        self.assertEqual(len(out), 10)


if __name__ == "__main__":
    unittest.main()
