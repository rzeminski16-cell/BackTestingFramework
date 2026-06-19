"""
Tests for the Alpha Vantage data-collection expansion:

  * commodities  (CommodityCollector + transform_flat_series)
  * macro        (MacroCollector, incl. Treasury maturity expansion)
  * corporate actions (CorporateActionsCollector + transforms)
  * utilities    (MARKET_STATUS + CSV payload parsing)
  * AlphaVantageClient CSV request path + new config dataclasses

All tests are offline: the API client is replaced by a fake that returns
canned ``APIResponse`` objects, and the one client-level test monkeypatches
``requests.get``.
"""

import io
import unittest
from unittest import mock

import pandas as pd

from Classes.DataCollection.alpha_vantage_client import (
    AlphaVantageClient,
    APIEndpoint,
    APIResponse,
)
from Classes.DataCollection.config import (
    APIConfig,
    CacheConfig,
    CommodityDataConfig,
    MacroDataConfig,
    CorporateActionsDataConfig,
    DataCollectionConfig,
    COMMODITY_SERIES,
    CORE_COMMODITIES,
    MACRO_SERIES,
)
from Classes.DataCollection.series_transforms import transform_flat_series
from Classes.DataCollection.commodity_collector import CommodityCollector
from Classes.DataCollection.macro_collector import MacroCollector
from Classes.DataCollection.corporate_actions_collector import (
    CorporateActionsCollector,
    transform_dividends,
    transform_splits,
)
from Classes.DataCollection.utilities_collector import (
    UtilitiesCollector,
    transform_market_status,
    _parse_csv_payload,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeClient:
    """Records the last call and returns a queued APIResponse per method."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def _resp(self, key, default=None):
        return self.responses.get(key, default)

    def get_commodity(self, function, interval="monthly"):
        self.calls.append(("commodity", function, interval))
        return self._resp(("commodity", function),
                          APIResponse(success=False, data=None, error_message="no fake"))

    def get_economic_indicator(self, function, interval=None, maturity=None):
        self.calls.append(("macro", function, interval, maturity))
        return self._resp(("macro", function),
                          APIResponse(success=False, data=None, error_message="no fake"))

    def get_dividends(self, symbol):
        self.calls.append(("dividends", symbol))
        return self._resp(("dividends", symbol),
                          APIResponse(success=False, data=None, error_message="no fake"))

    def get_splits(self, symbol):
        self.calls.append(("splits", symbol))
        return self._resp(("splits", symbol),
                          APIResponse(success=False, data=None, error_message="no fake"))

    def get_market_status(self):
        self.calls.append(("market_status",))
        return self._resp(("market_status",),
                          APIResponse(success=False, data=None, error_message="no fake"))

    def get_listing_status(self, date=None, state="active"):
        self.calls.append(("listing_status", date, state))
        return self._resp(("listing_status",),
                          APIResponse(success=False, data=None, error_message="no fake"))


def _av_series(name, interval, unit, points):
    """Build a fake AV flat-series payload."""
    return {"name": name, "interval": interval, "unit": unit,
            "data": [{"date": d, "value": v} for d, v in points]}


# --------------------------------------------------------------------------- #
# transform_flat_series
# --------------------------------------------------------------------------- #
class TestTransformFlatSeries(unittest.TestCase):
    def test_parses_sorts_and_coerces(self):
        payload = _av_series("WTI", "monthly", "dollars per barrel",
                             [("2024-03-01", "80.5"), ("2024-01-01", "73.85"),
                              ("2024-02-01", ".")])  # "." is a missing token
        df, meta = transform_flat_series(payload)
        self.assertEqual(meta["unit"], "dollars per barrel")
        self.assertEqual(list(df["observation_date"].dt.strftime("%Y-%m-%d")),
                         ["2024-01-01", "2024-02-01", "2024-03-01"])  # ascending
        self.assertTrue(pd.isna(df.loc[1, "value"]))  # "." -> NaN
        self.assertAlmostEqual(df.loc[0, "value"], 73.85)

    def test_empty_payload(self):
        df, meta = transform_flat_series({})
        self.assertTrue(df.empty)
        self.assertIsNone(meta["name"])

    def test_dedup_keeps_last(self):
        payload = _av_series("CPI", "monthly", "index",
                             [("2024-01-01", "100"), ("2024-01-01", "101")])
        df, _ = transform_flat_series(payload)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.loc[0, "value"], 101.0)


# --------------------------------------------------------------------------- #
# CommodityCollector
# --------------------------------------------------------------------------- #
class TestCommodityCollector(unittest.TestCase):
    def test_normalised_output(self):
        client = FakeClient({
            ("commodity", "WTI"): APIResponse(
                success=True,
                data=_av_series("Crude Oil WTI", "monthly", "dollars per barrel",
                                [("2024-01-01", "73.85"), ("2024-02-01", "78.2")]),
            )
        })
        res = CommodityCollector(client).collect("WTI", interval="monthly")
        self.assertFalse(res.empty)
        self.assertEqual(
            list(res.df.columns),
            ["series_id", "series_name", "native_function", "observation_date",
             "native_frequency", "value", "unit", "currency", "source_vendor",
             "retrieved_at"],
        )
        self.assertEqual(res.df["series_id"].iloc[0], "WTI")
        self.assertEqual(res.df["native_function"].iloc[0], "WTI")
        self.assertEqual(res.df["currency"].iloc[0], "USD")
        self.assertEqual(res.meta["tier"], "core")

    def test_unknown_series(self):
        res = CommodityCollector(FakeClient()).collect("UNOBTANIUM")
        self.assertTrue(res.empty)
        self.assertIn("Unknown commodity", res.error)

    def test_unsupported_interval(self):
        # COPPER does not support daily.
        res = CommodityCollector(FakeClient()).collect("COPPER", interval="daily")
        self.assertTrue(res.empty)
        self.assertIn("does not support interval", res.error)

    def test_api_failure_surfaced(self):
        client = FakeClient({
            ("commodity", "BRENT"): APIResponse(
                success=False, data=None, error_message="Premium endpoint")
        })
        res = CommodityCollector(client).collect("BRENT")
        self.assertTrue(res.empty)
        self.assertIn("Premium", res.error)

    def test_collect_many_defaults_to_core(self):
        client = FakeClient({
            ("commodity", spec["function"]): APIResponse(
                success=True,
                data=_av_series(spec["name"], spec["default_interval"], spec["unit"],
                                [("2024-01-01", "1.0")]),
            )
            for key, spec in COMMODITY_SERIES.items()
        })
        out = CommodityCollector(client).collect_many()
        self.assertEqual(set(out.keys()), set(CORE_COMMODITIES))
        self.assertTrue(all(not r.empty for r in out.values()))


# --------------------------------------------------------------------------- #
# MacroCollector
# --------------------------------------------------------------------------- #
class TestMacroCollector(unittest.TestCase):
    def test_normalised_output_has_geo_and_revision(self):
        client = FakeClient({
            ("macro", "CPI"): APIResponse(
                success=True,
                data=_av_series("CPI", "monthly", "index",
                                [("2024-01-01", "308.4")]),
            )
        })
        res = MacroCollector(client).collect("CPI")
        self.assertFalse(res.empty)
        self.assertEqual(res.df["geo_scope"].iloc[0], "US")
        self.assertTrue(bool(res.df["revision_risk_flag"].iloc[0]))
        self.assertIn("geo_scope", res.df.columns)
        self.assertIn("revision_risk_flag", res.df.columns)

    def test_treasury_maturity_suffix(self):
        client = FakeClient({
            ("macro", "TREASURY_YIELD"): APIResponse(
                success=True,
                data=_av_series("Treasury Yield", "monthly", "percent",
                                [("2024-01-01", "4.1")]),
            )
        })
        res = MacroCollector(client).collect("TREASURY_YIELD", maturity="10year")
        self.assertEqual(res.series_id, "TREASURY_YIELD_10year")
        self.assertIn("10year", res.df["series_name"].iloc[0])

    def test_collect_many_expands_treasury(self):
        client = FakeClient({
            ("macro", spec["function"]): APIResponse(
                success=True,
                data=_av_series(spec["name"], spec["default_interval"],
                                spec.get("unit", "x"), [("2024-01-01", "1.0")]),
            )
            for key, spec in MACRO_SERIES.items()
        })
        out = MacroCollector(client).collect_many(
            series_keys=["CPI", "TREASURY_YIELD"],
            treasury_maturities=["2year", "10year"],
        )
        self.assertIn("CPI", out)
        self.assertIn("TREASURY_YIELD_2year", out)
        self.assertIn("TREASURY_YIELD_10year", out)


# --------------------------------------------------------------------------- #
# Corporate actions
# --------------------------------------------------------------------------- #
class TestCorporateActions(unittest.TestCase):
    def test_transform_dividends(self):
        payload = {"symbol": "IBM", "data": [
            {"ex_dividend_date": "2024-02-09", "declaration_date": "2024-01-30",
             "record_date": "2024-02-12", "payment_date": "2024-03-09", "amount": "1.66"},
            {"ex_dividend_date": "2023-11-09", "declaration_date": "None",
             "record_date": "2023-11-10", "payment_date": "2023-12-09", "amount": "1.66"},
        ]}
        df = transform_dividends(payload, "IBM")
        self.assertEqual(len(df), 2)
        # sorted ascending by ex-date
        self.assertTrue(df["ex_dividend_date"].is_monotonic_increasing)
        self.assertEqual(df["action_type"].iloc[0], "dividend")
        # The 2023-11-09 row (now first after ascending sort) had "None" -> NaT.
        self.assertTrue(pd.isna(df["declaration_date"].iloc[0]))

    def test_transform_splits(self):
        payload = {"symbol": "AAPL", "data": [
            {"effective_date": "2020-08-31", "split_factor": "4.0"},
            {"effective_date": "2014-06-09", "split_factor": "7.0"},
        ]}
        df = transform_splits(payload, "AAPL")
        self.assertEqual(len(df), 2)
        self.assertTrue(df["effective_date"].is_monotonic_increasing)
        self.assertAlmostEqual(df["split_factor"].iloc[1], 4.0)

    def test_collector_combines_and_tags(self):
        client = FakeClient({
            ("dividends", "IBM"): APIResponse(success=True, data={
                "symbol": "IBM", "data": [
                    {"ex_dividend_date": "2024-02-09", "amount": "1.66"}]}),
            ("splits", "IBM"): APIResponse(success=True, data={
                "symbol": "IBM", "data": [
                    {"effective_date": "1999-05-27", "split_factor": "2.0"}]}),
        })
        res = CorporateActionsCollector(client).collect("IBM")
        self.assertFalse(res.dividends.empty)
        self.assertFalse(res.splits.empty)
        self.assertIn("source_vendor", res.dividends.columns)
        self.assertIn("retrieved_at", res.splits.columns)
        self.assertEqual(res.errors, [])


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class TestUtilities(unittest.TestCase):
    def test_transform_market_status(self):
        payload = {"endpoint": "Global Market Open & Close Status", "markets": [
            {"market_type": "Equity", "region": "United States",
             "current_status": "open"},
            {"market_type": "Equity", "region": "United Kingdom",
             "current_status": "closed"},
        ]}
        df = transform_market_status(payload)
        self.assertEqual(len(df), 2)
        self.assertIn("current_status", df.columns)

    def test_csv_payload_parsing(self):
        csv_text = "symbol,name,exchange,status\nIBM,Intl Business Machines,NYSE,Active\n"
        df = _parse_csv_payload({"csv": csv_text})
        self.assertEqual(len(df), 1)
        self.assertEqual(df["symbol"].iloc[0], "IBM")

    def test_collect_listing_status_via_csv(self):
        csv_text = "symbol,name,status\nAAPL,Apple Inc,Active\nMSFT,Microsoft,Active\n"
        client = FakeClient({("listing_status",): APIResponse(
            success=True, data={"csv": csv_text})})
        df = UtilitiesCollector(client).collect_listing_status()
        self.assertEqual(len(df), 2)


# --------------------------------------------------------------------------- #
# Client-level CSV request path
# --------------------------------------------------------------------------- #
class _FakeHTTP:
    def __init__(self, text, status=200):
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status

    def json(self):
        import json
        return json.loads(self.text)


class TestClientCsvPath(unittest.TestCase):
    def _client(self, tmpdir):
        return AlphaVantageClient(
            APIConfig(api_key="demo", rate_limit_per_minute=600, max_retries=0),
            CacheConfig(cache_dir=tmpdir, enabled=False),
        )

    def test_csv_success_wrapped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            csv_text = "symbol,reportDate\nIBM,2024-04-24\n"
            with mock.patch(
                "Classes.DataCollection.alpha_vantage_client.requests.get",
                return_value=_FakeHTTP(csv_text),
            ):
                resp = client.get_earnings_calendar(horizon="3month")
            self.assertTrue(resp.success)
            self.assertEqual(resp.data["csv"], csv_text)

    def test_csv_endpoint_json_advisory_is_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            advisory = '{"Information": "premium endpoint"}'
            with mock.patch(
                "Classes.DataCollection.alpha_vantage_client.requests.get",
                return_value=_FakeHTTP(advisory),
            ):
                resp = client.get_listing_status()
            self.assertFalse(resp.success)
            self.assertIn("Premium", resp.error_message or "")


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def test_commodity_config_roundtrip_and_interval(self):
        cfg = CommodityDataConfig(series=["WTI", "COPPER"], intervals={"WTI": "daily"})
        self.assertEqual(cfg.interval_for("WTI"), "daily")
        self.assertEqual(cfg.interval_for("COPPER"), "monthly")  # default
        rt = CommodityDataConfig.from_dict(cfg.to_dict())
        self.assertEqual(rt.series, ["WTI", "COPPER"])

    def test_commodity_config_rejects_unknown(self):
        with self.assertRaises(ValueError):
            CommodityDataConfig(series=["NOTREAL"])

    def test_macro_config_defaults(self):
        cfg = MacroDataConfig()
        self.assertIn("CPI", cfg.series)
        self.assertEqual(cfg.treasury_maturities, ["10year"])

    def test_master_config_includes_new_families(self):
        cfg = DataCollectionConfig(
            commodity=CommodityDataConfig(series=["WTI"]),
            macro=MacroDataConfig(series=["CPI"]),
            corporate_actions=CorporateActionsDataConfig(tickers=["IBM"]),
        )
        d = cfg.to_dict()
        self.assertIsNotNone(d["commodity"])
        self.assertIsNotNone(d["macro"])
        self.assertIsNotNone(d["corporate_actions"])


if __name__ == "__main__":
    unittest.main()
