"""
US macro / economic-indicator collector for Alpha Vantage.

Fetches the US economic indicators Alpha Vantage exposes (Real GDP, Treasury
yields, Fed funds rate, CPI, inflation, retail sales, durable goods,
unemployment, nonfarm payroll) and normalises them into a release-aware regime
panel.

Two cautions are baked into the output so the data-preparation layer can warn
downstream:

  * geo_scope is "US" for every series -- treat as a US / global-risk proxy.
  * revision_risk_flag is True for series that FRED/BEA/BLS revise, because the
    Alpha Vantage endpoints expose *latest-history* values, not point-in-time
    vintages. ``observation_date`` is the period the value describes, NOT when it
    became available; the data-prep timing layer converts it to ``available_ts``
    using a conservative publication lag.

Each row carries:

    series_id, series_name, native_function, observation_date,
    native_frequency, value, unit, geo_scope, revision_risk_flag,
    source_vendor, retrieved_at

For TREASURY_YIELD, ``series_id`` is suffixed with the maturity (e.g.
"TREASURY_YIELD_10year") so multiple maturities coexist in one panel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import MACRO_SERIES, DEFAULT_MACRO_SERIES
from .series_transforms import transform_flat_series

logger = logging.getLogger(__name__)

SOURCE_VENDOR = "alpha_vantage"


@dataclass
class MacroResult:
    """Outcome of a single macro-series fetch."""
    df: pd.DataFrame
    series_id: str
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def empty(self) -> bool:
        return self.df is None or self.df.empty


class MacroCollector:
    """Collects and normalises Alpha Vantage US economic indicators."""

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    def _data(response: Any) -> Dict[str, Any]:
        if response is not None and getattr(response, "success", False):
            return response.data or {}
        return {}

    def collect(
        self,
        series_key: str,
        interval: Optional[str] = None,
        maturity: Optional[str] = None,
    ) -> MacroResult:
        """
        Fetch and normalise one macro series.

        Args:
            series_key: Key into ``MACRO_SERIES`` (e.g. "CPI", "TREASURY_YIELD").
            interval: Native frequency override; defaults to the series default.
            maturity: For TREASURY_YIELD only (e.g. "10year").

        Returns:
            MacroResult with a normalised DataFrame (empty on failure).
        """
        spec = MACRO_SERIES.get(series_key)
        if spec is None:
            return MacroResult(pd.DataFrame(), series_key, error=f"Unknown macro series: {series_key}")

        interval = interval or spec["default_interval"]
        if interval not in spec["intervals"]:
            return MacroResult(
                pd.DataFrame(), series_key,
                error=f"{series_key} does not support interval '{interval}' "
                      f"(supported: {', '.join(spec['intervals'])})"
            )

        requires_maturity = spec.get("maturity_required", False)
        if requires_maturity:
            maturity = maturity or spec.get("default_maturity")
            series_id = f"{series_key}_{maturity}"
        else:
            maturity = None
            series_id = series_key

        try:
            resp = self.client.get_economic_indicator(
                spec["function"], interval=interval, maturity=maturity
            )
        except Exception as exc:  # pragma: no cover - defensive
            return MacroResult(pd.DataFrame(), series_id, error=str(exc))

        if not getattr(resp, "success", False):
            return MacroResult(
                pd.DataFrame(), series_id,
                error=getattr(resp, "error_message", "request failed")
            )

        df, payload_meta = transform_flat_series(self._data(resp))
        if df.empty:
            return MacroResult(
                pd.DataFrame(), series_id,
                error=f"{series_id}: response had no parseable data"
            )

        retrieved_at = datetime.now(timezone.utc).isoformat()
        name = spec["name"] + (f" ({maturity})" if maturity else "")
        df["series_id"] = series_id
        df["series_name"] = name
        df["native_function"] = spec["function"]
        df["native_frequency"] = interval
        df["unit"] = payload_meta.get("unit") or spec["unit"]
        df["geo_scope"] = spec.get("geo_scope", "US")
        df["revision_risk_flag"] = bool(spec.get("revision_risk", True))
        df["source_vendor"] = SOURCE_VENDOR
        df["retrieved_at"] = retrieved_at

        ordered = [
            "series_id", "series_name", "native_function", "observation_date",
            "native_frequency", "value", "unit", "geo_scope",
            "revision_risk_flag", "source_vendor", "retrieved_at",
        ]
        df = df[ordered]

        meta = {
            "series_id": series_id,
            "series_key": series_key,
            "name": name,
            "function": spec["function"],
            "interval": interval,
            "maturity": maturity,
            "unit": df["unit"].iloc[0],
            "geo_scope": spec.get("geo_scope", "US"),
            "revision_risk": bool(spec.get("revision_risk", True)),
            "rationale": spec.get("rationale"),
            "rows": len(df),
            "retrieved_at": retrieved_at,
        }
        return MacroResult(df, series_id, meta=meta)

    def collect_many(
        self,
        series_keys: Optional[List[str]] = None,
        intervals: Optional[Dict[str, str]] = None,
        treasury_maturities: Optional[List[str]] = None,
    ) -> Dict[str, MacroResult]:
        """
        Collect several macro series.

        TREASURY_YIELD is expanded across ``treasury_maturities`` (default
        ["10year"]); the resulting keys are suffixed with the maturity.

        Returns:
            ``{series_id: MacroResult}``.
        """
        series_keys = series_keys or list(DEFAULT_MACRO_SERIES)
        intervals = intervals or {}
        treasury_maturities = treasury_maturities or ["10year"]

        out: Dict[str, MacroResult] = {}
        for key in series_keys:
            spec = MACRO_SERIES.get(key)
            if spec and spec.get("maturity_required"):
                for mat in treasury_maturities:
                    res = self.collect(key, interval=intervals.get(key), maturity=mat)
                    out[res.series_id] = res
            else:
                res = self.collect(key, interval=intervals.get(key))
                out[res.series_id] = res
        return out
