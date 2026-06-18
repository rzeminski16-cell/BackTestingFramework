"""
Commodity data collector for Alpha Vantage.

Fetches commodity price series (WTI, Brent, natural gas, copper, the
all-commodities index, and the optional agricultural/metals tier) and normalises
them into a tidy panel suitable for the point-in-time data-preparation layer.

Each row carries the metadata the modelling stage needs to reason about
frequency and provenance:

    series_id, series_name, native_function, observation_date,
    native_frequency, value, unit, currency, source_vendor, retrieved_at

``available_ts``, ``run_id`` and ``quality_flag`` are intentionally *not* set
here -- they depend on the timing policy chosen per research run and are added by
the data-preparation layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import COMMODITY_SERIES, CORE_COMMODITIES
from .series_transforms import transform_flat_series

logger = logging.getLogger(__name__)

SOURCE_VENDOR = "alpha_vantage"
# Commodity series are quoted in USD (or USD-denominated indices).
DEFAULT_CURRENCY = "USD"


@dataclass
class CommodityResult:
    """Outcome of a single commodity fetch: normalised data plus any error."""
    df: pd.DataFrame
    series_key: str
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def empty(self) -> bool:
        return self.df is None or self.df.empty


class CommodityCollector:
    """Collects and normalises Alpha Vantage commodity series."""

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    def _data(response: Any) -> Dict[str, Any]:
        if response is not None and getattr(response, "success", False):
            return response.data or {}
        return {}

    def collect(self, series_key: str, interval: Optional[str] = None) -> CommodityResult:
        """
        Fetch and normalise one commodity series.

        Args:
            series_key: Key into ``COMMODITY_SERIES`` (e.g. "WTI", "COPPER").
            interval: Native frequency override; defaults to the series' default.

        Returns:
            CommodityResult with a normalised DataFrame (empty on failure).
        """
        spec = COMMODITY_SERIES.get(series_key)
        if spec is None:
            return CommodityResult(pd.DataFrame(), series_key, error=f"Unknown commodity: {series_key}")

        interval = interval or spec["default_interval"]
        if interval not in spec["intervals"]:
            return CommodityResult(
                pd.DataFrame(), series_key,
                error=f"{series_key} does not support interval '{interval}' "
                      f"(supported: {', '.join(spec['intervals'])})"
            )

        try:
            resp = self.client.get_commodity(spec["function"], interval=interval)
        except Exception as exc:  # pragma: no cover - defensive
            return CommodityResult(pd.DataFrame(), series_key, error=str(exc))

        if not getattr(resp, "success", False):
            return CommodityResult(
                pd.DataFrame(), series_key,
                error=getattr(resp, "error_message", "request failed")
            )

        df, payload_meta = transform_flat_series(self._data(resp))
        if df.empty:
            return CommodityResult(
                pd.DataFrame(), series_key,
                error=f"{series_key}: response had no parseable data"
            )

        retrieved_at = datetime.now(timezone.utc).isoformat()
        df["series_id"] = series_key
        df["series_name"] = spec["name"]
        df["native_function"] = spec["function"]
        df["native_frequency"] = interval
        df["unit"] = payload_meta.get("unit") or spec["unit"]
        df["currency"] = DEFAULT_CURRENCY
        df["source_vendor"] = SOURCE_VENDOR
        df["retrieved_at"] = retrieved_at

        ordered = [
            "series_id", "series_name", "native_function", "observation_date",
            "native_frequency", "value", "unit", "currency", "source_vendor",
            "retrieved_at",
        ]
        df = df[ordered]

        meta = {
            "series_key": series_key,
            "name": spec["name"],
            "function": spec["function"],
            "interval": interval,
            "unit": df["unit"].iloc[0],
            "tier": spec["tier"],
            "rationale": spec["rationale"],
            "rows": len(df),
            "retrieved_at": retrieved_at,
        }
        return CommodityResult(df, series_key, meta=meta)

    def collect_many(
        self,
        series_keys: Optional[List[str]] = None,
        intervals: Optional[Dict[str, str]] = None,
    ) -> Dict[str, CommodityResult]:
        """
        Collect several commodity series.

        Args:
            series_keys: Series to fetch; defaults to the recommended core set.
            intervals: Optional per-series interval overrides.

        Returns:
            ``{series_key: CommodityResult}``.
        """
        series_keys = series_keys or list(CORE_COMMODITIES)
        intervals = intervals or {}
        out: Dict[str, CommodityResult] = {}
        for key in series_keys:
            out[key] = self.collect(key, interval=intervals.get(key))
        return out
