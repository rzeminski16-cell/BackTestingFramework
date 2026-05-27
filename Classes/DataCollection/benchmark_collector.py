"""
Benchmark / index data collector for Alpha Vantage.

Collects index price series via the premium INDEX_DATA endpoint (SPX, DJI,
IXIC, NDX, RUT, VIX, ...) and stores them under ``raw_data/benchmarks/`` so
they can be used as benchmarks in generated reports.

A registry (``config/benchmarks.json``) maps friendly names ("S&P 500") to
INDEX_DATA symbols ("SPX"), so reports can resolve a benchmark by name.

The INDEX_DATA response mirrors the other Alpha Vantage time-series endpoints:
a ``Meta Data`` object plus one time-series object keyed by date. The exact
time-series key name varies by interval, so the parser locates it by structure
(the first non-metadata object whose values look like OHLC dicts) rather than
by a hard-coded key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("config/benchmarks.json")
DEFAULT_BENCHMARK_DIR = Path("raw_data/benchmarks")

# Alpha Vantage OHLC field prefixes -> canonical column names.
_OHLC_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjusted close": "close",
    "volume": "volume",
}

_NULL_TOKENS = {None, "", "None", "-", "NaN", "nan"}


def _strip_av_prefix(field: str) -> str:
    """'1. open' -> 'open', '5. adjusted close' -> 'adjusted close'."""
    field = field.strip().lower()
    if ". " in field:
        field = field.split(". ", 1)[1]
    return field


def transform_index_data(response_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse an INDEX_DATA (or any AV time-series) response into a tidy DataFrame.

    Returns a DataFrame with a ``date`` column plus available OHLC(V) columns,
    sorted ascending by date. Empty if no time series is found.
    """
    if not response_data:
        return pd.DataFrame()

    # Locate the time-series object: the first dict-valued key (other than
    # metadata) whose entries are themselves dicts of OHLC fields.
    series = None
    for key, value in response_data.items():
        if key in ("Meta Data", "Information", "Note", "Error Message"):
            continue
        if isinstance(value, dict) and value:
            first = next(iter(value.values()))
            if isinstance(first, dict):
                series = value
                break

    if not series:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for date_str, fields in series.items():
        row: Dict[str, Any] = {"date": date_str}
        for raw_field, raw_value in fields.items():
            canonical = _OHLC_MAP.get(_strip_av_prefix(raw_field))
            if canonical:
                row[canonical] = None if raw_value in _NULL_TOKENS else raw_value
        records.append(row)

    df = pd.DataFrame.from_records(records)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [c for c in df.columns if c != "date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    leading = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    rest = [c for c in df.columns if c not in leading]
    return df[leading + rest]


def load_benchmark_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the benchmark registry JSON (name -> {symbol, interval, aliases})."""
    path = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not path.exists():
        return {"default": None, "benchmarks": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_benchmark(name_or_symbol: str, registry: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Resolve a benchmark by friendly name, symbol, or alias.

    Returns (canonical_name, entry) or None if not found.
    """
    if not name_or_symbol:
        return None
    benchmarks = registry.get("benchmarks", {})
    needle = str(name_or_symbol).strip().lower()

    for name, entry in benchmarks.items():
        if name.lower() == needle:
            return name, entry
        if str(entry.get("symbol", "")).lower() == needle:
            return name, entry
        if any(str(a).lower() == needle for a in entry.get("aliases", [])):
            return name, entry
    return None


class BenchmarkCollector:
    """Fetches index series via INDEX_DATA and assembles tidy price frames."""

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    def _data(response: Any) -> Dict[str, Any]:
        if response is not None and getattr(response, "success", False):
            return response.data or {}
        return {}

    def collect(
        self,
        symbol: str,
        interval: str = "daily",
        outputsize: str = "full"
    ) -> pd.DataFrame:
        """Fetch and parse a single index series. Empty DataFrame on failure."""
        response = self.client.get_index_data(symbol, interval=interval, outputsize=outputsize)
        if not getattr(response, "success", False):
            logger.warning(
                "INDEX_DATA fetch failed for %s: %s",
                symbol, getattr(response, "error_message", "unknown error")
            )
            return pd.DataFrame()
        df = transform_index_data(self._data(response))
        if not df.empty:
            df["symbol"] = symbol
        return df

    def collect_registry(
        self,
        registry: Optional[Dict[str, Any]] = None,
        outputsize: str = "full"
    ) -> Dict[str, pd.DataFrame]:
        """Collect every benchmark in the registry. Returns {symbol: DataFrame}."""
        registry = registry or load_benchmark_registry()
        out: Dict[str, pd.DataFrame] = {}
        for name, entry in registry.get("benchmarks", {}).items():
            symbol = entry.get("symbol")
            interval = entry.get("interval", "daily")
            if not symbol:
                continue
            df = self.collect(symbol, interval=interval, outputsize=outputsize)
            if not df.empty:
                out[symbol] = df
        return out
