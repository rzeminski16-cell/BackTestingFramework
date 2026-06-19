"""
Shared transformer for Alpha Vantage "flat series" responses.

The commodities and economic-indicator endpoints all share one response shape::

    {
        "name": "Crude Oil Prices WTI",
        "interval": "monthly",
        "unit": "dollars per barrel",
        "data": [
            {"date": "2024-01-01", "value": "73.85"},
            {"date": "2023-12-01", "value": "71.90"},
            ...
        ]
    }

This module parses that into a tidy ``observation_date``/``value`` DataFrame
(ascending by date) plus the series metadata. Values use the FRED missing-data
token "." which is coerced to NaN. The output deliberately stops at the raw
observation level: availability timestamps, run identifiers and quality flags
are assigned later by the data-preparation layer, which owns timing policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

# Tokens Alpha Vantage / FRED use for "no observation".
_NULL_TOKENS = {None, "", ".", "-", "None", "NaN", "nan", "null"}


def transform_flat_series(response_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse a flat AV series response into ``(DataFrame, metadata)``.

    Returns:
        A tuple of:
          * DataFrame with columns ``observation_date`` (datetime64) and
            ``value`` (float), sorted ascending and de-duplicated by date.
            Empty if the payload has no parseable data.
          * metadata dict with ``name``, ``interval`` and ``unit`` (any missing
            key defaults to None).
    """
    meta = {
        "name": (response_data or {}).get("name"),
        "interval": (response_data or {}).get("interval"),
        "unit": (response_data or {}).get("unit"),
    }

    rows = (response_data or {}).get("data")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["observation_date", "value"]), meta

    records: List[Dict[str, Any]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        date_str = entry.get("date")
        raw_value = entry.get("value")
        value = None if raw_value in _NULL_TOKENS else raw_value
        records.append({"observation_date": date_str, "value": value})

    df = pd.DataFrame.from_records(records)
    if df.empty or "observation_date" not in df.columns:
        return pd.DataFrame(columns=["observation_date", "value"]), meta

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = (
        df.dropna(subset=["observation_date"])
        .drop_duplicates(subset=["observation_date"], keep="last")
        .sort_values("observation_date")
        .reset_index(drop=True)
    )
    return df, meta
