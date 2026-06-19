"""
Corporate-actions collector for Alpha Vantage (dividends & splits).

Dividends and splits are kept as *separate event tables* (never folded into a
price series) so the data-preparation layer can:

  * reconcile raw vs adjusted prices deterministically, and
  * compute age-since-event features (days_since_last_dividend / _split).

Dividend events are anchored on ``ex_dividend_date`` and split events on
``effective_date`` -- the dates at which the action affects tradable prices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_VENDOR = "alpha_vantage"
_NULL_TOKENS = {None, "", "None", "-", "NaN", "nan", "null"}

_DIVIDEND_DATE_COLS = ["declaration_date", "ex_dividend_date", "record_date", "payment_date"]


def _clean(value: Any) -> Any:
    return None if value in _NULL_TOKENS else value


def transform_dividends(response_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
    """Parse a DIVIDENDS response into a tidy dividend-event DataFrame."""
    rows = (response_data or {}).get("data")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        records.append({
            "symbol": symbol,
            "action_type": "dividend",
            "declaration_date": _clean(entry.get("declaration_date")),
            "ex_dividend_date": _clean(entry.get("ex_dividend_date")),
            "record_date": _clean(entry.get("record_date")),
            "payment_date": _clean(entry.get("payment_date")),
            "amount": _clean(entry.get("amount")),
        })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    for col in _DIVIDEND_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["ex_dividend_date"]).sort_values("ex_dividend_date").reset_index(drop=True)
    return df


def transform_splits(response_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
    """Parse a SPLITS response into a tidy split-event DataFrame."""
    rows = (response_data or {}).get("data")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        records.append({
            "symbol": symbol,
            "action_type": "split",
            "effective_date": _clean(entry.get("effective_date")),
            "split_factor": _clean(entry.get("split_factor")),
        })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")
    df = df.dropna(subset=["effective_date"]).sort_values("effective_date").reset_index(drop=True)
    return df


@dataclass
class CorporateActionsResult:
    """Dividends and splits for one symbol, as separate event tables."""
    symbol: str
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame)
    splits: pd.DataFrame = field(default_factory=pd.DataFrame)
    retrieved_at: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.dividends.empty and self.splits.empty


class CorporateActionsCollector:
    """Collects dividend and split events for one or more symbols."""

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
        include_dividends: bool = True,
        include_splits: bool = True,
    ) -> CorporateActionsResult:
        """Fetch dividends and/or splits for a single symbol."""
        retrieved_at = datetime.now(timezone.utc).isoformat()
        result = CorporateActionsResult(symbol=symbol, retrieved_at=retrieved_at)

        if include_dividends:
            try:
                resp = self.client.get_dividends(symbol)
                if getattr(resp, "success", False):
                    df = transform_dividends(self._data(resp), symbol)
                    if not df.empty:
                        df["source_vendor"] = SOURCE_VENDOR
                        df["retrieved_at"] = retrieved_at
                    result.dividends = df
                else:
                    result.errors.append(
                        f"DIVIDENDS({symbol}): {getattr(resp, 'error_message', 'request failed')}"
                    )
            except Exception as exc:  # pragma: no cover - defensive
                result.errors.append(f"DIVIDENDS({symbol}): {exc}")

        if include_splits:
            try:
                resp = self.client.get_splits(symbol)
                if getattr(resp, "success", False):
                    df = transform_splits(self._data(resp), symbol)
                    if not df.empty:
                        df["source_vendor"] = SOURCE_VENDOR
                        df["retrieved_at"] = retrieved_at
                    result.splits = df
                else:
                    result.errors.append(
                        f"SPLITS({symbol}): {getattr(resp, 'error_message', 'request failed')}"
                    )
            except Exception as exc:  # pragma: no cover - defensive
                result.errors.append(f"SPLITS({symbol}): {exc}")

        return result

    def collect_many(
        self,
        symbols: List[str],
        include_dividends: bool = True,
        include_splits: bool = True,
    ) -> Dict[str, CorporateActionsResult]:
        """Fetch corporate actions for several symbols. Returns {symbol: result}."""
        return {
            sym: self.collect(sym, include_dividends, include_splits)
            for sym in symbols
        }
