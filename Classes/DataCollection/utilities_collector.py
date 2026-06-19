"""
Utilities collector for Alpha Vantage.

"Utilities" are a *support* layer for the data-preparation pipeline rather than a
predictive feature family. This collector wraps the external utility endpoints:

  * MARKET_STATUS   -- current open/closed status of global markets (calendar
                       alignment, avoiding needless intraday refreshes).
  * LISTING_STATUS  -- active/delisted symbols as of a date (survivorship-aware
                       universes). CSV endpoint.
  * EARNINGS_CALENDAR -- upcoming earnings dates (release-timing helper). CSV.
  * INDEX_CATALOG   -- discoverable indices for benchmark mapping.

Computed utilities (trading-calendar alignment, feature-age counters, freshness
flags) are derived later in the data-preparation layer from the data actually
selected for a run; they do not require an API call and live there.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_VENDOR = "alpha_vantage"


def transform_market_status(response_data: Dict[str, Any]) -> pd.DataFrame:
    """Parse a MARKET_STATUS response into a tidy DataFrame, one row per market."""
    markets = (response_data or {}).get("markets")
    if not isinstance(markets, list) or not markets:
        return pd.DataFrame()
    df = pd.DataFrame.from_records([m for m in markets if isinstance(m, dict)])
    return df


def _parse_csv_payload(response_data: Dict[str, Any]) -> pd.DataFrame:
    """Parse a CSV payload wrapped as ``{"csv": <text>}`` into a DataFrame."""
    text = (response_data or {}).get("csv")
    if not text or not text.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse CSV utility payload: %s", exc)
        return pd.DataFrame()


class UtilitiesCollector:
    """Collects external utility data from Alpha Vantage."""

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    def _data(response: Any) -> Dict[str, Any]:
        if response is not None and getattr(response, "success", False):
            return response.data or {}
        return {}

    def collect_market_status(self) -> pd.DataFrame:
        """Fetch the current global market open/close status."""
        try:
            resp = self.client.get_market_status()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MARKET_STATUS failed: %s", exc)
            return pd.DataFrame()
        if not getattr(resp, "success", False):
            logger.warning("MARKET_STATUS failed: %s", getattr(resp, "error_message", "request failed"))
            return pd.DataFrame()
        df = transform_market_status(self._data(resp))
        if not df.empty:
            df["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        return df

    def collect_listing_status(
        self,
        date: Optional[str] = None,
        state: str = "active",
    ) -> pd.DataFrame:
        """Fetch active/delisted listings (optionally point-in-time by date)."""
        try:
            resp = self.client.get_listing_status(date=date, state=state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LISTING_STATUS failed: %s", exc)
            return pd.DataFrame()
        if not getattr(resp, "success", False):
            logger.warning("LISTING_STATUS failed: %s", getattr(resp, "error_message", "request failed"))
            return pd.DataFrame()
        return _parse_csv_payload(self._data(resp))

    def collect_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        horizon: str = "3month",
    ) -> pd.DataFrame:
        """Fetch the upcoming earnings calendar (optionally scoped to a symbol)."""
        try:
            resp = self.client.get_earnings_calendar(symbol=symbol, horizon=horizon)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("EARNINGS_CALENDAR failed: %s", exc)
            return pd.DataFrame()
        if not getattr(resp, "success", False):
            logger.warning("EARNINGS_CALENDAR failed: %s", getattr(resp, "error_message", "request failed"))
            return pd.DataFrame()
        return _parse_csv_payload(self._data(resp))

    def collect_index_catalog(self) -> Dict[str, Any]:
        """Fetch the index catalogue payload (best-effort; raw passthrough)."""
        try:
            resp = self.client.get_index_catalog()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("INDEX_CATALOG failed: %s", exc)
            return {}
        if not getattr(resp, "success", False):
            logger.warning("INDEX_CATALOG failed: %s", getattr(resp, "error_message", "request failed"))
            return {}
        return self._data(resp)
