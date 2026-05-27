"""
Point-in-time fundamental data collector for Alpha Vantage.

This replaces the previous assembly step (a vertical ``pd.concat`` of five
endpoint responses followed by a forward-fill) which produced a 120-column,
mostly-empty file in which:

  * each fiscal period was fragmented across up to four separate rows
    (earnings / income / balance / cash) that were never joined, and
  * the forward-fill smeared the *current* OVERVIEW snapshot (P/E, beta, EPS,
    moving averages, ...) across every historical row, collapsing those columns
    to a single value and injecting look-ahead bias.

The new layout is a tidy, "as-reported" panel:

  * one row per (symbol, frequency, fiscal_date_ending),
  * the three financial statements joined on ``fiscalDateEnding``,
  * a real publication date (``report_date``) recovered from the EARNINGS
    endpoint - the only Alpha Vantage endpoint that exposes when a period was
    actually reported - so downstream code can index point-in-time and avoid
    look-ahead bias,
  * honest ``NaN`` for missing line items (no forward-fill), and
  * no point-in-time-now snapshot columns.

The OVERVIEW snapshot is captured separately (:func:`extract_overview_snapshot`)
because it has no history and must never be written onto historical rows.

Column naming keeps the previous convention - the lowercased raw Alpha Vantage
field name (e.g. ``totalShareholderEquity`` -> ``totalshareholderequity``) - so
existing consumers (e.g. ``Classes/FactorAnalysis/data/fundamental_loader.py``)
keep working while the structure and values are corrected.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Default number of days to add to ``fiscal_date_ending`` to estimate the
# publication date when the EARNINGS endpoint has no matching ``reportedDate``
# (e.g. for periods older than earnings coverage). Conservative on purpose.
DEFAULT_REPORTING_LAG_DAYS = 60

# Leading/key columns, in display order.
KEY_COLUMNS = [
    "symbol",
    "frequency",
    "fiscaldateending",
    "report_date",
    "reporttime",
    "reportedcurrency",
    "reported_eps",
    "estimated_eps",
    "earnings_surprise",
    "surprise_pct",
]

# Columns that must never be coerced to numeric.
TEXT_COLUMNS = {"symbol", "frequency", "reportedcurrency", "reporttime"}

# Columns parsed as dates.
DATE_COLUMNS = {"fiscaldateending", "report_date"}

# EARNINGS quarterlyEarnings field -> canonical column name.
_EARNINGS_QUARTERLY_MAP = {
    "reportedDate": "report_date",
    "reportTime": "reporttime",
    "reportedEPS": "reported_eps",
    "estimatedEPS": "estimated_eps",
    "surprise": "earnings_surprise",
    "surprisePercentage": "surprise_pct",
}

_NULL_TOKENS = {None, "", "None", "-", "NaN", "nan"}


def _index_reports(reports: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Index a list of statement reports by ``fiscalDateEnding`` (lowercased keys)."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for report in reports or []:
        fde = report.get("fiscalDateEnding")
        if not fde:
            continue
        indexed[fde] = {k.lower(): v for k, v in report.items()}
    return indexed


def _index_earnings_quarterly(earnings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index quarterlyEarnings by ``fiscalDateEnding`` with canonical column names."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for entry in (earnings or {}).get("quarterlyEarnings", []) or []:
        fde = entry.get("fiscalDateEnding")
        if not fde:
            continue
        indexed[fde] = {
            canonical: entry.get(raw)
            for raw, canonical in _EARNINGS_QUARTERLY_MAP.items()
        }
    return indexed


def _index_earnings_annual(earnings: Dict[str, Any]) -> Dict[str, Any]:
    """Index annualEarnings reportedEPS by ``fiscalDateEnding``."""
    indexed: Dict[str, Any] = {}
    for entry in (earnings or {}).get("annualEarnings", []) or []:
        fde = entry.get("fiscalDateEnding")
        if fde:
            indexed[fde] = entry.get("reportedEPS")
    return indexed


def _merge_statement_row(*sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge per-statement field dicts for a single fiscal period.

    Sources are applied in priority order; a later source only fills keys that
    are still missing/empty. This resolves the fields shared across statements
    (``reportedcurrency`` in all three, ``netincome`` in income + cash flow) in
    favour of the first (authoritative) source without dropping anything.
    """
    row: Dict[str, Any] = {}
    for source in sources:
        if not source:
            continue
        for key, value in source.items():
            if key not in row or row[key] in _NULL_TOKENS:
                row[key] = value
    return row


def _build_frequency_records(
    inc_idx: Dict[str, Dict[str, Any]],
    bal_idx: Dict[str, Dict[str, Any]],
    cf_idx: Dict[str, Dict[str, Any]],
    earn_q_idx: Dict[str, Dict[str, Any]],
    earn_a_idx: Optional[Dict[str, Any]],
    symbol: str,
    frequency: str,
) -> List[Dict[str, Any]]:
    """Build the list of row dicts for one frequency (quarterly or annual)."""
    # For quarterly we also include earnings-only periods (Alpha Vantage's
    # earnings history reaches back further than its statement history), so the
    # reported-EPS series is preserved. For annual we union the statement and
    # annual-earnings periods.
    fiscal_dates = set(inc_idx) | set(bal_idx) | set(cf_idx)
    if frequency == "quarterly":
        fiscal_dates |= set(earn_q_idx)
    else:
        fiscal_dates |= set(earn_a_idx or {})

    records: List[Dict[str, Any]] = []
    for fde in sorted(fiscal_dates):
        record = _merge_statement_row(inc_idx.get(fde), bal_idx.get(fde), cf_idx.get(fde))
        record["fiscaldateending"] = fde
        record["symbol"] = symbol
        record["frequency"] = frequency

        earnings_match = earn_q_idx.get(fde, {})
        if frequency == "quarterly":
            for key, value in earnings_match.items():
                record.setdefault(key, value)
        else:
            # Annual fiscal-year-end equals the Q4 quarter-end, so the matching
            # quarterly earnings row gives a real publication date / report time.
            record.setdefault("report_date", earnings_match.get("report_date"))
            record.setdefault("reporttime", earnings_match.get("reporttime"))
            if earn_a_idx and fde in earn_a_idx:
                record["reported_eps"] = earn_a_idx[fde]

        records.append(record)
    return records


def build_fundamental_panel(
    raw: Dict[str, Any],
    symbol: str,
    frequency: str = "both",
    reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS,
) -> pd.DataFrame:
    """
    Assemble a tidy, point-in-time fundamental panel from raw endpoint responses.

    Args:
        raw: Mapping with keys ``income_statement``, ``balance_sheet``,
            ``cash_flow`` and ``earnings`` (each the parsed Alpha Vantage JSON).
        symbol: Ticker symbol.
        frequency: ``"both"``, ``"quarterly"`` or ``"annual"``.
        reporting_lag_days: Days added to ``fiscal_date_ending`` to estimate
            ``report_date`` when no earnings publication date is available.

    Returns:
        DataFrame with one row per (frequency, fiscal_date_ending), sorted by
        frequency then fiscal date. Empty if no data is available.
    """
    inc = raw.get("income_statement") or {}
    bal = raw.get("balance_sheet") or {}
    cf = raw.get("cash_flow") or {}
    earnings = raw.get("earnings") or {}

    earn_q_idx = _index_earnings_quarterly(earnings)
    earn_a_idx = _index_earnings_annual(earnings)

    want_quarterly = frequency in ("both", "quarterly")
    want_annual = frequency in ("both", "annual")

    records: List[Dict[str, Any]] = []
    if want_quarterly:
        records += _build_frequency_records(
            _index_reports(inc.get("quarterlyReports")),
            _index_reports(bal.get("quarterlyReports")),
            _index_reports(cf.get("quarterlyReports")),
            earn_q_idx,
            None,
            symbol,
            "quarterly",
        )
    if want_annual:
        records += _build_frequency_records(
            _index_reports(inc.get("annualReports")),
            _index_reports(bal.get("annualReports")),
            _index_reports(cf.get("annualReports")),
            earn_q_idx,
            earn_a_idx,
            symbol,
            "annual",
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df = _coerce_types(df, reporting_lag_days)
    df = _order_and_sort(df)
    return df


def _coerce_types(df: pd.DataFrame, reporting_lag_days: int) -> pd.DataFrame:
    """Parse dates, coerce numerics, and fill the report_date fallback."""
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # report_date fallback: fiscal_date_ending + reporting lag where unknown.
    if "fiscaldateending" in df.columns:
        if "report_date" not in df.columns:
            df["report_date"] = pd.NaT
        missing = df["report_date"].isna()
        df.loc[missing, "report_date"] = (
            df.loc[missing, "fiscaldateending"] + pd.Timedelta(days=reporting_lag_days)
        )

    for col in df.columns:
        if col in TEXT_COLUMNS or col in DATE_COLUMNS:
            if col in TEXT_COLUMNS:
                df[col] = df[col].astype("string").str.strip()
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _order_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Order key columns first, then remaining columns alphabetically; sort rows."""
    leading = [c for c in KEY_COLUMNS if c in df.columns]
    rest = sorted(c for c in df.columns if c not in leading)
    df = df[leading + rest]

    sort_cols = [c for c in ["frequency", "fiscaldateending"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def extract_overview_snapshot(
    overview: Dict[str, Any],
    symbol: str,
    collected_at: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Capture the current OVERVIEW snapshot as a single dated row.

    These values are point-in-time-NOW only (no history is available from Alpha
    Vantage) and must never be backfilled onto historical rows. They are stored
    separately for current screening and tagged with the collection date.
    """
    if not overview or "Symbol" not in overview:
        return pd.DataFrame()

    collected_at = collected_at or pd.Timestamp.utcnow().normalize()
    row: Dict[str, Any] = {"symbol": symbol, "as_of_date": collected_at}
    for key, value in overview.items():
        row[key.lower()] = None if value in _NULL_TOKENS else value
    return pd.DataFrame([row])


def merge_panels(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a freshly collected panel with a previously stored one.

    Because Alpha Vantage only serves a limited window of statement history,
    merging lets the local store accumulate periods that age out of the API.
    Rows are de-duplicated on (symbol, frequency, fiscal_date_ending), keeping
    the most recently collected version.
    """
    if existing is None or existing.empty:
        return new
    if new is None or new.empty:
        return existing

    combined = pd.concat([existing, new], ignore_index=True)
    if "fiscaldateending" in combined.columns:
        combined["fiscaldateending"] = pd.to_datetime(
            combined["fiscaldateending"], errors="coerce"
        )
    if "report_date" in combined.columns:
        combined["report_date"] = pd.to_datetime(combined["report_date"], errors="coerce")

    keys = [c for c in ["symbol", "frequency", "fiscaldateending"] if c in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    return _order_and_sort(combined)


class FundamentalCollector:
    """
    Orchestrates fetching the fundamental endpoints and assembling the panel.

    Usage::

        collector = FundamentalCollector(api_client)
        panel, snapshot = collector.collect("AAPL", frequency="both")
    """

    def __init__(self, client: Any, reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS):
        self.client = client
        self.reporting_lag_days = reporting_lag_days

    @staticmethod
    def _data(response: Any) -> Dict[str, Any]:
        """Extract the payload from an APIResponse, or {} on failure."""
        if response is not None and getattr(response, "success", False):
            return response.data or {}
        return {}

    def collect(
        self, symbol: str, frequency: str = "both"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect and assemble fundamentals for one symbol.

        Returns:
            (panel, overview_snapshot) - either may be empty if unavailable.
        """
        raw = {
            "income_statement": self._data(self.client.get_income_statement(symbol)),
            "balance_sheet": self._data(self.client.get_balance_sheet(symbol)),
            "cash_flow": self._data(self.client.get_cash_flow(symbol)),
            "earnings": self._data(self.client.get_earnings(symbol)),
        }
        overview = self._data(self.client.get_company_overview(symbol))

        panel = build_fundamental_panel(
            raw, symbol, frequency=frequency, reporting_lag_days=self.reporting_lag_days
        )
        snapshot = extract_overview_snapshot(overview, symbol)
        return panel, snapshot
