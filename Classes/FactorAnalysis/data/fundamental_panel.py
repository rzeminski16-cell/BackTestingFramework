"""
Point-in-time fundamental factor panel for factor analysis.

Loads the rebuilt per-symbol point-in-time panels
(raw_data/fundamentals/{SYMBOL}_fundamental.csv) and derives the value/quality/
growth ratios the factor engine expects (ROE, ROA, margins, debt/equity,
current ratio, YoY growth, FCF) using TTM rolling sums on the quarterly rows.

Why this matters: the panels store *raw* point-in-time line items
(totalrevenue, netincome, totalshareholderequity, ...). The factor engine wants
*derived ratios* that previously only existed in the current-snapshot _overview
files (no history). Computing them here - per symbol, using only data up to each
fiscal period and keyed by the real publication date (report_date) - makes the
full fundamental factor set usable without look-ahead bias, replacing the old
EPS-only fallback.

Reliability: every derived value is computed from the panel's as-reported line
items; nothing is read from the current snapshot, and TTM windows never span
data the market hadn't seen by that period's report_date.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FUNDAMENTALS_DIR = Path("raw_data/fundamentals")
TTM = 4  # quarters in a trailing-twelve-month window

# Raw panel line items used for derivations (lowercased Alpha Vantage names).
_FLOW_ITEMS = {
    "totalrevenue": "revenue_ttm",
    "netincome": "net_income_ttm",
    "operatingincome": "operating_income_ttm",
    "grossprofit": "gross_profit_ttm",
    "ebitda": "ebitda_ttm",
    "operatingcashflow": "operating_cash_flow_ttm",
    "capitalexpenditures": "capex_ttm",
}
# Balance-sheet (point-in-time) items used as-of each quarter.
_BALANCE_ITEMS = ["totalassets", "totalliabilities", "totalshareholderequity",
                  "totalcurrentassets", "totalcurrentliabilities"]


def load_panels(
    directory: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
    frequency: str = "quarterly",
) -> pd.DataFrame:
    """
    Load per-symbol fundamental panels into one long DataFrame.

    Args:
        directory: folder containing {SYMBOL}_fundamental.csv (default raw_data/fundamentals).
        symbols: restrict to these symbols (else all files in the folder).
        frequency: 'quarterly' (default; required for TTM), 'annual', or 'both'.
    """
    directory = Path(directory) if directory else DEFAULT_FUNDAMENTALS_DIR
    if not directory.exists():
        return pd.DataFrame()

    frames = []
    for path in sorted(directory.glob("*_fundamental.csv")):
        symbol = path.name[: -len("_fundamental.csv")].upper()
        if symbols and symbol not in {s.upper() for s in symbols}:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read %s: %s", path, exc)
            continue
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    if frequency != "both" and "frequency" in panel.columns:
        panel = panel[panel["frequency"] == frequency].copy()

    for col in ("fiscaldateending", "report_date"):
        if col in panel.columns:
            panel[col] = pd.to_datetime(panel[col], errors="coerce")
    # Numeric coercion for everything except known text/date columns.
    text_cols = {"symbol", "frequency", "reportedcurrency", "reporttime",
                 "fiscaldateending", "report_date"}
    for col in panel.columns:
        if col not in text_cols:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def add_pit_derived_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add point-in-time derived fundamental factors per symbol.

    Adds TTM aggregates and the ratios the factor engine consumes:
    profit_margin, operating_margin_ttm, gross_margin, return_on_equity_ttm,
    return_on_assets_ttm, debt_to_equity, currentratio, revenue_growth_yoy,
    earnings_growth_yoy, freecashflow, and an 'eps' alias. Computed within each
    symbol on quarterly rows ordered by fiscal date, so no look-ahead.
    """
    if panel.empty or "symbol" not in panel.columns or "fiscaldateending" not in panel.columns:
        return panel

    out = panel.sort_values(["symbol", "fiscaldateending"]).reset_index(drop=True)
    g = out.groupby("symbol", group_keys=False)

    # TTM rolling sums of flow items (need 4 consecutive quarters).
    for raw, ttm_name in _FLOW_ITEMS.items():
        if raw in out.columns:
            out[ttm_name] = g[raw].transform(
                lambda s: s.rolling(TTM, min_periods=TTM).sum()
            )

    def col(name):
        return out[name] if name in out.columns else pd.Series(np.nan, index=out.index)

    def pos(series):
        """Mask non-positive denominators so ratios stay meaningful (e.g. ROE with
        negative equity, growth from a negative base are reported as NaN, not noise)."""
        return series.where(series > 0)

    rev = col("revenue_ttm")
    ni = col("net_income_ttm")
    equity = col("totalshareholderequity")
    assets = col("totalassets")

    # EPS (TTM) for an EPS-growth factor.
    if "reported_eps" in out.columns:
        out["eps_ttm"] = g["reported_eps"].transform(lambda s: s.rolling(TTM, min_periods=TTM).sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        # Margins (TTM).
        out["profit_margin"] = (ni / pos(rev)) * 100
        out["operating_margin_ttm"] = (col("operating_income_ttm") / pos(rev)) * 100
        out["gross_margin"] = (col("gross_profit_ttm") / pos(rev)) * 100
        out["ebitda_margin"] = (col("ebitda_ttm") / pos(rev)) * 100
        # Returns (TTM earnings over as-of balance), guarded against non-positive denominators.
        out["return_on_equity_ttm"] = (ni / pos(equity)) * 100
        out["return_on_assets_ttm"] = (ni / pos(assets)) * 100
        # Efficiency / leverage / liquidity.
        out["asset_turnover"] = rev / pos(assets)
        out["debt_to_equity"] = col("totalliabilities") / pos(equity)
        out["currentratio"] = col("totalcurrentassets") / pos(col("totalcurrentliabilities"))
        # Free cash flow (TTM) and its margin.
        out["freecashflow"] = col("operating_cash_flow_ttm") - col("capex_ttm").abs()
        out["fcf_margin"] = (out["freecashflow"] / pos(rev)) * 100
        # Accruals (Sloan): (TTM net income - TTM operating cash flow) / assets; lower is higher quality.
        out["accruals_ratio"] = (ni - col("operating_cash_flow_ttm")) / pos(assets)

    # Year-over-year growth (TTM vs 4 quarters prior), per symbol. Re-group so the
    # transforms see the TTM columns just added.
    g2 = out.groupby("symbol", group_keys=False)

    def _yoy(series):
        prior = series.shift(TTM)
        return (series / pos(prior) - 1) * 100

    for ttm_col, growth_col in (("revenue_ttm", "revenue_growth_yoy"),
                                ("net_income_ttm", "earnings_growth_yoy"),
                                ("eps_ttm", "eps_growth_yoy"),
                                ("freecashflow", "fcf_growth_yoy")):
        if ttm_col in out.columns:
            out[growth_col] = g2[ttm_col].transform(_yoy)

    # Share-count change YoY (buyback/dilution signal).
    if "commonstocksharesoutstanding" in out.columns:
        out["shares_growth_yoy"] = g2["commonstocksharesoutstanding"].transform(_yoy)

    # 'eps' alias so EPS factors resolve (panel uses reported_eps).
    if "reported_eps" in out.columns and "eps" not in out.columns:
        out["eps"] = out["reported_eps"]

    # Tidy infinities from divisions.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an in-memory fundamental DataFrame (e.g. one the GUI assembled from
    per-symbol panels) and add the point-in-time derived factors.

    Lowercases columns, parses dates, coerces numerics, then derives ratios. On
    any failure it returns the normalized frame unchanged (never raises), so
    callers can use it defensively.
    """
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    for col in ("fiscaldateending", "report_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    text_cols = {"symbol", "frequency", "reportedcurrency", "reporttime",
                 "fiscaldateending", "report_date"}
    for col in out.columns:
        if col not in text_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "symbol" in out.columns and "fiscaldateending" in out.columns:
        try:
            out = add_pit_derived_factors(out)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Fundamental enrichment skipped: %s", exc)
    return out


def build_panel(
    directory: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
    frequency: str = "quarterly",
) -> pd.DataFrame:
    """Load panels and add point-in-time derived factors (the common entry point)."""
    return add_pit_derived_factors(load_panels(directory, symbols, frequency))


def build_aggregate(
    directory: Optional[Path] = None,
    out_path: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
    frequency: str = "quarterly",
) -> pd.DataFrame:
    """
    Build the aggregate point-in-time fundamental table and optionally write it.

    Returns the DataFrame; if out_path is given, also writes it as CSV
    (default destination raw_data/fundamentals/fundamental_data.csv).
    """
    directory = Path(directory) if directory else DEFAULT_FUNDAMENTALS_DIR
    df = build_panel(directory, symbols, frequency)
    if df.empty:
        return df
    if out_path is None:
        out_path = directory / "fundamental_data.csv"
    out_path = Path(out_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not write aggregate %s: %s", out_path, exc)
    return df
