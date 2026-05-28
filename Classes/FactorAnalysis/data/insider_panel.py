"""
Point-in-time insider-activity feature panel.

Turns cleaned per-symbol insider transactions (raw_data/insider_transactions/
{SYMBOL}_insider.csv) into trailing-window features indexed by the date the
information became public (transaction date + filing delay), so they can be
joined to trades without look-ahead bias.

For each (symbol, available_date) with activity it computes, over 30/90/180-day
trailing windows: buy/sell counts, net shares, net value, executive net,
distinct buyers, buy ratio, plus a cluster-buy flag and days-since-last-buy.

Reliability: every feature is computed only from transactions whose
available_date is on/before the row's available_date (no future leakage), and
the executive flag is derived from the SEC title rather than assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INSIDER_DIR = Path("raw_data/insider_transactions")
DEFAULT_WINDOWS = (30, 90, 180)
DEFAULT_FILING_DELAY_DAYS = 3
DEFAULT_CLUSTER_MIN_BUYERS = 3

# SEC title keywords that mark an insider as an executive/officer/director.
EXEC_TITLE_PATTERN = (
    r"\b(?:CEO|CFO|COO|CTO|CIO|President|Chief|Officer|Director|Chairman|"
    r"Chairwoman|VP|Vice[\s-]?President|Treasurer|Secretary|EVP|SVP|Founder|Partner)\b"
)


def is_executive_from_title(titles: pd.Series) -> pd.Series:
    return titles.astype(str).str.contains(EXEC_TITLE_PATTERN, case=False, regex=True, na=False)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize/clean an insider transaction frame (idempotent, source-agnostic)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).lower().strip().replace(" ", "_") for c in out.columns]

    # Map common alternative column names.
    renames = {"transaction_date": "date", "ticker": "symbol",
               "executive": "insider_name", "executive_title": "insider_title",
               "share_price": "price"}
    out = out.rename(columns={k: v for k, v in renames.items() if k in out.columns and v not in out.columns})

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "transaction_type" in out.columns:
        tt = out["transaction_type"].astype(str).str.strip().str.lower()
        out["transaction_type"] = tt.replace({
            "a": "buy", "d": "sell", "p": "buy", "s": "sell",
            "purchase": "buy", "sale": "sell",
            "acquisition": "buy", "disposition": "sell", "disposal": "sell",
        })

    for col in ("shares", "price", "value"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "value" not in out.columns and "shares" in out.columns and "price" in out.columns:
        out["value"] = out["shares"] * out["price"]

    # Executive flag: use existing boolean, else derive from title.
    if "is_executive" in out.columns:
        out["is_executive"] = out["is_executive"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes", "t")
        ) if out["is_executive"].dtype == object else out["is_executive"].astype(bool)
    elif "insider_title" in out.columns:
        out["is_executive"] = is_executive_from_title(out["insider_title"])
    else:
        out["is_executive"] = False

    if "date" in out.columns:
        out = out.dropna(subset=["date"])
    if "shares" in out.columns:
        out = out[out["shares"].fillna(0) > 0]
    keys = [c for c in ["date", "symbol", "insider_name", "transaction_type", "shares", "price"]
            if c in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys)
    return out.sort_values("date").reset_index(drop=True) if "date" in out.columns else out


def load_transactions(
    directory: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load and clean all per-symbol insider transactions into one long frame
    (raw transactions, not aggregated) - the input the analyzer's aligner needs."""
    directory = Path(directory) if directory else DEFAULT_INSIDER_DIR
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(directory.glob("*_insider.csv")):
        symbol = path.name[: -len("_insider.csv")].upper()
        if symbols and symbol not in {s.upper() for s in symbols}:
            continue
        try:
            raw = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if "symbol" not in (c.lower() for c in raw.columns):
            raw["symbol"] = symbol
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    return clean_transactions(pd.concat(frames, ignore_index=True))


def add_pit_features(
    df: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    filing_delay_days: int = DEFAULT_FILING_DELAY_DAYS,
    cluster_min_buyers: int = DEFAULT_CLUSTER_MIN_BUYERS,
) -> pd.DataFrame:
    """
    Build one feature row per (symbol, available_date) with trailing-window
    insider aggregates. available_date = transaction date + filing delay.
    """
    df = clean_transactions(df)
    if df.empty or "date" not in df.columns or "symbol" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["_available_date"] = df["date"] + pd.Timedelta(days=filing_delay_days)
    df["transaction_type"] = df["transaction_type"].fillna("")
    df["shares"] = df.get("shares", pd.Series(0, index=df.index)).fillna(0.0)
    df["value"] = df.get("value", pd.Series(np.nan, index=df.index))
    df["is_executive"] = df.get("is_executive", pd.Series(False, index=df.index)).fillna(False)
    if "insider_name" not in df.columns:
        df["insider_name"] = ""

    rows = []
    for symbol, sub in df.groupby("symbol"):
        sub = sub.sort_values("_available_date").reset_index(drop=True)
        avail = sub["_available_date"].to_numpy()
        is_buy = (sub["transaction_type"] == "buy").to_numpy()
        is_sell = (sub["transaction_type"] == "sell").to_numpy()
        shares = sub["shares"].to_numpy(dtype=float)
        value = pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype=float)
        is_exec = sub["is_executive"].to_numpy(dtype=bool)
        names = sub["insider_name"].astype(str).to_numpy()

        for d in pd.unique(sub["_available_date"]):
            row = {"symbol": symbol, "available_date": pd.Timestamp(d)}
            # most recent buy on/before d
            buy_dates = avail[is_buy & (avail <= d)]
            row["days_since_last_buy"] = (
                int((pd.Timestamp(d) - pd.Timestamp(buy_dates.max())).days) if len(buy_dates) else np.nan
            )
            for w in windows:
                lo = d - np.timedelta64(w, "D")
                mask = (avail > lo) & (avail <= d)
                b = mask & is_buy
                s = mask & is_sell
                bc, sc = int(b.sum()), int(s.sum())
                buy_sh, sell_sh = float(shares[b].sum()), float(shares[s].sum())
                buy_val = float(np.nansum(value[b]))
                sell_val = float(np.nansum(value[s]))
                pre = f"ins_{w}d_"
                row[pre + "buy_count"] = bc
                row[pre + "sell_count"] = sc
                row[pre + "net_count"] = bc - sc
                row[pre + "net_shares"] = buy_sh - sell_sh
                row[pre + "net_value"] = buy_val - sell_val
                row[pre + "buy_ratio"] = bc / (bc + sc) if (bc + sc) > 0 else np.nan
                row[pre + "exec_net"] = int((b & is_exec).sum()) - int((s & is_exec).sum())
                distinct_buyers = int(pd.unique(names[b]).size) if bc else 0
                row[pre + "distinct_buyers"] = distinct_buyers
                if w == 90:
                    row["cluster_buy_flag"] = int(distinct_buyers >= cluster_min_buyers)
            rows.append(row)

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["symbol", "available_date"]).reset_index(drop=True)
    return panel


def build_aggregate(
    directory: Optional[Path] = None,
    out_path: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    filing_delay_days: int = DEFAULT_FILING_DELAY_DAYS,
) -> pd.DataFrame:
    """
    Load every per-symbol insider file, clean + feature-engineer, concatenate,
    and (optionally) write the aggregate point-in-time panel to CSV.
    """
    directory = Path(directory) if directory else DEFAULT_INSIDER_DIR
    if not directory.exists():
        return pd.DataFrame()

    frames = []
    for path in sorted(directory.glob("*_insider.csv")):
        symbol = path.name[: -len("_insider.csv")].upper()
        if symbols and symbol not in {s.upper() for s in symbols}:
            continue
        try:
            raw = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if "symbol" not in (c.lower() for c in raw.columns):
            raw["symbol"] = symbol
        frames.append(raw)

    if not frames:
        return pd.DataFrame()

    panel = add_pit_features(pd.concat(frames, ignore_index=True),
                             windows=windows, filing_delay_days=filing_delay_days)
    if panel.empty:
        return panel

    if out_path is None:
        out_path = directory / "insider_data.csv"
    out_path = Path(out_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not write insider aggregate %s: %s", out_path, exc)
    return panel
