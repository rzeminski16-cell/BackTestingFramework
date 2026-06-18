"""
Trade source -- discover and load backtest trades into a trade-centred table.

The dataset is trade-centred, so the trade list is the anchor for everything
else. This module reads the framework's existing CSV trade logs (written by
``Classes/Analysis/trade_logger.py`` under ``logs/``) into a normalised
``selected_trades`` table, validates the required keys, and summarises the
universe (count, date range, instruments, asset-class mix) for the GUI.

It is deliberately standalone (it does not import the FactorAnalysis loader) so
the new pipeline can evolve independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .schema import TRADE_REQUIRED_COLUMNS

_FX_PAIR_RE = re.compile(r"^[A-Z]{6}$")
_TRADE_LOG_PATTERNS = ("*_trades.csv", "portfolio_trades.csv")


def _infer_asset_class(symbol: str) -> str:
    """Best-effort asset-class inference from a symbol when none is provided."""
    if not isinstance(symbol, str) or not symbol:
        return "unknown"
    s = symbol.upper().replace("/", "")
    if _FX_PAIR_RE.match(s):
        return "fx"
    return "equity"


@dataclass
class TradeUniverseSummary:
    """Summary statistics for a selected trade universe (for the GUI preview)."""
    trade_count: int
    n_symbols: int
    symbols: List[str]
    date_range: Tuple[Optional[str], Optional[str]]
    asset_class_mix: Dict[str, int]
    currencies: List[str]
    source_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_count": self.trade_count,
            "n_symbols": self.n_symbols,
            "symbols": self.symbols,
            "date_range": list(self.date_range),
            "asset_class_mix": self.asset_class_mix,
            "currencies": self.currencies,
            "source_files": self.source_files,
        }


class TradeSource:
    """Discovers, loads, validates and summarises backtest trade logs."""

    def __init__(self, logs_dir: Union[str, Path] = "logs"):
        self.logs_dir = Path(logs_dir)

    # -- discovery ---------------------------------------------------------- #
    def discover(self) -> List[Dict[str, Any]]:
        """
        Find available trade-log files under ``logs_dir``.

        Returns a list of ``{path, name, kind}`` dicts (kind is "portfolio",
        "single_security" or "other") for populating a selector. Consolidated
        ``portfolio_trades.csv`` files are preferred over per-symbol files in the
        same directory to avoid double counting.
        """
        if not self.logs_dir.exists():
            return []

        found: List[Dict[str, Any]] = []
        seen_dirs_with_portfolio = set()

        # First pass: consolidated portfolio files.
        for path in sorted(self.logs_dir.rglob("portfolio_trades.csv")):
            seen_dirs_with_portfolio.add(path.parent)
            found.append({
                "path": str(path),
                "name": path.parent.name,
                "kind": "portfolio",
            })

        # Second pass: per-symbol / single-security files, skipping dirs already
        # represented by a consolidated portfolio file.
        for path in sorted(self.logs_dir.rglob("*_trades.csv")):
            if path.name == "portfolio_trades.csv":
                continue
            if path.parent in seen_dirs_with_portfolio:
                continue
            kind = "single_security" if "single_security" in path.parts else "other"
            found.append({
                "path": str(path),
                "name": f"{path.parent.name}/{path.stem}",
                "kind": kind,
            })
        return found

    # -- loading ------------------------------------------------------------ #
    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

        if "date" not in df.columns and "time" in df.columns:
            df = df.rename(columns={"time": "date"})

        for col in ("entry_date", "exit_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        numeric = ["entry_price", "exit_price", "pl", "pl_pct", "quantity",
                   "duration_days", "commission_paid", "security_pl", "fx_pl",
                   "entry_fx_rate", "exit_fx_rate"]
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "asset_class" not in df.columns and "symbol" in df.columns:
            df["asset_class"] = df["symbol"].map(_infer_asset_class)

        if "security_currency" in df.columns and "currency" not in df.columns:
            df = df.rename(columns={"security_currency": "currency"})

        return df

    def load(
        self,
        paths: List[Union[str, Path]],
        validate: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and combine one or more trade-log files.

        Returns ``(trades_df, issues)``. ``issues`` is a list of human-readable
        problems; when ``validate`` is True and required keys are missing, the
        issues list is non-empty (callers should block export). ``trade_id`` is
        made unique across files by prefixing the source stem when needed.
        """
        frames: List[pd.DataFrame] = []
        issues: List[str] = []

        for p in paths:
            p = Path(p)
            if not p.exists():
                issues.append(f"Trade log not found: {p}")
                continue
            try:
                df = pd.read_csv(p)
            except Exception as exc:
                issues.append(f"Could not read {p.name}: {exc}")
                continue
            df = self._normalise(df)
            df["_source_file"] = p.stem
            frames.append(df)

        if not frames:
            return pd.DataFrame(), issues or ["No trade logs loaded."]

        trades = pd.concat(frames, ignore_index=True)

        # Make trade_id unique across multiple source files.
        if "trade_id" in trades.columns and trades["_source_file"].nunique() > 1:
            trades["trade_id"] = (
                trades["_source_file"].astype(str) + "_" + trades["trade_id"].astype(str)
            )

        if validate:
            issues.extend(self.validate(trades))

        return trades, issues

    # -- validation & summary ---------------------------------------------- #
    @staticmethod
    def validate(trades: pd.DataFrame) -> List[str]:
        """Return a list of blocking issues (missing required keys / bad dates)."""
        issues: List[str] = []
        if trades.empty:
            return ["Trade table is empty."]

        for col in TRADE_REQUIRED_COLUMNS:
            if col not in trades.columns:
                issues.append(f"Missing required trade column: '{col}'.")

        if "entry_date" in trades.columns and trades["entry_date"].isna().any():
            n = int(trades["entry_date"].isna().sum())
            issues.append(f"{n} trade(s) have an unparseable/missing entry_date.")

        if "trade_id" in trades.columns and trades["trade_id"].duplicated().any():
            n = int(trades["trade_id"].duplicated().sum())
            issues.append(f"{n} duplicate trade_id value(s) found.")

        return issues

    @staticmethod
    def summarise(trades: pd.DataFrame, source_files: Optional[List[str]] = None) -> TradeUniverseSummary:
        """Compute a :class:`TradeUniverseSummary` for the GUI preview."""
        if trades.empty:
            return TradeUniverseSummary(0, 0, [], (None, None), {}, [], source_files or [])

        symbols = sorted(trades["symbol"].dropna().unique().tolist()) if "symbol" in trades else []

        start = end = None
        if "entry_date" in trades.columns and trades["entry_date"].notna().any():
            start = trades["entry_date"].min().strftime("%Y-%m-%d")
        end_col = "exit_date" if "exit_date" in trades.columns else "entry_date"
        if end_col in trades.columns and trades[end_col].notna().any():
            end = trades[end_col].max().strftime("%Y-%m-%d")

        mix: Dict[str, int] = {}
        if "asset_class" in trades.columns:
            mix = {str(k): int(v) for k, v in trades["asset_class"].value_counts().items()}

        currencies = (
            sorted(trades["currency"].dropna().unique().tolist())
            if "currency" in trades.columns else []
        )

        return TradeUniverseSummary(
            trade_count=int(len(trades)),
            n_symbols=len(symbols),
            symbols=symbols,
            date_range=(start, end),
            asset_class_mix=mix,
            currencies=currencies,
            source_files=source_files or [],
        )
