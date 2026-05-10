"""
Trade log loading for Monte Carlo simulation.

Auto-detects the canonical BackTestingFramework trade log schema and falls back
to user-selected columns for foreign CSVs. Supports loading and concatenating
multiple files in a single batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class TradeLogReturnSource(str, Enum):
    """Which column should drive the simulation."""
    PCT_RETURN = "pct_return"   # pl_pct as a percentage (e.g. 1.5 == +1.5%)
    R_MULTIPLE = "r_multiple"   # R-multiple, computed or read from a column


# Columns that the framework writes natively (Trade.to_dict).
CANONICAL_COLUMNS = {
    "trade_id", "symbol", "entry_date", "entry_price", "exit_date", "exit_price",
    "quantity", "side", "initial_stop_loss", "final_stop_loss", "pl", "pl_pct",
}


@dataclass
class LoadedTradeLog:
    """
    Result of loading one or more trade log CSVs.

    Attributes:
        source_files: Paths the data came from (in load order).
        df: Concatenated DataFrame of all loaded rows (with a `_source_file`
            column added for traceability).
        pct_returns: numpy array of % returns as fractions (pl_pct / 100), one
            entry per loaded trade. Always present (NaN for unparseable rows
            are dropped before this is built).
        r_multiples: numpy array of R-multiples (computed when stop loss data
            is present). May be empty if no rows have valid stop data.
        warnings: Human-readable warnings raised during load.
        is_canonical: True if the loaded files all match the framework schema.
    """
    source_files: List[Path]
    df: pd.DataFrame
    pct_returns: np.ndarray
    r_multiples: np.ndarray
    warnings: List[str] = field(default_factory=list)
    is_canonical: bool = False

    # ---- selection --------------------------------------------------------

    def returns_for(self, source: TradeLogReturnSource) -> np.ndarray:
        """Return the numpy array the simulator should sample from.

        Returns are expressed as the fractional gain that should be multiplied
        by the risked capital. For % returns this is `pl_pct / 100` divided by
        an assumed reference risk (we just pass the fraction through and let
        the simulator scale by `risk_per_trade`). For R-multiples the value is
        used directly (a +1R trade → equity gain of `risk_per_trade * equity`).

        See SimulationConfig docstring for the full update equation.
        """
        if source == TradeLogReturnSource.PCT_RETURN:
            return self.pct_returns
        if source == TradeLogReturnSource.R_MULTIPLE:
            return self.r_multiples
        raise ValueError(f"Unknown return source: {source}")

    # ---- summary stats ---------------------------------------------------

    def summary_stats(self, source: TradeLogReturnSource) -> dict:
        """Return summary statistics for the chosen return series."""
        arr = self.returns_for(source)
        if arr.size == 0:
            return {
                "count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "win_rate": 0.0, "skew": 0.0,
            }
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "win_rate": float(np.mean(arr > 0)),
            "skew": _skew(arr),
        }


def _skew(arr: np.ndarray) -> float:
    if arr.size < 3:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _coerce_float_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="float64")
    return _robust_to_numeric(df[col])


# Strings that mean "missing" rather than "malformed".
_NULL_STRINGS = {
    "", "nan", "na", "n/a", "none", "null", "-", "--", "?",
}


def _robust_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a pandas Series of mixed/dirty strings to numeric.

    Handles:
      * already-numeric series (passed through unchanged)
      * leading/trailing whitespace
      * % / $ / GBP / EUR / JPY / commas (thousand separators)
      * parentheses for negatives, e.g. ``(1.23)`` -> ``-1.23``
      * common null-tokens (empty, ``nan``, ``N/A``, ``null``, ``-``)

    Returns:
        A numeric series of float64. Invalid values become NaN; null-tokens
        are also NaN (the caller can distinguish missing vs malformed by
        inspecting the original series).
    """
    # Fast path: already numeric.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    # Normalise to string, strip, lowercase for null detection.
    s = series.astype(str).str.strip()

    # Treat known null tokens as NaN
    null_mask = s.str.lower().isin(_NULL_STRINGS)

    # Strip currency, percent, comma thousands separators
    cleaned = (
        s.str.replace("%", "", regex=False)
         .str.replace("$", "", regex=False)
         .str.replace("£", "", regex=False)   # GBP
         .str.replace("€", "", regex=False)   # EUR
         .str.replace("¥", "", regex=False)   # JPY/CNY
         .str.replace(",", "", regex=False)
         .str.strip()
    )
    # Parentheses for negatives: (1.23) -> -1.23
    paren_mask = cleaned.str.startswith("(") & cleaned.str.endswith(")")
    cleaned = cleaned.where(
        ~paren_mask,
        "-" + cleaned.str.slice(1, -1),
    )

    out = pd.to_numeric(cleaned, errors="coerce")
    out[null_mask] = np.nan
    return out


def _parse_diagnostic(
    series: pd.Series, parsed: pd.Series, source_files: pd.Series
) -> tuple[int, int, list[str]]:
    """Return (missing_count, malformed_count, sample_bad_values).

    Distinguishes between values that were missing in the source (NaN /
    empty / common null-token) and values that look populated but failed
    to parse as numbers.
    """
    if pd.api.types.is_numeric_dtype(series):
        # Numeric column: only "missing" can apply.
        missing = int(parsed.isna().sum())
        return missing, 0, []

    s_str = series.astype(str).str.strip()
    null_mask = s_str.str.lower().isin(_NULL_STRINGS) | series.isna()
    parsed_nan = parsed.isna()
    malformed_mask = parsed_nan & ~null_mask

    missing = int(null_mask.sum())
    malformed = int(malformed_mask.sum())

    samples: list[str] = []
    if malformed:
        bad_values = series[malformed_mask].astype(str).head(5).tolist()
        bad_files = source_files[malformed_mask].head(5).tolist() if source_files is not None else []
        for i, v in enumerate(bad_values):
            file_hint = f" [{bad_files[i]}]" if i < len(bad_files) else ""
            samples.append(f"'{v}'{file_hint}")
    return missing, malformed, samples


def _column_presence_per_file(frames: List[pd.DataFrame], col: str) -> List[Path]:
    """Return list of source file names where the column is missing."""
    missing_files: List[Path] = []
    for f in frames:
        if col not in f.columns:
            # Stash the file name from the _source_file column we always add
            src = f["_source_file"].iloc[0] if "_source_file" in f.columns and len(f) > 0 else "?"
            missing_files.append(src)
    return missing_files


def _compute_r_multiples(df: pd.DataFrame) -> np.ndarray:
    """Compute R-multiples from price + stop loss columns when present.

    R = (exit - entry) for LONG, (entry - exit) for SHORT, divided by initial
    risk per share (entry - stop_loss for LONG; stop_loss - entry for SHORT).
    Rows with missing/invalid stop loss data are dropped.
    """
    needed = {"entry_price", "exit_price", "initial_stop_loss"}
    if not needed.issubset(df.columns):
        return np.array([], dtype="float64")

    entry = _coerce_float_column(df, "entry_price")
    exit_ = _coerce_float_column(df, "exit_price")
    stop = _coerce_float_column(df, "initial_stop_loss")

    if "side" in df.columns:
        side = df["side"].astype(str).str.upper().str.strip()
    else:
        side = pd.Series(["LONG"] * len(df), index=df.index)

    is_long = side != "SHORT"

    initial_risk = np.where(is_long, entry - stop, stop - entry)
    pl_per_unit = np.where(is_long, exit_ - entry, entry - exit_)

    valid = (
        ~entry.isna() & ~exit_.isna() & ~stop.isna()
        & (initial_risk > 0)
    )
    if not valid.any():
        return np.array([], dtype="float64")

    r_mult = pl_per_unit[valid] / initial_risk[valid]
    return np.asarray(r_mult, dtype="float64")


def _load_single(path: Path) -> pd.DataFrame:
    # utf-8-sig strips a BOM if present (which would otherwise hide the
    # first column behind '﻿trade_id' and break column lookups).
    df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    # Normalise column names: strip whitespace (handles trailing tabs etc.).
    df.columns = [str(c).strip() for c in df.columns]
    df["_source_file"] = path.name
    return df


def load_trade_logs(
    paths: Sequence[Union[str, Path]],
    *,
    pct_return_column: Optional[str] = None,
    r_multiple_column: Optional[str] = None,
    min_trades_warning: int = 100,
) -> LoadedTradeLog:
    """Load and concatenate one or more trade log CSV files.

    Args:
        paths: One or more CSV paths.
        pct_return_column: Override for the % return column. If None, the
            loader uses ``pl_pct`` when present.
        r_multiple_column: Override for an existing R-multiple column. If None
            the loader computes R from price + stop columns when available.
        min_trades_warning: Emit a warning if fewer trades were loaded.

    Returns:
        LoadedTradeLog with a concatenated DataFrame and ready-to-sample arrays.

    Raises:
        ValueError: if no rows could be parsed at all.
    """
    if not paths:
        raise ValueError("At least one CSV path is required")

    paths = [Path(p) for p in paths]
    frames = [_load_single(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    src_series = df["_source_file"] if "_source_file" in df.columns else pd.Series([""] * len(df))

    warnings: List[str] = []

    # --- % returns --------------------------------------------------------
    pct_col = pct_return_column or ("pl_pct" if "pl_pct" in df.columns else None)
    if pct_col is None:
        pct_returns = np.array([], dtype="float64")
        warnings.append(
            "No % return column found (looked for 'pl_pct'). Pass "
            "pct_return_column to override."
        )
    else:
        if pct_col not in df.columns:
            raise ValueError(f"Column '{pct_col}' not present in CSV")

        # Flag files that are missing the column entirely - their rows show
        # up as NaN after concat and shouldn't be reported as "malformed".
        missing_in_files = _column_presence_per_file(frames, pct_col)
        if missing_in_files:
            warnings.append(
                f"Column '{pct_col}' is missing in {len(missing_in_files)} "
                f"file(s): {', '.join(map(str, missing_in_files[:3]))}"
                + ("..." if len(missing_in_files) > 3 else "")
            )

        raw = _robust_to_numeric(df[pct_col])
        missing, malformed, samples = _parse_diagnostic(df[pct_col], raw, src_series)

        if malformed:
            sample_str = ", ".join(samples)
            warnings.append(
                f"Dropped {malformed} row(s) with unparseable '{pct_col}' "
                f"values. Examples: {sample_str}"
            )
        if missing:
            warnings.append(
                f"Dropped {missing} row(s) with missing/empty '{pct_col}' values."
            )

        # Convert percent units to fractional units (1.5% -> 0.015).
        pct_returns = raw.dropna().to_numpy(dtype="float64") / 100.0

    # --- R-multiples ------------------------------------------------------
    if r_multiple_column is not None:
        if r_multiple_column not in df.columns:
            raise ValueError(f"Column '{r_multiple_column}' not present in CSV")
        r_raw = _robust_to_numeric(df[r_multiple_column])
        missing_r, malformed_r, samples_r = _parse_diagnostic(
            df[r_multiple_column], r_raw, src_series
        )
        if malformed_r:
            warnings.append(
                f"Dropped {malformed_r} row(s) with unparseable "
                f"'{r_multiple_column}' values. Examples: {', '.join(samples_r)}"
            )
        r_multiples = r_raw.dropna().to_numpy(dtype="float64")
    else:
        r_multiples = _compute_r_multiples(df)
        if r_multiples.size == 0:
            warnings.append(
                "Could not compute R-multiples: missing entry_price, exit_price "
                "or initial_stop_loss columns (or all rows had invalid stops)."
            )

    is_canonical = CANONICAL_COLUMNS.issubset(set(df.columns))

    total_loaded = max(pct_returns.size, r_multiples.size)
    if total_loaded == 0:
        raise ValueError(
            "No usable trade returns could be parsed from the selected files."
        )
    if total_loaded < min_trades_warning:
        warnings.append(
            f"Only {total_loaded} trades loaded; results from a Monte Carlo "
            f"simulation are noisy below {min_trades_warning} trades."
        )

    return LoadedTradeLog(
        source_files=paths,
        df=df,
        pct_returns=pct_returns,
        r_multiples=r_multiples,
        warnings=warnings,
        is_canonical=is_canonical,
    )
