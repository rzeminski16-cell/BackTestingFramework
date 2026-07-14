"""
Parquet-first data store: typed, validated conversion of raw CSVs.

CSV is the collection format; Parquet is the *consumption* format — typed
columns, no repeated date parsing, ~5-10x faster loads and smaller files.
``ingest_directory`` converts a directory of price CSVs into sibling
``.parquet`` files (schema-validated), and :class:`~.data_loader.DataLoader`
transparently prefers a Parquet file when one exists next to the CSV.

Run it via the CLI:

    python -m btf ingest --data-dir raw_data/daily
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Outcome of one directory ingest."""
    written: List[Path] = field(default_factory=list)
    skipped_up_to_date: List[Path] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)      # "file: reason"
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def _normalise_frame(df: pd.DataFrame, name: str,
                     warnings: List[str]) -> pd.DataFrame:
    """Apply the same normalisation DataLoader performs, plus typing."""
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    if "date" not in df.columns:
        raise ValueError("no 'date' (or 'time') column")
    if "close" not in df.columns:
        raise ValueError("no 'close' column")

    from .data_loader import parse_date_column
    try:
        df["date"] = parse_date_column(df["date"])
    except (ValueError, TypeError):
        df["date"] = pd.to_datetime(df["date"], format="mixed",
                                    dayfirst=True, errors="coerce")
    n_bad_dates = int(df["date"].isna().sum())
    if n_bad_dates:
        warnings.append(f"{name}: dropped {n_bad_dates} rows with "
                        f"unparseable dates")
        df = df.dropna(subset=["date"])

    n_dupes = int(df["date"].duplicated().sum())
    if n_dupes:
        warnings.append(f"{name}: collapsed {n_dupes} duplicate dates "
                        f"(keeping last)")
        df = df.drop_duplicates("date", keep="last")

    df = df.sort_values("date").reset_index(drop=True)

    # Type every non-date column numerically where possible; genuinely
    # textual columns (e.g. a symbol column) stay as-is.
    for col in df.columns:
        if col == "date" or pd.api.types.is_numeric_dtype(df[col]):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() >= max(1, int(0.9 * df[col].notna().sum())):
            df[col] = coerced

    close = pd.to_numeric(df["close"], errors="coerce")
    n_bad_close = int((close.isna() | (close <= 0)).sum())
    if n_bad_close:
        warnings.append(f"{name}: dropped {n_bad_close} rows with "
                        f"missing/non-positive close")
        df = df[close.notna() & (close > 0)].reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("fewer than 2 valid rows after validation")

    return df


def ingest_file(csv_path: Path, parquet_path: Optional[Path] = None,
                force: bool = False) -> IngestResult:
    """Convert one CSV to a validated Parquet file next to it."""
    result = IngestResult()
    csv_path = Path(csv_path)
    parquet_path = parquet_path or csv_path.with_suffix(".parquet")

    if (not force and parquet_path.exists()
            and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime):
        result.skipped_up_to_date.append(parquet_path)
        return result

    try:
        df = pd.read_csv(csv_path)
        df = _normalise_frame(df, csv_path.name, result.warnings)
        df.to_parquet(parquet_path, index=False)
        result.written.append(parquet_path)
    except Exception as exc:
        result.failed.append(f"{csv_path.name}: {exc}")
    return result


def ingest_directory(data_dir: Path, force: bool = False) -> IngestResult:
    """
    Convert every ``*.csv`` in ``data_dir`` to a sibling ``.parquet``.

    Files whose Parquet is already newer than the CSV are skipped unless
    ``force``. Returns an aggregate IngestResult; failures don't stop the
    rest of the directory.
    """
    data_dir = Path(data_dir)
    total = IngestResult()
    for csv_path in sorted(data_dir.glob("*.csv")):
        one = ingest_file(csv_path, force=force)
        total.written.extend(one.written)
        total.skipped_up_to_date.extend(one.skipped_up_to_date)
        total.failed.extend(one.failed)
        total.warnings.extend(one.warnings)
    return total
