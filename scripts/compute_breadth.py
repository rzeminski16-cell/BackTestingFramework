"""
Compute market breadth (% of the daily universe trading above its 50-day SMA).

Market breadth is a regime indicator: when only a small fraction of stocks are
above their 50-day moving average the broad market is weak. The ShortOnlyBase
strategy uses ``breadth_pct_above_50dma < 40`` as one leg of its regime gate.

The framework feeds a strategy only its own ticker's data, so breadth has to be
pre-computed into a standalone file that the strategy loads at run time (the
same pattern it uses for the benchmark and VIX series in ``raw_data/benchmarks``).

For every trading date this script counts, across all ``raw_data/daily/*.csv``
files, how many securities closed above their pre-calculated ``sma_50_sma`` and
divides by the number of securities that had a valid 50-day SMA on that date.

Output: ``raw_data/benchmarks/BREADTH_daily.csv`` with columns
``date, breadth_pct_above_50dma, count_above, count_total``.

Usage:
    python scripts/compute_breadth.py
    python scripts/compute_breadth.py --daily-dir raw_data/daily --out raw_data/benchmarks/BREADTH_daily.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAILY_DIR = PROJECT_ROOT / "raw_data" / "daily"
DEFAULT_OUT = PROJECT_ROOT / "raw_data" / "benchmarks" / "BREADTH_daily.csv"

CLOSE_COL = "close"
SMA_50_COL = "sma_50_sma"
DATE_COL = "date"


def compute_breadth(daily_dir: Path) -> pd.DataFrame:
    """Aggregate the percentage of the universe above its 50-day SMA per date.

    Args:
        daily_dir: Directory containing ``*_daily.csv`` security files. Each
            file must expose ``date``, ``close`` and ``sma_50_sma`` columns.

    Returns:
        DataFrame indexed 0..N with columns ``date``,
        ``breadth_pct_above_50dma``, ``count_above`` and ``count_total``,
        sorted ascending by date.
    """
    files = sorted(daily_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No daily CSV files found in {daily_dir}")

    # Running per-date tallies: total securities with a valid SMA, and the
    # subset closing above it.
    total = pd.Series(dtype="int64")
    above = pd.Series(dtype="int64")

    used = 0
    for path in files:
        try:
            df = pd.read_csv(path, usecols=[DATE_COL, CLOSE_COL, SMA_50_COL])
        except (ValueError, pd.errors.EmptyDataError):
            # Missing required columns or empty file - skip this security.
            continue

        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=True)
        df = df.dropna(subset=[CLOSE_COL, SMA_50_COL])
        if df.empty:
            continue

        is_above = (df[CLOSE_COL] > df[SMA_50_COL]).astype("int64")
        valid = pd.Series(1, index=df.index, dtype="int64")

        total = total.add(valid.groupby(df[DATE_COL]).sum(), fill_value=0)
        above = above.add(is_above.groupby(df[DATE_COL]).sum(), fill_value=0)
        used += 1

    if total.empty:
        raise ValueError(
            f"None of the {len(files)} daily files exposed the required "
            f"columns {[DATE_COL, CLOSE_COL, SMA_50_COL]}."
        )

    total = total.sort_index().astype("int64")
    above = above.reindex(total.index).fillna(0).astype("int64")
    pct = (above / total * 100.0).round(4)

    result = pd.DataFrame(
        {
            DATE_COL: total.index,
            "breadth_pct_above_50dma": pct.values,
            "count_above": above.values,
            "count_total": total.values,
        }
    ).reset_index(drop=True)

    print(f"Processed {used}/{len(files)} daily files -> {len(result)} dates.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR,
                        help="Directory of *_daily.csv security files.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output CSV path for the breadth series.")
    args = parser.parse_args()

    result = compute_breadth(args.daily_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Wrote breadth series to {args.out}")


if __name__ == "__main__":
    main()
