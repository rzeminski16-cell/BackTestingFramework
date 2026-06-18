#!/usr/bin/env python3
"""
Clean and standardise the raw fundamentals datasets, and emit a data report.

Two datasets live under ``raw_data/fundamentals``:

    *_fundamental.csv           Point-in-time quarterly financial statements
                                (earnings + income statement + balance sheet +
                                cash flow), one row per fiscal quarter.
    _overview/*_overview.csv    A single-row company "overview" snapshot of
                                descriptive fields and valuation ratios.

What this script guarantees
---------------------------
1. **Identical schema within each dataset.** Every cleaned fundamental file has
   the same ordered columns; likewise every cleaned overview file. Source files
   that differed only by line endings (CRLF vs LF) or header casing/whitespace
   are normalised.
2. **No information loss.** Nothing is dropped that contains data. Values are
   read as raw text first, then coerced to numbers/dates with an explicit
   "had-a-value-before / is-null-after" check; any value that would be lost is
   *reported*, never silently discarded. The only columns removed are those that
   are empty across *every* file (see #3). Originals in ``raw_data`` are left
   untouched -- cleaned copies are written to a separate tree.
3. **Globally-empty columns removed.** A column is dropped only if it is null in
   *all* files of its dataset (decided in a first profiling pass).
4. **Extra hygiene.** Whitespace trimmed, missing-value tokens ("None", "-",
   "N/A", ...) unified to NA, exact duplicate rows removed, fundamentals sorted
   chronologically, dates emitted as ISO ``YYYY-MM-DD``, files written UTF-8/LF.

Reports written to ``processed_data/fundamentals/_reports``
-----------------------------------------------------------
    data_dictionary_fundamentals.{csv,md}   Per-column description + coverage.
    data_dictionary_overview.{csv,md}
    stats_summary_fundamentals.csv          Per-column numeric/categorical stats.
    stats_summary_overview.csv
    coverage_fundamentals.csv               Per-symbol row count and date span.
    cleaning_report.md                       Human-readable run summary.

Usage
-----
    python scripts/clean_fundamentals.py
    python scripts/clean_fundamentals.py --raw-dir raw_data/fundamentals \
        --out-dir processed_data/fundamentals --report-only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tokens that mean "missing" in this Alpha Vantage-sourced data. Read as NA on
# load so a literal "None"/"-" never masquerades as real text.
NA_TOKENS = [
    "", "none", "None", "NONE", "nan", "NaN", "NAN",
    "null", "NULL", "n/a", "N/A", "na", "NA", "-", "--", "#n/a",
]

# Columns that must stay textual (never coerced to numbers).
FUND_TEXT_COLS = {"symbol", "frequency", "reporttime", "reportedcurrency"}
FUND_DATE_COLS = {"fiscaldateending", "report_date"}

OVERVIEW_TEXT_COLS = {
    "symbol", "assettype", "name", "description", "exchange", "currency",
    "country", "sector", "industry", "address", "officialsite", "fiscalyearend",
}
OVERVIEW_DATE_COLS = {"as_of_date", "latestquarter", "dividenddate", "exdividenddate"}

# --------------------------------------------------------------------------- #
# Column dictionaries (descriptions for the report)
# --------------------------------------------------------------------------- #
# Keys are lower-cased column names. Each value is (description, unit/category).

FUND_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # --- identifiers / metadata ---
    "symbol": ("Ticker symbol the row belongs to.", "id"),
    "frequency": ("Reporting cadence of the statement (quarterly/annual).", "category"),
    "fiscaldateending": ("Last day of the fiscal period the figures cover.", "date"),
    "report_date": ("Date the earnings/results were publicly reported.", "date"),
    "reporttime": ("Timing of the earnings release (pre-market/post-market).", "category"),
    "reportedcurrency": ("Currency the financial statement is reported in.", "category"),
    # --- earnings ---
    "reported_eps": ("Actual reported earnings per share for the quarter.", "currency/share"),
    "estimated_eps": ("Consensus analyst estimated EPS for the quarter.", "currency/share"),
    "earnings_surprise": ("Reported EPS minus estimated EPS (absolute surprise).", "currency/share"),
    "surprise_pct": ("Earnings surprise as a percentage of the estimate.", "percent"),
    # --- income statement ---
    "grossprofit": ("Revenue less cost of goods/services sold.", "currency"),
    "totalrevenue": ("Total revenue / net sales for the period.", "currency"),
    "costofrevenue": ("Total cost of generating revenue.", "currency"),
    "costofgoodsandservicessold": ("Direct cost of goods and services sold (COGS).", "currency"),
    "operatingincome": ("Profit from core operations (revenue less operating costs).", "currency"),
    "sellinggeneralandadministrative": ("Selling, general & administrative expenses (SG&A).", "currency"),
    "researchanddevelopment": ("Research & development expenditure.", "currency"),
    "operatingexpenses": ("Total operating expenses.", "currency"),
    "investmentincomenet": ("Net income earned from investments.", "currency"),
    "netinterestincome": ("Interest income net of interest expense.", "currency"),
    "interestincome": ("Income earned from interest-bearing assets.", "currency"),
    "interestexpense": ("Expense incurred on debt/interest obligations.", "currency"),
    "noninterestincome": ("Income from sources other than interest (fees, trading).", "currency"),
    "othernonoperatingincome": ("Income/expense outside normal operations.", "currency"),
    "depreciation": ("Depreciation expense for the period.", "currency"),
    "depreciationandamortization": ("Combined depreciation and amortization expense.", "currency"),
    "incomebeforetax": ("Pre-tax income (EBT).", "currency"),
    "incometaxexpense": ("Income tax expense/provision.", "currency"),
    "interestanddebtexpense": ("Combined interest and debt-related expense.", "currency"),
    "netincomefromcontinuingoperations": ("Net income from continuing operations.", "currency"),
    "comprehensiveincomenetoftax": ("Comprehensive income net of tax.", "currency"),
    "ebit": ("Earnings before interest and taxes.", "currency"),
    "ebitda": ("Earnings before interest, taxes, depreciation & amortization.", "currency"),
    "netincome": ("Bottom-line net income (profit) for the period.", "currency"),
    # --- balance sheet: assets ---
    "totalassets": ("Total assets.", "currency"),
    "totalcurrentassets": ("Assets expected to convert to cash within a year.", "currency"),
    "totalnoncurrentassets": ("Long-term (non-current) assets.", "currency"),
    "cashandcashequivalentsatcarryingvalue": ("Cash and cash equivalents at carrying value.", "currency"),
    "cashandshortterminvestments": ("Cash plus short-term investments.", "currency"),
    "shortterminvestments": ("Short-term / marketable investments.", "currency"),
    "longterminvestments": ("Long-term investments.", "currency"),
    "investments": ("Total investments held.", "currency"),
    "currentnetreceivables": ("Net receivables due within a year.", "currency"),
    "inventory": ("Inventory carried on the balance sheet.", "currency"),
    "othercurrentassets": ("Other current assets.", "currency"),
    "othernoncurrentassets": ("Other non-current assets.", "currency"),
    "propertyplantequipment": ("Property, plant & equipment (net PP&E).", "currency"),
    "accumulateddepreciationamortizationppe": ("Accumulated depreciation/amortization on PP&E.", "currency"),
    "goodwill": ("Goodwill from acquisitions.", "currency"),
    "intangibleassets": ("Total intangible assets (incl. goodwill).", "currency"),
    "intangibleassetsexcludinggoodwill": ("Intangible assets excluding goodwill.", "currency"),
    # --- balance sheet: liabilities & equity ---
    "totalliabilities": ("Total liabilities.", "currency"),
    "totalcurrentliabilities": ("Obligations due within a year.", "currency"),
    "totalnoncurrentliabilities": ("Long-term (non-current) liabilities.", "currency"),
    "currentaccountspayable": ("Accounts payable due within a year.", "currency"),
    "deferredrevenue": ("Revenue received but not yet earned.", "currency"),
    "currentdebt": ("Debt due within a year.", "currency"),
    "shorttermdebt": ("Short-term debt.", "currency"),
    "currentlongtermdebt": ("Current portion of long-term debt.", "currency"),
    "longtermdebt": ("Long-term debt.", "currency"),
    "longtermdebtnoncurrent": ("Non-current portion of long-term debt.", "currency"),
    "shortlongtermdebttotal": ("Total short- plus long-term debt.", "currency"),
    "capitalleaseobligations": ("Capital/finance lease obligations.", "currency"),
    "othercurrentliabilities": ("Other current liabilities.", "currency"),
    "othernoncurrentliabilities": ("Other non-current liabilities.", "currency"),
    "commonstock": ("Common stock at par/stated value.", "currency"),
    "commonstocksharesoutstanding": ("Number of common shares outstanding.", "shares"),
    "retainedearnings": ("Cumulative retained earnings.", "currency"),
    "treasurystock": ("Treasury stock (repurchased shares).", "currency"),
    "totalshareholderequity": ("Total shareholders' equity.", "currency"),
    # --- cash flow ---
    "operatingcashflow": ("Net cash generated by operating activities.", "currency"),
    "cashflowfrominvestment": ("Net cash from investing activities.", "currency"),
    "cashflowfromfinancing": ("Net cash from financing activities.", "currency"),
    "capitalexpenditures": ("Capital expenditures (capex).", "currency"),
    "changeincashandcashequivalents": ("Net change in cash and equivalents.", "currency"),
    "changeinexchangerate": ("FX effect on cash balances.", "currency"),
    "changeininventory": ("Change in inventory (cash flow adjustment).", "currency"),
    "changeinoperatingassets": ("Change in operating assets.", "currency"),
    "changeinoperatingliabilities": ("Change in operating liabilities.", "currency"),
    "changeinreceivables": ("Change in receivables.", "currency"),
    "depreciationdepletionandamortization": ("Depreciation, depletion & amortization (cash flow).", "currency"),
    "stockbasedcompensation": ("Stock-based compensation expense.", "currency"),
    "profitloss": ("Profit/loss figure used in the cash flow statement.", "currency"),
    "dividendpayout": ("Total dividends paid.", "currency"),
    "dividendpayoutcommonstock": ("Dividends paid on common stock.", "currency"),
    "dividendpayoutpreferredstock": ("Dividends paid on preferred stock.", "currency"),
    "paymentsforoperatingactivities": ("Cash payments for operating activities.", "currency"),
    "proceedsfromoperatingactivities": ("Cash proceeds from operating activities.", "currency"),
    "paymentsforrepurchaseofcommonstock": ("Cash paid to repurchase common stock.", "currency"),
    "paymentsforrepurchaseofequity": ("Cash paid to repurchase equity.", "currency"),
    "paymentsforrepurchaseofpreferredstock": ("Cash paid to repurchase preferred stock.", "currency"),
    "proceedsfromrepurchaseofequity": ("Net proceeds/payments from equity repurchase.", "currency"),
    "proceedsfromsaleoftreasurystock": ("Proceeds from sale of treasury stock.", "currency"),
    "proceedsfromissuanceofcommonstock": ("Proceeds from issuing common stock.", "currency"),
    "proceedsfromissuanceofpreferredstock": ("Proceeds from issuing preferred stock.", "currency"),
    "proceedsfromissuanceoflongtermdebtandcapitalsecuritiesnet": ("Net proceeds from issuing long-term debt/capital securities.", "currency"),
    "proceedsfromrepaymentsofshorttermdebt": ("Net proceeds/repayments of short-term debt.", "currency"),
}

OVERVIEW_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "symbol": ("Ticker symbol.", "id"),
    "as_of_date": ("Date the overview snapshot was captured.", "date"),
    "assettype": ("Security type (e.g. Common Stock, ETF).", "category"),
    "name": ("Company name.", "text"),
    "description": ("Company business description.", "text"),
    "cik": ("SEC Central Index Key identifier.", "id"),
    "exchange": ("Listing exchange.", "category"),
    "currency": ("Trading/reporting currency.", "category"),
    "country": ("Country of domicile.", "category"),
    "sector": ("GICS-style sector.", "category"),
    "industry": ("Industry classification.", "category"),
    "address": ("Company headquarters address.", "text"),
    "officialsite": ("Company website URL.", "text"),
    "fiscalyearend": ("Month the fiscal year ends.", "category"),
    "latestquarter": ("Fiscal date of the most recent reported quarter.", "date"),
    "marketcapitalization": ("Market capitalisation.", "currency"),
    "ebitda": ("Trailing EBITDA.", "currency"),
    "peratio": ("Price-to-earnings ratio.", "ratio"),
    "pegratio": ("Price/earnings-to-growth ratio.", "ratio"),
    "bookvalue": ("Book value per share.", "currency/share"),
    "dividendpershare": ("Dividend per share.", "currency/share"),
    "dividendyield": ("Dividend yield (fraction).", "fraction"),
    "eps": ("Earnings per share.", "currency/share"),
    "revenuepersharettm": ("Trailing-twelve-month revenue per share.", "currency/share"),
    "profitmargin": ("Net profit margin (fraction).", "fraction"),
    "operatingmarginttm": ("Trailing-twelve-month operating margin (fraction).", "fraction"),
    "returnonassetsttm": ("Trailing-twelve-month return on assets (fraction).", "fraction"),
    "returnonequityttm": ("Trailing-twelve-month return on equity (fraction).", "fraction"),
    "revenuettm": ("Trailing-twelve-month revenue.", "currency"),
    "grossprofitttm": ("Trailing-twelve-month gross profit.", "currency"),
    "dilutedepsttm": ("Trailing-twelve-month diluted EPS.", "currency/share"),
    "quarterlyearningsgrowthyoy": ("Year-over-year quarterly earnings growth (fraction).", "fraction"),
    "quarterlyrevenuegrowthyoy": ("Year-over-year quarterly revenue growth (fraction).", "fraction"),
    "analysttargetprice": ("Mean analyst target price.", "currency"),
    "analystratingstrongbuy": ("Count of 'strong buy' analyst ratings.", "count"),
    "analystratingbuy": ("Count of 'buy' analyst ratings.", "count"),
    "analystratinghold": ("Count of 'hold' analyst ratings.", "count"),
    "analystratingsell": ("Count of 'sell' analyst ratings.", "count"),
    "analystratingstrongsell": ("Count of 'strong sell' analyst ratings.", "count"),
    "trailingpe": ("Trailing price-to-earnings ratio.", "ratio"),
    "forwardpe": ("Forward price-to-earnings ratio.", "ratio"),
    "pricetosalesratiottm": ("Trailing-twelve-month price-to-sales ratio.", "ratio"),
    "pricetobookratio": ("Price-to-book ratio.", "ratio"),
    "evtorevenue": ("Enterprise value to revenue.", "ratio"),
    "evtoebitda": ("Enterprise value to EBITDA.", "ratio"),
    "beta": ("Beta versus the market.", "ratio"),
    "52weekhigh": ("52-week high price.", "currency"),
    "52weeklow": ("52-week low price.", "currency"),
    "50daymovingaverage": ("50-day moving average price.", "currency"),
    "200daymovingaverage": ("200-day moving average price.", "currency"),
    "sharesoutstanding": ("Shares outstanding.", "shares"),
    "sharesfloat": ("Public float (shares available for trading).", "shares"),
    "percentinsiders": ("Percent of shares held by insiders.", "percent"),
    "percentinstitutions": ("Percent of shares held by institutions.", "percent"),
    "dividenddate": ("Next/most-recent dividend payment date.", "date"),
    "exdividenddate": ("Ex-dividend date.", "date"),
}


# --------------------------------------------------------------------------- #
# Loading & coercion helpers
# --------------------------------------------------------------------------- #

def discover_files(raw_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (fundamental_files, overview_files), sorted by symbol."""
    fund = sorted(raw_dir.glob("*_fundamental.csv"))
    over = sorted((raw_dir / "_overview").glob("*_overview.csv"))
    return fund, over


def symbol_from_path(path: Path) -> str:
    return path.stem.replace("_fundamental", "").replace("_overview", "")


def load_dataset(files: list[Path], suffix: str) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Read every file as text, normalise headers, and stack into one frame.

    Returns (combined_df, canonical_columns, schema_issues). The combined frame
    carries two helper columns ``__symbol`` and ``__source`` used for grouping
    and reporting; they are stripped before any file is written.
    """
    frames: list[pd.DataFrame] = []
    canonical: list[str] | None = None
    schema_issues: list[dict] = []

    for fp in files:
        # dtype=str + our NA tokens => we control every coercion ourselves.
        df = pd.read_csv(
            fp, dtype=str, na_values=NA_TOKENS, keep_default_na=True,
            skipinitialspace=True, encoding="utf-8-sig",
        )
        df.columns = [c.strip().lower() for c in df.columns]
        # Everything was read as text; trim cells and re-apply the NA tokens so
        # that whitespace-padded sentinels (e.g. " None ") also become NA.
        na_set = set(NA_TOKENS)
        for c in df.columns:
            col = df[c].str.strip()
            df[c] = col.mask(col.isin(na_set))

        cols = list(df.columns)
        if canonical is None:
            canonical = cols
        elif cols != canonical:
            missing = [c for c in canonical if c not in cols]
            extra = [c for c in cols if c not in canonical]
            reordered = set(cols) == set(canonical) and cols != canonical
            schema_issues.append({
                "file": fp.name, "missing_columns": missing,
                "extra_columns": extra, "reordered": reordered,
            })

        df["__symbol"] = symbol_from_path(fp)
        df["__source"] = fp.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, (canonical or []), schema_issues


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def coerce_datetime(s: pd.Series) -> pd.Series:
    """Strict ISO parse, then a flexible fallback for anything that failed."""
    out = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    failed = s.notna() & out.isna()
    if failed.any():
        out.loc[failed] = pd.to_datetime(s[failed], errors="coerce")
    return out


def coerce_dataset(
    df: pd.DataFrame, text_cols: set[str], date_cols: set[str]
) -> tuple[pd.DataFrame, list[dict]]:
    """Coerce columns to their target dtype, recording any value that is lost.

    Returns (coerced_df, coercion_warnings). A warning means a non-null source
    value failed to parse to the target type -- surfaced, never silently nulled.
    """
    out = df.copy()
    warnings: list[dict] = []
    helper = {"__symbol", "__source"}

    for col in df.columns:
        if col in helper or col in text_cols:
            continue
        before = df[col]
        if col in date_cols:
            after = coerce_datetime(before)
            kind = "date"
        else:
            after = coerce_numeric(before)
            kind = "numeric"
        lost_mask = before.notna() & after.isna()
        if lost_mask.any():
            samples = before[lost_mask].dropna().unique()[:5].tolist()
            warnings.append({
                "column": col, "target_type": kind,
                "values_lost": int(lost_mask.sum()),
                "sample_unparseable": samples,
            })
            # Preserve the original text rather than destroy data.
            after = after.astype(object)
            after[lost_mask] = before[lost_mask]
        out[col] = after

    return out, warnings


# --------------------------------------------------------------------------- #
# Profiling, dropping, writing
# --------------------------------------------------------------------------- #

def find_empty_columns(df: pd.DataFrame, canonical: list[str]) -> list[str]:
    """Columns null across *every* row of the whole dataset."""
    return [c for c in canonical if c in df.columns and df[c].notna().sum() == 0]


def constant_columns(df: pd.DataFrame, canonical: list[str], dropped: set[str]) -> list[str]:
    """Non-empty columns that carry a single distinct value everywhere."""
    const = []
    for c in canonical:
        if c in dropped or c not in df.columns:
            continue
        nun = df[c].nunique(dropna=True)
        if nun == 1:
            const.append(c)
    return const


def write_clean_files(
    df: pd.DataFrame, out_cols: list[str], out_dir: Path, suffix: str,
    sort_col: str | None,
) -> dict:
    """Write one cleaned CSV per symbol with the shared schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written, dupes_removed, rows_out = 0, 0, 0

    for symbol, g in df.groupby("__symbol", sort=True):
        sub = g[out_cols].copy()
        n_before = len(sub)
        sub = sub.drop_duplicates()
        dupes_removed += n_before - len(sub)
        if sort_col and sort_col in sub.columns:
            sub = sub.sort_values(sort_col, kind="stable", na_position="last")
        sub = sub.reset_index(drop=True)

        out_path = out_dir / f"{symbol}_{suffix}.csv"
        sub.to_csv(out_path, index=False, encoding="utf-8",
                   lineterminator="\n", date_format="%Y-%m-%d")
        written += 1
        rows_out += len(sub)

    return {"files_written": written, "duplicate_rows_removed": dupes_removed,
            "rows_written": rows_out}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_data_dictionary(
    df: pd.DataFrame, canonical: list[str], dropped: set[str],
    constants: set[str], descriptions: dict[str, tuple[str, str]],
    text_cols: set[str], date_cols: set[str],
) -> pd.DataFrame:
    rows = []
    n_files = df["__source"].nunique()
    for col in canonical:
        if col not in df.columns:
            continue
        ser = df[col]
        nonnull = int(ser.notna().sum())
        total = int(len(ser))
        files_present = int(df.loc[ser.notna(), "__source"].nunique())
        example = ser.dropna().iloc[0] if nonnull else ""
        if isinstance(example, float) and example.is_integer():
            example = int(example)
        elif isinstance(example, pd.Timestamp):
            example = example.date().isoformat()
        desc, unit = descriptions.get(col, ("(undocumented column)", "unknown"))
        if col in text_cols:
            dtype = "string"
        elif col in date_cols:
            dtype = "date"
        else:
            dtype = "numeric"
        status = ("dropped_all_null" if col in dropped
                  else "constant" if col in constants else "kept")
        rows.append({
            "column": col,
            "dtype": dtype,
            "unit_category": unit,
            "description": desc,
            "status": status,
            "non_null": nonnull,
            "total": total,
            "pct_populated": round(100 * nonnull / total, 2) if total else 0.0,
            "files_with_data": files_present,
            "pct_files_with_data": round(100 * files_present / n_files, 2) if n_files else 0.0,
            "example_value": str(example)[:80],
            "documented": col in descriptions,
        })
    return pd.DataFrame(rows)


def build_stats_summary(
    df: pd.DataFrame, canonical: list[str], dropped: set[str],
    text_cols: set[str], date_cols: set[str],
) -> pd.DataFrame:
    rows = []
    for col in canonical:
        if col in dropped or col not in df.columns:
            continue
        ser = df[col]
        nonnull = ser.dropna()
        base = {
            "column": col,
            "count": int(ser.notna().sum()),
            "n_missing": int(ser.isna().sum()),
            "pct_missing": round(100 * ser.isna().sum() / len(ser), 2) if len(ser) else 0.0,
            "n_unique": int(nonnull.nunique()),
        }
        if col in text_cols:
            top = nonnull.mode()
            base["kind"] = "categorical"
            base["top"] = str(top.iloc[0])[:60] if not top.empty else ""
            base["top_freq"] = int((nonnull == top.iloc[0]).sum()) if not top.empty else 0
        elif col in date_cols:
            dt = pd.to_datetime(nonnull, errors="coerce")
            base["kind"] = "date"
            base["min"] = dt.min().date().isoformat() if dt.notna().any() else ""
            base["max"] = dt.max().date().isoformat() if dt.notna().any() else ""
        else:
            num = pd.to_numeric(nonnull, errors="coerce").dropna()
            base["kind"] = "numeric"
            if not num.empty:
                base.update({
                    "mean": num.mean(), "std": num.std(),
                    "min": num.min(), "p25": num.quantile(0.25),
                    "median": num.median(), "p75": num.quantile(0.75),
                    "max": num.max(),
                    "n_zero": int((num == 0).sum()),
                    "n_negative": int((num < 0).sum()),
                })
        rows.append(base)
    return pd.DataFrame(rows)


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol fundamentals coverage: rows and fiscal-date span."""
    g = df.groupby("__symbol")
    cov = pd.DataFrame({
        "symbol": g.size().index,
        "n_rows": g.size().values,
    })
    if "fiscaldateending" in df.columns:
        dates = pd.to_datetime(df["fiscaldateending"], errors="coerce")
        tmp = df.assign(_d=dates).groupby("__symbol")["_d"]
        cov["first_fiscal_date"] = tmp.min().dt.date.astype(str).values
        cov["last_fiscal_date"] = tmp.max().dt.date.astype(str).values
    return cov.sort_values("symbol").reset_index(drop=True)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([head, sep, *body])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def process(args) -> int:
    raw_dir = (REPO_ROOT / args.raw_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    report_dir = out_dir / "_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    fund_files, over_files = discover_files(raw_dir)
    if not fund_files and not over_files:
        print(f"No CSVs found under {raw_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(fund_files)} fundamental and {len(over_files)} overview files.")

    summary: dict = {"generated": datetime.now().isoformat(timespec="seconds"),
                     "raw_dir": str(raw_dir), "out_dir": str(out_dir)}

    # ---- Fundamentals -----------------------------------------------------
    fund_raw, fund_cols, fund_schema_issues = load_dataset(fund_files, "fundamental")
    fund, fund_warnings = coerce_dataset(fund_raw, FUND_TEXT_COLS, FUND_DATE_COLS)
    fund_dropped = find_empty_columns(fund, fund_cols)
    fund_const = constant_columns(fund, fund_cols, set(fund_dropped))
    fund_out_cols = [c for c in fund_cols if c not in fund_dropped]

    fund_dict = build_data_dictionary(
        fund, fund_cols, set(fund_dropped), set(fund_const),
        FUND_DESCRIPTIONS, FUND_TEXT_COLS, FUND_DATE_COLS)
    fund_stats = build_stats_summary(fund, fund_cols, set(fund_dropped),
                                     FUND_TEXT_COLS, FUND_DATE_COLS)
    fund_cov = build_coverage(fund)

    # ---- Overview ---------------------------------------------------------
    over_raw, over_cols, over_schema_issues = load_dataset(over_files, "overview")
    over, over_warnings = coerce_dataset(over_raw, OVERVIEW_TEXT_COLS, OVERVIEW_DATE_COLS)
    over_dropped = find_empty_columns(over, over_cols)
    over_const = constant_columns(over, over_cols, set(over_dropped))
    over_out_cols = [c for c in over_cols if c not in over_dropped]

    over_dict = build_data_dictionary(
        over, over_cols, set(over_dropped), set(over_const),
        OVERVIEW_DESCRIPTIONS, OVERVIEW_TEXT_COLS, OVERVIEW_DATE_COLS)
    over_stats = build_stats_summary(over, over_cols, set(over_dropped),
                                     OVERVIEW_TEXT_COLS, OVERVIEW_DATE_COLS)

    # ---- Cross-dataset checks --------------------------------------------
    fund_syms = {symbol_from_path(p) for p in fund_files}
    over_syms = {symbol_from_path(p) for p in over_files}
    over_only = sorted(over_syms - fund_syms)
    fund_only = sorted(fund_syms - over_syms)

    # ---- Write cleaned data ----------------------------------------------
    if not args.report_only:
        fr = write_clean_files(fund, fund_out_cols, out_dir, "fundamental",
                               sort_col="fiscaldateending")
        ov = write_clean_files(over, over_out_cols, out_dir / "_overview",
                               "overview", sort_col="as_of_date")
        summary["fundamentals_write"] = fr
        summary["overview_write"] = ov
        print(f"Wrote {fr['files_written']} fundamental files "
              f"({fr['rows_written']} rows, {fr['duplicate_rows_removed']} dupes removed).")
        print(f"Wrote {ov['files_written']} overview files "
              f"({ov['rows_written']} rows, {ov['duplicate_rows_removed']} dupes removed).")
    else:
        print("Report-only mode: no cleaned files written.")

    # ---- Persist reports --------------------------------------------------
    fund_dict.to_csv(report_dir / "data_dictionary_fundamentals.csv", index=False)
    over_dict.to_csv(report_dir / "data_dictionary_overview.csv", index=False)
    fund_stats.to_csv(report_dir / "stats_summary_fundamentals.csv", index=False)
    over_stats.to_csv(report_dir / "stats_summary_overview.csv", index=False)
    fund_cov.to_csv(report_dir / "coverage_fundamentals.csv", index=False)

    (report_dir / "data_dictionary_fundamentals.md").write_text(
        "# Fundamentals data dictionary\n\n" + df_to_markdown(fund_dict) + "\n",
        encoding="utf-8")
    (report_dir / "data_dictionary_overview.md").write_text(
        "# Overview data dictionary\n\n" + df_to_markdown(over_dict) + "\n",
        encoding="utf-8")

    write_main_report(
        report_dir, summary,
        fund=(fund, fund_files, fund_out_cols, fund_dropped, fund_const,
              fund_warnings, fund_schema_issues),
        over=(over, over_files, over_out_cols, over_dropped, over_const,
              over_warnings, over_schema_issues),
        over_only=over_only, fund_only=fund_only,
    )

    print(f"Reports written to {report_dir}")
    return 0


def write_main_report(report_dir, summary, fund, over, over_only, fund_only):
    (fdf, ffiles, fout, fdrop, fconst, fwarn, fissues) = fund
    (odf, ofiles, oout, odrop, oconst, owarn, oissues) = over

    def section(title, files, df, out_cols, all_cols, dropped, const, warnings, issues):
        lines = [f"## {title}", ""]
        lines += [
            f"- **Source files:** {len(files)}",
            f"- **Total rows:** {len(df):,}",
            f"- **Columns in / out:** {len(all_cols)} -> {len(out_cols)}",
            f"- **Columns dropped (empty in *all* files):** "
            f"{', '.join(dropped) if dropped else 'none'}",
            f"- **Constant columns (single value everywhere):** "
            f"{', '.join(const) if const else 'none'}",
            "",
        ]
        if issues:
            lines.append(f"### Schema deviations ({len(issues)} file(s))")
            for it in issues[:20]:
                lines.append(f"- `{it['file']}`: missing={it['missing_columns']} "
                             f"extra={it['extra_columns']} reordered={it['reordered']}")
            lines.append("")
        else:
            lines += ["### Schema deviations", "- None -- all files shared an "
                      "identical column set (after header normalisation).", ""]
        if warnings:
            lines.append("### Value-coercion warnings (preserved, not dropped)")
            for w in warnings:
                lines.append(f"- `{w['column']}` -> {w['target_type']}: "
                             f"{w['values_lost']} value(s) could not be parsed; "
                             f"kept as text. Samples: {w['sample_unparseable']}")
            lines.append("")
        else:
            lines += ["### Value-coercion warnings",
                      "- None -- every non-null value coerced cleanly to its "
                      "target type (no information lost).", ""]
        return "\n".join(lines)

    parts = [
        "# Fundamentals cleaning report",
        "",
        f"_Generated {summary['generated']}_",
        "",
        f"- Raw input: `{summary['raw_dir']}`",
        f"- Cleaned output: `{summary['out_dir']}`",
        "",
        "This report accompanies the cleaned datasets. Per-column descriptions "
        "live in `data_dictionary_*.{csv,md}`; full per-field statistics live in "
        "`stats_summary_*.csv`; per-symbol fundamentals coverage lives in "
        "`coverage_fundamentals.csv`.",
        "",
        "## What the cleaning step does",
        "",
        "1. Normalises headers (lower-cased, trimmed) and unifies missing-value "
        "tokens (`None`, `-`, `N/A`, ...) to true NA.",
        "2. Enforces one shared schema per dataset and normalises encoding to "
        "UTF-8 with LF line endings (source files mixed CRLF/LF).",
        "3. Coerces numeric and date columns, **preserving** any value that "
        "fails to parse and reporting it (no silent loss).",
        "4. Drops only columns that are empty across *every* file.",
        "5. Removes exact duplicate rows and sorts fundamentals chronologically "
        "by `fiscaldateending`.",
        "",
        section("Fundamentals", ffiles, fdf, fout,
                [c for c in fdf.columns if not c.startswith("__")],
                fdrop, fconst, fwarn, fissues),
        section("Overview", ofiles, odf, oout,
                [c for c in odf.columns if not c.startswith("__")],
                odrop, oconst, owarn, oissues),
        "## Cross-dataset coverage",
        "",
        f"- Symbols with an overview but **no** fundamental file ({len(over_only)}): "
        f"{', '.join(over_only) if over_only else 'none'}",
        f"- Symbols with a fundamental file but **no** overview ({len(fund_only)}): "
        f"{', '.join(fund_only) if fund_only else 'none'}",
        "",
    ]
    if "fundamentals_write" in summary:
        fw, ow = summary["fundamentals_write"], summary["overview_write"]
        parts += [
            "## Output written",
            "",
            f"- Fundamentals: {fw['files_written']} files, {fw['rows_written']:,} rows "
            f"({fw['duplicate_rows_removed']} duplicate rows removed).",
            f"- Overview: {ow['files_written']} files, {ow['rows_written']:,} rows "
            f"({ow['duplicate_rows_removed']} duplicate rows removed).",
            "",
        ]
    (report_dir / "cleaning_report.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="raw_data/fundamentals",
                   help="Input directory (relative to repo root).")
    p.add_argument("--out-dir", default="processed_data/fundamentals",
                   help="Output directory for cleaned files and reports.")
    p.add_argument("--report-only", action="store_true",
                   help="Profile and write reports without writing cleaned CSVs.")
    return process(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
