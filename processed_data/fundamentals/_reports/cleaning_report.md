# Fundamentals cleaning report

_Generated 2026-06-18T11:00:57_

- Raw input: `C:\Users\rzemi\OneDrive\Desktop\Finance & Trading\Trading\Back Testing Framework\raw_data\fundamentals`
- Cleaned output: `C:\Users\rzemi\OneDrive\Desktop\Finance & Trading\Trading\Back Testing Framework\processed_data\fundamentals`

This report accompanies the cleaned datasets. Per-column descriptions live in `data_dictionary_*.{csv,md}`; full per-field statistics live in `stats_summary_*.csv`; per-symbol fundamentals coverage lives in `coverage_fundamentals.csv`.

## What the cleaning step does

1. Normalises headers (lower-cased, trimmed) and unifies missing-value tokens (`None`, `-`, `N/A`, ...) to true NA.
2. Enforces one shared schema per dataset and normalises encoding to UTF-8 with LF line endings (source files mixed CRLF/LF).
3. Coerces numeric and date columns, **preserving** any value that fails to parse and reporting it (no silent loss).
4. Drops only columns that are empty across *every* file.
5. Removes exact duplicate rows and sorts fundamentals chronologically by `fiscaldateending`.

## Fundamentals

- **Source files:** 493
- **Total rows:** 47,086
- **Columns in / out:** 97 -> 73
- **Columns dropped (empty in *all* files):** changeinoperatingassets, changeinoperatingliabilities, comprehensiveincomenetoftax, currentdebt, deferredrevenue, depreciation, dividendpayoutpreferredstock, interestanddebtexpense, investmentincomenet, investments, longtermdebtnoncurrent, noninterestincome, othernoncurrentassets, paymentsforoperatingactivities, paymentsforrepurchaseofcommonstock, paymentsforrepurchaseofequity, paymentsforrepurchaseofpreferredstock, proceedsfromissuanceofcommonstock, proceedsfromissuanceoflongtermdebtandcapitalsecuritiesnet, proceedsfromissuanceofpreferredstock, proceedsfromoperatingactivities, proceedsfromrepaymentsofshorttermdebt, proceedsfromsaleoftreasurystock, profitloss
- **Constant columns (single value everywhere):** frequency

### Schema deviations
- None -- all files shared an identical column set (after header normalisation).

### Value-coercion warnings
- None -- every non-null value coerced cleanly to its target type (no information lost).

## Overview

- **Source files:** 503
- **Total rows:** 503
- **Columns in / out:** 56 -> 56
- **Columns dropped (empty in *all* files):** none
- **Constant columns (single value everywhere):** assettype, currency

### Schema deviations
- None -- all files shared an identical column set (after header normalisation).

### Value-coercion warnings
- None -- every non-null value coerced cleanly to its target type (no information lost).

## Cross-dataset coverage

- Symbols with an overview but **no** fundamental file (10): AMP, AWR, BIP, BOKF, CCL, COF, CPK, CSCO, DTE, E
- Symbols with a fundamental file but **no** overview (0): none

## Output written

- Fundamentals: 493 files, 47,086 rows (0 duplicate rows removed).
- Overview: 503 files, 503 rows (0 duplicate rows removed).
