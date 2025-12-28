"""
Compare trade logs from Framework vs TradingView and generate Excel report.

This script:
1. Reads both trade log CSV files
2. Finds exact matches (same entry date)
3. Finds inferred matches (entry dates within N days)
4. Calculates differences in entry/exit prices and P/L
5. Generates comprehensive Excel report

Usage:
    python tools/compare_trade_logs.py --framework logs/AlphaTrendStrategy_test_1/AlphaTrendStrategy_test_1_AAPL_trades.csv --tradingview logs/AlphaTrendStrategy_test_1/AlphaTrendStrategy_test_1_AAPL_trades_Trading_view.csv
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys


def load_trade_log(filepath: str, source: str) -> pd.DataFrame:
    """Load and standardize trade log CSV."""
    print(f"Loading {source} trades from: {filepath}")

    df = pd.read_csv(filepath)

    # Standardize column names (handle different formats)
    # Add source column
    df['source'] = source

    # Convert date columns to datetime
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'entry' in col.lower() or 'exit' in col.lower()]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    print(f"  ✓ Loaded {len(df)} trades")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    return df


def find_exact_matches(df_framework: pd.DataFrame, df_tv: pd.DataFrame,
                       entry_col_fw: str, entry_col_tv: str) -> pd.DataFrame:
    """Find trades with exact matching entry dates."""
    print("\nFinding exact matches (same entry date)...")

    matches = []
    matched_tv_indices = set()

    for idx_fw, row_fw in df_framework.iterrows():
        entry_date_fw = row_fw[entry_col_fw]

        if pd.isna(entry_date_fw):
            continue

        # Find exact match in TradingView data
        for idx_tv, row_tv in df_tv.iterrows():
            if idx_tv in matched_tv_indices:
                continue

            entry_date_tv = row_tv[entry_col_tv]

            if pd.isna(entry_date_tv):
                continue

            # Check if dates match (same day)
            if entry_date_fw.date() == entry_date_tv.date():
                matches.append({
                    'match_type': 'EXACT',
                    'fw_index': idx_fw,
                    'tv_index': idx_tv,
                    'entry_date_fw': entry_date_fw,
                    'entry_date_tv': entry_date_tv,
                    'date_diff_days': 0,
                    'fw_trade': row_fw.to_dict(),
                    'tv_trade': row_tv.to_dict()
                })
                matched_tv_indices.add(idx_tv)
                break

    print(f"  ✓ Found {len(matches)} exact matches")
    return pd.DataFrame(matches), matched_tv_indices


def find_fuzzy_matches(df_framework: pd.DataFrame, df_tv: pd.DataFrame,
                       entry_col_fw: str, entry_col_tv: str,
                       matched_tv_indices: set, max_days: int = 5) -> pd.DataFrame:
    """Find trades with entry dates within max_days of each other."""
    print(f"\nFinding fuzzy matches (entry dates within {max_days} days)...")

    matches = []
    additional_matched_tv = set()

    for idx_fw, row_fw in df_framework.iterrows():
        entry_date_fw = row_fw[entry_col_fw]

        if pd.isna(entry_date_fw):
            continue

        best_match = None
        best_diff = max_days + 1

        # Find closest match in TradingView data
        for idx_tv, row_tv in df_tv.iterrows():
            if idx_tv in matched_tv_indices or idx_tv in additional_matched_tv:
                continue

            entry_date_tv = row_tv[entry_col_tv]

            if pd.isna(entry_date_tv):
                continue

            # Calculate date difference
            date_diff = abs((entry_date_fw.date() - entry_date_tv.date()).days)

            if date_diff <= max_days and date_diff < best_diff:
                best_diff = date_diff
                best_match = (idx_tv, row_tv, entry_date_tv)

        if best_match:
            idx_tv, row_tv, entry_date_tv = best_match
            matches.append({
                'match_type': 'FUZZY',
                'fw_index': idx_fw,
                'tv_index': idx_tv,
                'entry_date_fw': entry_date_fw,
                'entry_date_tv': entry_date_tv,
                'date_diff_days': best_diff,
                'fw_trade': row_fw.to_dict(),
                'tv_trade': row_tv.to_dict()
            })
            additional_matched_tv.add(idx_tv)

    print(f"  ✓ Found {len(matches)} fuzzy matches")
    return pd.DataFrame(matches), additional_matched_tv


def create_comparison_report(matches_df: pd.DataFrame,
                            df_framework: pd.DataFrame,
                            df_tv: pd.DataFrame,
                            matched_tv_indices: set,
                            output_file: str):
    """Create comprehensive Excel report with multiple sheets."""
    print(f"\nGenerating Excel report: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # Sheet 1: Summary Statistics
        summary_data = {
            'Metric': [
                'Total Framework Trades',
                'Total TradingView Trades',
                'Exact Matches',
                'Fuzzy Matches (within 5 days)',
                'Total Matched',
                'Unmatched Framework Trades',
                'Unmatched TradingView Trades'
            ],
            'Count': [
                len(df_framework),
                len(df_tv),
                len(matches_df[matches_df['match_type'] == 'EXACT']) if len(matches_df) > 0 else 0,
                len(matches_df[matches_df['match_type'] == 'FUZZY']) if len(matches_df) > 0 else 0,
                len(matches_df),
                len(df_framework) - len(matches_df),
                len(df_tv) - len(matched_tv_indices)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print("  ✓ Sheet 1: Summary")

        # Sheet 2: Exact Matches with Differences
        if len(matches_df) > 0 and (matches_df['match_type'] == 'EXACT').any():
            exact_matches = matches_df[matches_df['match_type'] == 'EXACT'].copy()
            exact_comparison = create_detailed_comparison(exact_matches)
            exact_comparison.to_excel(writer, sheet_name='Exact Matches', index=False)
            print(f"  ✓ Sheet 2: Exact Matches ({len(exact_comparison)} trades)")

        # Sheet 3: Fuzzy Matches with Differences
        if len(matches_df) > 0 and (matches_df['match_type'] == 'FUZZY').any():
            fuzzy_matches = matches_df[matches_df['match_type'] == 'FUZZY'].copy()
            fuzzy_comparison = create_detailed_comparison(fuzzy_matches)
            fuzzy_comparison.to_excel(writer, sheet_name='Fuzzy Matches', index=False)
            print(f"  ✓ Sheet 3: Fuzzy Matches ({len(fuzzy_comparison)} trades)")

        # Sheet 4: Unmatched Framework Trades
        matched_fw_indices = set(matches_df['fw_index'].tolist()) if len(matches_df) > 0 else set()
        unmatched_fw = df_framework[~df_framework.index.isin(matched_fw_indices)].copy()
        if len(unmatched_fw) > 0:
            unmatched_fw.to_excel(writer, sheet_name='Unmatched Framework', index=False)
            print(f"  ✓ Sheet 4: Unmatched Framework ({len(unmatched_fw)} trades)")

        # Sheet 5: Unmatched TradingView Trades
        unmatched_tv = df_tv[~df_tv.index.isin(matched_tv_indices)].copy()
        if len(unmatched_tv) > 0:
            unmatched_tv.to_excel(writer, sheet_name='Unmatched TradingView', index=False)
            print(f"  ✓ Sheet 5: Unmatched TradingView ({len(unmatched_tv)} trades)")

        # Sheet 6: Raw Framework Trades
        df_framework.to_excel(writer, sheet_name='Raw Framework', index=False)
        print(f"  ✓ Sheet 6: Raw Framework Data ({len(df_framework)} trades)")

        # Sheet 7: Raw TradingView Trades
        df_tv.to_excel(writer, sheet_name='Raw TradingView', index=False)
        print(f"  ✓ Sheet 7: Raw TradingView Data ({len(df_tv)} trades)")

    print(f"\n✓ Report saved: {output_file}")


def create_detailed_comparison(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Create detailed comparison showing differences in prices and P/L."""
    comparison_rows = []

    for _, match in matches_df.iterrows():
        fw = match['fw_trade']
        tv = match['tv_trade']

        # Extract key fields (handle different column names)
        entry_price_fw = extract_price(fw, ['entry_price', 'Entry Price', 'entry', 'Entry'])
        exit_price_fw = extract_price(fw, ['exit_price', 'Exit Price', 'exit', 'Exit'])
        pl_fw = extract_price(fw, ['profit_loss', 'P&L', 'pnl', 'PnL', 'Profit/Loss'])

        entry_price_tv = extract_price(tv, ['entry_price', 'Entry Price', 'entry', 'Entry'])
        exit_price_tv = extract_price(tv, ['exit_price', 'Exit Price', 'exit', 'Exit'])
        pl_tv = extract_price(tv, ['profit_loss', 'P&L', 'pnl', 'PnL', 'Profit/Loss'])

        row = {
            'Match Type': match['match_type'],
            'Date Diff (days)': match['date_diff_days'],

            # Entry comparison
            'Entry Date FW': match['entry_date_fw'],
            'Entry Date TV': match['entry_date_tv'],
            'Entry Price FW': entry_price_fw,
            'Entry Price TV': entry_price_tv,
            'Entry Price Diff': entry_price_fw - entry_price_tv if entry_price_fw and entry_price_tv else None,
            'Entry Price Diff %': ((entry_price_fw - entry_price_tv) / entry_price_tv * 100) if entry_price_fw and entry_price_tv and entry_price_tv != 0 else None,

            # Exit comparison
            'Exit Date FW': fw.get('exit_date', fw.get('Exit Date')),
            'Exit Date TV': tv.get('exit_date', tv.get('Exit Date')),
            'Exit Price FW': exit_price_fw,
            'Exit Price TV': exit_price_tv,
            'Exit Price Diff': exit_price_fw - exit_price_tv if exit_price_fw and exit_price_tv else None,
            'Exit Price Diff %': ((exit_price_fw - exit_price_tv) / exit_price_tv * 100) if exit_price_fw and exit_price_tv and exit_price_tv != 0 else None,

            # P/L comparison
            'P/L FW': pl_fw,
            'P/L TV': pl_tv,
            'P/L Diff': pl_fw - pl_tv if pl_fw and pl_tv else None,
            'P/L Diff %': ((pl_fw - pl_tv) / abs(pl_tv) * 100) if pl_fw and pl_tv and pl_tv != 0 else None,
        }

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def extract_price(trade_dict: dict, possible_keys: list) -> float:
    """Extract price from trade dict, trying multiple possible key names."""
    for key in possible_keys:
        if key in trade_dict:
            val = trade_dict[key]
            if pd.notna(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Compare Framework and TradingView trade logs'
    )
    parser.add_argument(
        '--framework',
        required=True,
        help='Path to Framework trade log CSV'
    )
    parser.add_argument(
        '--tradingview',
        required=True,
        help='Path to TradingView trade log CSV'
    )
    parser.add_argument(
        '--output',
        default='trade_comparison_report.xlsx',
        help='Output Excel report filename (default: trade_comparison_report.xlsx)'
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=5,
        help='Maximum days difference for fuzzy matching (default: 5)'
    )

    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.framework).exists():
        print(f"✗ Error: Framework file not found: {args.framework}")
        sys.exit(1)

    if not Path(args.tradingview).exists():
        print(f"✗ Error: TradingView file not found: {args.tradingview}")
        sys.exit(1)

    print("="*80)
    print("TRADE LOG COMPARISON TOOL")
    print("="*80)

    # Load trade logs
    df_fw = load_trade_log(args.framework, 'Framework')
    df_tv = load_trade_log(args.tradingview, 'TradingView')

    # Detect entry date columns (try common names)
    entry_col_fw = detect_column(df_fw, ['entry_date', 'Entry Date', 'date', 'Date'])
    entry_col_tv = detect_column(df_tv, ['entry_date', 'Entry Date', 'date', 'Date'])

    if not entry_col_fw or not entry_col_tv:
        print(f"\n✗ Error: Could not detect entry date columns")
        print(f"Framework columns: {df_fw.columns.tolist()}")
        print(f"TradingView columns: {df_tv.columns.tolist()}")
        sys.exit(1)

    print(f"\nUsing entry date columns:")
    print(f"  Framework: {entry_col_fw}")
    print(f"  TradingView: {entry_col_tv}")

    # Find exact matches
    exact_matches_df, matched_tv_exact = find_exact_matches(
        df_fw, df_tv, entry_col_fw, entry_col_tv
    )

    # Find fuzzy matches
    fuzzy_matches_df, matched_tv_fuzzy = find_fuzzy_matches(
        df_fw, df_tv, entry_col_fw, entry_col_tv,
        matched_tv_exact, args.max_days
    )

    # Combine all matches
    all_matches = pd.concat([exact_matches_df, fuzzy_matches_df], ignore_index=True)
    all_matched_tv = matched_tv_exact.union(matched_tv_fuzzy)

    # Generate report
    create_comparison_report(
        all_matches, df_fw, df_tv, all_matched_tv, args.output
    )

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


def detect_column(df: pd.DataFrame, possible_names: list) -> str:
    """Detect column name from list of possibilities."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


if __name__ == '__main__':
    main()
