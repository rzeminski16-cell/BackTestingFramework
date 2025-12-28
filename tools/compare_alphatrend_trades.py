"""
Compare AlphaTrend trade logs: Framework vs TradingView.

Handles TradingView's format where Entry and Exit are separate rows.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys


def parse_tradingview_trades(filepath: str) -> pd.DataFrame:
    """Parse TradingView CSV where each trade has Entry/Exit rows."""
    print(f"Parsing TradingView trades from: {filepath}")

    df = pd.read_csv(filepath)
    print(f"  Raw rows: {len(df)}")

    # TradingView format: alternating Entry/Exit rows
    # Combine them into single trade rows
    trades = []

    for trade_num in df['Trade #'].unique():
        trade_rows = df[df['Trade #'] == trade_num]

        # Find entry and exit rows
        entry_row = trade_rows[trade_rows['Type'] == 'Entry long']
        exit_row = trade_rows[trade_rows['Type'] == 'Exit long']

        if len(entry_row) == 0 or len(exit_row) == 0:
            continue

        entry = entry_row.iloc[0]
        exit = exit_row.iloc[0]

        trades.append({
            'trade_num': trade_num,
            'entry_date': pd.to_datetime(entry['Date/Time']),
            'entry_price': entry['Price USD'],
            'entry_signal': entry['Signal'],
            'exit_date': pd.to_datetime(exit['Date/Time']),
            'exit_price': exit['Price USD'],
            'exit_signal': exit['Signal'],
            'quantity': entry['Position size (qty)'],
            'position_value': entry['Position size (value)'],
            'net_pl': exit['Net P&L GBP'],
            'net_pl_pct': exit['Net P&L %']
        })

    result = pd.DataFrame(trades)
    print(f"  ✓ Parsed {len(result)} complete trades")
    return result


def load_framework_trades(filepath: str) -> pd.DataFrame:
    """Load framework trades CSV."""
    print(f"Loading Framework trades from: {filepath}")

    df = pd.read_csv(filepath)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])

    print(f"  ✓ Loaded {len(df)} trades")
    return df


def find_matches(df_fw: pd.DataFrame, df_tv: pd.DataFrame, max_days: int = 5):
    """Find exact and fuzzy matches between framework and TradingView trades."""

    exact_matches = []
    fuzzy_matches = []
    matched_tv_indices = set()

    print(f"\nFinding matches (max {max_days} days difference)...")

    for idx_fw, row_fw in df_fw.iterrows():
        entry_fw = row_fw['entry_date']

        best_match = None
        best_diff = float('inf')
        best_idx = None

        for idx_tv, row_tv in df_tv.iterrows():
            if idx_tv in matched_tv_indices:
                continue

            entry_tv = row_tv['entry_date']
            date_diff = abs((entry_fw - entry_tv).days)

            if date_diff < best_diff:
                best_diff = date_diff
                best_match = row_tv
                best_idx = idx_tv

        if best_match is not None and best_diff <= max_days:
            match_data = {
                'fw_index': idx_fw,
                'tv_index': best_idx,
                'match_type': 'EXACT' if best_diff == 0 else 'FUZZY',
                'date_diff_days': best_diff,

                # Framework data
                'fw_entry_date': row_fw['entry_date'],
                'fw_exit_date': row_fw['exit_date'],
                'fw_entry_price': row_fw['entry_price'],
                'fw_exit_price': row_fw['exit_price'],
                'fw_quantity': row_fw['quantity'],
                'fw_pl': row_fw['pl'],
                'fw_pl_pct': row_fw['pl_pct'],
                'fw_duration_days': row_fw['duration_days'],

                # TradingView data
                'tv_entry_date': best_match['entry_date'],
                'tv_exit_date': best_match['exit_date'],
                'tv_entry_price': best_match['entry_price'],
                'tv_exit_price': best_match['exit_price'],
                'tv_quantity': best_match['quantity'],
                'tv_pl': best_match['net_pl'],
                'tv_pl_pct': best_match['net_pl_pct'],
                'tv_trade_num': best_match['trade_num'],
            }

            # Calculate differences
            match_data['entry_price_diff'] = row_fw['entry_price'] - best_match['entry_price']
            match_data['entry_price_diff_pct'] = (match_data['entry_price_diff'] / best_match['entry_price']) * 100

            match_data['exit_price_diff'] = row_fw['exit_price'] - best_match['exit_price']
            match_data['exit_price_diff_pct'] = (match_data['exit_price_diff'] / best_match['exit_price']) * 100

            match_data['pl_diff'] = row_fw['pl'] - best_match['net_pl']
            match_data['pl_pct_diff'] = row_fw['pl_pct'] - best_match['net_pl_pct']

            if best_diff == 0:
                exact_matches.append(match_data)
            else:
                fuzzy_matches.append(match_data)

            matched_tv_indices.add(best_idx)

    print(f"  ✓ Found {len(exact_matches)} exact matches")
    print(f"  ✓ Found {len(fuzzy_matches)} fuzzy matches")

    return pd.DataFrame(exact_matches), pd.DataFrame(fuzzy_matches), matched_tv_indices


def create_excel_report(exact_df: pd.DataFrame, fuzzy_df: pd.DataFrame,
                        df_fw: pd.DataFrame, df_tv: pd.DataFrame,
                        matched_tv: set, output_file: str):
    """Create comprehensive Excel report."""

    print(f"\nGenerating Excel report: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # Sheet 1: Summary
        summary = pd.DataFrame({
            'Metric': [
                'Total Framework Trades',
                'Total TradingView Trades',
                'Exact Matches (same date)',
                'Fuzzy Matches (within 5 days)',
                'Total Matched',
                'Unmatched Framework',
                'Unmatched TradingView',
                'Match Rate (%)'
            ],
            'Count': [
                len(df_fw),
                len(df_tv),
                len(exact_df),
                len(fuzzy_df),
                len(exact_df) + len(fuzzy_df),
                len(df_fw) - len(exact_df) - len(fuzzy_df),
                len(df_tv) - len(matched_tv),
                round(((len(exact_df) + len(fuzzy_df)) / len(df_fw) * 100), 2) if len(df_fw) > 0 else 0
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        print(f"  ✓ Sheet 1: Summary")

        # Sheet 2: Exact Matches
        if len(exact_df) > 0:
            exact_df.to_excel(writer, sheet_name='Exact Matches', index=False)
            print(f"  ✓ Sheet 2: Exact Matches ({len(exact_df)} trades)")

        # Sheet 3: Fuzzy Matches
        if len(fuzzy_df) > 0:
            fuzzy_df.to_excel(writer, sheet_name='Fuzzy Matches', index=False)
            print(f"  ✓ Sheet 3: Fuzzy Matches ({len(fuzzy_df)} trades)")

        # Sheet 4: Unmatched Framework
        matched_fw_indices = set()
        if len(exact_df) > 0:
            matched_fw_indices.update(exact_df['fw_index'].tolist())
        if len(fuzzy_df) > 0:
            matched_fw_indices.update(fuzzy_df['fw_index'].tolist())

        unmatched_fw = df_fw[~df_fw.index.isin(matched_fw_indices)]
        if len(unmatched_fw) > 0:
            unmatched_fw.to_excel(writer, sheet_name='Unmatched Framework', index=False)
            print(f"  ✓ Sheet 4: Unmatched Framework ({len(unmatched_fw)} trades)")

        # Sheet 5: Unmatched TradingView
        unmatched_tv = df_tv[~df_tv.index.isin(matched_tv)]
        if len(unmatched_tv) > 0:
            unmatched_tv.to_excel(writer, sheet_name='Unmatched TradingView', index=False)
            print(f"  ✓ Sheet 5: Unmatched TradingView ({len(unmatched_tv)} trades)")

        # Sheet 6: All Framework Trades
        df_fw.to_excel(writer, sheet_name='All Framework Trades', index=False)
        print(f"  ✓ Sheet 6: All Framework Trades ({len(df_fw)} trades)")

        # Sheet 7: All TradingView Trades
        df_tv.to_excel(writer, sheet_name='All TradingView Trades', index=False)
        print(f"  ✓ Sheet 7: All TradingView Trades ({len(df_tv)} trades)")

    print(f"\n✓ Report saved: {output_file}")


def main():
    print("="*80)
    print("ALPHATREND TRADE COMPARISON TOOL")
    print("="*80)

    # File paths
    fw_file = "logs/AlphaTrendStrategy_test_1/AlphaTrendStrategy_test_1_AAPL_trades.csv"
    tv_file = "logs/AlphaTrendStrategy_test_1/AlphaTrendStrategy_test_1_AAPL_trades_Trading_view.csv"
    output_file = "alphatrend_trade_comparison.xlsx"

    # Load data
    df_fw = load_framework_trades(fw_file)
    df_tv = parse_tradingview_trades(tv_file)

    # Find matches
    exact_df, fuzzy_df, matched_tv = find_matches(df_fw, df_tv, max_days=5)

    # Generate report
    create_excel_report(exact_df, fuzzy_df, df_fw, df_tv, matched_tv, output_file)

    # Print statistics
    print("\n" + "="*80)
    print("COMPARISON STATISTICS")
    print("="*80)
    print(f"Framework Trades:    {len(df_fw)}")
    print(f"TradingView Trades:  {len(df_tv)}")
    print(f"Exact Matches:       {len(exact_df)} ({len(exact_df)/len(df_fw)*100:.1f}%)")
    print(f"Fuzzy Matches:       {len(fuzzy_df)} ({len(fuzzy_df)/len(df_fw)*100:.1f}%)")
    print(f"Unmatched FW:        {len(df_fw) - len(exact_df) - len(fuzzy_df)}")
    print(f"Unmatched TV:        {len(df_tv) - len(matched_tv)}")
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
