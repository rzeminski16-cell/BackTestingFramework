"""
Export indicator values from raw data to CSV for comparison.

This script reads pre-calculated indicators from raw data CSV files
and exports them to a new CSV file for analysis and comparison.

NOTE: This script no longer calculates indicators. All indicators are
expected to be pre-calculated in the raw data files.

Usage:
    python export_alphatrend_indicators.py --symbol AAPL

Output:
    Creates a CSV file in the current directory with all indicator values.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def load_data(symbol: str, data_dir: str = 'raw_data') -> pd.DataFrame:
    """Load OHLCV data with pre-calculated indicators for the given symbol."""
    data_path = Path(data_dir) / f"{symbol}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.rename(columns={'time': 'date'}, inplace=True)

    return df


def export_indicators_to_csv(symbol: str = 'AAPL', output_file: str = None):
    """
    Export pre-calculated indicators from raw data to CSV.

    Args:
        symbol: Stock symbol to process
        output_file: Output CSV filename (default: auto-generated)
    """
    print(f"Loading data for {symbol}...")

    # Load data with pre-calculated indicators
    try:
        data = load_data(symbol)
        print(f"✓ Loaded {len(data)} bars of data")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # List all standard indicators we expect
    standard_indicators = [
        'atr_14', 'ema_50', 'sma_200', 'rsi_14',
        'cmf', 'par_sar', 'base_bb', 'upper_bb', 'lower_bb'
    ]

    # Check which standard indicators are present
    present_indicators = [col for col in standard_indicators if col in data.columns]
    missing_indicators = [col for col in standard_indicators if col not in data.columns]

    print("\n" + "="*80)
    print("Standard Indicators Status:")
    print("="*80)
    if present_indicators:
        print("✓ Present indicators:")
        for ind in present_indicators:
            print(f"  - {ind}")
    if missing_indicators:
        print("\n✗ Missing indicators:")
        for ind in missing_indicators:
            print(f"  - {ind}")
    print("="*80)

    # Select columns to export (OHLCV + all available indicators)
    base_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    available_base = [col for col in base_columns if col in data.columns]

    # Get all other columns (these are indicators or custom columns)
    indicator_columns = [col for col in data.columns if col not in base_columns]

    # Combine columns for export
    columns_to_export = available_base + indicator_columns
    export_df = data[columns_to_export].copy()

    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'indicators_export_{symbol}_{timestamp}.csv'

    # Export to CSV
    print(f"\nExporting to {output_file}...")
    export_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"✓ Exported {len(export_df)} rows with {len(columns_to_export)} columns")

    # Print sample of first few rows
    print("\n" + "="*80)
    print("Sample of exported data (first 10 rows):")
    print("="*80)

    # Show sample with key columns
    sample_cols = ['date', 'close']
    # Add standard indicators that are present
    for ind in ['atr_14', 'ema_50', 'sma_200', 'rsi_14']:
        if ind in export_df.columns:
            sample_cols.append(ind)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(export_df[sample_cols].head(10))

    print("\n" + "="*80)
    print(f"✓ Complete! File contains {len(indicator_columns)} indicator columns.")
    print(f"✓ File location: {Path(output_file).absolute()}")
    print("="*80)

    # Print summary statistics for standard indicators
    if present_indicators:
        print("\nStandard Indicator Summary Statistics:")
        print("-" * 80)
        print(export_df[present_indicators].describe())

    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Export pre-calculated indicator values from raw data to CSV'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='SF',
        help='Stock symbol to process (default: SF)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV filename (default: auto-generated)'
    )

    args = parser.parse_args()

    try:
        export_indicators_to_csv(symbol=args.symbol, output_file=args.output)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
