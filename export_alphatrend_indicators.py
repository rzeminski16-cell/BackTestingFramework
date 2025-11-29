"""
Export AlphaTrend indicator values to CSV for comparison with TradingView.

This script calculates all AlphaTrend indicators using the framework's
indicator engine and exports them to a CSV file. This allows direct
comparison with TradingView's data window values to identify any
calculation discrepancies.

Usage:
    python export_alphatrend_indicators.py

Output:
    Creates a CSV file in the current directory with all indicator values.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Classes.Indicators.indicator_engine import IndicatorEngine
import json


def load_strategy_preset(preset_file: str) -> dict:
    """Load strategy preset configuration."""
    with open(preset_file, 'r') as f:
        preset = json.load(f)
    return preset['parameters']


def load_data(symbol: str, data_dir: str = 'raw_data/processed_exports') -> pd.DataFrame:
    """Load OHLCV data for the given symbol."""
    data_path = Path(data_dir) / f"{symbol}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df


def export_indicators_to_csv(symbol: str = 'AAPL', output_file: str = None):
    """
    Calculate and export AlphaTrend indicators to CSV.

    Args:
        symbol: Stock symbol to process
        output_file: Output CSV filename (default: auto-generated)
    """
    print(f"Loading data for {symbol}...")

    # Load data
    try:
        data = load_data(symbol)
        print(f"✓ Loaded {len(data)} bars of data")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # Load strategy parameters
    preset_file = 'config/strategy_presets/AlphaTrendStrategy__Default.json'
    print(f"\nLoading strategy parameters from {preset_file}...")
    params = load_strategy_preset(preset_file)
    print("✓ Parameters loaded:")
    for key, value in params.items():
        print(f"  - {key}: {value}")

    # Calculate indicators
    print(f"\nCalculating AlphaTrend indicators...")
    data_with_indicators = IndicatorEngine.calculate_alphatrend_indicators(
        data=data,
        atr_multiplier=params['atr_multiplier'],
        common_period=params['common_period'],
        source=params['source'],
        smoothing_length=params['smoothing_length'],
        percentile_period=params['percentile_period'],
        volume_short_ma=params['volume_short_ma'],
        volume_long_ma=params['volume_long_ma'],
        volume_alignment_window=params['volume_alignment_window'],
        signal_lookback=params['signal_lookback'],
        exit_ema_period=params['exit_ema_period']
    )
    print("✓ Indicators calculated")

    # Select columns to export
    columns_to_export = [
        # Date and OHLCV
        'date', 'open', 'high', 'low', 'close', 'volume',

        # True Range and ATR
        'tr', 'atr', 'atr_stop',

        # Adaptive coefficient components
        'atr_ema_long', 'volatility_ratio', 'adaptive_coeff',

        # AlphaTrend bands
        'up_band', 'down_band',

        # MFI components
        'typical_price', 'raw_money_flow', 'mfi',
        'mfi_upper', 'mfi_lower', 'mfi_threshold', 'momentum_bullish',

        # AlphaTrend and smoothing
        'alphatrend', 'smooth_at',

        # Crossover signals
        'at_cross_up', 'at_cross_down',
        'filtered_buy', 'filtered_sell',

        # Volume filter
        'vol_short_ma', 'vol_long_ma', 'volume_condition',

        # Exit EMA
        'exit_ema'
    ]

    # Filter to only columns that exist
    available_columns = [col for col in columns_to_export if col in data_with_indicators.columns]
    export_df = data_with_indicators[available_columns].copy()

    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'alphatrend_indicators_{symbol}_{timestamp}.csv'

    # Export to CSV
    print(f"\nExporting to {output_file}...")
    export_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"✓ Exported {len(export_df)} rows with {len(available_columns)} columns")

    # Print sample of first few rows
    print("\n" + "="*80)
    print("Sample of exported data (first 5 rows with key indicators):")
    print("="*80)

    sample_cols = ['date', 'close', 'atr', 'atr_stop', 'alphatrend', 'smooth_at',
                   'mfi', 'mfi_threshold', 'volume_condition', 'filtered_buy', 'filtered_sell']
    sample_cols = [col for col in sample_cols if col in export_df.columns]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(export_df[sample_cols].head(20))

    print("\n" + "="*80)
    print(f"✓ Complete! Compare this file with TradingView's data window values.")
    print(f"✓ File location: {Path(output_file).absolute()}")
    print("="*80)

    # Print summary statistics
    print("\nIndicator Summary Statistics:")
    print("-" * 80)
    stats_cols = ['atr', 'atr_stop', 'alphatrend', 'mfi', 'exit_ema']
    stats_cols = [col for col in stats_cols if col in export_df.columns]
    print(export_df[stats_cols].describe())

    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Export AlphaTrend indicator values to CSV for comparison with TradingView'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Stock symbol to process (default: AAPL)'
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
