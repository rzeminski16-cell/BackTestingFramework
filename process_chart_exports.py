#!/usr/bin/env python3
"""
This is a stand alone script to get the raw data ready for backtesting.
Process TradingView chart exports to standardize schema and fill Candle Position values.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict
import sys


def process_candle_position(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Process the Candle Position column by working backwards from non-(-1) values.

    Args:
        df: DataFrame with Candle Position column
        date_col: Name of the date column for sorting

    Returns:
        DataFrame with processed Candle Position values
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure data is sorted by date (oldest to newest)
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # Find all indices where Candle Position is not -1
    non_negative_indices = df[df['candle_position'] != -1].index.tolist()

    # Process each non-(-1) value by working backwards
    for idx in non_negative_indices:
        current_value = df.loc[idx, 'candle_position']

        # Work backwards from this index
        countdown = current_value - 1
        current_idx = idx - 1

        while current_idx >= 0 and countdown >= 1:
            # Stop if we encounter another non-(-1) value
            if df.loc[current_idx, 'candle_position'] != -1:
                break

            # Assign the countdown value
            df.loc[current_idx, 'candle_position'] = countdown
            countdown -= 1
            current_idx -= 1

    return df


def process_csv_files(input_folder: str, output_folder: str, column_rename: Dict[str, str]):
    """
    Process all CSV files in the input folder.

    Args:
        input_folder: Path to folder containing CSV files
        output_folder: Path to folder for processed CSV files
        column_rename: Dictionary mapping old column names to new names
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all CSV files
    csv_files = list(input_path.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process\n")

    # Store summary information
    summary = []

    for csv_file in csv_files:
        ticker = csv_file.stem  # Filename without extension
        print(f"Processing {ticker}...")

        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Rename columns
            df = df.rename(columns=column_rename)
            
            df['open'] = df['open'].round(2)
            df['high'] = df['high'].round(2)
            df['low'] = df['low'].round(2)
            df['close'] = df['close'].round(2)

            # Process Candle Position column
            df = process_candle_position(df, date_col='date')

            # Get date range for summary
            # Parse dates to find min/max
            df['Date_parsed'] = pd.to_datetime(df['date'], dayfirst=False, errors='coerce')
            min_date = df['Date_parsed'].min()
            max_date = df['Date_parsed'].max()
            num_records = len(df)

            summary.append({
                'Ticker': ticker,
                'Earliest Date': min_date.strftime('%d/%m/%Y') if pd.notna(min_date) else 'N/A',
                'Latest Date': max_date.strftime('%d/%m/%Y') if pd.notna(max_date) else 'N/A',
                'Records': num_records
            })

            # Drop the temporary parsed date column
            columns_to_drop = df.columns.tolist()

            values_to_remove = set(column_rename.values())
            values_to_remove.add("candle_position")
            
            columns_to_drop = [c for c in columns_to_drop if c not in values_to_remove]
            
            df = df.drop(columns=['Date_parsed'] + columns_to_drop)

            # Save to output folder
            output_file = output_path / csv_file.name
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved to {output_file}")

        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {str(e)}")
            summary.append({
                'Ticker': ticker,
                'Earliest Date': 'ERROR',
                'Latest Date': 'ERROR',
                'Records': 0
            })

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Date Ranges by Security")
    print("="*80)

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Ticker')

    print(summary_df.to_string(index=False))
    print("\n")


def main():
    """Main entry point for the script."""

    # Example column rename dictionary
    # Modify this according to your actual column names
    column_rename = {
        'time': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'candle_position': 'candle_position',
        'MA': 'sma_200',
        'EMA': 'ema_50',
        'RSI': 'rsi_14',
        'Volume': 'volume',
        'CMF': 'cmf',
        'ATR': 'atr_14',
        'ParabolicSAR': 'par_sar',
        'Basis': 'base_bb',
        'Upper': 'upper_bb',
        'Lower': 'lower_bb'
    }

    # Get input and output folders from command line arguments or use defaults
    if len(sys.argv) >= 3:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
    else:
        # Default folders
        input_folder = './raw_data'
        output_folder = 'raw_data/processed_exports'

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Column rename mapping: {column_rename}\n")

    process_csv_files(input_folder, output_folder, column_rename)


if __name__ == '__main__':
    main()