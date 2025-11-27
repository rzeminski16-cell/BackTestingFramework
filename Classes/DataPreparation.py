import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Excel reporting imports
try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.chart import LineChart, BarChart, Reference, PieChart
    from openpyxl.chart.label import DataLabelList
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: openpyxl not installed. Excel reporting disabled. Install with: pip install openpyxl")

class DataPreparation:
    """
    Handles data preparation by filtering raw data to only required columns
    and optionally renaming columns based on a mapping
    """
    
    def __init__(self, raw_data_folder: str, prepared_data_folder: str):
        """
        Initialize data preparation
        
        Args:
            raw_data_folder: Path to folder with raw CSV files (with all indicators)
            prepared_data_folder: Path to save prepared data
        """
        self.raw_data_folder = Path(raw_data_folder)
        self.prepared_data_folder = Path(prepared_data_folder)
    
    def get_required_columns(self, strategy: BaseStrategy) -> List[str]:
        """
        Get list of columns required by the strategy
        
        Args:
            strategy: Strategy instance
        
        Returns:
            List of required column names
        """
        return strategy.required_columns()
    
    def prepare_data_for_strategy(self, 
                                  strategy: BaseStrategy, 
                                  security_metadata: Dict[str, Dict],
                                  column_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Prepare data for all securities based on strategy requirements
        
        Args:
            strategy: Strategy instance
            security_metadata: Dict mapping symbol to metadata 
                              (e.g., {'AAPL': {'type': 'stock', 'sector': 'tech'}})
            column_mapping: Optional dict mapping strategy column names to raw data column names
                          Example: {'sma_50': 'SMA_50', 'close': 'Close', 'date': 'Date'}
                          If None, assumes column names in raw data match strategy requirements
        
        Returns:
            Dict mapping symbol to path of prepared data file
        """
        strategy_folder = self.prepared_data_folder / strategy.name
        strategy_folder.mkdir(parents=True, exist_ok=True)
        
        required_cols = strategy.required_columns()
        
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.name}")
        print(f"Required columns: {required_cols}")
        if column_mapping:
            print(f"Column mapping provided:")
            for strategy_col, raw_col in column_mapping.items():
                if strategy_col in required_cols:
                    print(f"  {strategy_col} <- {raw_col}")
        print(f"{'='*60}")
        
        prepared_files = {}
        missing_columns_report = {}
        
        for symbol, metadata in security_metadata.items():
            print(f"\nPreparing data for {symbol}...")
            
            # Load raw data
            raw_file = self.raw_data_folder / f"{symbol}.csv"
            if not raw_file.exists():
                print(f"  ⚠ Warning: Raw data file not found for {symbol}")
                continue
            
            try:
                df = pd.read_csv(raw_file)
                print(f"  Loaded {len(df)} rows")
                print(f"  Available columns: {list(df.columns)}")
                
                # Create column mapping
                if column_mapping:
                    # Use provided mapping
                    rename_dict = {}
                    for strategy_col in required_cols:
                        if strategy_col in column_mapping:
                            raw_col = column_mapping[strategy_col]
                            if raw_col in df.columns:
                                rename_dict[raw_col] = strategy_col
                            else:
                                print(f"  ⚠ Warning: Mapped column '{raw_col}' not found in {symbol}")
                                if symbol not in missing_columns_report:
                                    missing_columns_report[symbol] = []
                                missing_columns_report[symbol].append(f"{strategy_col} (mapped from {raw_col})")
                    
                    # Rename columns
                    df = df.rename(columns=rename_dict)
                
                # Check if all required columns are present
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  ⚠ Warning: Missing required columns: {missing_cols}")
                    if symbol not in missing_columns_report:
                        missing_columns_report[symbol] = []
                    missing_columns_report[symbol].extend(missing_cols)
                    continue
                
                # Keep only required columns
                df = df[required_cols].copy()
                
                # Convert date column to datetime
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # Remove rows with any NaN values in required columns
                initial_rows = len(df)
                df = df.dropna()
                if len(df) < initial_rows:
                    print(f"  Removed {initial_rows - len(df)} rows with missing values")
                
                # Add metadata columns
                for key, value in metadata.items():
                    df[f'metadata_{key}'] = value
                
                # Save prepared data
                output_file = strategy_folder / f"{symbol}.csv"
                df.to_csv(output_file, index=False)
                prepared_files[symbol] = str(output_file)
                
                print(f"  ✓ Saved prepared data: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"  ✗ Error processing {symbol}: {e}")
                continue
        
        # Save metadata about this preparation run
        metadata_file = strategy_folder / "preparation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'strategy': strategy.name,
                'strategy_params': strategy.params,
                'preparation_date': datetime.now().isoformat(),
                'required_columns': required_cols,
                'column_mapping_used': column_mapping,
                'securities_prepared': list(prepared_files.keys()),
                'securities_with_missing_columns': missing_columns_report
            }, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PREPARATION SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully prepared: {len(prepared_files)} securities")
        if missing_columns_report:
            print(f"\n⚠ Securities with missing columns:")
            for symbol, missing in missing_columns_report.items():
                print(f"  {symbol}: {missing}")
        print(f"\nPrepared data saved to: {strategy_folder}")
        print(f"{'='*60}")
        
        return prepared_files