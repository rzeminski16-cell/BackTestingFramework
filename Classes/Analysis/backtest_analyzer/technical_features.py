"""
Technical features generator for backtest analysis.

This module generates a master technical features CSV containing:
- All trades that satisfy the minimum trades per year filter
- Technical indicators at each trade entry
- GB_Flag classification (Good/Indeterminate/Bad based on profit)
- Outcomes_Flag classification (post-exit price behavior)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import AnalysisConfig
from .indicators import WeeklyIndicatorCalculator


class TechnicalFeaturesGenerator:
    """
    Generates technical features for backtested trades.

    Creates a master CSV with:
    - Filtered trades (>4 trades per security per year)
    - Technical indicators at entry
    - GB_Flag: G (profit ≥5%), I (0-5%), B (loss)
    - Outcomes_Flag: FullRideGood, EarlyExitGood, MissedOpportunity
    """

    def __init__(self, config: AnalysisConfig,
                 indicator_calculator: WeeklyIndicatorCalculator):
        """
        Initialize the generator.

        Args:
            config: Analysis configuration
            indicator_calculator: Pre-initialized indicator calculator
        """
        self.config = config
        self.indicator_calculator = indicator_calculator

    def load_trade_log(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a trade log CSV file.

        Args:
            file_path: Path to the trade log CSV

        Returns:
            DataFrame with trade data or None if failed
        """
        try:
            df = pd.read_csv(file_path)

            # Parse date columns
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])

            # Sort by entry date
            df.sort_values('entry_date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            print(f"Error loading trade log {file_path}: {e}")
            return None

    def filter_by_minimum_trades(self, trades: pd.DataFrame,
                                  min_trades: int) -> pd.DataFrame:
        """
        Filter trades to only include security-year combinations
        with more than min_trades.

        Note: A trade is registered on entry_date, not exit_date.

        Args:
            trades: DataFrame of trades
            min_trades: Minimum trades required per security per year

        Returns:
            Filtered DataFrame
        """
        if trades.empty:
            return trades

        # Add entry year column
        trades = trades.copy()
        trades['entry_year'] = trades['entry_date'].dt.year

        # Count trades per symbol-year
        trade_counts = trades.groupby(['symbol', 'entry_year']).size().reset_index(name='trade_count')

        # Filter to valid combinations (>= min_trades, not > min_trades)
        valid_combos = trade_counts[trade_counts['trade_count'] >= min_trades]

        # Merge to filter original trades
        filtered = trades.merge(
            valid_combos[['symbol', 'entry_year']],
            on=['symbol', 'entry_year'],
            how='inner'
        )

        return filtered

    def classify_gb_flag(self, pl_pct: float) -> str:
        """
        Classify a trade based on profit/loss percentage.

        Args:
            pl_pct: Profit/loss percentage (as stored in trade log, e.g., "4.72%" -> 4.72)

        Returns:
            Classification: "G", "I", or "B"
        """
        # Handle string percentage if needed
        if isinstance(pl_pct, str):
            pl_pct = float(pl_pct.replace('%', ''))

        threshold = self.config.gb_profit_good_threshold

        if pl_pct >= threshold:
            return "G"  # Good
        elif pl_pct >= 0:
            return "I"  # Indeterminate
        else:
            return "B"  # Bad

    def classify_outcomes_flag(self, trade: pd.Series,
                                price_after_21d: Optional[float],
                                max_price_after: Optional[float]) -> str:
        """
        Classify a trade based on post-exit price behavior.

        Args:
            trade: Trade data row
            price_after_21d: Price 21 trading days after exit
            max_price_after: Maximum price in the 21 days after exit

        Returns:
            Classification: "FullRideGood", "EarlyExitGood", "MissedOpportunity", or "Unknown"
        """
        cfg = self.config

        # Parse pl_pct
        pl_pct = trade['pl_pct']
        if isinstance(pl_pct, str):
            pl_pct = float(pl_pct.replace('%', ''))

        entry_price = trade['entry_price']
        exit_price = trade['exit_price']

        # If no future price data available
        if price_after_21d is None and max_price_after is None:
            return "Unknown"

        # Use last available price if 21d price is None
        price_21d = price_after_21d if price_after_21d is not None else exit_price

        # FullRideGood: Profit ≥5% AND price 21 days later ≤ exit price + 10% of entry price
        full_ride_threshold = cfg.full_ride_profit_min
        price_extension_limit = exit_price + (entry_price * cfg.full_ride_price_extension / 100)

        if pl_pct >= full_ride_threshold and price_21d <= price_extension_limit:
            return "FullRideGood"

        # EarlyExitGood: Profit ≥0% AND price 21 days later shows no major extension
        # Major extension is 5% on top of exit price
        major_extension_price = exit_price * (1 + cfg.early_exit_major_extension / 100)

        if pl_pct >= 0:
            # Check if no major extension occurred
            if max_price_after is None or max_price_after <= major_extension_price:
                return "EarlyExitGood"

        # MissedOpportunity: Exit at small profit (<5%) or loss but price surges >10% after exit
        surge_threshold_price = exit_price * (1 + cfg.missed_opportunity_surge / 100)

        if pl_pct < full_ride_threshold:  # Small profit or loss
            if max_price_after is not None and max_price_after > surge_threshold_price:
                return "MissedOpportunity"

        # Default: doesn't fit clear categories
        return "Neutral"

    def add_technical_indicators(self, trade: pd.Series,
                                  symbol: str) -> Dict:
        """
        Get all technical indicators for a trade at entry date.

        Args:
            trade: Trade data row
            symbol: Security symbol

        Returns:
            Dictionary of indicator values
        """
        entry_date = trade['entry_date']
        if isinstance(entry_date, str):
            entry_date = pd.to_datetime(entry_date)

        indicators = self.indicator_calculator.get_indicators_at_date(symbol, entry_date)
        return indicators

    def process_single_trade(self, trade: pd.Series,
                              symbol: str) -> Dict:
        """
        Process a single trade to add all features.

        Args:
            trade: Trade data row
            symbol: Security symbol

        Returns:
            Dictionary with all trade data and features
        """
        cfg = self.config

        # Start with original trade data
        result = trade.to_dict()

        # Add technical indicators
        indicators = self.add_technical_indicators(trade, symbol)
        result.update(indicators)

        # Parse pl_pct for classification
        pl_pct = trade['pl_pct']
        if isinstance(pl_pct, str):
            pl_pct_float = float(pl_pct.replace('%', ''))
        else:
            pl_pct_float = float(pl_pct)

        # Add GB_Flag
        result['GB_Flag'] = self.classify_gb_flag(pl_pct_float)

        # Get future prices for Outcomes_Flag
        exit_date = trade['exit_date']
        if isinstance(exit_date, str):
            exit_date = pd.to_datetime(exit_date)

        price_after_21d = self.indicator_calculator.get_price_after_days(
            symbol, exit_date, cfg.days_after_exit
        )
        max_price_after = self.indicator_calculator.get_max_price_in_period(
            symbol, exit_date, cfg.days_after_exit
        )

        # Add price after exit for reference
        result['price_21d_after_exit'] = price_after_21d
        result['max_price_21d_after_exit'] = max_price_after

        # Add Outcomes_Flag
        result['Outcomes_Flag'] = self.classify_outcomes_flag(
            trade, price_after_21d, max_price_after
        )

        return result

    def process_trade_log_file(self, file_path: Path,
                                symbol: str) -> Optional[pd.DataFrame]:
        """
        Process a single trade log file.

        Args:
            file_path: Path to the trade log CSV
            symbol: Security symbol

        Returns:
            DataFrame with processed trades or None if failed
        """
        trades = self.load_trade_log(file_path)
        if trades is None:
            return None

        # Ensure symbol column exists
        if 'symbol' not in trades.columns:
            trades['symbol'] = symbol

        return trades

    def generate_technical_features(self, trade_logs_folder: Path,
                                     save_weekly: bool = True,
                                     weekly_output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate technical features master file from all trade logs.

        Args:
            trade_logs_folder: Path to folder containing trade log CSVs
            save_weekly: Whether to save weekly indicator data for validation
            weekly_output_dir: Directory for weekly data output

        Returns:
            Master DataFrame with all filtered trades and features
        """
        trade_logs_folder = Path(trade_logs_folder)
        cfg = self.config

        # Collect all trades
        all_trades = []
        trade_log_files = list(trade_logs_folder.glob("*_trades.csv"))

        print(f"Found {len(trade_log_files)} trade log files")

        for file_path in trade_log_files:
            # Extract symbol from filename
            filename = file_path.stem
            parts = filename.rsplit('_', 2)

            if len(parts) >= 2 and parts[-1] == 'trades':
                symbol = parts[-2]
            else:
                print(f"Warning: Could not extract symbol from {filename}")
                continue

            print(f"Loading {symbol}...")

            trades = self.process_trade_log_file(file_path, symbol)
            if trades is not None and not trades.empty:
                all_trades.append(trades)

                # Save weekly data for validation
                if save_weekly and weekly_output_dir:
                    self.indicator_calculator.save_weekly_data(symbol, weekly_output_dir)

        if not all_trades:
            print("No trade logs loaded")
            return pd.DataFrame()

        # Combine all trades
        combined = pd.concat(all_trades, ignore_index=True)
        print(f"Total trades before filtering: {len(combined)}")

        # Filter by minimum trades per year
        filtered = self.filter_by_minimum_trades(combined, cfg.min_trades_per_year)
        print(f"Trades after filtering (>={cfg.min_trades_per_year} trades/year): {len(filtered)}")

        if filtered.empty:
            print("No trades remaining after filtering")
            return pd.DataFrame()

        # Process each trade to add features
        processed_trades = []
        total = len(filtered)

        for idx, (_, trade) in enumerate(filtered.iterrows()):
            if (idx + 1) % 100 == 0:
                print(f"Processing trade {idx + 1}/{total}...")

            symbol = trade['symbol']
            processed = self.process_single_trade(trade, symbol)
            processed_trades.append(processed)

        # Create master DataFrame
        master_df = pd.DataFrame(processed_trades)

        # Reorder columns for clarity
        priority_cols = [
            'trade_id', 'symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price',
            'pl', 'pl_pct', 'duration_days', 'entry_reason', 'exit_reason',
            'GB_Flag', 'Outcomes_Flag',
            'rsi_14w', 'high_low_distance_52w', 'bb_position', 'bb_width',
            'volume_trend', 'atr_14w', 'atr_pct',
            'price_vs_sma200', 'above_sma200', 'volume_ratio',
            'days_since_52w_high', 'days_since_52w_low',
            'price_21d_after_exit', 'max_price_21d_after_exit',
        ]

        # Get existing columns in priority order, then remaining
        existing_priority = [c for c in priority_cols if c in master_df.columns]
        remaining = [c for c in master_df.columns if c not in priority_cols]
        ordered_cols = existing_priority + remaining

        master_df = master_df[ordered_cols]

        return master_df

    def generate_summary_stats(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics from the master features file.

        Args:
            master_df: Master DataFrame with all features

        Returns:
            Summary statistics DataFrame
        """
        if master_df.empty:
            return pd.DataFrame()

        summary = []

        # Overall statistics
        overall = {
            'category': 'Overall',
            'subcategory': 'All',
            'total_trades': len(master_df),
            'gb_good_count': (master_df['GB_Flag'] == 'G').sum(),
            'gb_indeterminate_count': (master_df['GB_Flag'] == 'I').sum(),
            'gb_bad_count': (master_df['GB_Flag'] == 'B').sum(),
        }

        # Add Outcomes_Flag counts
        for flag in ['FullRideGood', 'EarlyExitGood', 'MissedOpportunity', 'Neutral', 'Unknown']:
            overall[f'outcome_{flag.lower()}_count'] = (master_df['Outcomes_Flag'] == flag).sum()

        summary.append(overall)

        # Per-symbol statistics
        for symbol in master_df['symbol'].unique():
            symbol_trades = master_df[master_df['symbol'] == symbol]

            symbol_stats = {
                'category': 'Symbol',
                'subcategory': symbol,
                'total_trades': len(symbol_trades),
                'gb_good_count': (symbol_trades['GB_Flag'] == 'G').sum(),
                'gb_indeterminate_count': (symbol_trades['GB_Flag'] == 'I').sum(),
                'gb_bad_count': (symbol_trades['GB_Flag'] == 'B').sum(),
            }

            for flag in ['FullRideGood', 'EarlyExitGood', 'MissedOpportunity', 'Neutral', 'Unknown']:
                symbol_stats[f'outcome_{flag.lower()}_count'] = (symbol_trades['Outcomes_Flag'] == flag).sum()

            summary.append(symbol_stats)

        # Per-year statistics
        if 'entry_year' in master_df.columns:
            for year in sorted(master_df['entry_year'].unique()):
                year_trades = master_df[master_df['entry_year'] == year]

                year_stats = {
                    'category': 'Year',
                    'subcategory': str(year),
                    'total_trades': len(year_trades),
                    'gb_good_count': (year_trades['GB_Flag'] == 'G').sum(),
                    'gb_indeterminate_count': (year_trades['GB_Flag'] == 'I').sum(),
                    'gb_bad_count': (year_trades['GB_Flag'] == 'B').sum(),
                }

                for flag in ['FullRideGood', 'EarlyExitGood', 'MissedOpportunity', 'Neutral', 'Unknown']:
                    year_stats[f'outcome_{flag.lower()}_count'] = (year_trades['Outcomes_Flag'] == flag).sum()

                summary.append(year_stats)

        return pd.DataFrame(summary)

    def generate_filtered_trade_counts(self, trade_logs_folder: Path) -> pd.DataFrame:
        """
        Generate a report of trade counts before and after filtering.

        Args:
            trade_logs_folder: Path to folder containing trade log CSVs

        Returns:
            DataFrame with trade counts per symbol-year
        """
        trade_logs_folder = Path(trade_logs_folder)
        cfg = self.config

        all_trades = []
        trade_log_files = list(trade_logs_folder.glob("*_trades.csv"))

        for file_path in trade_log_files:
            filename = file_path.stem
            parts = filename.rsplit('_', 2)

            if len(parts) >= 2 and parts[-1] == 'trades':
                symbol = parts[-2]
            else:
                continue

            trades = self.load_trade_log(file_path)
            if trades is not None and not trades.empty:
                trades['symbol'] = symbol
                all_trades.append(trades)

        if not all_trades:
            return pd.DataFrame()

        combined = pd.concat(all_trades, ignore_index=True)
        combined['entry_year'] = combined['entry_date'].dt.year

        # Count trades per symbol-year
        counts = combined.groupby(['symbol', 'entry_year']).size().reset_index(name='trade_count')
        counts['included'] = counts['trade_count'] >= cfg.min_trades_per_year

        return counts
