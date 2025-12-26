"""
Fundamental features generator for backtest analysis.

This module generates fundamental feature CSVs for each security containing:
- Year and month for the full trading range
- period_GB_flag calculated per year based on Calmar ratio and max drawdown

The user can then manually append fundamental data (EPS, growth, etc.)
for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .config import AnalysisConfig


class FundamentalFeaturesGenerator:
    """
    Generates fundamental feature CSVs for backtested securities.

    For each security, creates a CSV with:
    - year, month columns for the full trading range
    - period_GB_flag calculated per year:
        - "good": Calmar > 0.5 AND max_dd <= 25%
        - "indeterminate": Calmar > 0.5 AND max_dd > 25%
        - "bad": otherwise
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize the generator.

        Args:
            config: Analysis configuration
        """
        self.config = config

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

    def calculate_yearly_metrics(self, trades: pd.DataFrame,
                                  initial_capital: float = 10000.0) -> pd.DataFrame:
        """
        Calculate yearly performance metrics from trades.

        Args:
            trades: DataFrame of trades
            initial_capital: Starting capital for calculations

        Returns:
            DataFrame with yearly metrics including Calmar ratio and max drawdown
        """
        if trades.empty:
            return pd.DataFrame()

        # Get unique years based on entry_date (trades registered on entry)
        trades['entry_year'] = trades['entry_date'].dt.year

        yearly_metrics = []

        for year in sorted(trades['entry_year'].unique()):
            year_trades = trades[trades['entry_year'] == year].copy()

            if year_trades.empty:
                continue

            # Sort by exit date for sequential P/L calculation
            year_trades = year_trades.sort_values('exit_date')

            # Calculate total return for the year
            total_pl = year_trades['pl'].sum()
            year_return_pct = (total_pl / initial_capital) * 100

            # Calculate max drawdown from cumulative P/L
            cumulative_pl = year_trades['pl'].cumsum()
            equity_curve = initial_capital + cumulative_pl

            running_max = equity_curve.expanding().max()
            drawdown = running_max - equity_curve
            drawdown_pct = (drawdown / running_max) * 100

            max_dd_pct = drawdown_pct.max() if len(drawdown_pct) > 0 else 0.0

            # Handle edge cases
            if pd.isna(max_dd_pct):
                max_dd_pct = 0.0

            # Calculate Calmar ratio (actual yearly return / max drawdown)
            # Using actual return, not annualized, as per user specification
            if max_dd_pct > 0:
                calmar_ratio = year_return_pct / max_dd_pct
            elif year_return_pct > 0:
                calmar_ratio = 999.99  # Very high (all gains, no drawdown)
            else:
                calmar_ratio = 0.0

            yearly_metrics.append({
                'year': year,
                'num_trades': len(year_trades),
                'total_pl': total_pl,
                'return_pct': year_return_pct,
                'max_drawdown_pct': max_dd_pct,
                'calmar_ratio': calmar_ratio,
            })

        return pd.DataFrame(yearly_metrics)

    def classify_period(self, calmar_ratio: float,
                        max_dd_pct: float) -> str:
        """
        Classify a period based on Calmar ratio and max drawdown.

        Args:
            calmar_ratio: Calmar ratio for the period
            max_dd_pct: Maximum drawdown percentage for the period

        Returns:
            Classification: "good", "indeterminate", or "bad"
        """
        cfg = self.config

        if calmar_ratio > cfg.calmar_good_threshold:
            if max_dd_pct <= cfg.max_dd_good_threshold:
                return "good"
            else:
                return "indeterminate"
        else:
            return "bad"

    def generate_monthly_range(self, start_date: datetime,
                                end_date: datetime) -> List[Tuple[int, int]]:
        """
        Generate list of (year, month) tuples for the date range.

        Args:
            start_date: Start date
            end_date: End date (extended to cover the full month)

        Returns:
            List of (year, month) tuples
        """
        months = []

        # Start from the first day of the start month
        current = datetime(start_date.year, start_date.month, 1)

        # End at the first day of the month after end_date
        end = datetime(end_date.year, end_date.month, 1) + relativedelta(months=1)

        while current < end:
            months.append((current.year, current.month))
            current += relativedelta(months=1)

        return months

    def generate_fundamental_features(self, trades: pd.DataFrame,
                                       symbol: str,
                                       initial_capital: float = 10000.0) -> pd.DataFrame:
        """
        Generate fundamental features DataFrame for a security.

        Args:
            trades: DataFrame of trades for the security
            symbol: Security symbol
            initial_capital: Initial capital for metric calculations

        Returns:
            DataFrame with year, month, and period_GB_flag
        """
        if trades.empty:
            return pd.DataFrame()

        # Get date range
        start_date = trades['entry_date'].min()
        end_date = trades['exit_date'].max()

        # Generate monthly range
        months = self.generate_monthly_range(start_date, end_date)

        # Calculate yearly metrics
        yearly_metrics = self.calculate_yearly_metrics(trades, initial_capital)

        # Create yearly classification lookup
        yearly_flags = {}
        for _, row in yearly_metrics.iterrows():
            year = int(row['year'])
            flag = self.classify_period(row['calmar_ratio'], row['max_drawdown_pct'])
            yearly_flags[year] = {
                'period_GB_flag': flag,
                'calmar_ratio': row['calmar_ratio'],
                'max_drawdown_pct': row['max_drawdown_pct'],
                'num_trades': row['num_trades'],
                'return_pct': row['return_pct'],
            }

        # Build the features DataFrame
        features = []
        for year, month in months:
            row = {
                'symbol': symbol,
                'year': year,
                'month': month,
            }

            # Add yearly metrics (same for all months in the year)
            if year in yearly_flags:
                row.update(yearly_flags[year])
            else:
                # Year with no trades
                row.update({
                    'period_GB_flag': 'no_trades',
                    'calmar_ratio': np.nan,
                    'max_drawdown_pct': np.nan,
                    'num_trades': 0,
                    'return_pct': np.nan,
                })

            features.append(row)

        return pd.DataFrame(features)

    def process_trade_log_file(self, file_path: Path,
                                symbol: str,
                                initial_capital: float = 10000.0) -> Optional[pd.DataFrame]:
        """
        Process a single trade log file and generate fundamental features.

        Args:
            file_path: Path to the trade log CSV
            symbol: Security symbol
            initial_capital: Initial capital for metric calculations

        Returns:
            DataFrame with fundamental features or None if failed
        """
        trades = self.load_trade_log(file_path)
        if trades is None:
            return None

        return self.generate_fundamental_features(trades, symbol, initial_capital)

    def process_trade_logs_folder(self, folder_path: Path,
                                   output_dir: Path,
                                   initial_capital: float = 10000.0) -> Dict[str, Path]:
        """
        Process all trade logs in a folder and save fundamental features.

        Args:
            folder_path: Path to folder containing trade log CSVs
            output_dir: Directory to save output files
            initial_capital: Initial capital for metric calculations

        Returns:
            Dictionary mapping symbol to output file path
        """
        folder_path = Path(folder_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Find all trade log CSV files
        trade_log_files = list(folder_path.glob("*_trades.csv"))

        for file_path in trade_log_files:
            # Extract symbol from filename
            # Expected format: StrategyName_Symbol_trades.csv
            filename = file_path.stem  # Remove .csv
            parts = filename.rsplit('_', 2)  # Split from right

            if len(parts) >= 2 and parts[-1] == 'trades':
                symbol = parts[-2]
            else:
                print(f"Warning: Could not extract symbol from {filename}")
                continue

            print(f"Processing {symbol}...")

            features = self.process_trade_log_file(file_path, symbol, initial_capital)

            if features is not None and not features.empty:
                output_file = output_dir / f"{symbol}_fundamental_features.csv"
                features.to_csv(output_file, index=False)
                results[symbol] = output_file
                print(f"  Saved {output_file}")
            else:
                print(f"  No features generated for {symbol}")

        return results

    def generate_summary(self, fundamental_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate a summary of all fundamental features across securities.

        Args:
            fundamental_features: Dictionary mapping symbol to features DataFrame

        Returns:
            Summary DataFrame
        """
        summary = []

        for symbol, features in fundamental_features.items():
            if features.empty:
                continue

            # Count periods by flag
            flag_counts = features['period_GB_flag'].value_counts().to_dict()

            # Get yearly metrics
            yearly_data = features.drop_duplicates(subset=['year'])

            for _, row in yearly_data.iterrows():
                if row['num_trades'] > 0:
                    summary.append({
                        'symbol': symbol,
                        'year': row['year'],
                        'period_GB_flag': row['period_GB_flag'],
                        'calmar_ratio': row['calmar_ratio'],
                        'max_drawdown_pct': row['max_drawdown_pct'],
                        'num_trades': row['num_trades'],
                        'return_pct': row['return_pct'],
                    })

        return pd.DataFrame(summary)
