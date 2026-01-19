"""
Metrics Calculator for Rule Tester.

This module calculates before/after performance metrics for trade filtering.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..Core.stable_metrics import StableMetricsCalculator, BARS_PER_YEAR


class RuleMetricsCalculator:
    """
    Calculator for comparing metrics before and after rule application.

    Calculates:
    - Total trades
    - Win rate
    - RAR%
    - R-Cubed
    - Cumulative P/L over time
    - Cumulative RAR% over time
    """

    # Minimum trades for statistical significance warning
    MIN_TRADES_FOR_SIGNIFICANCE = 30

    def __init__(self, initial_capital: float = 100000):
        """
        Initialize RuleMetricsCalculator.

        Args:
            initial_capital: Starting capital for equity calculations
        """
        self.initial_capital = initial_capital

    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all metrics for a set of trades.

        Args:
            trades_df: DataFrame with trade data

        Returns:
            Dict with all calculated metrics
        """
        if trades_df is None or len(trades_df) == 0:
            return self._empty_metrics()

        # Sort by exit date
        trades = trades_df.copy()
        if 'exit_date' in trades.columns:
            trades['exit_date'] = pd.to_datetime(trades['exit_date'])
            trades = trades.sort_values('exit_date')

        # Basic metrics
        total_trades = len(trades)
        winning_trades = (trades['pl'] > 0).sum() if 'pl' in trades.columns else 0
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_pl = trades['pl'].sum() if 'pl' in trades.columns else 0
        avg_pl = trades['pl'].mean() if 'pl' in trades.columns else 0
        avg_pl_pct = trades['pl_pct'].mean() if 'pl_pct' in trades.columns else 0

        # Build equity curve
        equity_curve = self._build_equity_curve(trades)

        # Calculate stable metrics
        rar_pct = 0.0
        r_squared = 0.0
        r_cubed = 0.0

        if equity_curve is not None and len(equity_curve) > 1:
            stable_result = StableMetricsCalculator.calculate_all(
                equity_curve,
                equity_column='equity',
                date_column='date'
            )
            rar_pct = stable_result.rar_pct
            r_squared = stable_result.r_squared
            r_cubed = stable_result.r_cubed

        # Calculate cumulative P/L
        cumulative_pl = self.calculate_cumulative_pl(trades)

        # Calculate cumulative RAR%
        cumulative_rar = self.calculate_cumulative_rar(trades)

        # Sample size warning
        sample_warning = ""
        if total_trades < self.MIN_TRADES_FOR_SIGNIFICANCE:
            sample_warning = f"Sample size ({total_trades}) is below {self.MIN_TRADES_FOR_SIGNIFICANCE}. Results may not be statistically significant."

        return {
            'total_trades': total_trades,
            'winning_trades': int(winning_trades),
            'losing_trades': total_trades - int(winning_trades),
            'win_rate': win_rate,
            'total_pl': total_pl,
            'avg_pl': avg_pl,
            'avg_pl_pct': avg_pl_pct,
            'rar_pct': rar_pct,
            'r_squared': r_squared,
            'r_cubed': r_cubed,
            'cumulative_pl': cumulative_pl,
            'cumulative_rar': cumulative_rar,
            'sample_warning': sample_warning,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dict."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pl': 0.0,
            'avg_pl': 0.0,
            'avg_pl_pct': 0.0,
            'rar_pct': 0.0,
            'r_squared': 0.0,
            'r_cubed': 0.0,
            'cumulative_pl': pd.DataFrame(),
            'cumulative_rar': pd.DataFrame(),
            'sample_warning': 'No trades to analyze.',
        }

    def _build_equity_curve(self, trades: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Build equity curve from trades.

        Args:
            trades: DataFrame with trade data (must have pl and exit_date columns)

        Returns:
            DataFrame with date and equity columns
        """
        if 'pl' not in trades.columns or 'exit_date' not in trades.columns:
            return None

        if len(trades) == 0:
            return None

        # Sort by exit date
        trades_sorted = trades.sort_values('exit_date')

        # Build equity curve
        equity_values = [self.initial_capital]
        dates = [trades_sorted['exit_date'].min() - pd.Timedelta(days=1)]

        current_equity = self.initial_capital
        for _, trade in trades_sorted.iterrows():
            current_equity += trade['pl']
            equity_values.append(current_equity)
            dates.append(trade['exit_date'])

        return pd.DataFrame({
            'date': dates,
            'equity': equity_values
        })

    def calculate_cumulative_pl(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative P/L over time.

        Args:
            trades: DataFrame with trade data

        Returns:
            DataFrame with date and cumulative_pl columns
        """
        if 'pl' not in trades.columns or 'exit_date' not in trades.columns:
            return pd.DataFrame(columns=['date', 'cumulative_pl'])

        if len(trades) == 0:
            return pd.DataFrame(columns=['date', 'cumulative_pl'])

        # Sort by exit date
        trades_sorted = trades.sort_values('exit_date')

        # Calculate cumulative sum
        cumulative = trades_sorted['pl'].cumsum()

        return pd.DataFrame({
            'date': trades_sorted['exit_date'].values,
            'cumulative_pl': cumulative.values
        })

    def calculate_cumulative_rar(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative RAR% at each trade close.

        RAR% is recalculated using all trades up to that point.

        Args:
            trades: DataFrame with trade data

        Returns:
            DataFrame with date and cumulative_rar columns
        """
        if 'pl' not in trades.columns or 'exit_date' not in trades.columns:
            return pd.DataFrame(columns=['date', 'cumulative_rar'])

        if len(trades) < 2:
            return pd.DataFrame(columns=['date', 'cumulative_rar'])

        # Sort by exit date
        trades_sorted = trades.sort_values('exit_date').reset_index(drop=True)

        dates = []
        rar_values = []

        # Calculate RAR% incrementally
        for i in range(1, len(trades_sorted)):
            # Use trades up to this point
            subset = trades_sorted.iloc[:i + 1]

            # Build equity curve for subset
            equity_curve = self._build_equity_curve(subset)

            if equity_curve is not None and len(equity_curve) > 1:
                stable_result = StableMetricsCalculator.calculate_all(
                    equity_curve,
                    equity_column='equity',
                    date_column='date'
                )
                rar_values.append(stable_result.rar_pct)
            else:
                rar_values.append(0.0)

            dates.append(trades_sorted['exit_date'].iloc[i])

        return pd.DataFrame({
            'date': dates,
            'cumulative_rar': rar_values
        })

    def compare_metrics(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare before/after metrics with change calculation.

        Args:
            before: Metrics before filtering
            after: Metrics after filtering

        Returns:
            Dict with comparison data
        """
        comparison = {}

        # Metrics to compare
        metrics_config = [
            ('total_trades', 'Total Trades', False),  # name, label, is_percentage
            ('win_rate', 'Win Rate (%)', True),
            ('total_pl', 'Total P/L', False),
            ('avg_pl', 'Avg P/L', False),
            ('avg_pl_pct', 'Avg P/L (%)', True),
            ('rar_pct', 'RAR%', True),
            ('r_squared', 'R-Squared', False),
            ('r_cubed', 'R-Cubed', False),
        ]

        for key, label, is_pct in metrics_config:
            before_val = before.get(key, 0) or 0
            after_val = after.get(key, 0) or 0

            change = after_val - before_val

            if is_pct:
                change_str = f"{change:+.2f}%"
            elif isinstance(before_val, (int, np.integer)):
                change_str = f"{int(change):+d}"
            else:
                change_str = f"{change:+.2f}"

            # Calculate percentage change
            if before_val != 0:
                pct_change = (after_val - before_val) / abs(before_val) * 100
                pct_change_str = f"{pct_change:+.1f}%"
            else:
                pct_change_str = "N/A"

            comparison[key] = {
                'label': label,
                'before': before_val,
                'after': after_val,
                'change': change,
                'change_str': change_str,
                'pct_change_str': pct_change_str,
            }

        # Include warnings
        comparison['before_warning'] = before.get('sample_warning', '')
        comparison['after_warning'] = after.get('sample_warning', '')

        return comparison

    def get_comparison_table_data(
        self,
        comparison: Dict[str, Any]
    ) -> List[Tuple[str, str, str, str]]:
        """
        Get comparison data formatted for table display.

        Args:
            comparison: Comparison dict from compare_metrics

        Returns:
            List of (label, before, after, change) tuples
        """
        rows = []

        # Order of metrics to display
        metric_order = [
            'total_trades', 'win_rate', 'total_pl', 'avg_pl',
            'avg_pl_pct', 'rar_pct', 'r_squared', 'r_cubed'
        ]

        for key in metric_order:
            if key in comparison:
                data = comparison[key]
                label = data['label']

                # Format before/after values
                before_val = data['before']
                after_val = data['after']

                if key in ['win_rate', 'rar_pct', 'avg_pl_pct']:
                    before_str = f"{before_val:.2f}%"
                    after_str = f"{after_val:.2f}%"
                elif key == 'r_squared':
                    before_str = f"{before_val:.4f}"
                    after_str = f"{after_val:.4f}"
                elif key == 'r_cubed':
                    before_str = f"{before_val:.2f}"
                    after_str = f"{after_val:.2f}"
                elif key == 'total_trades':
                    before_str = f"{int(before_val)}"
                    after_str = f"{int(after_val)}"
                elif key in ['total_pl', 'avg_pl']:
                    before_str = f"${before_val:,.2f}"
                    after_str = f"${after_val:,.2f}"
                else:
                    before_str = f"{before_val:.2f}"
                    after_str = f"{after_val:.2f}"

                change_str = data['change_str']
                if key not in ['win_rate', 'rar_pct', 'avg_pl_pct', 'r_squared']:
                    change_str += f" ({data['pct_change_str']})"

                rows.append((label, before_str, after_str, change_str))

        return rows


def export_filtered_trades(
    trades: pd.DataFrame,
    output_path: str,
    rules_description: str = ""
) -> str:
    """
    Export filtered trades to CSV.

    Args:
        trades: Filtered trades DataFrame
        output_path: Path to save CSV
        rules_description: Optional description of rules applied

    Returns:
        Path to saved file
    """
    output = trades.copy()

    # Add metadata
    if rules_description:
        # Add as a comment at the top would require special handling
        # Instead, we'll just save the trades
        pass

    output.to_csv(output_path, index=False)
    return output_path


def export_comparison_report(
    comparison: Dict[str, Any],
    rules: List,
    output_path: str
) -> str:
    """
    Export comparison report to CSV.

    Args:
        comparison: Comparison dict from compare_metrics
        rules: List of rules applied
        output_path: Path to save CSV

    Returns:
        Path to saved file
    """
    rows = []

    # Header section
    rows.append(['Rule Tester Comparison Report', ''])
    rows.append(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    rows.append(['', ''])

    # Rules section
    rows.append(['Rules Applied', ''])
    for i, rule in enumerate(rules, 1):
        rows.append([f'  Rule {i}', str(rule)])
    rows.append(['', ''])

    # Metrics section
    rows.append(['Metric', 'Before', 'After', 'Change'])

    metric_order = [
        'total_trades', 'win_rate', 'total_pl', 'avg_pl',
        'avg_pl_pct', 'rar_pct', 'r_squared', 'r_cubed'
    ]

    for key in metric_order:
        if key in comparison:
            data = comparison[key]
            rows.append([
                data['label'],
                str(data['before']),
                str(data['after']),
                data['change_str']
            ])

    # Warnings section
    rows.append(['', ''])
    if comparison.get('before_warning'):
        rows.append(['Warning (Before)', comparison['before_warning']])
    if comparison.get('after_warning'):
        rows.append(['Warning (After)', comparison['after_warning']])

    # Write to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, header=False)

    return output_path


def export_exit_comparison_csv(
    original_trades: pd.DataFrame,
    modified_trades: pd.DataFrame,
    price_data_dict: Dict[str, pd.DataFrame],
    output_path: str,
    rules_description: str = "",
    indicator_columns: Optional[List[str]] = None
) -> str:
    """
    Export a comparison CSV showing original vs new exits for validation.

    Creates a side-by-side comparison with:
    - Trade identification (id, symbol)
    - Entry details (date, price)
    - Original exit (date, price, pnl)
    - New exit (date, price, pnl)
    - Delta columns (days difference, pnl difference)
    - Indicator values at both exit points

    Args:
        original_trades: Original trades DataFrame before rule application
        modified_trades: Modified trades DataFrame after rule application
        price_data_dict: Dict of ticker -> price DataFrame for indicator lookup
        output_path: Path to save CSV
        rules_description: Description of rules applied
        indicator_columns: List of indicator columns to include (auto-detected if None)

    Returns:
        Path to saved file
    """
    # Ensure we have the same trades in both DataFrames
    if len(original_trades) != len(modified_trades):
        raise ValueError("Original and modified trades must have same length")

    # Auto-detect indicator columns if not provided
    if indicator_columns is None:
        indicator_columns = _detect_indicator_columns(price_data_dict)

    # Build comparison DataFrame
    rows = []

    for i in range(len(original_trades)):
        orig = original_trades.iloc[i]
        mod = modified_trades.iloc[i]

        # Get basic trade info
        trade_id = orig.get('trade_id', orig.get('id', i))
        symbol = str(orig.get('symbol', '')).upper()
        entry_date = pd.to_datetime(orig.get('entry_date'))
        entry_price = orig.get('entry_price', 0)

        # Original exit info
        orig_exit_date = pd.to_datetime(orig.get('exit_date'))
        orig_exit_price = orig.get('exit_price', 0)
        orig_pl = orig.get('pl', 0)
        orig_pl_pct = orig.get('pl_pct', 0)

        # New exit info
        new_exit_date = pd.to_datetime(mod.get('exit_date'))
        new_exit_price = mod.get('exit_price', 0)
        new_pl = mod.get('pl', 0)
        new_pl_pct = mod.get('pl_pct', 0)

        # Calculate deltas
        exit_changed = False
        days_diff = 0
        if pd.notna(orig_exit_date) and pd.notna(new_exit_date):
            days_diff = (new_exit_date - orig_exit_date).days
            exit_changed = days_diff != 0

        pl_diff = new_pl - orig_pl if pd.notna(new_pl) and pd.notna(orig_pl) else 0
        pl_pct_diff = new_pl_pct - orig_pl_pct if pd.notna(new_pl_pct) and pd.notna(orig_pl_pct) else 0

        row = {
            'trade_id': trade_id,
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'orig_exit_date': orig_exit_date,
            'orig_exit_price': orig_exit_price,
            'orig_pl': orig_pl,
            'orig_pl_pct': orig_pl_pct,
            'new_exit_date': new_exit_date,
            'new_exit_price': new_exit_price,
            'new_pl': new_pl,
            'new_pl_pct': new_pl_pct,
            'exit_changed': exit_changed,
            'exit_days_diff': days_diff,
            'pl_diff': pl_diff,
            'pl_pct_diff': pl_pct_diff,
        }

        # Get indicator values at original and new exit dates
        pdf = price_data_dict.get(symbol)
        if pdf is not None:
            # Indicators at original exit
            orig_indicators = _get_indicators_at_date(pdf, orig_exit_date, indicator_columns)
            for col, val in orig_indicators.items():
                row[f'orig_{col}'] = val

            # Indicators at new exit
            new_indicators = _get_indicators_at_date(pdf, new_exit_date, indicator_columns)
            for col, val in new_indicators.items():
                row[f'new_{col}'] = val

            # Indicator deltas (for numeric columns)
            for col in indicator_columns:
                orig_val = orig_indicators.get(col)
                new_val = new_indicators.get(col)
                if isinstance(orig_val, (int, float)) and isinstance(new_val, (int, float)):
                    if pd.notna(orig_val) and pd.notna(new_val):
                        row[f'delta_{col}'] = new_val - orig_val

        rows.append(row)

    # Create DataFrame and sort columns logically
    df = pd.DataFrame(rows)

    # Define column order
    base_cols = [
        'trade_id', 'symbol', 'entry_date', 'entry_price',
        'orig_exit_date', 'orig_exit_price', 'orig_pl', 'orig_pl_pct',
        'new_exit_date', 'new_exit_price', 'new_pl', 'new_pl_pct',
        'exit_changed', 'exit_days_diff', 'pl_diff', 'pl_pct_diff'
    ]

    # Get indicator columns in order: orig_, new_, delta_
    indicator_cols = []
    for col in indicator_columns:
        if f'orig_{col}' in df.columns:
            indicator_cols.append(f'orig_{col}')
        if f'new_{col}' in df.columns:
            indicator_cols.append(f'new_{col}')
        if f'delta_{col}' in df.columns:
            indicator_cols.append(f'delta_{col}')

    # Reorder columns
    final_cols = [c for c in base_cols if c in df.columns] + indicator_cols
    remaining_cols = [c for c in df.columns if c not in final_cols]
    df = df[final_cols + remaining_cols]

    # Save to CSV
    df.to_csv(output_path, index=False)

    return output_path


def _detect_indicator_columns(price_data_dict: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Auto-detect indicator columns from price data.

    Returns columns that are likely indicators (not standard OHLCV columns).
    """
    excluded = {'date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'adj_close'}

    if not price_data_dict:
        return []

    # Get columns from first price DataFrame
    first_df = next(iter(price_data_dict.values()))
    all_cols = set(first_df.columns)

    # Filter to likely indicator columns
    indicator_cols = [c for c in all_cols if c.lower() not in excluded]

    # Prioritize common indicator patterns
    priority_patterns = ['ema', 'sma', 'atr', 'rsi', 'macd', 'bb_', 'alpha', 'momentum', 'trend']
    priority_cols = []
    other_cols = []

    for col in indicator_cols:
        col_lower = col.lower()
        if any(p in col_lower for p in priority_patterns):
            priority_cols.append(col)
        else:
            other_cols.append(col)

    # Return prioritized list (limit to avoid huge CSVs)
    result = sorted(priority_cols) + sorted(other_cols)
    return result[:15]  # Limit to 15 indicators max


def _get_indicators_at_date(
    pdf: pd.DataFrame,
    target_date: pd.Timestamp,
    indicator_columns: List[str]
) -> Dict[str, Any]:
    """
    Get indicator values at a specific date.

    Args:
        pdf: Price DataFrame with date column
        target_date: Date to look up
        indicator_columns: List of indicator columns to retrieve

    Returns:
        Dict of column -> value
    """
    result = {}

    if pdf is None or target_date is None or pd.isna(target_date):
        return {col: np.nan for col in indicator_columns}

    # Find the row for this date (or closest prior date)
    target_date = pd.to_datetime(target_date)
    matching = pdf[pdf['date'] <= target_date]

    if len(matching) == 0:
        return {col: np.nan for col in indicator_columns}

    row = matching.iloc[-1]

    for col in indicator_columns:
        if col in row.index:
            result[col] = row[col]
        else:
            result[col] = np.nan

    return result
