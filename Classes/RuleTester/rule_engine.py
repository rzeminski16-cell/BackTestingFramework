"""
Rule Engine for testing rule effects on strategy performance.

This module provides rule definition and filtering logic for trade logs.
Supports both entry and exit rules with configurable lookback periods.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
from enum import Enum


class RuleMode(Enum):
    """Mode for rule evaluation - entry or exit based."""
    ENTRY = "entry"
    EXIT = "exit"


@dataclass
class Rule:
    """
    Represents a single filtering rule for trades.

    Attributes:
        feature: Name of the feature/column to check
        operator: Comparison operator ('>', '<', '>=', '<=', 'between', '==', '!=', 'in')
        value: Value(s) to compare against (single value, tuple for between, list for in)
        lookback_bars: Number of bars before the reference date to check (0 = reference date only)
        is_discrete: Whether the feature is discrete (categorical)
    """
    feature: str
    operator: str  # '>', '<', '>=', '<=', 'between', '==', '!=', 'in'
    value: Any  # single value, tuple for between, list for in
    lookback_bars: int = 0  # N=0 means check reference date only, N>0 means check N bars before
    is_discrete: bool = False

    def __str__(self) -> str:
        """Human-readable representation of the rule."""
        lookback_str = ""
        if self.lookback_bars > 0:
            lookback_str = f" (within {self.lookback_bars} bars before)"

        if self.operator == 'between':
            return f"{self.feature} between {self.value[0]} and {self.value[1]}{lookback_str}"
        elif self.operator == 'in':
            values_str = ', '.join(str(v) for v in self.value[:5])
            if len(self.value) > 5:
                values_str += f", ... ({len(self.value)} total)"
            return f"{self.feature} in [{values_str}]{lookback_str}"
        else:
            return f"{self.feature} {self.operator} {self.value}{lookback_str}"


class RuleEngine:
    """
    Engine for applying filtering rules to trade logs.

    Supports:
    - Entry rules: evaluated at trade entry date
    - Exit rules: evaluated at trade exit date
    - Lookback periods: check if condition was true within N bars before reference date
    - Multiple rules combined with AND logic

    Data leakage prevention:
    - Only uses data on or before the reference date
    - Never looks at future data relative to entry/exit
    """

    # Threshold for discrete vs continuous detection
    DISCRETE_THRESHOLD = 10

    # Excluded columns from feature selection (standard price columns)
    EXCLUDED_COLUMNS = {'date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'adj_close'}

    def __init__(
        self,
        trades_df: pd.DataFrame,
        price_data_dict: Dict[str, pd.DataFrame],
        mode: RuleMode = RuleMode.ENTRY
    ):
        """
        Initialize RuleEngine with trades and price data.

        Args:
            trades_df: DataFrame with trade logs (must have entry_date, exit_date, symbol columns)
            price_data_dict: Dict of ticker -> price DataFrame (must have date column)
            mode: RuleMode.ENTRY or RuleMode.EXIT - determines which date to use as reference
        """
        self.trades_df = trades_df.copy()
        self.price_data_dict = price_data_dict
        self.mode = mode

        # Ensure dates are datetime
        if 'entry_date' in self.trades_df.columns:
            self.trades_df['entry_date'] = pd.to_datetime(self.trades_df['entry_date'])
        if 'exit_date' in self.trades_df.columns:
            self.trades_df['exit_date'] = pd.to_datetime(self.trades_df['exit_date'])

        for ticker, pdf in self.price_data_dict.items():
            if 'date' in pdf.columns:
                self.price_data_dict[ticker] = pdf.copy()
                self.price_data_dict[ticker]['date'] = pd.to_datetime(pdf['date'])

        # Cache for feature values (keyed by feature + mode)
        self._feature_cache: Dict[str, pd.Series] = {}

    def set_mode(self, mode: RuleMode) -> None:
        """
        Change the rule evaluation mode.

        Args:
            mode: RuleMode.ENTRY or RuleMode.EXIT
        """
        if mode != self.mode:
            self.mode = mode
            self._feature_cache.clear()  # Clear cache when mode changes

    def get_reference_date_column(self) -> str:
        """Get the column name for the current mode's reference date."""
        return 'entry_date' if self.mode == RuleMode.ENTRY else 'exit_date'

    def get_available_features(self) -> List[str]:
        """
        Get list of available features from price data.

        Returns:
            List of column names that can be used as features.
        """
        if not self.price_data_dict:
            return []

        # Get all columns from first price data DataFrame
        first_ticker = next(iter(self.price_data_dict))
        all_columns = set(self.price_data_dict[first_ticker].columns)

        # Check which columns exist in all price data
        for pdf in self.price_data_dict.values():
            all_columns &= set(pdf.columns)

        # Exclude standard columns
        features = sorted(all_columns - self.EXCLUDED_COLUMNS)

        return features

    def detect_feature_type(self, feature: str) -> str:
        """
        Detect if a feature is continuous or discrete.

        A feature is considered discrete if it has <= DISCRETE_THRESHOLD unique values.

        Args:
            feature: Feature name to check

        Returns:
            'continuous' or 'discrete'
        """
        values = self.get_feature_values_at_reference(feature)

        if values is None or len(values) == 0:
            return 'continuous'

        # Drop NaN and count unique values
        unique_count = values.dropna().nunique()

        if unique_count <= self.DISCRETE_THRESHOLD:
            return 'discrete'
        return 'continuous'

    def get_feature_values_at_reference(self, feature: str) -> Optional[pd.Series]:
        """
        Get feature values at each trade's reference date (entry or exit based on mode).

        Args:
            feature: Feature name to retrieve

        Returns:
            Series with feature values indexed like trades_df, or None if unavailable
        """
        cache_key = f"{feature}_{self.mode.value}"

        # Check cache
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        if self.trades_df is None or len(self.trades_df) == 0:
            return None

        date_col = self.get_reference_date_column()
        values = []

        for idx, trade in self.trades_df.iterrows():
            ref_date = trade.get(date_col)
            symbol = str(trade.get('symbol', '')).upper()

            # Get price data for this symbol
            pdf = self.price_data_dict.get(symbol)

            if pdf is None or ref_date is None or pd.isna(ref_date):
                values.append(np.nan)
                continue

            # Find the row for this date (or closest prior date to avoid lookahead)
            ref_date = pd.to_datetime(ref_date)
            matching = pdf[pdf['date'] <= ref_date]

            if len(matching) == 0:
                values.append(np.nan)
                continue

            # Get the most recent row on or before reference date
            row = matching.iloc[-1]

            # Get feature value
            if feature in row.index:
                values.append(row[feature])
            else:
                values.append(np.nan)

        result = pd.Series(values, index=self.trades_df.index)
        self._feature_cache[cache_key] = result
        return result

    def get_feature_statistics(self, feature: str) -> Dict[str, Any]:
        """
        Get statistics for a feature.

        Args:
            feature: Feature name

        Returns:
            Dict with min, max, mean, std, unique values (for discrete), etc.
        """
        values = self.get_feature_values_at_reference(feature)

        if values is None or len(values) == 0:
            return {}

        values_clean = values.dropna()

        if len(values_clean) == 0:
            return {'error': 'All values are NaN'}

        feature_type = self.detect_feature_type(feature)

        stats = {
            'feature_type': feature_type,
            'count': len(values_clean),
            'missing': len(values) - len(values_clean),
        }

        if feature_type == 'continuous':
            stats.update({
                'min': float(values_clean.min()),
                'max': float(values_clean.max()),
                'mean': float(values_clean.mean()),
                'std': float(values_clean.std()),
                'median': float(values_clean.median()),
                'q25': float(values_clean.quantile(0.25)),
                'q75': float(values_clean.quantile(0.75)),
            })
        else:
            unique_values = sorted(values_clean.unique().tolist())
            value_counts = values_clean.value_counts().to_dict()
            stats.update({
                'unique_values': unique_values,
                'value_counts': value_counts,
            })

        return stats

    def _check_condition(self, value: Any, rule: Rule) -> bool:
        """
        Check if a single value satisfies the rule condition.

        Args:
            value: Value to check
            rule: Rule with operator and comparison value

        Returns:
            True if condition is satisfied
        """
        if pd.isna(value):
            return False

        if rule.operator == '>':
            return value > rule.value
        elif rule.operator == '<':
            return value < rule.value
        elif rule.operator == '>=':
            return value >= rule.value
        elif rule.operator == '<=':
            return value <= rule.value
        elif rule.operator == 'between':
            low, high = rule.value
            return low <= value <= high
        elif rule.operator == '==':
            return value == rule.value
        elif rule.operator == '!=':
            return value != rule.value
        elif rule.operator == 'in':
            return value in rule.value
        else:
            raise ValueError(f"Unknown operator: {rule.operator}")

    def evaluate_rule(self, rule: Rule) -> pd.Series:
        """
        Evaluate a single rule against all trades.

        Supports lookback: if rule.lookback_bars > 0, checks if the condition
        was true at any point in the N bars BEFORE the reference date.
        If lookback_bars = 0, only checks the reference date.

        Args:
            rule: Rule to evaluate

        Returns:
            Boolean Series indicating which trades pass the rule
        """
        date_col = self.get_reference_date_column()
        results = []

        for idx, trade in self.trades_df.iterrows():
            ref_date = trade.get(date_col)
            symbol = str(trade.get('symbol', '')).upper()

            # Get price data for this symbol
            pdf = self.price_data_dict.get(symbol)

            if pdf is None or ref_date is None or pd.isna(ref_date):
                results.append(False)
                continue

            ref_date = pd.to_datetime(ref_date)

            # Get data up to (and including for N=0, or excluding for N>0) reference date
            if rule.lookback_bars == 0:
                # N=0: Check only the reference date
                # Get data on or before reference date
                available = pdf[pdf['date'] <= ref_date]
                if len(available) == 0:
                    results.append(False)
                    continue

                # Get the reference date value (most recent on or before ref_date)
                row = available.iloc[-1]
                value = row.get(rule.feature, np.nan)
                results.append(self._check_condition(value, rule))

            else:
                # N>0: Check if condition was true in any of the N bars BEFORE reference date
                # This means we look at bars strictly before ref_date
                available = pdf[pdf['date'] < ref_date]

                if len(available) == 0:
                    results.append(False)
                    continue

                # Get the last N bars before reference date
                lookback_data = available.tail(rule.lookback_bars)

                if rule.feature not in lookback_data.columns:
                    results.append(False)
                    continue

                # Check if condition is true for ANY bar in the lookback period
                values = lookback_data[rule.feature]
                condition_met = False
                for val in values:
                    if self._check_condition(val, rule):
                        condition_met = True
                        break

                results.append(condition_met)

        return pd.Series(results, index=self.trades_df.index)

    def apply_rules(self, rules: List[Rule]) -> pd.DataFrame:
        """
        Filter trades based on rules (AND logic).

        Args:
            rules: List of rules to apply

        Returns:
            Filtered DataFrame containing only trades that pass ALL rules
        """
        if not rules:
            return self.trades_df.copy()

        # Start with all True
        combined_mask = pd.Series([True] * len(self.trades_df), index=self.trades_df.index)

        # Apply each rule with AND logic
        for rule in rules:
            rule_mask = self.evaluate_rule(rule)
            combined_mask &= rule_mask

        return self.trades_df[combined_mask].copy()

    def count_passing_trades(self, rules: List[Rule]) -> int:
        """
        Count trades that pass all rules.

        Args:
            rules: List of rules to apply

        Returns:
            Number of trades passing all rules
        """
        filtered = self.apply_rules(rules)
        return len(filtered)

    def get_rule_preview(self, rules: List[Rule]) -> Dict[str, Any]:
        """
        Get a preview of rule application.

        Args:
            rules: List of rules to preview

        Returns:
            Dict with total_trades, passing_trades, pass_rate, and per-rule counts
        """
        total = len(self.trades_df)
        passing = self.count_passing_trades(rules)

        result = {
            'total_trades': total,
            'passing_trades': passing,
            'filtered_trades': total - passing,
            'pass_rate': passing / total * 100 if total > 0 else 0,
            'mode': self.mode.value,
        }

        # Per-rule breakdown
        rule_counts = []
        for rule in rules:
            mask = self.evaluate_rule(rule)
            count = mask.sum()
            rule_counts.append({
                'rule': str(rule),
                'passing': int(count),
                'pass_rate': count / total * 100 if total > 0 else 0
            })

        result['per_rule'] = rule_counts

        return result

    def clear_cache(self) -> None:
        """Clear the feature value cache."""
        self._feature_cache.clear()

    # Backward compatibility alias
    def get_feature_values_at_entry(self, feature: str) -> Optional[pd.Series]:
        """Alias for get_feature_values_at_reference for backward compatibility."""
        return self.get_feature_values_at_reference(feature)


def extract_ticker_from_filename(filename: str) -> Optional[str]:
    """
    Extract ticker symbol from a trade log filename.

    Assumes format like: AAPL_trades.csv, AAPL_backtest.csv, etc.

    Args:
        filename: Filename to parse

    Returns:
        Ticker symbol or None if not found
    """
    stem = Path(filename).stem
    parts = stem.split('_')

    if parts:
        # First part is typically the ticker
        ticker = parts[0].upper()
        # Validate it looks like a ticker (letters only, reasonable length)
        if ticker.isalpha() and 1 <= len(ticker) <= 5:
            return ticker

    return None


def load_price_data_for_tickers(
    tickers: List[str],
    base_path: str = "raw_data/daily"
) -> Dict[str, pd.DataFrame]:
    """
    Load price data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        base_path: Base path for daily price data files

    Returns:
        Dict of ticker -> DataFrame
    """
    result = {}
    base = Path(base_path)

    for ticker in tickers:
        # Try common naming patterns
        possible_files = [
            base / f"{ticker}_daily.csv",
            base / f"{ticker.upper()}_daily.csv",
            base / f"{ticker.lower()}_daily.csv",
            base / f"{ticker}.csv",
            base / f"{ticker.upper()}.csv",
        ]

        for file_path in possible_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Normalize column names
                    df.columns = [c.lower().strip() for c in df.columns]
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.sort_values('date')
                    result[ticker.upper()] = df
                    break
                except Exception:
                    pass

    return result
