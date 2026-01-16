"""
Rule Engine for testing rule effects on strategy performance.

This module provides rule definition and filtering logic for trade logs.
Supports both entry and exit rules with configurable lookback periods.
Supports comparing features to static values OR to other features.
Supports loading original strategy exit rules for proper AND logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np
from pathlib import Path
from enum import Enum

if TYPE_CHECKING:
    from .strategy_exit_rules import StrategyExitConfig, StrategyExitRuleEvaluator


class RuleMode(Enum):
    """Mode for rule evaluation - entry or exit based."""
    ENTRY = "entry"
    EXIT = "exit"


class CompareType(Enum):
    """Type of comparison value."""
    VALUE = "value"  # Compare to a static numeric value
    FEATURE = "feature"  # Compare to another feature


@dataclass
class Rule:
    """
    Represents a single filtering rule for trades.

    Attributes:
        feature: Name of the feature/column to check
        operator: Comparison operator ('>', '<', '>=', '<=', 'between', '==', '!=', 'in')
        value: Value(s) to compare against (single value, tuple for between, list for in)
        compare_type: Whether comparing to a VALUE or another FEATURE
        compare_feature: If compare_type is FEATURE, the name of the feature to compare to
        lookback_bars: Number of bars before the reference date to check (0 = reference date only)
        is_discrete: Whether the feature is discrete (categorical)
    """
    feature: str
    operator: str  # '>', '<', '>=', '<=', 'between', '==', '!=', 'in'
    value: Any  # single value, tuple for between, list for in
    compare_type: CompareType = CompareType.VALUE
    compare_feature: Optional[str] = None  # Feature to compare against if compare_type is FEATURE
    lookback_bars: int = 0  # N=0 means check reference date only, N>0 means check N bars before
    is_discrete: bool = False

    def __str__(self) -> str:
        """Human-readable representation of the rule."""
        lookback_str = ""
        if self.lookback_bars > 0:
            lookback_str = f" (within {self.lookback_bars} bars before)"

        # Determine what we're comparing to
        if self.compare_type == CompareType.FEATURE and self.compare_feature:
            compare_str = self.compare_feature
        else:
            compare_str = str(self.value)

        if self.operator == 'between':
            if self.compare_type == CompareType.FEATURE:
                return f"{self.feature} between {self.value[0]} and {self.value[1]}{lookback_str}"
            return f"{self.feature} between {self.value[0]} and {self.value[1]}{lookback_str}"
        elif self.operator == 'in':
            values_str = ', '.join(str(v) for v in self.value[:5])
            if len(self.value) > 5:
                values_str += f", ... ({len(self.value)} total)"
            return f"{self.feature} in [{values_str}]{lookback_str}"
        else:
            return f"{self.feature} {self.operator} {compare_str}{lookback_str}"


class RuleEngine:
    """
    Engine for applying filtering rules to trade logs.

    Supports:
    - Entry rules: Filter trades based on conditions at entry date
    - Exit rules: Recalculate exit points based on when exit conditions are met
    - Feature-to-feature comparison (e.g., sma20 > sma200)
    - Lookback periods: check if condition was true within N bars before reference date

    Data leakage prevention:
    - Only uses data on or before the reference date
    - Never looks at future data relative to entry
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
            mode: RuleMode.ENTRY or RuleMode.EXIT - determines behavior
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

        # Strategy configuration for exit rules (lazy import to avoid circular dependency)
        self._strategy_config: Optional['StrategyExitConfig'] = None
        self._strategy_evaluator: Optional['StrategyExitRuleEvaluator'] = None

    def set_strategy(self, strategy_name: str) -> bool:
        """
        Set the strategy for exit rule evaluation.

        Args:
            strategy_name: Name of the strategy (e.g., "AlphaTrendStrategy")

        Returns:
            True if strategy was found and set, False otherwise
        """
        from .strategy_exit_rules import StrategyExitRulesRegistry, StrategyExitRuleEvaluator

        config = StrategyExitRulesRegistry.get(strategy_name)
        if config is None:
            self._strategy_config = None
            self._strategy_evaluator = None
            return False

        self._strategy_config = config
        self._strategy_evaluator = StrategyExitRuleEvaluator(config)
        return True

    def get_strategy_config(self) -> Optional['StrategyExitConfig']:
        """Get the current strategy configuration."""
        return self._strategy_config

    def get_trade_direction(self) -> str:
        """Get trade direction from strategy config, default to LONG."""
        if self._strategy_config:
            return self._strategy_config.trade_direction
        return "LONG"

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

    def get_price_features(self) -> List[str]:
        """Get list of price-related features (open, high, low, close)."""
        return ['open', 'high', 'low', 'close']

    def get_all_comparable_features(self) -> List[str]:
        """Get all features that can be used for comparison (indicators + price)."""
        features = self.get_available_features()
        price_features = self.get_price_features()
        return sorted(set(features + price_features))

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

    def _get_feature_value_at_bar(self, pdf: pd.DataFrame, bar_idx: int, feature: str) -> Any:
        """Get feature value at a specific bar index."""
        if bar_idx < 0 or bar_idx >= len(pdf):
            return np.nan
        if feature not in pdf.columns:
            return np.nan
        return pdf.iloc[bar_idx][feature]

    def _check_condition_at_bar(
        self,
        pdf: pd.DataFrame,
        bar_idx: int,
        rule: Rule
    ) -> bool:
        """
        Check if a rule condition is satisfied at a specific bar.

        Args:
            pdf: Price DataFrame for the symbol
            bar_idx: Index of the bar to check
            rule: Rule to evaluate

        Returns:
            True if condition is satisfied
        """
        if bar_idx < 0 or bar_idx >= len(pdf):
            return False

        # Get feature value
        feature_val = self._get_feature_value_at_bar(pdf, bar_idx, rule.feature)
        if pd.isna(feature_val):
            return False

        # Get comparison value (either static or from another feature)
        if rule.compare_type == CompareType.FEATURE and rule.compare_feature:
            compare_val = self._get_feature_value_at_bar(pdf, bar_idx, rule.compare_feature)
            if pd.isna(compare_val):
                return False
        else:
            compare_val = rule.value

        # Evaluate the condition
        try:
            if rule.operator == '>':
                return float(feature_val) > float(compare_val)
            elif rule.operator == '<':
                return float(feature_val) < float(compare_val)
            elif rule.operator == '>=':
                return float(feature_val) >= float(compare_val)
            elif rule.operator == '<=':
                return float(feature_val) <= float(compare_val)
            elif rule.operator == 'between':
                if rule.value is None or not isinstance(rule.value, (tuple, list)) or len(rule.value) != 2:
                    return False
                low, high = rule.value
                return float(low) <= float(feature_val) <= float(high)
            elif rule.operator == '==':
                return feature_val == compare_val
            elif rule.operator == '!=':
                return feature_val != compare_val
            elif rule.operator == 'in':
                if rule.value is None:
                    return False
                return feature_val in rule.value
            else:
                return False
        except (TypeError, ValueError):
            # Type conversion failed - values are not comparable
            return False

    def _check_rule_with_lookback(
        self,
        pdf: pd.DataFrame,
        ref_bar_idx: int,
        rule: Rule
    ) -> bool:
        """
        Check if a rule is satisfied, considering lookback (for ENTRY mode).

        Args:
            pdf: Price DataFrame
            ref_bar_idx: Reference bar index
            rule: Rule to evaluate

        Returns:
            True if rule is satisfied (at ref bar for N=0, or any bar in lookback window for N>0)
        """
        if rule.lookback_bars == 0:
            return self._check_condition_at_bar(pdf, ref_bar_idx, rule)
        else:
            # Check N bars before reference (not including reference)
            for offset in range(1, rule.lookback_bars + 1):
                check_idx = ref_bar_idx - offset
                if check_idx >= 0 and self._check_condition_at_bar(pdf, check_idx, rule):
                    return True
            return False

    def _check_exit_rule_at_bar(
        self,
        pdf: pd.DataFrame,
        bar_idx: int,
        rule: Rule
    ) -> bool:
        """
        Check if an exit rule is satisfied at a specific bar.

        For exit rules, we always check AT the current bar.
        Lookback for exit rules means: condition must be true at current bar,
        AND we can optionally require it was also true at some point in lookback window.

        For simplicity, exit rules check AT the current bar (lookback is ignored for exit scanning).

        Args:
            pdf: Price DataFrame
            bar_idx: Bar index to check
            rule: Rule to evaluate

        Returns:
            True if exit condition is satisfied at this bar
        """
        return self._check_condition_at_bar(pdf, bar_idx, rule)

    def evaluate_rule(self, rule: Rule) -> pd.Series:
        """
        Evaluate a single rule against all trades (for ENTRY mode).

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

            pdf = self.price_data_dict.get(symbol)

            if pdf is None or ref_date is None or pd.isna(ref_date):
                results.append(False)
                continue

            ref_date = pd.to_datetime(ref_date)

            # Find the bar index for the reference date
            date_mask = pdf['date'] <= ref_date
            if not date_mask.any():
                results.append(False)
                continue

            ref_bar_idx = date_mask.sum() - 1  # Index of last bar on or before ref_date

            # Check rule with lookback
            results.append(self._check_rule_with_lookback(pdf, ref_bar_idx, rule))

        return pd.Series(results, index=self.trades_df.index)

    def apply_rules(self, rules: List[Rule]) -> pd.DataFrame:
        """
        Apply rules to filter trades (ENTRY mode) or recalculate exits (EXIT mode).

        Args:
            rules: List of rules to apply

        Returns:
            Filtered/modified DataFrame
        """
        if self.mode == RuleMode.ENTRY and not rules:
            return self.trades_df.copy()

        if self.mode == RuleMode.ENTRY:
            return self._apply_entry_rules(rules)
        else:
            return self._apply_exit_rules_to_df(self.trades_df, rules)

    def _apply_entry_rules(self, rules: List[Rule]) -> pd.DataFrame:
        """Apply entry rules to filter trades."""
        combined_mask = pd.Series([True] * len(self.trades_df), index=self.trades_df.index)

        for rule in rules:
            rule_mask = self.evaluate_rule(rule)
            combined_mask &= rule_mask

        return self.trades_df[combined_mask].copy()

    def _apply_exit_rules_to_df(self, trades_df: pd.DataFrame, rules: List[Rule]) -> pd.DataFrame:
        """
        Apply exit rules to recalculate exit points for a given trades DataFrame.

        Exit logic (AND):
        - Exit when (Original Strategy Exit Rules) AND (User-Defined Rules) are ALL satisfied
        - If no strategy is set, only user-defined rules are checked
        - If no user-defined rules, only strategy rules are checked
        - Scans from entry+1 to end of data to find first bar where all conditions met

        Args:
            trades_df: DataFrame of trades to process
            rules: List of user-defined rules

        Returns:
            DataFrame with recalculated exit points and P/L
        """
        result_rows = []

        # Determine trade direction
        is_long = self.get_trade_direction().upper() == "LONG"

        for idx, trade in trades_df.iterrows():
            entry_date = trade.get('entry_date')
            original_exit_date = trade.get('exit_date')
            symbol = str(trade.get('symbol', '')).upper()
            entry_price = trade.get('entry_price', 0)
            quantity = trade.get('quantity', 1)

            # Check for trade direction in the trade data itself
            trade_direction = trade.get('direction', trade.get('side', None))
            if trade_direction:
                trade_is_long = str(trade_direction).upper() in ('LONG', 'BUY', 'L')
            else:
                trade_is_long = is_long  # Use strategy default

            pdf = self.price_data_dict.get(symbol)

            if pdf is None or entry_date is None or pd.isna(entry_date):
                result_rows.append(trade.to_dict())
                continue

            entry_date = pd.to_datetime(entry_date)
            original_exit_date = pd.to_datetime(original_exit_date) if original_exit_date is not None else None

            # Find entry bar index
            entry_mask = pdf['date'] <= entry_date
            if not entry_mask.any():
                result_rows.append(trade.to_dict())
                continue

            entry_bar_idx = entry_mask.sum() - 1

            # For exit rules, scan to end of available data (not just original exit)
            # This allows finding exits that might occur after original exit if rules weren't met
            end_bar_idx = len(pdf) - 1

            # Search for first bar after entry where ALL exit rules are satisfied
            new_exit_bar_idx = None
            new_exit_date = None
            new_exit_price = None

            # Make sure we have bars to check (entry+1 to end of data)
            start_bar = entry_bar_idx + 1

            for bar_idx in range(start_bar, end_bar_idx + 1):
                # Check 1: Original strategy exit rules (if strategy is set)
                strategy_rules_met = True
                if self._strategy_evaluator and self._strategy_config and self._strategy_config.exit_rules:
                    strategy_rules_met = self._strategy_evaluator.check_exit_rules_at_bar(
                        pdf, bar_idx, entry_bar_idx, entry_price
                    )

                # Check 2: User-defined exit rules
                user_rules_met = True
                if rules:  # Only check if user has defined rules
                    for rule in rules:
                        if not self._check_exit_rule_at_bar(pdf, bar_idx, rule):
                            user_rules_met = False
                            break

                # AND logic: Exit when BOTH strategy rules AND user rules are satisfied
                # If no strategy rules defined, only user rules matter
                # If no user rules defined, only strategy rules matter
                has_strategy_rules = self._strategy_evaluator and self._strategy_config and self._strategy_config.exit_rules
                has_user_rules = bool(rules)

                if has_strategy_rules and has_user_rules:
                    # Both must be satisfied
                    all_conditions_met = strategy_rules_met and user_rules_met
                elif has_strategy_rules:
                    # Only strategy rules
                    all_conditions_met = strategy_rules_met
                elif has_user_rules:
                    # Only user rules
                    all_conditions_met = user_rules_met
                else:
                    # No rules defined - use original exit
                    all_conditions_met = False

                if all_conditions_met:
                    new_exit_bar_idx = bar_idx
                    new_exit_date = pdf.iloc[bar_idx]['date']
                    new_exit_price = pdf.iloc[bar_idx]['close']
                    break

            # If no exit rules met, use original exit
            if new_exit_bar_idx is None:
                result_rows.append(trade.to_dict())
                continue

            # Recalculate trade with new exit
            new_trade = trade.to_dict()
            new_trade['exit_date'] = new_exit_date
            new_trade['exit_price'] = new_exit_price

            # Recalculate P/L based on trade direction
            if entry_price > 0 and new_exit_price is not None:
                if trade_is_long:
                    # Long trade: profit when price goes up
                    new_pl = (new_exit_price - entry_price) * abs(quantity)
                    new_pl_pct = ((new_exit_price / entry_price) - 1) * 100
                else:
                    # Short trade: profit when price goes down
                    new_pl = (entry_price - new_exit_price) * abs(quantity)
                    new_pl_pct = ((entry_price / new_exit_price) - 1) * 100 if new_exit_price > 0 else 0
                new_trade['pl'] = new_pl
                new_trade['pl_pct'] = new_pl_pct

            # Recalculate duration
            if isinstance(new_exit_date, pd.Timestamp) and isinstance(entry_date, pd.Timestamp):
                new_trade['duration_days'] = (new_exit_date - entry_date).days

            result_rows.append(new_trade)

        return pd.DataFrame(result_rows)

    def count_passing_trades(self, rules: List[Rule]) -> int:
        """
        Count trades that pass all rules (entry mode) or total trades (exit mode).

        Args:
            rules: List of rules to apply

        Returns:
            Number of trades
        """
        filtered = self.apply_rules(rules)
        return len(filtered)

    def get_rule_preview(self, rules: List[Rule], sample_limit: int = 20) -> Dict[str, Any]:
        """
        Get a preview of rule application.

        Args:
            rules: List of rules to preview
            sample_limit: For exit mode, limit preview calculation to this many trades for speed

        Returns:
            Dict with total_trades, passing_trades, pass_rate, and per-rule counts
        """
        total = len(self.trades_df)

        if self.mode == RuleMode.ENTRY:
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
        else:
            # Exit mode - use a lightweight preview to avoid UI freezing
            # Only do full calculation on a sample of trades for preview
            has_strategy_rules = self._strategy_evaluator and self._strategy_config and self._strategy_config.exit_rules
            has_user_rules = bool(rules)

            if not has_strategy_rules and not has_user_rules:
                # No rules defined - quick return
                result = {
                    'total_trades': total,
                    'passing_trades': total,
                    'modified_exits': 0,
                    'unmodified_exits': total,
                    'pass_rate': 100.0,
                    'mode': self.mode.value,
                    'is_estimate': False,
                }
                result['per_rule'] = []
                return result

            # Sample trades for preview (to avoid freezing UI)
            if total > sample_limit:
                sample_df = self.trades_df.sample(n=sample_limit, random_state=42)
                is_estimate = True
            else:
                sample_df = self.trades_df
                is_estimate = False

            # Apply rules only to sample
            sample_modified = self._apply_exit_rules_to_df(sample_df, rules)

            # Count modified exits in sample
            original_exits = sample_df['exit_date'].values
            new_exits = sample_modified['exit_date'].values
            sample_modified_count = sum(1 for i in range(len(original_exits))
                               if not pd.isna(original_exits[i]) and not pd.isna(new_exits[i])
                               and original_exits[i] != new_exits[i])

            # Estimate for full dataset
            if is_estimate and len(sample_df) > 0:
                estimated_modified = int((sample_modified_count / len(sample_df)) * total)
            else:
                estimated_modified = sample_modified_count

            result = {
                'total_trades': total,
                'passing_trades': total,  # All trades kept in exit mode
                'modified_exits': estimated_modified,
                'unmodified_exits': total - estimated_modified,
                'pass_rate': 100.0,  # All trades pass in exit mode
                'mode': self.mode.value,
                'is_estimate': is_estimate,
            }
            result['per_rule'] = []

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
                        df = df.sort_values('date').reset_index(drop=True)
                    result[ticker.upper()] = df
                    break
                except Exception:
                    pass

    return result
