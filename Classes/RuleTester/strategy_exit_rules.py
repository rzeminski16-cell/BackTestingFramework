"""
Strategy Exit Rules Configuration.

Defines the exit rules for each strategy in a machine-readable format
so the Rule Tester can properly apply AND logic with user-defined rules.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np


class ExitRuleType(Enum):
    """Type of exit rule."""
    STOP_LOSS_ATR = "stop_loss_atr"  # ATR-based stop loss
    STOP_LOSS_PCT = "stop_loss_pct"  # Percentage-based stop loss
    TAKE_PROFIT_ATR = "take_profit_atr"  # ATR-based take profit
    TAKE_PROFIT_PCT = "take_profit_pct"  # Percentage-based take profit
    INDICATOR_CROSS = "indicator_cross"  # Price crosses indicator (e.g., close < EMA50)
    INDICATOR_THRESHOLD = "indicator_threshold"  # Indicator reaches threshold
    MAX_DURATION = "max_duration"  # Maximum holding period
    CUSTOM = "custom"  # Custom rule with callable


@dataclass
class StrategyExitRule:
    """
    Represents a single exit rule from a strategy.

    Attributes:
        rule_type: Type of exit rule
        params: Parameters for the rule
        description: Human-readable description
        grace_period_bars: Bars after entry before this rule is active (0 = always active)
        momentum_protection: If True, rule is disabled when trade has momentum gain
        momentum_gain_pct: Minimum gain % to qualify as "momentum protected"
    """
    rule_type: ExitRuleType
    params: Dict[str, Any]
    description: str
    grace_period_bars: int = 0
    momentum_protection: bool = False
    momentum_gain_pct: float = 0.0


@dataclass
class StrategyExitConfig:
    """
    Configuration for a strategy's exit rules.

    Attributes:
        strategy_name: Name of the strategy
        display_name: Human-readable name
        trade_direction: "LONG" or "SHORT"
        exit_rules: List of exit rules
        required_indicators: Indicators needed for exit rules
    """
    strategy_name: str
    display_name: str
    trade_direction: str  # "LONG" or "SHORT"
    exit_rules: List[StrategyExitRule]
    required_indicators: List[str] = field(default_factory=list)


class StrategyExitRulesRegistry:
    """
    Registry of all strategy exit configurations.
    """

    _strategies: Dict[str, StrategyExitConfig] = {}

    @classmethod
    def register(cls, config: StrategyExitConfig) -> None:
        """Register a strategy exit configuration."""
        cls._strategies[config.strategy_name] = config

    @classmethod
    def get(cls, strategy_name: str) -> Optional[StrategyExitConfig]:
        """Get a strategy exit configuration by name."""
        return cls._strategies.get(strategy_name)

    @classmethod
    def get_all(cls) -> Dict[str, StrategyExitConfig]:
        """Get all registered strategies."""
        return cls._strategies.copy()

    @classmethod
    def get_strategy_names(cls) -> List[str]:
        """Get list of all registered strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def get_display_names(cls) -> Dict[str, str]:
        """Get mapping of strategy_name -> display_name."""
        return {name: config.display_name for name, config in cls._strategies.items()}


class StrategyExitRuleEvaluator:
    """
    Evaluates strategy exit rules against price data.
    """

    def __init__(self, strategy_config: StrategyExitConfig):
        """
        Initialize evaluator with strategy configuration.

        Args:
            strategy_config: The strategy's exit configuration
        """
        self.config = strategy_config
        self.is_long = strategy_config.trade_direction.upper() == "LONG"

    def check_exit_rules_at_bar(
        self,
        pdf: pd.DataFrame,
        bar_idx: int,
        entry_bar_idx: int,
        entry_price: float
    ) -> bool:
        """
        Check if ALL strategy exit rules are satisfied at a specific bar.

        Args:
            pdf: Price DataFrame with indicators
            bar_idx: Current bar index to check
            entry_bar_idx: Bar index when trade was entered
            entry_price: Entry price of the trade

        Returns:
            True if ALL exit rules are satisfied (OR logic between rules - any rule triggers exit)
        """
        if bar_idx < 0 or bar_idx >= len(pdf):
            return False

        bars_since_entry = bar_idx - entry_bar_idx
        current_bar = pdf.iloc[bar_idx]
        current_price = current_bar.get('close', np.nan)

        if pd.isna(current_price):
            return False

        # Calculate current gain for momentum protection
        if self.is_long:
            current_gain_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        else:
            current_gain_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0

        # Check each exit rule - exit if ANY rule is triggered (OR logic between rules)
        for rule in self.config.exit_rules:
            if self._check_single_rule(
                rule, pdf, bar_idx, entry_bar_idx, entry_price,
                bars_since_entry, current_price, current_gain_pct
            ):
                return True

        return False

    def _check_single_rule(
        self,
        rule: StrategyExitRule,
        pdf: pd.DataFrame,
        bar_idx: int,
        entry_bar_idx: int,
        entry_price: float,
        bars_since_entry: int,
        current_price: float,
        current_gain_pct: float
    ) -> bool:
        """Check if a single exit rule is triggered."""

        # Check grace period
        if rule.grace_period_bars > 0 and bars_since_entry <= rule.grace_period_bars:
            return False

        # Check momentum protection
        if rule.momentum_protection and current_gain_pct >= rule.momentum_gain_pct:
            return False

        current_bar = pdf.iloc[bar_idx]

        # Evaluate based on rule type
        if rule.rule_type == ExitRuleType.STOP_LOSS_ATR:
            return self._check_stop_loss_atr(rule, pdf, bar_idx, entry_bar_idx, entry_price, current_price)

        elif rule.rule_type == ExitRuleType.STOP_LOSS_PCT:
            return self._check_stop_loss_pct(rule, entry_price, current_price)

        elif rule.rule_type == ExitRuleType.TAKE_PROFIT_ATR:
            return self._check_take_profit_atr(rule, pdf, bar_idx, entry_bar_idx, entry_price, current_price)

        elif rule.rule_type == ExitRuleType.TAKE_PROFIT_PCT:
            return self._check_take_profit_pct(rule, entry_price, current_price)

        elif rule.rule_type == ExitRuleType.INDICATOR_CROSS:
            return self._check_indicator_cross(rule, current_bar, current_price)

        elif rule.rule_type == ExitRuleType.INDICATOR_THRESHOLD:
            return self._check_indicator_threshold(rule, current_bar)

        elif rule.rule_type == ExitRuleType.MAX_DURATION:
            return self._check_max_duration(rule, bars_since_entry)

        return False

    def _check_stop_loss_atr(
        self,
        rule: StrategyExitRule,
        pdf: pd.DataFrame,
        bar_idx: int,
        entry_bar_idx: int,
        entry_price: float,
        current_price: float
    ) -> bool:
        """Check ATR-based stop loss."""
        atr_col = rule.params.get('atr_column', 'atr_14')
        multiple = rule.params.get('multiple', 2.0)

        # Get ATR at entry
        if entry_bar_idx >= 0 and entry_bar_idx < len(pdf):
            entry_atr = pdf.iloc[entry_bar_idx].get(atr_col, np.nan)
        else:
            return False

        if pd.isna(entry_atr):
            return False

        if self.is_long:
            stop_price = entry_price - (entry_atr * multiple)
            return current_price <= stop_price
        else:
            stop_price = entry_price + (entry_atr * multiple)
            return current_price >= stop_price

    def _check_stop_loss_pct(
        self,
        rule: StrategyExitRule,
        entry_price: float,
        current_price: float
    ) -> bool:
        """Check percentage-based stop loss."""
        pct = rule.params.get('percent', 2.0)

        if self.is_long:
            stop_price = entry_price * (1 - pct / 100)
            return current_price <= stop_price
        else:
            stop_price = entry_price * (1 + pct / 100)
            return current_price >= stop_price

    def _check_take_profit_atr(
        self,
        rule: StrategyExitRule,
        pdf: pd.DataFrame,
        bar_idx: int,
        entry_bar_idx: int,
        entry_price: float,
        current_price: float
    ) -> bool:
        """Check ATR-based take profit."""
        atr_col = rule.params.get('atr_column', 'atr_14')
        multiple = rule.params.get('multiple', 3.0)

        if entry_bar_idx >= 0 and entry_bar_idx < len(pdf):
            entry_atr = pdf.iloc[entry_bar_idx].get(atr_col, np.nan)
        else:
            return False

        if pd.isna(entry_atr):
            return False

        if self.is_long:
            target_price = entry_price + (entry_atr * multiple)
            return current_price >= target_price
        else:
            target_price = entry_price - (entry_atr * multiple)
            return current_price <= target_price

    def _check_take_profit_pct(
        self,
        rule: StrategyExitRule,
        entry_price: float,
        current_price: float
    ) -> bool:
        """Check percentage-based take profit."""
        pct = rule.params.get('percent', 5.0)

        if self.is_long:
            target_price = entry_price * (1 + pct / 100)
            return current_price >= target_price
        else:
            target_price = entry_price * (1 - pct / 100)
            return current_price <= target_price

    def _check_indicator_cross(
        self,
        rule: StrategyExitRule,
        current_bar: pd.Series,
        current_price: float
    ) -> bool:
        """Check if price crosses an indicator."""
        indicator = rule.params.get('indicator', 'ema_50')
        direction = rule.params.get('direction', 'below')  # 'below' or 'above'

        indicator_value = current_bar.get(indicator, np.nan)
        if pd.isna(indicator_value):
            return False

        if direction == 'below':
            return current_price < indicator_value
        else:
            return current_price > indicator_value

    def _check_indicator_threshold(
        self,
        rule: StrategyExitRule,
        current_bar: pd.Series
    ) -> bool:
        """Check if indicator reaches a threshold."""
        indicator = rule.params.get('indicator', 'rsi_14')
        operator = rule.params.get('operator', '>')
        threshold = rule.params.get('threshold', 70)

        indicator_value = current_bar.get(indicator, np.nan)
        if pd.isna(indicator_value):
            return False

        if operator == '>':
            return indicator_value > threshold
        elif operator == '<':
            return indicator_value < threshold
        elif operator == '>=':
            return indicator_value >= threshold
        elif operator == '<=':
            return indicator_value <= threshold

        return False

    def _check_max_duration(
        self,
        rule: StrategyExitRule,
        bars_since_entry: int
    ) -> bool:
        """Check if max holding duration exceeded."""
        max_bars = rule.params.get('max_bars', 100)
        return bars_since_entry >= max_bars


# =============================================================================
# REGISTER KNOWN STRATEGIES
# =============================================================================

def register_default_strategies():
    """Register the default strategy exit configurations."""

    # AlphaTrend Strategy
    alphatrend_config = StrategyExitConfig(
        strategy_name="AlphaTrendStrategy",
        display_name="Alpha Trend Strategy",
        trade_direction="LONG",
        required_indicators=['ema_50', 'atr_14'],
        exit_rules=[
            # Stop Loss - ATR based
            StrategyExitRule(
                rule_type=ExitRuleType.STOP_LOSS_ATR,
                params={'atr_column': 'atr_14', 'multiple': 2.5},
                description="Stop Loss: Entry - (ATR14 × 2.5)",
                grace_period_bars=0,
                momentum_protection=False
            ),
            # EMA Exit - with grace period and momentum protection
            StrategyExitRule(
                rule_type=ExitRuleType.INDICATOR_CROSS,
                params={'indicator': 'ema_50', 'direction': 'below'},
                description="Exit when Close < EMA(50)",
                grace_period_bars=14,  # Default grace period
                momentum_protection=True,
                momentum_gain_pct=2.0  # Default momentum threshold
            ),
        ]
    )
    StrategyExitRulesRegistry.register(alphatrend_config)

    # Random Base Strategy
    random_config = StrategyExitConfig(
        strategy_name="RandomBaseStrategy",
        display_name="Random Base Strategy",
        trade_direction="LONG",
        required_indicators=['atr_14'],
        exit_rules=[
            # Stop Loss - ATR based
            StrategyExitRule(
                rule_type=ExitRuleType.STOP_LOSS_ATR,
                params={'atr_column': 'atr_14', 'multiple': 2.0},
                description="Stop Loss: Entry - (ATR14 × 2.0)",
                grace_period_bars=0,
                momentum_protection=False
            ),
        ]
    )
    StrategyExitRulesRegistry.register(random_config)

    # RSI Strategy
    rsi_config = StrategyExitConfig(
        strategy_name="RSIStrategy",
        display_name="RSI Strategy",
        trade_direction="LONG",
        required_indicators=['rsi_14'],
        exit_rules=[
            # RSI Overbought Exit
            StrategyExitRule(
                rule_type=ExitRuleType.INDICATOR_THRESHOLD,
                params={'indicator': 'rsi_14', 'operator': '>', 'threshold': 70},
                description="Exit when RSI(14) > 70 (overbought)",
                grace_period_bars=0,
                momentum_protection=False
            ),
        ]
    )
    StrategyExitRulesRegistry.register(rsi_config)

    # No Strategy (user-defined rules only)
    no_strategy_config = StrategyExitConfig(
        strategy_name="None",
        display_name="No Strategy (Custom Rules Only)",
        trade_direction="LONG",
        required_indicators=[],
        exit_rules=[]  # No original exit rules - only user-defined rules apply
    )
    StrategyExitRulesRegistry.register(no_strategy_config)


# Register strategies on module load
register_default_strategies()
