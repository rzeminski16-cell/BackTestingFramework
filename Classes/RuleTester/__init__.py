"""
RuleTester module for testing the effect of rules on strategy performance.

This module provides tools to:
- Define and apply filtering rules to trade logs
- Calculate before/after performance metrics
- Compare cumulative P/L and RAR% over time
"""

from .rule_engine import Rule, RuleEngine, RuleMode, extract_ticker_from_filename, load_price_data_for_tickers
from .metrics_calculator import RuleMetricsCalculator, export_filtered_trades, export_comparison_report

__all__ = [
    'Rule',
    'RuleEngine',
    'RuleMode',
    'RuleMetricsCalculator',
    'extract_ticker_from_filename',
    'load_price_data_for_tickers',
    'export_filtered_trades',
    'export_comparison_report',
]
