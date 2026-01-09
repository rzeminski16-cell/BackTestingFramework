"""
Streamlit UI components for Per-Trade Analysis.
"""

from .trade_selector import render_trade_selector, render_sidebar_filters
from .tabs import (
    render_trade_summary_tab,
    render_price_action_tab,
    render_fundamentals_tab,
    render_insider_tab,
    render_options_tab,
    render_correlation_tab,
    render_mae_mfe_tab,
    render_pattern_analysis_tab
)

__all__ = [
    'render_trade_selector',
    'render_sidebar_filters',
    'render_trade_summary_tab',
    'render_price_action_tab',
    'render_fundamentals_tab',
    'render_insider_tab',
    'render_options_tab',
    'render_correlation_tab',
    'render_mae_mfe_tab',
    'render_pattern_analysis_tab'
]
