"""
Per-Trade Analysis Streamlit Application

A comprehensive tool for analyzing individual trades from backtesting results.
Launch with: streamlit run apps/per_trade_analysis/app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.per_trade_analysis.data_aggregation import (
    TradeLogLoader,
    RawDataLoader,
    TradeSampler,
    TradeDataAggregator,
    aggregate_trades_for_analysis
)
from apps.per_trade_analysis.pattern_analysis import (
    SignalStrengthCalculator,
    PatternFlagger,
    AggregatePatternAnalyzer,
    SignalStrengthScore,
    PatternFlag
)
from apps.per_trade_analysis.reporting import PerTradeExcelReport
from apps.per_trade_analysis.components.trade_selector import (
    render_sidebar_filters,
    apply_filters,
    render_trade_selector,
    render_backtest_selector,
    render_file_upload_section,
    render_sample_rationale
)
from apps.per_trade_analysis.components.tabs import (
    render_trade_summary_tab,
    render_price_action_tab,
    render_fundamentals_tab,
    render_insider_tab,
    render_options_tab,
    render_correlation_tab,
    render_mae_mfe_tab,
    render_pattern_analysis_tab
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Per-Trade Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'trades_df' not in st.session_state:
        st.session_state['trades_df'] = None

    if 'selected_trades' not in st.session_state:
        st.session_state['selected_trades'] = []

    if 'analyzed_trades' not in st.session_state:
        st.session_state['analyzed_trades'] = {}

    if 'signal_scores' not in st.session_state:
        st.session_state['signal_scores'] = {}

    if 'pattern_flags' not in st.session_state:
        st.session_state['pattern_flags'] = {}

    if 'pattern_summary' not in st.session_state:
        st.session_state['pattern_summary'] = None

    if 'sample_rationales' not in st.session_state:
        st.session_state['sample_rationales'] = []

    if 'auto_select' not in st.session_state:
        st.session_state['auto_select'] = False


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_backtest_trades(backtest_name: str, logs_path: Path) -> pd.DataFrame:
    """Load trades from a backtest with caching."""
    loader = TradeLogLoader(logs_path)
    trade_files = loader.discover_backtest_trades(backtest_name)

    if not trade_files:
        return pd.DataFrame()

    return loader.load_multiple_trade_logs(trade_files)


def get_paths() -> tuple:
    """Get project paths."""
    project_root = Path(__file__).parent.parent.parent
    logs_path = project_root / 'logs'
    data_path = project_root / 'raw_data'
    return project_root, logs_path, data_path


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_selected_trades(
    trades_df: pd.DataFrame,
    selected_ids: List[str],
    data_loader: RawDataLoader
) -> tuple:
    """
    Analyze selected trades.

    Returns:
        Tuple of (analyzed_trades dict, signal_scores dict, pattern_flags dict)
    """
    selected_trades = trades_df[trades_df['trade_id'].isin(selected_ids)]

    # Aggregate data for each trade
    analyzed_trades = {}
    signal_scores = {}
    pattern_flags = {}

    aggregator = TradeDataAggregator(data_loader)
    signal_calculator = SignalStrengthCalculator()
    flagger = PatternFlagger()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (_, trade) in enumerate(selected_trades.iterrows()):
        trade_id = trade['trade_id']
        status_text.text(f"Analyzing trade {idx + 1}/{len(selected_trades)}: {trade_id}")

        try:
            # Aggregate all data
            trade_data = aggregator.aggregate_trade_data(trade)
            analyzed_trades[trade_id] = trade_data

            # Calculate signal strength
            if trade_data.price_data is not None and len(trade_data.price_data) > 0:
                entry_date = pd.to_datetime(trade['entry_date'])
                price_data = trade_data.price_data
                price_data['date'] = pd.to_datetime(price_data['date'])

                # Get entry row
                entry_mask = price_data['date'] == entry_date
                if entry_mask.any():
                    entry_row = price_data[entry_mask].iloc[0]
                else:
                    price_data['date_diff'] = abs((price_data['date'] - entry_date).dt.days)
                    entry_idx = price_data['date_diff'].idxmin()
                    entry_row = price_data.loc[entry_idx]

                # Pre-entry history
                pre_entry = price_data[price_data['date'] < entry_date]

                score = signal_calculator.calculate_signal_strength(
                    entry_row,
                    pre_entry,
                    side=trade.get('side', 'LONG')
                )
                signal_scores[trade_id] = score

                # Generate pattern flags
                flags = flagger.flag_all_patterns(
                    entry_row=entry_row,
                    price_history=pre_entry,
                    insider_flags=trade_data.insider_flags,
                    options_data=trade_data.options_data,
                    correlation_data={
                        'sector_correlation': trade_data.sector_correlation,
                        'index_correlation': trade_data.index_correlation
                    },
                    market_regime=trade_data.market_regime,
                    side=trade.get('side', 'LONG')
                )
                pattern_flags[trade_id] = flags

        except Exception as e:
            st.warning(f"Error analyzing trade {trade_id}: {e}")

        progress_bar.progress((idx + 1) / len(selected_trades))

    progress_bar.empty()
    status_text.empty()

    return analyzed_trades, signal_scores, pattern_flags


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.title(" Per-Trade Analysis")
    st.markdown("Deep-dive forensics for individual trades from your backtests.")

    # Get paths
    project_root, logs_path, data_path = get_paths()

    # Sidebar - Data Source Selection
    st.sidebar.header("Data Source")

    data_source = st.sidebar.radio(
        "Select data source",
        options=["Load from Backtest", "Upload CSV Files"],
        help="Choose how to load trade data"
    )

    # Load trades based on source
    if data_source == "Load from Backtest":
        selected_backtest = render_backtest_selector(logs_path)

        if selected_backtest:
            with st.spinner("Loading trades..."):
                trades_df = load_backtest_trades(selected_backtest, logs_path)
                st.session_state['trades_df'] = trades_df

    else:
        uploaded_paths = render_file_upload_section()
        if uploaded_paths:
            loader = TradeLogLoader()
            trades_df = loader.load_multiple_trade_logs(uploaded_paths)
            st.session_state['trades_df'] = trades_df

    # Main content
    trades_df = st.session_state.get('trades_df')

    if trades_df is None or len(trades_df) == 0:
        st.info("Select a backtest or upload trade log CSV files to begin analysis.")
        return

    st.success(f"Loaded {len(trades_df)} trades from {trades_df['symbol'].nunique()} symbols")

    # Apply sidebar filters
    filters = render_sidebar_filters(trades_df)
    filtered_df = apply_filters(trades_df, filters)

    st.write(f"**Filtered:** {len(filtered_df)} trades")

    # Trade selection
    st.divider()

    # Handle auto-select
    if st.session_state.get('auto_select'):
        sampler = TradeSampler(target_sample_size=12)
        selected_df, rationales = sampler.sample_representative_trades(filtered_df)
        st.session_state['selected_trades'] = selected_df['trade_id'].tolist()
        st.session_state['sample_rationales'] = rationales
        st.session_state['auto_select'] = False
        st.rerun()

    # Render trade selector
    selected_ids, _ = render_trade_selector(
        filtered_df,
        st.session_state.get('selected_trades', [])
    )
    st.session_state['selected_trades'] = selected_ids

    # Show sample rationale if available
    if st.session_state.get('sample_rationales'):
        render_sample_rationale(st.session_state['sample_rationales'])

    # Analyze button
    st.divider()

    if selected_ids:
        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Analyze Selected Trades", type="primary"):
                data_loader = RawDataLoader(data_path)

                analyzed, scores, flags = analyze_selected_trades(
                    filtered_df, selected_ids, data_loader
                )

                st.session_state['analyzed_trades'] = analyzed
                st.session_state['signal_scores'] = scores
                st.session_state['pattern_flags'] = flags

                # Generate pattern summary
                if len(analyzed) > 1:
                    analyzer = AggregatePatternAnalyzer()
                    summary = analyzer.analyze_patterns(
                        list(analyzed.values()),
                        scores,
                        flags
                    )
                    st.session_state['pattern_summary'] = summary

                st.success(f"Analyzed {len(analyzed)} trades!")

        with col2:
            if st.session_state.get('analyzed_trades'):
                if st.button("Export to Excel"):
                    report_gen = PerTradeExcelReport()

                    analyzed = st.session_state['analyzed_trades']
                    scores = st.session_state['signal_scores']
                    flags = st.session_state['pattern_flags']
                    summary = st.session_state.get('pattern_summary')

                    if len(analyzed) == 1:
                        trade_id = list(analyzed.keys())[0]
                        path = report_gen.generate_report(
                            analyzed[trade_id],
                            scores.get(trade_id),
                            flags.get(trade_id, []),
                            summary
                        )
                    else:
                        path = report_gen.generate_multi_trade_report(
                            list(analyzed.values()),
                            scores,
                            flags,
                            summary
                        )

                    st.success(f"Report saved to: {path}")

    # Analysis tabs
    analyzed_trades = st.session_state.get('analyzed_trades', {})
    signal_scores = st.session_state.get('signal_scores', {})
    pattern_flags = st.session_state.get('pattern_flags', {})
    pattern_summary = st.session_state.get('pattern_summary')

    if analyzed_trades:
        st.divider()

        # Trade selector for detailed view
        trade_ids = list(analyzed_trades.keys())

        if len(trade_ids) > 1:
            selected_trade_id = st.selectbox(
                "Select trade for detailed analysis",
                options=trade_ids,
                format_func=lambda x: f"{analyzed_trades[x].symbol} - {x}"
            )
        else:
            selected_trade_id = trade_ids[0]

        current_trade = analyzed_trades[selected_trade_id]
        current_score = signal_scores.get(selected_trade_id)
        current_flags = pattern_flags.get(selected_trade_id, [])

        # Tabs
        tabs = st.tabs([
            " Summary",
            " Price Action",
            " Fundamentals",
            " Insider Activity",
            " Options",
            " Correlation",
            " MAE/MFE",
            " Patterns"
        ])

        with tabs[0]:
            render_trade_summary_tab(current_trade, current_score)

        with tabs[1]:
            render_price_action_tab(current_trade)

        with tabs[2]:
            render_fundamentals_tab(current_trade)

        with tabs[3]:
            render_insider_tab(current_trade)

        with tabs[4]:
            render_options_tab(current_trade)

        with tabs[5]:
            render_correlation_tab(current_trade)

        with tabs[6]:
            render_mae_mfe_tab(current_trade)

        with tabs[7]:
            render_pattern_analysis_tab(
                list(analyzed_trades.values()),
                signal_scores,
                pattern_flags,
                pattern_summary
            )

        # Data quality warnings
        if current_trade.data_quality.warnings:
            with st.expander("Data Quality Warnings"):
                for warning in current_trade.data_quality.warnings:
                    st.warning(warning)


if __name__ == "__main__":
    main()
