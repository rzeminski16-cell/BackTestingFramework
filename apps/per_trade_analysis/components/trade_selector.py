"""
Trade selection UI components for Per-Trade Analysis.
"""

import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def render_sidebar_filters(trades_df: pd.DataFrame) -> Dict:
    """
    Render sidebar filter controls.

    Args:
        trades_df: DataFrame with all trades

    Returns:
        Dict with filter values
    """
    st.sidebar.header("Filters")

    filters = {}

    # Symbol filter
    if 'symbol' in trades_df.columns:
        symbols = sorted(trades_df['symbol'].unique().tolist())
        filters['symbols'] = st.sidebar.multiselect(
            "Symbols",
            options=symbols,
            default=symbols,
            help="Filter by trading symbol"
        )

    # Side filter
    if 'side' in trades_df.columns:
        sides = trades_df['side'].unique().tolist()
        filters['sides'] = st.sidebar.multiselect(
            "Trade Direction",
            options=sides,
            default=sides,
            help="Filter by LONG or SHORT"
        )

    # Outcome filter
    st.sidebar.subheader("Outcome")
    filters['show_winners'] = st.sidebar.checkbox("Winners", value=True)
    filters['show_losers'] = st.sidebar.checkbox("Losers", value=True)
    filters['show_breakeven'] = st.sidebar.checkbox("Breakeven", value=True)

    # Date range filter
    if 'exit_date' in trades_df.columns:
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        min_date = trades_df['exit_date'].min().date()
        max_date = trades_df['exit_date'].max().date()

        st.sidebar.subheader("Date Range")
        filters['date_range'] = st.sidebar.date_input(
            "Exit Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter by trade exit date"
        )

    # P&L range filter
    if 'pl_pct' in trades_df.columns:
        min_pl = float(trades_df['pl_pct'].min())
        max_pl = float(trades_df['pl_pct'].max())

        st.sidebar.subheader("P&L Range")
        filters['pl_range'] = st.sidebar.slider(
            "P&L %",
            min_value=min_pl,
            max_value=max_pl,
            value=(min_pl, max_pl),
            format="%.1f%%"
        )

    return filters


def apply_filters(trades_df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply sidebar filters to trades DataFrame.

    Args:
        trades_df: Full trades DataFrame
        filters: Dict of filter values from render_sidebar_filters

    Returns:
        Filtered DataFrame
    """
    filtered = trades_df.copy()

    # Symbol filter
    if 'symbols' in filters and filters['symbols']:
        filtered = filtered[filtered['symbol'].isin(filters['symbols'])]

    # Side filter
    if 'sides' in filters and filters['sides']:
        filtered = filtered[filtered['side'].isin(filters['sides'])]

    # Outcome filter
    if 'pl_pct' in filtered.columns:
        outcome_mask = pd.Series(False, index=filtered.index)

        if filters.get('show_winners', True):
            outcome_mask |= (filtered['pl_pct'] > 0.5)

        if filters.get('show_losers', True):
            outcome_mask |= (filtered['pl_pct'] < -0.5)

        if filters.get('show_breakeven', True):
            outcome_mask |= ((filtered['pl_pct'] >= -0.5) & (filtered['pl_pct'] <= 0.5))

        filtered = filtered[outcome_mask]

    # Date range filter
    if 'date_range' in filters and 'exit_date' in filtered.columns:
        date_range = filters['date_range']
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filtered = filtered[
                (pd.to_datetime(filtered['exit_date']).dt.date >= date_range[0]) &
                (pd.to_datetime(filtered['exit_date']).dt.date <= date_range[1])
            ]

    # P&L range filter
    if 'pl_range' in filters and 'pl_pct' in filtered.columns:
        pl_min, pl_max = filters['pl_range']
        filtered = filtered[
            (filtered['pl_pct'] >= pl_min) &
            (filtered['pl_pct'] <= pl_max)
        ]

    return filtered


def render_trade_selector(
    trades_df: pd.DataFrame,
    selected_trade_ids: List[str]
) -> Tuple[List[str], pd.DataFrame]:
    """
    Render the trade selection panel.

    Args:
        trades_df: DataFrame with filtered trades
        selected_trade_ids: Currently selected trade IDs

    Returns:
        Tuple of (updated selected trade IDs, display DataFrame)
    """
    st.subheader("Trade Selection")

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", len(trades_df))

    with col2:
        if 'pl_pct' in trades_df.columns:
            winners = len(trades_df[trades_df['pl_pct'] > 0])
            st.metric("Winners", winners)

    with col3:
        if 'pl_pct' in trades_df.columns:
            losers = len(trades_df[trades_df['pl_pct'] < 0])
            st.metric("Losers", losers)

    with col4:
        if 'pl_pct' in trades_df.columns and len(trades_df) > 0:
            win_rate = len(trades_df[trades_df['pl_pct'] > 0]) / len(trades_df) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Auto-Select Representative Sample", type="primary"):
            st.session_state['auto_select'] = True

    with col2:
        if st.button("Clear Selection"):
            st.session_state['selected_trades'] = []
            selected_trade_ids = []

    with col3:
        if st.button("Select All Visible"):
            selected_trade_ids = trades_df['trade_id'].tolist()
            st.session_state['selected_trades'] = selected_trade_ids

    # Display trades table
    st.write(f"**Selected:** {len(selected_trade_ids)} trades")

    # Prepare display DataFrame
    display_cols = ['trade_id', 'symbol', 'side', 'entry_date', 'exit_date', 'pl_pct', 'duration_days', 'exit_reason']
    available_cols = [c for c in display_cols if c in trades_df.columns]
    display_df = trades_df[available_cols].copy()

    # Format columns for display
    if 'entry_date' in display_df.columns:
        display_df['entry_date'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d')

    if 'exit_date' in display_df.columns:
        display_df['exit_date'] = pd.to_datetime(display_df['exit_date']).dt.strftime('%Y-%m-%d')

    if 'pl_pct' in display_df.columns:
        display_df['pl_pct'] = display_df['pl_pct'].apply(lambda x: f"{x:+.2f}%")

    if 'exit_reason' in display_df.columns:
        display_df['exit_reason'] = display_df['exit_reason'].apply(lambda x: str(x)[:30] if pd.notna(x) else '')

    # Add selection column
    display_df.insert(0, 'selected', display_df['trade_id'].isin(selected_trade_ids))

    # Use data editor for interactive selection
    edited_df = st.data_editor(
        display_df,
        column_config={
            "selected": st.column_config.CheckboxColumn(
                "Select",
                help="Select trade for analysis",
                default=False
            ),
            "trade_id": st.column_config.TextColumn("Trade ID", width="small"),
            "symbol": st.column_config.TextColumn("Symbol", width="small"),
            "side": st.column_config.TextColumn("Side", width="small"),
            "entry_date": st.column_config.TextColumn("Entry", width="small"),
            "exit_date": st.column_config.TextColumn("Exit", width="small"),
            "pl_pct": st.column_config.TextColumn("P&L %", width="small"),
            "duration_days": st.column_config.NumberColumn("Days", width="small"),
            "exit_reason": st.column_config.TextColumn("Exit Reason", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        key="trade_selector"
    )

    # Update selected trades from editor
    new_selected = edited_df[edited_df['selected'] == True]['trade_id'].tolist()

    return new_selected, display_df


def render_backtest_selector(logs_path: Path) -> Optional[str]:
    """
    Render backtest folder selector.

    Args:
        logs_path: Path to logs directory

    Returns:
        Selected backtest name or None
    """
    portfolio_path = logs_path / 'backtests' / 'portfolio'

    if not portfolio_path.exists():
        st.warning("No backtest logs found. Run a portfolio backtest first.")
        return None

    # Find available backtests
    backtests = [d.name for d in portfolio_path.iterdir()
                 if d.is_dir() and (d / 'trades').exists()]

    if not backtests:
        st.warning("No portfolio backtests with trade logs found.")
        return None

    selected = st.selectbox(
        "Select Backtest",
        options=backtests,
        help="Choose a backtest to analyze"
    )

    return selected


def render_file_upload_section() -> List[Path]:
    """
    Render file upload section for trade log CSVs.

    Returns:
        List of paths to uploaded files
    """
    st.subheader("Upload Trade Logs")

    uploaded_files = st.file_uploader(
        "Upload trade log CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more trade log CSV files from your backtests"
    )

    if not uploaded_files:
        return []

    # Save uploaded files temporarily
    temp_paths = []
    for uploaded_file in uploaded_files:
        # Create temp directory if needed (cross-platform)
        temp_dir = Path(tempfile.gettempdir()) / "per_trade_analysis"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = temp_dir / uploaded_file.name

        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        temp_paths.append(temp_path)

    st.success(f"Uploaded {len(temp_paths)} file(s)")

    return temp_paths


def render_sample_rationale(rationales: List[Dict]):
    """
    Display rationale for auto-selected trades.

    Args:
        rationales: List of dicts with selection rationale
    """
    if not rationales:
        return

    with st.expander("Selection Rationale", expanded=False):
        for r in rationales:
            st.markdown(f"- **{r['trade_id']}** ({r['bucket']}): {r['reason']}")
