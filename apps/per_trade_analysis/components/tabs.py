"""
Analysis tab components for Per-Trade Analysis Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..data_aggregation import TradeAnalysisData, MAEMFEResult, MarketRegime
from ..pattern_analysis import SignalStrengthScore, PatternFlag, PatternSummary


# =============================================================================
# TAB 1: TRADE SUMMARY
# =============================================================================

def render_trade_summary_tab(
    trade: TradeAnalysisData,
    signal_score: Optional[SignalStrengthScore] = None
):
    """Render the trade summary tab."""
    info = trade.trade_info

    # Key metrics in columns
    st.subheader("Position Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pl = info.get('pl', 0)
        pl_pct = info.get('pl_pct', 0)
        delta_color = "normal" if pl >= 0 else "inverse"
        st.metric("P&L", f"${pl:,.2f}", f"{pl_pct:+.2f}%", delta_color=delta_color)

    with col2:
        st.metric("Duration", f"{info.get('duration_days', 0)} days")

    with col3:
        st.metric("Entry Price", f"${info.get('entry_price', 0):.4f}")

    with col4:
        st.metric("Exit Price", f"${info.get('exit_price', 0):.4f}")

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Side", info.get('side', 'LONG'))

    with col2:
        qty = info.get('quantity', 0)
        st.metric("Quantity", f"{qty:,.2f}")

    with col3:
        if trade.mae_mfe:
            st.metric("MAE", f"{trade.mae_mfe.mae_pct:.2f}%")
        else:
            st.metric("MAE", "N/A")

    with col4:
        if trade.mae_mfe:
            st.metric("MFE", f"{trade.mae_mfe.mfe_pct:.2f}%")
        else:
            st.metric("MFE", "N/A")

    # Signal strength gauge
    if signal_score:
        st.subheader("Entry Signal Strength")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Display as progress bar
            score_pct = signal_score.total_score / 100
            st.progress(score_pct)
            st.markdown(f"**{signal_score.total_score:.0f}/100** - {signal_score.tier.upper()}")

        with col2:
            st.info(signal_score.description)

    # Entry/Exit reasons
    st.subheader("Trade Execution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Entry Reason:**")
        st.write(info.get('entry_reason', 'N/A'))

    with col2:
        st.markdown("**Exit Reason:**")
        st.write(info.get('exit_reason', 'N/A'))

    # Position sizing context
    with st.expander("Position Sizing Details"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Entry Equity", f"${info.get('entry_equity', 0):,.2f}")

        with col2:
            st.metric("Capital Required", f"${info.get('entry_capital_required', 0):,.2f}")

        with col3:
            st.metric("Concurrent Positions", info.get('concurrent_positions', 0))

        # Stop loss info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Initial Stop", f"${info.get('initial_stop_loss', 0):.4f}")

        with col2:
            st.metric("Final Stop", f"${info.get('final_stop_loss', 0):.4f}")

        with col3:
            tp = info.get('take_profit')
            st.metric("Take Profit", f"${tp:.4f}" if tp else "N/A")


# =============================================================================
# TAB 2: PRICE ACTION & TECHNICALS
# =============================================================================

def render_price_action_tab(trade: TradeAnalysisData):
    """Render the price action and technicals tab."""

    if trade.price_data is None or len(trade.price_data) == 0:
        st.warning("No price data available for this trade.")
        return

    st.subheader("Pre-Entry Technical Snapshot")

    # Get entry date data
    entry_date = pd.to_datetime(trade.trade_info.get('entry_date'))
    prices = trade.price_data.copy()
    prices['date'] = pd.to_datetime(prices['date'])

    # Get entry row
    entry_mask = prices['date'] == entry_date
    if not entry_mask.any():
        # Find closest date
        prices['date_diff'] = abs((prices['date'] - entry_date).dt.days)
        entry_idx = prices['date_diff'].idxmin()
        entry_row = prices.loc[entry_idx]
    else:
        entry_row = prices[entry_mask].iloc[0]

    # Display key technical indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        atr = entry_row.get('atr_14_atr') or entry_row.get('atr')
        st.metric("ATR (14)", f"{atr:.4f}" if pd.notna(atr) else "N/A")

    with col2:
        rsi = entry_row.get('rsi_14_rsi') or entry_row.get('rsi')
        st.metric("RSI (14)", f"{rsi:.1f}" if pd.notna(rsi) else "N/A")

    with col3:
        macd = entry_row.get('macd_14_macd') or entry_row.get('macd')
        st.metric("MACD", f"{macd:.4f}" if pd.notna(macd) else "N/A")

    with col4:
        volume = entry_row.get('volume')
        st.metric("Volume", f"{volume:,.0f}" if pd.notna(volume) else "N/A")

    # SMA distances
    st.subheader("Moving Average Analysis")

    close = entry_row.get('close', 0)

    sma_data = []
    for sma_col, label in [('sma_20_sma', 'SMA 20'), ('sma_50_sma', 'SMA 50'), ('sma_200_sma', 'SMA 200')]:
        sma_val = entry_row.get(sma_col)
        if pd.notna(sma_val) and close > 0:
            distance_pct = ((close - sma_val) / sma_val) * 100
            sma_data.append({
                'MA': label,
                'Value': f"${sma_val:.4f}",
                'Distance': f"{distance_pct:+.2f}%",
                'Position': 'Above' if close > sma_val else 'Below'
            })

    if sma_data:
        st.table(pd.DataFrame(sma_data))

    # Bollinger Band position
    st.subheader("Bollinger Bands Position")

    bb_upper = entry_row.get('bbands_20_real upper band')
    bb_lower = entry_row.get('bbands_20_real lower band')
    bb_middle = entry_row.get('bbands_20_real middle band')

    if pd.notna(bb_upper) and pd.notna(bb_lower):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            position = (close - bb_lower) / bb_range

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lower Band", f"${bb_lower:.4f}")
            with col2:
                st.metric("Price Position", f"{position:.1%} of range")
            with col3:
                st.metric("Upper Band", f"${bb_upper:.4f}")

    # Market Regime
    if trade.market_regime:
        st.subheader("Market Regime at Entry")

        col1, col2, col3 = st.columns(3)

        with col1:
            trend_emoji = {"uptrend": "", "downtrend": "", "ranging": ""}
            st.metric("Trend", f"{trend_emoji.get(trade.market_regime.trend, '')} {trade.market_regime.trend.title()}")

        with col2:
            vol_emoji = {"high": "", "low": "", "normal": ""}
            st.metric("Volatility", f"{vol_emoji.get(trade.market_regime.volatility, '')} {trade.market_regime.volatility.title()}")

        with col3:
            st.metric("SMA Alignment", trade.market_regime.sma_alignment.title())

    # Price chart placeholder
    with st.expander("Price Data Table"):
        display_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        available = [c for c in display_cols if c in prices.columns]
        display_prices = prices[available].copy()
        display_prices['date'] = display_prices['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_prices.tail(30), use_container_width=True)


# =============================================================================
# TAB 3: FUNDAMENTALS
# =============================================================================

def render_fundamentals_tab(trade: TradeAnalysisData):
    """Render the fundamentals analysis tab."""

    if not trade.fundamentals_entry and not trade.fundamentals_exit:
        st.warning("No fundamental data available for this symbol.")
        st.info("Check if fundamental data exists in raw_data/fundamentals/")
        return

    st.subheader("Fundamental Comparison: Entry vs Exit")

    # Build comparison table
    metrics = [
        ('pe_ratio', 'P/E Ratio'),
        ('eps', 'EPS'),
        ('earnings_growth_yoy', 'Earnings Growth YoY %'),
        ('revenue_growth_yoy', 'Revenue Growth YoY %'),
        ('profit_margin', 'Profit Margin'),
        ('return_on_assets', 'Return on Assets'),
        ('return_on_equity', 'Return on Equity'),
    ]

    entry_data = trade.fundamentals_entry or {}
    exit_data = trade.fundamentals_exit or {}
    delta_data = trade.fundamentals_delta or {}

    comparison_data = []
    for key, label in metrics:
        entry_val = entry_data.get(key)
        exit_val = exit_data.get(key)
        delta = delta_data.get(key)

        row = {
            'Metric': label,
            'Entry': f"{entry_val:.2f}" if pd.notna(entry_val) else "N/A",
            'Exit': f"{exit_val:.2f}" if pd.notna(exit_val) else "N/A",
            'Change': f"{delta:+.2f}" if pd.notna(delta) else "N/A",
        }

        # Determine status
        if pd.notna(delta):
            if delta > 0:
                row['Status'] = ' Improved'
            elif delta < 0:
                row['Status'] = ' Declined'
            else:
                row['Status'] = ' Stable'
        else:
            row['Status'] = ' Unknown'

        comparison_data.append(row)

    st.table(pd.DataFrame(comparison_data))

    # Narrative summary
    st.subheader("Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**At Entry:**")
        entry_date = entry_data.get('date', 'N/A')
        if isinstance(entry_date, datetime):
            entry_date = entry_date.strftime('%Y-%m-%d')
        st.write(f"Data as of: {entry_date}")

    with col2:
        st.markdown("**At Exit:**")
        exit_date = exit_data.get('date', 'N/A')
        if isinstance(exit_date, datetime):
            exit_date = exit_date.strftime('%Y-%m-%d')
        st.write(f"Data as of: {exit_date}")


# =============================================================================
# TAB 4: INSIDER ACTIVITY
# =============================================================================

def render_insider_tab(trade: TradeAnalysisData):
    """Render the insider activity tab."""

    if trade.insider_activity is None or len(trade.insider_activity) == 0:
        st.warning("No insider transaction data available for this symbol.")
        st.info("Check if insider data exists in raw_data/insider_transactions/")
        return

    # Display flags
    if trade.insider_flags:
        st.subheader("Insider Activity Flags")
        for flag in trade.insider_flags:
            if 'BUYING' in flag:
                st.success(flag)
            elif 'SELLING' in flag:
                st.warning(flag)
            else:
                st.info(flag)

    # Summary metrics
    st.subheader("Activity Summary")

    insider_df = trade.insider_activity.copy()

    if 'transaction_type' in insider_df.columns and 'value' in insider_df.columns:
        buys = insider_df[insider_df['transaction_type'].str.upper() == 'BUY']
        sells = insider_df[insider_df['transaction_type'].str.upper() == 'SELL']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Transactions", len(insider_df))

        with col2:
            st.metric("Buy Transactions", len(buys))

        with col3:
            st.metric("Sell Transactions", len(sells))

        with col4:
            net_value = buys['value'].sum() - sells['value'].sum()
            st.metric("Net Activity", f"${net_value:,.0f}")

    # Transaction table
    st.subheader("Transaction History")

    display_cols = ['date', 'executive', 'insider_title', 'transaction_type', 'shares', 'price', 'value']
    available = [c for c in display_cols if c in insider_df.columns]
    display_df = insider_df[available].copy()

    if 'date' in display_df.columns:
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')

    if 'value' in display_df.columns:
        display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    if 'shares' in display_df.columns:
        display_df['shares'] = display_df['shares'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Pattern analysis
    st.subheader("Pattern Interpretation")

    if trade.insider_flags:
        if any('BUYING_PREENTRY' in f for f in trade.insider_flags):
            st.success("Insider buying before your entry may indicate bullish insider sentiment.")

        if any('SELLING_DURING' in f for f in trade.insider_flags):
            st.warning("Insider selling during your trade may indicate headwinds.")

        if any('COORDINATED' in f for f in trade.insider_flags):
            st.info("Multiple insiders acting together strengthens the signal.")
    else:
        st.write("No significant insider patterns detected for this trade period.")


# =============================================================================
# TAB 5: OPTIONS MARKET
# =============================================================================

def render_options_tab(trade: TradeAnalysisData):
    """Render the options market analysis tab."""

    if not trade.options_data:
        st.warning("No options data available for this symbol.")
        st.info("Options data is optional and may not exist for all symbols.")
        return

    opts = trade.options_data

    st.subheader("Implied Volatility Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        iv_entry = opts.get('iv_at_entry')
        st.metric("IV at Entry", f"{iv_entry:.1f}%" if iv_entry else "N/A")

    with col2:
        iv_pct = opts.get('iv_percentile_entry')
        st.metric("IV Percentile", f"{iv_pct:.0f}th" if iv_pct else "N/A")

    with col3:
        iv_exit = opts.get('iv_at_exit')
        st.metric("IV at Exit", f"{iv_exit:.1f}%" if iv_exit else "N/A")

    # Put/Call ratio
    st.subheader("Market Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        pcr = opts.get('put_call_ratio_entry')
        st.metric("Put/Call Ratio (Entry)", f"{pcr:.2f}" if pcr else "N/A")

    with col2:
        earnings = opts.get('earnings_during_trade')
        st.metric("Earnings During Trade", "Yes" if earnings else "No")

    # Interpretation
    st.subheader("Interpretation")

    iv_pct = opts.get('iv_percentile_entry')
    if iv_pct:
        if iv_pct >= 75:
            st.warning(
                "High IV at entry suggests elevated volatility expectations. "
                "Price may experience large moves but could also revert if volatility contracts."
            )
        elif iv_pct <= 25:
            st.info(
                "Low IV at entry suggests calm market expectations. "
                "Moves may be muted, but any surprises could trigger larger reactions."
            )
        else:
            st.success(
                "Normal IV environment. Standard volatility expectations priced in."
            )

    if opts.get('earnings_during_trade'):
        st.warning(
            "Earnings announcement occurred during your trade. "
            "This catalyst may have significantly impacted the outcome."
        )


# =============================================================================
# TAB 6: SECTOR & MARKET CORRELATION
# =============================================================================

def render_correlation_tab(trade: TradeAnalysisData):
    """Render the sector and market correlation tab."""

    st.subheader("Correlation Analysis")

    sector_corr = trade.sector_correlation
    index_corr = trade.index_correlation

    if sector_corr is None and index_corr is None:
        st.warning("Correlation data not available.")
        st.info("Ensure sector/index price data exists in raw_data/daily/")
        return

    col1, col2 = st.columns(2)

    with col1:
        if sector_corr is not None:
            # Color based on strength
            if abs(sector_corr) > 0.7:
                corr_color = "green"
            elif abs(sector_corr) > 0.3:
                corr_color = "orange"
            else:
                corr_color = "red"

            st.metric("Sector Correlation", f"{sector_corr:.3f}")

            if abs(sector_corr) > 0.85:
                st.success("Strong correlation with sector")
            elif abs(sector_corr) > 0.5:
                st.info("Moderate correlation with sector")
            else:
                st.warning("Weak correlation - idiosyncratic movement")
        else:
            st.metric("Sector Correlation", "N/A")

    with col2:
        if index_corr is not None:
            st.metric("Index (SPX) Correlation", f"{index_corr:.3f}")

            if abs(index_corr) > 0.7:
                st.success("Moves with the broader market")
            else:
                st.info("Less sensitive to market movements")
        else:
            st.metric("Index Correlation", "N/A")

    # Analysis
    st.subheader("Strategic Implications")

    if sector_corr is not None and abs(sector_corr) > 0.85:
        st.info(
            "With high sector correlation, consider whether trading the sector ETF "
            "could provide similar exposure with lower single-stock risk."
        )

    if trade.correlation_analysis:
        st.write(trade.correlation_analysis)


# =============================================================================
# TAB 7: MAE/MFE ANALYSIS
# =============================================================================

def render_mae_mfe_tab(trade: TradeAnalysisData):
    """Render the MAE/MFE analysis tab."""

    if not trade.mae_mfe:
        st.warning("MAE/MFE data not available.")
        st.info("This requires sufficient intraday price data during the trade period.")
        return

    mf = trade.mae_mfe

    st.subheader("Maximum Adverse Excursion (MAE)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE %", f"{mf.mae_pct:.2f}%")

    with col2:
        mae_date = mf.mae_date.strftime('%Y-%m-%d') if mf.mae_date else 'N/A'
        st.metric("Date", mae_date)

    with col3:
        st.metric("Days Into Trade", mf.mae_days_into_trade)

    st.subheader("Maximum Favorable Excursion (MFE)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MFE %", f"{mf.mfe_pct:.2f}%")

    with col2:
        mfe_date = mf.mfe_date.strftime('%Y-%m-%d') if mf.mfe_date else 'N/A'
        st.metric("Date", mfe_date)

    with col3:
        st.metric("Days Into Trade", mf.mfe_days_into_trade)

    # Comparison
    st.subheader("Trade Efficiency")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Actual P&L %", f"{mf.actual_pl_pct:.2f}%")

    with col2:
        st.metric("MFE Capture", f"{mf.mfe_capture_pct:.1f}%",
                  help="Percentage of maximum opportunity captured")

    with col3:
        ratio = mf.mfe_pct / mf.mae_pct if mf.mae_pct > 0 else 0
        st.metric("MFE/MAE Ratio", f"{ratio:.2f}",
                  help="Risk/reward ratio experienced during trade")

    # Interpretation
    st.subheader("Interpretation")

    st.write(f"- You weathered a **{mf.mae_pct:.2f}%** drawdown during this trade.")
    st.write(f"- The maximum opportunity was **{mf.mfe_pct:.2f}%**.")
    st.write(f"- You captured **{mf.mfe_capture_pct:.1f}%** of that opportunity.")

    if mf.mfe_capture_pct < 50 and mf.actual_pl_pct > 0:
        st.info("Consider whether exits could be optimized - you left significant gains on the table.")

    if mf.mae_pct > mf.mfe_pct:
        st.warning("MAE exceeded MFE - the trade spent more time underwater than profitable.")


# =============================================================================
# TAB 8: PATTERN ANALYSIS
# =============================================================================

def render_pattern_analysis_tab(
    trades_data: List[TradeAnalysisData],
    signal_scores: Dict[str, SignalStrengthScore],
    pattern_flags: Dict[str, List[PatternFlag]],
    pattern_summary: Optional[PatternSummary] = None
):
    """Render the aggregate pattern analysis tab."""

    if not pattern_summary:
        st.info("Pattern summary not available. Analyze multiple trades to see aggregate patterns.")
        return

    st.subheader("Aggregate Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", pattern_summary.total_trades)

    with col2:
        st.metric("Winners", pattern_summary.winners)

    with col3:
        st.metric("Losers", pattern_summary.losers)

    with col4:
        st.metric("Win Rate", f"{pattern_summary.win_rate:.1f}%")

    # Win rate by signal strength
    st.subheader("Win Rate by Signal Strength")

    signal_data = []
    for tier, (wins, total, rate) in pattern_summary.win_rate_by_signal_strength.items():
        if total > 0:
            signal_data.append({
                'Tier': tier.title(),
                'Wins': wins,
                'Total': total,
                'Win Rate': f"{rate:.1f}%"
            })

    if signal_data:
        st.table(pd.DataFrame(signal_data))

    # Win rate by market regime
    st.subheader("Win Rate by Market Regime")

    regime_data = []
    for regime, (wins, total, rate) in pattern_summary.win_rate_by_market_regime.items():
        if total > 0:
            regime_data.append({
                'Regime': regime.title(),
                'Wins': wins,
                'Total': total,
                'Win Rate': f"{rate:.1f}%"
            })

    if regime_data:
        st.table(pd.DataFrame(regime_data))

    # MAE/MFE comparison
    st.subheader("MAE/MFE by Outcome")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Winners**")
        st.write(f"- Avg MAE: {pattern_summary.avg_mae_winners:.2f}%")
        st.write(f"- Avg MFE: {pattern_summary.avg_mfe_winners:.2f}%")
        st.write(f"- Avg Duration: {pattern_summary.avg_duration_winners:.1f} days")

    with col2:
        st.markdown("**Losers**")
        st.write(f"- Avg MAE: {pattern_summary.avg_mae_losers:.2f}%")
        st.write(f"- Avg MFE: {pattern_summary.avg_mfe_losers:.2f}%")
        st.write(f"- Avg Duration: {pattern_summary.avg_duration_losers:.1f} days")

    # Common patterns
    st.subheader("Pattern Identification")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Common in Winners:**")
        for pattern, count in pattern_summary.common_patterns_winners:
            st.write(f"- {pattern}: {count} trades")

    with col2:
        st.markdown("**Common in Losers:**")
        for pattern, count in pattern_summary.common_patterns_losers:
            st.write(f"- {pattern}: {count} trades")

    # Key insights
    st.subheader("Key Insights")

    for insight in pattern_summary.insights:
        st.info(insight)
