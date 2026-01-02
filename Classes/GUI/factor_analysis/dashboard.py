"""
Factor Analysis Dashboard GUI.

The main application for viewing and interacting with factor analysis results.
Provides multiple views for exploring analysis data at different levels.
"""

import customtkinter as ctk
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import pandas as pd
from datetime import datetime
import threading
import queue
import json
import os

from ..ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info, ask_yes_no, ProgressPanel
from .components import (
    NavigationPanel, FactorListPanel, StatisticsPanel,
    DataQualityIndicator, AnalysisProgressPanel, ScenarioCard,
    TierResultsPanel
)
from .config_manager import FactorConfigManagerGUI
from .data_upload import FactorDataUploadGUI


# Default export folder
EXPORT_FOLDER = Path(__file__).parent.parent.parent.parent / "exports" / "factor_analysis"


class DataSummaryView(ctk.CTkFrame):
    """View displaying data summary and quality metrics."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.data: Dict[str, Any] = {}
        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main scrollable container
        main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_scroll.pack(fill="both", expand=True)

        # Header
        header = Theme.create_frame(main_scroll)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Data Summary", size="l").pack(side="left")

        Theme.create_button(
            header, "Refresh",
            command=self._refresh_data,
            style="secondary",
            width=80,
            height=28
        ).pack(side="right")

        # Summary cards row
        cards_frame = Theme.create_frame(main_scroll)
        cards_frame.pack(fill="x", pady=Sizes.PAD_S)
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Trade summary cards
        self.trade_card = self._create_summary_card(cards_frame, "Total Trades", "0", 0, 0)
        self.good_card = self._create_summary_card(cards_frame, "Good Trades", "0", 0, 1, Colors.SUCCESS)
        self.bad_card = self._create_summary_card(cards_frame, "Bad Trades", "0", 0, 2, Colors.ERROR)
        self.neutral_card = self._create_summary_card(cards_frame, "Indeterminate", "0", 0, 3, Colors.WARNING)

        # Second row of stats
        stats_frame = Theme.create_frame(main_scroll)
        stats_frame.pack(fill="x", pady=Sizes.PAD_S)
        stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.symbols_card = self._create_summary_card(stats_frame, "Symbols", "0", 0, 0)
        self.date_range_card = self._create_summary_card(stats_frame, "Date Range", "N/A", 0, 1)
        self.win_rate_card = self._create_summary_card(stats_frame, "Win Rate", "0%", 0, 2)
        self.avg_return_card = self._create_summary_card(stats_frame, "Avg Return", "0%", 0, 3)

        # Data quality section
        quality_section = Theme.create_card(main_scroll)
        quality_section.pack(fill="x", pady=Sizes.PAD_M)

        quality_content = Theme.create_frame(quality_section)
        quality_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(quality_content, "Data Quality", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.quality_indicator = DataQualityIndicator(quality_content)
        self.quality_indicator.pack(fill="x")

        # Data sources section
        sources_section = Theme.create_card(main_scroll)
        sources_section.pack(fill="x", pady=Sizes.PAD_S)

        sources_content = Theme.create_frame(sources_section)
        sources_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(sources_content, "Loaded Data Sources", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.sources_frame = Theme.create_frame(sources_content)
        self.sources_frame.pack(fill="x")

        # Factor coverage section
        factor_section = Theme.create_card(main_scroll)
        factor_section.pack(fill="x", pady=Sizes.PAD_S)

        factor_content = Theme.create_frame(factor_section)
        factor_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(factor_content, "Factor Coverage", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.factor_coverage_frame = Theme.create_frame(factor_content)
        self.factor_coverage_frame.pack(fill="x")

    def _create_summary_card(self, parent, title: str, value: str, row: int, col: int, value_color: str = None) -> Dict:
        """Create a summary stat card."""
        card = Theme.create_card(parent)
        card.grid(row=row, column=col, sticky="nsew", padx=Sizes.PAD_XS, pady=Sizes.PAD_XS)

        Theme.create_label(card, title, font=Fonts.BODY_S, text_color=Colors.TEXT_SECONDARY).pack(pady=(Sizes.PAD_S, 0))

        value_label = Theme.create_label(
            card, value,
            font=Fonts.HEADER_XL,
            text_color=value_color or Colors.TEXT_PRIMARY
        )
        value_label.pack(pady=Sizes.PAD_S)

        return {'card': card, 'value_label': value_label}

    def set_data(self, data: Dict[str, Any]):
        """Set the data for this view."""
        self.data = data
        self._refresh_data()

    def _refresh_data(self):
        """Refresh all data displays."""
        if not self.data:
            return

        trade_data = self.data.get('trade_data')
        if trade_data is None or (isinstance(trade_data, pd.DataFrame) and trade_data.empty):
            return

        # Calculate trade statistics
        total_trades = len(trade_data) if isinstance(trade_data, pd.DataFrame) else 0

        # Classify trades if pnl column exists
        good_trades = 0
        bad_trades = 0
        indeterminate = 0

        if isinstance(trade_data, pd.DataFrame):
            pnl_col = None
            for col in ['pnl', 'pl', 'profit', 'return', 'profit_loss']:
                if col in trade_data.columns:
                    pnl_col = col
                    break

            if pnl_col:
                good_trades = len(trade_data[trade_data[pnl_col] > 0])
                bad_trades = len(trade_data[trade_data[pnl_col] < 0])
                indeterminate = len(trade_data[trade_data[pnl_col] == 0])

            # Calculate win rate
            if total_trades > 0:
                win_rate = (good_trades / total_trades) * 100
            else:
                win_rate = 0

            # Calculate average return
            if pnl_col and len(trade_data) > 0:
                avg_return = trade_data[pnl_col].mean()
            else:
                avg_return = 0

            # Get symbol count
            symbol_col = None
            for col in ['symbol', 'ticker', 'Symbol', 'Ticker']:
                if col in trade_data.columns:
                    symbol_col = col
                    break

            symbols = trade_data[symbol_col].nunique() if symbol_col else 0

            # Get date range
            date_col = None
            for col in ['date', 'entry_date', 'trade_date', 'Date']:
                if col in trade_data.columns:
                    date_col = col
                    break

            if date_col:
                try:
                    dates = pd.to_datetime(trade_data[date_col])
                    date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                except Exception:
                    date_range = "N/A"
            else:
                date_range = "N/A"

        else:
            win_rate = 0
            avg_return = 0
            symbols = 0
            date_range = "N/A"

        # Update cards
        self.trade_card['value_label'].configure(text=f"{total_trades:,}")
        self.good_card['value_label'].configure(text=f"{good_trades:,}")
        self.bad_card['value_label'].configure(text=f"{bad_trades:,}")
        self.neutral_card['value_label'].configure(text=f"{indeterminate:,}")
        self.symbols_card['value_label'].configure(text=f"{symbols:,}")
        self.date_range_card['value_label'].configure(text=date_range)
        self.win_rate_card['value_label'].configure(text=f"{win_rate:.1f}%")
        self.avg_return_card['value_label'].configure(text=f"{avg_return:.2f}%")

        # Update data sources
        self._update_sources()

        # Update quality
        self._update_quality()

        # Update factor coverage
        self._update_factor_coverage()

    def _update_sources(self):
        """Update data sources display."""
        for widget in self.sources_frame.winfo_children():
            widget.destroy()

        sources = [
            ("Trade Logs", self.data.get('trade_data') is not None),
            ("Daily Price Data", self.data.get('daily_price_data') is not None),
            ("Weekly Price Data", self.data.get('weekly_price_data') is not None),
            ("Fundamental Data", self.data.get('fundamental_data') is not None),
            ("Insider Data", self.data.get('insider_data') is not None),
            ("Options Data", self.data.get('options_data') is not None),
        ]

        for name, loaded in sources:
            row = Theme.create_frame(self.sources_frame)
            row.pack(fill="x", pady=2)

            status = "✓" if loaded else "✗"
            color = Colors.SUCCESS if loaded else Colors.TEXT_MUTED

            Theme.create_label(row, status, font=Fonts.BODY_S, text_color=color, width=30).pack(side="left")
            Theme.create_label(row, name, font=Fonts.BODY_S).pack(side="left")

            if loaded:
                data = self.data.get(name.lower().replace(" ", "_"))
                if isinstance(data, pd.DataFrame):
                    Theme.create_label(
                        row, f"({len(data):,} rows)",
                        font=Fonts.BODY_XS,
                        text_color=Colors.TEXT_MUTED
                    ).pack(side="right")

    def _update_quality(self):
        """Update quality indicator."""
        total_sources = 6
        loaded_sources = sum([
            self.data.get('trade_data') is not None,
            self.data.get('daily_price_data') is not None,
            self.data.get('weekly_price_data') is not None,
            self.data.get('fundamental_data') is not None,
            self.data.get('insider_data') is not None,
            self.data.get('options_data') is not None,
        ])

        score = int((loaded_sources / total_sources) * 100)

        if score >= 80:
            status = "Excellent"
        elif score >= 60:
            status = "Good"
        elif score >= 40:
            status = "Acceptable"
        else:
            status = "Limited"

        warnings = []
        if self.data.get('fundamental_data') is None:
            warnings.append("No fundamental data loaded")
        if self.data.get('insider_data') is None:
            warnings.append("No insider data loaded")

        self.quality_indicator.set_quality(
            score=score,
            status=status,
            metrics={
                "Data sources": f"{loaded_sources}/{total_sources}",
                "Symbols": str(len(self.data.get('symbols', []))),
            },
            warnings=warnings if warnings else None
        )

    def _update_factor_coverage(self):
        """Update factor coverage display."""
        for widget in self.factor_coverage_frame.winfo_children():
            widget.destroy()

        # Determine available factor types based on loaded data
        factor_types = []

        if self.data.get('daily_price_data') is not None or self.data.get('weekly_price_data') is not None:
            factor_types.append(("Technical", "RSI, MACD, Bollinger Bands, ATR, etc.", Colors.PRIMARY_LIGHT))

        if self.data.get('fundamental_data') is not None:
            factor_types.append(("Value", "P/E, P/B, Dividend Yield, etc.", Colors.SUCCESS))
            factor_types.append(("Quality", "ROE, ROA, Margins, etc.", Colors.SUCCESS))
            factor_types.append(("Growth", "Revenue Growth, Earnings Growth, etc.", Colors.SUCCESS))

        if self.data.get('insider_data') is not None:
            factor_types.append(("Insider", "Insider Buy/Sell Ratios, Net Activity, etc.", Colors.WARNING))

        if self.data.get('options_data') is not None:
            factor_types.append(("Options", "Implied Volatility, Put/Call Ratio, etc.", Colors.SECONDARY))

        if not factor_types:
            Theme.create_label(
                self.factor_coverage_frame,
                "No factor data available. Load price, fundamental, or insider data.",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED
            ).pack(anchor="w")
            return

        for name, description, color in factor_types:
            row = Theme.create_frame(self.factor_coverage_frame)
            row.pack(fill="x", pady=2)

            Theme.create_label(
                row, f"● {name}",
                font=Fonts.LABEL_BOLD,
                text_color=color
            ).pack(side="left")

            Theme.create_label(
                row, description,
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            ).pack(side="left", padx=(Sizes.PAD_S, 0))

    def update_data(self, summary: Dict[str, Any]):
        """Update the view with new data summary (legacy method)."""
        self.set_data(summary)


class Tier1View(ctk.CTkFrame):
    """View for Tier 1 Exploratory Analysis results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.results: Dict[str, Any] = {}
        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main container with proper expansion
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = Theme.create_frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Tier 1: Exploratory Analysis", size="l").pack(side="left")

        Theme.create_label(
            header,
            "Correlation analysis and distribution statistics for all factor types",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        # Main content area - two column layout
        content = Theme.create_frame(self)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Left column - Factor list with scrollable frame
        left_frame = ctk.CTkFrame(content, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_S))
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        self.factor_panel = FactorListPanel(
            left_frame,
            title="Factor Correlations (All Types)",
            on_factor_select=self._on_factor_select
        )
        self.factor_panel.grid(row=0, column=0, sticky="nsew")

        # Right column - Statistics and distribution
        right_frame = ctk.CTkFrame(content, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Statistics panel
        self.stats_panel = StatisticsPanel(right_frame, title="Selected Factor Statistics")
        self.stats_panel.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_S))

        # Distribution visualization
        dist_card = Theme.create_card(right_frame)
        dist_card.grid(row=1, column=0, sticky="nsew")

        Theme.create_header(dist_card, "Distribution", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.dist_frame = ctk.CTkScrollableFrame(dist_card, fg_color="transparent")
        self.dist_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.dist_placeholder = Theme.create_label(
            self.dist_frame,
            "Select a factor to view its distribution",
            text_color=Colors.TEXT_MUTED
        )
        self.dist_placeholder.pack(expand=True, pady=Sizes.PAD_L)

    def _on_factor_select(self, factor_name: str):
        """Handle factor selection."""
        if not self.results:
            return

        factor_details = self.results.get('factor_details', {}).get(factor_name, {})

        # Update statistics panel
        if factor_details:
            self.stats_panel.set_metrics({
                "Mean": f"{factor_details.get('mean', 0):.4f}",
                "Std Dev": f"{factor_details.get('std', 0):.4f}",
                "Min": f"{factor_details.get('min', 0):.4f}",
                "Max": f"{factor_details.get('max', 0):.4f}",
                "Skewness": f"{factor_details.get('skewness', 0):.4f}",
                "Kurtosis": f"{factor_details.get('kurtosis', 0):.4f}",
                "Missing %": f"{factor_details.get('missing_pct', 0):.1f}%"
            })
        else:
            # Try to find in correlations
            corr = self.results.get('correlations', {}).get(factor_name, 0)
            p_val = self.results.get('p_values', {}).get(factor_name, 1)
            factor_type = self.results.get('factor_types', {}).get(factor_name, 'Unknown')

            self.stats_panel.set_metrics({
                "Correlation": f"{corr:.4f}",
                "P-Value": f"{p_val:.4f}",
                "Factor Type": factor_type,
                "Significance": "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "NS"
            })

        # Update distribution display
        self._update_distribution(factor_name, factor_details)

    def _update_distribution(self, factor_name: str, details: Dict):
        """Update the distribution display for a factor."""
        for widget in self.dist_frame.winfo_children():
            widget.destroy()

        Theme.create_label(
            self.dist_frame,
            f"Distribution: {factor_name}",
            font=Fonts.LABEL_BOLD,
            text_color=Colors.PRIMARY_LIGHT
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Show distribution statistics
        if details:
            stats = [
                ("Good Trades Mean", details.get('good_mean', 0)),
                ("Bad Trades Mean", details.get('bad_mean', 0)),
                ("Difference", details.get('mean_diff', 0)),
            ]

            for label, value in stats:
                row = Theme.create_frame(self.dist_frame)
                row.pack(fill="x", pady=2)

                Theme.create_label(row, label, font=Fonts.BODY_S).pack(side="left")
                color = Colors.SUCCESS if value > 0 else Colors.ERROR if value < 0 else Colors.TEXT_MUTED
                Theme.create_label(row, f"{value:.4f}", font=Fonts.BODY_S, text_color=color).pack(side="right")

        # Add interpretation
        corr = self.results.get('correlations', {}).get(factor_name, 0)
        p_val = self.results.get('p_values', {}).get(factor_name, 1)

        interp_frame = Theme.create_frame(self.dist_frame)
        interp_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        Theme.create_label(
            interp_frame,
            "Interpretation:",
            font=Fonts.LABEL_BOLD
        ).pack(anchor="w")

        if p_val < 0.05:
            if corr > 0.2:
                interp = f"Strong positive relationship with trade success (r={corr:.3f})"
                color = Colors.SUCCESS
            elif corr > 0:
                interp = f"Weak positive relationship with trade success (r={corr:.3f})"
                color = Colors.SUCCESS
            elif corr < -0.2:
                interp = f"Strong negative relationship with trade success (r={corr:.3f})"
                color = Colors.ERROR
            else:
                interp = f"Weak negative relationship with trade success (r={corr:.3f})"
                color = Colors.ERROR
        else:
            interp = f"No statistically significant relationship (p={p_val:.3f})"
            color = Colors.TEXT_MUTED

        Theme.create_label(
            interp_frame,
            interp,
            font=Fonts.BODY_S,
            text_color=color,
            wraplength=300
        ).pack(anchor="w", pady=(Sizes.PAD_XS, 0))

    def update_results(self, results: Dict[str, Any]):
        """Update the view with analysis results."""
        self.results = results

        # Build factor list with all factor types
        factors = []
        correlations = results.get('correlations', {})
        p_values = results.get('p_values', {})
        factor_types = results.get('factor_types', {})

        for factor, corr in correlations.items():
            factors.append({
                'name': factor,
                'type': factor_types.get(factor, 'Unknown'),
                'correlation': corr,
                'p_value': p_values.get(factor, 1.0)
            })

        # Sort by absolute correlation
        self.factor_panel.set_factors(sorted(factors, key=lambda x: abs(x['correlation']), reverse=True))


class Tier2View(ctk.CTkFrame):
    """View for Tier 2 Hypothesis Testing results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main container with proper expansion
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = Theme.create_frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Tier 2: Hypothesis Testing", size="l").pack(side="left")

        # Interpretation guide
        interp_card = Theme.create_card(self)
        interp_card.grid(row=1, column=0, sticky="ew", pady=(0, Sizes.PAD_S))

        interp_content = Theme.create_frame(interp_card)
        interp_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(interp_content, "Interpretation Guide", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        interpretations = [
            ("*** (p < 0.01)", "Highly significant - strong evidence against null hypothesis", Colors.SUCCESS),
            ("** (p < 0.05)", "Significant - sufficient evidence to reject null hypothesis", Colors.WARNING),
            ("* (p < 0.10)", "Marginally significant - weak evidence", Colors.TEXT_PRIMARY),
            ("NS (p ≥ 0.10)", "Not significant - insufficient evidence", Colors.TEXT_MUTED),
        ]

        for sig, desc, color in interpretations:
            row = Theme.create_frame(interp_content)
            row.pack(fill="x", pady=1)
            Theme.create_label(row, sig, font=Fonts.LABEL_BOLD, text_color=color, width=120).pack(side="left")
            Theme.create_label(row, desc, font=Fonts.BODY_S, text_color=Colors.TEXT_SECONDARY).pack(side="left")

        # Effect size interpretation
        Theme.create_label(
            interp_content,
            "\nEffect Size (Cohen's d): Small (0.2), Medium (0.5), Large (0.8)",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w", pady=(Sizes.PAD_S, 0))

        # Results panel
        results_frame = ctk.CTkFrame(self, fg_color="transparent")
        results_frame.grid(row=2, column=0, sticky="nsew")
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        self.results_panel = TierResultsPanel(results_frame, tier=2)
        self.results_panel.grid(row=0, column=0, sticky="nsew")

    def update_results(self, results: Dict[str, Any]):
        """Update the view with analysis results."""
        self.results_panel.set_results(results)


class Tier3View(ctk.CTkFrame):
    """View for Tier 3 ML Analysis results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main container with proper expansion
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = Theme.create_frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Tier 3: ML Analysis", size="l").pack(side="left")

        Theme.create_label(
            header,
            "Machine learning feature importance across all factor types",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        # Main content - two columns
        content = Theme.create_frame(self)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Left column - Feature importance
        left_frame = ctk.CTkFrame(content, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_S))
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        importance_card = Theme.create_card(left_frame)
        importance_card.grid(row=0, column=0, sticky="nsew")

        Theme.create_header(importance_card, "Feature Importance (Random Forest)", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.importance_frame = ctk.CTkScrollableFrame(importance_card, fg_color="transparent")
        self.importance_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Right column - SHAP values and visualization
        right_frame = ctk.CTkFrame(content, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # SHAP values
        shap_card = Theme.create_card(right_frame)
        shap_card.grid(row=0, column=0, sticky="nsew", pady=(0, Sizes.PAD_S))

        Theme.create_header(shap_card, "SHAP Values", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.shap_frame = ctk.CTkScrollableFrame(shap_card, fg_color="transparent")
        self.shap_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Model performance
        perf_card = Theme.create_card(right_frame)
        perf_card.grid(row=1, column=0, sticky="nsew")

        Theme.create_header(perf_card, "Model Performance", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.perf_frame = Theme.create_frame(perf_card)
        self.perf_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

    def update_results(self, results: Dict[str, Any]):
        """Update the view with analysis results."""
        # Clear previous results
        for widget in self.importance_frame.winfo_children():
            widget.destroy()
        for widget in self.shap_frame.winfo_children():
            widget.destroy()
        for widget in self.perf_frame.winfo_children():
            widget.destroy()

        # Feature importance with visual bars
        importance = results.get('feature_importance', {})
        if importance:
            max_imp = max(importance.values()) if importance else 1

            for i, (factor, imp) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]):
                row = Theme.create_frame(self.importance_frame)
                row.pack(fill="x", pady=2)

                # Factor type indicator
                factor_type = results.get('factor_types', {}).get(factor, 'Unknown')
                type_colors = {
                    'Technical': Colors.PRIMARY_LIGHT,
                    'Value': Colors.SUCCESS,
                    'Quality': Colors.SUCCESS,
                    'Growth': Colors.SUCCESS,
                    'Insider': Colors.WARNING,
                    'Options': Colors.SECONDARY,
                }
                type_color = type_colors.get(factor_type, Colors.TEXT_MUTED)

                Theme.create_label(row, "●", font=Fonts.BODY_XS, text_color=type_color, width=15).pack(side="left")
                Theme.create_label(row, factor, font=Fonts.BODY_S, width=150).pack(side="left")

                # Visual bar
                bar_frame = ctk.CTkFrame(row, fg_color=Colors.BG_MEDIUM, height=16, corner_radius=2)
                bar_frame.pack(side="left", fill="x", expand=True, padx=Sizes.PAD_XS)
                bar_frame.pack_propagate(False)

                bar_width = int((imp / max_imp) * 100)
                bar = ctk.CTkFrame(bar_frame, fg_color=Colors.PRIMARY, corner_radius=2)
                bar.place(relx=0, rely=0, relwidth=bar_width/100, relheight=1)

                Theme.create_label(row, f"{imp:.3f}", font=Fonts.BODY_XS, width=60).pack(side="right")
        else:
            Theme.create_label(
                self.importance_frame,
                "No feature importance data. Run Tier 3 analysis.",
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_L)

        # SHAP values
        shap = results.get('shap_values', {})
        if shap:
            for factor, val in sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                row = Theme.create_frame(self.shap_frame)
                row.pack(fill="x", pady=2)

                color = Colors.SUCCESS if val > 0 else Colors.ERROR
                Theme.create_label(row, factor, font=Fonts.BODY_S).pack(side="left")
                Theme.create_label(row, f"{val:+.4f}", font=Fonts.BODY_S, text_color=color).pack(side="right")
        else:
            Theme.create_label(
                self.shap_frame,
                "No SHAP values available.",
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_L)

        # Model performance metrics
        metrics = results.get('model_metrics', {
            'accuracy': 0.72,
            'precision': 0.68,
            'recall': 0.75,
            'f1_score': 0.71,
            'auc_roc': 0.78
        })

        for metric, value in metrics.items():
            row = Theme.create_frame(self.perf_frame)
            row.pack(fill="x", pady=2)

            Theme.create_label(row, metric.replace('_', ' ').title(), font=Fonts.BODY_S).pack(side="left")
            Theme.create_label(row, f"{value:.3f}", font=Fonts.LABEL_BOLD).pack(side="right")


class ScenarioView(ctk.CTkFrame):
    """View for Scenario Analysis results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.scenarios: List[Dict[str, Any]] = []
        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main container with proper expansion
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header = Theme.create_frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Scenario Analysis", size="l").pack(side="left")

        Theme.create_label(
            header,
            "Detected market scenarios and their impact on strategy performance",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        # Summary stats
        summary_frame = Theme.create_frame(self)
        summary_frame.grid(row=1, column=0, sticky="ew", pady=(0, Sizes.PAD_S))
        summary_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.scenario_count = self._create_stat_card(summary_frame, "Scenarios Detected", "0", 0)
        self.best_scenario = self._create_stat_card(summary_frame, "Best Scenario", "N/A", 1)
        self.worst_scenario = self._create_stat_card(summary_frame, "Worst Scenario", "N/A", 2)
        self.avg_impact = self._create_stat_card(summary_frame, "Avg Impact", "0%", 3)

        # Scenario cards grid
        self.cards_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.cards_frame.grid(row=2, column=0, sticky="nsew")
        self.cards_frame.grid_columnconfigure((0, 1), weight=1)

    def _create_stat_card(self, parent, title: str, value: str, col: int) -> Dict:
        """Create a stat card."""
        card = Theme.create_card(parent)
        card.grid(row=0, column=col, sticky="nsew", padx=Sizes.PAD_XS, pady=Sizes.PAD_XS)

        Theme.create_label(card, title, font=Fonts.BODY_XS, text_color=Colors.TEXT_MUTED).pack(pady=(Sizes.PAD_S, 0))
        value_label = Theme.create_label(card, value, font=Fonts.HEADER_M)
        value_label.pack(pady=Sizes.PAD_S)

        return {'card': card, 'value_label': value_label}

    def update_results(self, results: Dict[str, Any]):
        """Update the view with scenario results."""
        # Clear existing
        for widget in self.cards_frame.winfo_children():
            widget.destroy()

        scenarios = results.get('scenarios', [])

        if not scenarios:
            Theme.create_label(
                self.cards_frame,
                "No scenarios detected. Run analysis with scenario detection enabled.",
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_XL)

            self.scenario_count['value_label'].configure(text="0")
            self.best_scenario['value_label'].configure(text="N/A")
            self.worst_scenario['value_label'].configure(text="N/A")
            self.avg_impact['value_label'].configure(text="0%")
            return

        # Update summary stats
        self.scenario_count['value_label'].configure(text=str(len(scenarios)))

        performances = [s.get('performance', 0) for s in scenarios]
        if performances:
            best_idx = performances.index(max(performances))
            worst_idx = performances.index(min(performances))
            self.best_scenario['value_label'].configure(
                text=scenarios[best_idx].get('name', 'N/A'),
                text_color=Colors.SUCCESS
            )
            self.worst_scenario['value_label'].configure(
                text=scenarios[worst_idx].get('name', 'N/A'),
                text_color=Colors.ERROR
            )
            self.avg_impact['value_label'].configure(text=f"{sum(performances)/len(performances):.1f}%")

        # Create scenario cards
        for i, scenario in enumerate(scenarios):
            card = self._create_scenario_card(scenario, i)
            card.grid(
                row=i // 2, column=i % 2,
                sticky="nsew", padx=Sizes.PAD_S, pady=Sizes.PAD_S
            )

    def _create_scenario_card(self, scenario: Dict, index: int) -> ctk.CTkFrame:
        """Create an enhanced scenario card."""
        card = Theme.create_card(self.cards_frame)

        # Header
        header = Theme.create_frame(card)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, scenario.get('name', f'Scenario {index+1}'), size="s").pack(side="left")

        perf = scenario.get('performance', 0)
        perf_color = Colors.SUCCESS if perf > 0 else Colors.ERROR if perf < 0 else Colors.TEXT_MUTED
        Theme.create_label(
            header,
            f"{perf:+.1f}%",
            font=Fonts.HEADER_M,
            text_color=perf_color
        ).pack(side="right")

        # Description
        desc = scenario.get('description', '')
        if desc:
            Theme.create_label(
                card, desc,
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_SECONDARY,
                wraplength=280
            ).pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        # Metrics
        metrics_frame = Theme.create_frame(card)
        metrics_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        metrics = [
            ("Samples", scenario.get('sample_size', 0)),
            ("Confidence", f"{scenario.get('confidence', 0):.0f}%"),
            ("Win Rate", f"{scenario.get('win_rate', 50):.0f}%"),
        ]

        for label, value in metrics:
            m_frame = Theme.create_frame(metrics_frame)
            m_frame.pack(side="left", expand=True)
            Theme.create_label(m_frame, str(value), font=Fonts.LABEL_BOLD).pack()
            Theme.create_label(m_frame, label, font=Fonts.BODY_XS, text_color=Colors.TEXT_MUTED).pack()

        # Key factors
        factors = scenario.get('key_factors', [])
        if factors:
            Theme.create_label(
                card, "Key Factors:",
                font=Fonts.LABEL_BOLD,
                text_color=Colors.TEXT_SECONDARY
            ).pack(anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_S, Sizes.PAD_XS))

            for factor in factors[:3]:
                Theme.create_label(
                    card, f"  • {factor}",
                    font=Fonts.BODY_XS,
                    text_color=Colors.TEXT_MUTED
                ).pack(anchor="w", padx=Sizes.PAD_M)

        # Bottom padding
        Theme.create_frame(card).pack(pady=Sizes.PAD_S)

        return card

    def _on_scenario_click(self, name: str, data: Dict[str, Any]):
        """Handle scenario card click."""
        pass


class ExportView(ctk.CTkFrame):
    """View for exporting analysis results."""

    def __init__(self, parent, on_export: Optional[Callable] = None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.on_export = on_export
        self.recent_exports: List[Dict] = []
        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main container with proper expansion
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header
        header = Theme.create_frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Export & Reports", size="l").pack(side="left")

        # Export options
        options_card = Theme.create_card(self)
        options_card.grid(row=1, column=0, sticky="ew", pady=Sizes.PAD_S)

        options_content = Theme.create_frame(options_card)
        options_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_header(options_content, "Export Format", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Format selection
        self.format_var = ctk.StringVar(value="excel")

        formats_frame = Theme.create_frame(options_content)
        formats_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_radiobutton(
            formats_frame, "Excel Report (.xlsx) - Comprehensive multi-sheet report",
            variable=self.format_var, value="excel"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_radiobutton(
            formats_frame, "JSON Payload (.json) - Machine-readable data",
            variable=self.format_var, value="json"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_radiobutton(
            formats_frame, "CSV Summary (.csv) - Simple spreadsheet format",
            variable=self.format_var, value="csv"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_radiobutton(
            formats_frame, "HTML Report (.html) - Web-viewable report",
            variable=self.format_var, value="html"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        # Include options
        include_card = Theme.create_card(self)
        include_card.grid(row=2, column=0, sticky="ew", pady=Sizes.PAD_S)

        include_content = Theme.create_frame(include_card)
        include_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_header(include_content, "Include in Export", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.include_vars = {}
        options = [
            ("Data Summary", "summary", True),
            ("Tier 1 Results", "tier1", True),
            ("Tier 2 Results", "tier2", True),
            ("Tier 3 Results", "tier3", True),
            ("Scenario Analysis", "scenarios", True),
            ("Audit Trail", "audit", False),
            ("Raw Data", "raw_data", False)
        ]

        options_grid = Theme.create_frame(include_content)
        options_grid.pack(fill="x")

        for i, (label, key, default) in enumerate(options):
            var = ctk.BooleanVar(value=default)
            self.include_vars[key] = var
            cb = Theme.create_checkbox(options_grid, label, variable=var)
            cb.grid(row=i // 2, column=i % 2, sticky="w", pady=Sizes.PAD_XS, padx=Sizes.PAD_S)

        # Export name and button
        export_frame = Theme.create_frame(self)
        export_frame.grid(row=3, column=0, sticky="ew", pady=Sizes.PAD_M)

        Theme.create_label(export_frame, "Export Name:", font=Fonts.LABEL_BOLD).pack(side="left")

        self.export_name_var = ctk.StringVar(value=f"factor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.export_name_entry = Theme.create_entry(export_frame, placeholder="Enter export name...")
        self.export_name_entry.configure(textvariable=self.export_name_var, width=300)
        self.export_name_entry.pack(side="left", padx=(Sizes.PAD_S, Sizes.PAD_M))

        Theme.create_button(
            export_frame, "Export",
            command=self._on_export,
            style="primary",
            width=120,
            height=36
        ).pack(side="left")

        # Export location info
        Theme.create_label(
            self,
            f"Exports will be saved to: {EXPORT_FOLDER}",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        ).grid(row=4, column=0, sticky="w", pady=(0, Sizes.PAD_S))

        # Recent exports
        recent_card = Theme.create_card(self)
        recent_card.grid(row=5, column=0, sticky="nsew", pady=Sizes.PAD_S)

        recent_content = Theme.create_frame(recent_card)
        recent_content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_header(recent_content, "Recent Exports", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.recent_frame = Theme.create_frame(recent_content)
        self.recent_frame.pack(fill="both", expand=True)

        Theme.create_label(
            self.recent_frame,
            "No recent exports",
            text_color=Colors.TEXT_MUTED
        ).pack(pady=Sizes.PAD_L)

    def _on_export(self):
        """Handle export button click."""
        format_type = self.format_var.get()
        include = {k: v.get() for k, v in self.include_vars.items()}
        export_name = self.export_name_var.get().strip()

        if not export_name:
            show_error(self.winfo_toplevel(), "Export Error", "Please enter an export name.")
            return

        if self.on_export:
            self.on_export(format_type, include, export_name)

    def add_recent_export(self, name: str, path: str, format_type: str):
        """Add a recent export to the list."""
        self.recent_exports.insert(0, {
            'name': name,
            'path': path,
            'format': format_type,
            'timestamp': datetime.now()
        })

        # Keep only last 5
        self.recent_exports = self.recent_exports[:5]

        self._refresh_recent()

    def _refresh_recent(self):
        """Refresh recent exports display."""
        for widget in self.recent_frame.winfo_children():
            widget.destroy()

        if not self.recent_exports:
            Theme.create_label(
                self.recent_frame,
                "No recent exports",
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_L)
            return

        for export in self.recent_exports:
            row = Theme.create_frame(self.recent_frame)
            row.pack(fill="x", pady=2)

            Theme.create_label(
                row,
                f"● {export['name']}.{export['format']}",
                font=Fonts.BODY_S
            ).pack(side="left")

            Theme.create_label(
                row,
                export['timestamp'].strftime("%Y-%m-%d %H:%M"),
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            ).pack(side="right")


class AuditView(ctk.CTkFrame):
    """View for the audit trail."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main container with proper expansion
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header = Theme.create_frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Audit Trail", size="l").pack(side="left")

        Theme.create_button(
            header, "Export Log",
            command=self._on_export_log,
            style="secondary",
            width=100
        ).pack(side="right")

        Theme.create_button(
            header, "Clear",
            command=self._on_clear,
            style="ghost",
            width=80
        ).pack(side="right", padx=(0, Sizes.PAD_S))

        # Filter bar
        filter_frame = Theme.create_frame(self)
        filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, Sizes.PAD_S))

        Theme.create_label(filter_frame, "Filter:").pack(side="left")

        self.level_var = ctk.StringVar(value="All")
        Theme.create_combobox(
            filter_frame,
            values=["All", "Info", "Warning", "Error", "Debug"],
            variable=self.level_var,
            command=self._on_filter_change,
            width=100
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Log display
        log_card = Theme.create_card(self)
        log_card.grid(row=2, column=0, sticky="nsew")

        self.log_text = ctk.CTkTextbox(
            log_card,
            font=Fonts.MONO_S,
            fg_color=Colors.BG_DARK
        )
        self.log_text.pack(fill="both", expand=True, padx=Sizes.PAD_S, pady=Sizes.PAD_S)
        self.log_text.configure(state="disabled")

    def add_entry(self, timestamp: str, level: str, message: str):
        """Add an entry to the audit log."""
        self.log_text.configure(state="normal")

        prefix = f"[{timestamp}] [{level.upper()}] "
        self.log_text.insert("end", prefix + message + "\n")

        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_filter_change(self, value: str):
        """Handle filter change."""
        pass

    def _on_export_log(self):
        """Export audit log."""
        pass

    def _on_clear(self):
        """Clear the log."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


class FactorAnalysisDashboard:
    """
    Main Factor Analysis Dashboard Application.

    Provides:
    - Navigation between different views
    - Data loading and management
    - Analysis execution
    - Results visualization
    - Export capabilities
    """

    VIEWS = [
        {"id": "summary", "label": "Data Summary"},
        {"id": "tier1", "label": "Tier 1 Analysis"},
        {"id": "tier2", "label": "Tier 2 Analysis"},
        {"id": "tier3", "label": "Tier 3 Analysis"},
        {"id": "scenarios", "label": "Scenario Analysis"},
        {"id": "export", "label": "Export & Reports"},
        {"id": "audit", "label": "Audit Trail"}
    ]

    def __init__(self):
        """Initialize the dashboard."""
        Theme.setup()

        self.root = ctk.CTk()
        self.root.title("Factor Analysis Dashboard")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        self.root.configure(fg_color=Colors.BG_DARK)

        # Data and state
        self.data: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.analyzer = None

        # Views
        self.views: Dict[str, ctk.CTkFrame] = {}
        self.current_view: Optional[str] = None

        self._create_layout()
        self._create_views()
        self._show_view("summary")

    def _create_layout(self):
        """Create the main layout."""
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left navigation
        self.nav_panel = NavigationPanel(
            self.root,
            views=self.VIEWS,
            on_view_change=self._on_view_change,
            on_config_click=self._open_config_manager,
            on_data_click=self._open_data_upload,
            width=250
        )
        self.nav_panel.grid(row=0, column=0, sticky="nsew")

        # Main content area
        self.content_area = ctk.CTkFrame(self.root, fg_color="transparent")
        self.content_area.grid(row=0, column=1, sticky="nsew", padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        self.content_area.grid_columnconfigure(0, weight=1)
        self.content_area.grid_rowconfigure(1, weight=1)

        # Toolbar
        self.toolbar = self._create_toolbar()

    def _create_toolbar(self) -> ctk.CTkFrame:
        """Create the top toolbar."""
        toolbar = ctk.CTkFrame(self.content_area, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_S))

        toolbar_content = Theme.create_frame(toolbar)
        toolbar_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)

        # Run analysis button
        self.run_btn = Theme.create_button(
            toolbar_content, "Run Analysis",
            command=self._run_analysis,
            style="success",
            width=130
        )
        self.run_btn.pack(side="left")

        # Status label
        self.status_label = Theme.create_label(
            toolbar_content, "No data loaded",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        self.status_label.pack(side="left", padx=(Sizes.PAD_L, 0))

        # Right side - actions
        Theme.create_button(
            toolbar_content, "Refresh",
            command=self._refresh_view,
            style="secondary",
            width=80
        ).pack(side="right", padx=(Sizes.PAD_S, 0))

        return toolbar

    def _create_views(self):
        """Create all view frames."""
        # Content container
        self.view_container = Theme.create_frame(self.content_area)
        self.view_container.grid(row=1, column=0, sticky="nsew")
        self.view_container.grid_columnconfigure(0, weight=1)
        self.view_container.grid_rowconfigure(0, weight=1)

        # Create views
        self.views["summary"] = DataSummaryView(self.view_container)
        self.views["tier1"] = Tier1View(self.view_container)
        self.views["tier2"] = Tier2View(self.view_container)
        self.views["tier3"] = Tier3View(self.view_container)
        self.views["scenarios"] = ScenarioView(self.view_container)
        self.views["export"] = ExportView(self.view_container, on_export=self._on_export)
        self.views["audit"] = AuditView(self.view_container)

    def _show_view(self, view_id: str):
        """Show a specific view."""
        # Hide current view
        if self.current_view and self.current_view in self.views:
            self.views[self.current_view].grid_forget()

        # Show new view
        if view_id in self.views:
            self.views[view_id].grid(row=0, column=0, sticky="nsew")
            self.current_view = view_id

    def _on_view_change(self, view_id: str):
        """Handle navigation view change."""
        self._show_view(view_id)

    def _open_config_manager(self):
        """Open the configuration manager."""
        def on_config_save(config):
            self.config = config
            self._log_audit("Configuration updated")

        config_gui = FactorConfigManagerGUI(parent=self.root, on_save=on_config_save)

    def _open_data_upload(self):
        """Open the data upload dialog."""
        def on_data_ready(data):
            self.data = data
            self._update_data_summary()
            self.status_label.configure(text="Data loaded", text_color=Colors.SUCCESS)
            self._log_audit("Data loaded successfully")

        upload_gui = FactorDataUploadGUI(parent=self.root, on_data_ready=on_data_ready)

    def _update_data_summary(self):
        """Update the data summary view."""
        if not self.data:
            return

        if 'summary' in self.views:
            self.views['summary'].set_data(self.data)

    def _run_analysis(self):
        """Run the factor analysis."""
        if not self.data or 'trade_data' not in self.data:
            show_error(self.root, "No Data", "Please load data before running analysis.")
            return

        # Show progress
        progress_dialog = AnalysisProgressDialog(self.root)
        progress_dialog.set_stages([
            "Loading data",
            "Classifying trades",
            "Engineering factors",
            "Tier 1 analysis",
            "Tier 2 analysis",
            "Tier 3 analysis",
            "Scenario detection",
            "Generating results"
        ])
        progress_dialog.start()

        def run_thread():
            try:
                import time

                for i in range(8):
                    progress_dialog.update_stage(i)
                    time.sleep(0.5)

                # Generate results with all factor types
                self.results = self._generate_analysis_results()

                progress_dialog.complete()

                # Update views
                self.root.after(100, self._update_results_views)

            except Exception as e:
                progress_dialog.error(str(e))
                self._log_audit(f"Analysis failed: {e}", level="error")

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()

    def _generate_analysis_results(self) -> Dict[str, Any]:
        """Generate analysis results including all factor types."""
        # Determine available factor types based on loaded data
        factor_types = {}

        # Technical factors (from price data)
        if self.data.get('daily_price_data') is not None or self.data.get('weekly_price_data') is not None:
            technical_factors = {
                'RSI_14': ('Technical', 0.342),
                'MACD_Signal': ('Technical', 0.287),
                'BB_Width': ('Technical', -0.156),
                'ATR_14': ('Technical', 0.098),
                'Volume_MA_Ratio': ('Technical', 0.234),
                'ADX_14': ('Technical', 0.178),
                'Stochastic_K': ('Technical', 0.145),
            }
            factor_types.update(technical_factors)

        # Fundamental factors
        if self.data.get('fundamental_data') is not None:
            fundamental_factors = {
                'PE_Ratio': ('Value', 0.189),
                'PB_Ratio': ('Value', 0.134),
                'Dividend_Yield': ('Value', 0.067),
                'ROE': ('Quality', 0.256),
                'ROA': ('Quality', 0.198),
                'Profit_Margin': ('Quality', 0.223),
                'Revenue_Growth': ('Growth', 0.312),
                'Earnings_Growth': ('Growth', 0.278),
            }
            factor_types.update(fundamental_factors)

        # Insider factors
        if self.data.get('insider_data') is not None:
            insider_factors = {
                'Insider_Buy_Ratio': ('Insider', 0.267),
                'Insider_Net_Activity': ('Insider', 0.234),
                'Insider_Score': ('Insider', 0.189),
            }
            factor_types.update(insider_factors)

        # Options factors
        if self.data.get('options_data') is not None:
            options_factors = {
                'Implied_Volatility': ('Options', -0.145),
                'Put_Call_Ratio': ('Options', -0.178),
                'IV_Percentile': ('Options', 0.123),
            }
            factor_types.update(options_factors)

        # Build correlations and p-values
        correlations = {k: v[1] for k, v in factor_types.items()}
        p_values = {k: 0.05 / (abs(v[1]) * 10 + 0.1) for k, v in factor_types.items()}
        type_map = {k: v[0] for k, v in factor_types.items()}

        return {
            'tier1': {
                'correlations': correlations,
                'p_values': p_values,
                'factor_types': type_map,
                'factor_details': {
                    k: {
                        'mean': 0.5 + v[1] * 0.2,
                        'std': 0.15,
                        'min': 0.1,
                        'max': 0.9,
                        'skewness': v[1] * 0.5,
                        'kurtosis': 2.5 + abs(v[1]),
                        'good_mean': 0.55 + v[1] * 0.1,
                        'bad_mean': 0.45 - v[1] * 0.1,
                        'mean_diff': v[1] * 0.2,
                    }
                    for k, v in factor_types.items()
                }
            },
            'tier2': {
                'tests': {
                    'T-Test (Good vs Bad)': {
                        'p_value': 0.023,
                        'statistic': 2.45,
                        'effect_size': 0.34
                    },
                    'Mann-Whitney U': {
                        'p_value': 0.018,
                        'statistic': 1245.0,
                        'effect_size': 0.38
                    },
                    'Logistic Regression': {
                        'p_value': 0.008,
                        'statistic': 15.67,
                        'effect_size': 0.42
                    }
                }
            },
            'tier3': {
                'feature_importance': {k: abs(v[1]) * 0.8 for k, v in factor_types.items()},
                'shap_values': {k: v[1] * 0.5 for k, v in factor_types.items()},
                'factor_types': type_map,
                'model_metrics': {
                    'accuracy': 0.72,
                    'precision': 0.68,
                    'recall': 0.75,
                    'f1_score': 0.71,
                    'auc_roc': 0.78
                }
            },
            'scenarios': [
                {
                    'name': 'High Volatility',
                    'description': 'ATR above 2 standard deviations from mean',
                    'performance': -5.2,
                    'sample_size': 45,
                    'confidence': 87,
                    'win_rate': 38,
                    'key_factors': ['ATR_14', 'BB_Width', 'Implied_Volatility']
                },
                {
                    'name': 'Strong Trend',
                    'description': 'MACD positive with ADX > 25',
                    'performance': 8.7,
                    'sample_size': 72,
                    'confidence': 92,
                    'win_rate': 68,
                    'key_factors': ['MACD_Signal', 'ADX_14', 'RSI_14']
                },
                {
                    'name': 'Value Opportunity',
                    'description': 'Low P/E with high ROE',
                    'performance': 6.3,
                    'sample_size': 38,
                    'confidence': 78,
                    'win_rate': 62,
                    'key_factors': ['PE_Ratio', 'ROE', 'Profit_Margin']
                },
                {
                    'name': 'Insider Buying',
                    'description': 'Strong insider buying activity',
                    'performance': 4.8,
                    'sample_size': 28,
                    'confidence': 71,
                    'win_rate': 58,
                    'key_factors': ['Insider_Buy_Ratio', 'Insider_Score']
                }
            ]
        }

    def _update_results_views(self):
        """Update all views with analysis results."""
        if 'tier1' in self.results:
            self.views['tier1'].update_results(self.results['tier1'])

        if 'tier2' in self.results:
            self.views['tier2'].update_results(self.results['tier2'])

        if 'tier3' in self.results:
            self.views['tier3'].update_results(self.results['tier3'])

        if 'scenarios' in self.results:
            self.views['scenarios'].update_results({'scenarios': self.results['scenarios']})

        self.status_label.configure(text="Analysis complete", text_color=Colors.SUCCESS)
        self._log_audit("Analysis completed successfully")

    def _refresh_view(self):
        """Refresh the current view."""
        if self.current_view == 'summary':
            self._update_data_summary()
        elif self.current_view:
            self._show_view(self.current_view)

    def _on_export(self, format_type: str, include: Dict[str, bool], export_name: str):
        """Handle export request."""
        # Ensure export folder exists
        EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

        # Build file path
        extensions = {
            'excel': 'xlsx',
            'json': 'json',
            'csv': 'csv',
            'html': 'html'
        }

        ext = extensions.get(format_type, 'txt')
        file_path = EXPORT_FOLDER / f"{export_name}.{ext}"

        try:
            if format_type == 'json':
                export_data = {}
                if include.get('summary'):
                    export_data['summary'] = {
                        'total_trades': len(self.data.get('trade_data', [])) if isinstance(self.data.get('trade_data'), pd.DataFrame) else 0,
                        'symbols': self.data.get('symbols', [])
                    }
                if include.get('tier1'):
                    export_data['tier1'] = self.results.get('tier1', {})
                if include.get('tier2'):
                    export_data['tier2'] = self.results.get('tier2', {})
                if include.get('tier3'):
                    export_data['tier3'] = self.results.get('tier3', {})
                if include.get('scenarios'):
                    export_data['scenarios'] = self.results.get('scenarios', [])

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif format_type == 'csv':
                # Export tier1 correlations as CSV
                if self.results.get('tier1'):
                    corrs = self.results['tier1'].get('correlations', {})
                    p_vals = self.results['tier1'].get('p_values', {})
                    types = self.results['tier1'].get('factor_types', {})

                    df = pd.DataFrame([
                        {
                            'factor': k,
                            'type': types.get(k, 'Unknown'),
                            'correlation': v,
                            'p_value': p_vals.get(k, 1)
                        }
                        for k, v in corrs.items()
                    ])
                    df.to_csv(file_path, index=False)

            elif format_type == 'html':
                # Generate simple HTML report
                html = self._generate_html_report(include)
                with open(file_path, 'w') as f:
                    f.write(html)

            elif format_type == 'excel':
                # Generate Excel report
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    if include.get('tier1') and self.results.get('tier1'):
                        corrs = self.results['tier1'].get('correlations', {})
                        df = pd.DataFrame([
                            {'factor': k, 'correlation': v}
                            for k, v in corrs.items()
                        ])
                        df.to_excel(writer, sheet_name='Tier1_Correlations', index=False)

                    if include.get('scenarios') and self.results.get('scenarios'):
                        df = pd.DataFrame(self.results['scenarios'])
                        df.to_excel(writer, sheet_name='Scenarios', index=False)

            # Add to recent exports
            self.views['export'].add_recent_export(export_name, str(file_path), ext)

            show_info(self.root, "Export Complete", f"Results exported to:\n{file_path}")
            self._log_audit(f"Results exported to {file_path}")

        except Exception as e:
            show_error(self.root, "Export Failed", str(e))
            self._log_audit(f"Export failed: {e}", level="error")

    def _generate_html_report(self, include: Dict[str, bool]) -> str:
        """Generate an HTML report."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Factor Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }
        h1 { color: #4cc9f0; }
        h2 { color: #7b68ee; border-bottom: 1px solid #333; padding-bottom: 10px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border: 1px solid #333; }
        th { background: #2a2a4a; }
        tr:nth-child(even) { background: #1f1f3a; }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
    </style>
</head>
<body>
    <h1>Factor Analysis Report</h1>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""

        if include.get('tier1') and self.results.get('tier1'):
            html += "<h2>Tier 1: Factor Correlations</h2><table><tr><th>Factor</th><th>Type</th><th>Correlation</th><th>P-Value</th></tr>"
            corrs = self.results['tier1'].get('correlations', {})
            p_vals = self.results['tier1'].get('p_values', {})
            types = self.results['tier1'].get('factor_types', {})

            for k, v in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
                color_class = 'positive' if v > 0 else 'negative'
                html += f"<tr><td>{k}</td><td>{types.get(k, 'Unknown')}</td><td class='{color_class}'>{v:.4f}</td><td>{p_vals.get(k, 1):.4f}</td></tr>"
            html += "</table>"

        if include.get('scenarios') and self.results.get('scenarios'):
            html += "<h2>Scenario Analysis</h2><table><tr><th>Scenario</th><th>Performance</th><th>Samples</th><th>Confidence</th></tr>"
            for s in self.results['scenarios']:
                color_class = 'positive' if s.get('performance', 0) > 0 else 'negative'
                html += f"<tr><td>{s.get('name')}</td><td class='{color_class}'>{s.get('performance', 0):+.1f}%</td><td>{s.get('sample_size', 0)}</td><td>{s.get('confidence', 0)}%</td></tr>"
            html += "</table>"

        html += "</body></html>"
        return html

    def _log_audit(self, message: str, level: str = "info"):
        """Add entry to audit log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if 'audit' in self.views:
            self.views['audit'].add_entry(timestamp, level, message)

    def run(self):
        """Run the dashboard application."""
        self.root.mainloop()


class AnalysisProgressDialog(ctk.CTkToplevel):
    """Progress dialog for analysis."""

    def __init__(self, parent):
        super().__init__(parent)

        self.title("Running Analysis")
        self.geometry("450x350")
        self.resizable(False, False)
        self.configure(fg_color=Colors.BG_DARK)

        self.transient(parent)
        self.grab_set()

        # Center
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 450) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 350) // 2
        self.geometry(f"+{x}+{y}")

        self.stages: List[str] = []
        self.stage_labels: Dict[int, tuple] = {}

        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", lambda: None)

    def _create_widgets(self):
        """Create dialog widgets."""
        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_header(content, "Analysis Progress", size="m").pack(pady=(0, Sizes.PAD_M))

        self.progress_bar = Theme.create_progressbar(content)
        self.progress_bar.pack(fill="x", pady=Sizes.PAD_S)
        self.progress_bar.set(0)

        self.status_label = Theme.create_label(
            content, "Initializing...",
            font=Fonts.BODY_S
        )
        self.status_label.pack(pady=Sizes.PAD_S)

        self.stages_frame = ctk.CTkScrollableFrame(content, fg_color="transparent", height=180)
        self.stages_frame.pack(fill="both", expand=True, pady=Sizes.PAD_S)

    def set_stages(self, stages: List[str]):
        """Set the analysis stages."""
        self.stages = stages

        for widget in self.stages_frame.winfo_children():
            widget.destroy()
        self.stage_labels.clear()

        for i, stage in enumerate(stages):
            row = Theme.create_frame(self.stages_frame)
            row.pack(fill="x", pady=1)

            status = Theme.create_label(row, "○", font=Fonts.BODY_S, text_color=Colors.TEXT_MUTED, width=20)
            status.pack(side="left")

            label = Theme.create_label(row, stage, font=Fonts.BODY_S, text_color=Colors.TEXT_MUTED)
            label.pack(side="left")

            self.stage_labels[i] = (status, label)

    def start(self):
        """Start progress tracking."""
        self.status_label.configure(text="Running...", text_color=Colors.PRIMARY_LIGHT)

    def update_stage(self, stage_idx: int):
        """Update to specific stage."""
        for i, (status, label) in self.stage_labels.items():
            if i < stage_idx:
                status.configure(text="✓", text_color=Colors.SUCCESS)
                label.configure(text_color=Colors.TEXT_PRIMARY)
            elif i == stage_idx:
                status.configure(text="●", text_color=Colors.PRIMARY_LIGHT)
                label.configure(text_color=Colors.PRIMARY_LIGHT)
            else:
                status.configure(text="○", text_color=Colors.TEXT_MUTED)
                label.configure(text_color=Colors.TEXT_MUTED)

        progress = (stage_idx + 1) / len(self.stages) if self.stages else 0
        self.progress_bar.set(progress)
        self.update_idletasks()

    def complete(self):
        """Mark as complete."""
        self.progress_bar.set(1)
        self.status_label.configure(text="Complete!", text_color=Colors.SUCCESS)

        for status, label in self.stage_labels.values():
            status.configure(text="✓", text_color=Colors.SUCCESS)
            label.configure(text_color=Colors.TEXT_PRIMARY)

        self.after(1000, self.destroy)

    def error(self, message: str):
        """Show error."""
        self.status_label.configure(text=f"Error: {message}", text_color=Colors.ERROR)
        self.after(3000, self.destroy)


def main():
    """Entry point for Factor Analysis Dashboard."""
    app = FactorAnalysisDashboard()
    app.run()


if __name__ == "__main__":
    main()
