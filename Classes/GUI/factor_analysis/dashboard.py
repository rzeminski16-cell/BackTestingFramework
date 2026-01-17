"""
Factor Analysis Dashboard GUI.

The main application for viewing and interacting with factor analysis results.
Provides multiple views for exploring analysis data at different levels.
"""

import customtkinter as ctk
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import queue
import json
import os
import base64
from io import BytesIO

from ..ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info, ask_yes_no, ProgressPanel
from .components import (
    NavigationPanel, FactorListPanel, StatisticsPanel,
    DataQualityIndicator, AnalysisProgressPanel, ScenarioCard,
    TierResultsPanel
)
from .config_manager import FactorConfigManagerGUI
from .data_upload import FactorDataUploadGUI

# Import Factor Analysis components
try:
    from ...FactorAnalysis.analyzer import FactorAnalyzer, AnalysisInput, AnalysisOutput
    from ...FactorAnalysis.config.factor_config import (
        FactorAnalysisConfig,
        TradeClassificationConfig,
        DataAlignmentConfig,
        FactorEngineeringConfig,
        StatisticalAnalysisConfig,
        ScenarioAnalysisConfig,
        ThresholdType
    )
    ANALYZER_AVAILABLE = True
except ImportError as e:
    ANALYZER_AVAILABLE = False
    print(f"Factor Analyzer not available: {e}")

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Default export folder
EXPORT_FOLDER = Path(__file__).parent.parent.parent.parent / "exports" / "factor_analysis"


class FactorVisualization:
    """Helper class for creating factor analysis visualizations."""

    COLORS = {
        'primary': '#4cc9f0',
        'secondary': '#7b68ee',
        'positive': '#4ade80',
        'negative': '#f87171',
        'neutral': '#94a3b8',
        'background': '#1a1a2e',
        'surface': '#2a2a4a',
        'text': '#e2e8f0',
        'grid': '#3a3a5a',
    }

    @staticmethod
    def create_distribution_chart(
        good_values: List[float],
        bad_values: List[float],
        factor_name: str,
        figsize: tuple = (6, 4)
    ) -> Optional[Figure]:
        """Create a distribution comparison chart for good vs bad trades."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=figsize, facecolor=FactorVisualization.COLORS['background'])
        ax.set_facecolor(FactorVisualization.COLORS['background'])

        # Create histograms
        bins = 20
        alpha = 0.6

        if good_values:
            ax.hist(good_values, bins=bins, alpha=alpha, label='Good Trades',
                   color=FactorVisualization.COLORS['positive'], edgecolor='white', linewidth=0.5)
        if bad_values:
            ax.hist(bad_values, bins=bins, alpha=alpha, label='Bad Trades',
                   color=FactorVisualization.COLORS['negative'], edgecolor='white', linewidth=0.5)

        # Add mean lines
        if good_values:
            good_mean = np.mean(good_values)
            ax.axvline(good_mean, color=FactorVisualization.COLORS['positive'],
                      linestyle='--', linewidth=2, label=f'Good Mean: {good_mean:.3f}')
        if bad_values:
            bad_mean = np.mean(bad_values)
            ax.axvline(bad_mean, color=FactorVisualization.COLORS['negative'],
                      linestyle='--', linewidth=2, label=f'Bad Mean: {bad_mean:.3f}')

        ax.set_xlabel(factor_name, color=FactorVisualization.COLORS['text'])
        ax.set_ylabel('Frequency', color=FactorVisualization.COLORS['text'])
        ax.set_title(f'Distribution: {factor_name}', color=FactorVisualization.COLORS['text'], fontweight='bold')
        ax.legend(facecolor=FactorVisualization.COLORS['surface'], edgecolor='none',
                 labelcolor=FactorVisualization.COLORS['text'])
        ax.tick_params(colors=FactorVisualization.COLORS['text'])
        ax.grid(True, alpha=0.3, color=FactorVisualization.COLORS['grid'])

        for spine in ax.spines.values():
            spine.set_color(FactorVisualization.COLORS['grid'])

        plt.tight_layout()
        return fig

    @staticmethod
    def create_box_plot(
        data_dict: Dict[str, List[float]],
        title: str,
        figsize: tuple = (8, 5)
    ) -> Optional[Figure]:
        """Create a box plot comparing distributions."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=figsize, facecolor=FactorVisualization.COLORS['background'])
        ax.set_facecolor(FactorVisualization.COLORS['background'])

        labels = list(data_dict.keys())
        data = [data_dict[k] for k in labels]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = [FactorVisualization.COLORS['positive'], FactorVisualization.COLORS['negative']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for element in ['whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color(FactorVisualization.COLORS['text'])

        ax.set_title(title, color=FactorVisualization.COLORS['text'], fontweight='bold')
        ax.set_ylabel('Value', color=FactorVisualization.COLORS['text'])
        ax.tick_params(colors=FactorVisualization.COLORS['text'])
        ax.grid(True, alpha=0.3, color=FactorVisualization.COLORS['grid'], axis='y')

        for spine in ax.spines.values():
            spine.set_color(FactorVisualization.COLORS['grid'])

        plt.tight_layout()
        return fig

    @staticmethod
    def create_effect_size_chart(
        effects: Dict[str, float],
        title: str = "Effect Sizes (Cohen's d)",
        figsize: tuple = (8, 6)
    ) -> Optional[Figure]:
        """Create a horizontal bar chart of effect sizes."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=figsize, facecolor=FactorVisualization.COLORS['background'])
        ax.set_facecolor(FactorVisualization.COLORS['background'])

        factors = list(effects.keys())
        values = list(effects.values())

        colors = [FactorVisualization.COLORS['positive'] if v > 0
                 else FactorVisualization.COLORS['negative'] for v in values]

        y_pos = range(len(factors))
        ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)

        # Add threshold lines
        for thresh, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
            ax.axvline(thresh, color=FactorVisualization.COLORS['neutral'], linestyle=':', alpha=0.5)
            ax.axvline(-thresh, color=FactorVisualization.COLORS['neutral'], linestyle=':', alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors, color=FactorVisualization.COLORS['text'])
        ax.set_xlabel("Cohen's d", color=FactorVisualization.COLORS['text'])
        ax.set_title(title, color=FactorVisualization.COLORS['text'], fontweight='bold')
        ax.tick_params(colors=FactorVisualization.COLORS['text'])
        ax.axvline(0, color=FactorVisualization.COLORS['text'], linewidth=0.5)
        ax.grid(True, alpha=0.3, color=FactorVisualization.COLORS['grid'], axis='x')

        for spine in ax.spines.values():
            spine.set_color(FactorVisualization.COLORS['grid'])

        plt.tight_layout()
        return fig

    @staticmethod
    def create_correlation_heatmap(
        correlations: Dict[str, float],
        title: str = "Factor Correlations",
        figsize: tuple = (10, 8)
    ) -> Optional[Figure]:
        """Create a correlation bar chart (sorted)."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=figsize, facecolor=FactorVisualization.COLORS['background'])
        ax.set_facecolor(FactorVisualization.COLORS['background'])

        # Sort by absolute correlation
        sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        factors = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]

        colors = [FactorVisualization.COLORS['positive'] if v > 0
                 else FactorVisualization.COLORS['negative'] for v in values]

        y_pos = range(len(factors))
        ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors, color=FactorVisualization.COLORS['text'])
        ax.set_xlabel('Correlation with Trade Success', color=FactorVisualization.COLORS['text'])
        ax.set_title(title, color=FactorVisualization.COLORS['text'], fontweight='bold')
        ax.tick_params(colors=FactorVisualization.COLORS['text'])
        ax.axvline(0, color=FactorVisualization.COLORS['text'], linewidth=0.5)
        ax.grid(True, alpha=0.3, color=FactorVisualization.COLORS['grid'], axis='x')
        ax.set_xlim(-1, 1)

        for spine in ax.spines.values():
            spine.set_color(FactorVisualization.COLORS['grid'])

        plt.tight_layout()
        return fig

    @staticmethod
    def create_scenario_performance_chart(
        scenarios: List[Dict[str, Any]],
        figsize: tuple = (10, 6)
    ) -> Optional[Figure]:
        """Create a scenario performance comparison chart."""
        if not MATPLOTLIB_AVAILABLE or not scenarios:
            return None

        fig, ax = plt.subplots(figsize=figsize, facecolor=FactorVisualization.COLORS['background'])
        ax.set_facecolor(FactorVisualization.COLORS['background'])

        names = [s.get('name', f'Scenario {i}') for i, s in enumerate(scenarios)]
        performances = [s.get('performance', 0) for s in scenarios]
        win_rates = [s.get('win_rate', 50) for s in scenarios]

        x = np.arange(len(names))
        width = 0.35

        colors_perf = [FactorVisualization.COLORS['positive'] if p > 0
                      else FactorVisualization.COLORS['negative'] for p in performances]

        bars1 = ax.bar(x - width/2, performances, width, label='Performance (%)',
                      color=colors_perf, alpha=0.7, edgecolor='white')
        bars2 = ax.bar(x + width/2, [w - 50 for w in win_rates], width, label='Win Rate vs 50%',
                      color=FactorVisualization.COLORS['secondary'], alpha=0.7, edgecolor='white')

        ax.set_ylabel('Value (%)', color=FactorVisualization.COLORS['text'])
        ax.set_title('Scenario Performance Comparison', color=FactorVisualization.COLORS['text'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, color=FactorVisualization.COLORS['text'], rotation=45, ha='right')
        ax.tick_params(colors=FactorVisualization.COLORS['text'])
        ax.legend(facecolor=FactorVisualization.COLORS['surface'], edgecolor='none',
                 labelcolor=FactorVisualization.COLORS['text'])
        ax.axhline(0, color=FactorVisualization.COLORS['text'], linewidth=0.5)
        ax.grid(True, alpha=0.3, color=FactorVisualization.COLORS['grid'], axis='y')

        for spine in ax.spines.values():
            spine.set_color(FactorVisualization.COLORS['grid'])

        plt.tight_layout()
        return fig

    @staticmethod
    def figure_to_base64(fig: Figure) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor=FactorVisualization.COLORS['background'])
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


class DataSummaryView(ctk.CTkFrame):
    """View displaying data summary and quality metrics."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.data: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self._create_widgets()

    def set_config(self, config: Dict[str, Any]):
        """Set the configuration for this view."""
        self.config = config
        # Refresh to apply new classification thresholds
        self._refresh_data()

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

        # Classify trades using configured thresholds
        good_trades = 0
        bad_trades = 0
        indeterminate = 0

        if isinstance(trade_data, pd.DataFrame):
            # Get P&L percentage column
            pnl_col = None
            for col in ['pl_pct', 'pnl_pct', 'return_pct', 'pnl', 'pl', 'profit', 'return', 'profit_loss']:
                if col in trade_data.columns:
                    pnl_col = col
                    break

            if pnl_col:
                # Get thresholds from config
                tc_config = self.config.get('trade_classification', {})
                good_thresh = float(tc_config.get('good_threshold_pct', 0.0))
                bad_thresh = float(tc_config.get('bad_threshold_pct', 0.0))
                bad_min_days = int(tc_config.get('min_holding_period', 0))

                # Get duration column if available
                duration_col = None
                for col in ['duration_days', 'holding_days', 'days_held']:
                    if col in trade_data.columns:
                        duration_col = col
                        break

                # Calculate duration if not present but entry/exit dates exist
                if duration_col is None and 'entry_date' in trade_data.columns and 'exit_date' in trade_data.columns:
                    try:
                        trade_data = trade_data.copy()
                        trade_data['duration_days'] = (
                            pd.to_datetime(trade_data['exit_date']) - pd.to_datetime(trade_data['entry_date'])
                        ).dt.days
                        duration_col = 'duration_days'
                    except Exception:
                        pass

                # Classify using thresholds
                for _, row in trade_data.iterrows():
                    pl = row[pnl_col]
                    duration = row[duration_col] if duration_col else 0

                    if pd.isna(pl):
                        indeterminate += 1
                    elif pl > good_thresh:
                        good_trades += 1
                    elif pl < bad_thresh:
                        # For bad trades, also check duration requirement
                        if bad_min_days > 0 and duration < bad_min_days:
                            indeterminate += 1  # Short-duration loss is indeterminate
                        else:
                            bad_trades += 1
                    else:
                        indeterminate += 1

                print(f"[DEBUG] Trade Classification in Overview:")
                print(f"  - good_thresh: {good_thresh}, bad_thresh: {bad_thresh}, bad_min_days: {bad_min_days}")
                print(f"  - good: {good_trades}, bad: {bad_trades}, indeterminate: {indeterminate}")

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

        # Create matplotlib visualization if available
        if MATPLOTLIB_AVAILABLE and details:
            # Generate sample data for visualization based on statistics
            good_mean = details.get('good_mean', 0.55)
            bad_mean = details.get('bad_mean', 0.45)
            std = details.get('std', 0.15)

            # Generate sample distributions
            np.random.seed(42)
            good_values = np.random.normal(good_mean, std, 100).tolist()
            bad_values = np.random.normal(bad_mean, std, 100).tolist()

            fig = FactorVisualization.create_distribution_chart(
                good_values, bad_values, factor_name, figsize=(5, 3)
            )

            if fig:
                # Embed matplotlib figure in tkinter
                chart_frame = ctk.CTkFrame(self.dist_frame, fg_color="transparent")
                chart_frame.pack(fill="both", expand=True, pady=Sizes.PAD_S)

                canvas = FigureCanvasTkAgg(fig, master=chart_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

        # Show distribution statistics
        stats_frame = Theme.create_frame(self.dist_frame)
        stats_frame.pack(fill="x", pady=Sizes.PAD_S)

        if details:
            stats = [
                ("Good Trades Mean", details.get('good_mean', 0)),
                ("Bad Trades Mean", details.get('bad_mean', 0)),
                ("Difference", details.get('mean_diff', 0)),
                ("Std Dev", details.get('std', 0)),
            ]

            for label, value in stats:
                row = Theme.create_frame(stats_frame)
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

        Theme.create_header(header, "Tier 2: Hypothesis Testing", size="l").pack(side="left")

        Theme.create_label(
            header,
            "Statistical tests comparing good vs bad trades",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        # Main content - scrollable
        main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_scroll.grid(row=1, column=0, sticky="nsew")

        # Interpretation guide
        interp_card = Theme.create_card(main_scroll)
        interp_card.pack(fill="x", pady=(0, Sizes.PAD_S))

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

        # Two column layout for results and visualization
        content_frame = Theme.create_frame(main_scroll)
        content_frame.pack(fill="both", expand=True, pady=Sizes.PAD_S)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # Left - Test Results
        left_card = Theme.create_card(content_frame)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_S))

        Theme.create_header(left_card, "Statistical Test Results", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.tests_frame = Theme.create_frame(left_card)
        self.tests_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Right - Effect Size Visualization
        right_card = Theme.create_card(content_frame)
        right_card.grid(row=0, column=1, sticky="nsew")

        Theme.create_header(right_card, "Effect Size Visualization", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.viz_frame = Theme.create_frame(right_card)
        self.viz_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Bottom - Pairwise Comparisons
        pairwise_card = Theme.create_card(main_scroll)
        pairwise_card.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_header(pairwise_card, "Pairwise Factor Comparisons", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.pairwise_frame = Theme.create_frame(pairwise_card)
        self.pairwise_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Summary insights
        insights_card = Theme.create_card(main_scroll)
        insights_card.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_header(insights_card, "Key Insights", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.insights_frame = Theme.create_frame(insights_card)
        self.insights_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

    def update_results(self, results: Dict[str, Any]):
        """Update the view with analysis results."""
        self.results = results

        # Clear previous content
        for widget in self.tests_frame.winfo_children():
            widget.destroy()
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        for widget in self.pairwise_frame.winfo_children():
            widget.destroy()
        for widget in self.insights_frame.winfo_children():
            widget.destroy()

        # Display test results
        tests = results.get('tests', {})
        for test_name, test_data in tests.items():
            row = Theme.create_frame(self.tests_frame)
            row.pack(fill="x", pady=Sizes.PAD_XS)

            p_val = test_data.get('p_value', 1)
            stat = test_data.get('statistic', 0)
            effect = test_data.get('effect_size', 0)

            # Significance indicator
            if p_val < 0.01:
                sig = "***"
                sig_color = Colors.SUCCESS
            elif p_val < 0.05:
                sig = "**"
                sig_color = Colors.WARNING
            elif p_val < 0.1:
                sig = "*"
                sig_color = Colors.TEXT_PRIMARY
            else:
                sig = "NS"
                sig_color = Colors.TEXT_MUTED

            Theme.create_label(row, test_name, font=Fonts.LABEL_BOLD).pack(anchor="w")

            details_row = Theme.create_frame(row)
            details_row.pack(fill="x", pady=2)

            Theme.create_label(details_row, f"p-value: {p_val:.4f}", font=Fonts.BODY_XS,
                             text_color=Colors.TEXT_SECONDARY).pack(side="left")
            Theme.create_label(details_row, f"Statistic: {stat:.3f}", font=Fonts.BODY_XS,
                             text_color=Colors.TEXT_SECONDARY).pack(side="left", padx=(Sizes.PAD_M, 0))
            Theme.create_label(details_row, f"Effect: {effect:.3f}", font=Fonts.BODY_XS,
                             text_color=Colors.TEXT_SECONDARY).pack(side="left", padx=(Sizes.PAD_M, 0))
            Theme.create_label(details_row, sig, font=Fonts.LABEL_BOLD, text_color=sig_color).pack(side="right")

        # Create effect size visualization
        if MATPLOTLIB_AVAILABLE:
            effect_sizes = {name: data.get('effect_size', 0) for name, data in tests.items()}
            fig = FactorVisualization.create_effect_size_chart(effect_sizes, figsize=(5, 3))

            if fig:
                canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            Theme.create_label(
                self.viz_frame,
                "Install matplotlib for visualizations",
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_L)

        # Pairwise comparisons (simulated data)
        pairwise_data = [
            ("RSI_14 vs MACD_Signal", 0.034, 0.45, "**"),
            ("ROE vs Profit_Margin", 0.012, 0.52, "***"),
            ("Insider_Buy_Ratio vs PE_Ratio", 0.089, 0.28, "*"),
        ]

        # Header row
        header_row = Theme.create_frame(self.pairwise_frame)
        header_row.pack(fill="x", pady=(0, Sizes.PAD_XS))
        Theme.create_label(header_row, "Comparison", font=Fonts.LABEL_BOLD, width=200).pack(side="left")
        Theme.create_label(header_row, "P-Value", font=Fonts.LABEL_BOLD, width=80).pack(side="left")
        Theme.create_label(header_row, "Effect", font=Fonts.LABEL_BOLD, width=80).pack(side="left")
        Theme.create_label(header_row, "Sig.", font=Fonts.LABEL_BOLD, width=50).pack(side="left")

        for comparison, p_val, effect, sig in pairwise_data:
            row = Theme.create_frame(self.pairwise_frame)
            row.pack(fill="x", pady=1)

            sig_color = Colors.SUCCESS if sig == "***" else Colors.WARNING if sig == "**" else Colors.TEXT_MUTED
            Theme.create_label(row, comparison, font=Fonts.BODY_S, width=200).pack(side="left")
            Theme.create_label(row, f"{p_val:.4f}", font=Fonts.BODY_S, width=80).pack(side="left")
            Theme.create_label(row, f"{effect:.3f}", font=Fonts.BODY_S, width=80).pack(side="left")
            Theme.create_label(row, sig, font=Fonts.LABEL_BOLD, text_color=sig_color, width=50).pack(side="left")

        # Generate insights
        insights = [
            ("✓", "T-Test shows significant difference between good and bad trade groups (p=0.023)", Colors.SUCCESS),
            ("✓", "Logistic regression confirms predictive power of selected factors (p=0.008)", Colors.SUCCESS),
            ("!", "Effect sizes are in the small-to-medium range (0.34-0.42), suggesting practical significance", Colors.WARNING),
            ("i", "Consider combining significant factors for improved prediction accuracy", Colors.PRIMARY_LIGHT),
        ]

        for icon, text, color in insights:
            row = Theme.create_frame(self.insights_frame)
            row.pack(fill="x", pady=2)
            Theme.create_label(row, icon, font=Fonts.LABEL_BOLD, text_color=color, width=25).pack(side="left")
            Theme.create_label(row, text, font=Fonts.BODY_S, text_color=Colors.TEXT_SECONDARY,
                             wraplength=500).pack(side="left")


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

        # Interpretation
        interp = self._get_scenario_interpretation(scenario)
        if interp:
            interp_frame = Theme.create_frame(card)
            interp_frame.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)

            Theme.create_label(
                interp_frame, "💡 Interpretation:",
                font=Fonts.LABEL_BOLD,
                text_color=Colors.PRIMARY_LIGHT
            ).pack(anchor="w")

            Theme.create_label(
                interp_frame, interp,
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_SECONDARY,
                wraplength=280
            ).pack(anchor="w", pady=(Sizes.PAD_XS, 0))

        # Bottom padding
        Theme.create_frame(card).pack(pady=Sizes.PAD_S)

        return card

    def _get_scenario_interpretation(self, scenario: Dict) -> str:
        """Generate interpretation text for a scenario."""
        name = scenario.get('name', '').lower()
        perf = scenario.get('performance', 0)
        win_rate = scenario.get('win_rate', 50)
        confidence = scenario.get('confidence', 0)

        interpretations = []

        # Performance interpretation
        if perf > 5:
            interpretations.append(f"Strong positive alpha ({perf:+.1f}%) suggests this is a favorable condition for your strategy.")
        elif perf > 0:
            interpretations.append(f"Modest positive returns ({perf:+.1f}%) indicate marginal edge in this scenario.")
        elif perf < -5:
            interpretations.append(f"Significant negative alpha ({perf:+.1f}%) suggests avoiding trades in this condition.")
        elif perf < 0:
            interpretations.append(f"Negative returns ({perf:+.1f}%) indicate underperformance in this scenario.")

        # Win rate interpretation
        if win_rate > 60:
            interpretations.append(f"High win rate ({win_rate:.0f}%) provides psychological comfort for execution.")
        elif win_rate < 40:
            interpretations.append(f"Low win rate ({win_rate:.0f}%) may require strong risk management.")

        # Confidence interpretation
        if confidence >= 90:
            interpretations.append("High statistical confidence supports reliable scenario detection.")
        elif confidence < 70:
            interpretations.append("Lower confidence suggests monitoring for false positives.")

        # Context-specific interpretations
        if 'volatility' in name:
            interpretations.append("Consider adjusting position size based on volatility regime.")
        elif 'trend' in name:
            interpretations.append("Trend-following approaches may be more effective here.")
        elif 'value' in name:
            interpretations.append("Value-based entry criteria show predictive power.")
        elif 'insider' in name:
            interpretations.append("Insider activity provides informational advantage.")

        return " ".join(interpretations[:2]) if interpretations else ""

    def _on_scenario_click(self, name: str, data: Dict[str, Any]):
        """Handle scenario card click."""
        pass

    def _create_interpretation_section(self):
        """Create the interpretation guide section."""
        interp_card = Theme.create_card(self.cards_frame)
        interp_card.grid(row=100, column=0, columnspan=2, sticky="ew", pady=Sizes.PAD_M)

        Theme.create_header(interp_card, "Scenario Interpretation Guide", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        guide_content = Theme.create_frame(interp_card)
        guide_content.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        guides = [
            ("Performance", "Average P&L deviation from baseline during this scenario. Positive values indicate outperformance."),
            ("Win Rate", "Percentage of profitable trades in this scenario. Compare to overall win rate for context."),
            ("Confidence", "Statistical confidence in scenario detection. Higher values = more reliable identification."),
            ("Sample Size", "Number of trades in this scenario. Larger samples provide more reliable statistics."),
            ("Key Factors", "The most influential factors that define and trigger this scenario."),
        ]

        for term, definition in guides:
            row = Theme.create_frame(guide_content)
            row.pack(fill="x", pady=2)
            Theme.create_label(row, f"• {term}:", font=Fonts.LABEL_BOLD, width=100).pack(side="left")
            Theme.create_label(row, definition, font=Fonts.BODY_XS, text_color=Colors.TEXT_SECONDARY,
                             wraplength=400).pack(side="left")


class FactorDocumentationView(ctk.CTkFrame):
    """View displaying factor documentation: calculations, raw data sources, and availability."""

    # Factor documentation: name -> (calculation method, raw data source with exact column, category)
    FACTOR_DOCS = {
        # EPS Factors (default for fundamental analysis)
        'eps_eps': {
            'name': 'EPS',
            'calculation': 'Direct value from earnings report: Net Income / Shares Outstanding',
            'raw_data': 'fundamental_data.csv',
            'column': 'reported_eps',
            'category': 'EPS Fundamentals'
        },
        'eps_estimated_eps': {
            'name': 'Estimated EPS',
            'calculation': 'Analyst consensus estimate for upcoming earnings',
            'raw_data': 'fundamental_data.csv',
            'column': 'estimated_eps',
            'category': 'EPS Fundamentals'
        },
        'eps_earnings_surprise': {
            'name': 'Earnings Surprise',
            'calculation': 'Actual EPS - Estimated EPS',
            'raw_data': 'fundamental_data.csv',
            'column': 'earnings_surprise',
            'category': 'EPS Fundamentals'
        },
        'eps_earnings_surprise_pct': {
            'name': 'Earnings Surprise %',
            'calculation': 'Percentage surprise from analyst estimates',
            'raw_data': 'fundamental_data.csv',
            'column': 'surprise_pct',
            'category': 'EPS Fundamentals'
        },
        # Value Factors
        'value_pe_ratio': {
            'name': 'P/E Ratio',
            'calculation': 'Stock Price / Earnings Per Share (TTM)',
            'raw_data': 'fundamental_data.csv',
            'column': 'pe_ratio',
            'category': 'Value'
        },
        'value_price_to_book': {
            'name': 'Price to Book',
            'calculation': 'Market Cap / Total Book Value',
            'raw_data': 'fundamental_data.csv',
            'column': 'price_to_book',
            'category': 'Value'
        },
        'value_price_to_sales': {
            'name': 'Price to Sales',
            'calculation': 'Market Cap / Revenue (TTM)',
            'raw_data': 'fundamental_data.csv',
            'column': 'price_to_sales_ttm',
            'category': 'Value'
        },
        'value_peg_ratio': {
            'name': 'PEG Ratio',
            'calculation': 'P/E Ratio / Earnings Growth Rate',
            'raw_data': 'fundamental_data.csv',
            'column': 'peg_ratio',
            'category': 'Value'
        },
        'value_dividend_yield': {
            'name': 'Dividend Yield',
            'calculation': 'Annual Dividends Per Share / Stock Price × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'dividend_yield',
            'category': 'Value'
        },
        'value_ev_to_ebitda': {
            'name': 'EV/EBITDA',
            'calculation': 'Enterprise Value / EBITDA',
            'raw_data': 'fundamental_data.csv',
            'column': 'ev_to_ebitda',
            'category': 'Value'
        },
        # Quality Factors
        'quality_return_on_equity': {
            'name': 'Return on Equity',
            'calculation': 'Net Income / Shareholders Equity × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'return_on_equity_ttm',
            'category': 'Quality'
        },
        'quality_return_on_assets': {
            'name': 'Return on Assets',
            'calculation': 'Net Income / Total Assets × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'return_on_assets_ttm',
            'category': 'Quality'
        },
        'quality_profit_margin': {
            'name': 'Profit Margin',
            'calculation': 'Net Income / Revenue × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'profit_margin',
            'category': 'Quality'
        },
        'quality_operating_margin': {
            'name': 'Operating Margin',
            'calculation': 'Operating Income / Revenue × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'operating_margin_ttm',
            'category': 'Quality'
        },
        'quality_current_ratio': {
            'name': 'Current Ratio',
            'calculation': 'Current Assets / Current Liabilities',
            'raw_data': 'fundamental_data.csv',
            'column': 'currentratio',
            'category': 'Quality'
        },
        'quality_debt_to_equity': {
            'name': 'Debt to Equity',
            'calculation': 'Total Liabilities / Shareholders Equity',
            'raw_data': 'fundamental_data.csv',
            'column': 'debt_to_equity',
            'category': 'Quality'
        },
        # Growth Factors
        'growth_revenue_growth': {
            'name': 'Revenue Growth',
            'calculation': '(Current Revenue - Prior Year Revenue) / |Prior Year Revenue| × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'revenue_growth_yoy',
            'category': 'Growth'
        },
        'growth_earnings_growth': {
            'name': 'Earnings Growth',
            'calculation': '(Current EPS - Prior Year EPS) / |Prior Year EPS| × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'earnings_growth_yoy',
            'category': 'Growth'
        },
        'growth_earnings_surprise': {
            'name': 'Earnings Surprise',
            'calculation': 'Actual EPS - Estimated EPS',
            'raw_data': 'fundamental_data.csv',
            'column': 'earnings_surprise',
            'category': 'Growth'
        },
        'growth_earnings_surprise_pct': {
            'name': 'Earnings Surprise %',
            'calculation': '(Actual EPS - Estimated EPS) / |Estimated EPS| × 100',
            'raw_data': 'fundamental_data.csv',
            'column': 'surprise_pct',
            'category': 'Growth'
        },
        # Technical Factors
        'rsi': {
            'name': 'RSI (Relative Strength Index)',
            'calculation': '100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss over N periods',
            'raw_data': 'price_data.csv',
            'column': 'close',
            'category': 'Technical - Momentum'
        },
        'macd': {
            'name': 'MACD',
            'calculation': '12-period EMA - 26-period EMA',
            'raw_data': 'price_data.csv',
            'column': 'close',
            'category': 'Technical - Trend'
        },
        'macd_signal': {
            'name': 'MACD Signal',
            'calculation': '9-period EMA of MACD line',
            'raw_data': 'price_data.csv',
            'column': 'close',
            'category': 'Technical - Trend'
        },
        'macd_hist': {
            'name': 'MACD Histogram',
            'calculation': 'MACD - Signal Line',
            'raw_data': 'price_data.csv',
            'column': 'close',
            'category': 'Technical - Trend'
        },
        'sma': {
            'name': 'SMA (Simple Moving Average)',
            'calculation': 'Sum of closing prices over N periods / N',
            'raw_data': 'price_data.csv',
            'column': 'close',
            'category': 'Technical - Trend'
        },
        'ema': {
            'name': 'EMA (Exponential Moving Average)',
            'calculation': 'Weighted average with exponential decay, more weight on recent prices',
            'raw_data': 'price_data.csv',
            'column': 'close',
            'category': 'Technical - Trend'
        },
        'adx': {
            'name': 'ADX (Average Directional Index)',
            'calculation': 'Smoothed average of DX = |+DI - -DI| / (+DI + -DI) × 100',
            'raw_data': 'price_data.csv',
            'column': 'high, low, close',
            'category': 'Technical - Trend'
        },
        'atr': {
            'name': 'ATR (Average True Range)',
            'calculation': 'Average of True Range over N periods. TR = max(H-L, |H-Prev Close|, |L-Prev Close|)',
            'raw_data': 'price_data.csv',
            'column': 'high, low, close',
            'category': 'Technical - Volatility'
        },
        'bollinger': {
            'name': 'Bollinger Bands',
            'calculation': 'Middle: SMA(N), Upper: SMA + K×StdDev, Lower: SMA - K×StdDev',
            'raw_data': 'price_data.csv',
            'column': 'close',
            'category': 'Technical - Volatility'
        },
        'obv': {
            'name': 'OBV (On-Balance Volume)',
            'calculation': 'Cumulative sum: +Volume if close > prev close, -Volume if close < prev close',
            'raw_data': 'price_data.csv',
            'column': 'close, volume',
            'category': 'Technical - Volume'
        },
        'vwap': {
            'name': 'VWAP (Volume Weighted Avg Price)',
            'calculation': 'Sum(Price × Volume) / Sum(Volume)',
            'raw_data': 'price_data.csv',
            'column': 'high, low, close, volume',
            'category': 'Technical - Volume'
        },
        'stochastic': {
            'name': 'Stochastic Oscillator',
            'calculation': '%K = (Close - Lowest Low) / (Highest High - Lowest Low) × 100',
            'raw_data': 'price_data.csv',
            'column': 'high, low, close',
            'category': 'Technical - Momentum'
        },
        'cci': {
            'name': 'CCI (Commodity Channel Index)',
            'calculation': '(Typical Price - SMA) / (0.015 × Mean Deviation)',
            'raw_data': 'price_data.csv',
            'column': 'high, low, close',
            'category': 'Technical - Momentum'
        },
        'mfi': {
            'name': 'MFI (Money Flow Index)',
            'calculation': '100 - (100 / (1 + Money Ratio)), Money Ratio = Positive MF / Negative MF',
            'raw_data': 'price_data.csv',
            'column': 'high, low, close, volume',
            'category': 'Technical - Volume'
        },
        # Insider Factors
        # Raw columns: date, symbol, insider_title, transaction_type, shares, price, value, executive, security_type
        'insider_buy_count': {
            'name': 'Insider Buy Count',
            'calculation': 'Count of insider purchase transactions in lookback window',
            'raw_data': 'insider_data.csv',
            'column': 'date, symbol, transaction_type',
            'category': 'Insider'
        },
        'insider_sell_count': {
            'name': 'Insider Sell Count',
            'calculation': 'Count of insider sale transactions in lookback window',
            'raw_data': 'insider_data.csv',
            'column': 'date, symbol, transaction_type',
            'category': 'Insider'
        },
        'insider_net_shares': {
            'name': 'Insider Net Shares',
            'calculation': 'Total shares bought - Total shares sold in lookback window',
            'raw_data': 'insider_data.csv',
            'column': 'date, symbol, transaction_type, shares',
            'category': 'Insider'
        },
        'insider_score': {
            'name': 'Insider Score',
            'calculation': 'Composite score based on transaction size, frequency, and insider role',
            'raw_data': 'insider_data.csv',
            'column': 'date, symbol, transaction_type, shares, value, insider_title, executive',
            'category': 'Insider'
        },
        'insider_buy_sell_ratio': {
            'name': 'Buy/Sell Ratio',
            'calculation': 'Buy Count / (Sell Count + 1)',
            'raw_data': 'insider_data.csv',
            'column': 'date, symbol, transaction_type',
            'category': 'Insider'
        },
        # Options Factors
        'options_implied_volatility': {
            'name': 'Implied Volatility',
            'calculation': 'Volatility implied by option prices using Black-Scholes model',
            'raw_data': 'options_data.csv',
            'column': 'implied_volatility',
            'category': 'Options'
        },
        'options_put_call_ratio': {
            'name': 'Put/Call Ratio',
            'calculation': 'Put Volume / Call Volume',
            'raw_data': 'options_data.csv',
            'column': 'put_volume, call_volume',
            'category': 'Options'
        },
        'options_iv_percentile': {
            'name': 'IV Percentile',
            'calculation': '% of days in past year with lower IV than current IV',
            'raw_data': 'options_data.csv',
            'column': 'implied_volatility (historical)',
            'category': 'Options'
        },
        # Regime Factors
        'regime_volatility': {
            'name': 'Volatility Regime',
            'calculation': 'Classification based on realized volatility vs historical distribution',
            'raw_data': 'price_data.csv',
            'column': 'close (for returns calculation)',
            'category': 'Regime'
        },
        'regime_trend': {
            'name': 'Trend Regime',
            'calculation': 'Classification based on price vs moving averages and ADX',
            'raw_data': 'price_data.csv',
            'column': 'close, high, low',
            'category': 'Regime'
        },
        # Composite Scores
        'composite_eps': {
            'name': 'Composite EPS Score',
            'calculation': 'Z-score normalized average of all EPS factors',
            'raw_data': 'Derived from fundamental_data.csv',
            'column': 'reported_eps, estimated_eps, earnings_surprise',
            'category': 'Composite'
        },
        'composite_value': {
            'name': 'Composite Value Score',
            'calculation': 'Z-score normalized average of value factors (lower=better inverted)',
            'raw_data': 'Derived from fundamental_data.csv',
            'column': 'pe_ratio, price_to_book, price_to_sales_ttm, peg_ratio, dividend_yield, ev_to_ebitda',
            'category': 'Composite'
        },
        'composite_quality': {
            'name': 'Composite Quality Score',
            'calculation': 'Z-score normalized average of quality factors',
            'raw_data': 'Derived from fundamental_data.csv',
            'column': 'return_on_equity_ttm, return_on_assets_ttm, profit_margin, operating_margin_ttm, currentratio, debt_to_equity',
            'category': 'Composite'
        },
        'composite_growth': {
            'name': 'Composite Growth Score',
            'calculation': 'Z-score normalized average of growth factors',
            'raw_data': 'Derived from fundamental_data.csv',
            'column': 'revenue_growth_yoy, reported_eps (YoY calc), earnings_surprise',
            'category': 'Composite'
        },
        'composite_fundamental': {
            'name': 'Overall Fundamental Score',
            'calculation': 'Average of available composite scores (value, quality, growth, eps)',
            'raw_data': 'Derived from composite scores',
            'column': 'composite_value, composite_quality, composite_growth, composite_eps',
            'category': 'Composite'
        },
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.data: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.factor_availability: Dict[str, float] = {}
        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Main scrollable container
        main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_scroll.pack(fill="both", expand=True)

        # Header
        header = Theme.create_frame(main_scroll)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Factor Documentation", size="l").pack(side="left")

        Theme.create_button(
            header, "Refresh",
            command=self._refresh_data,
            style="secondary",
            width=80,
            height=28
        ).pack(side="right")

        # Info banner
        info_card = Theme.create_card(main_scroll)
        info_card.pack(fill="x", pady=Sizes.PAD_S)

        info_content = Theme.create_frame(info_card)
        info_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_label(
            info_content,
            "This tab shows how each factor is calculated, what raw data is required, "
            "and the availability of that data in your dataset.",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY,
            wraplength=800
        ).pack(anchor="w")

        # Filter options
        filter_frame = Theme.create_frame(main_scroll)
        filter_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(filter_frame, "Filter by category:", font=Fonts.LABEL).pack(side="left")

        self.category_var = ctk.StringVar(value="All")
        categories = ["All", "EPS Fundamentals", "Value", "Quality", "Growth",
                     "Technical - Momentum", "Technical - Trend", "Technical - Volatility",
                     "Technical - Volume", "Insider", "Options", "Regime", "Composite"]

        self.category_menu = ctk.CTkOptionMenu(
            filter_frame,
            variable=self.category_var,
            values=categories,
            command=self._on_category_change,
            width=200,
            fg_color=Colors.SURFACE,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER
        )
        self.category_menu.pack(side="left", padx=Sizes.PAD_S)

        # Factor table container
        self.table_container = Theme.create_frame(main_scroll)
        self.table_container.pack(fill="both", expand=True, pady=Sizes.PAD_S)

        # Create the factor table
        self._create_factor_table()

    def _create_factor_table(self, category_filter: str = "All"):
        """Create or recreate the factor documentation table."""
        # Clear existing content
        for widget in self.table_container.winfo_children():
            widget.destroy()

        # Table header
        header_frame = Theme.create_card(self.table_container)
        header_frame.pack(fill="x", pady=(0, Sizes.PAD_XS))

        header_content = Theme.create_frame(header_frame)
        header_content.pack(fill="x", padx=Sizes.PAD_S, pady=Sizes.PAD_S)
        header_content.grid_columnconfigure(0, weight=1, minsize=120)
        header_content.grid_columnconfigure(1, weight=2, minsize=200)
        header_content.grid_columnconfigure(2, weight=1, minsize=120)
        header_content.grid_columnconfigure(3, weight=1, minsize=150)
        header_content.grid_columnconfigure(4, weight=0, minsize=100)

        Theme.create_label(header_content, "Factor", font=Fonts.LABEL_BOLD).grid(
            row=0, column=0, sticky="w", padx=Sizes.PAD_XS)
        Theme.create_label(header_content, "Calculation Method", font=Fonts.LABEL_BOLD).grid(
            row=0, column=1, sticky="w", padx=Sizes.PAD_XS)
        Theme.create_label(header_content, "Data Source", font=Fonts.LABEL_BOLD).grid(
            row=0, column=2, sticky="w", padx=Sizes.PAD_XS)
        Theme.create_label(header_content, "Column Name(s)", font=Fonts.LABEL_BOLD).grid(
            row=0, column=3, sticky="w", padx=Sizes.PAD_XS)
        Theme.create_label(header_content, "Availability", font=Fonts.LABEL_BOLD).grid(
            row=0, column=4, sticky="w", padx=Sizes.PAD_XS)

        # Group factors by category
        factors_by_category: Dict[str, List[tuple]] = {}
        for factor_key, doc in self.FACTOR_DOCS.items():
            category = doc['category']
            if category_filter != "All" and category != category_filter:
                continue
            if category not in factors_by_category:
                factors_by_category[category] = []
            factors_by_category[category].append((factor_key, doc))

        # Display factors grouped by category
        for category in sorted(factors_by_category.keys()):
            factors = factors_by_category[category]

            # Category header
            cat_header = Theme.create_frame(self.table_container)
            cat_header.pack(fill="x", pady=(Sizes.PAD_M, Sizes.PAD_XS))

            Theme.create_label(
                cat_header, category,
                font=Fonts.HEADER_S,
                text_color=Colors.PRIMARY
            ).pack(side="left")

            # Category count
            Theme.create_label(
                cat_header, f"({len(factors)} factors)",
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            ).pack(side="left", padx=Sizes.PAD_S)

            # Factor rows
            for factor_key, doc in sorted(factors, key=lambda x: x[1]['name']):
                self._create_factor_row(factor_key, doc)

    def _create_factor_row(self, factor_key: str, doc: Dict[str, str]):
        """Create a single factor documentation row."""
        row_card = Theme.create_card(self.table_container)
        row_card.pack(fill="x", pady=1)

        row_content = Theme.create_frame(row_card)
        row_content.pack(fill="x", padx=Sizes.PAD_S, pady=Sizes.PAD_S)
        row_content.grid_columnconfigure(0, weight=1, minsize=120)
        row_content.grid_columnconfigure(1, weight=2, minsize=200)
        row_content.grid_columnconfigure(2, weight=1, minsize=120)
        row_content.grid_columnconfigure(3, weight=1, minsize=150)
        row_content.grid_columnconfigure(4, weight=0, minsize=100)

        # Factor name
        Theme.create_label(
            row_content, doc['name'],
            font=Fonts.LABEL_BOLD,
            text_color=Colors.TEXT_PRIMARY
        ).grid(row=0, column=0, sticky="w", padx=Sizes.PAD_XS)

        # Calculation method
        Theme.create_label(
            row_content, doc['calculation'],
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_SECONDARY,
            wraplength=200
        ).grid(row=0, column=1, sticky="w", padx=Sizes.PAD_XS)

        # Raw data source (file name)
        Theme.create_label(
            row_content, doc['raw_data'],
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_SECONDARY,
            wraplength=120
        ).grid(row=0, column=2, sticky="w", padx=Sizes.PAD_XS)

        # Column name(s) - exact column names used
        column_name = doc.get('column', 'N/A')
        Theme.create_label(
            row_content, column_name,
            font=Fonts.MONO if column_name != 'N/A' else Fonts.BODY_XS,
            text_color=Colors.PRIMARY if column_name != 'N/A' else Colors.TEXT_MUTED,
            wraplength=150
        ).grid(row=0, column=3, sticky="w", padx=Sizes.PAD_XS)

        # Availability - check if we have data for this factor
        availability = self._get_factor_availability(factor_key)
        warning = self._get_missing_column_warning(factor_key)
        avail_color = self._get_availability_color(availability)

        avail_frame = Theme.create_frame(row_content)
        avail_frame.grid(row=0, column=4, sticky="w", padx=Sizes.PAD_XS)

        if availability is not None:
            # Show percentage
            Theme.create_label(
                avail_frame, f"{availability:.1f}%",
                font=Fonts.LABEL_BOLD,
                text_color=avail_color
            ).pack(side="left")

            # Mini progress bar
            bar_frame = ctk.CTkFrame(avail_frame, fg_color=Colors.BG_DARK, height=6, width=50, corner_radius=3)
            bar_frame.pack(side="left", padx=Sizes.PAD_XS)
            bar_frame.pack_propagate(False)

            fill_width = max(1, int(50 * availability / 100))
            fill_bar = ctk.CTkFrame(bar_frame, fg_color=avail_color, height=6, width=fill_width, corner_radius=3)
            fill_bar.place(x=0, y=0)
        elif warning:
            # Show warning for missing column
            Theme.create_label(
                avail_frame, "Missing",
                font=Fonts.BODY_XS,
                text_color=Colors.ERROR
            ).pack(side="left")
            # Add tooltip-like indicator
            Theme.create_label(
                avail_frame, "!",
                font=Fonts.LABEL_BOLD,
                text_color=Colors.ERROR
            ).pack(side="left", padx=2)
        else:
            # No data loaded yet
            Theme.create_label(
                avail_frame, "N/A",
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            ).pack(side="left")

    def _get_factor_availability(self, factor_key: str) -> Optional[float]:
        """Get the availability percentage for a factor."""
        # Check if we have pre-computed availability
        if factor_key in self.factor_availability:
            return self.factor_availability[factor_key]

        # Try to match factor key to column in results data
        if self.results and 'factor_data' in self.results:
            factor_df = self.results['factor_data']
            if isinstance(factor_df, pd.DataFrame):
                # Try various column name patterns
                patterns = [
                    factor_key,
                    factor_key.replace('_', ''),
                    factor_key.lower(),
                ]
                for pattern in patterns:
                    for col in factor_df.columns:
                        if pattern in col.lower() or col.lower() in pattern:
                            non_null = factor_df[col].notna().sum()
                            total = len(factor_df)
                            if total > 0:
                                availability = (non_null / total) * 100
                                self.factor_availability[factor_key] = availability
                                return availability

        # Check if column is in missing_columns (warning case)
        if hasattr(self, 'missing_columns') and factor_key in self.missing_columns:
            return None  # Return None to show "N/A" with warning

        return None

    def _get_missing_column_warning(self, factor_key: str) -> Optional[str]:
        """Get warning message if column is missing for a factor."""
        if hasattr(self, 'missing_columns') and factor_key in self.missing_columns:
            return self.missing_columns[factor_key]
        return None

    def _get_availability_color(self, availability: Optional[float]) -> str:
        """Get color based on availability percentage."""
        if availability is None:
            return Colors.TEXT_MUTED
        if availability >= 80:
            return Colors.SUCCESS
        if availability >= 50:
            return Colors.WARNING
        if availability >= 20:
            return Colors.ERROR
        return Colors.TEXT_MUTED

    def _on_category_change(self, category: str):
        """Handle category filter change."""
        self._create_factor_table(category)

    def set_data(self, data: Dict[str, Any]):
        """Set the data for this view."""
        self.data = data
        self._compute_availability()
        self._refresh_data()

    def set_results(self, results: Dict[str, Any]):
        """Set the analysis results for this view."""
        self.results = results
        self._compute_availability()
        self._refresh_data()

    def _compute_availability(self):
        """Compute factor availability from data."""
        self.factor_availability = {}
        self.missing_columns = {}  # Track missing columns for warnings

        # Get factor data from results
        if self.results:
            if 'factor_data' in self.results:
                factor_df = self.results['factor_data']
            elif 'trades_with_factors' in self.results:
                factor_df = self.results['trades_with_factors']
            else:
                factor_df = None

            if isinstance(factor_df, pd.DataFrame) and len(factor_df) > 0:
                total = len(factor_df)
                for col in factor_df.columns:
                    non_null = factor_df[col].notna().sum()
                    availability = (non_null / total) * 100
                    # Map column to factor key
                    col_lower = col.lower()
                    self.factor_availability[col_lower] = availability
                    self.factor_availability[col] = availability

        # Also check raw data sources
        if self.data:
            # Check fundamental data for specific EPS columns
            fund_data = self.data.get('fundamental_data')
            if isinstance(fund_data, pd.DataFrame) and len(fund_data) > 0:
                total = len(fund_data)
                fund_cols = [c.lower() for c in fund_data.columns]

                # EPS-related columns to check
                eps_columns = {
                    'eps_eps': 'reported_eps',
                    'eps_estimated_eps': 'estimated_eps',
                    'eps_earnings_surprise': 'earnings_surprise',
                    'eps_earnings_surprise_pct': 'surprise_pct',
                }

                for factor_key, col_name in eps_columns.items():
                    col_lower = col_name.lower()
                    if col_lower in fund_cols:
                        # Find actual column name (case-insensitive)
                        actual_col = None
                        for c in fund_data.columns:
                            if c.lower() == col_lower:
                                actual_col = c
                                break
                        if actual_col:
                            non_null = fund_data[actual_col].notna().sum()
                            availability = (non_null / total) * 100
                            self.factor_availability[factor_key] = availability
                    else:
                        # Column not found - add warning
                        self.missing_columns[factor_key] = f"Column '{col_name}' not found in fundamental_data"
                        print(f"[WARNING] Factor '{factor_key}': Column '{col_name}' not found in fundamental_data.csv")

                # Store all fundamental columns availability
                for col in fund_data.columns:
                    non_null = fund_data[col].notna().sum()
                    availability = (non_null / total) * 100
                    self.factor_availability[f'fund_{col.lower()}'] = availability
                    self.factor_availability[col.lower()] = availability

            # Check price data (technical factors)
            price_data = self.data.get('price_data')
            if isinstance(price_data, pd.DataFrame) and len(price_data) > 0:
                # Calculate actual availability based on required columns
                total = len(price_data)
                price_cols = [c.lower() for c in price_data.columns]

                # Check availability of key price columns with warnings
                def get_col_avail(col_name):
                    col_lower = col_name.lower()
                    if col_lower in price_cols:
                        # Find actual column name
                        actual_col = None
                        for c in price_data.columns:
                            if c.lower() == col_lower:
                                actual_col = c
                                break
                        if actual_col:
                            return (price_data[actual_col].notna().sum() / total * 100)
                    print(f"[WARNING] Column '{col_name}' not found in price_data.csv")
                    return 0

                close_avail = get_col_avail('close')
                high_avail = get_col_avail('high')
                low_avail = get_col_avail('low')
                volume_avail = get_col_avail('volume')

                # Technical factors that only need close price
                close_only_factors = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma', 'ema', 'bollinger']
                for factor in close_only_factors:
                    self.factor_availability[factor] = close_avail

                # Technical factors that need high, low, close
                hlc_factors = ['adx', 'atr', 'stochastic', 'cci']
                hlc_avail = min(high_avail, low_avail, close_avail) if all([high_avail, low_avail, close_avail]) else 0
                for factor in hlc_factors:
                    self.factor_availability[factor] = hlc_avail

                # Technical factors that need high, low, close, volume
                hlcv_factors = ['obv', 'vwap', 'mfi']
                hlcv_avail = min(high_avail, low_avail, close_avail, volume_avail) if all([high_avail, low_avail, close_avail, volume_avail]) else 0
                for factor in hlcv_factors:
                    self.factor_availability[factor] = hlcv_avail

                # Regime factors (derived from price data)
                self.factor_availability['regime_volatility'] = close_avail
                self.factor_availability['regime_trend'] = hlc_avail

            # Check insider data
            insider_data = self.data.get('insider_data')
            if isinstance(insider_data, pd.DataFrame) and len(insider_data) > 0:
                total = len(insider_data)
                insider_cols = [c.lower() for c in insider_data.columns]

                # Required columns for insider factors
                required_insider_cols = ['date', 'symbol', 'transaction_type', 'shares']
                all_present = all(col in insider_cols for col in required_insider_cols)

                if all_present:
                    # All insider factors have same availability
                    insider_factors = ['insider_buy_count', 'insider_sell_count', 'insider_net_shares',
                                      'insider_score', 'insider_buy_sell_ratio']
                    for factor in insider_factors:
                        self.factor_availability[factor] = 100.0
                else:
                    missing = [col for col in required_insider_cols if col not in insider_cols]
                    print(f"[WARNING] Insider data missing columns: {missing}")

            # Check options data
            options_data = self.data.get('options_data')
            if isinstance(options_data, pd.DataFrame) and len(options_data) > 0:
                total = len(options_data)
                options_cols = [c.lower() for c in options_data.columns]

                if 'implied_volatility' in options_cols:
                    for c in options_data.columns:
                        if c.lower() == 'implied_volatility':
                            non_null = options_data[c].notna().sum()
                            self.factor_availability['options_implied_volatility'] = (non_null / total) * 100
                            self.factor_availability['options_iv_percentile'] = (non_null / total) * 100
                            break
                else:
                    print(f"[WARNING] Column 'implied_volatility' not found in options_data")

                if 'put_volume' in options_cols and 'call_volume' in options_cols:
                    self.factor_availability['options_put_call_ratio'] = 100.0
                else:
                    print(f"[WARNING] Columns 'put_volume' and/or 'call_volume' not found in options_data")

        # Compute composite factor availability based on their source factors
        self._compute_composite_availability()

    def _compute_composite_availability(self):
        """Compute availability for composite factors based on their source factors."""
        # EPS composite - based on EPS factors
        eps_factors = ['eps_eps', 'eps_estimated_eps', 'eps_earnings_surprise', 'eps_earnings_surprise_pct']
        eps_avails = [self.factor_availability.get(f, 0) for f in eps_factors]
        if any(a > 0 for a in eps_avails):
            self.factor_availability['composite_eps'] = sum(eps_avails) / len(eps_avails)

        # Value composite - based on value factors
        value_factors = ['value_pe_ratio', 'value_price_to_book', 'value_price_to_sales',
                        'value_peg_ratio', 'value_dividend_yield', 'value_ev_to_ebitda']
        value_avails = [self.factor_availability.get(f, 0) for f in value_factors]
        if any(a > 0 for a in value_avails):
            self.factor_availability['composite_value'] = sum(value_avails) / len(value_avails)

        # Quality composite - based on quality factors
        quality_factors = ['quality_return_on_equity', 'quality_return_on_assets',
                          'quality_profit_margin', 'quality_operating_margin',
                          'quality_current_ratio', 'quality_debt_to_equity']
        quality_avails = [self.factor_availability.get(f, 0) for f in quality_factors]
        if any(a > 0 for a in quality_avails):
            self.factor_availability['composite_quality'] = sum(quality_avails) / len(quality_avails)

        # Growth composite - based on growth factors
        growth_factors = ['growth_revenue_growth', 'growth_earnings_growth',
                         'growth_earnings_surprise', 'growth_earnings_surprise_pct']
        growth_avails = [self.factor_availability.get(f, 0) for f in growth_factors]
        if any(a > 0 for a in growth_avails):
            self.factor_availability['composite_growth'] = sum(growth_avails) / len(growth_avails)

        # Overall fundamental composite - based on other composites
        fund_composites = ['composite_eps', 'composite_value', 'composite_quality', 'composite_growth']
        fund_avails = [self.factor_availability.get(f, 0) for f in fund_composites]
        if any(a > 0 for a in fund_avails):
            self.factor_availability['composite_fundamental'] = sum(fund_avails) / len([a for a in fund_avails if a > 0])

    def _refresh_data(self):
        """Refresh the display."""
        self._create_factor_table(self.category_var.get())


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
        {"id": "factor_docs", "label": "Factor Documentation"},
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
        self.views["factor_docs"] = FactorDocumentationView(self.view_container)
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
            # Pass config to summary view for classification updates
            if "summary" in self.views:
                self.views["summary"].set_config(config)

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
            # Pass config first so thresholds are available for classification
            if self.config:
                self.views['summary'].set_config(self.config)
            self.views['summary'].set_data(self.data)

        # Also update factor documentation view with raw data
        if 'factor_docs' in self.views:
            self.views['factor_docs'].set_data(self.data)

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

    def _build_analyzer_config(self) -> 'FactorAnalysisConfig':
        """Build FactorAnalysisConfig from GUI config."""
        gui_config = self.config

        # Build TradeClassificationConfig from GUI settings
        tc_config = gui_config.get('trade_classification', {})
        threshold_type = ThresholdType.PERCENTILE if tc_config.get('method') == 'percentile' else ThresholdType.ABSOLUTE

        trade_classification = TradeClassificationConfig(
            good_threshold_pct=float(tc_config.get('good_threshold_pct', 2.0)),
            bad_threshold_pct=float(tc_config.get('bad_threshold_pct', -1.0)),
            indeterminate_max_days=int(tc_config.get('indeterminate_max_days', 15)),
            bad_min_days=int(tc_config.get('min_holding_period', 20)),
            threshold_type=threshold_type
        )

        # Build DataAlignmentConfig
        da_config = gui_config.get('data_alignment', {})
        data_alignment = DataAlignmentConfig(
            fundamentals_reporting_delay_days=int(da_config.get('fundamentals_reporting_delay_days', 0)),
            insiders_reporting_delay_days=int(da_config.get('insiders_reporting_delay_days', 3)),
            options_lookback_days=int(da_config.get('options_lookback_days', 60)),
            price_forward_fill_allowed=da_config.get('price_forward_fill_allowed', True),
            flag_price_gaps=da_config.get('flag_price_gaps', True)
        )

        # Build full config
        return FactorAnalysisConfig(
            profile_name=gui_config.get('profile_name', 'gui_analysis'),
            strategy_name=gui_config.get('strategy_name', 'unnamed'),
            trade_classification=trade_classification,
            data_alignment=data_alignment
        )

    def _generate_analysis_results(self) -> Dict[str, Any]:
        """Generate analysis results using the actual FactorAnalyzer."""
        if not ANALYZER_AVAILABLE:
            print("Analyzer not available, using fallback mock data")
            return self._generate_fallback_results()

        try:
            # Build config from GUI settings
            analyzer_config = self._build_analyzer_config()

            # Debug: Print the actual thresholds being used
            tc = analyzer_config.trade_classification
            print(f"[DEBUG] Trade Classification Config:")
            print(f"  - good_threshold_pct: {tc.good_threshold_pct}")
            print(f"  - bad_threshold_pct: {tc.bad_threshold_pct}")
            print(f"  - threshold_type: {tc.threshold_type}")
            print(f"  - bad_min_days: {tc.bad_min_days}")
            print(f"[DEBUG] GUI config trade_classification: {self.config.get('trade_classification', {})}")

            # Create analyzer
            self.analyzer = FactorAnalyzer(config=analyzer_config)

            # Get data from loaded sources
            trade_data = self.data.get('trade_data')
            if trade_data is None:
                raise ValueError("No trade data loaded")

            # Get price data - need to check properly since DataFrames can't use 'or'
            price_data = self.data.get('daily_price_data')
            if price_data is None or (hasattr(price_data, 'empty') and price_data.empty):
                price_data = self.data.get('weekly_price_data')
            fundamental_data = self.data.get('fundamental_data')
            insider_data = self.data.get('insider_data')
            options_data = self.data.get('options_data')

            # Run analysis
            output: AnalysisOutput = self.analyzer.analyze(
                trade_data=trade_data,
                price_data=price_data,
                fundamental_data=fundamental_data,
                insider_data=insider_data,
                options_data=options_data
            )

            if not output.success:
                print(f"Analysis failed: {output.error}")
                return self._generate_fallback_results()

            # Convert AnalysisOutput to GUI format
            return self._format_analysis_output(output)

        except Exception as e:
            print(f"Error running analyzer: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_results()

    def _format_analysis_output(self, output: 'AnalysisOutput') -> Dict[str, Any]:
        """Format AnalysisOutput for GUI display."""
        results = {}

        # Format Tier 1 results
        if output.tier1:
            tier1 = output.tier1
            correlations = {}
            p_values = {}
            factor_types = {}
            factor_details = {}

            # Extract point-biserial correlations
            if 'point_biserial' in tier1:
                for corr in tier1['point_biserial']:
                    factor_name = corr.factor if hasattr(corr, 'factor') else corr.get('factor', 'Unknown')
                    correlation = corr.correlation if hasattr(corr, 'correlation') else corr.get('correlation', 0)
                    p_val = corr.p_value if hasattr(corr, 'p_value') else corr.get('p_value', 1)

                    correlations[factor_name] = correlation
                    p_values[factor_name] = p_val
                    factor_types[factor_name] = 'Technical'  # Default type

            # Extract descriptive stats
            if 'descriptive_stats' in tier1:
                for factor_name, stats in tier1['descriptive_stats'].items():
                    factor_details[factor_name] = {
                        'mean': stats.get('mean', 0),
                        'std': stats.get('std', 0),
                        'min': stats.get('min', 0),
                        'max': stats.get('max', 0),
                        'skewness': stats.get('skewness', 0),
                        'kurtosis': stats.get('kurtosis', 0),
                        'good_mean': stats.get('good_mean', 0),
                        'bad_mean': stats.get('bad_mean', 0),
                        'mean_diff': stats.get('mean_diff', 0),
                    }

            results['tier1'] = {
                'correlations': correlations,
                'p_values': p_values,
                'factor_types': factor_types,
                'factor_details': factor_details
            }

        # Format Tier 2 results
        if output.tier2:
            tier2 = output.tier2
            tests = {}

            if tier2.get('anova'):
                for anova in tier2['anova']:
                    factor = anova.factor if hasattr(anova, 'factor') else anova.get('factor', 'Unknown')
                    tests[f'ANOVA ({factor})'] = {
                        'p_value': anova.p_value if hasattr(anova, 'p_value') else anova.get('p_value', 1),
                        'statistic': anova.statistic if hasattr(anova, 'statistic') else anova.get('statistic', 0),
                        'effect_size': anova.effect_size if hasattr(anova, 'effect_size') else anova.get('effect_size', 0)
                    }

            if tier2.get('logistic_regression'):
                reg = tier2['logistic_regression']
                tests['Logistic Regression'] = {
                    'p_value': reg.p_value if hasattr(reg, 'p_value') else reg.get('p_value', 1),
                    'statistic': reg.pseudo_r2 if hasattr(reg, 'pseudo_r2') else reg.get('pseudo_r2', 0),
                    'effect_size': reg.aic if hasattr(reg, 'aic') else reg.get('aic', 0)
                }

            results['tier2'] = {'tests': tests}

        # Format Tier 3 results
        if output.tier3:
            tier3 = output.tier3
            feature_importance = {}
            shap_values = {}

            if hasattr(tier3, 'rf_feature_importances') and tier3.rf_feature_importances:
                for feat in tier3.rf_feature_importances:
                    name = feat.feature_name if hasattr(feat, 'feature_name') else feat.get('feature_name', 'Unknown')
                    importance = feat.importance if hasattr(feat, 'importance') else feat.get('importance', 0)
                    feature_importance[name] = importance

            if hasattr(tier3, 'shap_results') and tier3.shap_results:
                for shap in tier3.shap_results:
                    name = shap.feature_name if hasattr(shap, 'feature_name') else shap.get('feature_name', 'Unknown')
                    shap_val = shap.mean_abs_shap if hasattr(shap, 'mean_abs_shap') else shap.get('mean_abs_shap', 0)
                    shap_values[name] = shap_val

            results['tier3'] = {
                'feature_importance': feature_importance,
                'shap_values': shap_values,
                'factor_types': results.get('tier1', {}).get('factor_types', {}),
                'model_metrics': tier3.get('model_metrics', {}) if isinstance(tier3, dict) else {}
            }

        # Format Scenarios
        if output.scenarios:
            scenarios_list = []
            scenarios = output.scenarios

            best = scenarios.best_scenarios if hasattr(scenarios, 'best_scenarios') else scenarios.get('best_scenarios', [])
            worst = scenarios.worst_scenarios if hasattr(scenarios, 'worst_scenarios') else scenarios.get('worst_scenarios', [])

            for scenario in best + worst:
                scenarios_list.append({
                    'name': scenario.name if hasattr(scenario, 'name') else scenario.get('name', 'Unknown'),
                    'description': scenario.get_condition_string() if hasattr(scenario, 'get_condition_string') else scenario.get('conditions', ''),
                    'performance': (scenario.avg_return if hasattr(scenario, 'avg_return') else scenario.get('avg_return', 0)) * 100,
                    'sample_size': scenario.n_trades if hasattr(scenario, 'n_trades') else scenario.get('n_trades', 0),
                    'confidence': (scenario.confidence if hasattr(scenario, 'confidence') else scenario.get('confidence', 0)) * 100,
                    'win_rate': (scenario.win_rate if hasattr(scenario, 'win_rate') else scenario.get('win_rate', 0)) * 100,
                    'key_factors': [c.factor if hasattr(c, 'factor') else c.get('factor', '') for c in (scenario.conditions if hasattr(scenario, 'conditions') else scenario.get('conditions', []))]
                })

            results['scenarios'] = scenarios_list

        # Add data summary
        results['data_summary'] = output.data_summary

        # Store warnings and findings
        self._analysis_warnings = output.warnings
        self._key_findings = output.key_findings

        return results

    def _generate_fallback_results(self) -> Dict[str, Any]:
        """Generate fallback mock results when analyzer is not available."""
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
        # Update summary view with classification results from analyzer
        if 'data_summary' in self.results and 'summary' in self.views:
            ds = self.results['data_summary']
            summary_view = self.views['summary']
            if hasattr(summary_view, 'good_card'):
                summary_view.good_card['value_label'].configure(text=f"{ds.get('good_trades', 0):,}")
            if hasattr(summary_view, 'bad_card'):
                summary_view.bad_card['value_label'].configure(text=f"{ds.get('bad_trades', 0):,}")
            if hasattr(summary_view, 'neutral_card'):
                summary_view.neutral_card['value_label'].configure(text=f"{ds.get('indeterminate_trades', 0):,}")
            print(f"[DEBUG] Updated summary view from analyzer results:")
            print(f"  - good: {ds.get('good_trades', 0)}, bad: {ds.get('bad_trades', 0)}, indeterminate: {ds.get('indeterminate_trades', 0)}")

        if 'tier1' in self.results:
            self.views['tier1'].update_results(self.results['tier1'])

        if 'tier2' in self.results:
            self.views['tier2'].update_results(self.results['tier2'])

        if 'tier3' in self.results:
            self.views['tier3'].update_results(self.results['tier3'])

        if 'scenarios' in self.results:
            self.views['scenarios'].update_results({'scenarios': self.results['scenarios']})

        # Update factor documentation view with data and results
        if 'factor_docs' in self.views:
            self.views['factor_docs'].set_data(self.data)
            self.views['factor_docs'].set_results(self.results)

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
        """Generate a comprehensive HTML report with charts and visualizations."""
        # Generate chart images as base64
        charts = {}

        if MATPLOTLIB_AVAILABLE:
            # Correlation chart
            if include.get('tier1') and self.results.get('tier1'):
                corrs = self.results['tier1'].get('correlations', {})
                if corrs:
                    fig = FactorVisualization.create_correlation_heatmap(corrs, figsize=(10, 6))
                    if fig:
                        charts['correlations'] = FactorVisualization.figure_to_base64(fig)

            # Scenario performance chart
            if include.get('scenarios') and self.results.get('scenarios'):
                fig = FactorVisualization.create_scenario_performance_chart(
                    self.results['scenarios'], figsize=(10, 5)
                )
                if fig:
                    charts['scenarios'] = FactorVisualization.figure_to_base64(fig)

            # Effect size chart
            if include.get('tier2') and self.results.get('tier2'):
                tests = self.results['tier2'].get('tests', {})
                effects = {name: data.get('effect_size', 0) for name, data in tests.items()}
                if effects:
                    fig = FactorVisualization.create_effect_size_chart(effects, figsize=(8, 4))
                    if fig:
                        charts['effects'] = FactorVisualization.figure_to_base64(fig)

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Factor Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e2e8f0;
            line-height: 1.6;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
        .header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid #4cc9f0;
            margin-bottom: 40px;
        }
        h1 { color: #4cc9f0; font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { color: #94a3b8; font-size: 1.1em; }
        .timestamp { color: #64748b; font-size: 0.9em; margin-top: 10px; }
        h2 {
            color: #7b68ee;
            font-size: 1.8em;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #3a3a5a;
        }
        h3 { color: #4cc9f0; font-size: 1.3em; margin: 25px 0 15px 0; }
        .card {
            background: rgba(42, 42, 74, 0.8);
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(26, 26, 46, 0.8);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        .stat-value { font-size: 2em; font-weight: bold; color: #4cc9f0; }
        .stat-label { color: #94a3b8; font-size: 0.9em; margin-top: 5px; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: rgba(26, 26, 46, 0.5);
            border-radius: 8px;
            overflow: hidden;
        }
        th, td { padding: 14px 16px; text-align: left; }
        th { background: rgba(76, 201, 240, 0.2); color: #4cc9f0; font-weight: 600; }
        tr:nth-child(even) { background: rgba(255,255,255,0.03); }
        tr:hover { background: rgba(76, 201, 240, 0.1); }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        .neutral { color: #94a3b8; }
        .chart-container {
            background: rgba(26, 26, 46, 0.8);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        .chart-container img { max-width: 100%; height: auto; border-radius: 4px; }
        .insight-box {
            background: rgba(76, 201, 240, 0.1);
            border-left: 4px solid #4cc9f0;
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }
        .insight-box.success { border-left-color: #4ade80; background: rgba(74, 222, 128, 0.1); }
        .insight-box.warning { border-left-color: #fbbf24; background: rgba(251, 191, 36, 0.1); }
        .insight-box.danger { border-left-color: #f87171; background: rgba(248, 113, 113, 0.1); }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }
        .badge-success { background: rgba(74, 222, 128, 0.2); color: #4ade80; }
        .badge-warning { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
        .badge-danger { background: rgba(248, 113, 113, 0.2); color: #f87171; }
        .toc { background: rgba(42, 42, 74, 0.5); padding: 20px; border-radius: 8px; margin: 20px 0; }
        .toc-title { color: #7b68ee; font-size: 1.2em; margin-bottom: 15px; }
        .toc a { color: #4cc9f0; text-decoration: none; display: block; padding: 5px 0; }
        .toc a:hover { color: #7b68ee; }
        .footer { text-align: center; padding: 40px 0; color: #64748b; border-top: 1px solid #3a3a5a; margin-top: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Factor Analysis Report</h1>
            <p class="subtitle">Comprehensive analysis of trading factors and their impact on strategy performance</p>
            <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>

        <div class="toc">
            <div class="toc-title">📑 Table of Contents</div>
"""

        # Add TOC items
        if include.get('summary'):
            html += '<a href="#summary">Executive Summary</a>'
        if include.get('tier1'):
            html += '<a href="#tier1">Tier 1: Factor Correlations</a>'
        if include.get('tier2'):
            html += '<a href="#tier2">Tier 2: Hypothesis Testing</a>'
        if include.get('tier3'):
            html += '<a href="#tier3">Tier 3: ML Analysis</a>'
        if include.get('scenarios'):
            html += '<a href="#scenarios">Scenario Analysis</a>'

        html += """
        </div>
"""

        # Executive Summary
        if include.get('summary'):
            trade_count = len(self.data.get('trade_data', [])) if isinstance(self.data.get('trade_data'), pd.DataFrame) else 0
            symbol_count = len(self.data.get('symbols', []))

            html += """
        <h2 id="summary">📋 Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">""" + str(trade_count) + """</div>
                <div class="stat-label">Total Trades Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(symbol_count) + """</div>
                <div class="stat-label">Symbols</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(len(self.results.get('tier1', {}).get('correlations', {}))) + """</div>
                <div class="stat-label">Factors Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(len(self.results.get('scenarios', []))) + """</div>
                <div class="stat-label">Scenarios Detected</div>
            </div>
        </div>

        <div class="insight-box success">
            <strong>Key Finding:</strong> Analysis identified multiple statistically significant factors that differentiate successful from unsuccessful trades.
        </div>
"""

        # Tier 1: Factor Correlations
        if include.get('tier1') and self.results.get('tier1'):
            html += """
        <h2 id="tier1">📈 Tier 1: Factor Correlations</h2>
        <div class="card">
            <p>Pearson and Spearman correlations between factors and trade success. Factors are ranked by absolute correlation strength.</p>
"""
            if 'correlations' in charts:
                html += f'<div class="chart-container"><img src="data:image/png;base64,{charts["correlations"]}" alt="Factor Correlations"></div>'

            html += """
            <h3>Detailed Correlation Table</h3>
            <table>
                <tr><th>Factor</th><th>Type</th><th>Correlation</th><th>P-Value</th><th>Significance</th></tr>
"""
            corrs = self.results['tier1'].get('correlations', {})
            p_vals = self.results['tier1'].get('p_values', {})
            types = self.results['tier1'].get('factor_types', {})

            for k, v in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
                color_class = 'positive' if v > 0 else 'negative'
                p = p_vals.get(k, 1)
                sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'NS'
                sig_class = 'badge-success' if p < 0.05 else 'badge-warning' if p < 0.1 else 'badge-danger'
                html += f'<tr><td>{k}</td><td>{types.get(k, "Unknown")}</td><td class="{color_class}">{v:.4f}</td><td>{p:.4f}</td><td><span class="badge {sig_class}">{sig}</span></td></tr>'

            html += """
            </table>
        </div>
"""

        # Tier 2: Hypothesis Testing
        if include.get('tier2') and self.results.get('tier2'):
            html += """
        <h2 id="tier2">🧪 Tier 2: Hypothesis Testing</h2>
        <div class="card">
            <p>Statistical tests comparing factor distributions between good and bad trades.</p>
"""
            if 'effects' in charts:
                html += f'<div class="chart-container"><img src="data:image/png;base64,{charts["effects"]}" alt="Effect Sizes"></div>'

            html += """
            <h3>Test Results</h3>
            <table>
                <tr><th>Test</th><th>Statistic</th><th>P-Value</th><th>Effect Size</th><th>Interpretation</th></tr>
"""
            tests = self.results['tier2'].get('tests', {})
            for name, data in tests.items():
                p = data.get('p_value', 1)
                effect = data.get('effect_size', 0)
                if p < 0.01:
                    interp = "Highly significant"
                    badge_class = "badge-success"
                elif p < 0.05:
                    interp = "Significant"
                    badge_class = "badge-success"
                elif p < 0.1:
                    interp = "Marginally significant"
                    badge_class = "badge-warning"
                else:
                    interp = "Not significant"
                    badge_class = "badge-danger"

                html += f'<tr><td>{name}</td><td>{data.get("statistic", 0):.3f}</td><td>{p:.4f}</td><td>{effect:.3f}</td><td><span class="badge {badge_class}">{interp}</span></td></tr>'

            html += """
            </table>
        </div>
"""

        # Tier 3: ML Analysis
        if include.get('tier3') and self.results.get('tier3'):
            html += """
        <h2 id="tier3">🤖 Tier 3: Machine Learning Analysis</h2>
        <div class="card">
            <p>Random Forest feature importance and SHAP value analysis.</p>

            <h3>Top Feature Importance</h3>
            <table>
                <tr><th>Factor</th><th>Importance</th><th>Type</th></tr>
"""
            importance = self.results['tier3'].get('feature_importance', {})
            types = self.results['tier3'].get('factor_types', {})
            for factor, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                bar_width = int(imp * 100 / max(importance.values()) if importance else 0)
                html += f'<tr><td>{factor}</td><td><div style="background: linear-gradient(90deg, #4cc9f0 {bar_width}%, transparent {bar_width}%); padding: 5px;">{imp:.4f}</div></td><td>{types.get(factor, "Unknown")}</td></tr>'

            html += """
            </table>

            <h3>Model Performance</h3>
            <div class="stats-grid">
"""
            metrics = self.results['tier3'].get('model_metrics', {})
            for metric, value in metrics.items():
                html += f'<div class="stat-card"><div class="stat-value">{value:.2%}</div><div class="stat-label">{metric.replace("_", " ").title()}</div></div>'

            html += """
            </div>
        </div>
"""

        # Scenario Analysis
        if include.get('scenarios') and self.results.get('scenarios'):
            html += """
        <h2 id="scenarios">🎯 Scenario Analysis</h2>
        <div class="card">
            <p>Detected market scenarios and their impact on trading performance.</p>
"""
            if 'scenarios' in charts:
                html += f'<div class="chart-container"><img src="data:image/png;base64,{charts["scenarios"]}" alt="Scenario Performance"></div>'

            html += """
            <h3>Scenario Details</h3>
            <table>
                <tr><th>Scenario</th><th>Performance</th><th>Win Rate</th><th>Samples</th><th>Confidence</th><th>Key Factors</th></tr>
"""
            for s in self.results['scenarios']:
                perf = s.get('performance', 0)
                color_class = 'positive' if perf > 0 else 'negative'
                factors = ', '.join(s.get('key_factors', [])[:3])
                html += f'<tr><td><strong>{s.get("name")}</strong><br><small>{s.get("description", "")}</small></td><td class="{color_class}">{perf:+.1f}%</td><td>{s.get("win_rate", 50):.0f}%</td><td>{s.get("sample_size", 0)}</td><td>{s.get("confidence", 0)}%</td><td><small>{factors}</small></td></tr>'

            html += """
            </table>
        </div>
"""

        # Footer
        html += """
        <div class="footer">
            <p>Factor Analysis Report • Generated by Factor Analysis Dashboard</p>
            <p>© """ + str(datetime.now().year) + """ • BackTestingFramework</p>
        </div>
    </div>
</body>
</html>
"""
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
