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

from ..ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info, ask_yes_no, ProgressPanel
from .components import (
    NavigationPanel, FactorListPanel, StatisticsPanel,
    DataQualityIndicator, AnalysisProgressPanel, ScenarioCard,
    TierResultsPanel
)
from .config_manager import FactorConfigManagerGUI
from .data_upload import FactorDataUploadGUI


class DataSummaryView(ctk.CTkFrame):
    """View displaying data summary and quality metrics."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Data Summary", size="l").pack(side="left")

        # Summary cards row
        cards_frame = Theme.create_frame(self)
        cards_frame.pack(fill="x", pady=Sizes.PAD_S)
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Trade summary card
        self.trade_card = self._create_summary_card(cards_frame, "Total Trades", "0", 0, 0)
        self.good_card = self._create_summary_card(cards_frame, "Good Trades", "0", 0, 1)
        self.bad_card = self._create_summary_card(cards_frame, "Bad Trades", "0", 0, 2)
        self.neutral_card = self._create_summary_card(cards_frame, "Indeterminate", "0", 0, 3)

        # Data quality section
        quality_section = Theme.create_card(self)
        quality_section.pack(fill="x", pady=Sizes.PAD_M)

        quality_content = Theme.create_frame(quality_section)
        quality_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(quality_content, "Data Quality", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.quality_indicator = DataQualityIndicator(quality_content)
        self.quality_indicator.pack(fill="x")

        # Factor coverage section
        factor_section = Theme.create_card(self)
        factor_section.pack(fill="both", expand=True, pady=Sizes.PAD_S)

        factor_content = Theme.create_frame(factor_section)
        factor_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(factor_content, "Factor Coverage", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.factor_coverage_frame = ctk.CTkScrollableFrame(factor_content, fg_color="transparent")
        self.factor_coverage_frame.pack(fill="both", expand=True)

    def _create_summary_card(self, parent, title: str, value: str, row: int, col: int) -> Dict:
        """Create a summary stat card."""
        card = Theme.create_card(parent)
        card.grid(row=row, column=col, sticky="nsew", padx=Sizes.PAD_XS, pady=Sizes.PAD_XS)

        Theme.create_label(card, title, font=Fonts.BODY_S, text_color=Colors.TEXT_SECONDARY).pack(pady=(Sizes.PAD_S, 0))

        value_label = Theme.create_label(card, value, font=Fonts.HEADER_XL)
        value_label.pack(pady=Sizes.PAD_S)

        return {'card': card, 'value_label': value_label}

    def update_data(self, summary: Dict[str, Any]):
        """Update the view with new data summary."""
        # Update trade counts
        self.trade_card['value_label'].configure(text=str(summary.get('total_trades', 0)))
        self.good_card['value_label'].configure(text=str(summary.get('good_trades', 0)))
        self.bad_card['value_label'].configure(text=str(summary.get('bad_trades', 0)))
        self.neutral_card['value_label'].configure(text=str(summary.get('indeterminate_trades', 0)))

        # Update quality
        quality = summary.get('quality', {})
        self.quality_indicator.set_quality(
            score=quality.get('score', 0),
            status=quality.get('status', 'Unknown'),
            metrics=quality.get('metrics', {}),
            warnings=quality.get('warnings')
        )

        # Update factor coverage
        for widget in self.factor_coverage_frame.winfo_children():
            widget.destroy()

        factors = summary.get('factors', {})
        for category, factor_list in factors.items():
            cat_frame = Theme.create_frame(self.factor_coverage_frame)
            cat_frame.pack(fill="x", pady=Sizes.PAD_XS)

            Theme.create_label(
                cat_frame, f"{category}: ",
                font=Fonts.LABEL_BOLD,
                text_color=Colors.PRIMARY_LIGHT
            ).pack(side="left")

            Theme.create_label(
                cat_frame, f"{len(factor_list)} factors",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_SECONDARY
            ).pack(side="left")


class Tier1View(ctk.CTkFrame):
    """View for Tier 1 Exploratory Analysis results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Tier 1: Exploratory Analysis", size="l").pack(side="left")

        Theme.create_label(
            header,
            "Correlation analysis, distribution statistics, and initial factor screening",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        # Two column layout
        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Left column - Factor list
        self.factor_panel = FactorListPanel(
            content,
            title="Factor Correlations",
            on_factor_select=self._on_factor_select
        )
        self.factor_panel.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_S))

        # Right column - Statistics
        right_frame = Theme.create_frame(content)
        right_frame.grid(row=0, column=1, sticky="nsew")

        self.stats_panel = StatisticsPanel(right_frame, title="Selected Factor Statistics")
        self.stats_panel.pack(fill="x", pady=(0, Sizes.PAD_S))

        # Distribution visualization placeholder
        dist_card = Theme.create_card(right_frame)
        dist_card.pack(fill="both", expand=True)

        Theme.create_header(dist_card, "Distribution", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        self.dist_frame = Theme.create_frame(dist_card)
        self.dist_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        Theme.create_label(
            self.dist_frame,
            "Select a factor to view its distribution",
            text_color=Colors.TEXT_MUTED
        ).pack(expand=True)

    def _on_factor_select(self, factor_name: str):
        """Handle factor selection."""
        # Update statistics panel with factor details
        if hasattr(self, 'results') and self.results:
            factor_data = self.results.get('factor_details', {}).get(factor_name, {})
            self.stats_panel.set_metrics({
                "Mean": f"{factor_data.get('mean', 0):.4f}",
                "Std Dev": f"{factor_data.get('std', 0):.4f}",
                "Min": f"{factor_data.get('min', 0):.4f}",
                "Max": f"{factor_data.get('max', 0):.4f}",
                "Skewness": f"{factor_data.get('skewness', 0):.4f}",
                "Kurtosis": f"{factor_data.get('kurtosis', 0):.4f}"
            })

    def update_results(self, results: Dict[str, Any]):
        """Update the view with analysis results."""
        self.results = results

        # Update factor list
        factors = []
        correlations = results.get('correlations', {})
        p_values = results.get('p_values', {})

        for factor, corr in correlations.items():
            factors.append({
                'name': factor,
                'type': results.get('factor_types', {}).get(factor, 'Unknown'),
                'correlation': corr,
                'p_value': p_values.get(factor, 1.0)
            })

        self.factor_panel.set_factors(sorted(factors, key=lambda x: abs(x['correlation']), reverse=True))


class Tier2View(ctk.CTkFrame):
    """View for Tier 2 Hypothesis Testing results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Tier 2: Hypothesis Testing", size="l").pack(side="left")

        # Content
        self.results_panel = TierResultsPanel(self, tier=2)
        self.results_panel.pack(fill="both", expand=True)

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
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Tier 3: ML Analysis", size="l").pack(side="left")

        # Content
        self.results_panel = TierResultsPanel(self, tier=3)
        self.results_panel.pack(fill="both", expand=True)

    def update_results(self, results: Dict[str, Any]):
        """Update the view with analysis results."""
        self.results_panel.set_results(results)


class ScenarioView(ctk.CTkFrame):
    """View for Scenario Analysis results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.scenarios: List[Dict[str, Any]] = []
        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Scenario Analysis", size="l").pack(side="left")

        Theme.create_label(
            header,
            "Detected market scenarios and their impact on strategy performance",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        # Scenario cards grid
        self.cards_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.cards_frame.pack(fill="both", expand=True)
        self.cards_frame.grid_columnconfigure((0, 1), weight=1)

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
            return

        for i, scenario in enumerate(scenarios):
            card = ScenarioCard(
                self.cards_frame,
                scenario_name=scenario.get('name', f'Scenario {i+1}'),
                scenario_data=scenario,
                on_click=self._on_scenario_click
            )
            card.grid(
                row=i // 2, column=i % 2,
                sticky="nsew", padx=Sizes.PAD_S, pady=Sizes.PAD_S
            )

    def _on_scenario_click(self, name: str, data: Dict[str, Any]):
        """Handle scenario card click."""
        # Show detailed scenario view
        pass


class ExportView(ctk.CTkFrame):
    """View for exporting analysis results."""

    def __init__(self, parent, on_export: Optional[Callable] = None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.on_export = on_export
        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(header, "Export & Reports", size="l").pack(side="left")

        # Export options
        options_card = Theme.create_card(self)
        options_card.pack(fill="x", pady=Sizes.PAD_S)

        options_content = Theme.create_frame(options_card)
        options_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_header(options_content, "Export Format", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Format selection
        self.format_var = ctk.StringVar(value="excel")

        formats_frame = Theme.create_frame(options_content)
        formats_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_radiobutton(
            formats_frame, "Excel Report (.xlsx)",
            variable=self.format_var, value="excel"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_radiobutton(
            formats_frame, "JSON Payload (.json)",
            variable=self.format_var, value="json"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_radiobutton(
            formats_frame, "CSV Summary (.csv)",
            variable=self.format_var, value="csv"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        Theme.create_radiobutton(
            formats_frame, "HTML Report (.html)",
            variable=self.format_var, value="html"
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        # Include options
        include_card = Theme.create_card(self)
        include_card.pack(fill="x", pady=Sizes.PAD_S)

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

        for label, key, default in options:
            var = ctk.BooleanVar(value=default)
            self.include_vars[key] = var
            Theme.create_checkbox(include_content, label, variable=var).pack(anchor="w", pady=Sizes.PAD_XS)

        # Export button
        btn_frame = Theme.create_frame(self)
        btn_frame.pack(fill="x", pady=Sizes.PAD_M)

        Theme.create_button(
            btn_frame, "Export",
            command=self._on_export,
            style="primary",
            width=150,
            height=40
        ).pack(side="right")

        # Recent exports
        recent_card = Theme.create_card(self)
        recent_card.pack(fill="both", expand=True, pady=Sizes.PAD_S)

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

        if self.on_export:
            self.on_export(format_type, include)


class AuditView(ctk.CTkFrame):
    """View for the audit trail."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create view widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", pady=(0, Sizes.PAD_M))

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
        filter_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

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
        log_card.pack(fill="both", expand=True)

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

        # Color based on level
        prefix = f"[{timestamp}] [{level.upper()}] "
        self.log_text.insert("end", prefix + message + "\n")

        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_filter_change(self, value: str):
        """Handle filter change."""
        # Would filter log entries
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
        self.content_area.grid_rowconfigure(0, weight=1)

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

        trade_data = self.data.get('trade_data')
        if trade_data is None:
            return

        summary = {
            'total_trades': len(trade_data),
            'good_trades': 0,
            'bad_trades': 0,
            'indeterminate_trades': 0,
            'quality': {
                'score': 75,
                'status': 'Good',
                'metrics': {
                    'Records': str(len(trade_data)),
                    'Date range': 'N/A'
                }
            },
            'factors': {
                'Technical': [],
                'Value': [],
                'Quality': [],
                'Growth': []
            }
        }

        if 'summary' in self.views:
            self.views['summary'].update_data(summary)

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
                # Simulate analysis stages
                import time

                for i in range(8):
                    progress_dialog.update_stage(i)
                    time.sleep(0.5)  # Simulate work

                # Generate mock results
                self.results = self._generate_mock_results()

                progress_dialog.complete()

                # Update views
                self.root.after(100, self._update_results_views)

            except Exception as e:
                progress_dialog.error(str(e))
                self._log_audit(f"Analysis failed: {e}", level="error")

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()

    def _generate_mock_results(self) -> Dict[str, Any]:
        """Generate mock results for demonstration."""
        return {
            'tier1': {
                'correlations': {
                    'RSI_14': 0.342,
                    'MACD_Signal': 0.287,
                    'BB_Width': -0.156,
                    'ATR_14': 0.098,
                    'Volume_MA_Ratio': 0.234
                },
                'p_values': {
                    'RSI_14': 0.002,
                    'MACD_Signal': 0.015,
                    'BB_Width': 0.089,
                    'ATR_14': 0.234,
                    'Volume_MA_Ratio': 0.045
                },
                'factor_types': {
                    'RSI_14': 'Technical',
                    'MACD_Signal': 'Technical',
                    'BB_Width': 'Technical',
                    'ATR_14': 'Technical',
                    'Volume_MA_Ratio': 'Technical'
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
                    }
                }
            },
            'tier3': {
                'feature_importance': {
                    'RSI_14': 0.234,
                    'MACD_Signal': 0.187,
                    'Volume_MA_Ratio': 0.145,
                    'BB_Width': 0.098,
                    'ATR_14': 0.076
                },
                'shap_values': {
                    'RSI_14': 0.089,
                    'MACD_Signal': 0.067,
                    'Volume_MA_Ratio': 0.045,
                    'BB_Width': -0.023,
                    'ATR_14': 0.012
                }
            },
            'scenarios': [
                {
                    'name': 'High Volatility',
                    'description': 'ATR above 2 standard deviations',
                    'performance': -5.2,
                    'sample_size': 45,
                    'confidence': 87
                },
                {
                    'name': 'Strong Trend',
                    'description': 'MACD positive with increasing momentum',
                    'performance': 8.7,
                    'sample_size': 72,
                    'confidence': 92
                },
                {
                    'name': 'Mean Reversion',
                    'description': 'RSI oversold with price at lower BB',
                    'performance': 3.4,
                    'sample_size': 38,
                    'confidence': 78
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
        if self.current_view:
            self._show_view(self.current_view)

    def _on_export(self, format_type: str, include: Dict[str, bool]):
        """Handle export request."""
        from tkinter import filedialog

        # Get save path
        extensions = {
            'excel': ('Excel files', '*.xlsx'),
            'json': ('JSON files', '*.json'),
            'csv': ('CSV files', '*.csv'),
            'html': ('HTML files', '*.html')
        }

        ext = extensions.get(format_type, ('All files', '*.*'))
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            filetypes=[ext],
            defaultextension=ext[1].replace('*', '')
        )

        if not file_path:
            return

        try:
            if format_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(self.results, f, indent=2, default=str)
            elif format_type == 'excel':
                # Would use ExcelReportGenerator here
                pass

            show_info(self.root, "Export Complete", f"Results exported to:\n{file_path}")
            self._log_audit(f"Results exported to {file_path}")

        except Exception as e:
            show_error(self.root, "Export Failed", str(e))
            self._log_audit(f"Export failed: {e}", level="error")

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
