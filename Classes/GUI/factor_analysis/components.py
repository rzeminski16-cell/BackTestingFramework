"""
Factor Analysis GUI Components.

Reusable CustomTkinter components for the Factor Analysis GUIs.
"""

import customtkinter as ctk
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import json

from ..ctk_theme import Theme, Colors, Fonts, Sizes


class NavigationPanel(ctk.CTkFrame):
    """
    Left navigation panel with view selection.

    Layout:
    ┌─────────────────────┐
    │ [Logo/Title]        │
    ├─────────────────────┤
    │ ▶ Data Summary      │
    │   Tier 1 Analysis   │
    │   Tier 2 Analysis   │
    │   Tier 3 Analysis   │
    │   Scenario Analysis │
    │   Export & Reports  │
    │   Audit Trail       │
    ├─────────────────────┤
    │ [Config] [Data]     │
    └─────────────────────┘
    """

    def __init__(
        self,
        parent,
        views: List[Dict[str, str]],
        on_view_change: Optional[Callable[[str], None]] = None,
        on_config_click: Optional[Callable] = None,
        on_data_click: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize navigation panel.

        Args:
            parent: Parent widget
            views: List of dicts with 'id', 'label', and optional 'icon'
            on_view_change: Callback when view selection changes
            on_config_click: Callback for config button
            on_data_click: Callback for data button
        """
        super().__init__(parent, fg_color=Colors.BG_MEDIUM, corner_radius=0, **kwargs)

        self.views = views
        self.on_view_change = on_view_change
        self.on_config_click = on_config_click
        self.on_data_click = on_data_click
        self.current_view = views[0]['id'] if views else None
        self.view_buttons: Dict[str, ctk.CTkButton] = {}

        self._create_widgets()

    def _create_widgets(self):
        """Create navigation widgets."""
        # Header
        header_frame = Theme.create_frame(self)
        header_frame.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_L)

        Theme.create_header(
            header_frame,
            "Factor Analysis",
            size="l"
        ).pack(anchor="w")

        Theme.create_label(
            header_frame,
            "Strategy Performance Module",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w")

        # Separator
        ctk.CTkFrame(self, fg_color=Colors.BORDER, height=1).pack(fill="x", pady=Sizes.PAD_M)

        # Navigation buttons
        nav_frame = Theme.create_frame(self)
        nav_frame.pack(fill="both", expand=True, padx=Sizes.PAD_S)

        for view in self.views:
            btn = ctk.CTkButton(
                nav_frame,
                text=f"  {view['label']}",
                font=Fonts.BODY_M,
                fg_color="transparent",
                hover_color=Colors.SURFACE_HOVER,
                text_color=Colors.TEXT_SECONDARY,
                anchor="w",
                height=40,
                corner_radius=Sizes.RADIUS_M,
                command=lambda v=view['id']: self._on_view_click(v)
            )
            btn.pack(fill="x", pady=Sizes.PAD_XS)
            self.view_buttons[view['id']] = btn

        # Update initial selection
        if self.current_view:
            self._update_selection()

        # Separator before action buttons
        ctk.CTkFrame(self, fg_color=Colors.BORDER, height=1).pack(fill="x", pady=Sizes.PAD_M)

        # Action buttons
        action_frame = Theme.create_frame(self)
        action_frame.pack(fill="x", padx=Sizes.PAD_S, pady=Sizes.PAD_M)

        Theme.create_button(
            action_frame,
            "Configuration",
            command=self._on_config,
            style="secondary",
            width=110,
            height=32
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        Theme.create_button(
            action_frame,
            "Load Data",
            command=self._on_data,
            style="secondary",
            width=100,
            height=32
        ).pack(side="left")

    def _on_view_click(self, view_id: str):
        """Handle view button click."""
        self.current_view = view_id
        self._update_selection()
        if self.on_view_change:
            self.on_view_change(view_id)

    def _update_selection(self):
        """Update button styles based on selection."""
        for view_id, btn in self.view_buttons.items():
            if view_id == self.current_view:
                btn.configure(
                    fg_color=Colors.PRIMARY,
                    text_color=Colors.TEXT_PRIMARY
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=Colors.TEXT_SECONDARY
                )

    def _on_config(self):
        """Handle config button click."""
        if self.on_config_click:
            self.on_config_click()

    def _on_data(self):
        """Handle data button click."""
        if self.on_data_click:
            self.on_data_click()

    def set_view(self, view_id: str):
        """Programmatically set the current view."""
        if view_id in self.view_buttons:
            self._on_view_click(view_id)


class FactorListPanel(ctk.CTkFrame):
    """
    Panel displaying a list of factors with their statistics.

    Features:
    - Sortable columns
    - Factor type filtering
    - Selection highlighting
    - Detailed view on click
    """

    def __init__(
        self,
        parent,
        title: str = "Factors",
        on_factor_select: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.on_factor_select = on_factor_select
        self.factors: List[Dict[str, Any]] = []
        self.selected_factor: Optional[str] = None
        self.factor_widgets: Dict[str, ctk.CTkFrame] = {}

        self._create_widgets(title)

    def _create_widgets(self, title: str):
        """Create panel widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, title, size="s").pack(side="left")

        self.count_label = Theme.create_label(
            header,
            "(0 factors)",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        self.count_label.pack(side="right")

        # Filter dropdown
        filter_frame = Theme.create_frame(self)
        filter_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        Theme.create_label(filter_frame, "Filter:", font=Fonts.BODY_S).pack(side="left")

        self.filter_var = ctk.StringVar(value="All")
        self.filter_combo = Theme.create_combobox(
            filter_frame,
            values=["All", "Technical", "EPS Fundamentals", "Value", "Quality", "Growth", "Insider", "Regime", "Composite", "Options", "Other"],
            variable=self.filter_var,
            command=self._on_filter_change,
            width=140
        )
        self.filter_combo.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Column headers
        columns_frame = Theme.create_frame(self)
        columns_frame.pack(fill="x", padx=Sizes.PAD_M)

        Theme.create_label(columns_frame, "Factor", font=Fonts.LABEL_BOLD, width=150).pack(side="left")
        Theme.create_label(columns_frame, "Type", font=Fonts.LABEL_BOLD, width=80).pack(side="left")
        Theme.create_label(columns_frame, "Correlation", font=Fonts.LABEL_BOLD, width=90).pack(side="left")
        Theme.create_label(columns_frame, "P-Value", font=Fonts.LABEL_BOLD, width=70).pack(side="left")

        # Scrollable factor list
        self.list_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0
        )
        self.list_frame.pack(fill="both", expand=True, padx=Sizes.PAD_S, pady=Sizes.PAD_S)

    def set_factors(self, factors: List[Dict[str, Any]]):
        """
        Set the factors to display.

        Args:
            factors: List of factor dicts with keys:
                - name: Factor name
                - type: Factor category
                - correlation: Correlation coefficient
                - p_value: Statistical p-value
        """
        self.factors = factors
        self._refresh_list()

    def _refresh_list(self):
        """Refresh the factor list display."""
        # Clear existing
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        self.factor_widgets.clear()

        # Filter factors
        filter_type = self.filter_var.get()
        filtered = self.factors
        if filter_type != "All":
            filtered = [f for f in self.factors if f.get('type', '').lower() == filter_type.lower()]

        self.count_label.configure(text=f"({len(filtered)} factors)")

        # Create factor rows
        for factor in filtered:
            self._create_factor_row(factor)

    def _create_factor_row(self, factor: Dict[str, Any]):
        """Create a row for a single factor."""
        name = factor.get('name', 'Unknown')

        row = ctk.CTkFrame(
            self.list_frame,
            fg_color="transparent",
            corner_radius=Sizes.RADIUS_S,
            height=36
        )
        row.pack(fill="x", pady=1)
        row.bind("<Enter>", lambda e, r=row: r.configure(fg_color=Colors.SURFACE_HOVER))
        row.bind("<Leave>", lambda e, r=row, n=name: r.configure(
            fg_color=Colors.PRIMARY if n == self.selected_factor else "transparent"
        ))
        row.bind("<Button-1>", lambda e, n=name: self._on_factor_click(n))

        # Factor name
        name_label = Theme.create_label(row, name, width=150)
        name_label.pack(side="left", padx=(Sizes.PAD_S, 0))
        name_label.bind("<Button-1>", lambda e, n=name: self._on_factor_click(n))

        # Type
        factor_type = factor.get('type', '-')
        type_label = Theme.create_label(
            row, factor_type, width=80,
            text_color=Colors.TEXT_SECONDARY
        )
        type_label.pack(side="left")
        type_label.bind("<Button-1>", lambda e, n=name: self._on_factor_click(n))

        # Correlation (with color coding) - use correlation_display if available, else correlation
        corr_display = factor.get('correlation_display', factor.get('correlation', 0))
        if corr_display is None:
            corr_text = "N/A"
            corr_color = Colors.TEXT_MUTED
        else:
            corr_text = f"{corr_display:.3f}"
            corr_color = Colors.SUCCESS if corr_display > 0 else Colors.ERROR if corr_display < 0 else Colors.TEXT_SECONDARY
        corr_label = Theme.create_label(
            row, corr_text, width=90,
            text_color=corr_color
        )
        corr_label.pack(side="left")
        corr_label.bind("<Button-1>", lambda e, n=name: self._on_factor_click(n))

        # P-value (with significance indicator)
        p_val = factor.get('p_value')
        if p_val is None:
            p_text = "N/A"
            p_color = Colors.TEXT_MUTED
        else:
            p_text = f"{p_val:.4f}"
            if p_val < 0.01:
                p_text += " ***"
                p_color = Colors.SUCCESS
            elif p_val < 0.05:
                p_text += " **"
                p_color = Colors.WARNING
            elif p_val < 0.1:
                p_text += " *"
                p_color = Colors.TEXT_PRIMARY
            else:
                p_color = Colors.TEXT_MUTED

        p_label = Theme.create_label(row, p_text, width=90, text_color=p_color)
        p_label.pack(side="left")
        p_label.bind("<Button-1>", lambda e, n=name: self._on_factor_click(n))

        self.factor_widgets[name] = row

    def _on_factor_click(self, factor_name: str):
        """Handle factor selection."""
        # Update selection
        if self.selected_factor and self.selected_factor in self.factor_widgets:
            self.factor_widgets[self.selected_factor].configure(fg_color="transparent")

        self.selected_factor = factor_name
        if factor_name in self.factor_widgets:
            self.factor_widgets[factor_name].configure(fg_color=Colors.PRIMARY)

        if self.on_factor_select:
            self.on_factor_select(factor_name)

    def _on_filter_change(self, value: str):
        """Handle filter change."""
        self._refresh_list()


class StatisticsPanel(ctk.CTkFrame):
    """
    Panel displaying statistical analysis results.

    Features:
    - Key metrics display
    - Significance indicators
    - Expandable details
    """

    def __init__(self, parent, title: str = "Statistics", **kwargs):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.metrics: Dict[str, Any] = {}
        self._create_widgets(title)

    def _create_widgets(self, title: str):
        """Create panel widgets."""
        # Header
        Theme.create_header(
            self, title, size="s"
        ).pack(anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        # Metrics container
        self.metrics_frame = Theme.create_frame(self)
        self.metrics_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

    def set_metrics(self, metrics: Dict[str, Any]):
        """
        Set the metrics to display.

        Args:
            metrics: Dict of metric_name -> value or dict with 'value', 'label', 'color'
        """
        self.metrics = metrics
        self._refresh_display()

    def _refresh_display(self):
        """Refresh the metrics display."""
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        for key, value in self.metrics.items():
            row = Theme.create_frame(self.metrics_frame)
            row.pack(fill="x", pady=Sizes.PAD_XS)

            if isinstance(value, dict):
                label = value.get('label', key)
                val = value.get('value', '-')
                color = value.get('color', Colors.TEXT_PRIMARY)
            else:
                label = key
                val = value
                color = Colors.TEXT_PRIMARY

            Theme.create_label(
                row, f"{label}:",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_SECONDARY
            ).pack(side="left")

            Theme.create_label(
                row, str(val),
                font=Fonts.LABEL_BOLD,
                text_color=color
            ).pack(side="right")

    def add_metric(self, key: str, value: Any, label: str = None, color: str = None):
        """Add or update a single metric."""
        self.metrics[key] = {
            'value': value,
            'label': label or key,
            'color': color or Colors.TEXT_PRIMARY
        }
        self._refresh_display()


class DataQualityIndicator(ctk.CTkFrame):
    """
    Visual indicator for data quality status.

    Shows:
    - Overall quality score
    - Missing data percentage
    - Data coverage
    - Warnings/issues
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """Create indicator widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, "Data Quality", size="s").pack(side="left")

        self.status_label = Theme.create_label(
            header, "Unknown",
            font=Fonts.LABEL_BOLD,
            text_color=Colors.TEXT_MUTED
        )
        self.status_label.pack(side="right")

        # Quality score bar
        score_frame = Theme.create_frame(self)
        score_frame.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)

        Theme.create_label(
            score_frame, "Quality Score:",
            font=Fonts.BODY_S
        ).pack(side="left")

        self.score_bar = Theme.create_progressbar(score_frame)
        self.score_bar.pack(side="left", fill="x", expand=True, padx=(Sizes.PAD_S, 0))
        self.score_bar.set(0)

        self.score_label = Theme.create_label(
            score_frame, "0%",
            font=Fonts.BODY_S,
            width=50
        )
        self.score_label.pack(side="right")

        # Metrics
        self.metrics_frame = Theme.create_frame(self)
        self.metrics_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Warnings area (collapsible)
        self.warnings_frame = Theme.create_frame(self)
        self.warnings_label = Theme.create_label(
            self.warnings_frame, "",
            font=Fonts.BODY_XS,
            text_color=Colors.WARNING,
            wraplength=250
        )
        self.warnings_label.pack(anchor="w", padx=Sizes.PAD_M)

    def set_quality(
        self,
        score: float,
        status: str,
        metrics: Dict[str, str] = None,
        warnings: List[str] = None
    ):
        """
        Set the data quality information.

        Args:
            score: Quality score 0-100
            status: Status text (e.g., "Good", "Warning", "Poor")
            metrics: Additional metrics to display
            warnings: List of warning messages
        """
        # Update score
        self.score_bar.set(score / 100)
        self.score_label.configure(text=f"{score:.0f}%")

        # Update status with color
        if score >= 80:
            color = Colors.SUCCESS
        elif score >= 60:
            color = Colors.WARNING
        else:
            color = Colors.ERROR

        self.status_label.configure(text=status, text_color=color)

        # Update metrics
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        if metrics:
            for key, value in metrics.items():
                row = Theme.create_frame(self.metrics_frame)
                row.pack(fill="x", pady=1)

                Theme.create_label(row, key, font=Fonts.BODY_XS, text_color=Colors.TEXT_MUTED).pack(side="left")
                Theme.create_label(row, value, font=Fonts.BODY_XS).pack(side="right")

        # Update warnings
        if warnings:
            self.warnings_frame.pack(fill="x", pady=(0, Sizes.PAD_S))
            self.warnings_label.configure(text="\n".join(f"⚠ {w}" for w in warnings))
        else:
            self.warnings_frame.pack_forget()


class AnalysisProgressPanel(ctk.CTkFrame):
    """
    Progress panel for factor analysis operations.

    Shows:
    - Current stage
    - Progress bar
    - Stage-by-stage status
    - Cancel button
    """

    def __init__(
        self,
        parent,
        on_cancel: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.on_cancel = on_cancel
        self.stages: List[str] = []
        self.current_stage_idx = 0
        self.stage_labels: Dict[int, ctk.CTkLabel] = {}

        self._create_widgets()

    def _create_widgets(self):
        """Create progress widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(header, "Analysis Progress", size="s").pack(side="left")

        self.status_label = Theme.create_label(
            header, "Idle",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        self.status_label.pack(side="right")

        # Overall progress
        progress_frame = Theme.create_frame(self)
        progress_frame.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)

        self.progress_bar = Theme.create_progressbar(progress_frame)
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)

        self.progress_text = Theme.create_label(
            progress_frame, "0%",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        )
        self.progress_text.pack(anchor="e", pady=(Sizes.PAD_XS, 0))

        # Stage list
        self.stages_frame = Theme.create_frame(self)
        self.stages_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_S)

        # Cancel button
        self.cancel_btn = Theme.create_button(
            self, "Cancel",
            command=self._on_cancel,
            style="danger",
            width=100
        )
        self.cancel_btn.pack(pady=(0, Sizes.PAD_M))
        self.cancel_btn.configure(state="disabled")

    def set_stages(self, stages: List[str]):
        """Set the analysis stages."""
        self.stages = stages
        self.current_stage_idx = 0

        for widget in self.stages_frame.winfo_children():
            widget.destroy()
        self.stage_labels.clear()

        for i, stage in enumerate(stages):
            row = Theme.create_frame(self.stages_frame)
            row.pack(fill="x", pady=1)

            status = Theme.create_label(
                row, "○",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED,
                width=20
            )
            status.pack(side="left")

            label = Theme.create_label(
                row, stage,
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED
            )
            label.pack(side="left")

            self.stage_labels[i] = (status, label)

    def start(self):
        """Start the progress tracking."""
        self.cancel_btn.configure(state="normal")
        self.status_label.configure(text="Running...", text_color=Colors.PRIMARY_LIGHT)
        self.update_stage(0)

    def update_stage(self, stage_idx: int, progress: float = None):
        """
        Update to a specific stage.

        Args:
            stage_idx: Stage index
            progress: Optional progress within stage (0-1)
        """
        self.current_stage_idx = stage_idx

        # Update stage labels
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

        # Update overall progress
        total_stages = len(self.stages)
        if total_stages > 0:
            base_progress = stage_idx / total_stages
            if progress is not None:
                base_progress += progress / total_stages
            self.progress_bar.set(base_progress)
            self.progress_text.configure(text=f"{int(base_progress * 100)}%")

    def complete(self, success: bool = True):
        """Mark analysis as complete."""
        self.cancel_btn.configure(state="disabled")

        if success:
            self.progress_bar.set(1)
            self.progress_text.configure(text="100%")
            self.status_label.configure(text="Complete", text_color=Colors.SUCCESS)

            # Mark all stages complete
            for status, label in self.stage_labels.values():
                status.configure(text="✓", text_color=Colors.SUCCESS)
                label.configure(text_color=Colors.TEXT_PRIMARY)
        else:
            self.status_label.configure(text="Failed", text_color=Colors.ERROR)

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.on_cancel:
            self.on_cancel()


class ScenarioCard(ctk.CTkFrame):
    """
    Card displaying a detected scenario.

    Shows:
    - Scenario name/description
    - Detection criteria
    - Performance in scenario
    - Factor effects
    """

    def __init__(
        self,
        parent,
        scenario_name: str,
        scenario_data: Dict[str, Any],
        on_click: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            parent,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_L,
            border_width=1,
            border_color=Colors.BORDER,
            **kwargs
        )

        self.scenario_name = scenario_name
        self.scenario_data = scenario_data
        self.on_click = on_click

        self._create_widgets()

        # Bind click events
        self.bind("<Button-1>", self._on_card_click)
        self.bind("<Enter>", self._on_hover_enter)
        self.bind("<Leave>", self._on_hover_leave)

    def _create_widgets(self):
        """Create card widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, self.scenario_name, size="s").pack(side="left")

        # Performance badge
        perf = self.scenario_data.get('performance', 0)
        perf_color = Colors.SUCCESS if perf > 0 else Colors.ERROR if perf < 0 else Colors.TEXT_MUTED
        perf_label = Theme.create_label(
            header,
            f"{perf:+.1f}%",
            font=Fonts.LABEL_BOLD,
            text_color=perf_color
        )
        perf_label.pack(side="right")

        # Description
        desc = self.scenario_data.get('description', '')
        if desc:
            Theme.create_label(
                self, desc,
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_SECONDARY,
                wraplength=280
            ).pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        # Key metrics
        metrics_frame = Theme.create_frame(self)
        metrics_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Sample size
        sample_size = self.scenario_data.get('sample_size', 0)
        Theme.create_label(
            metrics_frame,
            f"Samples: {sample_size}",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        ).pack(side="left")

        # Confidence
        confidence = self.scenario_data.get('confidence', 0)
        conf_text = f"Conf: {confidence:.0f}%"
        Theme.create_label(
            metrics_frame,
            conf_text,
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        ).pack(side="right")

    def _on_card_click(self, event):
        """Handle card click."""
        if self.on_click:
            self.on_click(self.scenario_name, self.scenario_data)

    def _on_hover_enter(self, event):
        """Handle mouse enter."""
        self.configure(border_color=Colors.PRIMARY_LIGHT)

    def _on_hover_leave(self, event):
        """Handle mouse leave."""
        self.configure(border_color=Colors.BORDER)


class TierResultsPanel(ctk.CTkFrame):
    """
    Panel for displaying tier analysis results.

    Features:
    - Tier selection tabs
    - Results table
    - Significance indicators
    - Export options
    """

    def __init__(
        self,
        parent,
        tier: int = 1,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.tier = tier
        self.results: Dict[str, Any] = {}

        self._create_widgets()

    def _create_widgets(self):
        """Create panel widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        tier_titles = {
            1: "Tier 1: Exploratory Analysis",
            2: "Tier 2: Hypothesis Testing",
            3: "Tier 3: ML Analysis"
        }

        Theme.create_header(
            header,
            tier_titles.get(self.tier, f"Tier {self.tier}"),
            size="s"
        ).pack(side="left")

        self.export_btn = Theme.create_button(
            header,
            "Export",
            command=self._on_export,
            style="ghost",
            width=70,
            height=28
        )
        self.export_btn.pack(side="right")

        # Description
        tier_descriptions = {
            1: "Correlation analysis, distribution statistics, and initial factor screening",
            2: "Statistical hypothesis tests with significance levels and effect sizes",
            3: "Machine learning factor importance and SHAP values"
        }

        Theme.create_label(
            self,
            tier_descriptions.get(self.tier, ""),
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY,
            wraplength=500
        ).pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Results content area
        self.content_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent"
        )
        self.content_frame.pack(fill="both", expand=True, padx=Sizes.PAD_S, pady=(0, Sizes.PAD_S))

        # Placeholder text
        self.placeholder = Theme.create_label(
            self.content_frame,
            "No analysis results yet. Run analysis to see results.",
            text_color=Colors.TEXT_MUTED
        )
        self.placeholder.pack(pady=Sizes.PAD_XL)

    def set_results(self, results: Dict[str, Any]):
        """
        Set the tier results.

        Args:
            results: Results dictionary from the analysis
        """
        self.results = results
        self._refresh_display()

    def _refresh_display(self):
        """Refresh the results display."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        if not self.results:
            self.placeholder = Theme.create_label(
                self.content_frame,
                "No analysis results yet. Run analysis to see results.",
                text_color=Colors.TEXT_MUTED
            )
            self.placeholder.pack(pady=Sizes.PAD_XL)
            return

        # Display based on tier type
        if self.tier == 1:
            self._display_tier1_results()
        elif self.tier == 2:
            self._display_tier2_results()
        elif self.tier == 3:
            self._display_tier3_results()

    def _display_tier1_results(self):
        """Display Tier 1 exploratory results."""
        # Correlations section
        correlations = self.results.get('correlations', {})
        if correlations:
            section = self._create_section("Factor Correlations")
            for factor, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                self._add_result_row(section, factor, f"{corr:.4f}",
                                    Colors.SUCCESS if corr > 0 else Colors.ERROR)

        # Distribution stats
        distributions = self.results.get('distributions', {})
        if distributions:
            section = self._create_section("Distribution Statistics")
            for factor, stats in list(distributions.items())[:5]:
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                self._add_result_row(section, factor, f"μ={mean:.2f}, σ={std:.2f}")

    def _display_tier2_results(self):
        """Display Tier 2 hypothesis testing results."""
        tests = self.results.get('tests', {})
        if tests:
            for test_name, test_results in tests.items():
                section = self._create_section(test_name)

                p_value = test_results.get('p_value', 1)
                statistic = test_results.get('statistic', 0)
                effect_size = test_results.get('effect_size', 0)

                # Significance indicator
                sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""

                self._add_result_row(section, "P-Value", f"{p_value:.4f} {sig}")
                self._add_result_row(section, "Test Statistic", f"{statistic:.4f}")
                self._add_result_row(section, "Effect Size", f"{effect_size:.4f}")

    def _display_tier3_results(self):
        """Display Tier 3 ML results."""
        # Feature importance
        importance = self.results.get('feature_importance', {})
        if importance:
            section = self._create_section("Feature Importance (Random Forest)")
            for factor, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                self._add_result_row(section, factor, f"{imp:.4f}")

        # SHAP values
        shap = self.results.get('shap_values', {})
        if shap:
            section = self._create_section("SHAP Values")
            for factor, val in sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                color = Colors.SUCCESS if val > 0 else Colors.ERROR
                self._add_result_row(section, factor, f"{val:.4f}", color)

    def _create_section(self, title: str) -> ctk.CTkFrame:
        """Create a results section."""
        section = Theme.create_frame(self.content_frame)
        section.pack(fill="x", pady=(Sizes.PAD_S, Sizes.PAD_M))

        Theme.create_label(
            section, title,
            font=Fonts.LABEL_BOLD,
            text_color=Colors.PRIMARY_LIGHT
        ).pack(anchor="w")

        ctk.CTkFrame(section, fg_color=Colors.BORDER, height=1).pack(fill="x", pady=Sizes.PAD_XS)

        return section

    def _add_result_row(self, parent: ctk.CTkFrame, label: str, value: str, color: str = None):
        """Add a result row to a section."""
        row = Theme.create_frame(parent)
        row.pack(fill="x", pady=1)

        Theme.create_label(
            row, label,
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left")

        Theme.create_label(
            row, value,
            font=Fonts.BODY_S,
            text_color=color or Colors.TEXT_PRIMARY
        ).pack(side="right")

    def _on_export(self):
        """Handle export button click."""
        # TODO: Implement export functionality
        pass


class ProfileSelector(ctk.CTkFrame):
    """
    Configuration profile selector widget.

    Features:
    - Profile dropdown
    - Create/Edit/Delete buttons
    - Profile description
    """

    def __init__(
        self,
        parent,
        profiles: List[str] = None,
        on_profile_change: Optional[Callable[[str], None]] = None,
        on_create: Optional[Callable] = None,
        on_edit: Optional[Callable[[str], None]] = None,
        on_delete: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.profiles = profiles or []
        self.on_profile_change = on_profile_change
        self.on_create = on_create
        self.on_edit = on_edit
        self.on_delete = on_delete

        self._create_widgets()

    def _create_widgets(self):
        """Create selector widgets."""
        # Label and dropdown
        label_frame = Theme.create_frame(self)
        label_frame.pack(fill="x")

        Theme.create_label(label_frame, "Configuration Profile:").pack(side="left")

        self.profile_var = ctk.StringVar(value=self.profiles[0] if self.profiles else "")
        self.profile_combo = Theme.create_combobox(
            label_frame,
            values=self.profiles,
            variable=self.profile_var,
            command=self._on_profile_select,
            width=200
        )
        self.profile_combo.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Buttons
        btn_frame = Theme.create_frame(self)
        btn_frame.pack(fill="x", pady=(Sizes.PAD_S, 0))

        Theme.create_button(
            btn_frame, "New",
            command=self._on_create,
            style="secondary",
            width=70,
            height=28
        ).pack(side="left", padx=(0, Sizes.PAD_XS))

        Theme.create_button(
            btn_frame, "Edit",
            command=self._on_edit,
            style="secondary",
            width=70,
            height=28
        ).pack(side="left", padx=(0, Sizes.PAD_XS))

        Theme.create_button(
            btn_frame, "Delete",
            command=self._on_delete,
            style="ghost",
            width=70,
            height=28
        ).pack(side="left")

    def _on_profile_select(self, value: str):
        """Handle profile selection."""
        if self.on_profile_change:
            self.on_profile_change(value)

    def _on_create(self):
        """Handle create button."""
        if self.on_create:
            self.on_create()

    def _on_edit(self):
        """Handle edit button."""
        if self.on_edit:
            self.on_edit(self.profile_var.get())

    def _on_delete(self):
        """Handle delete button."""
        if self.on_delete:
            self.on_delete(self.profile_var.get())

    def update_profiles(self, profiles: List[str], selected: str = None):
        """Update the available profiles."""
        self.profiles = profiles
        self.profile_combo.configure(values=profiles)
        if selected:
            self.profile_var.set(selected)
        elif profiles:
            self.profile_var.set(profiles[0])


class ConfigSection(ctk.CTkFrame):
    """
    Collapsible configuration section.

    Features:
    - Expand/collapse
    - Section title
    - Status indicator
    """

    def __init__(
        self,
        parent,
        title: str,
        expanded: bool = True,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.title = title
        self.expanded = expanded
        self.content_frame: Optional[ctk.CTkFrame] = None

        self._create_widgets()

    def _create_widgets(self):
        """Create section widgets."""
        # Header (clickable)
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)
        self.header.bind("<Button-1>", self._toggle)

        self.expand_label = Theme.create_label(
            self.header,
            "▼" if self.expanded else "▶",
            font=Fonts.BODY_S,
            width=20
        )
        self.expand_label.pack(side="left")
        self.expand_label.bind("<Button-1>", self._toggle)

        self.title_label = Theme.create_header(self.header, self.title, size="s")
        self.title_label.pack(side="left")
        self.title_label.bind("<Button-1>", self._toggle)

        self.status_label = Theme.create_label(
            self.header, "",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        )
        self.status_label.pack(side="right")

        # Content area
        self.content_frame = Theme.create_frame(self)
        if self.expanded:
            self.content_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

    def _toggle(self, event=None):
        """Toggle expand/collapse."""
        self.expanded = not self.expanded
        self.expand_label.configure(text="▼" if self.expanded else "▶")

        if self.expanded:
            self.content_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))
        else:
            self.content_frame.pack_forget()

    def set_status(self, text: str, color: str = None):
        """Set the status indicator text."""
        self.status_label.configure(text=text, text_color=color or Colors.TEXT_MUTED)

    def get_content_frame(self) -> ctk.CTkFrame:
        """Get the content frame for adding widgets."""
        return self.content_frame
