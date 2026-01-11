"""
Univariate Parameter Optimization GUI.

A modern CustomTkinter interface for univariate parameter optimization.

Flow:
1. Select securities (single or portfolio)
2. Choose run mode (together or separate) for portfolio
3. Select strategy
4. Set control values for all parameters
5. Select which parameters to optimize with range (slider + interval)
6. Select performance metrics to examine
7. Run optimization
8. Export results to Excel

Uses sliders with interval input boxes for parameter range selection.
"""

import logging
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import customtkinter as ctk

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes, ask_yes_no, show_error, show_info, ProgressPanel
from Classes.GUI.ctk_components import SecuritySelector
from Classes.Data.data_loader import DataLoader
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Optimization.univariate_optimizer import (
    UnivariateOptimizer, ParameterConfig, AVAILABLE_METRICS, METRIC_DEFINITIONS
)
from Classes.Optimization.univariate_report import UnivariateReportGenerator

# Import available strategies
from strategies.alphatrend_strategy import AlphaTrendStrategy
from strategies.random_base_strategy import RandomBaseStrategy

# Import centralized strategy configuration
from config.strategy_config import StrategyConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_strategy_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Load strategy parameters from centralized configuration.

    Returns dict in format: {strategy_name: {param_name: {type, default, min, max, step}}}
    """
    result = {}
    for strategy_name in StrategyConfig.get_strategies():
        opt_params = StrategyConfig.get_optimization_params(strategy_name)
        if opt_params:
            result[strategy_name] = opt_params
    return result


# Load strategy parameters from centralized config
STRATEGY_PARAMETERS = get_strategy_parameters()

STRATEGIES = {
    'AlphaTrendStrategy': AlphaTrendStrategy,
    'RandomBaseStrategy': RandomBaseStrategy
}


class ParameterRangeWidget(ctk.CTkFrame):
    """Widget for configuring a parameter's optimization range with slider and interval."""

    def __init__(
        self,
        parent,
        param_name: str,
        param_config: Dict[str, Any],
        on_change: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.param_name = param_name
        self.param_config = param_config
        self.on_change = on_change

        self.param_type = param_config.get('type', 'float')
        self.default = param_config.get('default', 0)
        self.min_val = param_config.get('min', 0)
        self.max_val = param_config.get('max', 100)
        self.step = param_config.get('step', 1)

        self._create_widgets()

    def _create_widgets(self):
        # Top row: Checkbox + Parameter name
        top_frame = Theme.create_frame(self)
        top_frame.pack(fill="x", pady=(Sizes.PAD_XS, 0))

        self.enabled_var = ctk.BooleanVar(value=False)
        self.enabled_cb = Theme.create_checkbox(
            top_frame, self.param_name,
            variable=self.enabled_var,
            command=self._on_enabled_change
        )
        self.enabled_cb.pack(side="left")

        # Default value hint
        Theme.create_hint(
            top_frame, f"(default: {self.default})"
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Control value entry (always visible)
        control_frame = Theme.create_frame(top_frame)
        control_frame.pack(side="right")
        Theme.create_label(control_frame, "Control:").pack(side="left", padx=(0, Sizes.PAD_XS))
        self.control_var = ctk.StringVar(value=str(self.default))
        self.control_entry = Theme.create_entry(control_frame, width=80)
        self.control_entry.configure(textvariable=self.control_var)
        self.control_entry.pack(side="left")

        # Range configuration frame (shown when enabled)
        self.range_frame = Theme.create_frame(self)

        # Range row
        range_row = Theme.create_frame(self.range_frame)
        range_row.pack(fill="x", pady=Sizes.PAD_XS)

        # Min value
        Theme.create_label(range_row, "Min:").pack(side="left", padx=(Sizes.PAD_L, 0))
        self.min_var = ctk.StringVar(value=str(self.min_val))
        self.min_entry = Theme.create_entry(range_row, width=70)
        self.min_entry.configure(textvariable=self.min_var)
        self.min_entry.pack(side="left", padx=(Sizes.PAD_XS, 0))

        # Max value
        Theme.create_label(range_row, "Max:").pack(side="left", padx=(Sizes.PAD_M, 0))
        self.max_var = ctk.StringVar(value=str(self.max_val))
        self.max_entry = Theme.create_entry(range_row, width=70)
        self.max_entry.configure(textvariable=self.max_var)
        self.max_entry.pack(side="left", padx=(Sizes.PAD_XS, 0))

        # Interval
        Theme.create_label(range_row, "Interval:").pack(side="left", padx=(Sizes.PAD_M, 0))
        self.interval_var = ctk.StringVar(value=str(self.step))
        self.interval_entry = Theme.create_entry(range_row, width=70)
        self.interval_entry.configure(textvariable=self.interval_var)
        self.interval_entry.pack(side="left", padx=(Sizes.PAD_XS, 0))

        # Values count label
        self.count_label = Theme.create_hint(range_row, "")
        self.count_label.pack(side="left", padx=(Sizes.PAD_M, 0))

        # Slider row
        slider_row = Theme.create_frame(self.range_frame)
        slider_row.pack(fill="x", pady=Sizes.PAD_XS, padx=(Sizes.PAD_L, 0))

        Theme.create_label(slider_row, "Range preview:").pack(side="left")

        self.slider = Theme.create_slider(
            slider_row,
            from_=self.min_val,
            to=self.max_val,
            command=self._on_slider_change
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=Sizes.PAD_S)
        self.slider.set(self.default)

        self.slider_value_label = Theme.create_label(slider_row, f"{self.default}")
        self.slider_value_label.pack(side="left", padx=(0, Sizes.PAD_S))

        # Bind entry changes
        self.min_var.trace_add('write', self._update_count)
        self.max_var.trace_add('write', self._update_count)
        self.interval_var.trace_add('write', self._update_count)

        self._update_count()

    def _on_enabled_change(self):
        if self.enabled_var.get():
            self.range_frame.pack(fill="x", pady=(Sizes.PAD_XS, Sizes.PAD_S))
        else:
            self.range_frame.pack_forget()

        if self.on_change:
            self.on_change()

    def _on_slider_change(self, value):
        if self.param_type == 'int':
            val = int(round(float(value)))
        else:
            val = round(float(value), 2)
        self.slider_value_label.configure(text=str(val))

    def _update_count(self, *args):
        try:
            min_val = float(self.min_var.get())
            max_val = float(self.max_var.get())
            interval = float(self.interval_var.get())

            if interval <= 0 or min_val > max_val:
                self.count_label.configure(text="Invalid range")
                return

            count = int((max_val - min_val) / interval) + 1
            self.count_label.configure(text=f"({count} values)")

            # Update slider range
            self.slider.configure(from_=min_val, to=max_val)

        except (ValueError, ZeroDivisionError):
            self.count_label.configure(text="")

    def is_enabled(self) -> bool:
        return self.enabled_var.get()

    def get_control_value(self) -> Any:
        try:
            val = self.control_var.get()
            if self.param_type == 'int':
                return int(val)
            return float(val)
        except ValueError:
            return self.default

    def get_parameter_config(self) -> Optional[ParameterConfig]:
        if not self.is_enabled():
            return None

        try:
            min_val = float(self.min_var.get())
            max_val = float(self.max_var.get())
            interval = float(self.interval_var.get())

            return ParameterConfig.from_range(
                name=self.param_name,
                min_val=min_val,
                max_val=max_val,
                interval=interval,
                param_type=self.param_type
            )
        except ValueError:
            return None


class CTkUnivariateOptimizationGUI(ctk.CTk):
    """Main GUI for univariate parameter optimization."""

    def __init__(self):
        super().__init__()

        # Apply theme
        Theme.setup()

        # Window configuration
        self.title("Univariate Parameter Optimization")
        self.geometry("1200x900")
        self.minsize(1000, 700)
        self.configure(fg_color=Colors.BG_DARK)

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/daily'))
        self.available_securities = self.data_loader.get_available_symbols()

        # State
        self.parameter_widgets: Dict[str, ParameterRangeWidget] = {}
        self.metric_vars: Dict[str, ctk.BooleanVar] = {}
        self.optimization_thread = None
        self.optimizer = None
        self.is_running = False

        # Build UI
        self._create_ui()

    def _create_ui(self):
        """Create the main UI layout."""
        # Main container
        main_frame = Theme.create_frame(self)
        main_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Left panel - Configuration
        left_panel = Theme.create_frame(main_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, Sizes.PAD_S))

        # Right panel - Results
        right_panel = Theme.create_frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(Sizes.PAD_S, 0))

        self._create_left_panel(left_panel)
        self._create_right_panel(right_panel)

    def _create_left_panel(self, parent):
        """Create configuration panel."""
        # Title
        Theme.create_header(parent, "Configuration", size="l").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Scrollable content
        scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True)

        # Section 1: Securities
        self._create_securities_section(scroll_frame)

        # Section 2: Strategy Selection
        self._create_strategy_section(scroll_frame)

        # Section 3: Parameters
        self._create_parameters_section(scroll_frame)

        # Section 4: Metrics
        self._create_metrics_section(scroll_frame)

        # Section 5: Settings
        self._create_settings_section(scroll_frame)

        # Run button
        btn_frame = Theme.create_frame(parent)
        btn_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        self.run_button = Theme.create_button(
            btn_frame, "Run Optimization",
            command=self._start_optimization,
            style="success",
            width=200,
            height=40
        )
        self.run_button.pack(side="right")

        self.cancel_button = Theme.create_button(
            btn_frame, "Cancel",
            command=self._cancel_optimization,
            style="danger",
            width=100,
            height=40
        )
        self.cancel_button.pack(side="right", padx=(0, Sizes.PAD_S))
        self.cancel_button.configure(state="disabled")

    def _create_securities_section(self, parent):
        """Create securities selection section."""
        card = Theme.create_card(parent)
        card.pack(fill="x", pady=(0, Sizes.PAD_M))

        content = Theme.create_frame(card)
        content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(content, "Securities", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Mode selection
        mode_frame = Theme.create_frame(content)
        mode_frame.pack(fill="x", pady=Sizes.PAD_S)

        self.mode_var = ctk.StringVar(value="single")

        Theme.create_radiobutton(
            mode_frame, "Single Security",
            variable=self.mode_var, value="single",
            command=self._on_mode_change
        ).pack(side="left", padx=(0, Sizes.PAD_L))

        Theme.create_radiobutton(
            mode_frame, "Portfolio",
            variable=self.mode_var, value="portfolio",
            command=self._on_mode_change
        ).pack(side="left")

        # Run mode for portfolio (hidden initially)
        self.run_mode_frame = Theme.create_frame(content)

        Theme.create_label(self.run_mode_frame, "Run Mode:").pack(side="left")
        self.run_mode_var = ctk.StringVar(value="together")

        Theme.create_radiobutton(
            self.run_mode_frame, "Together",
            variable=self.run_mode_var, value="together"
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        Theme.create_radiobutton(
            self.run_mode_frame, "Separate",
            variable=self.run_mode_var, value="separate"
        ).pack(side="left", padx=(Sizes.PAD_M, 0))

        Theme.create_hint(
            self.run_mode_frame,
            "(Together: combined metrics, Separate: averaged)"
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Securities selector
        self.security_selector = SecuritySelector(
            content,
            available_securities=sorted(self.available_securities),
            title="",
            multi_select=False,
            show_search=True,
            show_buttons=True,
            height=150,
            on_selection_change=self._on_security_change
        )
        self.security_selector.pack(fill="x", pady=(Sizes.PAD_S, 0))

    def _create_strategy_section(self, parent):
        """Create strategy selection section."""
        card = Theme.create_card(parent)
        card.pack(fill="x", pady=(0, Sizes.PAD_M))

        content = Theme.create_frame(card)
        content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(content, "Strategy", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        strategy_frame = Theme.create_frame(content)
        strategy_frame.pack(fill="x")

        Theme.create_label(strategy_frame, "Strategy:").pack(side="left")

        self.strategy_var = ctk.StringVar(value=list(STRATEGIES.keys())[0])
        self.strategy_combo = Theme.create_combobox(
            strategy_frame,
            values=list(STRATEGIES.keys()),
            variable=self.strategy_var,
            command=self._on_strategy_change,
            width=250
        )
        self.strategy_combo.pack(side="left", padx=(Sizes.PAD_S, 0))

    def _create_parameters_section(self, parent):
        """Create parameters configuration section."""
        card = Theme.create_card(parent)
        card.pack(fill="x", pady=(0, Sizes.PAD_M))

        content = Theme.create_frame(card)
        content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        header_frame = Theme.create_frame(content)
        header_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_header(header_frame, "Parameters", size="s").pack(side="left")

        # Select/Deselect buttons
        btn_frame = Theme.create_frame(header_frame)
        btn_frame.pack(side="right")

        Theme.create_button(
            btn_frame, "Select All",
            command=self._select_all_params,
            style="secondary", width=90, height=28
        ).pack(side="left", padx=2)

        Theme.create_button(
            btn_frame, "Deselect All",
            command=self._deselect_all_params,
            style="secondary", width=90, height=28
        ).pack(side="left", padx=2)

        # Parameters count
        self.params_count_var = ctk.StringVar(value="0 parameters selected for optimization")
        Theme.create_label(
            content, "",
            textvariable=self.params_count_var,
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Hint
        Theme.create_hint(
            content,
            "Check parameters to optimize. Set 'Control' value (baseline) and range for optimization."
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Parameters container (scrollable within card)
        self.params_container = ctk.CTkScrollableFrame(
            content,
            fg_color="transparent",
            height=300
        )
        self.params_container.pack(fill="x")

        # Build parameters for initial strategy
        self._build_parameter_widgets()

    def _create_metrics_section(self, parent):
        """Create metrics selection section."""
        card = Theme.create_card(parent)
        card.pack(fill="x", pady=(0, Sizes.PAD_M))

        content = Theme.create_frame(card)
        content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        header_frame = Theme.create_frame(content)
        header_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_header(header_frame, "Performance Metrics", size="s").pack(side="left")

        # Quick select buttons
        btn_frame = Theme.create_frame(header_frame)
        btn_frame.pack(side="right")

        Theme.create_button(
            btn_frame, "All",
            command=lambda: self._set_all_metrics(True),
            style="secondary", width=50, height=28
        ).pack(side="left", padx=2)

        Theme.create_button(
            btn_frame, "None",
            command=lambda: self._set_all_metrics(False),
            style="secondary", width=50, height=28
        ).pack(side="left", padx=2)

        Theme.create_button(
            btn_frame, "Default",
            command=self._set_default_metrics,
            style="secondary", width=60, height=28
        ).pack(side="left", padx=2)

        # Metrics checkboxes in grid
        metrics_frame = Theme.create_frame(content)
        metrics_frame.pack(fill="x")

        row = 0
        col = 0
        max_cols = 3

        for metric in AVAILABLE_METRICS:
            metric_def = METRIC_DEFINITIONS.get(metric, {})
            display_name = metric_def.get('name', metric)

            var = ctk.BooleanVar(value=metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'])
            self.metric_vars[metric] = var

            cb = Theme.create_checkbox(metrics_frame, display_name, variable=var)
            cb.grid(row=row, column=col, sticky="w", padx=Sizes.PAD_S, pady=Sizes.PAD_XS)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _create_settings_section(self, parent):
        """Create optimization settings section."""
        card = Theme.create_card(parent)
        card.pack(fill="x")

        content = Theme.create_frame(card)
        content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(content, "Settings", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        settings_frame = Theme.create_frame(content)
        settings_frame.pack(fill="x")

        # Initial capital
        cap_frame = Theme.create_frame(settings_frame)
        cap_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(cap_frame, "Initial Capital:").pack(side="left")
        self.capital_var = ctk.StringVar(value="100000")
        Theme.create_entry(cap_frame, width=120).configure(textvariable=self.capital_var)
        self.capital_var.trace_add('write', lambda *a: None)
        cap_entry = Theme.create_entry(cap_frame, width=120)
        cap_entry.configure(textvariable=self.capital_var)
        cap_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Commission
        comm_frame = Theme.create_frame(settings_frame)
        comm_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(comm_frame, "Commission (%):").pack(side="left")
        self.commission_var = ctk.StringVar(value="0.1")
        comm_entry = Theme.create_entry(comm_frame, width=80)
        comm_entry.configure(textvariable=self.commission_var)
        comm_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Slippage
        slip_frame = Theme.create_frame(settings_frame)
        slip_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(slip_frame, "Slippage (%):").pack(side="left")
        self.slippage_var = ctk.StringVar(value="0.1")
        slip_entry = Theme.create_entry(slip_frame, width=80)
        slip_entry.configure(textvariable=self.slippage_var)
        slip_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Date range section
        Theme.create_label(
            content, "Date Range (optional):",
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(Sizes.PAD_M, Sizes.PAD_XS))

        date_frame = Theme.create_frame(content)
        date_frame.pack(fill="x")

        # Start date
        start_date_frame = Theme.create_frame(date_frame)
        start_date_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(start_date_frame, "Start Date:").pack(side="left")
        self.start_date_var = ctk.StringVar(value="")
        start_date_entry = Theme.create_entry(start_date_frame, width=120)
        start_date_entry.configure(textvariable=self.start_date_var)
        start_date_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_label(
            start_date_frame, "(YYYY-MM-DD)",
            text_color=Colors.TEXT_MUTED
        ).pack(side="left", padx=(Sizes.PAD_XS, 0))

        # End date
        end_date_frame = Theme.create_frame(date_frame)
        end_date_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(end_date_frame, "End Date:").pack(side="left")
        self.end_date_var = ctk.StringVar(value="")
        end_date_entry = Theme.create_entry(end_date_frame, width=120)
        end_date_entry.configure(textvariable=self.end_date_var)
        end_date_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_label(
            end_date_frame, "(YYYY-MM-DD)",
            text_color=Colors.TEXT_MUTED
        ).pack(side="left", padx=(Sizes.PAD_XS, 0))

    def _create_right_panel(self, parent):
        """Create results panel."""
        Theme.create_header(parent, "Results", size="l").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Progress panel
        progress_card = Theme.create_card(parent)
        progress_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        progress_content = Theme.create_frame(progress_card)
        progress_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        self.progress_panel = ProgressPanel(progress_content, show_log=True, log_height=400)
        self.progress_panel.pack(fill="both", expand=True)

        # Export button
        export_frame = Theme.create_frame(parent)
        export_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        self.export_button = Theme.create_button(
            export_frame, "Export to Excel",
            command=self._export_results,
            style="primary",
            width=150
        )
        self.export_button.pack(side="right")
        self.export_button.configure(state="disabled")

        # Results info
        self.results_label = Theme.create_label(
            export_frame, "",
            text_color=Colors.TEXT_SECONDARY
        )
        self.results_label.pack(side="left")

    def _on_mode_change(self):
        """Handle mode change between single and portfolio."""
        mode = self.mode_var.get()
        if mode == "portfolio":
            self.security_selector.multi_select = True
            self.security_selector.update_securities(sorted(self.available_securities))
            self.run_mode_frame.pack(fill="x", pady=Sizes.PAD_S, after=self.security_selector.master.winfo_children()[1])
        else:
            self.security_selector.multi_select = False
            selected = self.security_selector.get_selected()
            self.security_selector.update_securities(sorted(self.available_securities))
            if selected:
                self.security_selector.set_selected([selected[0]])
            self.run_mode_frame.pack_forget()

    def _on_security_change(self, selected: List[str]):
        """Handle security selection change."""
        pass

    def _on_strategy_change(self, strategy_name: str):
        """Handle strategy selection change."""
        self._build_parameter_widgets()

    def _build_parameter_widgets(self):
        """Build parameter widgets for current strategy."""
        # Clear existing widgets
        for widget in self.params_container.winfo_children():
            widget.destroy()
        self.parameter_widgets.clear()

        strategy_name = self.strategy_var.get()
        params = STRATEGY_PARAMETERS.get(strategy_name, {})

        if not params:
            Theme.create_label(
                self.params_container,
                "No configurable parameters for this strategy."
            ).pack(anchor="w")
            return

        for param_name, param_config in params.items():
            widget = ParameterRangeWidget(
                self.params_container,
                param_name=param_name,
                param_config=param_config,
                on_change=self._update_param_count
            )
            widget.pack(fill="x", pady=Sizes.PAD_XS)
            self.parameter_widgets[param_name] = widget

        self._update_param_count()

    def _update_param_count(self):
        """Update parameter selection count."""
        enabled = sum(1 for w in self.parameter_widgets.values() if w.is_enabled())
        total = len(self.parameter_widgets)
        self.params_count_var.set(f"{enabled}/{total} parameters selected for optimization")

    def _select_all_params(self):
        """Select all parameters for optimization."""
        for widget in self.parameter_widgets.values():
            widget.enabled_var.set(True)
            widget._on_enabled_change()
        self._update_param_count()

    def _deselect_all_params(self):
        """Deselect all parameters."""
        for widget in self.parameter_widgets.values():
            widget.enabled_var.set(False)
            widget._on_enabled_change()
        self._update_param_count()

    def _set_all_metrics(self, value: bool):
        """Set all metrics to value."""
        for var in self.metric_vars.values():
            var.set(value)

    def _set_default_metrics(self):
        """Set default metrics."""
        defaults = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        for metric, var in self.metric_vars.items():
            var.set(metric in defaults)

    def _validate_config(self) -> bool:
        """Validate configuration before running."""
        # Check securities
        securities = self.security_selector.get_selected()
        if not securities:
            show_error(self, "Validation Error", "Please select at least one security.")
            return False

        mode = self.mode_var.get()
        if mode == "portfolio" and len(securities) < 2:
            show_error(self, "Validation Error", "Portfolio mode requires at least 2 securities.")
            return False

        # Check parameters
        params_to_optimize = []
        for widget in self.parameter_widgets.values():
            if widget.is_enabled():
                config = widget.get_parameter_config()
                if config:
                    params_to_optimize.append(config)

        if not params_to_optimize:
            show_error(self, "Validation Error", "Please select at least one parameter to optimize.")
            return False

        # Check metrics
        selected_metrics = [m for m, v in self.metric_vars.items() if v.get()]
        if not selected_metrics:
            show_error(self, "Validation Error", "Please select at least one metric.")
            return False

        return True

    def _start_optimization(self):
        """Start the optimization process."""
        if not self._validate_config():
            return

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.export_button.configure(state="disabled")
        self.progress_panel.clear()

        # Gather configuration
        securities = self.security_selector.get_selected()
        strategy_name = self.strategy_var.get()
        strategy_class = STRATEGIES[strategy_name]
        run_mode = self.run_mode_var.get() if self.mode_var.get() == "portfolio" else "together"

        # Get control values and parameters to optimize
        control_values = {}
        params_to_optimize = []

        for param_name, widget in self.parameter_widgets.items():
            control_values[param_name] = widget.get_control_value()
            if widget.is_enabled():
                config = widget.get_parameter_config()
                if config:
                    params_to_optimize.append(config)

        # Get selected metrics
        metrics = [m for m, v in self.metric_vars.items() if v.get()]

        # Get settings
        try:
            initial_capital = float(self.capital_var.get())
            commission = float(self.commission_var.get()) / 100
            slippage = float(self.slippage_var.get()) / 100
        except ValueError:
            show_error(self, "Error", "Invalid settings values.")
            self._reset_ui()
            return

        # Get date range
        start_date = None
        end_date = None
        start_date_str = self.start_date_var.get().strip()
        end_date_str = self.end_date_var.get().strip()

        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            except ValueError:
                show_error(self, "Error", "Invalid start date format. Use YYYY-MM-DD.")
                self._reset_ui()
                return

        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            except ValueError:
                show_error(self, "Error", "Invalid end date format. Use YYYY-MM-DD.")
                self._reset_ui()
                return

        if start_date and end_date and start_date >= end_date:
            show_error(self, "Error", "Start date must be before end date.")
            self._reset_ui()
            return

        # Log configuration
        self.progress_panel.log(f"Strategy: {strategy_name}")
        self.progress_panel.log(f"Securities: {', '.join(securities)}")
        self.progress_panel.log(f"Run Mode: {run_mode}")
        self.progress_panel.log(f"Parameters to optimize: {len(params_to_optimize)}")
        self.progress_panel.log(f"Metrics: {', '.join(metrics)}")
        if start_date or end_date:
            date_range_str = f"{start_date.strftime('%Y-%m-%d') if start_date else 'Start'} to {end_date.strftime('%Y-%m-%d') if end_date else 'End'}"
            self.progress_panel.log(f"Date Range: {date_range_str}")
        self.progress_panel.log("")

        # Create optimizer
        self.optimizer = UnivariateOptimizer(
            data_loader=self.data_loader,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )

        # Store result for export
        self.optimization_result = None

        # Run in thread
        self.optimization_thread = threading.Thread(
            target=self._run_optimization,
            args=(strategy_class, securities, params_to_optimize,
                  control_values, metrics, run_mode, start_date, end_date),
            daemon=True
        )
        self.optimization_thread.start()

    def _run_optimization(
        self,
        strategy_class,
        securities,
        params_to_optimize,
        control_values,
        metrics,
        run_mode,
        start_date,
        end_date
    ):
        """Run optimization in background thread."""
        try:
            def progress_callback(stage, current, total):
                self.after(0, lambda: self._update_progress(stage, current, total))

            result = self.optimizer.optimize(
                strategy_class=strategy_class,
                securities=securities,
                parameters_to_optimize=params_to_optimize,
                control_values=control_values,
                metrics=metrics,
                run_mode=run_mode,
                start_date=start_date,
                end_date=end_date,
                progress_callback=progress_callback
            )

            self.optimization_result = result
            self.after(0, lambda: self._on_optimization_complete(result))

        except Exception as e:
            logger.exception("Optimization failed")
            self.after(0, lambda: self._on_optimization_error(str(e)))

    def _update_progress(self, stage: str, current: int, total: int):
        """Update progress display."""
        progress = current / total if total > 0 else 0
        self.progress_panel.update_progress(progress, stage)
        self.progress_panel.log(f"[{current}/{total}] {stage}")

    def _on_optimization_complete(self, result):
        """Handle optimization completion."""
        self.progress_panel.set_complete("Optimization Complete")

        # Display summary
        self.progress_panel.log("\n" + "=" * 50)
        self.progress_panel.log("OPTIMIZATION COMPLETE")
        self.progress_panel.log("=" * 50)

        for param_name, param_result in result.parameter_results.items():
            self.progress_panel.log(f"\nParameter: {param_name}")
            self.progress_panel.log(f"  Control value: {param_result.control_value}")
            self.progress_panel.log(f"  Values tested: {len(param_result.values_tested)}")

            # Show best values for first few metrics
            for metric in result.metrics_calculated[:3]:
                metric_def = METRIC_DEFINITIONS.get(metric, {})
                higher_is_better = metric_def.get("higher_is_better", True)
                best = param_result.get_best_value(metric, higher_is_better)
                self.progress_panel.log(f"  Best for {metric}: {best}")

        total_runs = sum(len(p.results) for p in result.parameter_results.values())
        self.results_label.configure(text=f"Completed {total_runs} backtests")

        self._reset_ui()
        self.export_button.configure(state="normal")

    def _on_optimization_error(self, error: str):
        """Handle optimization error."""
        self.progress_panel.log(f"\nERROR: {error}", level="error")
        show_error(self, "Optimization Error", f"Optimization failed: {error}")
        self._reset_ui()

    def _cancel_optimization(self):
        """Cancel running optimization."""
        if self.optimizer:
            self.optimizer.cancel()
            self.progress_panel.log("\nCancellation requested...")

    def _reset_ui(self):
        """Reset UI after optimization."""
        self.is_running = False
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    def _export_results(self):
        """Export results to Excel."""
        if not self.optimization_result:
            show_error(self, "Error", "No results to export.")
            return

        try:
            report_gen = UnivariateReportGenerator()
            filepath = report_gen.generate_report(self.optimization_result)
            show_info(self, "Export Complete", f"Report saved to:\n{filepath}")
            self.progress_panel.log(f"\nReport exported to: {filepath}")
        except Exception as e:
            logger.exception("Export failed")
            show_error(self, "Export Error", f"Failed to export: {e}")


def main():
    """Main entry point."""
    app = CTkUnivariateOptimizationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
