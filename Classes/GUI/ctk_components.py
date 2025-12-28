"""
Reusable CustomTkinter Components.

This module provides shared widget components that can be reused across
all GUIs in the BackTestingFramework.

Components:
- SecuritySelector: Multi-select security/ticker list
- StrategySelector: Strategy selection with parameter configuration
- DatePeriodSelector: Date range selection
- ParameterEditor: Dynamic parameter editing widget
- BasketSelector: Portfolio basket selection
"""

import customtkinter as ctk
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, date
from pathlib import Path
import json

from .ctk_theme import Theme, Colors, Fonts, Sizes


class SecuritySelector(ctk.CTkFrame):
    """
    Multi-select security/ticker selector widget.

    Features:
    - Scrollable list with checkboxes
    - Search/filter functionality
    - Select All / Clear All buttons
    - Load from preset or file
    - Selection count display
    """

    def __init__(
        self,
        parent,
        available_securities: List[str],
        title: str = "Securities",
        multi_select: bool = True,
        show_search: bool = True,
        show_buttons: bool = True,
        height: int = 300,
        on_selection_change: Optional[Callable[[List[str]], None]] = None,
        **kwargs
    ):
        """
        Initialize the security selector.

        Args:
            parent: Parent widget
            available_securities: List of available security symbols
            title: Section title
            multi_select: Allow multiple selections
            show_search: Show search/filter box
            show_buttons: Show Select All/Clear All buttons
            height: Height of the scrollable list
            on_selection_change: Callback when selection changes
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.available_securities = available_securities
        self.multi_select = multi_select
        self.on_selection_change = on_selection_change

        # Selection tracking
        self.selection_vars: Dict[str, ctk.BooleanVar] = {}
        self.checkbox_widgets: Dict[str, ctk.CTkCheckBox] = {}

        # Radio button var for single-select mode
        self.single_select_var = ctk.StringVar(value="")

        self._create_widgets(title, show_search, show_buttons, height)

    def _create_widgets(self, title: str, show_search: bool, show_buttons: bool, height: int):
        """Create the selector widgets."""
        # Title with selection count
        title_frame = Theme.create_frame(self)
        title_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_header(title_frame, title, size="s").pack(side="left")

        self.count_label = Theme.create_label(
            title_frame, "(0 selected)",
            font=Fonts.HINT,
            text_color=Colors.TEXT_MUTED
        )
        self.count_label.pack(side="right")

        # Search box
        if show_search:
            search_frame = Theme.create_frame(self)
            search_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

            self.search_var = ctk.StringVar()
            self.search_var.trace_add("write", self._on_search_change)

            self.search_entry = Theme.create_entry(
                search_frame,
                placeholder="Search securities..."
            )
            self.search_entry.pack(fill="x")
            self.search_entry.configure(textvariable=self.search_var)

        # Action buttons
        if show_buttons and self.multi_select:
            button_frame = Theme.create_frame(self)
            button_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

            Theme.create_button(
                button_frame, "Select All",
                command=self.select_all,
                style="secondary",
                width=90,
                height=28
            ).pack(side="left", padx=(0, Sizes.PAD_S))

            Theme.create_button(
                button_frame, "Clear All",
                command=self.clear_all,
                style="secondary",
                width=90,
                height=28
            ).pack(side="left")

        # Scrollable security list
        self.list_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_M,
            height=height
        )
        self.list_frame.pack(fill="both", expand=True)

        # Create checkboxes/radiobuttons for each security
        self._populate_list()

    def _populate_list(self, filter_text: str = ""):
        """Populate the security list."""
        # Clear existing widgets
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        filter_text = filter_text.lower()

        for symbol in self.available_securities:
            if filter_text and filter_text not in symbol.lower():
                continue

            if self.multi_select:
                # Initialize variable if not exists
                if symbol not in self.selection_vars:
                    self.selection_vars[symbol] = ctk.BooleanVar(value=False)

                cb = Theme.create_checkbox(
                    self.list_frame,
                    symbol,
                    variable=self.selection_vars[symbol],
                    command=self._on_selection_change
                )
                cb.pack(fill="x", padx=Sizes.PAD_S, pady=Sizes.PAD_XS)
                self.checkbox_widgets[symbol] = cb
            else:
                rb = Theme.create_radiobutton(
                    self.list_frame,
                    symbol,
                    variable=self.single_select_var,
                    value=symbol,
                    command=self._on_selection_change
                )
                rb.pack(fill="x", padx=Sizes.PAD_S, pady=Sizes.PAD_XS)

    def _on_search_change(self, *args):
        """Handle search text change."""
        self._populate_list(self.search_var.get())

    def _on_selection_change(self):
        """Handle selection change."""
        selected = self.get_selected()
        self.count_label.configure(text=f"({len(selected)} selected)")

        if self.on_selection_change:
            self.on_selection_change(selected)

    def get_selected(self) -> List[str]:
        """Get list of selected securities."""
        if self.multi_select:
            return [s for s, var in self.selection_vars.items() if var.get()]
        else:
            val = self.single_select_var.get()
            return [val] if val else []

    def set_selected(self, securities: List[str]):
        """Set the selected securities."""
        if self.multi_select:
            for symbol, var in self.selection_vars.items():
                var.set(symbol in securities)
        else:
            if securities:
                self.single_select_var.set(securities[0])
            else:
                self.single_select_var.set("")

        self._on_selection_change()

    def select_all(self):
        """Select all visible securities."""
        if self.multi_select:
            for var in self.selection_vars.values():
                var.set(True)
            self._on_selection_change()

    def clear_all(self):
        """Clear all selections."""
        if self.multi_select:
            for var in self.selection_vars.values():
                var.set(False)
        else:
            self.single_select_var.set("")
        self._on_selection_change()

    def update_securities(self, securities: List[str]):
        """Update the available securities list."""
        self.available_securities = securities
        self.selection_vars.clear()
        self.checkbox_widgets.clear()
        self._populate_list()


class StrategySelector(ctk.CTkFrame):
    """
    Strategy selection widget with parameter configuration.

    Features:
    - Strategy dropdown selection
    - Dynamic parameter display based on selected strategy
    - Parameter validation
    """

    def __init__(
        self,
        parent,
        strategies: Dict[str, Dict[str, Any]],
        title: str = "Strategy",
        on_strategy_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs
    ):
        """
        Initialize the strategy selector.

        Args:
            parent: Parent widget
            strategies: Dict of strategy name -> parameter definitions
                       Format: {"strategy_name": {"param": {"default": val, "type": type, "min": min, "max": max}, ...}}
            title: Section title
            on_strategy_change: Callback when strategy or params change
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.strategies = strategies
        self.on_strategy_change = on_strategy_change
        self.param_vars: Dict[str, ctk.Variable] = {}
        self.param_widgets: List[Any] = []

        self._create_widgets(title)

    def _create_widgets(self, title: str):
        """Create the strategy selector widgets."""
        # Title
        Theme.create_header(self, title, size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Strategy dropdown
        strategy_frame = Theme.create_frame(self)
        strategy_frame.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_label(strategy_frame, "Strategy:").pack(side="left", padx=(0, Sizes.PAD_S))

        self.strategy_var = ctk.StringVar()
        strategy_names = list(self.strategies.keys())

        self.strategy_combo = Theme.create_combobox(
            strategy_frame,
            values=strategy_names,
            variable=self.strategy_var,
            command=self._on_strategy_selected,
            width=200
        )
        self.strategy_combo.pack(side="left", fill="x", expand=True)

        if strategy_names:
            self.strategy_var.set(strategy_names[0])

        # Parameters section
        self.params_frame = Theme.create_frame(self)
        self.params_frame.pack(fill="x")

        # Load initial parameters
        self._on_strategy_selected(self.strategy_var.get())

    def _on_strategy_selected(self, strategy_name: str):
        """Handle strategy selection."""
        # Clear existing parameter widgets
        for widget in self.param_widgets:
            widget.destroy()
        self.param_widgets.clear()
        self.param_vars.clear()

        if not strategy_name or strategy_name not in self.strategies:
            return

        params = self.strategies[strategy_name]

        # Create parameter widgets
        for param_name, param_def in params.items():
            frame = Theme.create_frame(self.params_frame)
            frame.pack(fill="x", pady=Sizes.PAD_XS)
            self.param_widgets.append(frame)

            # Label
            label = Theme.create_label(frame, f"{param_name}:")
            label.pack(side="left", padx=(0, Sizes.PAD_S))

            # Value widget based on type
            param_type = param_def.get("type", "float")
            default_val = param_def.get("default", 0)

            if param_type in ("int", "float"):
                var = ctk.StringVar(value=str(default_val))
                entry = Theme.create_entry(frame, width=100)
                entry.configure(textvariable=var)
                entry.pack(side="left")

                # Add hint for range if available
                min_val = param_def.get("min")
                max_val = param_def.get("max")
                if min_val is not None and max_val is not None:
                    hint = Theme.create_hint(frame, f"[{min_val} - {max_val}]")
                    hint.pack(side="left", padx=(Sizes.PAD_S, 0))

                self.param_vars[param_name] = var

            elif param_type == "bool":
                var = ctk.BooleanVar(value=default_val)
                cb = Theme.create_checkbox(frame, "", variable=var)
                cb.pack(side="left")
                self.param_vars[param_name] = var

            elif param_type == "choice":
                var = ctk.StringVar(value=str(default_val))
                choices = param_def.get("choices", [])
                combo = Theme.create_combobox(frame, values=choices, variable=var, width=150)
                combo.pack(side="left")
                self.param_vars[param_name] = var

        self._notify_change()

    def _notify_change(self):
        """Notify of strategy/param change."""
        if self.on_strategy_change:
            strategy = self.get_selected_strategy()
            params = self.get_parameters()
            self.on_strategy_change(strategy, params)

    def get_selected_strategy(self) -> str:
        """Get the selected strategy name."""
        return self.strategy_var.get()

    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameter values."""
        result = {}
        strategy = self.strategy_var.get()

        if not strategy or strategy not in self.strategies:
            return result

        for param_name, param_def in self.strategies[strategy].items():
            if param_name not in self.param_vars:
                continue

            var = self.param_vars[param_name]
            param_type = param_def.get("type", "float")

            try:
                if param_type == "int":
                    result[param_name] = int(var.get())
                elif param_type == "float":
                    result[param_name] = float(var.get())
                elif param_type == "bool":
                    result[param_name] = var.get()
                else:
                    result[param_name] = var.get()
            except (ValueError, TypeError):
                result[param_name] = param_def.get("default", 0)

        return result

    def set_strategy(self, strategy_name: str, parameters: Dict[str, Any] = None):
        """Set the strategy and optionally its parameters."""
        if strategy_name in self.strategies:
            self.strategy_var.set(strategy_name)
            self._on_strategy_selected(strategy_name)

            if parameters:
                for param_name, value in parameters.items():
                    if param_name in self.param_vars:
                        self.param_vars[param_name].set(value)


class DatePeriodSelector(ctk.CTkFrame):
    """
    Date range selection widget.

    Features:
    - All available data option
    - Custom date range with from/to entries
    - Last N periods option
    - Date format validation
    """

    def __init__(
        self,
        parent,
        title: str = "Date Range",
        show_all_option: bool = True,
        show_last_n_option: bool = True,
        date_format: str = "%Y-%m-%d",
        on_change: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the date period selector.

        Args:
            parent: Parent widget
            title: Section title
            show_all_option: Show "All available data" option
            show_last_n_option: Show "Last N days" option
            date_format: Expected date format
            on_change: Callback when selection changes
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.date_format = date_format
        self.on_change = on_change

        self._create_widgets(title, show_all_option, show_last_n_option)

    def _create_widgets(self, title: str, show_all_option: bool, show_last_n_option: bool):
        """Create the date selector widgets."""
        # Title
        Theme.create_header(self, title, size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Mode selection
        self.mode_var = ctk.StringVar(value="all" if show_all_option else "custom")

        # All available data option
        if show_all_option:
            rb_all = Theme.create_radiobutton(
                self, "All available data",
                variable=self.mode_var,
                value="all",
                command=self._on_mode_change
            )
            rb_all.pack(anchor="w", pady=Sizes.PAD_XS)

        # Custom date range
        rb_custom = Theme.create_radiobutton(
            self, "Custom date range",
            variable=self.mode_var,
            value="custom",
            command=self._on_mode_change
        )
        rb_custom.pack(anchor="w", pady=Sizes.PAD_XS)

        # Custom date inputs frame
        self.custom_frame = Theme.create_frame(self)
        self.custom_frame.pack(fill="x", padx=(Sizes.PAD_XL, 0), pady=(Sizes.PAD_XS, 0))

        # From date
        from_frame = Theme.create_frame(self.custom_frame)
        from_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(from_frame, "From:", width=50).pack(side="left")
        self.from_var = ctk.StringVar()
        self.from_entry = Theme.create_entry(from_frame, placeholder="YYYY-MM-DD", width=150)
        self.from_entry.configure(textvariable=self.from_var)
        self.from_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # To date
        to_frame = Theme.create_frame(self.custom_frame)
        to_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(to_frame, "To:", width=50).pack(side="left")
        self.to_var = ctk.StringVar()
        self.to_entry = Theme.create_entry(to_frame, placeholder="YYYY-MM-DD", width=150)
        self.to_entry.configure(textvariable=self.to_var)
        self.to_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Last N days option
        if show_last_n_option:
            rb_last = Theme.create_radiobutton(
                self, "Last N days",
                variable=self.mode_var,
                value="last_n",
                command=self._on_mode_change
            )
            rb_last.pack(anchor="w", pady=Sizes.PAD_XS)

            self.last_n_frame = Theme.create_frame(self)
            self.last_n_frame.pack(fill="x", padx=(Sizes.PAD_XL, 0), pady=(Sizes.PAD_XS, 0))

            Theme.create_label(self.last_n_frame, "Days:").pack(side="left")
            self.last_n_var = ctk.StringVar(value="365")
            self.last_n_entry = Theme.create_entry(self.last_n_frame, width=80)
            self.last_n_entry.configure(textvariable=self.last_n_var)
            self.last_n_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Initial state
        self._on_mode_change()

    def _on_mode_change(self):
        """Handle mode change."""
        mode = self.mode_var.get()

        # Enable/disable custom frame
        state = "normal" if mode == "custom" else "disabled"
        self.from_entry.configure(state=state)
        self.to_entry.configure(state=state)

        # Enable/disable last N frame if exists
        if hasattr(self, 'last_n_entry'):
            state = "normal" if mode == "last_n" else "disabled"
            self.last_n_entry.configure(state=state)

        if self.on_change:
            self.on_change()

    def get_config(self) -> Dict[str, Any]:
        """Get the current date configuration."""
        mode = self.mode_var.get()

        if mode == "all":
            return {"mode": "all"}
        elif mode == "custom":
            return {
                "mode": "custom",
                "from_date": self.from_var.get(),
                "to_date": self.to_var.get()
            }
        elif mode == "last_n":
            try:
                days = int(self.last_n_var.get())
            except ValueError:
                days = 365
            return {
                "mode": "last_n",
                "days": days
            }

        return {"mode": "all"}

    def validate(self) -> Tuple[bool, str]:
        """
        Validate the date selection.

        Returns:
            Tuple of (is_valid, error_message)
        """
        mode = self.mode_var.get()

        if mode == "custom":
            from_date = self.from_var.get().strip()
            to_date = self.to_var.get().strip()

            if not from_date or not to_date:
                return False, "Please enter both From and To dates."

            try:
                from_dt = datetime.strptime(from_date, self.date_format)
                to_dt = datetime.strptime(to_date, self.date_format)

                if from_dt >= to_dt:
                    return False, "From date must be before To date."

            except ValueError:
                return False, f"Invalid date format. Use {self.date_format}"

        elif mode == "last_n":
            try:
                days = int(self.last_n_var.get())
                if days <= 0:
                    return False, "Number of days must be positive."
            except ValueError:
                return False, "Invalid number of days."

        return True, ""

    def set_config(self, config: Dict[str, Any]):
        """Set the date configuration."""
        mode = config.get("mode", "all")
        self.mode_var.set(mode)

        if mode == "custom":
            self.from_var.set(config.get("from_date", ""))
            self.to_var.set(config.get("to_date", ""))
        elif mode == "last_n":
            self.last_n_var.set(str(config.get("days", 365)))

        self._on_mode_change()


class CapitalCostsPanel(ctk.CTkFrame):
    """
    Capital and trading costs configuration panel.

    Features:
    - Initial capital input
    - Commission type (fixed/percentage)
    - Commission value
    - Slippage percentage
    """

    def __init__(
        self,
        parent,
        title: str = "Capital & Costs",
        default_capital: float = 100000,
        default_commission_type: str = "fixed",
        default_commission: float = 0,
        default_slippage: float = 0,
        **kwargs
    ):
        """
        Initialize the capital/costs panel.

        Args:
            parent: Parent widget
            title: Section title
            default_capital: Default initial capital
            default_commission_type: "fixed" or "percentage"
            default_commission: Default commission value
            default_slippage: Default slippage percentage
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.default_capital = default_capital
        self.default_commission_type = default_commission_type
        self.default_commission = default_commission
        self.default_slippage = default_slippage

        self._create_widgets(title)

    def _create_widgets(self, title: str):
        """Create the panel widgets."""
        # Title
        Theme.create_header(self, title, size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Initial Capital
        capital_frame = Theme.create_frame(self)
        capital_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(capital_frame, "Initial Capital ($):").pack(side="left")
        self.capital_var = ctk.StringVar(value=str(self.default_capital))
        self.capital_entry = Theme.create_entry(capital_frame, width=150)
        self.capital_entry.configure(textvariable=self.capital_var)
        self.capital_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Commission Type
        comm_type_frame = Theme.create_frame(self)
        comm_type_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(comm_type_frame, "Commission Type:").pack(side="left")
        self.comm_type_var = ctk.StringVar(value=self.default_commission_type)
        self.comm_type_combo = Theme.create_combobox(
            comm_type_frame,
            values=["fixed", "percentage"],
            variable=self.comm_type_var,
            width=120
        )
        self.comm_type_combo.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Commission Value
        comm_val_frame = Theme.create_frame(self)
        comm_val_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(comm_val_frame, "Commission Value:").pack(side="left")
        self.commission_var = ctk.StringVar(value=str(self.default_commission))
        self.commission_entry = Theme.create_entry(comm_val_frame, width=100)
        self.commission_entry.configure(textvariable=self.commission_var)
        self.commission_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(comm_val_frame, "($ per trade or %)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Slippage
        slip_frame = Theme.create_frame(self)
        slip_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(slip_frame, "Slippage (%):").pack(side="left")
        self.slippage_var = ctk.StringVar(value=str(self.default_slippage))
        self.slippage_entry = Theme.create_entry(slip_frame, width=100)
        self.slippage_entry.configure(textvariable=self.slippage_var)
        self.slippage_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

    def get_config(self) -> Dict[str, Any]:
        """Get the capital and costs configuration."""
        try:
            capital = float(self.capital_var.get())
        except ValueError:
            capital = self.default_capital

        try:
            commission = float(self.commission_var.get())
        except ValueError:
            commission = self.default_commission

        try:
            slippage = float(self.slippage_var.get())
        except ValueError:
            slippage = self.default_slippage

        return {
            "initial_capital": capital,
            "commission_type": self.comm_type_var.get(),
            "commission": commission,
            "slippage": slippage
        }

    def validate(self) -> Tuple[bool, str]:
        """Validate the configuration."""
        try:
            capital = float(self.capital_var.get())
            if capital <= 0:
                return False, "Initial capital must be positive."
        except ValueError:
            return False, "Invalid initial capital value."

        try:
            commission = float(self.commission_var.get())
            if commission < 0:
                return False, "Commission cannot be negative."
        except ValueError:
            return False, "Invalid commission value."

        try:
            slippage = float(self.slippage_var.get())
            if slippage < 0:
                return False, "Slippage cannot be negative."
        except ValueError:
            return False, "Invalid slippage value."

        return True, ""


class WalkForwardConfigPanel(ctk.CTkFrame):
    """
    Walk-forward optimization configuration panel.

    Features:
    - Training window size
    - Validation window size
    - Step size configuration
    - Window visualization
    """

    def __init__(
        self,
        parent,
        title: str = "Walk-Forward Configuration",
        default_training_days: int = 252,
        default_validation_days: int = 63,
        default_step_days: int = 21,
        **kwargs
    ):
        """
        Initialize the walk-forward config panel.

        Args:
            parent: Parent widget
            title: Section title
            default_training_days: Default training window size
            default_validation_days: Default validation window size
            default_step_days: Default step size
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._create_widgets(
            title,
            default_training_days,
            default_validation_days,
            default_step_days
        )

    def _create_widgets(
        self,
        title: str,
        default_training: int,
        default_validation: int,
        default_step: int
    ):
        """Create the panel widgets."""
        # Title
        Theme.create_header(self, title, size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Training window
        train_frame = Theme.create_frame(self)
        train_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(train_frame, "Training Window (days):").pack(side="left")
        self.training_var = ctk.StringVar(value=str(default_training))
        train_entry = Theme.create_entry(train_frame, width=100)
        train_entry.configure(textvariable=self.training_var)
        train_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(train_frame, "(252 = 1 year)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Validation window
        val_frame = Theme.create_frame(self)
        val_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(val_frame, "Validation Window (days):").pack(side="left")
        self.validation_var = ctk.StringVar(value=str(default_validation))
        val_entry = Theme.create_entry(val_frame, width=100)
        val_entry.configure(textvariable=self.validation_var)
        val_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(val_frame, "(63 = 1 quarter)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Step size
        step_frame = Theme.create_frame(self)
        step_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(step_frame, "Step Size (days):").pack(side="left")
        self.step_var = ctk.StringVar(value=str(default_step))
        step_entry = Theme.create_entry(step_frame, width=100)
        step_entry.configure(textvariable=self.step_var)
        step_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(step_frame, "(21 = 1 month)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Description
        desc_frame = Theme.create_card(self)
        desc_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        desc_text = (
            "Walk-forward optimization trains on historical data and validates on "
            "subsequent out-of-sample data, stepping forward through time to avoid "
            "overfitting and ensure robust parameter selection."
        )
        Theme.create_label(
            desc_frame,
            desc_text,
            wraplength=400,
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(padx=Sizes.PAD_M, pady=Sizes.PAD_M)

    def get_config(self) -> Dict[str, Any]:
        """Get the walk-forward configuration."""
        try:
            training = int(self.training_var.get())
        except ValueError:
            training = 252

        try:
            validation = int(self.validation_var.get())
        except ValueError:
            validation = 63

        try:
            step = int(self.step_var.get())
        except ValueError:
            step = 21

        return {
            "training_days": training,
            "validation_days": validation,
            "step_days": step
        }

    def validate(self) -> Tuple[bool, str]:
        """Validate the configuration."""
        try:
            training = int(self.training_var.get())
            if training <= 0:
                return False, "Training window must be positive."
        except ValueError:
            return False, "Invalid training window value."

        try:
            validation = int(self.validation_var.get())
            if validation <= 0:
                return False, "Validation window must be positive."
        except ValueError:
            return False, "Invalid validation window value."

        try:
            step = int(self.step_var.get())
            if step <= 0:
                return False, "Step size must be positive."
            if step > validation:
                return False, "Step size should not exceed validation window."
        except ValueError:
            return False, "Invalid step size value."

        return True, ""


class ExecutionOptionsPanel(ctk.CTkFrame):
    """
    Execution options configuration panel for optimization.

    Features:
    - Speed mode selection
    - CPU cores selection
    - Additional analysis options
    """

    def __init__(
        self,
        parent,
        title: str = "Execution Options",
        **kwargs
    ):
        """Initialize the execution options panel."""
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._create_widgets(title)

    def _create_widgets(self, title: str):
        """Create the panel widgets."""
        # Title
        Theme.create_header(self, title, size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Speed mode
        speed_frame = Theme.create_frame(self)
        speed_frame.pack(fill="x", pady=Sizes.PAD_XS)

        Theme.create_label(speed_frame, "Speed Mode:").pack(side="left")
        self.speed_var = ctk.StringVar(value="normal")
        speed_combo = Theme.create_combobox(
            speed_frame,
            values=["fast", "normal", "thorough"],
            variable=self.speed_var,
            width=120
        )
        speed_combo.pack(side="left", padx=(Sizes.PAD_S, 0))

        # CPU cores
        cores_frame = Theme.create_frame(self)
        cores_frame.pack(fill="x", pady=Sizes.PAD_XS)

        import os
        max_cores = os.cpu_count() or 4

        Theme.create_label(cores_frame, "CPU Cores:").pack(side="left")
        self.cores_var = ctk.StringVar(value=str(max(1, max_cores - 1)))
        cores_entry = Theme.create_entry(cores_frame, width=80)
        cores_entry.configure(textvariable=self.cores_var)
        cores_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(cores_frame, f"(Max: {max_cores})").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Additional options
        options_frame = Theme.create_frame(self)
        options_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        self.sensitivity_var = ctk.BooleanVar(value=False)
        Theme.create_checkbox(
            options_frame,
            "Run sensitivity analysis",
            variable=self.sensitivity_var
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        self.export_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            options_frame,
            "Export results to file",
            variable=self.export_var
        ).pack(anchor="w", pady=Sizes.PAD_XS)

    def get_config(self) -> Dict[str, Any]:
        """Get the execution options configuration."""
        try:
            cores = int(self.cores_var.get())
        except ValueError:
            cores = 1

        return {
            "speed_mode": self.speed_var.get(),
            "cpu_cores": cores,
            "run_sensitivity": self.sensitivity_var.get(),
            "export_results": self.export_var.get()
        }

    def validate(self) -> Tuple[bool, str]:
        """Validate the configuration."""
        try:
            cores = int(self.cores_var.get())
            import os
            max_cores = os.cpu_count() or 4
            if cores < 1 or cores > max_cores:
                return False, f"CPU cores must be between 1 and {max_cores}."
        except ValueError:
            return False, "Invalid CPU cores value."

        return True, ""
