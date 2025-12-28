"""
CustomTkinter Basket Manager Dialog for creating and managing security baskets.
"""
import customtkinter as ctk
from typing import List, Optional, Callable

from .ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info, ask_yes_no
from ..Config.basket import Basket, BasketManager
from ..Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig,
    EnhancedVulnerabilityConfig, FeatureWeightConfig,
    ENHANCED_FEATURE_DEFINITIONS, FEATURE_PARAMETER_DEFINITIONS, VULNERABILITY_CORE_PARAM_DEFINITIONS
)


class CTkBasketManagerDialog:
    """
    Dialog for creating, editing, and deleting security baskets using CustomTkinter.
    """

    def __init__(self, parent, available_securities: List[str],
                 on_basket_selected: Optional[Callable[[Optional[Basket]], None]] = None):
        """
        Initialize the basket manager dialog.

        Args:
            parent: Parent window
            available_securities: List of available security symbols
            on_basket_selected: Callback when a basket is selected/created
        """
        self.parent = parent
        self.available_securities = sorted(available_securities)
        self.on_basket_selected = on_basket_selected
        self.basket_manager = BasketManager()
        self.selected_basket: Optional[Basket] = None

        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Basket Manager")
        self.dialog.geometry("850x650")
        self.dialog.transient(parent)
        self.dialog.configure(fg_color=Colors.BG_DARK)

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 850) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 650) // 2
        self.dialog.geometry(f"+{x}+{y}")

        self.dialog.grab_set()

        self._create_widgets()
        self._load_baskets()

    def _create_widgets(self):
        """Create dialog widgets."""
        # Main container
        main_frame = Theme.create_frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)

        # Left Panel: Basket List
        left_card = Theme.create_card(main_frame)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_S))

        left_content = Theme.create_frame(left_card)
        left_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(left_content, "Available Baskets", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Basket list (scrollable)
        self.basket_scroll = ctk.CTkScrollableFrame(left_content, fg_color=Colors.SURFACE, height=400)
        self.basket_scroll.pack(fill="both", expand=True)

        self.basket_buttons: List[ctk.CTkButton] = []

        # Action buttons
        btn_frame = Theme.create_frame(left_content)
        btn_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        Theme.create_button(btn_frame, "New", command=self._new_basket, style="primary", width=70).pack(side="left", padx=2)
        Theme.create_button(btn_frame, "Delete", command=self._delete_basket, style="danger", width=70).pack(side="left", padx=2)
        Theme.create_button(btn_frame, "Refresh", command=self._load_baskets, style="secondary", width=70).pack(side="left", padx=2)

        # Right Panel: Basket Details/Editor
        right_card = Theme.create_card(main_frame)
        right_card.grid(row=0, column=1, sticky="nsew")

        right_content = Theme.create_frame(right_card)
        right_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Basket Details
        details_frame = Theme.create_frame(right_content)
        details_frame.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(details_frame, "Basket Details", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Name
        name_frame = Theme.create_frame(details_frame)
        name_frame.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(name_frame, "Name:", width=100).pack(side="left")
        self.name_var = ctk.StringVar()
        self.name_entry = Theme.create_entry(name_frame, width=200)
        self.name_entry.configure(textvariable=self.name_var)
        self.name_entry.pack(side="left", fill="x", expand=True)

        # Description
        desc_frame = Theme.create_frame(details_frame)
        desc_frame.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(desc_frame, "Description:", width=100).pack(side="left")
        self.desc_var = ctk.StringVar()
        desc_entry = Theme.create_entry(desc_frame, width=200)
        desc_entry.configure(textvariable=self.desc_var)
        desc_entry.pack(side="left", fill="x", expand=True)

        # Securities Selection
        securities_frame = Theme.create_frame(right_content)
        securities_frame.pack(fill="both", expand=True, pady=(0, Sizes.PAD_M))

        Theme.create_header(securities_frame, "Securities in Basket", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Two columns: available and selected
        columns_frame = Theme.create_frame(securities_frame)
        columns_frame.pack(fill="both", expand=True)
        columns_frame.grid_columnconfigure(0, weight=1)
        columns_frame.grid_columnconfigure(2, weight=1)
        columns_frame.grid_rowconfigure(1, weight=1)

        # Available securities
        Theme.create_label(columns_frame, "Available:").grid(row=0, column=0, sticky="w")
        self.available_scroll = ctk.CTkScrollableFrame(columns_frame, fg_color=Colors.SURFACE, height=180)
        self.available_scroll.grid(row=1, column=0, sticky="nsew")

        self.available_checkboxes: dict[str, ctk.CTkCheckBox] = {}
        self.available_vars: dict[str, ctk.BooleanVar] = {}

        for symbol in self.available_securities:
            var = ctk.BooleanVar(value=False)
            self.available_vars[symbol] = var
            cb = Theme.create_checkbox(self.available_scroll, symbol, variable=var)
            cb.pack(anchor="w", pady=1)
            self.available_checkboxes[symbol] = cb

        # Add/Remove buttons
        btn_col = Theme.create_frame(columns_frame)
        btn_col.grid(row=1, column=1, padx=Sizes.PAD_S)
        Theme.create_button(btn_col, ">>", command=self._add_to_basket, width=40, height=30).pack(pady=5)
        Theme.create_button(btn_col, "<<", command=self._remove_from_basket, width=40, height=30).pack(pady=5)

        # Selected securities
        Theme.create_label(columns_frame, "In Basket:").grid(row=0, column=2, sticky="w")
        self.selected_scroll = ctk.CTkScrollableFrame(columns_frame, fg_color=Colors.SURFACE, height=180)
        self.selected_scroll.grid(row=1, column=2, sticky="nsew")

        self.selected_checkboxes: dict[str, ctk.CTkCheckBox] = {}
        self.selected_vars: dict[str, ctk.BooleanVar] = {}

        # Capital Contention
        contention_frame = Theme.create_frame(right_content)
        contention_frame.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_header(contention_frame, "Default Capital Contention", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.contention_mode_var = ctk.StringVar(value="default")
        Theme.create_radiobutton(
            contention_frame, "Default (Ignore signals)",
            variable=self.contention_mode_var, value="default"
        ).pack(anchor="w")
        Theme.create_radiobutton(
            contention_frame, "Vulnerability Score",
            variable=self.contention_mode_var, value="vulnerability"
        ).pack(anchor="w")

        # Action buttons
        action_frame = Theme.create_frame(right_content)
        action_frame.pack(fill="x")

        Theme.create_button(action_frame, "Save Basket", command=self._save_basket, style="success", width=120).pack(side="left", padx=2)
        Theme.create_button(action_frame, "Use Selected", command=self._use_basket, style="primary", width=120).pack(side="left", padx=2)
        Theme.create_button(action_frame, "Cancel", command=self.dialog.destroy, style="secondary", width=80).pack(side="right", padx=2)

    def _load_baskets(self):
        """Load available baskets into list."""
        # Clear existing buttons
        for btn in self.basket_buttons:
            btn.destroy()
        self.basket_buttons.clear()

        baskets = self.basket_manager.list_baskets()
        for name in baskets:
            btn = ctk.CTkButton(
                self.basket_scroll,
                text=name,
                command=lambda n=name: self._on_basket_select(n),
                fg_color="transparent",
                hover_color=Colors.SURFACE_HOVER,
                anchor="w",
                height=32
            )
            btn.pack(fill="x", pady=1)
            self.basket_buttons.append(btn)

    def _on_basket_select(self, basket_name: str):
        """Handle basket selection."""
        basket = self.basket_manager.load(basket_name)
        if basket:
            self._populate_basket_details(basket)

    def _populate_basket_details(self, basket: Basket):
        """Populate the details panel with basket info."""
        self.name_var.set(basket.name)
        self.desc_var.set(basket.description)

        # Clear selected
        for widget in self.selected_scroll.winfo_children():
            widget.destroy()
        self.selected_checkboxes.clear()
        self.selected_vars.clear()

        # Add basket securities to selected
        for symbol in basket.securities:
            var = ctk.BooleanVar(value=False)
            self.selected_vars[symbol] = var
            cb = Theme.create_checkbox(self.selected_scroll, symbol, variable=var)
            cb.pack(anchor="w", pady=1)
            self.selected_checkboxes[symbol] = cb

        # Set capital contention mode
        if basket.default_capital_contention:
            if basket.default_capital_contention.mode == CapitalContentionMode.VULNERABILITY_SCORE:
                self.contention_mode_var.set("vulnerability")
            else:
                self.contention_mode_var.set("default")
        else:
            self.contention_mode_var.set("default")

    def _add_to_basket(self):
        """Add selected securities to basket."""
        current_in_basket = set(self.selected_vars.keys())

        for symbol, var in self.available_vars.items():
            if var.get() and symbol not in current_in_basket:
                # Add to selected
                sel_var = ctk.BooleanVar(value=False)
                self.selected_vars[symbol] = sel_var
                cb = Theme.create_checkbox(self.selected_scroll, symbol, variable=sel_var)
                cb.pack(anchor="w", pady=1)
                self.selected_checkboxes[symbol] = cb

        # Uncheck all available
        for var in self.available_vars.values():
            var.set(False)

    def _remove_from_basket(self):
        """Remove selected securities from basket."""
        to_remove = [symbol for symbol, var in self.selected_vars.items() if var.get()]

        for symbol in to_remove:
            if symbol in self.selected_checkboxes:
                self.selected_checkboxes[symbol].destroy()
                del self.selected_checkboxes[symbol]
                del self.selected_vars[symbol]

    def _new_basket(self):
        """Clear form for new basket."""
        self.name_var.set("")
        self.desc_var.set("")
        self.contention_mode_var.set("default")

        # Clear selected
        for widget in self.selected_scroll.winfo_children():
            widget.destroy()
        self.selected_checkboxes.clear()
        self.selected_vars.clear()

        self.name_entry.focus_set()

    def _delete_basket(self):
        """Delete selected basket."""
        name = self.name_var.get().strip()
        if not name:
            show_error(self.dialog, "No Selection", "Please select a basket to delete.")
            return

        if ask_yes_no(self.dialog, "Confirm Delete", f"Delete basket '{name}'?"):
            if self.basket_manager.delete(name):
                self._load_baskets()
                self._new_basket()
                show_info(self.dialog, "Success", f"Basket '{name}' deleted.")
            else:
                show_error(self.dialog, "Error", f"Failed to delete basket '{name}'.")

    def _save_basket(self):
        """Save current basket configuration."""
        name = self.name_var.get().strip()
        if not name:
            show_error(self.dialog, "Error", "Basket name is required.")
            return

        securities = list(self.selected_vars.keys())
        if not securities:
            show_error(self.dialog, "Error", "Basket must contain at least one security.")
            return

        description = self.desc_var.get().strip()

        contention_mode = self.contention_mode_var.get()
        if contention_mode == "vulnerability":
            capital_contention = CapitalContentionConfig.vulnerability_score_mode()
        else:
            capital_contention = CapitalContentionConfig.default_mode()

        try:
            basket = self.basket_manager.create(
                name=name,
                securities=securities,
                description=description,
                capital_contention=capital_contention
            )
            self._load_baskets()
            show_info(self.dialog, "Success", f"Basket '{name}' saved with {len(securities)} securities.")
        except Exception as e:
            show_error(self.dialog, "Error", f"Failed to save basket: {e}")

    def _use_basket(self):
        """Use the currently displayed basket."""
        securities = list(self.selected_vars.keys())
        if not securities:
            show_error(self.dialog, "No Securities", "No securities in basket.")
            return

        name = self.name_var.get().strip() or "Untitled"
        description = self.desc_var.get().strip()

        contention_mode = self.contention_mode_var.get()
        if contention_mode == "vulnerability":
            capital_contention = CapitalContentionConfig.vulnerability_score_mode()
        else:
            capital_contention = CapitalContentionConfig.default_mode()

        basket = Basket(
            name=name,
            securities=securities,
            description=description,
            default_capital_contention=capital_contention
        )

        self.selected_basket = basket

        if self.on_basket_selected:
            self.on_basket_selected(basket)

        self.dialog.destroy()

    def get_selected_basket(self) -> Optional[Basket]:
        """Get the selected/created basket."""
        return self.selected_basket


class CTkVulnerabilityScoreConfigDialog:
    """
    Dialog for configuring vulnerability score parameters using CustomTkinter.
    """

    def __init__(self, parent, current_config=None,
                 on_save: Optional[Callable] = None):
        """
        Initialize the vulnerability score config dialog.

        Args:
            parent: Parent window
            current_config: Current VulnerabilityScoreConfig
            on_save: Callback when configuration is saved
        """
        self.parent = parent
        self.on_save = on_save
        self.current_config = current_config if current_config else VulnerabilityScoreConfig()

        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Vulnerability Score Configuration")
        self.dialog.geometry("550x500")
        self.dialog.transient(parent)
        self.dialog.configure(fg_color=Colors.BG_DARK)

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 550) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 500) // 2
        self.dialog.geometry(f"+{x}+{y}")

        self.dialog.grab_set()

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = Theme.create_frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Title
        Theme.create_header(main_frame, "Vulnerability Score Parameters", size="l").pack(pady=(0, Sizes.PAD_L))

        # Parameters card
        params_card = Theme.create_card(main_frame)
        params_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        params_content = Theme.create_frame(params_card)
        params_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Immunity Days
        imm_def = VULNERABILITY_CORE_PARAM_DEFINITIONS['immunity_days']
        imm_frame = Theme.create_frame(params_content)
        imm_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(imm_frame, "Immunity Days:", width=150).pack(side="left")
        self.immunity_var = ctk.StringVar(value=str(self.current_config.immunity_days))
        Theme.create_entry(imm_frame, width=80).configure(textvariable=self.immunity_var)
        imm_entry = Theme.create_entry(imm_frame, width=80)
        imm_entry.configure(textvariable=self.immunity_var)
        imm_entry.pack(side="left")
        Theme.create_hint(imm_frame, f"[{imm_def['min']}-{imm_def['max']}]").pack(side="left", padx=Sizes.PAD_S)

        # Min Profit Threshold
        profit_frame = Theme.create_frame(params_content)
        profit_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(profit_frame, "Min Profit Threshold:", width=150).pack(side="left")
        self.profit_var = ctk.StringVar(value=str(self.current_config.min_profit_threshold))
        profit_entry = Theme.create_entry(profit_frame, width=80)
        profit_entry.configure(textvariable=self.profit_var)
        profit_entry.pack(side="left")
        Theme.create_hint(profit_frame, "[0.0-0.20] (0.02=2%)").pack(side="left", padx=Sizes.PAD_S)

        # Decay Rate Fast
        fast_def = FEATURE_PARAMETER_DEFINITIONS['fast_decay_rate']
        fast_frame = Theme.create_frame(params_content)
        fast_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(fast_frame, "Decay Rate (Fast):", width=150).pack(side="left")
        self.decay_fast_var = ctk.StringVar(value=str(self.current_config.decay_rate_fast))
        fast_entry = Theme.create_entry(fast_frame, width=80)
        fast_entry.configure(textvariable=self.decay_fast_var)
        fast_entry.pack(side="left")
        Theme.create_hint(fast_frame, f"[{fast_def['min']}-{fast_def['max']}] pts/day").pack(side="left", padx=Sizes.PAD_S)

        # Decay Rate Slow
        slow_def = FEATURE_PARAMETER_DEFINITIONS['slow_decay_rate']
        slow_frame = Theme.create_frame(params_content)
        slow_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(slow_frame, "Decay Rate (Slow):", width=150).pack(side="left")
        self.decay_slow_var = ctk.StringVar(value=str(self.current_config.decay_rate_slow))
        slow_entry = Theme.create_entry(slow_frame, width=80)
        slow_entry.configure(textvariable=self.decay_slow_var)
        slow_entry.pack(side="left")
        Theme.create_hint(slow_frame, f"[{slow_def['min']}-{slow_def['max']}] pts/day").pack(side="left", padx=Sizes.PAD_S)

        # Swap Threshold
        thresh_def = VULNERABILITY_CORE_PARAM_DEFINITIONS['swap_threshold']
        thresh_frame = Theme.create_frame(params_content)
        thresh_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(thresh_frame, "Swap Threshold:", width=150).pack(side="left")
        self.threshold_var = ctk.StringVar(value=str(self.current_config.swap_threshold))
        thresh_entry = Theme.create_entry(thresh_frame, width=80)
        thresh_entry.configure(textvariable=self.threshold_var)
        thresh_entry.pack(side="left")
        Theme.create_hint(thresh_frame, f"[{int(thresh_def['min'])}-{int(thresh_def['max'])}]").pack(side="left", padx=Sizes.PAD_S)

        # Description card
        desc_card = Theme.create_card(main_frame)
        desc_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        desc_content = Theme.create_frame(desc_card)
        desc_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(desc_content, "How It Works", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        desc_text = (
            "When a new BUY signal arrives with no capital available:\n"
            "1. Each open position gets a vulnerability score (0-100)\n"
            "2. New trades start at 100 and are protected during immunity period\n"
            "3. After immunity, stagnant trades (low P/L) decay faster\n"
            "4. If weakest position < swap threshold, it's closed for new signal"
        )
        Theme.create_label(desc_content, desc_text, text_color=Colors.TEXT_SECONDARY, justify="left").pack(anchor="w")

        # Buttons
        btn_frame = Theme.create_frame(main_frame)
        btn_frame.pack(fill="x")

        Theme.create_button(btn_frame, "Reset Defaults", command=self._reset_defaults, style="secondary", width=120).pack(side="left")
        Theme.create_button(btn_frame, "Cancel", command=self.dialog.destroy, style="secondary", width=80).pack(side="right")
        Theme.create_button(btn_frame, "Save", command=self._save, style="success", width=80).pack(side="right", padx=Sizes.PAD_S)

    def _reset_defaults(self):
        """Reset to default values."""
        defaults = VulnerabilityScoreConfig()
        self.immunity_var.set(str(defaults.immunity_days))
        self.profit_var.set(str(defaults.min_profit_threshold))
        self.decay_fast_var.set(str(defaults.decay_rate_fast))
        self.decay_slow_var.set(str(defaults.decay_rate_slow))
        self.threshold_var.set(str(defaults.swap_threshold))

    def _save(self):
        """Save configuration."""
        try:
            config = VulnerabilityScoreConfig(
                immunity_days=int(self.immunity_var.get()),
                min_profit_threshold=float(self.profit_var.get()),
                decay_rate_fast=float(self.decay_fast_var.get()),
                decay_rate_slow=float(self.decay_slow_var.get()),
                swap_threshold=float(self.threshold_var.get())
            )

            if self.on_save:
                self.on_save(config)

            self.dialog.destroy()

        except ValueError as e:
            show_error(self.dialog, "Invalid Value", f"Please enter valid numbers: {e}")
