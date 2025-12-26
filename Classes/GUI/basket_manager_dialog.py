"""
Basket Manager Dialog for creating and managing security baskets.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Callable

from ..Config.basket import Basket, BasketManager
from ..Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig,
    EnhancedVulnerabilityConfig, FeatureWeightConfig,
    ENHANCED_FEATURE_DEFINITIONS, FEATURE_PARAMETER_DEFINITIONS, VULNERABILITY_CORE_PARAM_DEFINITIONS
)
from ..VulnerabilityScorer.scoring import PresetManager


class BasketManagerDialog:
    """
    Dialog for creating, editing, and deleting security baskets.
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
        self.available_securities = available_securities
        self.on_basket_selected = on_basket_selected
        self.basket_manager = BasketManager()
        self.selected_basket: Optional[Basket] = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Basket Manager")
        self.dialog.geometry("700x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._create_widgets()
        self._load_baskets()

    def _create_widgets(self):
        """Create dialog widgets."""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== Left Panel: Basket List =====
        left_frame = ttk.LabelFrame(main_frame, text="Available Baskets", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Basket listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.basket_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            height=15,
            yscrollcommand=scrollbar.set
        )
        self.basket_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.basket_listbox.yview)
        self.basket_listbox.bind('<<ListboxSelect>>', self._on_basket_select)

        # Basket action buttons
        basket_buttons = ttk.Frame(left_frame)
        basket_buttons.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(basket_buttons, text="New", command=self._new_basket).pack(side=tk.LEFT, padx=2)
        ttk.Button(basket_buttons, text="Delete", command=self._delete_basket).pack(side=tk.LEFT, padx=2)
        ttk.Button(basket_buttons, text="Refresh", command=self._load_baskets).pack(side=tk.LEFT, padx=2)

        # ===== Right Panel: Basket Details/Editor =====
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Basket name
        name_frame = ttk.LabelFrame(right_frame, text="Basket Details", padding="10")
        name_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(name_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(name_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.desc_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.desc_var, width=30).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5
        )

        # Securities selection
        securities_frame = ttk.LabelFrame(right_frame, text="Securities in Basket", padding="10")
        securities_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Two columns: available and selected
        columns_frame = ttk.Frame(securities_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True)

        # Available securities
        avail_frame = ttk.Frame(columns_frame)
        avail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(avail_frame, text="Available:").pack(anchor=tk.W)
        avail_scroll = ttk.Scrollbar(avail_frame)
        avail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.available_listbox = tk.Listbox(
            avail_frame,
            selectmode=tk.MULTIPLE,
            height=10,
            yscrollcommand=avail_scroll.set
        )
        self.available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        avail_scroll.config(command=self.available_listbox.yview)

        for symbol in self.available_securities:
            self.available_listbox.insert(tk.END, symbol)

        # Add/Remove buttons
        btn_frame = ttk.Frame(columns_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text=">>", width=5, command=self._add_to_basket).pack(pady=5)
        ttk.Button(btn_frame, text="<<", width=5, command=self._remove_from_basket).pack(pady=5)

        # Selected securities
        sel_frame = ttk.Frame(columns_frame)
        sel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(sel_frame, text="In Basket:").pack(anchor=tk.W)
        sel_scroll = ttk.Scrollbar(sel_frame)
        sel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_listbox = tk.Listbox(
            sel_frame,
            selectmode=tk.MULTIPLE,
            height=10,
            yscrollcommand=sel_scroll.set
        )
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_scroll.config(command=self.selected_listbox.yview)

        # Capital Contention defaults (optional)
        contention_frame = ttk.LabelFrame(right_frame, text="Default Capital Contention", padding="10")
        contention_frame.pack(fill=tk.X, pady=(0, 10))

        self.contention_mode_var = tk.StringVar(value="default")
        ttk.Radiobutton(contention_frame, text="Default (Ignore signals)",
                       variable=self.contention_mode_var, value="default").grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=2
        )
        ttk.Radiobutton(contention_frame, text="Vulnerability Score",
                       variable=self.contention_mode_var, value="vulnerability").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=2
        )

        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="Save Basket", command=self._save_basket).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Use Selected", command=self._use_basket).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _load_baskets(self):
        """Load available baskets into listbox."""
        self.basket_listbox.delete(0, tk.END)
        baskets = self.basket_manager.list_baskets()
        for name in baskets:
            self.basket_listbox.insert(tk.END, name)

    def _on_basket_select(self, event):
        """Handle basket selection."""
        selection = self.basket_listbox.curselection()
        if not selection:
            return

        basket_name = self.basket_listbox.get(selection[0])
        basket = self.basket_manager.load(basket_name)
        if basket:
            self._populate_basket_details(basket)

    def _populate_basket_details(self, basket: Basket):
        """Populate the details panel with basket info."""
        self.name_var.set(basket.name)
        self.desc_var.set(basket.description)

        # Clear and populate selected securities
        self.selected_listbox.delete(0, tk.END)
        for symbol in basket.securities:
            self.selected_listbox.insert(tk.END, symbol)

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
        selection = self.available_listbox.curselection()
        for idx in selection:
            symbol = self.available_listbox.get(idx)
            # Check if not already in basket
            current = list(self.selected_listbox.get(0, tk.END))
            if symbol not in current:
                self.selected_listbox.insert(tk.END, symbol)

    def _remove_from_basket(self):
        """Remove selected securities from basket."""
        selection = self.selected_listbox.curselection()
        for idx in reversed(selection):  # Reverse to maintain indices
            self.selected_listbox.delete(idx)

    def _new_basket(self):
        """Clear form for new basket."""
        self.name_var.set("")
        self.desc_var.set("")
        self.selected_listbox.delete(0, tk.END)
        self.contention_mode_var.set("default")
        self.basket_listbox.selection_clear(0, tk.END)
        self.name_entry.focus_set()

    def _delete_basket(self):
        """Delete selected basket."""
        selection = self.basket_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a basket to delete.")
            return

        basket_name = self.basket_listbox.get(selection[0])
        if messagebox.askyesno("Confirm Delete", f"Delete basket '{basket_name}'?"):
            if self.basket_manager.delete(basket_name):
                self._load_baskets()
                self._new_basket()
                messagebox.showinfo("Success", f"Basket '{basket_name}' deleted.")
            else:
                messagebox.showerror("Error", f"Failed to delete basket '{basket_name}'.")

    def _save_basket(self):
        """Save current basket configuration."""
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Basket name is required.")
            return

        securities = list(self.selected_listbox.get(0, tk.END))
        if not securities:
            messagebox.showerror("Error", "Basket must contain at least one security.")
            return

        description = self.desc_var.get().strip()

        # Create capital contention config
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
            messagebox.showinfo("Success", f"Basket '{name}' saved with {len(securities)} securities.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save basket: {e}")

    def _use_basket(self):
        """Use the currently displayed basket."""
        securities = list(self.selected_listbox.get(0, tk.END))
        if not securities:
            messagebox.showwarning("No Securities", "No securities in basket.")
            return

        name = self.name_var.get().strip() or "Untitled"
        description = self.desc_var.get().strip()

        # Create capital contention config
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


class VulnerabilityScoreConfigDialog:
    """
    Dialog for configuring vulnerability score parameters.
    Supports both simple (VulnerabilityScoreConfig) and enhanced (EnhancedVulnerabilityConfig) modes.
    """

    def __init__(self, parent, current_config=None,
                 on_save: Optional[Callable] = None,
                 enhanced_mode: bool = False):
        """
        Initialize the vulnerability score config dialog.

        Args:
            parent: Parent window
            current_config: Current configuration (VulnerabilityScoreConfig or EnhancedVulnerabilityConfig)
            on_save: Callback when configuration is saved
            enhanced_mode: If True, use enhanced feature-based configuration
        """
        self.parent = parent
        self.enhanced_mode = enhanced_mode
        self.on_save = on_save

        # Initialize configuration
        if enhanced_mode:
            self.enhanced_config = current_config if isinstance(current_config, EnhancedVulnerabilityConfig) else EnhancedVulnerabilityConfig()
            self.current_config = None
        else:
            self.current_config = current_config if isinstance(current_config, VulnerabilityScoreConfig) else VulnerabilityScoreConfig()
            self.enhanced_config = None

        # Initialize preset manager for loading custom presets
        self.preset_manager = PresetManager()

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Vulnerability Score Configuration")
        self.dialog.geometry("750x650" if enhanced_mode else "500x450")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Feature variable storage
        self.feature_vars = {}

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        if self.enhanced_mode:
            self._create_enhanced_widgets(main_frame)
        else:
            self._create_simple_widgets(main_frame)

    def _create_simple_widgets(self, main_frame):
        """Create simple mode widgets (backward compatible)."""
        # Title
        title = ttk.Label(main_frame, text="Vulnerability Score Parameters",
                         font=('TkDefaultFont', 12, 'bold'))
        title.pack(pady=(0, 15))

        # Parameters frame
        params_frame = ttk.Frame(main_frame)
        params_frame.pack(fill=tk.X)

        row = 0

        # Immunity Days
        self._add_param_row(params_frame, row, "Immunity Days:",
                           str(self.current_config.immunity_days),
                           VULNERABILITY_CORE_PARAM_DEFINITIONS['immunity_days'])
        self.immunity_var = self._last_var
        row += 1

        # Min Profit Threshold (as percentage display)
        ttk.Label(params_frame, text="Min Profit Threshold:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.profit_var = tk.StringVar(value=str(self.current_config.min_profit_threshold))
        ttk.Entry(params_frame, textvariable=self.profit_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(params_frame, text="[0.0 - 0.20] (0.02 = 2%, below is 'stagnant')",
                 font=('TkDefaultFont', 8), foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # Decay Rate Fast
        self._add_param_row(params_frame, row, "Decay Rate (Fast):",
                           str(self.current_config.decay_rate_fast),
                           FEATURE_PARAMETER_DEFINITIONS['fast_decay_rate'])
        self.decay_fast_var = self._last_var
        row += 1

        # Decay Rate Slow
        self._add_param_row(params_frame, row, "Decay Rate (Slow):",
                           str(self.current_config.decay_rate_slow),
                           FEATURE_PARAMETER_DEFINITIONS['slow_decay_rate'])
        self.decay_slow_var = self._last_var
        row += 1

        # Swap Threshold
        self._add_param_row(params_frame, row, "Swap Threshold:",
                           str(self.current_config.swap_threshold),
                           VULNERABILITY_CORE_PARAM_DEFINITIONS['swap_threshold'])
        self.threshold_var = self._last_var
        row += 1

        # Description
        desc_frame = ttk.LabelFrame(main_frame, text="How It Works", padding="10")
        desc_frame.pack(fill=tk.X, pady=15)

        desc_text = (
            "When a new BUY signal arrives with no capital available:\n"
            "1. Each open position gets a vulnerability score (0-100)\n"
            "2. New trades start at 100 and are protected during immunity period\n"
            "3. After immunity, stagnant trades (low P/L) decay faster\n"
            "4. If weakest position < swap threshold, it's closed for new signal"
        )
        ttk.Label(desc_frame, text=desc_text, justify=tk.LEFT,
                 font=('TkDefaultFont', 9)).pack()

        # Mode switch
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Button(mode_frame, text="Switch to Enhanced Mode (Per-Feature Config)",
                  command=self._switch_to_enhanced).pack(side=tk.LEFT)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Reset to Defaults",
                  command=self._reset_defaults).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Save", command=self._save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.dialog.destroy).pack(side=tk.RIGHT)

    def _create_enhanced_widgets(self, main_frame):
        """Create enhanced mode widgets with per-feature configuration."""
        # Title
        title = ttk.Label(main_frame, text="Enhanced Vulnerability Score Configuration",
                         font=('TkDefaultFont', 12, 'bold'))
        title.pack(pady=(0, 10))

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Tab 1: Core Parameters
        core_frame = ttk.Frame(notebook, padding="10")
        notebook.add(core_frame, text="Core Parameters")
        self._create_core_params_tab(core_frame)

        # Tab 2: Feature Configuration
        features_frame = ttk.Frame(notebook, padding="10")
        notebook.add(features_frame, text="Feature Configuration")
        self._create_features_tab(features_frame)

        # Tab 3: Presets
        presets_frame = ttk.Frame(notebook, padding="10")
        notebook.add(presets_frame, text="Presets")
        self._create_presets_tab(presets_frame)

        # Buttons at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Reset All",
                  command=self._reset_enhanced_defaults).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Switch to Simple Mode",
                  command=self._switch_to_simple).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save", command=self._save_enhanced).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.dialog.destroy).pack(side=tk.RIGHT)

    def _create_core_params_tab(self, parent):
        """Create core parameters tab."""
        row = 0

        # Immunity Days
        self._add_param_row(parent, row, "Immunity Days:",
                           str(self.enhanced_config.immunity_days),
                           VULNERABILITY_CORE_PARAM_DEFINITIONS['immunity_days'])
        self.immunity_var = self._last_var
        row += 1

        # Base Score
        self._add_param_row(parent, row, "Base Score:",
                           str(self.enhanced_config.base_score),
                           VULNERABILITY_CORE_PARAM_DEFINITIONS['base_score'])
        self.base_score_var = self._last_var
        row += 1

        # Swap Threshold
        self._add_param_row(parent, row, "Swap Threshold:",
                           str(self.enhanced_config.swap_threshold),
                           VULNERABILITY_CORE_PARAM_DEFINITIONS['swap_threshold'])
        self.threshold_var = self._last_var
        row += 1

        # Info
        info_frame = ttk.LabelFrame(parent, text="Score Interpretation", padding="10")
        info_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=15, padx=5)

        info_text = (
            "100: Immune (within immunity period)\n"
            "50-99: Healthy but aging\n"
            "1-49: Vulnerable (below default swap threshold)\n"
            "0: Extremely vulnerable"
        )
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT,
                 font=('TkDefaultFont', 9)).pack(anchor='w')

    def _create_features_tab(self, parent):
        """Create features configuration tab with scrollable content."""
        # Instructions
        ttk.Label(parent, text="Configure which features to include and their parameters:",
                 font=('TkDefaultFont', 9, 'italic'), foreground='gray').pack(anchor='w', pady=(0, 10))

        # Scrollable frame for features
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Create feature sections
        for feature_name, feature_def in ENHANCED_FEATURE_DEFINITIONS.items():
            self._create_feature_section(scrollable_frame, feature_name, feature_def)

    def _create_feature_section(self, parent, feature_name: str, feature_def: dict):
        """Create a collapsible section for a single feature."""
        # Get current feature config
        if feature_name in self.enhanced_config.features:
            feature_config = self.enhanced_config.features[feature_name]
        else:
            feature_config = FeatureWeightConfig()

        # Feature frame
        frame = ttk.LabelFrame(parent, text=f"{feature_def['name']} [{feature_def['importance']}]", padding="8")
        frame.pack(fill=tk.X, pady=5, padx=5)

        # Initialize storage for this feature
        self.feature_vars[feature_name] = {}

        # Row 0: Enable checkbox and description
        row0 = ttk.Frame(frame)
        row0.pack(fill=tk.X)

        enabled_var = tk.BooleanVar(value=feature_config.enabled)
        self.feature_vars[feature_name]['enabled'] = enabled_var
        ttk.Checkbutton(row0, text="Enabled", variable=enabled_var).pack(side=tk.LEFT)

        ttk.Label(row0, text=f"  {feature_def['description']}",
                 font=('TkDefaultFont', 8), foreground='gray').pack(side=tk.LEFT, padx=10)

        # Row 1: Weight and hint
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(row1, text="Weight:", width=15).pack(side=tk.LEFT)
        weight_var = tk.DoubleVar(value=feature_config.weight)
        self.feature_vars[feature_name]['weight'] = weight_var
        weight_spin = ttk.Spinbox(row1, from_=feature_def['weight_range'][0],
                                  to=feature_def['weight_range'][1],
                                  textvariable=weight_var, width=8, increment=0.5)
        weight_spin.pack(side=tk.LEFT)

        range_str = f"[{feature_def['weight_range'][0]} to {feature_def['weight_range'][1]}]"
        ttk.Label(row1, text=range_str, font=('TkDefaultFont', 8), foreground='#666').pack(side=tk.LEFT, padx=5)

        # Row 2: Weight hint
        if 'weight_hint' in feature_def:
            ttk.Label(frame, text=f"  Hint: {feature_def['weight_hint']}",
                     font=('TkDefaultFont', 8), foreground='#888').pack(anchor='w')

        # Row 3: Advanced parameters (collapsible)
        adv_frame = ttk.Frame(frame)
        adv_frame.pack(fill=tk.X, pady=(5, 0))

        # Decay Point
        ttk.Label(adv_frame, text="Decay Point:", width=15).pack(side=tk.LEFT)
        decay_point_var = tk.IntVar(value=feature_config.decay_point)
        self.feature_vars[feature_name]['decay_point'] = decay_point_var
        dp_def = FEATURE_PARAMETER_DEFINITIONS['decay_point']
        ttk.Spinbox(adv_frame, from_=dp_def['min'], to=dp_def['max'],
                   textvariable=decay_point_var, width=6).pack(side=tk.LEFT)
        ttk.Label(adv_frame, text=f"days [{dp_def['min']}-{dp_def['max']}]",
                 font=('TkDefaultFont', 8), foreground='#666').pack(side=tk.LEFT, padx=5)

        # Stagnation Threshold
        ttk.Label(adv_frame, text="Stagnation:", width=10).pack(side=tk.LEFT, padx=(10, 0))
        stag_var = tk.DoubleVar(value=feature_config.stagnation_threshold)
        self.feature_vars[feature_name]['stagnation_threshold'] = stag_var
        stag_def = FEATURE_PARAMETER_DEFINITIONS['stagnation_threshold']
        ttk.Spinbox(adv_frame, from_=stag_def['min'], to=stag_def['max'],
                   textvariable=stag_var, width=6, increment=0.5).pack(side=tk.LEFT)
        ttk.Label(adv_frame, text=f"% [{stag_def['min']}-{stag_def['max']}]",
                 font=('TkDefaultFont', 8), foreground='#666').pack(side=tk.LEFT, padx=5)

        # Row 4: Decay rates
        decay_frame = ttk.Frame(frame)
        decay_frame.pack(fill=tk.X, pady=(3, 0))

        # Fast Decay
        ttk.Label(decay_frame, text="Fast Decay:", width=15).pack(side=tk.LEFT)
        fast_var = tk.DoubleVar(value=feature_config.fast_decay_rate)
        self.feature_vars[feature_name]['fast_decay_rate'] = fast_var
        fast_def = FEATURE_PARAMETER_DEFINITIONS['fast_decay_rate']
        ttk.Spinbox(decay_frame, from_=fast_def['min'], to=fast_def['max'],
                   textvariable=fast_var, width=6, increment=0.5).pack(side=tk.LEFT)
        ttk.Label(decay_frame, text=f"pts/day [{fast_def['min']}-{fast_def['max']}]",
                 font=('TkDefaultFont', 8), foreground='#666').pack(side=tk.LEFT, padx=5)

        # Slow Decay
        ttk.Label(decay_frame, text="Slow Decay:", width=10).pack(side=tk.LEFT, padx=(10, 0))
        slow_var = tk.DoubleVar(value=feature_config.slow_decay_rate)
        self.feature_vars[feature_name]['slow_decay_rate'] = slow_var
        slow_def = FEATURE_PARAMETER_DEFINITIONS['slow_decay_rate']
        ttk.Spinbox(decay_frame, from_=slow_def['min'], to=slow_def['max'],
                   textvariable=slow_var, width=6, increment=0.1).pack(side=tk.LEFT)
        ttk.Label(decay_frame, text=f"pts/day [{slow_def['min']}-{slow_def['max']}]",
                 font=('TkDefaultFont', 8), foreground='#666').pack(side=tk.LEFT, padx=5)

    def _create_presets_tab(self, parent):
        """Create presets tab."""
        # Create scrollable frame for presets
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Built-in presets section
        ttk.Label(scrollable_frame, text="Built-in Presets:",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0, 10))

        builtin_presets = [
            ("Conservative", "Protects positions longer, unlikely to swap",
             EnhancedVulnerabilityConfig.conservative_preset),
            ("Aggressive", "Swaps quickly if no progress",
             EnhancedVulnerabilityConfig.aggressive_preset),
            ("Momentum Focused", "Emphasizes recent momentum over time held",
             EnhancedVulnerabilityConfig.momentum_focused_preset),
            ("Default", "Balanced default settings",
             lambda: EnhancedVulnerabilityConfig()),
        ]

        for name, desc, preset_func in builtin_presets:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=5)

            ttk.Button(frame, text=f"Load {name}",
                      command=lambda f=preset_func: self._load_preset(f())).pack(side=tk.LEFT)
            ttk.Label(frame, text=f"  {desc}",
                     font=('TkDefaultFont', 9), foreground='gray').pack(side=tk.LEFT, padx=10)

        # Custom presets section
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(scrollable_frame, text="Custom Presets (from vulnerability_presets folder):",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0, 10))

        # Load custom presets from PresetManager
        custom_preset_names = self.preset_manager.get_custom_presets().keys()

        if custom_preset_names:
            for preset_name in custom_preset_names:
                frame = ttk.Frame(scrollable_frame)
                frame.pack(fill=tk.X, pady=5)

                ttk.Button(frame, text=f"Load {preset_name}",
                          command=lambda name=preset_name: self._load_custom_preset(name)).pack(side=tk.LEFT)
                ttk.Label(frame, text=f"  Saved preset",
                         font=('TkDefaultFont', 9), foreground='gray').pack(side=tk.LEFT, padx=10)
        else:
            ttk.Label(scrollable_frame, text="No custom presets found. Use vulnerability_gui to create and save presets.",
                     font=('TkDefaultFont', 9), foreground='#999', wraplength=400).pack(anchor='w', padx=20)

        # Preset description section (in scrollable area)
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        desc_frame = ttk.LabelFrame(scrollable_frame, text="Preset Details", padding="10")
        desc_frame.pack(fill=tk.X, pady=(0, 10))

        preset_info = """
Conservative:
  - Immunity: 14 days, Swap Threshold: 30
  - Slow aging (-1/day), high P/L weight (2.0)

Aggressive:
  - Immunity: 3 days, Swap Threshold: 70
  - Fast aging (-3/day), strong momentum focus (5.0)
  - Penalizes volatility and drawdowns

Momentum Focused:
  - Immunity: 7 days, Swap Threshold: 50
  - Both 7d and 14d momentum enabled
  - Good for trend-following strategies
"""
        ttk.Label(desc_frame, text=preset_info, justify=tk.LEFT,
                 font=('TkDefaultFont', 9)).pack(anchor='w')

    def _add_param_row(self, parent, row: int, label: str, value: str, param_def: dict):
        """Add a parameter row with label, entry, and hint."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)

        self._last_var = tk.StringVar(value=value)
        ttk.Entry(parent, textvariable=self._last_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )

        # Build range and hint text
        range_text = ""
        if 'min' in param_def and 'max' in param_def:
            range_text = f"[{param_def['min']} - {param_def['max']}]"
        hint_text = param_def.get('hint', param_def.get('description', ''))

        display_text = f"{range_text} {hint_text}" if range_text else hint_text
        ttk.Label(parent, text=display_text,
                 font=('TkDefaultFont', 8), foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=5)

    def _switch_to_enhanced(self):
        """Switch to enhanced mode."""
        self.dialog.destroy()
        VulnerabilityScoreConfigDialog(
            self.parent,
            current_config=EnhancedVulnerabilityConfig(),
            on_save=self.on_save,
            enhanced_mode=True
        )

    def _switch_to_simple(self):
        """Switch to simple mode."""
        self.dialog.destroy()
        VulnerabilityScoreConfigDialog(
            self.parent,
            current_config=VulnerabilityScoreConfig(),
            on_save=self.on_save,
            enhanced_mode=False
        )

    def _load_preset(self, preset: EnhancedVulnerabilityConfig):
        """Load a preset configuration."""
        self.enhanced_config = preset

        # Update UI
        self.immunity_var.set(str(preset.immunity_days))
        self.base_score_var.set(str(preset.base_score))
        self.threshold_var.set(str(preset.swap_threshold))

        # Update feature vars
        for feature_name, config in preset.features.items():
            if feature_name in self.feature_vars:
                self.feature_vars[feature_name]['enabled'].set(config.enabled)
                self.feature_vars[feature_name]['weight'].set(config.weight)
                self.feature_vars[feature_name]['decay_point'].set(config.decay_point)
                self.feature_vars[feature_name]['stagnation_threshold'].set(config.stagnation_threshold)
                self.feature_vars[feature_name]['fast_decay_rate'].set(config.fast_decay_rate)
                self.feature_vars[feature_name]['slow_decay_rate'].set(config.slow_decay_rate)

        messagebox.showinfo("Preset Loaded", f"Loaded preset: {preset.name}")

    def _load_custom_preset(self, preset_name: str):
        """Load a custom preset from the vulnerability_presets directory."""
        # Load the preset using PresetManager
        vuln_score_params = self.preset_manager.load_preset(preset_name)

        if vuln_score_params is None:
            messagebox.showerror("Error", f"Could not load preset: {preset_name}")
            return

        # Convert VulnerabilityScoreParams to EnhancedVulnerabilityConfig
        # Build feature configs from the params
        features = {}
        for feature_name, feature_weight in vuln_score_params.features.items():
            features[feature_name] = FeatureWeightConfig(
                enabled=feature_weight.enabled,
                weight=feature_weight.weight,
                use_advanced_params=feature_weight.use_advanced_params,
                decay_point=feature_weight.decay_point,
                fast_decay_rate=feature_weight.fast_decay_rate,
                slow_decay_rate=feature_weight.slow_decay_rate,
                stagnation_threshold=feature_weight.stagnation_threshold,
                normalize_min=feature_weight.normalize_min,
                normalize_max=feature_weight.normalize_max
            )

        enhanced_config = EnhancedVulnerabilityConfig(
            name=vuln_score_params.name,
            description=vuln_score_params.description,
            immunity_days=vuln_score_params.immunity_days,
            base_score=vuln_score_params.base_score,
            swap_threshold=vuln_score_params.swap_threshold,
            features=features,
            tiebreaker_order=vuln_score_params.tiebreaker_order
        )

        # Load it using the standard method
        self._load_preset(enhanced_config)

    def _reset_defaults(self):
        """Reset to default values (simple mode)."""
        defaults = VulnerabilityScoreConfig()
        self.immunity_var.set(str(defaults.immunity_days))
        self.profit_var.set(str(defaults.min_profit_threshold))
        self.decay_fast_var.set(str(defaults.decay_rate_fast))
        self.decay_slow_var.set(str(defaults.decay_rate_slow))
        self.threshold_var.set(str(defaults.swap_threshold))

    def _reset_enhanced_defaults(self):
        """Reset to default values (enhanced mode)."""
        self._load_preset(EnhancedVulnerabilityConfig())

    def _save(self):
        """Save configuration (simple mode)."""
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
            messagebox.showerror("Invalid Value", f"Please enter valid numbers: {e}")

    def _save_enhanced(self):
        """Save configuration (enhanced mode)."""
        try:
            # Build feature configs
            features = {}
            for feature_name in ENHANCED_FEATURE_DEFINITIONS.keys():
                if feature_name in self.feature_vars:
                    fv = self.feature_vars[feature_name]
                    features[feature_name] = FeatureWeightConfig(
                        enabled=fv['enabled'].get(),
                        weight=fv['weight'].get(),
                        decay_point=fv['decay_point'].get(),
                        fast_decay_rate=fv['fast_decay_rate'].get(),
                        slow_decay_rate=fv['slow_decay_rate'].get(),
                        stagnation_threshold=fv['stagnation_threshold'].get()
                    )

            config = EnhancedVulnerabilityConfig(
                name="Custom",
                description="User-defined configuration",
                immunity_days=int(self.immunity_var.get()),
                base_score=float(self.base_score_var.get()),
                swap_threshold=float(self.threshold_var.get()),
                features=features
            )

            if self.on_save:
                self.on_save(config)

            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Invalid Value", f"Please enter valid numbers: {e}")
