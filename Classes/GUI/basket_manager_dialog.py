"""
Basket Manager Dialog for creating and managing security baskets.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Callable

from ..Config.basket import Basket, BasketManager
from ..Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig
)


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
    """

    def __init__(self, parent, current_config: Optional[VulnerabilityScoreConfig] = None,
                 on_save: Optional[Callable[[VulnerabilityScoreConfig], None]] = None):
        """
        Initialize the vulnerability score config dialog.

        Args:
            parent: Parent window
            current_config: Current configuration (or None for defaults)
            on_save: Callback when configuration is saved
        """
        self.parent = parent
        self.current_config = current_config or VulnerabilityScoreConfig()
        self.on_save = on_save

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Vulnerability Score Configuration")
        self.dialog.geometry("450x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(main_frame, text="Vulnerability Score Parameters",
                         font=('Arial', 12, 'bold'))
        title.pack(pady=(0, 15))

        # Parameters frame
        params_frame = ttk.Frame(main_frame)
        params_frame.pack(fill=tk.X)

        row = 0

        # Immunity Days
        ttk.Label(params_frame, text="Immunity Days:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.immunity_var = tk.StringVar(value=str(self.current_config.immunity_days))
        ttk.Entry(params_frame, textvariable=self.immunity_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(params_frame, text="(New trades protected from swap)",
                 font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # Min Profit Threshold
        ttk.Label(params_frame, text="Min Profit Threshold:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.profit_var = tk.StringVar(value=str(self.current_config.min_profit_threshold))
        ttk.Entry(params_frame, textvariable=self.profit_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(params_frame, text="(0.02 = 2%, below is 'stagnant')",
                 font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # Decay Rate Fast
        ttk.Label(params_frame, text="Decay Rate (Fast):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.decay_fast_var = tk.StringVar(value=str(self.current_config.decay_rate_fast))
        ttk.Entry(params_frame, textvariable=self.decay_fast_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(params_frame, text="(Points/day for stagnant trades)",
                 font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # Decay Rate Slow
        ttk.Label(params_frame, text="Decay Rate (Slow):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.decay_slow_var = tk.StringVar(value=str(self.current_config.decay_rate_slow))
        ttk.Entry(params_frame, textvariable=self.decay_slow_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(params_frame, text="(Points/day for performing trades)",
                 font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        # Swap Threshold
        ttk.Label(params_frame, text="Swap Threshold:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.StringVar(value=str(self.current_config.swap_threshold))
        ttk.Entry(params_frame, textvariable=self.threshold_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(params_frame, text="(Score below which can be swapped)",
                 font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=5)
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
                 font=('Arial', 9)).pack()

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Reset to Defaults",
                  command=self._reset_defaults).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Save", command=self._save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.dialog.destroy).pack(side=tk.RIGHT)

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
            messagebox.showerror("Invalid Value", f"Please enter valid numbers: {e}")
