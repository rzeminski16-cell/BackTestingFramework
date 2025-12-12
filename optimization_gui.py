"""
Optimization GUI

A graphical interface for walk-forward optimization with Bayesian optimization.

Features:
- Select strategy to optimize
- Choose optimization mode:
  * Single Security: Optimize one security at a time
  * Portfolio: Optimize with shared capital across multiple securities
- Choose which parameters to optimize
- Configure optimization settings
- Configure capital contention settings for portfolio mode
- Create and manage baskets of securities
- Run walk-forward optimization
- View progress with real-time updates
- Generate comprehensive Excel reports
- Cancel optimization in progress
"""

import logging
import os
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

import pandas as pd

from Classes.Data.data_loader import DataLoader
from Classes.Config.basket import Basket, BasketManager
from Classes.Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig
)
from Classes.Optimization.optimization_report_generator import \
    OptimizationReportGenerator
from Classes.Optimization.sensitivity_analyzer import SensitivityAnalyzer
from Classes.Optimization.walk_forward_optimizer import WalkForwardOptimizer, WalkForwardMode
from Classes.GUI.basket_manager_dialog import BasketManagerDialog, VulnerabilityScoreConfigDialog

# Import available strategies
from strategies.alphatrend_strategy import AlphaTrendStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationGUI:
    """Main GUI application for walk-forward optimization."""

    # Available strategies
    STRATEGIES = {
        'AlphaTrendStrategy': AlphaTrendStrategy
    }

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Walk-Forward Optimization")
        self.root.geometry("1000x700")

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/processed_exports'))
        self.available_securities = self.data_loader.get_available_symbols()

        # Optimization components
        self.optimizer = WalkForwardOptimizer()
        self.sensitivity_analyzer = SensitivityAnalyzer(self.optimizer.config)
        self.report_generator = OptimizationReportGenerator(self.optimizer.config, use_enhanced=True)

        # State
        self.optimization_thread = None
        self.is_running = False
        self.selected_parameters = {}  # Track which parameters to optimize

        # Basket manager for portfolio mode
        self.basket_manager = BasketManager()
        self.selected_basket: Optional[Basket] = None

        # Capital contention configuration for portfolio mode
        self.capital_contention_config = CapitalContentionConfig.default_mode()
        self.vulnerability_config = VulnerabilityScoreConfig()

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ===== Left Panel: Configuration =====
        config_frame = ttk.LabelFrame(main_frame, text="Optimization Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        row = 0

        # Strategy Selection
        ttk.Label(config_frame, text="Strategy:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(
            config_frame,
            textvariable=self.strategy_var,
            values=list(self.STRATEGIES.keys()),
            state="readonly",
            width=28
        )
        self.strategy_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        if self.STRATEGIES:
            self.strategy_combo.current(0)
        self.strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        row += 1

        # Optimization Mode Selection
        ttk.Label(config_frame, text="Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        mode_frame = ttk.Frame(config_frame)
        mode_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single Security", variable=self.mode_var,
                       value="single", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Portfolio", variable=self.mode_var,
                       value="portfolio", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        row += 1

        # Securities Selection
        ttk.Label(config_frame, text="Securities:").grid(row=row, column=0, sticky=tk.W, pady=5)
        securities_container = ttk.Frame(config_frame)
        securities_container.grid(row=row, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Listbox with scrollbar
        securities_frame = ttk.Frame(securities_container)
        securities_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(securities_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.securities_listbox = tk.Listbox(
            securities_frame,
            selectmode=tk.MULTIPLE,
            height=6,
            yscrollcommand=scrollbar.set
        )
        self.securities_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.securities_listbox.yview)

        # Populate securities
        for symbol in sorted(self.available_securities):
            self.securities_listbox.insert(tk.END, symbol)

        # Selection buttons frame (changes based on mode)
        self.selection_buttons_frame = ttk.Frame(securities_container)
        self.selection_buttons_frame.pack(fill=tk.X, pady=2)
        self.select_all_btn = ttk.Button(self.selection_buttons_frame, text="Select All",
                                          command=self.select_all_securities)
        self.deselect_all_btn = ttk.Button(self.selection_buttons_frame, text="Deselect All",
                                            command=self.deselect_all_securities)
        # Initially hidden for single mode
        self.select_all_btn.pack_forget()
        self.deselect_all_btn.pack_forget()

        row += 1

        # Portfolio-specific settings (initially hidden)
        self.portfolio_frame = ttk.LabelFrame(config_frame, text="Portfolio/Basket Settings", padding="5")
        self.portfolio_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.portfolio_frame.grid_remove()  # Hide initially

        # Basket Selection
        basket_row = 0
        ttk.Label(self.portfolio_frame, text="Basket:").grid(row=basket_row, column=0, sticky=tk.W, pady=2)
        basket_select_frame = ttk.Frame(self.portfolio_frame)
        basket_select_frame.grid(row=basket_row, column=1, sticky=tk.W, pady=2)

        self.basket_var = tk.StringVar(value="(Select securities manually)")
        self.basket_combo = ttk.Combobox(basket_select_frame, textvariable=self.basket_var, width=20, state='readonly')
        self.basket_combo.pack(side=tk.LEFT, padx=(0, 5))
        self._refresh_basket_list()
        self.basket_combo.bind('<<ComboboxSelected>>', self._on_basket_selected)

        ttk.Button(basket_select_frame, text="Manage...", command=self._open_basket_manager).pack(side=tk.LEFT)
        basket_row += 1

        # Capital Contention Mode
        ttk.Label(self.portfolio_frame, text="Capital Contention:").grid(row=basket_row, column=0, sticky=tk.W, pady=2)
        contention_frame = ttk.Frame(self.portfolio_frame)
        contention_frame.grid(row=basket_row, column=1, sticky=tk.W, pady=2)

        self.contention_mode_var = tk.StringVar(value="default")
        ttk.Radiobutton(contention_frame, text="Default (Ignore)",
                       variable=self.contention_mode_var, value="default",
                       command=self._on_contention_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(contention_frame, text="Vulnerability Score",
                       variable=self.contention_mode_var, value="vulnerability",
                       command=self._on_contention_mode_change).pack(side=tk.LEFT, padx=2)
        basket_row += 1

        # Vulnerability Score Config Button (shown when vulnerability mode selected)
        self.vuln_config_btn = ttk.Button(self.portfolio_frame, text="Configure Vulnerability Score...",
                                         command=self._open_vulnerability_config)
        self.vuln_config_btn.grid(row=basket_row, column=0, columnspan=2, pady=5)
        self.vuln_config_btn.grid_remove()  # Hide initially

        row += 1

        # Run Sensitivity Analysis checkbox
        self.run_sensitivity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_frame,
            text="Run Sensitivity Analysis",
            variable=self.run_sensitivity_var
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # Start date filter
        ttk.Label(config_frame, text="Start Date (optional):").grid(row=row, column=0, sticky=tk.W, pady=5)
        date_frame = ttk.Frame(config_frame)
        date_frame.grid(row=row, column=1, sticky=tk.W, pady=5)

        self.use_start_date_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(date_frame, text="Use", variable=self.use_start_date_var,
                       command=self.toggle_start_date).pack(side=tk.LEFT, padx=(0, 5))

        self.start_date_entry = ttk.Entry(date_frame, width=12, state=tk.DISABLED)
        self.start_date_entry.pack(side=tk.LEFT)
        self.start_date_entry.insert(0, "YYYY-MM-DD")

        ttk.Label(date_frame, text="(filters to this date or oldest available)",
                 font=('TkDefaultFont', 8, 'italic')).pack(side=tk.LEFT, padx=5)
        row += 1

        # Speed mode selection
        ttk.Label(config_frame, text="Speed Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.speed_mode_var = tk.StringVar(value="full")
        speed_frame = ttk.Frame(config_frame)
        speed_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(speed_frame, text="Full (100 iter)", variable=self.speed_mode_var,
                       value="full").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(speed_frame, text="Fast (50 iter)", variable=self.speed_mode_var,
                       value="fast").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(speed_frame, text="Quick (25 iter)", variable=self.speed_mode_var,
                       value="quick").pack(side=tk.LEFT, padx=5)
        row += 1

        # Parallel jobs setting (only show on non-Windows platforms)
        import platform
        self.n_jobs_var = tk.IntVar(value=1)
        if platform.system() != 'Windows':
            ttk.Label(config_frame, text="CPU Cores:").grid(row=row, column=0, sticky=tk.W, pady=5)
            jobs_frame = ttk.Frame(config_frame)
            jobs_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
            ttk.Radiobutton(jobs_frame, text="1 (Recommended)", variable=self.n_jobs_var,
                           value=1).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(jobs_frame, text="2", variable=self.n_jobs_var,
                           value=2).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(jobs_frame, text="4", variable=self.n_jobs_var,
                           value=4).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(jobs_frame, text="All", variable=self.n_jobs_var,
                           value=-1).pack(side=tk.LEFT, padx=5)
            row += 1

        # Walk-Forward Window Settings
        wf_frame = ttk.LabelFrame(config_frame, text="Walk-Forward Window Settings", padding="5")
        wf_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Get default values from config
        wf_defaults = self.optimizer.config.get('walk_forward', {})
        default_train = wf_defaults.get('training_period_days', 365)
        default_test = wf_defaults.get('testing_period_days', 365)
        default_step_min = wf_defaults.get('step_size_min_days', 7)
        default_step_max = wf_defaults.get('step_size_max_days', 30)
        default_mode = wf_defaults.get('mode', 'rolling')

        # Walk-Forward Mode
        mode_row = ttk.Frame(wf_frame)
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Walk-Forward Mode:", width=20).pack(side=tk.LEFT)
        self.wf_mode_var = tk.StringVar(value=default_mode)
        ttk.Radiobutton(mode_row, text="Rolling (sliding window)",
                       variable=self.wf_mode_var, value="rolling").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_row, text="Anchored (expanding window)",
                       variable=self.wf_mode_var, value="anchored").pack(side=tk.LEFT, padx=5)

        # Mode description
        mode_desc_row = ttk.Frame(wf_frame)
        mode_desc_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(mode_desc_row, text="   Rolling: Fixed window slides forward (good for frequent trading)",
                 font=('TkDefaultFont', 8, 'italic')).pack(anchor=tk.W)
        ttk.Label(mode_desc_row, text="   Anchored: Start fixed, window expands (good for low-frequency trading)",
                 font=('TkDefaultFont', 8, 'italic')).pack(anchor=tk.W)

        # Training period
        train_row = ttk.Frame(wf_frame)
        train_row.pack(fill=tk.X, pady=2)
        ttk.Label(train_row, text="Training Period (days):", width=20).pack(side=tk.LEFT)
        self.training_period_var = tk.IntVar(value=default_train)
        training_spin = ttk.Spinbox(train_row, from_=90, to=1825, width=8,
                                     textvariable=self.training_period_var)
        training_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(train_row, text="(90-1825)", font=('TkDefaultFont', 8, 'italic')).pack(side=tk.LEFT)

        # Testing period
        test_row = ttk.Frame(wf_frame)
        test_row.pack(fill=tk.X, pady=2)
        ttk.Label(test_row, text="Testing Period (days):", width=20).pack(side=tk.LEFT)
        self.testing_period_var = tk.IntVar(value=default_test)
        testing_spin = ttk.Spinbox(test_row, from_=30, to=730, width=8,
                                    textvariable=self.testing_period_var)
        testing_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(test_row, text="(30-730)", font=('TkDefaultFont', 8, 'italic')).pack(side=tk.LEFT)

        # Step size
        step_row = ttk.Frame(wf_frame)
        step_row.pack(fill=tk.X, pady=2)
        ttk.Label(step_row, text="Step Size (days):", width=20).pack(side=tk.LEFT)
        self.step_min_var = tk.IntVar(value=default_step_min)
        step_min_spin = ttk.Spinbox(step_row, from_=7, to=365, width=6,
                                     textvariable=self.step_min_var)
        step_min_spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(step_row, text="to").pack(side=tk.LEFT, padx=2)
        self.step_max_var = tk.IntVar(value=default_step_max)
        step_max_spin = ttk.Spinbox(step_row, from_=7, to=365, width=6,
                                     textvariable=self.step_max_var)
        step_max_spin.pack(side=tk.LEFT, padx=2)

        # Window estimation label
        self.window_estimate_label = ttk.Label(
            wf_frame,
            text="",
            font=('TkDefaultFont', 8, 'italic'),
            foreground='blue'
        )
        self.window_estimate_label.pack(fill=tk.X, pady=2)

        # Bind changes to update estimate
        self.training_period_var.trace_add('write', self._update_window_estimate)
        self.testing_period_var.trace_add('write', self._update_window_estimate)
        self.step_min_var.trace_add('write', self._update_window_estimate)
        self.step_max_var.trace_add('write', self._update_window_estimate)
        # Also update when start date filter changes
        self.use_start_date_var.trace_add('write', self._update_window_estimate)
        self.start_date_entry.bind('<KeyRelease>', lambda e: self._update_window_estimate())
        # Update when security selection changes
        self.securities_listbox.bind('<<ListboxSelect>>', lambda e: self._update_window_estimate())

        # Config file path display
        ttk.Label(config_frame, text="Config File:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(
            config_frame,
            text="config/optimization_config.yaml",
            font=('TkDefaultFont', 8, 'italic')
        ).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)

        self.configure_params_button = ttk.Button(
            button_frame,
            text="Select Parameters...",
            command=self.configure_parameters
        )
        self.configure_params_button.pack(side=tk.LEFT, padx=5)

        # Parameter selection status label
        self.param_status_label = ttk.Label(
            config_frame,
            text="(All parameters will be optimized)",
            font=('TkDefaultFont', 8, 'italic'),
            foreground='gray'
        )
        self.param_status_label.grid(row=row + 1, column=0, columnspan=2, pady=(0, 5))

        self.optimize_button = ttk.Button(
            button_frame,
            text="Start Optimization",
            command=self.start_optimization
        )
        self.optimize_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_optimization,
            state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # ===== Right Panel: Results and Progress =====
        results_frame = ttk.LabelFrame(main_frame, text="Progress & Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Progress bar
        ttk.Label(results_frame, text="Progress:").pack(anchor=tk.W, pady=(0, 5))
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(results_frame, textvariable=self.progress_var)
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(results_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Results text area
        ttk.Label(results_frame, text="Results:").pack(anchor=tk.W, pady=(10, 5))
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            width=60,
            height=30,
            wrap=tk.WORD,
            font=('Courier', 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def select_all_securities(self):
        """Select all securities in the listbox."""
        self.securities_listbox.select_set(0, tk.END)

    def deselect_all_securities(self):
        """Deselect all securities in the listbox."""
        self.securities_listbox.select_clear(0, tk.END)

    def on_mode_change(self):
        """Handle optimization mode change."""
        mode = self.mode_var.get()
        if mode == "portfolio":
            # Portfolio mode: show portfolio settings, enable multi-select
            self.portfolio_frame.grid()
            self.securities_listbox.config(selectmode=tk.MULTIPLE)
            self.select_all_btn.pack(side=tk.LEFT, padx=2)
            self.deselect_all_btn.pack(side=tk.LEFT, padx=2)
        else:
            # Single mode: hide portfolio settings, single-select only
            self.portfolio_frame.grid_remove()
            self.securities_listbox.config(selectmode=tk.SINGLE)
            self.select_all_btn.pack_forget()
            self.deselect_all_btn.pack_forget()
            # Clear multiple selections if any
            selected = self.securities_listbox.curselection()
            if len(selected) > 1:
                self.securities_listbox.selection_clear(0, tk.END)
                self.securities_listbox.selection_set(selected[0])

    def _refresh_basket_list(self):
        """Refresh the basket dropdown list."""
        baskets = self.basket_manager.list_baskets()
        values = ["(Select securities manually)"] + baskets
        self.basket_combo['values'] = values

    def _on_basket_selected(self, event):
        """Handle basket selection from dropdown."""
        basket_name = self.basket_var.get()
        if basket_name == "(Select securities manually)":
            self.selected_basket = None
            return

        basket = self.basket_manager.load(basket_name)
        if basket:
            self.selected_basket = basket
            # Update securities listbox to show basket securities
            self.securities_listbox.selection_clear(0, tk.END)
            for i, symbol in enumerate(self.securities_listbox.get(0, tk.END)):
                if symbol in basket.securities:
                    self.securities_listbox.selection_set(i)

            # Load basket's default capital contention if set
            if basket.default_capital_contention:
                if basket.default_capital_contention.mode == CapitalContentionMode.VULNERABILITY_SCORE:
                    self.contention_mode_var.set("vulnerability")
                    self.vulnerability_config = basket.default_capital_contention.vulnerability_config
                else:
                    self.contention_mode_var.set("default")
                self._on_contention_mode_change()

    def _open_basket_manager(self):
        """Open the basket manager dialog."""
        def on_basket_selected(basket):
            if basket:
                self.selected_basket = basket
                self._refresh_basket_list()
                self.basket_var.set(basket.name)
                self._on_basket_selected(None)

        BasketManagerDialog(
            self.root,
            self.available_securities,
            on_basket_selected=on_basket_selected
        )

    def _on_contention_mode_change(self):
        """Handle capital contention mode change."""
        mode = self.contention_mode_var.get()
        if mode == "vulnerability":
            self.vuln_config_btn.grid()
            self.capital_contention_config = CapitalContentionConfig(
                mode=CapitalContentionMode.VULNERABILITY_SCORE,
                vulnerability_config=self.vulnerability_config
            )
        else:
            self.vuln_config_btn.grid_remove()
            self.capital_contention_config = CapitalContentionConfig.default_mode()

    def _open_vulnerability_config(self):
        """Open vulnerability score configuration dialog."""
        def on_save(config):
            self.vulnerability_config = config
            self.capital_contention_config = CapitalContentionConfig(
                mode=CapitalContentionMode.VULNERABILITY_SCORE,
                vulnerability_config=config
            )

        VulnerabilityScoreConfigDialog(
            self.root,
            current_config=self.vulnerability_config,
            on_save=on_save
        )

    def toggle_start_date(self):
        """Toggle the start date entry field."""
        if self.use_start_date_var.get():
            self.start_date_entry.config(state=tk.NORMAL)
            if self.start_date_entry.get() == "YYYY-MM-DD":
                self.start_date_entry.delete(0, tk.END)
        else:
            self.start_date_entry.config(state=tk.DISABLED)

    def _update_window_estimate(self, *args):
        """Update the estimated number of windows based on current settings."""
        try:
            train_days = self.training_period_var.get()
            test_days = self.testing_period_var.get()
            step_min = self.step_min_var.get()
            step_max = self.step_max_var.get()

            # Validate inputs
            if train_days <= 0 or test_days <= 0 or step_min <= 0 or step_max <= 0:
                self.window_estimate_label.config(text="Invalid settings")
                return

            if step_min > step_max:
                self.window_estimate_label.config(text="Step min > max")
                return

            # Estimate based on selected securities data range
            selected_indices = self.securities_listbox.curselection()
            data_days = None
            filter_note = ""

            if selected_indices:
                # Try to get actual data range from first selected security
                try:
                    symbol = self.securities_listbox.get(selected_indices[0])
                    data = self.data_loader.load_csv(symbol)
                    data_start = data['date'].min()
                    data_end = data['date'].max()

                    # Check if start date filter is applied
                    if self.use_start_date_var.get():
                        start_date_str = self.start_date_entry.get().strip()
                        if start_date_str and start_date_str != "YYYY-MM-DD":
                            try:
                                import pandas as pd
                                filter_start = pd.to_datetime(start_date_str)
                                # Use the later of filter start or data start
                                effective_start = max(filter_start, data_start)
                                data_days = (data_end - effective_start).days
                                filter_note = " (filtered)"
                            except:
                                pass

                    # If no filter or filter parsing failed, use full range
                    if data_days is None:
                        data_days = (data_end - data_start).days
                except:
                    pass

            # Default if no security selected or loading failed
            if data_days is None:
                data_days = 15 * 365

            # Calculate estimated windows (matching _split_into_windows logic)
            window_size = train_days + test_days
            avg_step = (step_min + step_max) // 2  # Integer division to match optimizer

            if data_days < window_size:
                self.window_estimate_label.config(
                    text=f"Not enough data ({data_days} days < {window_size} required)"
                )
                return

            # Calculate windows: start at day 0, step forward until test_end exceeds data
            # Window N starts at day (N-1) * avg_step, ends at (N-1) * avg_step + window_size
            # Last window: (N-1) * avg_step + window_size <= data_days
            # Solving: N <= (data_days - window_size) / avg_step + 1
            est_windows = int((data_days - window_size) / avg_step) + 1

            # Show estimate with explanation
            self.window_estimate_label.config(
                text=f"≈ {est_windows} windows (step {avg_step} days, {data_days} days{filter_note})"
            )

        except (tk.TclError, ValueError):
            # Invalid input during typing
            pass

    def on_strategy_change(self, event=None):
        """Handle strategy selection change."""
        strategy_name = self.strategy_var.get()
        self.log_message(f"Selected strategy: {strategy_name}")
        # Reset parameter selection when strategy changes
        self.selected_parameters = {}
        self.param_status_label.config(text="(All parameters will be optimized)")

    def configure_parameters(self):
        """Open dialog to select which parameters to optimize."""
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            messagebox.showerror("Error", "Please select a strategy first")
            return

        # Get parameter configuration for this strategy
        strategy_config = self.optimizer.config['strategy_parameters'].get(strategy_name)
        if not strategy_config:
            messagebox.showerror("Error", f"No parameter configuration found for {strategy_name}")
            return

        # Get strategy class to read default values
        strategy_class = self.STRATEGIES[strategy_name]

        # Initialize selected_parameters if empty (first time for this strategy)
        if not self.selected_parameters:
            for param_name in strategy_config.keys():
                self.selected_parameters[param_name] = True
            # Initialize vulnerability score parameters (only for portfolio mode)
            self.selected_parameters['_vuln_immunity_days'] = False
            self.selected_parameters['_vuln_min_profit_threshold'] = False
            self.selected_parameters['_vuln_decay_rate_fast'] = False
            self.selected_parameters['_vuln_decay_rate_slow'] = False
            self.selected_parameters['_vuln_swap_threshold'] = False

        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Select Parameters to Optimize - {strategy_name}")
        dialog.geometry("700x650")
        dialog.transient(self.root)
        dialog.grab_set()

        # Title
        title_label = ttk.Label(
            dialog,
            text="Select Parameters to Optimize",
            font=('TkDefaultFont', 12, 'bold')
        )
        title_label.pack(pady=10)

        # Instructions
        instr_label = ttk.Label(
            dialog,
            text="✓ = Optimize | ✗ = Use default value",
            font=('TkDefaultFont', 9, 'italic')
        )
        instr_label.pack(pady=5)

        # Create scrollable frame for parameters
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Get default values from strategy
        try:
            default_strategy = strategy_class()
            default_params = {
                param: getattr(default_strategy, param, None)
                for param in strategy_config.keys()
            }
        except:
            default_params = {
                param: spec.get('min', 0)
                for param, spec in strategy_config.items()
            }

        param_vars = {}

        # Helper function to add a section header
        def add_section_header(parent, text):
            header_frame = ttk.Frame(parent)
            header_frame.pack(fill=tk.X, padx=10, pady=(15, 5))
            ttk.Separator(header_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 5))
            ttk.Label(header_frame, text=text, font=('TkDefaultFont', 10, 'bold'),
                     foreground='#2E5994').pack(anchor=tk.W)

        # Helper function to add a parameter row
        def add_param_row(parent, param_name, param_spec, default_val, display_name=None):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, padx=30, pady=3)

            var = tk.BooleanVar(value=self.selected_parameters.get(param_name, True))
            param_vars[param_name] = var

            cb = ttk.Checkbutton(frame, variable=var, width=3)
            cb.pack(side=tk.LEFT)

            param_type = param_spec.get('type', 'float')
            min_val = param_spec.get('min', 'N/A')
            max_val = param_spec.get('max', 'N/A')

            if 'values' in param_spec:
                range_str = f"Options: {param_spec['values']}"
            else:
                if param_type == 'int':
                    range_str = f"Range: {int(min_val)} - {int(max_val)}"
                else:
                    range_str = f"Range: {min_val:.2f} - {max_val:.2f}"

            name = display_name or param_name
            info_text = f"{name}  [{range_str}]  Default: {default_val}"
            label = ttk.Label(frame, text=info_text, font=('TkDefaultFont', 9))
            label.pack(side=tk.LEFT, padx=10)

        # Categorize strategy parameters by type
        entry_params = {}
        exit_params = {}
        indicator_params = {}
        other_params = {}

        for param_name, param_spec in strategy_config.items():
            name_lower = param_name.lower()
            if any(kw in name_lower for kw in ['entry', 'buy', 'signal']):
                entry_params[param_name] = param_spec
            elif any(kw in name_lower for kw in ['exit', 'sell', 'stop', 'take', 'trailing']):
                exit_params[param_name] = param_spec
            elif any(kw in name_lower for kw in ['period', 'length', 'lookback', 'window', 'multiplier', 'atr']):
                indicator_params[param_name] = param_spec
            else:
                other_params[param_name] = param_spec

        # --- STRATEGY PARAMETERS SECTION ---
        add_section_header(scrollable_frame, "STRATEGY PARAMETERS")

        # Indicator/Core parameters
        if indicator_params:
            ttk.Label(scrollable_frame, text="Indicator Settings:",
                     font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W, padx=20, pady=(10, 2))
            for param_name, param_spec in indicator_params.items():
                add_param_row(scrollable_frame, param_name, param_spec, default_params.get(param_name, 'N/A'))

        # Entry parameters
        if entry_params:
            ttk.Label(scrollable_frame, text="Entry Settings:",
                     font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W, padx=20, pady=(10, 2))
            for param_name, param_spec in entry_params.items():
                add_param_row(scrollable_frame, param_name, param_spec, default_params.get(param_name, 'N/A'))

        # Exit parameters
        if exit_params:
            ttk.Label(scrollable_frame, text="Exit Settings:",
                     font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W, padx=20, pady=(10, 2))
            for param_name, param_spec in exit_params.items():
                add_param_row(scrollable_frame, param_name, param_spec, default_params.get(param_name, 'N/A'))

        # Other parameters
        if other_params:
            ttk.Label(scrollable_frame, text="Other Settings:",
                     font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W, padx=20, pady=(10, 2))
            for param_name, param_spec in other_params.items():
                add_param_row(scrollable_frame, param_name, param_spec, default_params.get(param_name, 'N/A'))

        # --- VULNERABILITY SCORE PARAMETERS SECTION (Portfolio mode only) ---
        mode = self.mode_var.get()
        if mode == "portfolio" and self.contention_mode_var.get() == "vulnerability":
            add_section_header(scrollable_frame, "VULNERABILITY SCORE PARAMETERS (Portfolio Only)")

            ttk.Label(scrollable_frame, text="These parameters control position swapping when capital is limited:",
                     font=('TkDefaultFont', 8, 'italic'), foreground='gray').pack(anchor=tk.W, padx=20, pady=(5, 10))

            vuln_params = {
                '_vuln_immunity_days': {'type': 'int', 'min': 1, 'max': 30},
                '_vuln_min_profit_threshold': {'type': 'float', 'min': 0.0, 'max': 0.10},
                '_vuln_decay_rate_fast': {'type': 'float', 'min': 1.0, 'max': 15.0},
                '_vuln_decay_rate_slow': {'type': 'float', 'min': 0.1, 'max': 5.0},
                '_vuln_swap_threshold': {'type': 'float', 'min': 20.0, 'max': 80.0},
            }
            vuln_defaults = {
                '_vuln_immunity_days': self.vulnerability_config.immunity_days,
                '_vuln_min_profit_threshold': self.vulnerability_config.min_profit_threshold,
                '_vuln_decay_rate_fast': self.vulnerability_config.decay_rate_fast,
                '_vuln_decay_rate_slow': self.vulnerability_config.decay_rate_slow,
                '_vuln_swap_threshold': self.vulnerability_config.swap_threshold,
            }
            vuln_display_names = {
                '_vuln_immunity_days': 'Immunity Days',
                '_vuln_min_profit_threshold': 'Min Profit Threshold',
                '_vuln_decay_rate_fast': 'Fast Decay Rate',
                '_vuln_decay_rate_slow': 'Slow Decay Rate',
                '_vuln_swap_threshold': 'Swap Threshold',
            }

            for param_name, param_spec in vuln_params.items():
                add_param_row(scrollable_frame, param_name, param_spec,
                            vuln_defaults.get(param_name, 'N/A'),
                            vuln_display_names.get(param_name))

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)

        # --- BUTTONS ---
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def select_all_strategy():
            for name, var in param_vars.items():
                if not name.startswith('_vuln_'):
                    var.set(True)

        def deselect_all_strategy():
            for name, var in param_vars.items():
                if not name.startswith('_vuln_'):
                    var.set(False)

        def select_all_vuln():
            for name, var in param_vars.items():
                if name.startswith('_vuln_'):
                    var.set(True)

        def deselect_all_vuln():
            for name, var in param_vars.items():
                if name.startswith('_vuln_'):
                    var.set(False)

        # Strategy param buttons
        strategy_btn_frame = ttk.Frame(button_frame)
        strategy_btn_frame.pack(pady=5)
        ttk.Label(strategy_btn_frame, text="Strategy:").pack(side=tk.LEFT, padx=5)
        ttk.Button(strategy_btn_frame, text="Select All", command=select_all_strategy).pack(side=tk.LEFT, padx=2)
        ttk.Button(strategy_btn_frame, text="Deselect All", command=deselect_all_strategy).pack(side=tk.LEFT, padx=2)

        # Vulnerability param buttons (only if shown)
        if mode == "portfolio" and self.contention_mode_var.get() == "vulnerability":
            vuln_btn_frame = ttk.Frame(button_frame)
            vuln_btn_frame.pack(pady=5)
            ttk.Label(vuln_btn_frame, text="Vulnerability:").pack(side=tk.LEFT, padx=5)
            ttk.Button(vuln_btn_frame, text="Select All", command=select_all_vuln).pack(side=tk.LEFT, padx=2)
            ttk.Button(vuln_btn_frame, text="Deselect All", command=deselect_all_vuln).pack(side=tk.LEFT, padx=2)

        # Save/Cancel buttons
        action_frame = ttk.Frame(button_frame)
        action_frame.pack(pady=10)

        def save_and_close():
            for param_name, var in param_vars.items():
                self.selected_parameters[param_name] = var.get()

            # Count selected (excluding vulnerability params from main count)
            strategy_selected = sum(1 for k, v in self.selected_parameters.items()
                                   if v and not k.startswith('_vuln_'))
            strategy_total = sum(1 for k in self.selected_parameters.keys()
                                if not k.startswith('_vuln_'))
            vuln_selected = sum(1 for k, v in self.selected_parameters.items()
                               if v and k.startswith('_vuln_'))

            if strategy_selected == strategy_total:
                status_text = "(All strategy params optimized"
            elif strategy_selected == 0:
                status_text = "(No strategy params"
            else:
                status_text = f"({strategy_selected}/{strategy_total} strategy params"

            if vuln_selected > 0:
                status_text += f", {vuln_selected} vuln params)"
            else:
                status_text += ")"

            self.param_status_label.config(text=status_text)
            self.log_message(f"Parameter selection updated: {strategy_selected}/{strategy_total} strategy + {vuln_selected} vulnerability")
            dialog.destroy()

        ttk.Button(action_frame, text="Save", command=save_and_close).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def log_message(self, message: str):
        """Log a message to the results text area."""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.results_text.update()

    def update_progress(self, stage: str, current: int, total: int):
        """Update progress bar and label."""
        # Cap current at total (Bayesian optimizer sometimes goes slightly over)
        current = min(current, total)
        percentage = (current / total * 100) if total > 0 else 0
        percentage = min(percentage, 100.0)  # Cap at 100%
        self.progress_var.set(f"{stage}: {current}/{total} ({percentage:.1f}%)")
        self.progress_bar['value'] = percentage
        self.root.update_idletasks()

    def start_optimization(self):
        """Start the optimization process."""
        # Validate inputs
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            messagebox.showerror("Error", "Please select a strategy")
            return

        selected_indices = self.securities_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one security")
            return

        selected_securities = [self.securities_listbox.get(i) for i in selected_indices]
        mode = self.mode_var.get()

        # Validate based on mode
        if mode == "single" and len(selected_securities) > 1:
            messagebox.showerror("Error", "Single mode only allows one security. Switch to Portfolio mode for multiple securities.")
            return

        if mode == "portfolio" and len(selected_securities) < 2:
            messagebox.showerror("Error", "Portfolio mode requires at least 2 securities.")
            return

        # Build confirmation message
        if mode == "single":
            confirm_msg = (
                f"Start SINGLE SECURITY optimization for:\n"
                f"Strategy: {strategy_name}\n"
                f"Security: {selected_securities[0]}\n\n"
                f"This may take a long time depending on data size and configuration."
            )
        else:
            confirm_msg = (
                f"Start PORTFOLIO optimization for:\n"
                f"Strategy: {strategy_name}\n"
                f"Securities: {', '.join(selected_securities)}\n"
                f"Capital Contention: {self.capital_contention_config.mode.value}\n\n"
                f"Note: Portfolio optimization optimizes parameters for the combined\n"
                f"portfolio performance with shared capital.\n\n"
                f"This may take a long time depending on data size and configuration."
            )

        if not messagebox.askyesno("Confirm Optimization", confirm_msg):
            return

        # Clear results
        self.results_text.delete('1.0', tk.END)

        # Disable optimize button, enable cancel
        self.optimize_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.is_running = True

        # Start optimization in background thread
        self.optimization_thread = threading.Thread(
            target=self.run_optimization,
            args=(strategy_name, selected_securities),
            daemon=True
        )
        self.optimization_thread.start()

    def cancel_optimization(self):
        """Cancel the running optimization."""
        if self.is_running:
            self.optimizer.cancel()
            self.sensitivity_analyzer.cancel()
            self.log_message("\n=== CANCELLATION REQUESTED ===")
            self.log_message("Waiting for current operation to complete...")

    def run_optimization(self, strategy_name: str, securities: List[str]):
        """
        Run optimization for selected strategy and securities.
        This runs in a background thread.

        Single mode:
        - Optimizes parameters for a single security

        Portfolio mode:
        - Optimizes parameters for the combined portfolio performance
        - All securities share capital with capital contention rules
        """
        try:
            strategy_class = self.STRATEGIES[strategy_name]
            mode = self.mode_var.get()

            # Apply GUI settings to optimizer config
            speed_mode = self.speed_mode_var.get()
            n_jobs = self.n_jobs_var.get()

            self.optimizer.config['bayesian_optimization']['speed_mode'] = speed_mode
            self.optimizer.config['bayesian_optimization']['n_jobs'] = n_jobs

            # Apply walk-forward window settings (ensure all are integers)
            train_days = int(self.training_period_var.get())
            test_days = int(self.testing_period_var.get())
            step_min = int(self.step_min_var.get())
            step_max = int(self.step_max_var.get())
            wf_mode = self.wf_mode_var.get()

            # Validate step sizes
            if step_min > step_max:
                step_min, step_max = step_max, step_min

            self.optimizer.config['walk_forward']['training_period_days'] = train_days
            self.optimizer.config['walk_forward']['testing_period_days'] = test_days
            self.optimizer.config['walk_forward']['step_size_min_days'] = step_min
            self.optimizer.config['walk_forward']['step_size_max_days'] = step_max
            self.optimizer.config['walk_forward']['mode'] = wf_mode

            # Get walk-forward mode enum
            walk_forward_mode = WalkForwardMode(wf_mode)

            self.log_message(f"Walk-Forward Mode: {wf_mode.upper()}")
            self.log_message(f"Walk-Forward Settings: Train={train_days} days, Test={test_days} days, Step={step_min}-{step_max} days")

            import platform
            is_windows = platform.system() == 'Windows'

            self.log_message(f"Speed Mode: {speed_mode.upper()}")
            if is_windows:
                self.log_message(f"CPU Cores: 1 (Windows - parallel processing not available)")
            else:
                self.log_message(f"CPU Cores: {n_jobs if n_jobs > 0 else 'All available'}")
                if n_jobs != 1:
                    self.log_message("Note: If parallel processing fails, will automatically fall back to serial processing")

            # Collect all results for combined report (portfolio mode)
            all_wf_results = {}  # symbol -> WalkForwardResults
            all_sensitivity_results = {}  # symbol -> SensitivityResults

            # Log mode
            self.log_message(f"\nOptimization Mode: {mode.upper()}")

            # Portfolio mode: Run combined optimization
            if mode == "portfolio":
                self.log_message(f"Capital Contention: {self.capital_contention_config.mode.value}")
                if self.capital_contention_config.mode == CapitalContentionMode.VULNERABILITY_SCORE:
                    vc = self.capital_contention_config.vulnerability_config
                    self.log_message(f"  - Immunity Days: {vc.immunity_days}")
                    self.log_message(f"  - Swap Threshold: {vc.swap_threshold}")
                self.log_message(f"\n*** PORTFOLIO OPTIMIZATION: {len(securities)} securities ***")
                self.log_message("Optimizing parameters for COMBINED portfolio performance with shared capital.\n")

                # Load data for all securities
                data_dict = {}
                for symbol in securities:
                    self.log_message(f"Loading data for {symbol}...")
                    try:
                        data = self.data_loader.load_csv(symbol)
                        data_dict[symbol] = data
                        self.log_message(f"  Loaded {len(data)} bars")
                    except Exception as e:
                        self.log_message(f"  ERROR: {e}")
                        continue

                if len(data_dict) < 2:
                    self.log_message("ERROR: Need at least 2 securities with valid data for portfolio optimization")
                    return

                # Run portfolio optimization
                self.log_message("\nStarting portfolio walk-forward optimization...")
                try:
                    wf_results = self.optimizer.optimize_portfolio(
                        strategy_class=strategy_class,
                        data_dict=data_dict,
                        capital_contention=self.capital_contention_config,
                        initial_capital=100000.0,
                        selected_params=self.selected_parameters if self.selected_parameters else None,
                        progress_callback=lambda stage, curr, total: self.root.after(
                            0, self.update_progress, stage, curr, total
                        ),
                        walk_forward_mode=walk_forward_mode
                    )

                    # Display results
                    self.log_message("\n" + "=" * 60)
                    self.log_message("PORTFOLIO WALK-FORWARD OPTIMIZATION RESULTS")
                    self.log_message("=" * 60)
                    self.log_message(f"Total Windows: {wf_results.total_windows}")
                    self.log_message(f"Windows Passed Constraints: {wf_results.windows_passed_constraints}")
                    self.log_message(f"Success Rate: {wf_results.windows_passed_constraints / wf_results.total_windows * 100:.1f}%")
                    self.log_message(f"\nAvg In-Sample Sortino: {wf_results.avg_in_sample_sortino:.4f}")
                    self.log_message(f"Avg Out-Sample Sortino: {wf_results.avg_out_sample_sortino:.4f}")
                    self.log_message(f"Sortino Degradation: {wf_results.avg_sortino_degradation_pct:.2f}%")
                    self.log_message(f"\nAvg In-Sample Sharpe: {wf_results.avg_in_sample_sharpe:.4f}")
                    self.log_message(f"Avg Out-Sample Sharpe: {wf_results.avg_out_sample_sharpe:.4f}")
                    self.log_message(f"Sharpe Degradation: {wf_results.avg_sharpe_degradation_pct:.2f}%")

                    self.log_message("\nRecommended Parameters:")
                    for param_name, param_value in wf_results.most_common_params.items():
                        min_val, max_val = wf_results.parameter_ranges[param_name]
                        self.log_message(f"  {param_name}: {param_value:.4f} (range: {min_val:.2f} - {max_val:.2f})")

                    # Generate report
                    self.log_message("\n\nGenerating portfolio optimization report...")
                    report_path = self.report_generator.generate_report(
                        wf_results=wf_results,
                        sensitivity_results=None  # Skip sensitivity for portfolio for now
                    )
                    self.log_message(f"Report saved to: {report_path}")

                except Exception as e:
                    self.log_message(f"ERROR during portfolio optimization: {e}")
                    logger.exception("Portfolio optimization failed")

                self.log_message(f"\n\n{'=' * 60}")
                self.log_message("PORTFOLIO OPTIMIZATION COMPLETE")
                self.log_message(f"{'=' * 60}")
                messagebox.showinfo("Complete", "Portfolio optimization completed!")
                return

            # Single mode: existing per-security optimization
            all_wf_results = {}
            all_sensitivity_results = {}

            for sec_idx, symbol in enumerate(securities):
                if not self.is_running:
                    self.log_message("Optimization cancelled by user")
                    break

                self.log_message(f"\n{'=' * 60}")
                self.log_message(f"OPTIMIZING: {symbol} ({sec_idx + 1}/{len(securities)})")
                self.log_message(f"{'=' * 60}\n")

                # Load data
                self.log_message(f"Loading data for {symbol}...")
                try:
                    data = self.data_loader.load_csv(symbol)
                    self.log_message(f"Loaded {len(data)} bars of data (from {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')})")

                    # Apply start date filter if specified
                    if self.use_start_date_var.get():
                        start_date_str = self.start_date_entry.get().strip()
                        if start_date_str and start_date_str != "YYYY-MM-DD":
                            try:
                                start_date = pd.to_datetime(start_date_str)
                                original_len = len(data)
                                oldest_available = data['date'].min()

                                # Filter data
                                data = data[data['date'] >= start_date].copy()

                                if len(data) == 0:
                                    self.log_message(f"WARNING: No data after {start_date_str}. Using all available data from {oldest_available.strftime('%Y-%m-%d')}")
                                    data = self.data_loader.load_csv(symbol)  # Reload original data
                                elif len(data) < original_len:
                                    self.log_message(f"Applied start date filter: Using data from {data['date'].min().strftime('%Y-%m-%d')} ({len(data)} bars)")
                                else:
                                    self.log_message(f"Start date {start_date_str} is before oldest available data ({oldest_available.strftime('%Y-%m-%d')}). Using all available data.")

                            except Exception as e:
                                self.log_message(f"WARNING: Invalid date format '{start_date_str}'. Using all available data. Error: {e}")

                except Exception as e:
                    self.log_message(f"ERROR: Failed to load data for {symbol}: {e}")
                    continue

                # Run walk-forward optimization
                self.log_message("\nStarting walk-forward optimization...")
                self.log_message(f"Optimizing {sum(self.selected_parameters.values())} parameters, " +
                               f"fixing {len(self.selected_parameters) - sum(self.selected_parameters.values())} parameters")

                try:
                    wf_results = self.optimizer.optimize(
                        strategy_class=strategy_class,
                        symbol=symbol,
                        data=data,
                        selected_params=self.selected_parameters if self.selected_parameters else None,
                        progress_callback=lambda stage, curr, total: self.root.after(
                            0, self.update_progress, stage, curr, total
                        ),
                        walk_forward_mode=walk_forward_mode
                    )

                    if not self.is_running:
                        break

                    # Display results
                    self.log_message("\n" + "=" * 60)
                    self.log_message("WALK-FORWARD OPTIMIZATION RESULTS")
                    self.log_message("=" * 60)
                    self.log_message(f"Total Windows: {wf_results.total_windows}")
                    self.log_message(f"Windows Passed Constraints: {wf_results.windows_passed_constraints}")
                    self.log_message(f"Success Rate: {wf_results.windows_passed_constraints / wf_results.total_windows * 100:.1f}%")
                    self.log_message(f"\nAvg In-Sample Sortino: {wf_results.avg_in_sample_sortino:.4f}")
                    self.log_message(f"Avg Out-Sample Sortino: {wf_results.avg_out_sample_sortino:.4f}")
                    self.log_message(f"Sortino Degradation: {wf_results.avg_sortino_degradation_pct:.2f}%")
                    self.log_message(f"\nAvg In-Sample Sharpe: {wf_results.avg_in_sample_sharpe:.4f}")
                    self.log_message(f"Avg Out-Sample Sharpe: {wf_results.avg_out_sample_sharpe:.4f}")
                    self.log_message(f"Sharpe Degradation: {wf_results.avg_sharpe_degradation_pct:.2f}%")

                    self.log_message("\nRecommended Parameters:")
                    for param_name, param_value in wf_results.most_common_params.items():
                        min_val, max_val = wf_results.parameter_ranges[param_name]
                        self.log_message(f"  {param_name}: {param_value:.4f} (range: {min_val:.2f} - {max_val:.2f})")

                except Exception as e:
                    self.log_message(f"ERROR during walk-forward optimization: {e}")
                    logger.exception("Walk-forward optimization failed")
                    continue

                # Run sensitivity analysis
                sensitivity_results = None
                if self.run_sensitivity_var.get() and self.is_running:
                    self.log_message("\n\nStarting sensitivity analysis...")
                    try:
                        sensitivity_results = self.sensitivity_analyzer.analyze(
                            strategy_class=strategy_class,
                            base_parameters=wf_results.most_common_params,
                            data=data,
                            symbol=symbol,
                            progress_callback=lambda curr, total: self.root.after(
                                0, self.update_progress, "Sensitivity Analysis", curr, total
                            )
                        )

                        if not self.is_running:
                            break

                        # Display sensitivity results
                        self.log_message("\n" + "=" * 60)
                        self.log_message("SENSITIVITY ANALYSIS RESULTS")
                        self.log_message("=" * 60)
                        self.log_message(f"Total Tests: {len(sensitivity_results.all_tests)}")
                        self.log_message(f"Overall Sortino Range: {sensitivity_results.overall_sortino_range_pct:.2f}%")
                        self.log_message(f"Overall Assessment: {'ROBUST' if sensitivity_results.is_overall_robust else 'SENSITIVE'}")

                        self.log_message("\nMost Robust Parameters:")
                        for param in sensitivity_results.most_robust_params:
                            self.log_message(f"  - {param}")

                        self.log_message("\nLeast Robust Parameters (require monitoring):")
                        for param in sensitivity_results.least_robust_params:
                            self.log_message(f"  - {param}")

                    except Exception as e:
                        self.log_message(f"ERROR during sensitivity analysis: {e}")
                        logger.exception("Sensitivity analysis failed")

                # Store results for combined report
                all_wf_results[symbol] = wf_results
                if sensitivity_results:
                    all_sensitivity_results[symbol] = sensitivity_results

                # Generate individual Excel report
                if self.is_running:
                    self.log_message("\n\nGenerating individual Excel report...")
                    try:
                        report_path = self.report_generator.generate_report(
                            wf_results=wf_results,
                            sensitivity_results=sensitivity_results
                        )
                        self.log_message(f"Report saved to: {report_path}")
                        self.log_message("\nOPTIMIZATION COMPLETE!")

                    except Exception as e:
                        self.log_message(f"ERROR generating report: {e}")
                        logger.exception("Report generation failed")

            # Generate COMBINED report for portfolio mode
            if self.is_running and mode == "portfolio" and len(all_wf_results) > 1:
                self.log_message(f"\n\n{'=' * 60}")
                self.log_message("GENERATING COMBINED PORTFOLIO REPORT")
                self.log_message(f"{'=' * 60}")
                try:
                    from Classes.Optimization.walk_forward_optimizer import MultiSecurityResults

                    # Create combined results
                    multi_results = MultiSecurityResults.from_individual_results(
                        strategy_name=strategy_name,
                        results_dict=all_wf_results
                    )

                    # Display combined summary
                    self.log_message(f"\nCombined Results Summary:")
                    self.log_message(f"  Securities Analyzed: {len(multi_results.securities)}")
                    self.log_message(f"  Total Windows: {multi_results.total_windows_all_securities}")
                    self.log_message(f"  Overall Success Rate: {multi_results.total_passed_all_securities / multi_results.total_windows_all_securities * 100:.1f}%")
                    self.log_message(f"  Securities with Positive OOS: {multi_results.securities_with_positive_oos}/{len(multi_results.securities)}")
                    self.log_message(f"\n  Combined Avg OOS Sortino: {multi_results.combined_avg_out_sample_sortino:.4f}")
                    self.log_message(f"  Combined Avg Sortino Degradation: {multi_results.combined_avg_sortino_degradation_pct:.2f}%")
                    self.log_message(f"\n  Best Security: {multi_results.best_security}")
                    self.log_message(f"  Worst Security: {multi_results.worst_security}")

                    self.log_message(f"\n  Consistent Parameters (across all securities):")
                    for param_name, value in multi_results.consistent_params.items():
                        consistency = multi_results.param_consistency_scores.get(param_name, 0)
                        self.log_message(f"    {param_name}: {value:.4f} (consistency: {consistency:.0f}%)")

                    # Generate combined report (uses enhanced portfolio report when available)
                    combined_report_path = self.report_generator.generate_portfolio_optimization_report(
                        multi_results=multi_results,
                        sensitivity_results_dict=all_sensitivity_results if all_sensitivity_results else None
                    )
                    self.log_message(f"\nCombined report saved to: {combined_report_path}")
                    if self.report_generator.is_enhanced_available():
                        self.log_message("  (Enhanced report with advanced visualizations)")

                except Exception as e:
                    self.log_message(f"ERROR generating combined report: {e}")
                    logger.exception("Combined report generation failed")

            if self.is_running:
                self.log_message(f"\n\n{'=' * 60}")
                self.log_message("OPTIMIZATION COMPLETE")
                self.log_message(f"{'=' * 60}")
                if mode == "portfolio" and len(all_wf_results) > 1:
                    self.log_message(f"Individual reports generated for {len(all_wf_results)} securities in portfolio")
                    self.log_message("Combined portfolio report also generated")
                messagebox.showinfo("Complete", "Optimization completed successfully!")

        except Exception as e:
            self.log_message(f"\n\nFATAL ERROR: {e}")
            logger.exception("Optimization failed")
            messagebox.showerror("Error", f"Optimization failed: {e}")

        finally:
            # Re-enable buttons
            self.root.after(0, self._reset_ui)

    def _reset_ui(self):
        """Reset UI state after optimization completes or is cancelled."""
        self.is_running = False
        self.optimize_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_var.set("Ready")
        self.progress_bar['value'] = 0


def main():
    """Main entry point for the optimization GUI."""
    root = tk.Tk()
    app = OptimizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
