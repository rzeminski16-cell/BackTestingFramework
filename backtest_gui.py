"""
Backtesting Framework GUI

A graphical user interface for running backtests with configurable parameters.

Features:
- Select securities (single or multiple with Select All button)
- Choose backtest mode:
  * Single Security: Test one security at a time
  * Portfolio: Run backtests on multiple securities with shared capital
- Select and configure strategies
- Configure commission settings (percentage or fixed)
- Configure capital contention settings for portfolio mode
- Create and manage baskets of securities
- Set date ranges
- Name backtests
- View results with detailed metrics
- Generate Excel reports
- Save trade logs

Portfolio Mode Features:
- Shared capital across all securities
- Capital contention handling (Default or Vulnerability Score mode)
- Basket management for quick security group selection
- Portfolio-level performance reporting
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import yaml

from Classes.Config.config import (
    BacktestConfig, PortfolioConfig, CommissionConfig,
    CommissionMode, OptimizationConfig
)
from Classes.Config.strategy_preset import StrategyParameterPreset
from Classes.Config.basket import Basket, BasketManager
from Classes.Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig
)
from Classes.Data.data_loader import DataLoader
from Classes.Data.security_registry import SecurityRegistry
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine, PortfolioBacktestResult
from Classes.Analysis.trade_logger import TradeLogger, PortfolioTradeLogger, LoggingPath
from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
from Classes.Analysis.portfolio_report_generator import PortfolioReportGenerator
from Classes.Optimization.optimizer import StrategyOptimizer
from Classes.GUI.basket_manager_dialog import BasketManagerDialog, VulnerabilityScoreConfigDialog

# Import available strategies
from strategies.alphatrend_strategy import AlphaTrendStrategy


class BacktestGUI:
    """Main GUI application for backtesting."""

    # Available strategies
    STRATEGIES = {
        'AlphaTrendStrategy': AlphaTrendStrategy
    }

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Backtesting Framework")
        self.root.geometry("1200x800")

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/processed_exports'))
        self.available_securities = self.data_loader.get_available_symbols()

        # Currency support
        self.security_registry = SecurityRegistry(Path('config/security_metadata.json'))
        self.currency_converter = CurrencyConverter(base_currency='GBP')
        try:
            self.currency_converter.load_rates_directory(Path('currency_rates/'))
        except Exception as e:
            print(f"Warning: Could not load currency rates: {e}")

        # Strategy parameters cache
        self.strategy_params = {}

        # Strategy preset manager
        self.preset_manager = StrategyParameterPreset()

        # Basket manager
        self.basket_manager = BasketManager()
        self.selected_basket: Optional[Basket] = None

        # Capital contention configuration
        self.capital_contention_config = CapitalContentionConfig.default_mode()
        self.vulnerability_config = VulnerabilityScoreConfig()

        # Load parameter configuration from optimization config
        self.param_config = self._load_param_config()

        # Create GUI components
        self.create_widgets()

    def _load_param_config(self) -> Dict[str, Any]:
        """Load parameter configuration from optimization config file."""
        config_path = Path("config/optimization_config.yaml")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('strategy_parameters', {})
        except Exception as e:
            print(f"Warning: Could not load parameter config: {e}")
            return {}

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # ===== Left Panel: Configuration =====
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        row = 0

        # Backtest Name
        ttk.Label(config_frame, text="Backtest Name:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.backtest_name_var = tk.StringVar(value="test_1")
        ttk.Entry(config_frame, textvariable=self.backtest_name_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Backtest Mode
        ttk.Label(config_frame, text="Backtest Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="single")
        mode_frame = ttk.Frame(config_frame)
        mode_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
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
        for symbol in self.available_securities:
            self.securities_listbox.insert(tk.END, symbol)

        # Selection buttons
        selection_buttons_frame = ttk.Frame(securities_container)
        selection_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(selection_buttons_frame, text="Select All",
                  command=self.select_all_securities).pack(side=tk.LEFT, padx=2)
        ttk.Button(selection_buttons_frame, text="Deselect All",
                  command=self.deselect_all_securities).pack(side=tk.LEFT, padx=2)

        row += 1

        # Strategy Selection
        ttk.Label(config_frame, text="Strategy:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(
            config_frame,
            textvariable=self.strategy_var,
            values=list(self.STRATEGIES.keys()),
            state="readonly",
            width=30
        )
        strategy_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        if self.STRATEGIES:
            strategy_combo.current(0)
            self.on_strategy_change(None)
        row += 1

        # Strategy Parameters Button
        ttk.Button(config_frame, text="Configure Strategy Parameters",
                  command=self.open_strategy_params_window).grid(
            row=row, column=0, columnspan=2, pady=10
        )
        row += 1

        # Capital
        ttk.Label(config_frame, text="Initial Capital:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.capital_var = tk.StringVar(value="100000")
        ttk.Entry(config_frame, textvariable=self.capital_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Commission Mode
        ttk.Label(config_frame, text="Commission Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.commission_mode_var = tk.StringVar(value="percentage")
        comm_frame = ttk.Frame(config_frame)
        comm_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(comm_frame, text="Percentage", variable=self.commission_mode_var,
                       value="percentage").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(comm_frame, text="Fixed", variable=self.commission_mode_var,
                       value="fixed").pack(side=tk.LEFT, padx=5)
        row += 1

        # Commission Value
        ttk.Label(config_frame, text="Commission Value:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.commission_value_var = tk.StringVar(value="0.001")
        ttk.Entry(config_frame, textvariable=self.commission_value_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Label(config_frame, text="(e.g., 0.001 for 0.1% or 3.0 for £3)",
                 font=('Arial', 8)).grid(row=row+1, column=1, sticky=tk.W)
        row += 2

        # Slippage
        ttk.Label(config_frame, text="Slippage (%):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.slippage_var = tk.StringVar(value="0.1")
        ttk.Entry(config_frame, textvariable=self.slippage_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Label(config_frame, text="(e.g., 0.1 for 0.1% slippage)",
                 font=('Arial', 8)).grid(row=row+1, column=1, sticky=tk.W)
        row += 2

        # Date Range
        ttk.Label(config_frame, text="Start Date (optional):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.start_date_var = tk.StringVar(value="")
        ttk.Entry(config_frame, textvariable=self.start_date_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Label(config_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(
            row=row+1, column=1, sticky=tk.W
        )
        row += 2

        ttk.Label(config_frame, text="End Date (optional):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.end_date_var = tk.StringVar(value="")
        ttk.Entry(config_frame, textvariable=self.end_date_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Label(config_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(
            row=row+1, column=1, sticky=tk.W
        )
        row += 2

        # Excel Report Generation
        self.generate_excel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Generate Excel Report",
                       variable=self.generate_excel_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5
        )
        row += 1

        # Portfolio-specific settings (initially hidden)
        self.portfolio_frame = ttk.LabelFrame(config_frame, text="Portfolio/Basket Settings", padding="5")
        self.portfolio_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
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

        # Run Button
        ttk.Button(config_frame, text="Run Backtest", command=self.run_backtest,
                  style='Accent.TButton').grid(
            row=row, column=0, columnspan=2, pady=20, ipadx=20, ipady=10
        )

        # ===== Right Panel: Results =====
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=70,
            height=40,
            font=('Courier', 9)
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.grid(row=0, column=1, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def on_mode_change(self):
        """Handle backtest mode change."""
        mode = self.mode_var.get()
        if mode == "portfolio":
            # Portfolio/Basket mode: show portfolio settings
            self.portfolio_frame.grid()
            self.securities_listbox.config(selectmode=tk.MULTIPLE)
        else:
            # Single mode
            self.portfolio_frame.grid_remove()
            self.securities_listbox.config(selectmode=tk.SINGLE)

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
                # Select the new basket in dropdown
                self.basket_var.set(basket.name)
                # Update securities listbox
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

    def on_strategy_change(self, event):
        """Handle strategy selection change."""
        strategy_name = self.strategy_var.get()
        if strategy_name not in self.STRATEGIES:
            return

        # Initialize default parameters for this strategy
        strategy_class = self.STRATEGIES[strategy_name]

        # Get default parameters from strategy
        if strategy_name == 'AlphaTrendStrategy':
            self.strategy_params[strategy_name] = {
                'volume_short_ma': 4,
                'volume_long_ma': 30,
                'volume_alignment_window': 14,
                'stop_loss_percent': 0.0,
                'atr_stop_loss_multiple': 2.5,
                'grace_period_bars': 14,
                'momentum_gain_pct': 2.0,
                'momentum_lookback': 7,
                'risk_percent': 2.0
            }

    def open_strategy_params_window(self):
        """Open window to configure strategy parameters with preset management."""
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            messagebox.showwarning("No Strategy", "Please select a strategy first.")
            return

        if strategy_name not in self.strategy_params:
            self.on_strategy_change(None)

        # Create parameter window
        param_window = tk.Toplevel(self.root)
        param_window.title(f"Configure {strategy_name} Parameters")
        param_window.geometry("500x700")

        # ===== Preset Management Section =====
        preset_frame = ttk.LabelFrame(param_window, text="Parameter Presets", padding="10")
        preset_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=10)

        # Load available presets
        presets = self.preset_manager.list_presets(strategy_name)
        preset_names = [p['preset_name'] for p in presets]

        ttk.Label(preset_frame, text="Load Preset:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=preset_var, values=preset_names, width=20, state='readonly')
        preset_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        def load_preset():
            """Load selected preset into parameter fields."""
            preset_name = preset_var.get()
            if not preset_name:
                messagebox.showwarning("No Preset", "Please select a preset to load.")
                return

            params = self.preset_manager.get_preset_parameters(strategy_name, preset_name)
            if params:
                # Update parameter vars with preset values
                for param_name, param_value in params.items():
                    if param_name in param_vars:
                        param_vars[param_name].set(str(param_value))
                messagebox.showinfo("Success", f"Preset '{preset_name}' loaded!")
            else:
                messagebox.showerror("Error", f"Failed to load preset '{preset_name}'")

        def save_preset():
            """Save current parameters as a new preset."""
            # Ask for preset name
            preset_name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
            if not preset_name:
                return

            # Ask for description (optional)
            description = tk.simpledialog.askstring("Preset Description", "Enter description (optional):", initialvalue="")

            # Get current parameter values
            current_params = {}
            try:
                for param_name, var in param_vars.items():
                    value = var.get()
                    original_value = self.strategy_params[strategy_name][param_name]

                    # Convert to appropriate type
                    if isinstance(original_value, str):
                        current_params[param_name] = value
                    elif isinstance(original_value, float):
                        current_params[param_name] = float(value)
                    elif isinstance(original_value, int):
                        current_params[param_name] = int(value)
                    else:
                        if '.' in value:
                            current_params[param_name] = float(value)
                        else:
                            try:
                                current_params[param_name] = int(value)
                            except ValueError:
                                current_params[param_name] = value

                # Save preset
                self.preset_manager.save_preset(strategy_name, preset_name, current_params, description or "")

                # Refresh preset list
                presets = self.preset_manager.list_presets(strategy_name)
                preset_names = [p['preset_name'] for p in presets]
                preset_combo['values'] = preset_names

                messagebox.showinfo("Success", f"Preset '{preset_name}' saved!")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid parameter value: {e}")

        def delete_preset():
            """Delete selected preset."""
            preset_name = preset_var.get()
            if not preset_name:
                messagebox.showwarning("No Preset", "Please select a preset to delete.")
                return

            # Confirm deletion
            if messagebox.askyesno("Confirm Delete", f"Delete preset '{preset_name}'?"):
                if self.preset_manager.delete_preset(strategy_name, preset_name):
                    # Refresh preset list
                    presets = self.preset_manager.list_presets(strategy_name)
                    preset_names = [p['preset_name'] for p in presets]
                    preset_combo['values'] = preset_names
                    preset_var.set('')
                    messagebox.showinfo("Success", f"Preset '{preset_name}' deleted!")
                else:
                    messagebox.showerror("Error", f"Failed to delete preset '{preset_name}'")

        # Preset buttons
        ttk.Button(preset_frame, text="Load", command=load_preset).grid(row=0, column=2, padx=5)
        ttk.Button(preset_frame, text="Save As...", command=save_preset).grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(preset_frame, text="Delete", command=delete_preset).grid(row=1, column=2, padx=5)

        # ===== Parameter Configuration Section =====
        param_label_frame = ttk.LabelFrame(param_window, text="Parameters", padding="10")
        param_label_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Create scrollable frame for parameters
        canvas = tk.Canvas(param_label_frame, height=350)
        scrollbar = ttk.Scrollbar(param_label_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Get parameter spec from config
        strategy_spec = self.param_config.get(strategy_name, {})

        # Parameter entries
        params = self.strategy_params[strategy_name]
        param_vars = {}

        # Helper function to add section header
        def add_section_header(text, row_num):
            separator = ttk.Separator(scrollable_frame, orient='horizontal')
            separator.grid(row=row_num, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=(15, 5))
            header = ttk.Label(scrollable_frame, text=text, font=('TkDefaultFont', 10, 'bold'),
                              foreground='#2E5994')
            header.grid(row=row_num + 1, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(0, 5))
            return row_num + 2

        # Helper function to add a parameter row with range info
        def add_param_row(param_name, param_value, row_num):
            # Get spec info if available
            spec = strategy_spec.get(param_name, {})
            min_val = spec.get('min', '')
            max_val = spec.get('max', '')
            param_type = spec.get('type', 'float')

            # Create range string
            if min_val != '' and max_val != '':
                if param_type == 'int':
                    range_str = f"[{int(min_val)} - {int(max_val)}]"
                else:
                    range_str = f"[{min_val:.2f} - {max_val:.2f}]"
            else:
                range_str = ""

            # Parameter name label
            name_label = ttk.Label(scrollable_frame, text=f"{param_name}:")
            name_label.grid(row=row_num, column=0, sticky=tk.W, padx=10, pady=3)

            # Entry field
            var = tk.StringVar(value=str(param_value))
            param_vars[param_name] = var
            entry = ttk.Entry(scrollable_frame, textvariable=var, width=15)
            entry.grid(row=row_num, column=1, sticky=(tk.W, tk.E), padx=5, pady=3)

            # Range info label
            if range_str:
                range_label = ttk.Label(scrollable_frame, text=range_str, font=('TkDefaultFont', 8),
                                       foreground='#666666')
                range_label.grid(row=row_num, column=2, sticky=tk.W, padx=5, pady=3)

            return row_num + 1

        # Categorize parameters
        indicator_params = {}
        entry_params = {}
        exit_params = {}
        other_params = {}

        for param_name, param_value in params.items():
            name_lower = param_name.lower()
            if any(kw in name_lower for kw in ['entry', 'buy', 'signal']):
                entry_params[param_name] = param_value
            elif any(kw in name_lower for kw in ['exit', 'sell', 'stop', 'take', 'trailing', 'grace', 'momentum']):
                exit_params[param_name] = param_value
            elif any(kw in name_lower for kw in ['period', 'length', 'lookback', 'window', 'multiplier', 'atr', 'volume', 'risk']):
                indicator_params[param_name] = param_value
            else:
                other_params[param_name] = param_value

        row = 0

        # Indicator/Core parameters section
        if indicator_params:
            row = add_section_header("Indicator & Position Settings", row)
            for param_name, param_value in indicator_params.items():
                row = add_param_row(param_name, param_value, row)

        # Entry parameters section
        if entry_params:
            row = add_section_header("Entry Settings", row)
            for param_name, param_value in entry_params.items():
                row = add_param_row(param_name, param_value, row)

        # Exit parameters section
        if exit_params:
            row = add_section_header("Exit Settings", row)
            for param_name, param_value in exit_params.items():
                row = add_param_row(param_name, param_value, row)

        # Other parameters section
        if other_params:
            row = add_section_header("Other Parameters", row)
            for param_name, param_value in other_params.items():
                row = add_param_row(param_name, param_value, row)

        # ===== Action Buttons =====
        action_frame = ttk.Frame(param_window)
        action_frame.grid(row=2, column=0, columnspan=2, pady=20)

        # Save button (apply parameters without closing)
        def save_params():
            try:
                for param_name, var in param_vars.items():
                    value = var.get()
                    # Get original parameter type to preserve it
                    original_value = params[param_name]

                    # Try to convert to appropriate type
                    if isinstance(original_value, str):
                        # Keep as string
                        self.strategy_params[strategy_name][param_name] = value
                    elif isinstance(original_value, float):
                        self.strategy_params[strategy_name][param_name] = float(value)
                    elif isinstance(original_value, int):
                        self.strategy_params[strategy_name][param_name] = int(value)
                    else:
                        # Fallback: auto-detect type
                        if '.' in value:
                            self.strategy_params[strategy_name][param_name] = float(value)
                        else:
                            try:
                                self.strategy_params[strategy_name][param_name] = int(value)
                            except ValueError:
                                self.strategy_params[strategy_name][param_name] = value
                messagebox.showinfo("Success", "Parameters saved!")
                param_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid parameter value: {e}")

        ttk.Button(action_frame, text="Apply & Close", command=save_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=param_window.destroy).pack(side=tk.LEFT, padx=5)

    def get_selected_securities(self) -> List[str]:
        """Get list of selected securities."""
        selection = self.securities_listbox.curselection()
        return [self.securities_listbox.get(i) for i in selection]

    def select_all_securities(self):
        """Select all securities in the listbox."""
        self.securities_listbox.selection_set(0, tk.END)

    def deselect_all_securities(self):
        """Deselect all securities in the listbox."""
        self.securities_listbox.selection_clear(0, tk.END)

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str.strip():
            return None
        try:
            return datetime.strptime(date_str.strip(), '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

    def run_backtest(self):
        """Run the backtest with configured parameters."""
        try:
            # Validate inputs
            securities = self.get_selected_securities()
            if not securities:
                messagebox.showerror("Error", "Please select at least one security.")
                return

            strategy_name = self.strategy_var.get()
            if not strategy_name:
                messagebox.showerror("Error", "Please select a strategy.")
                return

            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.status_var.set("Running backtest...")
            self.root.update()

            # Get configuration
            mode = self.mode_var.get()
            capital = float(self.capital_var.get())
            commission_mode = CommissionMode.PERCENTAGE if self.commission_mode_var.get() == "percentage" else CommissionMode.FIXED
            commission_value = float(self.commission_value_var.get())
            slippage_percent = float(self.slippage_var.get())

            commission = CommissionConfig(mode=commission_mode, value=commission_value)

            start_date = self.parse_date(self.start_date_var.get())
            end_date = self.parse_date(self.end_date_var.get())

            # Create strategy instance
            strategy_class = self.STRATEGIES[strategy_name]
            strategy_params = self.strategy_params.get(strategy_name, {})
            strategy = strategy_class(**strategy_params)

            # Generate backtest name
            user_name = self.backtest_name_var.get().strip()
            full_backtest_name = f"{strategy_name}_{user_name}"

            # Run backtest based on mode
            if mode == "single":
                self.run_single_backtest(
                    securities[0], strategy, capital, commission,
                    start_date, end_date, full_backtest_name, slippage_percent
                )
            else:
                self.run_portfolio_backtest(
                    securities, strategy, capital, commission,
                    start_date, end_date, full_backtest_name, slippage_percent
                )

            self.status_var.set("Backtest completed!")

        except Exception as e:
            messagebox.showerror("Error", f"Backtest failed: {str(e)}")
            self.status_var.set("Error!")
            import traceback
            traceback.print_exc()

    def update_progress(self, current: int, total: int):
        """Update progress bar and label."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_var.set(percentage)
            self.progress_label.config(text=f"{current}/{total} ({percentage}%)")
            self.root.update_idletasks()  # Force GUI update

    def reset_progress(self):
        """Reset progress bar."""
        self.progress_var.set(0)
        self.progress_label.config(text="")
        self.root.update_idletasks()

    def run_single_backtest(self, symbol: str, strategy, capital: float,
                          commission: CommissionConfig, start_date, end_date,
                          backtest_name: str, slippage_percent: float):
        """Run single security backtest."""
        # Configure backtest
        config = BacktestConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            slippage_percent=slippage_percent
        )

        # Load data
        self.log_result(f"Loading data for {symbol}...")
        data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
        self.log_result(f"Loaded {len(data)} bars\n")

        # Run backtest
        self.log_result(f"Running backtest: {backtest_name}")
        self.log_result(f"Strategy: {strategy}")
        self.log_result(f"Security: {symbol}")
        self.log_result(f"Capital: ${capital:,.2f}\n")

        # Reset progress bar
        self.reset_progress()
        self.status_var.set(f"Processing {len(data)} bars...")

        # Create engine with currency support
        engine = SingleSecurityEngine(
            config=config,
            currency_converter=self.currency_converter,
            security_registry=self.security_registry
        )
        result = engine.run(symbol, data, strategy, progress_callback=self.update_progress)

        # Reset progress bar
        self.reset_progress()

        # Display results
        self.display_result(symbol, result, backtest_name)

    def run_portfolio_backtest(self, symbols: List[str], strategy, capital: float,
                              commission: CommissionConfig, start_date, end_date,
                              backtest_name: str, slippage_percent: float):
        """Run portfolio backtest with shared capital and capital contention."""
        # Get basket name if using a basket
        basket_name = self.selected_basket.name if self.selected_basket else None

        # Configure portfolio backtest
        config = PortfolioConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            capital_contention=self.capital_contention_config,
            slippage_percent=slippage_percent,
            basket_name=basket_name
        )

        self.log_result(f"Running PORTFOLIO backtest: {backtest_name}")
        self.log_result(f"Strategy: {strategy}")
        self.log_result(f"Securities: {', '.join(symbols)}")
        self.log_result(f"Shared Capital: ${capital:,.2f}")
        self.log_result(f"Capital Contention: {self.capital_contention_config.mode.value}")
        if self.capital_contention_config.mode == CapitalContentionMode.VULNERABILITY_SCORE:
            vc = self.capital_contention_config.vulnerability_config
            self.log_result(f"  - Immunity Days: {vc.immunity_days}")
            self.log_result(f"  - Swap Threshold: {vc.swap_threshold}")
        self.log_result("=" * 70)

        # Load data for all securities
        data_dict = {}
        for symbol in symbols:
            try:
                data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
                data_dict[symbol] = data
                self.log_result(f"Loaded {symbol}: {len(data)} bars")
            except Exception as e:
                self.log_result(f"WARNING: Could not load {symbol}: {e}")

        if not data_dict:
            raise ValueError("No data loaded for any security")

        self.log_result("")

        # Create portfolio engine
        engine = PortfolioEngine(
            config=config,
            currency_converter=self.currency_converter,
            security_registry=self.security_registry
        )

        # Run portfolio backtest
        self.status_var.set("Running portfolio backtest...")
        self.reset_progress()

        result = engine.run(data_dict, strategy, progress_callback=self.update_progress)

        # Reset progress bar
        self.reset_progress()

        # Display portfolio results
        self.display_portfolio_results(result, backtest_name, strategy.get_parameters())

    def display_portfolio_results(self, result: PortfolioBacktestResult, backtest_name: str,
                                  strategy_params: Dict):
        """Display portfolio backtest results."""
        self.log_result("\n" + "=" * 70)
        self.log_result("PORTFOLIO RESULTS")
        self.log_result("=" * 70)

        # Overall metrics
        self.log_result(f"\nInitial Capital:     ${result.config.initial_capital:,.2f}")
        self.log_result(f"Final Equity:        ${result.final_equity:,.2f}")
        self.log_result(f"Total Return:        ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")

        # Per-symbol breakdown
        self.log_result("\n--- Per-Security Performance ---")
        total_trades = 0
        for symbol, sym_result in result.symbol_results.items():
            num_trades = len(sym_result.trades)
            total_trades += num_trades
            wins = len([t for t in sym_result.trades if t.pl > 0])
            win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
            self.log_result(f"  {symbol}: {num_trades} trades, "
                          f"Win Rate: {win_rate:.1f}%, "
                          f"P/L: ${sym_result.total_return:,.2f}")

        self.log_result(f"\nTotal Trades: {total_trades}")

        # Signal rejections
        if result.signal_rejections:
            self.log_result(f"\n--- Signal Rejections: {len(result.signal_rejections)} ---")
            rejection_summary = {}
            for r in result.signal_rejections:
                rejection_summary[r.symbol] = rejection_summary.get(r.symbol, 0) + 1
            for symbol, count in sorted(rejection_summary.items(), key=lambda x: -x[1])[:5]:
                self.log_result(f"  {symbol}: {count} rejections")

        # Vulnerability swaps
        if result.vulnerability_swaps:
            self.log_result(f"\n--- Vulnerability Swaps: {len(result.vulnerability_swaps)} ---")
            for swap in result.vulnerability_swaps[:5]:
                self.log_result(f"  {swap.date.strftime('%Y-%m-%d')}: "
                              f"{swap.closed_symbol} (score: {swap.closed_score:.1f}) -> {swap.new_symbol}")
            if len(result.vulnerability_swaps) > 5:
                self.log_result(f"  ... and {len(result.vulnerability_swaps) - 5} more")

        self.log_result("=" * 70)

        # Log portfolio results
        basket_name = result.config.basket_name
        portfolio_logger = PortfolioTradeLogger(backtest_name, basket_name)
        logged_files = portfolio_logger.log_portfolio_result(result, strategy_params)
        self.log_result(f"\nResults logged to: {portfolio_logger.base_dir}")

        # Generate portfolio report if enabled
        if self.generate_excel_var.get():
            try:
                report_gen = PortfolioReportGenerator(portfolio_logger.reports_dir)
                report_path = report_gen.generate_portfolio_report(result)
                self.log_result(f"Portfolio report: {report_path}")
            except Exception as e:
                self.log_result(f"Warning: Could not generate portfolio report: {e}")
                import traceback
                traceback.print_exc()

    def display_result(self, symbol: str, result, backtest_name: str):
        """Display single backtest result."""
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_metrics(result)

        # Display metrics
        self.log_result("=" * 70)
        self.log_result(f"RESULTS: {symbol}")
        self.log_result("=" * 70)
        self.log_result(f"Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        self.log_result(f"Final Equity:        ${metrics['final_equity']:,.2f}")
        self.log_result(f"Number of Trades:    {metrics['num_trades']}")
        self.log_result(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
        self.log_result(f"Winning Trades:      {metrics['num_wins']}")
        self.log_result(f"Losing Trades:       {metrics['num_losses']}")
        self.log_result(f"Average Win:         ${metrics['avg_win']:,.2f}")
        self.log_result(f"Average Loss:        ${metrics['avg_loss']:,.2f}")
        self.log_result(f"Largest Win:         ${metrics['largest_win']:,.2f}")
        self.log_result(f"Largest Loss:        ${metrics['largest_loss']:,.2f}")
        self.log_result(f"Profit Factor:       {metrics['profit_factor']:.2f}")
        self.log_result(f"Avg Trade Duration:  {metrics['avg_trade_duration']:.1f} days")
        self.log_result(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        self.log_result(f"Max Drawdown:        ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        self.log_result("=" * 70)

        # Save trade log
        logger = TradeLogger(Path('logs') / backtest_name)
        logger.log_trades(symbol, backtest_name, result.trades, result.strategy_params)
        self.log_result(f"\nTrade log saved to: logs/{backtest_name}/{backtest_name}_{symbol}_trades.csv")
        if result.strategy_params:
            self.log_result(f"Strategy parameters saved to: logs/{backtest_name}/{backtest_name}_{symbol}_parameters.json")

        # Generate Excel report if enabled
        if self.generate_excel_var.get():
            try:
                self.log_result("\nGenerating Excel report...")
                excel_generator = ExcelReportGenerator(
                    output_directory=Path('logs') / backtest_name / 'reports',
                    initial_capital=float(self.capital_var.get()),
                    risk_free_rate=0.02,
                    benchmark_name="S&P 500"
                )
                report_path = excel_generator.generate_report(
                    result=result,
                    filename=f"{backtest_name}_{symbol}_report.xlsx"
                )
                self.log_result(f"✓ Excel report saved to: {report_path}")
            except Exception as e:
                self.log_result(f"⚠ Excel report generation failed: {str(e)}")

    def log_result(self, message: str):
        """Log message to results text area."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update()


def main():
    """Main entry point for GUI."""
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    style.theme_use('clam')

    app = BacktestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
