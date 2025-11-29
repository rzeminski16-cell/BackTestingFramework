"""
Backtesting Framework GUI

A graphical user interface for running backtests with configurable parameters.

Features:
- Select securities (single or multiple with Select All button)
- Choose backtest mode:
  * Single Security: Test one security at a time
  * Batch (Individual Tests): Run separate backtests on multiple securities
- Select and configure strategies
- Configure commission settings
- Set date ranges
- Name backtests
- View results with detailed metrics
- Generate Excel reports:
  * Single mode: Individual report per security
  * Batch mode: Individual reports + comprehensive summary report with correlation analysis
- Save trade logs

Batch Mode Reporting:
- Creates organized folder structure: batch_reports/individual/ for each security
- Generates batch summary report with:
  * Aggregate performance across all securities
  * Correlation analysis (returns, drawdowns)
  * Comparative performance metrics
  * Risk analysis
  * Visual comparisons
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from Classes.Config.config import (
    BacktestConfig, PortfolioConfig, CommissionConfig,
    CommissionMode, OptimizationConfig
)
from Classes.Config.strategy_preset import StrategyParameterPreset
from Classes.Data.data_loader import DataLoader
from Classes.Data.security_registry import SecurityRegistry
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Analysis.trade_logger import TradeLogger
from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
from Classes.Analysis.batch_summary_report import BatchSummaryReportGenerator
from Classes.Optimization.optimizer import StrategyOptimizer

# Import available strategies
from strategies.examples import (
    AdvancedTrailingStopStrategy,
    PartialExitStrategy
)
from strategies.alphatrend_strategy import AlphaTrendStrategy


class BacktestGUI:
    """Main GUI application for backtesting."""

    # Available strategies
    STRATEGIES = {
        'AdvancedTrailingStopStrategy': AdvancedTrailingStopStrategy,
        'PartialExitStrategy': PartialExitStrategy,
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
        ttk.Radiobutton(mode_frame, text="Batch (Individual Tests)", variable=self.mode_var,
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
        self.portfolio_frame = ttk.LabelFrame(config_frame, text="Portfolio Settings", padding="5")
        self.portfolio_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.portfolio_frame.grid_remove()  # Hide initially

        # Max Positions
        ttk.Label(self.portfolio_frame, text="Max Positions:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_positions_var = tk.StringVar(value="3")
        ttk.Entry(self.portfolio_frame, textvariable=self.max_positions_var, width=10).grid(
            row=0, column=1, sticky=tk.W, pady=2
        )

        # Position Size Limit
        ttk.Label(self.portfolio_frame, text="Position Size Limit:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.position_limit_var = tk.StringVar(value="0.3")
        ttk.Entry(self.portfolio_frame, textvariable=self.position_limit_var, width=10).grid(
            row=1, column=1, sticky=tk.W, pady=2
        )

        # Total Allocation Limit
        ttk.Label(self.portfolio_frame, text="Total Allocation Limit:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.total_allocation_var = tk.StringVar(value="0.9")
        ttk.Entry(self.portfolio_frame, textvariable=self.total_allocation_var, width=10).grid(
            row=2, column=1, sticky=tk.W, pady=2
        )

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
            # Batch mode: multiple securities, individual tests
            # No portfolio-specific settings needed
            self.portfolio_frame.grid_remove()
            self.securities_listbox.config(selectmode=tk.MULTIPLE)
        else:
            # Single mode
            self.portfolio_frame.grid_remove()
            self.securities_listbox.config(selectmode=tk.SINGLE)

    def on_strategy_change(self, event):
        """Handle strategy selection change."""
        strategy_name = self.strategy_var.get()
        if strategy_name not in self.STRATEGIES:
            return

        # Initialize default parameters for this strategy
        strategy_class = self.STRATEGIES[strategy_name]

        # Get default parameters from strategy
        if strategy_name == 'AdvancedTrailingStopStrategy':
            self.strategy_params[strategy_name] = {
                'sma_period': 200,
                'ema_period': 14,
                'rsi_period': 14,
                'rsi_threshold': 70,
                'position_size': 0.25,
                'initial_stop_pct': 0.08,
                'breakeven_profit_pct': 0.05
            }
        elif strategy_name == 'PartialExitStrategy':
            self.strategy_params[strategy_name] = {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'position_size': 0.3,
                'first_target_pct': 0.10,
                'second_target_pct': 0.20,
                'stop_loss_pct': 0.06
            }
        elif strategy_name == 'AlphaTrendStrategy':
            self.strategy_params[strategy_name] = {
                'atr_multiplier': 1.0,
                'common_period': 14,
                'source': 'close',
                'smoothing_length': 3,
                'percentile_period': 100,
                'volume_short_ma': 4,
                'volume_long_ma': 30,
                'volume_alignment_window': 14,
                'signal_lookback': 9,
                'exit_ema_period': 50,
                'stop_atr_multiplier': 2.5,
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
        canvas = tk.Canvas(param_label_frame, height=300)
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

        # Parameter entries
        params = self.strategy_params[strategy_name]
        param_vars = {}

        row = 0
        for param_name, param_value in params.items():
            ttk.Label(scrollable_frame, text=f"{param_name}:").grid(
                row=row, column=0, sticky=tk.W, padx=10, pady=5
            )
            var = tk.StringVar(value=str(param_value))
            param_vars[param_name] = var
            ttk.Entry(scrollable_frame, textvariable=var, width=20).grid(
                row=row, column=1, sticky=(tk.W, tk.E), padx=10, pady=5
            )
            row += 1

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
                    start_date, end_date, full_backtest_name
                )
            else:
                self.run_portfolio_backtest(
                    securities, strategy, capital, commission,
                    start_date, end_date, full_backtest_name
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
                          backtest_name: str):
        """Run single security backtest."""
        # Configure backtest
        config = BacktestConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date
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
                              backtest_name: str):
        """Run batch backtest - individual backtests for each selected security."""
        # Configure backtest (same config for all securities)
        config = BacktestConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date
        )

        self.log_result(f"Running batch backtest: {backtest_name}")
        self.log_result(f"Strategy: {strategy}")
        self.log_result(f"Securities: {', '.join(symbols)}")
        self.log_result(f"Capital per security: ${capital:,.2f}\n")
        self.log_result("=" * 70)

        # Run individual backtest for each security
        results = {}
        total_securities = len(symbols)

        for idx, symbol in enumerate(symbols, 1):
            try:
                self.log_result(f"\n[{idx}/{total_securities}] Testing {symbol}...")
                self.status_var.set(f"Testing {symbol} ({idx}/{total_securities})...")

                # Load data
                data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
                self.log_result(f"  Loaded {len(data)} bars")

                # Reset progress bar for this security
                self.reset_progress()

                # Create engine with currency support
                engine = SingleSecurityEngine(
                    config=config,
                    currency_converter=self.currency_converter,
                    security_registry=self.security_registry
                )

                # Run backtest
                result = engine.run(symbol, data, strategy, progress_callback=self.update_progress)
                results[symbol] = result

                # Quick summary
                self.log_result(f"  ✓ Completed: {result.num_trades} trades, "
                              f"Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")

            except Exception as e:
                self.log_result(f"  ✗ FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Reset progress bar
        self.reset_progress()

        if not results:
            raise ValueError("No backtests completed successfully")

        # Display aggregate results
        self.display_batch_results(results, backtest_name, config.initial_capital)

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

    def display_batch_results(self, results: Dict, backtest_name: str, initial_capital: float):
        """Display batch backtest results - individual tests for multiple securities."""
        self.log_result("\n" + "=" * 70)
        self.log_result("BATCH BACKTEST RESULTS (Individual Tests)")
        self.log_result("=" * 70)

        # Collect metrics for all securities
        all_metrics = []
        for symbol, result in results.items():
            metrics = PerformanceMetrics.calculate_metrics(result)
            metrics['symbol'] = symbol
            all_metrics.append(metrics)

        # Sort by total return descending
        all_metrics.sort(key=lambda x: x['total_return'], reverse=True)

        # Display summary table
        self.log_result(f"\n{'Symbol':<10} {'Trades':<8} {'Win Rate':<10} {'P/L':<15} {'Return %':<12} {'Sharpe':<8}")
        self.log_result("-" * 70)

        total_pl = 0
        total_trades = 0
        winning_securities = 0

        for m in all_metrics:
            total_pl += m['total_return']
            total_trades += m['num_trades']
            if m['total_return'] > 0:
                winning_securities += 1

            self.log_result(
                f"{m['symbol']:<10} {m['num_trades']:<8} "
                f"{m['win_rate']*100:>8.1f}% "
                f"${m['total_return']:>12,.2f} {m['total_return_pct']:>10.2f}% "
                f"{m['sharpe_ratio']:>6.2f}"
            )

        self.log_result("-" * 70)

        # Calculate aggregate metrics
        avg_return = total_pl / len(results)
        avg_return_pct = (avg_return / initial_capital) * 100
        num_securities = len(results)

        self.log_result(
            f"{'TOTAL':<10} {total_trades:<8} "
            f"{'':>9} "
            f"${total_pl:>12,.2f} {(total_pl/(initial_capital*num_securities))*100:>10.2f}%"
        )
        self.log_result(
            f"{'AVERAGE':<10} {total_trades//num_securities:<8} "
            f"{'':>9} "
            f"${avg_return:>12,.2f} {avg_return_pct:>10.2f}%"
        )
        self.log_result("=" * 70)

        # Additional statistics
        self.log_result(f"\nBatch Statistics:")
        self.log_result(f"  Securities Tested: {num_securities}")
        self.log_result(f"  Profitable: {winning_securities} ({winning_securities/num_securities*100:.1f}%)")
        self.log_result(f"  Unprofitable: {num_securities - winning_securities}")
        self.log_result(f"  Total Trades: {total_trades}")
        self.log_result(f"  Average Trades per Security: {total_trades/num_securities:.1f}")

        # Best and worst performers
        best = all_metrics[0]
        worst = all_metrics[-1]
        self.log_result(f"\n  Best Performer: {best['symbol']} (${best['total_return']:,.2f}, {best['total_return_pct']:.2f}%)")
        self.log_result(f"  Worst Performer: {worst['symbol']} (${worst['total_return']:,.2f}, {worst['total_return_pct']:.2f}%)")
        self.log_result("=" * 70)

        # Save trade logs
        logger = TradeLogger(Path('logs') / backtest_name)
        for symbol, result in results.items():
            logger.log_trades(symbol, backtest_name, result.trades, result.strategy_params)

        self.log_result(f"\nTrade logs saved to: logs/{backtest_name}/")

        # Generate Excel reports if enabled (batch-specific structure)
        if self.generate_excel_var.get():
            try:
                self.log_result("\nGenerating batch reports...")

                # Create batch reports folder structure
                batch_reports_dir = Path('logs') / backtest_name / 'batch_reports'
                individual_reports_dir = batch_reports_dir / 'individual'

                # Generate individual reports for each security
                self.log_result("\n  Generating individual security reports...")
                excel_generator = ExcelReportGenerator(
                    output_directory=individual_reports_dir,
                    initial_capital=initial_capital,
                    risk_free_rate=0.02,
                    benchmark_name="S&P 500"
                )

                for symbol, result in results.items():
                    report_path = excel_generator.generate_report(
                        result=result,
                        filename=f"{backtest_name}_{symbol}_report.xlsx"
                    )
                    self.log_result(f"    ✓ {symbol}: {report_path.name}")

                self.log_result(f"\n  ✓ Individual reports saved to: {individual_reports_dir}/")

                # Generate batch summary report with correlation analysis
                self.log_result("\n  Generating batch summary report...")
                batch_summary_generator = BatchSummaryReportGenerator(
                    output_directory=batch_reports_dir,
                    initial_capital=initial_capital
                )

                summary_report_path = batch_summary_generator.generate_batch_summary(
                    results=results,
                    backtest_name=backtest_name
                )

                self.log_result(f"  ✓ Summary report: {summary_report_path.name}")
                self.log_result(f"\n✓ All batch reports saved to: {batch_reports_dir}/")
                self.log_result("\nThe summary report includes:")
                self.log_result("  • Aggregate performance metrics across all securities")
                self.log_result("  • Detailed performance comparison")
                self.log_result("  • Correlation analysis (return, drawdown correlations)")
                self.log_result("  • Risk analysis and comparative visualizations")

            except Exception as e:
                self.log_result(f"⚠ Batch report generation failed: {str(e)}")
                import traceback
                traceback.print_exc()

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
