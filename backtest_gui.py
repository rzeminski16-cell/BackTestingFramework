"""
Backtesting Framework GUI (Wizard Flow)

A step-by-step graphical user interface for running backtests with clear configuration flow.

Steps:
1. Mode & Securities - Choose single/portfolio mode and select securities
2. Strategy - Select and configure strategy parameters
3. Capital & Costs - Set initial capital, commission, and slippage
4. Time Period - Optional date range filtering
5. Review & Run - Verify all settings and run backtest

Features:
- Clear wizard-based flow for easy configuration
- Selection summary visible at each step
- Review step before running to verify settings
- Separate results window showing configuration used
- Real-time validation with helpful messages
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import yaml
import threading
import queue

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
from Classes.GUI.wizard_base import WizardBase, WizardStep, ReviewStep
from Classes.GUI.results_window import ResultsWindow

# Import available strategies
from strategies.alphatrend_strategy import AlphaTrendStrategy


# =============================================================================
# WIZARD STEPS
# =============================================================================

class ModeSecuritiesStep(WizardStep):
    """Step 1: Select backtest mode and securities."""

    def __init__(self, wizard: 'BacktestWizard'):
        super().__init__(wizard, "Mode & Securities")
        self.wizard: BacktestWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # Mode Selection
        mode_frame = ttk.LabelFrame(parent, text="Backtest Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 15))

        self.wizard.mode_var = tk.StringVar(value="single")

        mode_desc_frame = ttk.Frame(mode_frame)
        mode_desc_frame.pack(fill=tk.X)

        single_frame = ttk.Frame(mode_desc_frame)
        single_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=20)
        ttk.Radiobutton(
            single_frame, text="Single Security",
            variable=self.wizard.mode_var, value="single",
            command=self._on_mode_change
        ).pack(anchor=tk.W)
        ttk.Label(
            single_frame,
            text="Test strategy on one security at a time",
            font=('TkDefaultFont', 8, 'italic'),
            foreground='gray'
        ).pack(anchor=tk.W, padx=(20, 0))

        portfolio_frame = ttk.Frame(mode_desc_frame)
        portfolio_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=20)
        ttk.Radiobutton(
            portfolio_frame, text="Portfolio",
            variable=self.wizard.mode_var, value="portfolio",
            command=self._on_mode_change
        ).pack(anchor=tk.W)
        ttk.Label(
            portfolio_frame,
            text="Test with shared capital across multiple securities",
            font=('TkDefaultFont', 8, 'italic'),
            foreground='gray'
        ).pack(anchor=tk.W, padx=(20, 0))

        # Securities Selection
        securities_frame = ttk.LabelFrame(parent, text="Select Securities", padding="10")
        securities_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        securities_frame.columnconfigure(0, weight=1)
        securities_frame.rowconfigure(1, weight=1)

        # Selection buttons
        btn_frame = ttk.Frame(securities_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        self.select_all_btn = ttk.Button(btn_frame, text="Select All", command=self._select_all)
        self.select_all_btn.pack(side=tk.LEFT, padx=2)
        self.deselect_all_btn = ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all)
        self.deselect_all_btn.pack(side=tk.LEFT, padx=2)

        # Selection count label
        self.selection_count_var = tk.StringVar(value="0 securities selected")
        ttk.Label(securities_frame, textvariable=self.selection_count_var,
                  font=('TkDefaultFont', 9, 'italic')).pack(side=tk.RIGHT, padx=10)

        # Listbox with scrollbar
        list_frame = ttk.Frame(securities_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.securities_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            height=10,
            yscrollcommand=scrollbar.set,
            font=('TkDefaultFont', 10)
        )
        self.securities_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.securities_listbox.yview)

        # Populate securities
        for symbol in self.wizard.available_securities:
            self.securities_listbox.insert(tk.END, symbol)

        # Bind selection change
        self.securities_listbox.bind('<<ListboxSelect>>', self._on_selection_change)

        # Portfolio-specific settings (Basket and Capital Contention)
        self.portfolio_settings_frame = ttk.LabelFrame(parent, text="Portfolio Settings", padding="10")
        self.portfolio_settings_frame.pack(fill=tk.X, pady=(0, 10))
        self.portfolio_settings_frame.pack_forget()  # Hidden initially

        # Basket selection
        basket_frame = ttk.Frame(self.portfolio_settings_frame)
        basket_frame.pack(fill=tk.X, pady=5)

        ttk.Label(basket_frame, text="Basket:", width=15).pack(side=tk.LEFT)
        self.wizard.basket_var = tk.StringVar(value="(Select securities manually)")
        self.basket_combo = ttk.Combobox(
            basket_frame, textvariable=self.wizard.basket_var,
            width=25, state='readonly'
        )
        self.basket_combo.pack(side=tk.LEFT, padx=5)
        self._refresh_basket_list()
        self.basket_combo.bind('<<ComboboxSelected>>', self._on_basket_selected)

        ttk.Button(basket_frame, text="Manage Baskets...",
                  command=self._open_basket_manager).pack(side=tk.LEFT, padx=5)

        # Capital Contention
        contention_frame = ttk.Frame(self.portfolio_settings_frame)
        contention_frame.pack(fill=tk.X, pady=5)

        ttk.Label(contention_frame, text="Capital Contention:", width=15).pack(side=tk.LEFT)
        self.wizard.contention_mode_var = tk.StringVar(value="default")
        ttk.Radiobutton(
            contention_frame, text="Default (Ignore)",
            variable=self.wizard.contention_mode_var, value="default",
            command=self._on_contention_mode_change
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            contention_frame, text="Vulnerability Score",
            variable=self.wizard.contention_mode_var, value="vulnerability",
            command=self._on_contention_mode_change
        ).pack(side=tk.LEFT, padx=5)

        self.vuln_config_btn = ttk.Button(
            contention_frame, text="Configure...",
            command=self._open_vulnerability_config
        )
        self.vuln_config_btn.pack(side=tk.LEFT, padx=10)
        self.vuln_config_btn.pack_forget()

    def _on_mode_change(self):
        """Handle mode change between single and portfolio."""
        mode = self.wizard.mode_var.get()
        if mode == "portfolio":
            self.securities_listbox.config(selectmode=tk.MULTIPLE)
            self.portfolio_settings_frame.pack(fill=tk.X, pady=(0, 10))
            self.select_all_btn.config(state=tk.NORMAL)
            self.deselect_all_btn.config(state=tk.NORMAL)
        else:
            self.securities_listbox.config(selectmode=tk.SINGLE)
            self.portfolio_settings_frame.pack_forget()
            # Keep only first selection if multiple were selected
            selections = self.securities_listbox.curselection()
            if len(selections) > 1:
                self.securities_listbox.selection_clear(0, tk.END)
                self.securities_listbox.selection_set(selections[0])
        self._on_selection_change(None)

    def _select_all(self):
        self.securities_listbox.selection_set(0, tk.END)
        self._on_selection_change(None)

    def _deselect_all(self):
        self.securities_listbox.selection_clear(0, tk.END)
        self._on_selection_change(None)

    def _on_selection_change(self, event):
        """Update selection count."""
        count = len(self.securities_listbox.curselection())
        self.selection_count_var.set(f"{count} securities selected")

    def _refresh_basket_list(self):
        baskets = self.wizard.basket_manager.list_baskets()
        self.basket_combo['values'] = ["(Select securities manually)"] + baskets

    def _on_basket_selected(self, event):
        basket_name = self.wizard.basket_var.get()
        if basket_name == "(Select securities manually)":
            self.wizard.selected_basket = None
            return

        basket = self.wizard.basket_manager.load(basket_name)
        if basket:
            self.wizard.selected_basket = basket
            self.securities_listbox.selection_clear(0, tk.END)
            for i, symbol in enumerate(self.securities_listbox.get(0, tk.END)):
                if symbol in basket.securities:
                    self.securities_listbox.selection_set(i)
            self._on_selection_change(None)

            # Load basket's capital contention settings
            if basket.default_capital_contention:
                if basket.default_capital_contention.mode == CapitalContentionMode.VULNERABILITY_SCORE:
                    self.wizard.contention_mode_var.set("vulnerability")
                    self.wizard.vulnerability_config = basket.default_capital_contention.vulnerability_config
                else:
                    self.wizard.contention_mode_var.set("default")
                self._on_contention_mode_change()

    def _open_basket_manager(self):
        def on_basket_selected(basket):
            if basket:
                self.wizard.selected_basket = basket
                self._refresh_basket_list()
                self.wizard.basket_var.set(basket.name)
                self._on_basket_selected(None)

        BasketManagerDialog(
            self.wizard.root,
            self.wizard.available_securities,
            on_basket_selected=on_basket_selected
        )

    def _on_contention_mode_change(self):
        mode = self.wizard.contention_mode_var.get()
        if mode == "vulnerability":
            self.vuln_config_btn.pack(side=tk.LEFT, padx=10)
            self.wizard.capital_contention_config = CapitalContentionConfig(
                mode=CapitalContentionMode.VULNERABILITY_SCORE,
                vulnerability_config=self.wizard.vulnerability_config
            )
        else:
            self.vuln_config_btn.pack_forget()
            self.wizard.capital_contention_config = CapitalContentionConfig.default_mode()

    def _open_vulnerability_config(self):
        def on_save(config):
            self.wizard.vulnerability_config = config
            self.wizard.capital_contention_config = CapitalContentionConfig(
                mode=CapitalContentionMode.VULNERABILITY_SCORE,
                vulnerability_config=config
            )

        VulnerabilityScoreConfigDialog(
            self.wizard.root,
            current_config=self.wizard.vulnerability_config,
            on_save=on_save
        )

    def on_enter(self):
        """Restore security selections when returning to this step."""
        if hasattr(self.wizard, 'selected_securities') and self.wizard.selected_securities:
            # Restore mode first
            mode = self.wizard.mode_var.get()
            if mode == "portfolio":
                self.securities_listbox.config(selectmode=tk.MULTIPLE)
            else:
                self.securities_listbox.config(selectmode=tk.SINGLE)

            # Restore selections
            self.securities_listbox.selection_clear(0, tk.END)
            all_symbols = list(self.securities_listbox.get(0, tk.END))
            for symbol in self.wizard.selected_securities:
                if symbol in all_symbols:
                    idx = all_symbols.index(symbol)
                    self.securities_listbox.selection_set(idx)
            self._on_selection_change(None)

    def get_summary(self) -> Dict[str, str]:
        mode = self.wizard.mode_var.get()
        # Use stored securities if available (set after validation), otherwise read from listbox
        if hasattr(self.wizard, 'selected_securities') and self.wizard.selected_securities:
            securities = self.wizard.selected_securities
        else:
            selections = self.securities_listbox.curselection()
            securities = [self.securities_listbox.get(i) for i in selections]

        summary = {
            "Mode": "Portfolio" if mode == "portfolio" else "Single Security"
        }

        if securities:
            if len(securities) <= 3:
                summary["Securities"] = ", ".join(securities)
            else:
                summary["Securities"] = f"{', '.join(securities[:3])}... ({len(securities)} total)"
        else:
            summary["Securities"] = "None selected"

        if mode == "portfolio":
            if self.wizard.selected_basket:
                summary["Basket"] = self.wizard.selected_basket.name
            contention = self.wizard.contention_mode_var.get()
            summary["Capital Contention"] = "Vulnerability Score" if contention == "vulnerability" else "Default"

        return summary

    def validate(self) -> bool:
        self.validation_errors = []
        mode = self.wizard.mode_var.get()
        selections = self.securities_listbox.curselection()

        if not selections:
            self.validation_errors.append("Please select at least one security.")
            return False

        if mode == "single" and len(selections) > 1:
            self.validation_errors.append("Single mode only allows one security. Switch to Portfolio mode for multiple.")
            return False

        if mode == "portfolio" and len(selections) < 2:
            self.validation_errors.append("Portfolio mode requires at least 2 securities.")
            return False

        # Store selected securities
        self.wizard.selected_securities = [self.securities_listbox.get(i) for i in selections]
        return True


class StrategyStep(WizardStep):
    """Step 2: Select and configure strategy."""

    def __init__(self, wizard: 'BacktestWizard'):
        super().__init__(wizard, "Strategy")
        self.wizard: BacktestWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # Strategy Selection
        selection_frame = ttk.LabelFrame(parent, text="Select Strategy", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(selection_frame, text="Strategy:").pack(side=tk.LEFT, padx=(0, 10))
        self.wizard.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.wizard.strategy_var,
            values=list(self.wizard.STRATEGIES.keys()),
            state="readonly",
            width=30
        )
        self.strategy_combo.pack(side=tk.LEFT)
        self.strategy_combo.bind('<<ComboboxSelected>>', self._on_strategy_change)

        # Strategy Parameters
        params_frame = ttk.LabelFrame(parent, text="Strategy Parameters", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        params_frame.columnconfigure(0, weight=1)
        params_frame.rowconfigure(1, weight=1)

        # Preset management
        preset_frame = ttk.Frame(params_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(preset_frame, text="Load Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar()
        self.preset_combo = ttk.Combobox(
            preset_frame, textvariable=self.preset_var,
            width=20, state='readonly'
        )
        self.preset_combo.pack(side=tk.LEFT, padx=5)
        self._refresh_presets()

        ttk.Button(preset_frame, text="Load", command=self._load_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Save As...", command=self._save_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Delete", command=self._delete_preset).pack(side=tk.LEFT, padx=2)

        # Parameter editing area (scrollable)
        param_container = ttk.Frame(params_frame)
        param_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(param_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(param_container, orient="vertical", command=canvas.yview)
        self.param_frame = ttk.Frame(canvas)

        self.param_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.param_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.param_entries: Dict[str, tk.StringVar] = {}

        # Initialize with first strategy (after all widgets are created)
        if self.wizard.STRATEGIES:
            self.strategy_combo.current(0)
            self._on_strategy_change(None)

    def _on_strategy_change(self, event):
        """Handle strategy selection change."""
        strategy_name = self.wizard.strategy_var.get()
        if not strategy_name:
            return

        # Initialize default parameters
        if strategy_name == 'AlphaTrendStrategy':
            self.wizard.strategy_params[strategy_name] = {
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

        self._refresh_presets()
        self._build_param_ui()

    def _refresh_presets(self):
        strategy_name = self.wizard.strategy_var.get()
        if strategy_name:
            presets = self.wizard.preset_manager.list_presets(strategy_name)
            self.preset_combo['values'] = [p['preset_name'] for p in presets]

    def _build_param_ui(self):
        """Build the parameter editing UI."""
        # Clear existing
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        strategy_name = self.wizard.strategy_var.get()
        if strategy_name not in self.wizard.strategy_params:
            return

        params = self.wizard.strategy_params[strategy_name]
        param_spec = self.wizard.param_config.get(strategy_name, {})

        # Categorize parameters
        categories = {
            'Indicator Settings': [],
            'Entry Settings': [],
            'Exit Settings': [],
            'Other Settings': []
        }

        for param_name, param_value in params.items():
            name_lower = param_name.lower()
            if any(kw in name_lower for kw in ['entry', 'buy', 'signal']):
                categories['Entry Settings'].append((param_name, param_value))
            elif any(kw in name_lower for kw in ['exit', 'sell', 'stop', 'take', 'trailing', 'grace', 'momentum']):
                categories['Exit Settings'].append((param_name, param_value))
            elif any(kw in name_lower for kw in ['period', 'length', 'lookback', 'window', 'multiplier', 'atr', 'volume', 'risk']):
                categories['Indicator Settings'].append((param_name, param_value))
            else:
                categories['Other Settings'].append((param_name, param_value))

        row = 0
        for category, param_list in categories.items():
            if not param_list:
                continue

            # Category header
            ttk.Separator(self.param_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=3, sticky='ew', pady=(10, 5), padx=5
            )
            row += 1
            ttk.Label(
                self.param_frame, text=category,
                font=('TkDefaultFont', 10, 'bold'), foreground='#2E5994'
            ).grid(row=row, column=0, columnspan=3, sticky='w', padx=10)
            row += 1

            # Parameters
            for param_name, param_value in param_list:
                spec = param_spec.get(param_name, {})
                min_val = spec.get('min', '')
                max_val = spec.get('max', '')
                param_type = spec.get('type', 'float')

                ttk.Label(self.param_frame, text=f"{param_name}:").grid(
                    row=row, column=0, sticky='w', padx=(20, 10), pady=3
                )

                var = tk.StringVar(value=str(param_value))
                self.param_entries[param_name] = var
                entry = ttk.Entry(self.param_frame, textvariable=var, width=12)
                entry.grid(row=row, column=1, sticky='w', pady=3)

                if min_val != '' and max_val != '':
                    if param_type == 'int':
                        range_str = f"[{int(min_val)} - {int(max_val)}]"
                    else:
                        range_str = f"[{min_val:.2f} - {max_val:.2f}]"
                    ttk.Label(
                        self.param_frame, text=range_str,
                        font=('TkDefaultFont', 8), foreground='#666666'
                    ).grid(row=row, column=2, sticky='w', padx=5, pady=3)

                row += 1

    def _load_preset(self):
        preset_name = self.preset_var.get()
        strategy_name = self.wizard.strategy_var.get()
        if not preset_name:
            messagebox.showwarning("No Preset", "Please select a preset to load.")
            return

        params = self.wizard.preset_manager.get_preset_parameters(strategy_name, preset_name)
        if params:
            for name, value in params.items():
                if name in self.param_entries:
                    self.param_entries[name].set(str(value))
            messagebox.showinfo("Success", f"Preset '{preset_name}' loaded!")

    def _save_preset(self):
        preset_name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if not preset_name:
            return

        strategy_name = self.wizard.strategy_var.get()
        try:
            params = self._get_current_params()
            self.wizard.preset_manager.save_preset(strategy_name, preset_name, params, "")
            self._refresh_presets()
            messagebox.showinfo("Success", f"Preset '{preset_name}' saved!")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {e}")

    def _delete_preset(self):
        preset_name = self.preset_var.get()
        strategy_name = self.wizard.strategy_var.get()
        if not preset_name:
            messagebox.showwarning("No Preset", "Please select a preset to delete.")
            return

        if messagebox.askyesno("Confirm Delete", f"Delete preset '{preset_name}'?"):
            if self.wizard.preset_manager.delete_preset(strategy_name, preset_name):
                self._refresh_presets()
                self.preset_var.set('')
                messagebox.showinfo("Success", f"Preset '{preset_name}' deleted!")

    def _get_current_params(self) -> Dict[str, Any]:
        """Get current parameter values from UI."""
        strategy_name = self.wizard.strategy_var.get()
        original_params = self.wizard.strategy_params[strategy_name]
        current_params = {}

        for param_name, var in self.param_entries.items():
            value_str = var.get()
            original = original_params.get(param_name)

            if isinstance(original, float):
                current_params[param_name] = float(value_str)
            elif isinstance(original, int):
                current_params[param_name] = int(value_str)
            else:
                try:
                    current_params[param_name] = int(value_str) if '.' not in value_str else float(value_str)
                except ValueError:
                    current_params[param_name] = value_str

        return current_params

    def on_leave(self):
        """Save parameters when leaving this step."""
        try:
            strategy_name = self.wizard.strategy_var.get()
            self.wizard.strategy_params[strategy_name] = self._get_current_params()
        except (ValueError, KeyError):
            pass  # Validation will catch issues

    def get_summary(self) -> Dict[str, str]:
        strategy_name = self.wizard.strategy_var.get()
        summary = {"Strategy": strategy_name if strategy_name else "None selected"}

        if strategy_name and strategy_name in self.wizard.strategy_params:
            params = self.wizard.strategy_params[strategy_name]
            modified = sum(1 for _ in params)
            summary["Parameters"] = f"{modified} parameters configured"

        return summary

    def validate(self) -> bool:
        self.validation_errors = []

        strategy_name = self.wizard.strategy_var.get()
        if not strategy_name:
            self.validation_errors.append("Please select a strategy.")
            return False

        try:
            self._get_current_params()
        except ValueError as e:
            self.validation_errors.append(f"Invalid parameter value: {e}")
            return False

        return True


class CapitalCostsStep(WizardStep):
    """Step 3: Configure capital and costs."""

    def __init__(self, wizard: 'BacktestWizard'):
        super().__init__(wizard, "Capital & Costs")
        self.wizard: BacktestWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # Initial Capital
        capital_frame = ttk.LabelFrame(parent, text="Initial Capital", padding="10")
        capital_frame.pack(fill=tk.X, pady=(0, 15))

        cap_input_frame = ttk.Frame(capital_frame)
        cap_input_frame.pack(fill=tk.X)

        ttk.Label(cap_input_frame, text="Amount:").pack(side=tk.LEFT, padx=(0, 10))
        self.wizard.capital_var = tk.StringVar(value="100000")
        capital_entry = ttk.Entry(cap_input_frame, textvariable=self.wizard.capital_var, width=15)
        capital_entry.pack(side=tk.LEFT)
        ttk.Label(cap_input_frame, text="GBP", font=('TkDefaultFont', 9, 'italic')).pack(side=tk.LEFT, padx=5)

        # Commission Settings
        commission_frame = ttk.LabelFrame(parent, text="Commission", padding="10")
        commission_frame.pack(fill=tk.X, pady=(0, 15))

        # Commission Mode
        mode_frame = ttk.Frame(commission_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 10))
        self.wizard.commission_mode_var = tk.StringVar(value="percentage")
        ttk.Radiobutton(
            mode_frame, text="Percentage",
            variable=self.wizard.commission_mode_var, value="percentage",
            command=self._update_commission_hint
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Fixed Amount",
            variable=self.wizard.commission_mode_var, value="fixed",
            command=self._update_commission_hint
        ).pack(side=tk.LEFT, padx=5)

        # Commission Value
        value_frame = ttk.Frame(commission_frame)
        value_frame.pack(fill=tk.X)

        ttk.Label(value_frame, text="Value:").pack(side=tk.LEFT, padx=(0, 10))
        self.wizard.commission_value_var = tk.StringVar(value="0.001")
        ttk.Entry(value_frame, textvariable=self.wizard.commission_value_var, width=10).pack(side=tk.LEFT)

        self.commission_hint_var = tk.StringVar(value="(0.1% per trade)")
        ttk.Label(
            value_frame, textvariable=self.commission_hint_var,
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(side=tk.LEFT, padx=10)

        # Slippage
        slippage_frame = ttk.LabelFrame(parent, text="Slippage", padding="10")
        slippage_frame.pack(fill=tk.X, pady=(0, 15))

        slip_input_frame = ttk.Frame(slippage_frame)
        slip_input_frame.pack(fill=tk.X)

        ttk.Label(slip_input_frame, text="Slippage (%):").pack(side=tk.LEFT, padx=(0, 10))
        self.wizard.slippage_var = tk.StringVar(value="0.1")
        ttk.Entry(slip_input_frame, textvariable=self.wizard.slippage_var, width=10).pack(side=tk.LEFT)
        ttk.Label(
            slip_input_frame,
            text="(Simulates price impact when entering/exiting positions)",
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(side=tk.LEFT, padx=10)

    def _update_commission_hint(self):
        mode = self.wizard.commission_mode_var.get()
        try:
            value = float(self.wizard.commission_value_var.get())
            if mode == "percentage":
                self.commission_hint_var.set(f"({value * 100:.2f}% per trade)")
            else:
                self.commission_hint_var.set(f"(£{value:.2f} per trade)")
        except ValueError:
            self.commission_hint_var.set("(enter a valid number)")

    def get_summary(self) -> Dict[str, str]:
        try:
            capital = float(self.wizard.capital_var.get())
            capital_str = f"£{capital:,.2f}"
        except ValueError:
            capital_str = self.wizard.capital_var.get()

        mode = self.wizard.commission_mode_var.get()
        try:
            comm_val = float(self.wizard.commission_value_var.get())
            if mode == "percentage":
                comm_str = f"{comm_val * 100:.2f}%"
            else:
                comm_str = f"£{comm_val:.2f}"
        except ValueError:
            comm_str = self.wizard.commission_value_var.get()

        try:
            slippage = float(self.wizard.slippage_var.get())
            slip_str = f"{slippage:.2f}%"
        except ValueError:
            slip_str = self.wizard.slippage_var.get()

        return {
            "Initial Capital": capital_str,
            "Commission": f"{comm_str} ({mode})",
            "Slippage": slip_str
        }

    def validate(self) -> bool:
        self.validation_errors = []

        try:
            capital = float(self.wizard.capital_var.get())
            if capital <= 0:
                self.validation_errors.append("Capital must be greater than 0.")
                return False
        except ValueError:
            self.validation_errors.append("Please enter a valid capital amount.")
            return False

        try:
            comm = float(self.wizard.commission_value_var.get())
            if comm < 0:
                self.validation_errors.append("Commission cannot be negative.")
                return False
        except ValueError:
            self.validation_errors.append("Please enter a valid commission value.")
            return False

        try:
            slip = float(self.wizard.slippage_var.get())
            if slip < 0:
                self.validation_errors.append("Slippage cannot be negative.")
                return False
        except ValueError:
            self.validation_errors.append("Please enter a valid slippage value.")
            return False

        return True


class TimePeriodStep(WizardStep):
    """Step 4: Configure time period."""

    def __init__(self, wizard: 'BacktestWizard'):
        super().__init__(wizard, "Time Period")
        self.wizard: BacktestWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        info_label = ttk.Label(
            parent,
            text="Optionally filter the backtest to a specific date range.\nLeave blank to use all available data.",
            font=('TkDefaultFont', 9, 'italic'),
            foreground='gray'
        )
        info_label.pack(anchor=tk.W, pady=(0, 15))

        # Date range frame
        date_frame = ttk.LabelFrame(parent, text="Date Range", padding="15")
        date_frame.pack(fill=tk.X, pady=(0, 15))

        # Start Date
        start_frame = ttk.Frame(date_frame)
        start_frame.pack(fill=tk.X, pady=5)

        ttk.Label(start_frame, text="Start Date:", width=12).pack(side=tk.LEFT)
        self.wizard.start_date_var = tk.StringVar(value="")
        start_entry = ttk.Entry(start_frame, textvariable=self.wizard.start_date_var, width=15)
        start_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(
            start_frame, text="YYYY-MM-DD (or leave blank)",
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(side=tk.LEFT, padx=10)

        # End Date
        end_frame = ttk.Frame(date_frame)
        end_frame.pack(fill=tk.X, pady=5)

        ttk.Label(end_frame, text="End Date:", width=12).pack(side=tk.LEFT)
        self.wizard.end_date_var = tk.StringVar(value="")
        end_entry = ttk.Entry(end_frame, textvariable=self.wizard.end_date_var, width=15)
        end_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(
            end_frame, text="YYYY-MM-DD (or leave blank)",
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(side=tk.LEFT, padx=10)

        # Data range info
        self.data_range_label = ttk.Label(
            date_frame,
            text="",
            font=('TkDefaultFont', 9),
            foreground='#2E5994'
        )
        self.data_range_label.pack(anchor=tk.W, pady=(15, 0))

    def on_enter(self):
        """Update data range info when entering this step."""
        if hasattr(self.wizard, 'selected_securities') and self.wizard.selected_securities:
            try:
                symbol = self.wizard.selected_securities[0]
                data = self.wizard.data_loader.load_csv(symbol)
                start = data['date'].min().strftime('%Y-%m-%d')
                end = data['date'].max().strftime('%Y-%m-%d')
                self.data_range_label.config(
                    text=f"Available data for {symbol}: {start} to {end}"
                )
            except Exception:
                self.data_range_label.config(text="")

    def get_summary(self) -> Dict[str, str]:
        start = self.wizard.start_date_var.get().strip()
        end = self.wizard.end_date_var.get().strip()

        if start or end:
            date_range = f"{start or 'earliest'} to {end or 'latest'}"
        else:
            date_range = "All available data"

        return {"Date Range": date_range}

    def validate(self) -> bool:
        self.validation_errors = []

        start_str = self.wizard.start_date_var.get().strip()
        end_str = self.wizard.end_date_var.get().strip()

        start_date = None
        end_date = None

        if start_str:
            try:
                start_date = datetime.strptime(start_str, '%Y-%m-%d')
            except ValueError:
                self.validation_errors.append("Start date must be in YYYY-MM-DD format.")
                return False

        if end_str:
            try:
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
            except ValueError:
                self.validation_errors.append("End date must be in YYYY-MM-DD format.")
                return False

        if start_date and end_date and start_date >= end_date:
            self.validation_errors.append("Start date must be before end date.")
            return False

        return True


class BacktestReviewStep(ReviewStep):
    """Step 5: Review and run backtest."""

    def __init__(self, wizard: 'BacktestWizard'):
        super().__init__(wizard, "Review & Run")
        self.wizard: BacktestWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        super().create_widgets(parent)

        # Add backtest name and options at the bottom
        options_frame = ttk.LabelFrame(parent, text="Backtest Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)

        # Backtest name
        name_frame = ttk.Frame(options_frame)
        name_frame.pack(fill=tk.X, pady=5)

        ttk.Label(name_frame, text="Backtest Name:", width=15).pack(side=tk.LEFT)
        self.wizard.backtest_name_var = tk.StringVar(value="test_1")
        ttk.Entry(name_frame, textvariable=self.wizard.backtest_name_var, width=30).pack(side=tk.LEFT, padx=5)

        # Excel report option
        self.wizard.generate_excel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Generate Excel Report",
            variable=self.wizard.generate_excel_var
        ).pack(anchor=tk.W, pady=5)


# =============================================================================
# MAIN WIZARD CLASS
# =============================================================================

class BacktestWizard(WizardBase):
    """Main Backtest Wizard application."""

    STRATEGIES = {
        'AlphaTrendStrategy': AlphaTrendStrategy
    }

    def __init__(self, root: tk.Tk):
        super().__init__(root, "Backtesting Framework", width=1100, height=750)

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

        # Load parameter configuration
        self.param_config = self._load_param_config()

        # Will be set by steps
        self.mode_var: Optional[tk.StringVar] = None
        self.strategy_var: Optional[tk.StringVar] = None
        self.capital_var: Optional[tk.StringVar] = None
        self.commission_mode_var: Optional[tk.StringVar] = None
        self.commission_value_var: Optional[tk.StringVar] = None
        self.slippage_var: Optional[tk.StringVar] = None
        self.start_date_var: Optional[tk.StringVar] = None
        self.end_date_var: Optional[tk.StringVar] = None
        self.backtest_name_var: Optional[tk.StringVar] = None
        self.generate_excel_var: Optional[tk.BooleanVar] = None
        self.basket_var: Optional[tk.StringVar] = None
        self.contention_mode_var: Optional[tk.StringVar] = None
        self.selected_securities: List[str] = []

        # Add wizard steps
        self.add_step(ModeSecuritiesStep(self))
        self.add_step(StrategyStep(self))
        self.add_step(CapitalCostsStep(self))
        self.add_step(TimePeriodStep(self))
        self.add_step(BacktestReviewStep(self))

        # Set callbacks
        self.on_complete = self._run_backtest
        self.on_cancel = self._on_cancel

        # Start wizard
        self.start()

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

    def _get_final_button_text(self) -> str:
        return "Run Backtest"

    def _on_cancel(self):
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel?"):
            self.root.destroy()

    def _run_backtest(self):
        """Run the backtest with configured settings in a background thread."""
        try:
            # Gather all settings for results window
            settings = self.get_all_summaries()

            # Parse configuration
            mode = self.mode_var.get()
            securities = self.selected_securities
            strategy_name = self.strategy_var.get()
            capital = float(self.capital_var.get())
            commission_mode = CommissionMode.PERCENTAGE if self.commission_mode_var.get() == "percentage" else CommissionMode.FIXED
            commission_value = float(self.commission_value_var.get())
            slippage_percent = float(self.slippage_var.get())
            commission = CommissionConfig(mode=commission_mode, value=commission_value)

            start_date = self._parse_date(self.start_date_var.get())
            end_date = self._parse_date(self.end_date_var.get())

            # Create strategy instance
            strategy_class = self.STRATEGIES[strategy_name]
            strategy_params = self.strategy_params.get(strategy_name, {})
            strategy = strategy_class(**strategy_params)

            # Generate backtest name
            user_name = self.backtest_name_var.get().strip()
            full_backtest_name = f"{strategy_name}_{user_name}"

            # Create results window
            results_window = ResultsWindow(
                self.root,
                f"Backtest Results: {full_backtest_name}",
                settings
            )

            # Create message queue for thread-safe UI updates
            msg_queue = queue.Queue()

            def update_ui():
                """Process messages from background thread to update UI."""
                try:
                    while True:
                        msg_type, data = msg_queue.get_nowait()
                        if msg_type == "log":
                            results_window.log(data)
                        elif msg_type == "progress":
                            current, total, detail = data
                            results_window.update_progress(current, total, detail)
                        elif msg_type == "complete":
                            results_window.on_complete()
                            return  # Stop polling
                        elif msg_type == "error":
                            results_window.on_error(data)
                            return  # Stop polling
                except queue.Empty:
                    pass
                # Schedule next check
                self.root.after(100, update_ui)

            def run_in_thread():
                """Run backtest in background thread."""
                try:
                    if mode == "single":
                        self._run_single_backtest_threaded(
                            msg_queue, securities[0], strategy, capital, commission,
                            start_date, end_date, full_backtest_name, slippage_percent, strategy_params
                        )
                    else:
                        self._run_portfolio_backtest_threaded(
                            msg_queue, securities, strategy, capital, commission,
                            start_date, end_date, full_backtest_name, slippage_percent, strategy_params
                        )
                except Exception as e:
                    import traceback
                    msg_queue.put(("error", f"{str(e)}\n{traceback.format_exc()}"))

            # Start background thread
            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()

            # Start UI update polling
            self.root.after(100, update_ui)

        except Exception as e:
            messagebox.showerror("Error", f"Backtest failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _run_backtest_sync(self):
        """Run the backtest synchronously (kept for reference)."""
        try:
            # Gather all settings for results window
            settings = self.get_all_summaries()

            # Parse configuration
            mode = self.mode_var.get()
            securities = self.selected_securities
            strategy_name = self.strategy_var.get()
            capital = float(self.capital_var.get())
            commission_mode = CommissionMode.PERCENTAGE if self.commission_mode_var.get() == "percentage" else CommissionMode.FIXED
            commission_value = float(self.commission_value_var.get())
            slippage_percent = float(self.slippage_var.get())
            commission = CommissionConfig(mode=commission_mode, value=commission_value)

            start_date = self._parse_date(self.start_date_var.get())
            end_date = self._parse_date(self.end_date_var.get())

            # Create strategy instance
            strategy_class = self.STRATEGIES[strategy_name]
            strategy_params = self.strategy_params.get(strategy_name, {})
            strategy = strategy_class(**strategy_params)

            # Generate backtest name
            user_name = self.backtest_name_var.get().strip()
            full_backtest_name = f"{strategy_name}_{user_name}"

            # Create results window
            results_window = ResultsWindow(
                self.root,
                f"Backtest Results: {full_backtest_name}",
                settings
            )

            # Run backtest based on mode
            if mode == "single":
                self._run_single_backtest(
                    results_window, securities[0], strategy, capital, commission,
                    start_date, end_date, full_backtest_name, slippage_percent, strategy_params
                )
            else:
                self._run_portfolio_backtest(
                    results_window, securities, strategy, capital, commission,
                    start_date, end_date, full_backtest_name, slippage_percent, strategy_params
                )

        except Exception as e:
            messagebox.showerror("Error", f"Backtest failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str.strip():
            return None
        try:
            return datetime.strptime(date_str.strip(), '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

    def _run_single_backtest_threaded(self, msg_queue: queue.Queue, symbol: str, strategy,
                                       capital: float, commission: CommissionConfig, start_date,
                                       end_date, backtest_name: str, slippage_percent: float,
                                       strategy_params: Dict):
        """Run single security backtest in background thread."""
        config = BacktestConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            slippage_percent=slippage_percent
        )

        msg_queue.put(("log", f"Loading data for {symbol}..."))
        data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
        msg_queue.put(("log", f"Loaded {len(data)} bars\n"))

        msg_queue.put(("log", f"Running backtest: {backtest_name}"))
        msg_queue.put(("log", f"Strategy: {strategy}"))
        msg_queue.put(("log", f"Security: {symbol}"))
        msg_queue.put(("log", f"Capital: £{capital:,.2f}\n"))

        def progress_callback(current: int, total: int):
            msg_queue.put(("progress", (current, total, f"Processing {symbol}")))

        engine = SingleSecurityEngine(
            config=config,
            currency_converter=self.currency_converter,
            security_registry=self.security_registry
        )
        result = engine.run(symbol, data, strategy, progress_callback=progress_callback)

        # Display results
        self._display_single_results_threaded(msg_queue, symbol, result, backtest_name, strategy_params)
        msg_queue.put(("complete", None))

    def _run_single_backtest(self, results_window: ResultsWindow, symbol: str, strategy,
                             capital: float, commission: CommissionConfig, start_date,
                             end_date, backtest_name: str, slippage_percent: float,
                             strategy_params: Dict):
        """Run single security backtest (sync version)."""
        config = BacktestConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            slippage_percent=slippage_percent
        )

        results_window.log(f"Loading data for {symbol}...")
        data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
        results_window.log(f"Loaded {len(data)} bars\n")

        results_window.log(f"Running backtest: {backtest_name}")
        results_window.log(f"Strategy: {strategy}")
        results_window.log(f"Security: {symbol}")
        results_window.log(f"Capital: £{capital:,.2f}\n")

        def progress_callback(current: int, total: int):
            results_window.update_progress(current, total, f"Processing {symbol}")
            self.root.update_idletasks()

        engine = SingleSecurityEngine(
            config=config,
            currency_converter=self.currency_converter,
            security_registry=self.security_registry
        )
        result = engine.run(symbol, data, strategy, progress_callback=progress_callback)

        # Display results
        self._display_single_results(results_window, symbol, result, backtest_name, strategy_params)
        results_window.on_complete()

    def _run_portfolio_backtest_threaded(self, msg_queue: queue.Queue, symbols: List[str],
                                          strategy, capital: float, commission: CommissionConfig,
                                          start_date, end_date, backtest_name: str,
                                          slippage_percent: float, strategy_params: Dict):
        """Run portfolio backtest in background thread."""
        basket_name = self.selected_basket.name if self.selected_basket else None

        config = PortfolioConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            capital_contention=self.capital_contention_config,
            slippage_percent=slippage_percent,
            basket_name=basket_name
        )

        msg_queue.put(("log", f"Running PORTFOLIO backtest: {backtest_name}"))
        msg_queue.put(("log", f"Strategy: {strategy}"))
        msg_queue.put(("log", f"Securities: {', '.join(symbols)}"))
        msg_queue.put(("log", f"Shared Capital: £{capital:,.2f}"))
        msg_queue.put(("log", f"Capital Contention: {self.capital_contention_config.mode.value}"))
        if self.capital_contention_config.mode == CapitalContentionMode.VULNERABILITY_SCORE:
            vc = self.capital_contention_config.vulnerability_config
            msg_queue.put(("log", f"  - Immunity Days: {vc.immunity_days}"))
            msg_queue.put(("log", f"  - Swap Threshold: {vc.swap_threshold}"))
        msg_queue.put(("log", "=" * 60))

        # Load data for all securities
        data_dict = {}
        for symbol in symbols:
            try:
                data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
                data_dict[symbol] = data
                msg_queue.put(("log", f"Loaded {symbol}: {len(data)} bars"))
            except Exception as e:
                msg_queue.put(("log", f"WARNING: Could not load {symbol}: {e}"))

        if not data_dict:
            msg_queue.put(("error", "No data loaded for any security"))
            return

        msg_queue.put(("log", ""))

        def progress_callback(current: int, total: int):
            msg_queue.put(("progress", (current, total, "Processing portfolio")))

        engine = PortfolioEngine(
            config=config,
            currency_converter=self.currency_converter,
            security_registry=self.security_registry
        )
        result = engine.run(data_dict, strategy, progress_callback=progress_callback)

        # Display portfolio results
        self._display_portfolio_results_threaded(msg_queue, result, backtest_name, strategy_params)
        msg_queue.put(("complete", None))

    def _run_portfolio_backtest(self, results_window: ResultsWindow, symbols: List[str],
                                strategy, capital: float, commission: CommissionConfig,
                                start_date, end_date, backtest_name: str,
                                slippage_percent: float, strategy_params: Dict):
        """Run portfolio backtest."""
        basket_name = self.selected_basket.name if self.selected_basket else None

        config = PortfolioConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            capital_contention=self.capital_contention_config,
            slippage_percent=slippage_percent,
            basket_name=basket_name
        )

        results_window.log(f"Running PORTFOLIO backtest: {backtest_name}")
        results_window.log(f"Strategy: {strategy}")
        results_window.log(f"Securities: {', '.join(symbols)}")
        results_window.log(f"Shared Capital: £{capital:,.2f}")
        results_window.log(f"Capital Contention: {self.capital_contention_config.mode.value}")
        if self.capital_contention_config.mode == CapitalContentionMode.VULNERABILITY_SCORE:
            vc = self.capital_contention_config.vulnerability_config
            results_window.log(f"  - Immunity Days: {vc.immunity_days}")
            results_window.log(f"  - Swap Threshold: {vc.swap_threshold}")
        results_window.log("=" * 60)

        # Load data for all securities
        data_dict = {}
        for symbol in symbols:
            try:
                data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
                data_dict[symbol] = data
                results_window.log(f"Loaded {symbol}: {len(data)} bars")
            except Exception as e:
                results_window.log(f"WARNING: Could not load {symbol}: {e}")

        if not data_dict:
            results_window.on_error("No data loaded for any security")
            return

        results_window.log("")

        def progress_callback(current: int, total: int):
            results_window.update_progress(current, total, "Processing portfolio")
            self.root.update_idletasks()

        engine = PortfolioEngine(
            config=config,
            currency_converter=self.currency_converter,
            security_registry=self.security_registry
        )
        result = engine.run(data_dict, strategy, progress_callback=progress_callback)

        # Display portfolio results
        self._display_portfolio_results(results_window, result, backtest_name, strategy_params)
        results_window.on_complete()

    def _display_single_results_threaded(self, msg_queue: queue.Queue, symbol: str,
                                          result, backtest_name: str, strategy_params: Dict):
        """Display single backtest results via message queue."""
        metrics = PerformanceMetrics.calculate_metrics(result)

        msg_queue.put(("log", "\n" + "=" * 60))
        msg_queue.put(("log", f"RESULTS: {symbol}"))
        msg_queue.put(("log", "=" * 60))
        msg_queue.put(("log", f"Total Return:        £{metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)"))
        msg_queue.put(("log", f"Final Equity:        £{metrics['final_equity']:,.2f}"))
        msg_queue.put(("log", f"Number of Trades:    {metrics['num_trades']}"))
        msg_queue.put(("log", f"Win Rate:            {metrics['win_rate']*100:.2f}%"))
        msg_queue.put(("log", f"Winning Trades:      {metrics['num_wins']}"))
        msg_queue.put(("log", f"Losing Trades:       {metrics['num_losses']}"))
        msg_queue.put(("log", f"Average Win:         £{metrics['avg_win']:,.2f}"))
        msg_queue.put(("log", f"Average Loss:        £{metrics['avg_loss']:,.2f}"))
        msg_queue.put(("log", f"Largest Win:         £{metrics['largest_win']:,.2f}"))
        msg_queue.put(("log", f"Largest Loss:        £{metrics['largest_loss']:,.2f}"))
        msg_queue.put(("log", f"Profit Factor:       {metrics['profit_factor']:.2f}"))
        msg_queue.put(("log", f"Avg Trade Duration:  {metrics['avg_trade_duration']:.1f} days"))
        msg_queue.put(("log", f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}"))
        msg_queue.put(("log", f"Max Drawdown:        £{metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)"))
        msg_queue.put(("log", "=" * 60))

        # Save trade log
        logger = TradeLogger(Path('logs') / backtest_name)
        logger.log_trades(symbol, backtest_name, result.trades, result.strategy_params)
        msg_queue.put(("log", f"\nTrade log saved to: logs/{backtest_name}/"))

        # Generate Excel report if enabled
        if self.generate_excel_var.get():
            try:
                msg_queue.put(("log", "\nGenerating Excel report..."))
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
                msg_queue.put(("log", f"Excel report saved to: {report_path}"))
            except Exception as e:
                msg_queue.put(("log", f"Excel report generation failed: {str(e)}"))

    def _display_single_results(self, results_window: ResultsWindow, symbol: str,
                                result, backtest_name: str, strategy_params: Dict):
        """Display single backtest results."""
        metrics = PerformanceMetrics.calculate_metrics(result)

        results_window.log("\n" + "=" * 60)
        results_window.log(f"RESULTS: {symbol}")
        results_window.log("=" * 60)
        results_window.log(f"Total Return:        £{metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        results_window.log(f"Final Equity:        £{metrics['final_equity']:,.2f}")
        results_window.log(f"Number of Trades:    {metrics['num_trades']}")
        results_window.log(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
        results_window.log(f"Winning Trades:      {metrics['num_wins']}")
        results_window.log(f"Losing Trades:       {metrics['num_losses']}")
        results_window.log(f"Average Win:         £{metrics['avg_win']:,.2f}")
        results_window.log(f"Average Loss:        £{metrics['avg_loss']:,.2f}")
        results_window.log(f"Largest Win:         £{metrics['largest_win']:,.2f}")
        results_window.log(f"Largest Loss:        £{metrics['largest_loss']:,.2f}")
        results_window.log(f"Profit Factor:       {metrics['profit_factor']:.2f}")
        results_window.log(f"Avg Trade Duration:  {metrics['avg_trade_duration']:.1f} days")
        results_window.log(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        results_window.log(f"Max Drawdown:        £{metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        results_window.log("=" * 60)

        # Save trade log
        logger = TradeLogger(Path('logs') / backtest_name)
        logger.log_trades(symbol, backtest_name, result.trades, result.strategy_params)
        results_window.log(f"\nTrade log saved to: logs/{backtest_name}/")

        # Generate Excel report if enabled
        if self.generate_excel_var.get():
            try:
                results_window.log("\nGenerating Excel report...")
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
                results_window.log(f"Excel report saved to: {report_path}")
            except Exception as e:
                results_window.log(f"Excel report generation failed: {str(e)}")

    def _display_portfolio_results_threaded(self, msg_queue: queue.Queue,
                                             result: PortfolioBacktestResult, backtest_name: str,
                                             strategy_params: Dict):
        """Display portfolio backtest results via message queue."""
        msg_queue.put(("log", "\n" + "=" * 60))
        msg_queue.put(("log", "PORTFOLIO RESULTS"))
        msg_queue.put(("log", "=" * 60))

        msg_queue.put(("log", f"\nInitial Capital:     £{result.config.initial_capital:,.2f}"))
        msg_queue.put(("log", f"Final Equity:        £{result.final_equity:,.2f}"))
        msg_queue.put(("log", f"Total Return:        £{result.total_return:,.2f} ({result.total_return_pct:.2f}%)"))

        msg_queue.put(("log", "\n--- Per-Security Performance ---"))
        total_trades = 0
        for symbol, sym_result in result.symbol_results.items():
            num_trades = len(sym_result.trades)
            total_trades += num_trades
            wins = len([t for t in sym_result.trades if t.pl > 0])
            win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
            msg_queue.put(("log", f"  {symbol}: {num_trades} trades, Win Rate: {win_rate:.1f}%, P/L: £{sym_result.total_return:,.2f}"))

        msg_queue.put(("log", f"\nTotal Trades: {total_trades}"))

        if result.signal_rejections:
            msg_queue.put(("log", f"\n--- Signal Rejections: {len(result.signal_rejections)} ---"))
            rejection_summary = {}
            for r in result.signal_rejections:
                rejection_summary[r.symbol] = rejection_summary.get(r.symbol, 0) + 1
            for symbol, count in sorted(rejection_summary.items(), key=lambda x: -x[1])[:5]:
                msg_queue.put(("log", f"  {symbol}: {count} rejections"))

        if result.vulnerability_swaps:
            msg_queue.put(("log", f"\n--- Vulnerability Swaps: {len(result.vulnerability_swaps)} ---"))
            for swap in result.vulnerability_swaps[:5]:
                msg_queue.put(("log", f"  {swap.date.strftime('%Y-%m-%d')}: {swap.closed_symbol} -> {swap.new_symbol}"))
            if len(result.vulnerability_swaps) > 5:
                msg_queue.put(("log", f"  ... and {len(result.vulnerability_swaps) - 5} more"))

        msg_queue.put(("log", "=" * 60))

        # Log portfolio results
        basket_name = result.config.basket_name
        portfolio_logger = PortfolioTradeLogger(backtest_name, basket_name)
        portfolio_logger.log_portfolio_result(result, strategy_params)
        msg_queue.put(("log", f"\nResults logged to: {portfolio_logger.base_dir}"))

        # Generate portfolio report if enabled
        if self.generate_excel_var.get():
            try:
                report_gen = PortfolioReportGenerator(portfolio_logger.reports_dir, use_enhanced=True)
                report_path = report_gen.generate_portfolio_report(result)
                msg_queue.put(("log", f"Portfolio report: {report_path}"))
            except Exception as e:
                msg_queue.put(("log", f"Warning: Could not generate portfolio report: {e}"))

    def _display_portfolio_results(self, results_window: ResultsWindow,
                                   result: PortfolioBacktestResult, backtest_name: str,
                                   strategy_params: Dict):
        """Display portfolio backtest results."""
        results_window.log("\n" + "=" * 60)
        results_window.log("PORTFOLIO RESULTS")
        results_window.log("=" * 60)

        results_window.log(f"\nInitial Capital:     £{result.config.initial_capital:,.2f}")
        results_window.log(f"Final Equity:        £{result.final_equity:,.2f}")
        results_window.log(f"Total Return:        £{result.total_return:,.2f} ({result.total_return_pct:.2f}%)")

        results_window.log("\n--- Per-Security Performance ---")
        total_trades = 0
        for symbol, sym_result in result.symbol_results.items():
            num_trades = len(sym_result.trades)
            total_trades += num_trades
            wins = len([t for t in sym_result.trades if t.pl > 0])
            win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
            results_window.log(f"  {symbol}: {num_trades} trades, Win Rate: {win_rate:.1f}%, P/L: £{sym_result.total_return:,.2f}")

        results_window.log(f"\nTotal Trades: {total_trades}")

        if result.signal_rejections:
            results_window.log(f"\n--- Signal Rejections: {len(result.signal_rejections)} ---")
            rejection_summary = {}
            for r in result.signal_rejections:
                rejection_summary[r.symbol] = rejection_summary.get(r.symbol, 0) + 1
            for symbol, count in sorted(rejection_summary.items(), key=lambda x: -x[1])[:5]:
                results_window.log(f"  {symbol}: {count} rejections")

        if result.vulnerability_swaps:
            results_window.log(f"\n--- Vulnerability Swaps: {len(result.vulnerability_swaps)} ---")
            for swap in result.vulnerability_swaps[:5]:
                results_window.log(f"  {swap.date.strftime('%Y-%m-%d')}: {swap.closed_symbol} -> {swap.new_symbol}")
            if len(result.vulnerability_swaps) > 5:
                results_window.log(f"  ... and {len(result.vulnerability_swaps) - 5} more")

        results_window.log("=" * 60)

        # Log portfolio results
        basket_name = result.config.basket_name
        portfolio_logger = PortfolioTradeLogger(backtest_name, basket_name)
        portfolio_logger.log_portfolio_result(result, strategy_params)
        results_window.log(f"\nResults logged to: {portfolio_logger.base_dir}")

        # Generate portfolio report if enabled
        if self.generate_excel_var.get():
            try:
                report_gen = PortfolioReportGenerator(portfolio_logger.reports_dir, use_enhanced=True)
                report_path = report_gen.generate_portfolio_report(result)
                results_window.log(f"Portfolio report: {report_path}")
            except Exception as e:
                results_window.log(f"Warning: Could not generate portfolio report: {e}")


def main():
    """Main entry point for GUI."""
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')

    # Maximize window (fullscreen)
    try:
        root.state('zoomed')  # Windows
    except tk.TclError:
        try:
            root.attributes('-zoomed', True)  # Linux
        except tk.TclError:
            root.geometry("1400x900")  # Fallback to large size

    app = BacktestWizard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
