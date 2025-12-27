"""
Optimization GUI (Wizard Flow)

A step-by-step graphical interface for walk-forward optimization with clear configuration flow.

Steps:
1. Mode & Securities - Choose single/portfolio mode and select securities
2. Strategy & Parameters - Select strategy and choose which parameters to optimize
3. Walk-Forward Settings - Configure window sizes and optimization mode
4. Execution Options - Speed mode, CPU cores, sensitivity analysis
5. Review & Run - Verify all settings and start optimization

Features:
- Clear wizard-based flow for easy configuration
- Selection summary visible at each step
- Review step before running to verify settings
- Separate results window showing configuration used
- Real-time validation with helpful messages
- Cancel optimization in progress
"""

import logging
import platform
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

import pandas as pd

from Classes.Data.data_loader import DataLoader
from Classes.Config.basket import Basket, BasketManager
from Classes.Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig,
    VULNERABILITY_SCORE_PARAM_DEFINITIONS
)
from Classes.Optimization.optimization_report_generator import OptimizationReportGenerator
from Classes.Optimization.sensitivity_analyzer import SensitivityAnalyzer
from Classes.Optimization.walk_forward_optimizer import WalkForwardOptimizer, WalkForwardMode
from Classes.GUI.basket_manager_dialog import BasketManagerDialog, VulnerabilityScoreConfigDialog
from Classes.GUI.wizard_base import WizardBase, WizardStep, ReviewStep
from Classes.GUI.results_window import OptimizationResultsWindow

# Import available strategies
from strategies.alphatrend_strategy import AlphaTrendStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# WIZARD STEPS
# =============================================================================

class OptModeSecuritiesStep(WizardStep):
    """Step 1: Select optimization mode and securities."""

    def __init__(self, wizard: 'OptimizationWizard'):
        super().__init__(wizard, "Mode & Securities")
        self.wizard: OptimizationWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # Mode Selection
        mode_frame = ttk.LabelFrame(parent, text="Optimization Mode", padding="10")
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
            text="Optimize parameters for one security at a time",
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
            text="Optimize for combined portfolio performance",
            font=('TkDefaultFont', 8, 'italic'),
            foreground='gray'
        ).pack(anchor=tk.W, padx=(20, 0))

        # Securities Selection
        securities_frame = ttk.LabelFrame(parent, text="Select Securities", padding="10")
        securities_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Selection buttons
        btn_frame = ttk.Frame(securities_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        self.select_all_btn = ttk.Button(btn_frame, text="Select All", command=self._select_all)
        self.select_all_btn.pack(side=tk.LEFT, padx=2)
        self.deselect_all_btn = ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all)
        self.deselect_all_btn.pack(side=tk.LEFT, padx=2)
        # Initially hidden for single mode
        self.select_all_btn.pack_forget()
        self.deselect_all_btn.pack_forget()

        # Selection count label
        self.selection_count_var = tk.StringVar(value="0 securities selected")
        ttk.Label(securities_frame, textvariable=self.selection_count_var,
                  font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.E, padx=10)

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
        for symbol in sorted(self.wizard.available_securities):
            self.securities_listbox.insert(tk.END, symbol)

        self.securities_listbox.bind('<<ListboxSelect>>', self._on_selection_change)

        # Portfolio-specific settings
        self.portfolio_settings_frame = ttk.LabelFrame(parent, text="Portfolio Settings", padding="10")
        self.portfolio_settings_frame.pack(fill=tk.X, pady=(0, 10))
        self.portfolio_settings_frame.pack_forget()

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
        mode = self.wizard.mode_var.get()
        if mode == "portfolio":
            self.securities_listbox.config(selectmode=tk.MULTIPLE)
            self.portfolio_settings_frame.pack(fill=tk.X, pady=(0, 10))
            self.select_all_btn.pack(side=tk.LEFT, padx=2)
            self.deselect_all_btn.pack(side=tk.LEFT, padx=2)
        else:
            self.securities_listbox.config(selectmode=tk.SINGLE)
            self.portfolio_settings_frame.pack_forget()
            self.select_all_btn.pack_forget()
            self.deselect_all_btn.pack_forget()
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

        summary = {"Mode": "Portfolio" if mode == "portfolio" else "Single Security"}

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
            summary["Capital Contention"] = "Vulnerability Score" if self.wizard.contention_mode_var.get() == "vulnerability" else "Default"

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

        self.wizard.selected_securities = [self.securities_listbox.get(i) for i in selections]
        return True


class StrategyParamsStep(WizardStep):
    """Step 2: Select strategy and which parameters to optimize."""

    def __init__(self, wizard: 'OptimizationWizard'):
        super().__init__(wizard, "Strategy & Parameters")
        self.wizard: OptimizationWizard = wizard
        self.vuln_checkboxes: Dict[str, tk.BooleanVar] = {}

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # Strategy Selection
        selection_frame = ttk.LabelFrame(parent, text="Select Strategy", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(selection_frame, text="Strategy:").pack(side=tk.LEFT, padx=(0, 10))
        self.wizard.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.wizard.strategy_var,
            values=list(self.wizard.STRATEGIES.keys()),
            state="readonly",
            width=30
        )
        strategy_combo.pack(side=tk.LEFT)
        if self.wizard.STRATEGIES:
            strategy_combo.current(0)
        strategy_combo.bind('<<ComboboxSelected>>', self._on_strategy_change)

        # Parameter Selection
        params_frame = ttk.LabelFrame(parent, text="Parameters to Optimize", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            params_frame,
            text="Select which parameters to optimize. Unselected parameters will use default values.",
            font=('TkDefaultFont', 9, 'italic'),
            foreground='gray'
        ).pack(anchor=tk.W, pady=(0, 10))

        # Buttons for select/deselect all
        btn_frame = ttk.Frame(params_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btn_frame, text="Select All", command=self._select_all_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all_params).pack(side=tk.LEFT, padx=2)

        self.param_count_var = tk.StringVar(value="0 parameters selected for optimization")
        ttk.Label(btn_frame, textvariable=self.param_count_var,
                  font=('TkDefaultFont', 9, 'italic')).pack(side=tk.RIGHT, padx=10)

        # Scrollable parameter list
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

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.param_checkboxes: Dict[str, tk.BooleanVar] = {}

        # Initialize with first strategy
        if self.wizard.STRATEGIES:
            self._on_strategy_change(None)

    def on_enter(self):
        """Rebuild UI when entering step (in case mode or contention changed)."""
        self._build_param_ui()

    def _on_strategy_change(self, event):
        strategy_name = self.wizard.strategy_var.get()
        if not strategy_name:
            return

        # Reset parameter selection
        self.wizard.selected_parameters = {}
        self._build_param_ui()

    def _build_param_ui(self):
        """Build the parameter selection UI."""
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_checkboxes.clear()
        self.vuln_checkboxes.clear()

        strategy_name = self.wizard.strategy_var.get()
        strategy_config = self.wizard.optimizer.config['strategy_parameters'].get(strategy_name, {})

        if not strategy_config:
            ttk.Label(self.param_frame, text="No parameters configured for this strategy.").pack()
            return

        # Get default values from strategy class
        try:
            strategy_class = self.wizard.STRATEGIES[strategy_name]
            default_strategy = strategy_class()
            default_params = {
                param: getattr(default_strategy, param, None)
                for param in strategy_config.keys()
            }
        except:
            default_params = {}

        # Categorize parameters
        categories = {
            'Indicator Settings': [],
            'Entry Settings': [],
            'Exit Settings': [],
            'Other Settings': []
        }

        for param_name, param_spec in strategy_config.items():
            name_lower = param_name.lower()
            if any(kw in name_lower for kw in ['entry', 'buy', 'signal']):
                categories['Entry Settings'].append((param_name, param_spec))
            elif any(kw in name_lower for kw in ['exit', 'sell', 'stop', 'take', 'trailing']):
                categories['Exit Settings'].append((param_name, param_spec))
            elif any(kw in name_lower for kw in ['period', 'length', 'lookback', 'window', 'multiplier', 'atr']):
                categories['Indicator Settings'].append((param_name, param_spec))
            else:
                categories['Other Settings'].append((param_name, param_spec))

        row = 0
        for category, param_list in categories.items():
            if not param_list:
                continue

            # Category header
            ttk.Separator(self.param_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=4, sticky='ew', pady=(10, 5), padx=5
            )
            row += 1
            ttk.Label(
                self.param_frame, text=category,
                font=('TkDefaultFont', 10, 'bold'), foreground='#2E5994'
            ).grid(row=row, column=0, columnspan=4, sticky='w', padx=10)
            row += 1

            for param_name, param_spec in param_list:
                # Initialize as selected by default
                var = tk.BooleanVar(value=True)
                self.param_checkboxes[param_name] = var
                self.wizard.selected_parameters[param_name] = True

                cb = ttk.Checkbutton(
                    self.param_frame, variable=var,
                    command=self._update_param_count
                )
                cb.grid(row=row, column=0, sticky='w', padx=(20, 5), pady=2)

                ttk.Label(self.param_frame, text=param_name).grid(
                    row=row, column=1, sticky='w', pady=2
                )

                # Range info
                min_val = param_spec.get('min', 'N/A')
                max_val = param_spec.get('max', 'N/A')
                param_type = param_spec.get('type', 'float')

                if min_val != 'N/A' and max_val != 'N/A':
                    if param_type == 'int':
                        range_str = f"[{int(min_val)} - {int(max_val)}]"
                    else:
                        range_str = f"[{min_val:.2f} - {max_val:.2f}]"
                else:
                    range_str = ""

                ttk.Label(
                    self.param_frame, text=range_str,
                    font=('TkDefaultFont', 8), foreground='#666666'
                ).grid(row=row, column=2, sticky='w', padx=5, pady=2)

                # Default value
                default_val = default_params.get(param_name, 'N/A')
                ttk.Label(
                    self.param_frame, text=f"Default: {default_val}",
                    font=('TkDefaultFont', 8), foreground='gray'
                ).grid(row=row, column=3, sticky='w', padx=5, pady=2)

                row += 1

        # Add Vulnerability Score parameters if portfolio mode with vulnerability score
        if (hasattr(self.wizard, 'mode_var') and self.wizard.mode_var and
            self.wizard.mode_var.get() == "portfolio" and
            hasattr(self.wizard, 'contention_mode_var') and self.wizard.contention_mode_var and
            self.wizard.contention_mode_var.get() == "vulnerability"):

            # Separator and header for vulnerability score parameters
            ttk.Separator(self.param_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=4, sticky='ew', pady=(15, 5), padx=5
            )
            row += 1
            ttk.Label(
                self.param_frame, text="Vulnerability Score Parameters",
                font=('TkDefaultFont', 10, 'bold'), foreground='#8B4513'
            ).grid(row=row, column=0, columnspan=4, sticky='w', padx=10)
            row += 1
            ttk.Label(
                self.param_frame,
                text="(These control position swapping behavior in portfolio mode)",
                font=('TkDefaultFont', 8, 'italic'), foreground='gray'
            ).grid(row=row, column=0, columnspan=4, sticky='w', padx=10)
            row += 1

            # Add each vulnerability score parameter
            for param_name, param_spec in VULNERABILITY_SCORE_PARAM_DEFINITIONS.items():
                # Initialize as NOT selected by default (these are advanced parameters)
                var = tk.BooleanVar(value=False)
                self.vuln_checkboxes[param_name] = var

                # Initialize in wizard's selected_vuln_parameters
                if not hasattr(self.wizard, 'selected_vuln_parameters'):
                    self.wizard.selected_vuln_parameters = {}
                self.wizard.selected_vuln_parameters[param_name] = False

                cb = ttk.Checkbutton(
                    self.param_frame, variable=var,
                    command=self._update_param_count
                )
                cb.grid(row=row, column=0, sticky='w', padx=(20, 5), pady=2)

                ttk.Label(self.param_frame, text=param_name).grid(
                    row=row, column=1, sticky='w', pady=2
                )

                # Range info
                min_val = param_spec.get('min', 'N/A')
                max_val = param_spec.get('max', 'N/A')
                param_type = param_spec.get('type', 'float')

                if min_val != 'N/A' and max_val != 'N/A':
                    if param_type == 'int':
                        range_str = f"[{int(min_val)} - {int(max_val)}]"
                    else:
                        range_str = f"[{min_val:.2f} - {max_val:.2f}]"
                else:
                    range_str = ""

                ttk.Label(
                    self.param_frame, text=range_str,
                    font=('TkDefaultFont', 8), foreground='#666666'
                ).grid(row=row, column=2, sticky='w', padx=5, pady=2)

                # Default value
                default_val = param_spec.get('default', 'N/A')
                ttk.Label(
                    self.param_frame, text=f"Default: {default_val}",
                    font=('TkDefaultFont', 8), foreground='gray'
                ).grid(row=row, column=3, sticky='w', padx=5, pady=2)

                row += 1

        self._update_param_count()

    def _select_all_params(self):
        for var in self.param_checkboxes.values():
            var.set(True)
        for var in self.vuln_checkboxes.values():
            var.set(True)
        self._update_param_count()

    def _deselect_all_params(self):
        for var in self.param_checkboxes.values():
            var.set(False)
        for var in self.vuln_checkboxes.values():
            var.set(False)
        self._update_param_count()

    def _update_param_count(self):
        strategy_count = sum(1 for var in self.param_checkboxes.values() if var.get())
        vuln_count = sum(1 for var in self.vuln_checkboxes.values() if var.get())
        total_count = strategy_count + vuln_count
        total_params = len(self.param_checkboxes) + len(self.vuln_checkboxes)

        if self.vuln_checkboxes:
            self.param_count_var.set(f"{total_count}/{total_params} parameters selected ({strategy_count} strategy, {vuln_count} vulnerability)")
        else:
            self.param_count_var.set(f"{strategy_count}/{len(self.param_checkboxes)} parameters selected for optimization")

        # Update wizard's selected_parameters
        for name, var in self.param_checkboxes.items():
            self.wizard.selected_parameters[name] = var.get()

        # Update wizard's selected_vuln_parameters
        if not hasattr(self.wizard, 'selected_vuln_parameters'):
            self.wizard.selected_vuln_parameters = {}
        for name, var in self.vuln_checkboxes.items():
            self.wizard.selected_vuln_parameters[name] = var.get()

    def get_summary(self) -> Dict[str, str]:
        strategy_name = self.wizard.strategy_var.get()
        strategy_count = sum(1 for var in self.param_checkboxes.values() if var.get())
        vuln_count = sum(1 for var in self.vuln_checkboxes.values() if var.get())
        total = len(self.param_checkboxes) + len(self.vuln_checkboxes)

        summary = {"Strategy": strategy_name}

        if self.vuln_checkboxes:
            summary["Parameters"] = f"{strategy_count + vuln_count}/{total} selected ({strategy_count} strategy, {vuln_count} vulnerability)"
        else:
            summary["Parameters"] = f"{strategy_count}/{len(self.param_checkboxes)} selected for optimization"

        return summary

    def validate(self) -> bool:
        self.validation_errors = []

        if not self.wizard.strategy_var.get():
            self.validation_errors.append("Please select a strategy.")
            return False

        strategy_count = sum(1 for var in self.param_checkboxes.values() if var.get())
        vuln_count = sum(1 for var in self.vuln_checkboxes.values() if var.get())
        total_count = strategy_count + vuln_count

        if total_count == 0:
            self.validation_errors.append("Please select at least one parameter to optimize.")
            return False

        return True


class WalkForwardStep(WizardStep):
    """Step 3: Configure walk-forward optimization settings."""

    def __init__(self, wizard: 'OptimizationWizard'):
        super().__init__(wizard, "Walk-Forward Settings")
        self.wizard: OptimizationWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # Walk-Forward Mode
        mode_frame = ttk.LabelFrame(parent, text="Walk-Forward Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 15))

        wf_defaults = self.wizard.optimizer.config.get('walk_forward', {})
        default_mode = wf_defaults.get('mode', 'rolling')

        self.wizard.wf_mode_var = tk.StringVar(value=default_mode)

        rolling_frame = ttk.Frame(mode_frame)
        rolling_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(
            rolling_frame, text="Rolling (sliding window)",
            variable=self.wizard.wf_mode_var, value="rolling"
        ).pack(side=tk.LEFT)
        ttk.Label(
            rolling_frame,
            text="Fixed-size window slides forward. Good for frequent trading.",
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(side=tk.LEFT, padx=20)

        anchored_frame = ttk.Frame(mode_frame)
        anchored_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(
            anchored_frame, text="Anchored (expanding window)",
            variable=self.wizard.wf_mode_var, value="anchored"
        ).pack(side=tk.LEFT)
        ttk.Label(
            anchored_frame,
            text="Start fixed, window expands. Good for low-frequency trading.",
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(side=tk.LEFT, padx=20)

        # Window Settings
        window_frame = ttk.LabelFrame(parent, text="Window Configuration", padding="10")
        window_frame.pack(fill=tk.X, pady=(0, 15))

        default_train = wf_defaults.get('training_period_days', 365)
        default_test = wf_defaults.get('testing_period_days', 365)
        default_step_min = wf_defaults.get('step_size_min_days', 7)
        default_step_max = wf_defaults.get('step_size_max_days', 30)

        # Training Period
        train_frame = ttk.Frame(window_frame)
        train_frame.pack(fill=tk.X, pady=5)
        ttk.Label(train_frame, text="Training Period:", width=20).pack(side=tk.LEFT)
        self.wizard.training_period_var = tk.IntVar(value=default_train)
        ttk.Spinbox(
            train_frame, from_=90, to=1825, width=8,
            textvariable=self.wizard.training_period_var
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(train_frame, text="days (90-1825)", font=('TkDefaultFont', 8, 'italic'),
                  foreground='gray').pack(side=tk.LEFT, padx=5)

        # Testing Period
        test_frame = ttk.Frame(window_frame)
        test_frame.pack(fill=tk.X, pady=5)
        ttk.Label(test_frame, text="Testing Period:", width=20).pack(side=tk.LEFT)
        self.wizard.testing_period_var = tk.IntVar(value=default_test)
        ttk.Spinbox(
            test_frame, from_=30, to=730, width=8,
            textvariable=self.wizard.testing_period_var
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(test_frame, text="days (30-730)", font=('TkDefaultFont', 8, 'italic'),
                  foreground='gray').pack(side=tk.LEFT, padx=5)

        # Step Size
        step_frame = ttk.Frame(window_frame)
        step_frame.pack(fill=tk.X, pady=5)
        ttk.Label(step_frame, text="Step Size:", width=20).pack(side=tk.LEFT)
        self.wizard.step_min_var = tk.IntVar(value=default_step_min)
        ttk.Spinbox(
            step_frame, from_=7, to=365, width=6,
            textvariable=self.wizard.step_min_var
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(step_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.wizard.step_max_var = tk.IntVar(value=default_step_max)
        ttk.Spinbox(
            step_frame, from_=7, to=365, width=6,
            textvariable=self.wizard.step_max_var
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(step_frame, text="days", font=('TkDefaultFont', 8, 'italic'),
                  foreground='gray').pack(side=tk.LEFT, padx=5)

        # Window Estimation
        self.window_estimate_var = tk.StringVar(value="")
        estimate_label = ttk.Label(
            window_frame, textvariable=self.window_estimate_var,
            font=('TkDefaultFont', 9), foreground='#2E5994'
        )
        estimate_label.pack(anchor=tk.W, pady=(15, 0))

        # Bind changes to update estimate
        self.wizard.training_period_var.trace_add('write', self._update_window_estimate)
        self.wizard.testing_period_var.trace_add('write', self._update_window_estimate)
        self.wizard.step_min_var.trace_add('write', self._update_window_estimate)
        self.wizard.step_max_var.trace_add('write', self._update_window_estimate)

        # Date Filter
        date_frame = ttk.LabelFrame(parent, text="Date Filter (Optional)", padding="10")
        date_frame.pack(fill=tk.X)

        self.wizard.use_start_date_var = tk.BooleanVar(value=False)
        date_check_frame = ttk.Frame(date_frame)
        date_check_frame.pack(fill=tk.X)

        ttk.Checkbutton(
            date_check_frame, text="Filter data start date:",
            variable=self.wizard.use_start_date_var,
            command=self._toggle_start_date
        ).pack(side=tk.LEFT)

        self.wizard.start_date_var = tk.StringVar(value="")
        self.start_date_entry = ttk.Entry(
            date_check_frame, textvariable=self.wizard.start_date_var,
            width=12, state=tk.DISABLED
        )
        self.start_date_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(
            date_check_frame, text="YYYY-MM-DD",
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(side=tk.LEFT, padx=5)

    def _toggle_start_date(self):
        if self.wizard.use_start_date_var.get():
            self.start_date_entry.config(state=tk.NORMAL)
        else:
            self.start_date_entry.config(state=tk.DISABLED)

    def on_enter(self):
        self._update_window_estimate()

    def _update_window_estimate(self, *args):
        try:
            train_days = self.wizard.training_period_var.get()
            test_days = self.wizard.testing_period_var.get()
            step_min = self.wizard.step_min_var.get()
            step_max = self.wizard.step_max_var.get()

            if train_days <= 0 or test_days <= 0 or step_min <= 0 or step_max <= 0:
                self.window_estimate_var.set("Invalid settings")
                return

            if step_min > step_max:
                self.window_estimate_var.set("Step min > max")
                return

            # Estimate based on selected securities
            data_days = None
            if hasattr(self.wizard, 'selected_securities') and self.wizard.selected_securities:
                try:
                    symbol = self.wizard.selected_securities[0]
                    data = self.wizard.data_loader.load_csv(symbol)
                    data_start = data['date'].min()
                    data_end = data['date'].max()
                    data_days = (data_end - data_start).days
                except:
                    pass

            if data_days is None:
                data_days = 15 * 365

            window_size = train_days + test_days
            avg_step = (step_min + step_max) // 2  # Integer division to match optimizer

            if data_days < window_size:
                self.window_estimate_var.set(f"Not enough data ({data_days} days < {window_size} required)")
                return

            usable_days = data_days - window_size
            est_windows = int(usable_days / avg_step) + 1

            self.window_estimate_var.set(
                f"Estimated: {est_windows} windows (avg step {avg_step:.0f} days, {data_days} days of data)"
            )

        except (tk.TclError, ValueError):
            pass

    def get_summary(self) -> Dict[str, str]:
        mode = self.wizard.wf_mode_var.get()
        train = self.wizard.training_period_var.get()
        test = self.wizard.testing_period_var.get()
        step_min = self.wizard.step_min_var.get()
        step_max = self.wizard.step_max_var.get()

        summary = {
            "Mode": "Rolling" if mode == "rolling" else "Anchored",
            "Training Period": f"{train} days",
            "Testing Period": f"{test} days",
            "Step Size": f"{step_min}-{step_max} days"
        }

        if self.wizard.use_start_date_var.get() and self.wizard.start_date_var.get():
            summary["Start Date Filter"] = self.wizard.start_date_var.get()

        return summary

    def validate(self) -> bool:
        self.validation_errors = []

        try:
            train = self.wizard.training_period_var.get()
            test = self.wizard.testing_period_var.get()
            step_min = self.wizard.step_min_var.get()
            step_max = self.wizard.step_max_var.get()

            if train < 90:
                self.validation_errors.append("Training period must be at least 90 days.")
                return False

            if test < 30:
                self.validation_errors.append("Testing period must be at least 30 days.")
                return False

            if step_min > step_max:
                self.validation_errors.append("Minimum step size cannot exceed maximum.")
                return False

        except (tk.TclError, ValueError):
            self.validation_errors.append("Please enter valid numeric values.")
            return False

        if self.wizard.use_start_date_var.get():
            date_str = self.wizard.start_date_var.get().strip()
            if date_str:
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    self.validation_errors.append("Start date must be in YYYY-MM-DD format.")
                    return False

        return True


class ExecutionOptionsStep(WizardStep):
    """Step 4: Configure execution options."""

    def __init__(self, wizard: 'OptimizationWizard'):
        super().__init__(wizard, "Execution Options")
        self.wizard: OptimizationWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # Speed Mode
        speed_frame = ttk.LabelFrame(parent, text="Optimization Speed", padding="10")
        speed_frame.pack(fill=tk.X, pady=(0, 15))

        self.wizard.speed_mode_var = tk.StringVar(value="full")

        for mode, (label, desc, iters) in [
            ("full", ("Full", "Most thorough optimization", "100 iterations")),
            ("fast", ("Fast", "Balanced speed/quality", "50 iterations")),
            ("quick", ("Quick", "Fastest, for testing", "25 iterations"))
        ]:
            frame = ttk.Frame(speed_frame)
            frame.pack(fill=tk.X, pady=3)
            ttk.Radiobutton(
                frame, text=label,
                variable=self.wizard.speed_mode_var, value=mode
            ).pack(side=tk.LEFT)
            ttk.Label(
                frame, text=f"- {desc} ({iters})",
                font=('TkDefaultFont', 8, 'italic'), foreground='gray'
            ).pack(side=tk.LEFT, padx=10)

        # CPU Cores (non-Windows only)
        self.wizard.n_jobs_var = tk.IntVar(value=1)
        if platform.system() != 'Windows':
            cores_frame = ttk.LabelFrame(parent, text="CPU Cores", padding="10")
            cores_frame.pack(fill=tk.X, pady=(0, 15))

            ttk.Label(
                cores_frame,
                text="More cores = faster optimization, but may use more memory.",
                font=('TkDefaultFont', 8, 'italic'), foreground='gray'
            ).pack(anchor=tk.W, pady=(0, 10))

            cores_options = ttk.Frame(cores_frame)
            cores_options.pack(fill=tk.X)

            for cores, label in [(1, "1 (Recommended)"), (2, "2"), (4, "4"), (-1, "All Available")]:
                ttk.Radiobutton(
                    cores_options, text=label,
                    variable=self.wizard.n_jobs_var, value=cores
                ).pack(side=tk.LEFT, padx=10)
        else:
            # Info label for Windows
            info_frame = ttk.LabelFrame(parent, text="CPU Cores", padding="10")
            info_frame.pack(fill=tk.X, pady=(0, 15))
            ttk.Label(
                info_frame,
                text="Parallel processing is not available on Windows. Using single core.",
                font=('TkDefaultFont', 9, 'italic'), foreground='gray'
            ).pack(anchor=tk.W)

        # Sensitivity Analysis
        analysis_frame = ttk.LabelFrame(parent, text="Additional Analysis", padding="10")
        analysis_frame.pack(fill=tk.X)

        self.wizard.run_sensitivity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            analysis_frame, text="Run Sensitivity Analysis",
            variable=self.wizard.run_sensitivity_var
        ).pack(anchor=tk.W)
        ttk.Label(
            analysis_frame,
            text="Tests how robust the optimized parameters are to small changes.",
            font=('TkDefaultFont', 8, 'italic'), foreground='gray'
        ).pack(anchor=tk.W, padx=(25, 0))

    def get_summary(self) -> Dict[str, str]:
        speed = self.wizard.speed_mode_var.get()
        speed_labels = {"full": "Full (100 iter)", "fast": "Fast (50 iter)", "quick": "Quick (25 iter)"}

        summary = {"Speed Mode": speed_labels.get(speed, speed)}

        cores = self.wizard.n_jobs_var.get()
        if platform.system() != 'Windows':
            summary["CPU Cores"] = "All" if cores == -1 else str(cores)
        else:
            summary["CPU Cores"] = "1 (Windows)"

        summary["Sensitivity Analysis"] = "Yes" if self.wizard.run_sensitivity_var.get() else "No"

        return summary

    def validate(self) -> bool:
        return True


class OptimizationReviewStep(ReviewStep):
    """Step 5: Review and run optimization."""

    def __init__(self, wizard: 'OptimizationWizard'):
        super().__init__(wizard, "Review & Run")
        self.wizard: OptimizationWizard = wizard

    def create_widgets(self, parent: ttk.Frame) -> None:
        super().create_widgets(parent)

        # Add note about duration
        note_frame = ttk.Frame(parent)
        note_frame.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)

        ttk.Label(
            note_frame,
            text="Note: Optimization may take a long time depending on data size and settings.",
            font=('TkDefaultFont', 9, 'italic'),
            foreground='#cc6600'
        ).pack(anchor=tk.W)


# =============================================================================
# MAIN WIZARD CLASS
# =============================================================================

class OptimizationWizard(WizardBase):
    """Main Optimization Wizard application."""

    STRATEGIES = {
        'AlphaTrendStrategy': AlphaTrendStrategy
    }

    def __init__(self, root: tk.Tk):
        super().__init__(root, "Walk-Forward Optimization", width=1100, height=750)

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/daily'))
        self.available_securities = self.data_loader.get_available_symbols()

        # Optimization components
        self.optimizer = WalkForwardOptimizer()
        self.sensitivity_analyzer = SensitivityAnalyzer(self.optimizer.config)
        self.report_generator = OptimizationReportGenerator(self.optimizer.config)

        # State
        self.optimization_thread = None
        self.is_running = False
        self.selected_parameters: Dict[str, bool] = {}

        # Basket manager
        self.basket_manager = BasketManager()
        self.selected_basket: Optional[Basket] = None

        # Capital contention configuration
        self.capital_contention_config = CapitalContentionConfig.default_mode()
        self.vulnerability_config = VulnerabilityScoreConfig()

        # Will be set by steps
        self.mode_var: Optional[tk.StringVar] = None
        self.strategy_var: Optional[tk.StringVar] = None
        self.basket_var: Optional[tk.StringVar] = None
        self.contention_mode_var: Optional[tk.StringVar] = None
        self.wf_mode_var: Optional[tk.StringVar] = None
        self.training_period_var: Optional[tk.IntVar] = None
        self.testing_period_var: Optional[tk.IntVar] = None
        self.step_min_var: Optional[tk.IntVar] = None
        self.step_max_var: Optional[tk.IntVar] = None
        self.use_start_date_var: Optional[tk.BooleanVar] = None
        self.start_date_var: Optional[tk.StringVar] = None
        self.speed_mode_var: Optional[tk.StringVar] = None
        self.n_jobs_var: Optional[tk.IntVar] = None
        self.run_sensitivity_var: Optional[tk.BooleanVar] = None
        self.selected_securities: List[str] = []

        # Add wizard steps
        self.add_step(OptModeSecuritiesStep(self))
        self.add_step(StrategyParamsStep(self))
        self.add_step(WalkForwardStep(self))
        self.add_step(ExecutionOptionsStep(self))
        self.add_step(OptimizationReviewStep(self))

        # Set callbacks
        self.on_complete = self._start_optimization
        self.on_cancel = self._on_cancel

        # Start wizard
        self.start()

    def _get_final_button_text(self) -> str:
        return "Start Optimization"

    def _on_cancel(self):
        if self.is_running:
            if messagebox.askyesno("Cancel", "Optimization is running. Are you sure you want to cancel?"):
                self.optimizer.cancel()
                self.sensitivity_analyzer.cancel()
        else:
            if messagebox.askyesno("Cancel", "Are you sure you want to cancel?"):
                self.root.destroy()

    def _start_optimization(self):
        """Start the optimization process."""
        # Gather settings for results window
        settings = self.get_all_summaries()

        # Create results window with cancel callback
        self.results_window = OptimizationResultsWindow(
            self.root,
            "Optimization Results",
            settings,
            on_cancel=self._cancel_optimization
        )

        # Disable the wizard's next button
        self.next_button.config(state=tk.DISABLED)
        self.is_running = True

        # Start optimization in background thread
        self.optimization_thread = threading.Thread(
            target=self._run_optimization,
            daemon=True
        )
        self.optimization_thread.start()

    def _cancel_optimization(self):
        """Cancel the running optimization."""
        if self.is_running:
            self.optimizer.cancel()
            self.sensitivity_analyzer.cancel()
            self.results_window.log("\n=== CANCELLATION REQUESTED ===")
            self.results_window.log("Waiting for current operation to complete...")

    def _run_optimization(self):
        """Run optimization in background thread."""
        try:
            strategy_name = self.strategy_var.get()
            strategy_class = self.STRATEGIES[strategy_name]
            mode = self.mode_var.get()
            securities = self.selected_securities

            # Apply settings to optimizer
            self.optimizer.config['bayesian_optimization']['speed_mode'] = self.speed_mode_var.get()
            self.optimizer.config['bayesian_optimization']['n_jobs'] = self.n_jobs_var.get()
            self.optimizer.config['walk_forward']['training_period_days'] = self.training_period_var.get()
            self.optimizer.config['walk_forward']['testing_period_days'] = self.testing_period_var.get()
            self.optimizer.config['walk_forward']['step_size_min_days'] = self.step_min_var.get()
            self.optimizer.config['walk_forward']['step_size_max_days'] = self.step_max_var.get()
            self.optimizer.config['walk_forward']['mode'] = self.wf_mode_var.get()

            walk_forward_mode = WalkForwardMode(self.wf_mode_var.get())

            # Log configuration
            self.results_window.log(f"Strategy: {strategy_name}")
            self.results_window.log(f"Mode: {mode.upper()}")
            self.results_window.log(f"Securities: {', '.join(securities)}")
            self.results_window.log(f"Walk-Forward Mode: {self.wf_mode_var.get().upper()}")
            self.results_window.log(f"Training: {self.training_period_var.get()} days, Testing: {self.testing_period_var.get()} days")
            self.results_window.log(f"Speed Mode: {self.speed_mode_var.get().upper()}")
            self.results_window.log("")

            if mode == "portfolio":
                self._run_portfolio_optimization(strategy_class, securities, walk_forward_mode)
            else:
                self._run_single_optimization(strategy_class, securities[0], walk_forward_mode)

        except Exception as e:
            self.results_window.log(f"\nFATAL ERROR: {e}")
            logger.exception("Optimization failed")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Optimization failed: {e}"))

        finally:
            self.root.after(0, self._reset_ui)

    def _run_single_optimization(self, strategy_class, symbol: str, walk_forward_mode):
        """Run single security optimization."""
        self.results_window.log(f"Loading data for {symbol}...")

        try:
            data = self.data_loader.load_csv(symbol)
            self.results_window.log(f"Loaded {len(data)} bars")

            # Apply date filter if specified
            if self.use_start_date_var.get():
                start_date_str = self.start_date_var.get().strip()
                if start_date_str:
                    try:
                        start_date = pd.to_datetime(start_date_str)
                        original_len = len(data)
                        data = data[data['date'] >= start_date].copy()
                        if len(data) > 0:
                            self.results_window.log(f"Filtered to {len(data)} bars (from {start_date_str})")
                        else:
                            self.results_window.log(f"No data after {start_date_str}, using all data")
                            data = self.data_loader.load_csv(symbol)
                    except Exception as e:
                        self.results_window.log(f"Warning: Invalid date filter: {e}")

        except Exception as e:
            self.results_window.log(f"ERROR: Failed to load data: {e}")
            return

        # Run walk-forward optimization
        self.results_window.log("\nStarting walk-forward optimization...")

        def progress_callback(stage, curr, total):
            self.root.after(0, lambda: self.results_window.update_progress(curr, total, stage))

        try:
            wf_results = self.optimizer.optimize(
                strategy_class=strategy_class,
                symbol=symbol,
                data=data,
                selected_params=self.selected_parameters,
                progress_callback=progress_callback,
                walk_forward_mode=walk_forward_mode
            )

            if not self.is_running:
                return

            # Display results
            self._display_wf_results(wf_results)

            # Run sensitivity analysis if enabled
            sensitivity_results = None
            if self.run_sensitivity_var.get() and self.is_running:
                self.results_window.log("\n\nStarting sensitivity analysis...")

                sensitivity_results = self.sensitivity_analyzer.analyze(
                    strategy_class=strategy_class,
                    base_parameters=wf_results.most_common_params,
                    data=data,
                    symbol=symbol,
                    progress_callback=lambda curr, total: self.root.after(
                        0, lambda c=curr, t=total: self.results_window.update_progress(c, t, "Sensitivity")
                    )
                )

                if self.is_running:
                    self._display_sensitivity_results(sensitivity_results)

            # Generate report
            if self.is_running:
                self.results_window.log("\n\nGenerating Excel report...")
                report_path = self.report_generator.generate_report(
                    wf_results=wf_results,
                    sensitivity_results=sensitivity_results
                )
                self.results_window.log(f"Report saved to: {report_path}")

        except Exception as e:
            self.results_window.log(f"ERROR during optimization: {e}")
            logger.exception("Optimization failed")

        self.results_window.on_complete()

    def _run_portfolio_optimization(self, strategy_class, securities: List[str], walk_forward_mode):
        """Run portfolio optimization."""
        self.results_window.log(f"Capital Contention: {self.capital_contention_config.mode.value}")
        self.results_window.log("")

        # Load data for all securities
        data_dict = {}
        for symbol in securities:
            try:
                data = self.data_loader.load_csv(symbol)
                data_dict[symbol] = data
                self.results_window.log(f"Loaded {symbol}: {len(data)} bars")
            except Exception as e:
                self.results_window.log(f"ERROR loading {symbol}: {e}")

        if len(data_dict) < 2:
            self.results_window.log("ERROR: Need at least 2 securities with valid data")
            return

        self.results_window.log("\nStarting portfolio walk-forward optimization...")

        def progress_callback(stage, curr, total):
            self.root.after(0, lambda: self.results_window.update_progress(curr, total, stage))

        try:
            wf_results = self.optimizer.optimize_portfolio(
                strategy_class=strategy_class,
                data_dict=data_dict,
                capital_contention=self.capital_contention_config,
                initial_capital=100000.0,
                selected_params=self.selected_parameters,
                progress_callback=progress_callback,
                walk_forward_mode=walk_forward_mode
            )

            if not self.is_running:
                return

            self._display_wf_results(wf_results)

            # Generate report
            self.results_window.log("\n\nGenerating portfolio optimization report...")
            report_path = self.report_generator.generate_report(
                wf_results=wf_results,
                sensitivity_results=None
            )
            self.results_window.log(f"Report saved to: {report_path}")

        except Exception as e:
            self.results_window.log(f"ERROR during portfolio optimization: {e}")
            logger.exception("Portfolio optimization failed")

        self.results_window.on_complete()

    def _display_wf_results(self, wf_results):
        """Display walk-forward optimization results."""
        self.results_window.log("\n" + "=" * 60)
        self.results_window.log("WALK-FORWARD OPTIMIZATION RESULTS")
        self.results_window.log("=" * 60)
        self.results_window.log(f"Total Windows: {wf_results.total_windows}")
        self.results_window.log(f"Windows Passed: {wf_results.windows_passed_constraints}")
        self.results_window.log(f"Success Rate: {wf_results.windows_passed_constraints / wf_results.total_windows * 100:.1f}%")
        self.results_window.log(f"\nAvg In-Sample Sortino: {wf_results.avg_in_sample_sortino:.4f}")
        self.results_window.log(f"Avg Out-Sample Sortino: {wf_results.avg_out_sample_sortino:.4f}")
        self.results_window.log(f"Sortino Degradation: {wf_results.avg_sortino_degradation_pct:.2f}%")
        self.results_window.log(f"\nAvg In-Sample Sharpe: {wf_results.avg_in_sample_sharpe:.4f}")
        self.results_window.log(f"Avg Out-Sample Sharpe: {wf_results.avg_out_sample_sharpe:.4f}")
        self.results_window.log(f"Sharpe Degradation: {wf_results.avg_sharpe_degradation_pct:.2f}%")

        self.results_window.log("\nRecommended Parameters:")
        for param_name, param_value in wf_results.most_common_params.items():
            min_val, max_val = wf_results.parameter_ranges[param_name]
            self.results_window.log(f"  {param_name}: {param_value:.4f} (range: {min_val:.2f} - {max_val:.2f})")

    def _display_sensitivity_results(self, sensitivity_results):
        """Display sensitivity analysis results."""
        self.results_window.log("\n" + "=" * 60)
        self.results_window.log("SENSITIVITY ANALYSIS RESULTS")
        self.results_window.log("=" * 60)
        self.results_window.log(f"Total Tests: {len(sensitivity_results.all_tests)}")
        self.results_window.log(f"Overall Sortino Range: {sensitivity_results.overall_sortino_range_pct:.2f}%")
        self.results_window.log(f"Assessment: {'ROBUST' if sensitivity_results.is_overall_robust else 'SENSITIVE'}")

        self.results_window.log("\nMost Robust Parameters:")
        for param in sensitivity_results.most_robust_params:
            self.results_window.log(f"  - {param}")

        self.results_window.log("\nLeast Robust Parameters:")
        for param in sensitivity_results.least_robust_params:
            self.results_window.log(f"  - {param}")

    def _reset_ui(self):
        """Reset UI after optimization completes."""
        self.is_running = False
        self.next_button.config(state=tk.NORMAL)
        if hasattr(self, 'results_window'):
            self.results_window.disable_cancel()


def main():
    """Main entry point for the optimization GUI."""
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

    app = OptimizationWizard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
