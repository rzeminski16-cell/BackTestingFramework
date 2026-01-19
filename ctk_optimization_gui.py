"""
Optimization GUI (CustomTkinter Wizard)

A modern step-by-step graphical interface for walk-forward optimization with dark/blue theme.

Steps:
1. Mode & Securities - Choose single/portfolio mode and select securities
2. Strategy & Parameters - Select strategy and choose which parameters to optimize
3. Walk-Forward Settings - Configure window sizes and optimization mode
4. Execution Options - Speed mode, CPU cores, sensitivity analysis
5. Review & Run - Verify all settings and start optimization

Features:
- Modern CustomTkinter dark/blue theme
- Clear wizard-based flow for easy configuration
- Reusable components (SecuritySelector, etc.)
- Real-time validation with helpful messages
- Cancel optimization in progress
"""

import logging
import platform
import threading
import queue
import customtkinter as ctk
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes, ask_yes_no, show_error, ProgressPanel
from Classes.GUI.ctk_wizard_base import CTkWizardBase, CTkWizardStep, CTkReviewStep
from Classes.GUI.ctk_components import SecuritySelector

from Classes.Data.data_loader import DataLoader
from Classes.Config.basket import Basket, BasketManager
from Classes.Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig,
    VULNERABILITY_SCORE_PARAM_DEFINITIONS
)
from Classes.Optimization.optimization_report_generator import OptimizationReportGenerator
from Classes.Optimization.sensitivity_analyzer import SensitivityAnalyzer
from Classes.Optimization.walk_forward_optimizer import WalkForwardOptimizer, WalkForwardMode

# Import available strategies
from strategies.base_alphatrend_strategy import BaseAlphaTrendStrategy
from strategies.random_control_strategy import RandomControlStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# WIZARD STEPS
# =============================================================================

class CTkOptModeSecuritiesStep(CTkWizardStep):
    """Step 1: Select optimization mode and securities."""

    def __init__(self, wizard: 'CTkOptimizationWizard'):
        super().__init__(wizard, "Mode & Securities")
        self.wizard: CTkOptimizationWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        # Mode Selection Card
        mode_card = Theme.create_card(parent)
        mode_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        mode_content = Theme.create_frame(mode_card)
        mode_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(mode_content, "Optimization Mode", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.wizard.mode_var = ctk.StringVar(value="single")

        mode_options = Theme.create_frame(mode_content)
        mode_options.pack(fill="x")

        # Single Security option
        single_frame = Theme.create_frame(mode_options)
        single_frame.pack(side="left", expand=True, fill="x", padx=(0, Sizes.PAD_L))

        Theme.create_radiobutton(
            single_frame, "Single Security",
            variable=self.wizard.mode_var, value="single",
            command=self._on_mode_change
        ).pack(anchor="w")
        Theme.create_hint(
            single_frame,
            "Optimize parameters for one security at a time"
        ).pack(anchor="w", padx=(Sizes.PAD_XL, 0))

        # Portfolio option
        portfolio_frame = Theme.create_frame(mode_options)
        portfolio_frame.pack(side="left", expand=True, fill="x")

        Theme.create_radiobutton(
            portfolio_frame, "Portfolio",
            variable=self.wizard.mode_var, value="portfolio",
            command=self._on_mode_change
        ).pack(anchor="w")
        Theme.create_hint(
            portfolio_frame,
            "Optimize for combined portfolio performance"
        ).pack(anchor="w", padx=(Sizes.PAD_XL, 0))

        # Securities Selection
        securities_card = Theme.create_card(parent)
        securities_card.pack(fill="both", expand=True, pady=(0, Sizes.PAD_M))

        securities_content = Theme.create_frame(securities_card)
        securities_content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        self.security_selector = SecuritySelector(
            securities_content,
            available_securities=sorted(self.wizard.available_securities),
            title="Select Securities",
            multi_select=False,
            show_search=True,
            show_buttons=True,
            height=200,
            on_selection_change=self._on_selection_change
        )
        self.security_selector.pack(fill="both", expand=True)

        # Portfolio Settings (hidden initially)
        self.portfolio_card = Theme.create_card(parent)
        self.portfolio_content = Theme.create_frame(self.portfolio_card)
        self.portfolio_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(self.portfolio_content, "Portfolio Settings", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Basket selection
        basket_frame = Theme.create_frame(self.portfolio_content)
        basket_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(basket_frame, "Basket:").pack(side="left", padx=(0, Sizes.PAD_S))
        self.wizard.basket_var = ctk.StringVar(value="(Select securities manually)")
        basket_values = ["(Select securities manually)"] + self.wizard.basket_manager.list_baskets()
        self.basket_combo = Theme.create_combobox(
            basket_frame,
            values=basket_values,
            variable=self.wizard.basket_var,
            command=self._on_basket_selected,
            width=250
        )
        self.basket_combo.pack(side="left", padx=(0, Sizes.PAD_M))

        # Capital Contention
        contention_frame = Theme.create_frame(self.portfolio_content)
        contention_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(contention_frame, "Capital Contention:").pack(side="left", padx=(0, Sizes.PAD_S))
        self.wizard.contention_mode_var = ctk.StringVar(value="default")

        Theme.create_radiobutton(
            contention_frame, "Default (Ignore)",
            variable=self.wizard.contention_mode_var, value="default",
            command=self._on_contention_mode_change
        ).pack(side="left", padx=(0, Sizes.PAD_M))

        Theme.create_radiobutton(
            contention_frame, "Vulnerability Score",
            variable=self.wizard.contention_mode_var, value="vulnerability",
            command=self._on_contention_mode_change
        ).pack(side="left")

    def _on_mode_change(self):
        mode = self.wizard.mode_var.get()
        if mode == "portfolio":
            self.security_selector.multi_select = True
            self.security_selector.update_securities(sorted(self.wizard.available_securities))
            self.portfolio_card.pack(fill="x", pady=(0, Sizes.PAD_M))
        else:
            self.security_selector.multi_select = False
            selected = self.security_selector.get_selected()
            self.security_selector.update_securities(sorted(self.wizard.available_securities))
            if selected:
                self.security_selector.set_selected([selected[0]])
            self.portfolio_card.pack_forget()

    def _on_selection_change(self, selected: List[str]):
        pass

    def _on_basket_selected(self, basket_name: str):
        if basket_name == "(Select securities manually)":
            self.wizard.selected_basket = None
            return

        basket = self.wizard.basket_manager.load(basket_name)
        if basket:
            self.wizard.selected_basket = basket
            self.security_selector.set_selected(basket.securities)

            if basket.default_capital_contention:
                if basket.default_capital_contention.mode == CapitalContentionMode.VULNERABILITY_SCORE:
                    self.wizard.contention_mode_var.set("vulnerability")
                    self.wizard.vulnerability_config = basket.default_capital_contention.vulnerability_config
                else:
                    self.wizard.contention_mode_var.set("default")
                self._on_contention_mode_change()

    def _on_contention_mode_change(self):
        mode = self.wizard.contention_mode_var.get()
        if mode == "vulnerability":
            self.wizard.capital_contention_config = CapitalContentionConfig(
                mode=CapitalContentionMode.VULNERABILITY_SCORE,
                vulnerability_config=self.wizard.vulnerability_config
            )
        else:
            self.wizard.capital_contention_config = CapitalContentionConfig.default_mode()

    def on_enter(self):
        if hasattr(self.wizard, 'selected_securities') and self.wizard.selected_securities:
            mode = self.wizard.mode_var.get()
            self.security_selector.multi_select = (mode == "portfolio")
            self.security_selector.set_selected(self.wizard.selected_securities)

    def get_summary(self) -> Dict[str, str]:
        mode = self.wizard.mode_var.get()
        if hasattr(self.wizard, 'selected_securities') and self.wizard.selected_securities:
            securities = self.wizard.selected_securities
        else:
            securities = self.security_selector.get_selected()

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
        selections = self.security_selector.get_selected()

        if not selections:
            self.validation_errors.append("Please select at least one security.")
            return False

        if mode == "single" and len(selections) > 1:
            self.validation_errors.append("Single mode only allows one security. Switch to Portfolio mode for multiple.")
            return False

        if mode == "portfolio" and len(selections) < 2:
            self.validation_errors.append("Portfolio mode requires at least 2 securities.")
            return False

        self.wizard.selected_securities = selections
        return True


class CTkStrategyParamsStep(CTkWizardStep):
    """Step 2: Select strategy and which parameters to optimize."""

    def __init__(self, wizard: 'CTkOptimizationWizard'):
        super().__init__(wizard, "Strategy & Parameters")
        self.wizard: CTkOptimizationWizard = wizard
        self.param_checkboxes: Dict[str, ctk.BooleanVar] = {}
        self.vuln_checkboxes: Dict[str, ctk.BooleanVar] = {}

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        # Strategy Selection Card
        selection_card = Theme.create_card(parent)
        selection_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        selection_content = Theme.create_frame(selection_card)
        selection_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(selection_content, "Select Strategy", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        strategy_frame = Theme.create_frame(selection_content)
        strategy_frame.pack(fill="x")

        Theme.create_label(strategy_frame, "Strategy:").pack(side="left", padx=(0, Sizes.PAD_S))
        self.wizard.strategy_var = ctk.StringVar()
        strategy_names = list(self.wizard.STRATEGIES.keys())
        self.strategy_combo = Theme.create_combobox(
            strategy_frame,
            values=strategy_names,
            variable=self.wizard.strategy_var,
            command=self._on_strategy_change,
            width=250
        )
        self.strategy_combo.pack(side="left")

        # Parameters Card
        params_card = Theme.create_card(parent)
        params_card.pack(fill="both", expand=True)

        params_header = Theme.create_frame(params_card)
        params_header.pack(fill="x", padx=Sizes.PAD_L, pady=(Sizes.PAD_M, 0))

        Theme.create_header(params_header, "Parameters to Optimize", size="s").pack(side="left")

        # Buttons
        btn_frame = Theme.create_frame(params_header)
        btn_frame.pack(side="right")

        Theme.create_button(btn_frame, "Select All", command=self._select_all_params, style="secondary", width=90, height=28).pack(side="left", padx=2)
        Theme.create_button(btn_frame, "Deselect All", command=self._deselect_all_params, style="secondary", width=90, height=28).pack(side="left", padx=2)

        # Count label
        self.param_count_var = ctk.StringVar(value="0 parameters selected")
        Theme.create_label(params_card, "", textvariable=self.param_count_var, text_color=Colors.TEXT_SECONDARY).pack(anchor="w", padx=Sizes.PAD_L)

        # Hint
        Theme.create_hint(
            params_card,
            "Select which parameters to optimize. Unselected parameters will use default values."
        ).pack(anchor="w", padx=Sizes.PAD_L, pady=(0, Sizes.PAD_S))

        # Scrollable parameters area
        self.params_scroll = ctk.CTkScrollableFrame(
            params_card,
            fg_color="transparent"
        )
        self.params_scroll.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        # Initialize with first strategy
        if strategy_names:
            self.wizard.strategy_var.set(strategy_names[0])
            self._on_strategy_change(strategy_names[0])

    def on_enter(self):
        self._build_param_ui()

    def _on_strategy_change(self, strategy_name: str):
        if not strategy_name:
            return
        self.wizard.selected_parameters = {}
        self._build_param_ui()

    def _build_param_ui(self):
        for widget in self.params_scroll.winfo_children():
            widget.destroy()
        self.param_checkboxes.clear()
        self.vuln_checkboxes.clear()

        strategy_name = self.wizard.strategy_var.get()
        strategy_config = self.wizard.optimizer.config['strategy_parameters'].get(strategy_name, {})

        if not strategy_config:
            Theme.create_label(self.params_scroll, "No parameters configured for this strategy.").pack()
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

        for category, param_list in categories.items():
            if not param_list:
                continue

            # Category header
            cat_frame = Theme.create_frame(self.params_scroll)
            cat_frame.pack(fill="x", pady=(Sizes.PAD_M, Sizes.PAD_S))

            ctk.CTkFrame(cat_frame, fg_color=Colors.BORDER, height=1).pack(fill="x", pady=(0, Sizes.PAD_S))
            Theme.create_label(
                cat_frame, category,
                font=Fonts.LABEL_BOLD,
                text_color=Colors.PRIMARY_LIGHT
            ).pack(anchor="w")

            for param_name, param_spec in param_list:
                var = ctk.BooleanVar(value=True)
                self.param_checkboxes[param_name] = var
                self.wizard.selected_parameters[param_name] = True

                param_frame = Theme.create_frame(self.params_scroll)
                param_frame.pack(fill="x", pady=Sizes.PAD_XS, padx=(Sizes.PAD_M, 0))

                Theme.create_checkbox(
                    param_frame, param_name,
                    variable=var,
                    command=self._update_param_count
                ).pack(side="left")

                # Range info
                min_val = param_spec.get('min', 'N/A')
                max_val = param_spec.get('max', 'N/A')
                param_type = param_spec.get('type', 'float')

                if min_val != 'N/A' and max_val != 'N/A':
                    if param_type == 'int':
                        range_str = f"[{int(min_val)} - {int(max_val)}]"
                    else:
                        range_str = f"[{min_val:.2f} - {max_val:.2f}]"
                    Theme.create_hint(param_frame, range_str).pack(side="left", padx=(Sizes.PAD_M, 0))

                # Default value
                default_val = default_params.get(param_name, 'N/A')
                Theme.create_hint(param_frame, f"Default: {default_val}").pack(side="right")

        # Add Vulnerability Score parameters if portfolio mode with vulnerability score
        if (hasattr(self.wizard, 'mode_var') and self.wizard.mode_var and
            self.wizard.mode_var.get() == "portfolio" and
            hasattr(self.wizard, 'contention_mode_var') and self.wizard.contention_mode_var and
            self.wizard.contention_mode_var.get() == "vulnerability"):

            # Separator and header
            vuln_frame = Theme.create_frame(self.params_scroll)
            vuln_frame.pack(fill="x", pady=(Sizes.PAD_L, Sizes.PAD_S))

            ctk.CTkFrame(vuln_frame, fg_color=Colors.WARNING, height=2).pack(fill="x", pady=(0, Sizes.PAD_S))
            Theme.create_label(
                vuln_frame, "Vulnerability Score Parameters",
                font=Fonts.LABEL_BOLD,
                text_color=Colors.WARNING
            ).pack(anchor="w")
            Theme.create_hint(
                vuln_frame,
                "(These control position swapping behavior in portfolio mode)"
            ).pack(anchor="w")

            for param_name, param_spec in VULNERABILITY_SCORE_PARAM_DEFINITIONS.items():
                var = ctk.BooleanVar(value=False)
                self.vuln_checkboxes[param_name] = var

                if not hasattr(self.wizard, 'selected_vuln_parameters'):
                    self.wizard.selected_vuln_parameters = {}
                self.wizard.selected_vuln_parameters[param_name] = False

                param_frame = Theme.create_frame(self.params_scroll)
                param_frame.pack(fill="x", pady=Sizes.PAD_XS, padx=(Sizes.PAD_M, 0))

                Theme.create_checkbox(
                    param_frame, param_name,
                    variable=var,
                    command=self._update_param_count
                ).pack(side="left")

                min_val = param_spec.get('min', 'N/A')
                max_val = param_spec.get('max', 'N/A')
                param_type = param_spec.get('type', 'float')

                if min_val != 'N/A' and max_val != 'N/A':
                    if param_type == 'int':
                        range_str = f"[{int(min_val)} - {int(max_val)}]"
                    else:
                        range_str = f"[{min_val:.2f} - {max_val:.2f}]"
                    Theme.create_hint(param_frame, range_str).pack(side="left", padx=(Sizes.PAD_M, 0))

                default_val = param_spec.get('default', 'N/A')
                Theme.create_hint(param_frame, f"Default: {default_val}").pack(side="right")

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

        for name, var in self.param_checkboxes.items():
            self.wizard.selected_parameters[name] = var.get()

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
            summary["Parameters"] = f"{strategy_count + vuln_count}/{total} selected"
        else:
            summary["Parameters"] = f"{strategy_count}/{len(self.param_checkboxes)} selected"

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


class CTkWalkForwardStep(CTkWizardStep):
    """Step 3: Configure walk-forward optimization settings."""

    def __init__(self, wizard: 'CTkOptimizationWizard'):
        super().__init__(wizard, "Walk-Forward Settings")
        self.wizard: CTkOptimizationWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        wf_defaults = self.wizard.optimizer.config.get('walk_forward', {})

        # Walk-Forward Mode Card
        mode_card = Theme.create_card(parent)
        mode_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        mode_content = Theme.create_frame(mode_card)
        mode_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(mode_content, "Walk-Forward Mode", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        default_mode = wf_defaults.get('mode', 'rolling')
        self.wizard.wf_mode_var = ctk.StringVar(value=default_mode)

        # Rolling option
        rolling_frame = Theme.create_frame(mode_content)
        rolling_frame.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_radiobutton(
            rolling_frame, "Rolling (sliding window)",
            variable=self.wizard.wf_mode_var, value="rolling"
        ).pack(side="left")
        Theme.create_hint(rolling_frame, "Fixed-size window slides forward. Good for frequent trading.").pack(side="left", padx=(Sizes.PAD_M, 0))

        # Anchored option
        anchored_frame = Theme.create_frame(mode_content)
        anchored_frame.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_radiobutton(
            anchored_frame, "Anchored (expanding window)",
            variable=self.wizard.wf_mode_var, value="anchored"
        ).pack(side="left")
        Theme.create_hint(anchored_frame, "Start fixed, window expands. Good for low-frequency trading.").pack(side="left", padx=(Sizes.PAD_M, 0))

        # Window Configuration Card
        window_card = Theme.create_card(parent)
        window_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        window_content = Theme.create_frame(window_card)
        window_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(window_content, "Window Configuration", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        default_train = wf_defaults.get('training_period_days', 365)
        default_test = wf_defaults.get('testing_period_days', 365)
        default_step_min = wf_defaults.get('step_size_min_days', 7)
        default_step_max = wf_defaults.get('step_size_max_days', 30)

        # Training Period
        train_frame = Theme.create_frame(window_content)
        train_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(train_frame, "Training Period:", width=150).pack(side="left")
        self.wizard.training_period_var = ctk.StringVar(value=str(default_train))
        Theme.create_entry(train_frame, width=80).configure(textvariable=self.wizard.training_period_var)
        self.wizard.training_period_var.trace_add('write', self._update_window_estimate)
        Theme.create_entry(train_frame, width=80).pack(side="left")
        Theme.create_hint(train_frame, "days (90-1825)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Recreate entry properly
        for widget in train_frame.winfo_children():
            widget.destroy()
        Theme.create_label(train_frame, "Training Period:", width=150).pack(side="left")
        train_entry = Theme.create_entry(train_frame, width=80)
        train_entry.configure(textvariable=self.wizard.training_period_var)
        train_entry.pack(side="left")
        Theme.create_hint(train_frame, "days (90-1825)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Testing Period
        test_frame = Theme.create_frame(window_content)
        test_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(test_frame, "Testing Period:", width=150).pack(side="left")
        self.wizard.testing_period_var = ctk.StringVar(value=str(default_test))
        test_entry = Theme.create_entry(test_frame, width=80)
        test_entry.configure(textvariable=self.wizard.testing_period_var)
        test_entry.pack(side="left")
        Theme.create_hint(test_frame, "days (30-730)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Step Size
        step_frame = Theme.create_frame(window_content)
        step_frame.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(step_frame, "Step Size:", width=150).pack(side="left")
        self.wizard.step_min_var = ctk.StringVar(value=str(default_step_min))
        min_entry = Theme.create_entry(step_frame, width=60)
        min_entry.configure(textvariable=self.wizard.step_min_var)
        min_entry.pack(side="left")
        Theme.create_label(step_frame, "to").pack(side="left", padx=Sizes.PAD_S)
        self.wizard.step_max_var = ctk.StringVar(value=str(default_step_max))
        max_entry = Theme.create_entry(step_frame, width=60)
        max_entry.configure(textvariable=self.wizard.step_max_var)
        max_entry.pack(side="left")
        Theme.create_hint(step_frame, "days").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Window Estimation
        self.window_estimate_label = Theme.create_label(
            window_content, "",
            text_color=Colors.PRIMARY_LIGHT
        )
        self.window_estimate_label.pack(anchor="w", pady=(Sizes.PAD_M, 0))

        # Bind changes
        self.wizard.training_period_var.trace_add('write', self._update_window_estimate)
        self.wizard.testing_period_var.trace_add('write', self._update_window_estimate)
        self.wizard.step_min_var.trace_add('write', self._update_window_estimate)
        self.wizard.step_max_var.trace_add('write', self._update_window_estimate)

        # Date Filter Card
        date_card = Theme.create_card(parent)
        date_card.pack(fill="x")

        date_content = Theme.create_frame(date_card)
        date_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(date_content, "Date Filter (Optional)", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        date_frame = Theme.create_frame(date_content)
        date_frame.pack(fill="x")

        self.wizard.use_start_date_var = ctk.BooleanVar(value=False)
        Theme.create_checkbox(
            date_frame, "Filter data start date:",
            variable=self.wizard.use_start_date_var,
            command=self._toggle_start_date
        ).pack(side="left")

        self.wizard.start_date_var = ctk.StringVar(value="")
        self.start_date_entry = Theme.create_entry(date_frame, placeholder="YYYY-MM-DD", width=120)
        self.start_date_entry.configure(textvariable=self.wizard.start_date_var, state="disabled")
        self.start_date_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

    def _toggle_start_date(self):
        if self.wizard.use_start_date_var.get():
            self.start_date_entry.configure(state="normal")
        else:
            self.start_date_entry.configure(state="disabled")

    def on_enter(self):
        self._update_window_estimate()

    def _update_window_estimate(self, *args):
        try:
            train_days = int(self.wizard.training_period_var.get())
            test_days = int(self.wizard.testing_period_var.get())
            step_min = int(self.wizard.step_min_var.get())
            step_max = int(self.wizard.step_max_var.get())

            if train_days <= 0 or test_days <= 0 or step_min <= 0 or step_max <= 0:
                self.window_estimate_label.configure(text="Invalid settings")
                return

            if step_min > step_max:
                self.window_estimate_label.configure(text="Step min > max")
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
            avg_step = (step_min + step_max) // 2

            if data_days < window_size:
                self.window_estimate_label.configure(text=f"Not enough data ({data_days} days < {window_size} required)")
                return

            usable_days = data_days - window_size
            est_windows = int(usable_days / avg_step) + 1

            self.window_estimate_label.configure(
                text=f"Estimated: {est_windows} windows (avg step {avg_step} days, {data_days} days of data)"
            )

        except (ValueError, TypeError):
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
            train = int(self.wizard.training_period_var.get())
            test = int(self.wizard.testing_period_var.get())
            step_min = int(self.wizard.step_min_var.get())
            step_max = int(self.wizard.step_max_var.get())

            if train < 90:
                self.validation_errors.append("Training period must be at least 90 days.")
                return False

            if test < 30:
                self.validation_errors.append("Testing period must be at least 30 days.")
                return False

            if step_min > step_max:
                self.validation_errors.append("Minimum step size cannot exceed maximum.")
                return False

        except ValueError:
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


class CTkExecutionOptionsStep(CTkWizardStep):
    """Step 4: Configure execution options."""

    def __init__(self, wizard: 'CTkOptimizationWizard'):
        super().__init__(wizard, "Execution Options")
        self.wizard: CTkOptimizationWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        # Speed Mode Card
        speed_card = Theme.create_card(parent)
        speed_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        speed_content = Theme.create_frame(speed_card)
        speed_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(speed_content, "Optimization Speed", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.wizard.speed_mode_var = ctk.StringVar(value="full")

        for mode, label, desc, iters in [
            ("full", "Full", "Most thorough optimization", "100 iterations"),
            ("fast", "Fast", "Balanced speed/quality", "50 iterations"),
            ("quick", "Quick", "Fastest, for testing", "25 iterations")
        ]:
            frame = Theme.create_frame(speed_content)
            frame.pack(fill="x", pady=Sizes.PAD_XS)
            Theme.create_radiobutton(
                frame, label,
                variable=self.wizard.speed_mode_var, value=mode
            ).pack(side="left")
            Theme.create_hint(frame, f"- {desc} ({iters})").pack(side="left", padx=(Sizes.PAD_M, 0))

        # CPU Cores Card
        cores_card = Theme.create_card(parent)
        cores_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        cores_content = Theme.create_frame(cores_card)
        cores_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(cores_content, "CPU Cores", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.wizard.n_jobs_var = ctk.IntVar(value=1)

        if platform.system() != 'Windows':
            Theme.create_hint(
                cores_content,
                "More cores = faster optimization, but may use more memory."
            ).pack(anchor="w", pady=(0, Sizes.PAD_S))

            cores_frame = Theme.create_frame(cores_content)
            cores_frame.pack(fill="x")

            for cores, label in [(1, "1 (Recommended)"), (2, "2"), (4, "4"), (-1, "All Available")]:
                Theme.create_radiobutton(
                    cores_frame, label,
                    variable=self.wizard.n_jobs_var, value=cores
                ).pack(side="left", padx=(0, Sizes.PAD_L))
        else:
            Theme.create_label(
                cores_content,
                "Parallel processing is not available on Windows. Using single core.",
                text_color=Colors.TEXT_SECONDARY
            ).pack(anchor="w")

        # Sensitivity Analysis Card
        analysis_card = Theme.create_card(parent)
        analysis_card.pack(fill="x")

        analysis_content = Theme.create_frame(analysis_card)
        analysis_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(analysis_content, "Additional Analysis", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.wizard.run_sensitivity_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            analysis_content, "Run Sensitivity Analysis",
            variable=self.wizard.run_sensitivity_var
        ).pack(anchor="w")
        Theme.create_hint(
            analysis_content,
            "Tests how robust the optimized parameters are to small changes."
        ).pack(anchor="w", padx=(Sizes.PAD_XL, 0))

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


class CTkOptimizationReviewStep(CTkReviewStep):
    """Step 5: Review and run optimization."""

    def __init__(self, wizard: 'CTkOptimizationWizard'):
        super().__init__(wizard, "Review & Run")
        self.wizard: CTkOptimizationWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        super().create_widgets(parent)

        # Note about duration
        note_card = Theme.create_card(parent)
        note_card.pack(fill="x", pady=(Sizes.PAD_M, 0), side="bottom")

        note_content = Theme.create_frame(note_card)
        note_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_label(
            note_content,
            "Note: Optimization may take a long time depending on data size and settings.",
            text_color=Colors.WARNING
        ).pack(anchor="w")


# =============================================================================
# RESULTS WINDOW
# =============================================================================

class CTkOptimizationResultsWindow(ctk.CTkToplevel):
    """Modern results window for displaying optimization results."""

    def __init__(self, parent, title: str, settings: Dict[str, Dict[str, str]], on_cancel=None):
        super().__init__(parent)

        self.title(title)
        self.geometry("950x750")
        self.configure(fg_color=Colors.BG_DARK)

        self.on_cancel_callback = on_cancel
        self._create_layout(settings)

    def _create_layout(self, settings: Dict[str, Dict[str, str]]):
        main_frame = Theme.create_frame(self)
        main_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Configuration summary
        config_card = Theme.create_card(main_frame)
        config_card.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        config_header = Theme.create_frame(config_card)
        config_header.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)
        Theme.create_header(config_header, "Configuration", size="s").pack(side="left")

        config_content = Theme.create_frame(config_card)
        config_content.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        for step_name, step_settings in settings.items():
            step_frame = Theme.create_frame(config_content)
            step_frame.pack(fill="x", pady=Sizes.PAD_XS)

            Theme.create_label(step_frame, f"{step_name}: ", font=Fonts.LABEL_BOLD).pack(side="left")
            summary_parts = [f"{k}: {v}" for k, v in step_settings.items()]
            Theme.create_label(step_frame, " | ".join(summary_parts), text_color=Colors.TEXT_SECONDARY).pack(side="left")

        # Progress section
        progress_card = Theme.create_card(main_frame)
        progress_card.grid(row=1, column=0, sticky="nsew")

        progress_content = Theme.create_frame(progress_card)
        progress_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        self.progress_panel = ProgressPanel(progress_content, show_log=True, log_height=450)
        self.progress_panel.pack(fill="both", expand=True)

        # Buttons
        btn_frame = Theme.create_frame(main_frame)
        btn_frame.grid(row=2, column=0, sticky="e", pady=(Sizes.PAD_M, 0))

        self.cancel_btn = Theme.create_button(btn_frame, "Cancel", command=self._on_cancel, style="danger", width=100)
        self.cancel_btn.pack(side="right", padx=(Sizes.PAD_S, 0))

        Theme.create_button(btn_frame, "Close", command=self.destroy, style="secondary", width=100).pack(side="right")

    def log(self, message: str):
        self.progress_panel.log(message)

    def update_progress(self, current: int, total: int, detail: str = ""):
        progress = current / total if total > 0 else 0
        self.progress_panel.update_progress(progress, detail)

    def on_complete(self):
        self.progress_panel.set_complete("Optimization Complete")
        self.cancel_btn.configure(state="disabled")

    def on_error(self, error: str):
        self.progress_panel.log(f"ERROR: {error}", level="error")
        self.progress_panel.status_label.configure(text="Error", text_color=Colors.ERROR)

    def _on_cancel(self):
        if self.on_cancel_callback:
            self.on_cancel_callback()

    def disable_cancel(self):
        self.cancel_btn.configure(state="disabled")


# =============================================================================
# MAIN WIZARD CLASS
# =============================================================================

class CTkOptimizationWizard(CTkWizardBase):
    """Main Optimization Wizard application using CustomTkinter."""

    STRATEGIES = {
        'BaseAlphaTrendStrategy': BaseAlphaTrendStrategy,
        'RandomControlStrategy': RandomControlStrategy
    }

    def __init__(self):
        super().__init__(
            title="Walk-Forward Optimization",
            width=1100,
            height=750
        )

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
        self.selected_vuln_parameters: Dict[str, bool] = {}

        # Basket manager
        self.basket_manager = BasketManager()
        self.selected_basket: Optional[Basket] = None

        # Capital contention configuration
        self.capital_contention_config = CapitalContentionConfig.default_mode()
        self.vulnerability_config = VulnerabilityScoreConfig()

        # Variables (will be set by steps)
        self.mode_var: Optional[ctk.StringVar] = None
        self.strategy_var: Optional[ctk.StringVar] = None
        self.basket_var: Optional[ctk.StringVar] = None
        self.contention_mode_var: Optional[ctk.StringVar] = None
        self.wf_mode_var: Optional[ctk.StringVar] = None
        self.training_period_var: Optional[ctk.StringVar] = None
        self.testing_period_var: Optional[ctk.StringVar] = None
        self.step_min_var: Optional[ctk.StringVar] = None
        self.step_max_var: Optional[ctk.StringVar] = None
        self.use_start_date_var: Optional[ctk.BooleanVar] = None
        self.start_date_var: Optional[ctk.StringVar] = None
        self.speed_mode_var: Optional[ctk.StringVar] = None
        self.n_jobs_var: Optional[ctk.IntVar] = None
        self.run_sensitivity_var: Optional[ctk.BooleanVar] = None
        self.selected_securities: List[str] = []

        # Add wizard steps
        self.add_step(CTkOptModeSecuritiesStep(self))
        self.add_step(CTkStrategyParamsStep(self))
        self.add_step(CTkWalkForwardStep(self))
        self.add_step(CTkExecutionOptionsStep(self))
        self.add_step(CTkOptimizationReviewStep(self))

        # Set callbacks
        self.on_complete = self._start_optimization
        self.on_cancel = self._on_cancel

        # Start wizard
        self.start()

    def _get_final_button_text(self) -> str:
        return "Start Optimization"

    def _on_cancel(self):
        if self.is_running:
            if ask_yes_no(self.root, "Cancel", "Optimization is running. Are you sure you want to cancel?"):
                self.optimizer.cancel()
                self.sensitivity_analyzer.cancel()
        else:
            if ask_yes_no(self.root, "Cancel", "Are you sure you want to cancel?"):
                self.root.destroy()

    def _start_optimization(self):
        settings = self.get_all_summaries()

        self.results_window = CTkOptimizationResultsWindow(
            self.root,
            "Optimization Results",
            settings,
            on_cancel=self._cancel_optimization
        )

        self.next_button.configure(state="disabled")
        self.is_running = True

        self.optimization_thread = threading.Thread(
            target=self._run_optimization,
            daemon=True
        )
        self.optimization_thread.start()

    def _cancel_optimization(self):
        if self.is_running:
            self.optimizer.cancel()
            self.sensitivity_analyzer.cancel()
            self.results_window.log("\n=== CANCELLATION REQUESTED ===")
            self.results_window.log("Waiting for current operation to complete...")

    def _run_optimization(self):
        try:
            strategy_name = self.strategy_var.get()
            strategy_class = self.STRATEGIES[strategy_name]
            mode = self.mode_var.get()
            securities = self.selected_securities

            # Apply settings to optimizer
            self.optimizer.config['bayesian_optimization']['speed_mode'] = self.speed_mode_var.get()
            self.optimizer.config['bayesian_optimization']['n_jobs'] = self.n_jobs_var.get()
            self.optimizer.config['walk_forward']['training_period_days'] = int(self.training_period_var.get())
            self.optimizer.config['walk_forward']['testing_period_days'] = int(self.testing_period_var.get())
            self.optimizer.config['walk_forward']['step_size_min_days'] = int(self.step_min_var.get())
            self.optimizer.config['walk_forward']['step_size_max_days'] = int(self.step_max_var.get())
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
            self.root.after(0, lambda: show_error(self.root, "Error", f"Optimization failed: {e}"))

        finally:
            self.root.after(0, self._reset_ui)

    def _run_single_optimization(self, strategy_class, symbol: str, walk_forward_mode):
        self.results_window.log(f"Loading data for {symbol}...")

        try:
            data = self.data_loader.load_csv(symbol)
            self.results_window.log(f"Loaded {len(data)} bars")

            if self.use_start_date_var.get():
                start_date_str = self.start_date_var.get().strip()
                if start_date_str:
                    try:
                        start_date = pd.to_datetime(start_date_str)
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

            self._display_wf_results(wf_results)

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
        self.results_window.log(f"Capital Contention: {self.capital_contention_config.mode.value}")
        self.results_window.log("")

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
        self.is_running = False
        self.next_button.configure(state="normal")
        if hasattr(self, 'results_window'):
            self.results_window.disable_cancel()


def main():
    """Main entry point for the optimization GUI."""
    app = CTkOptimizationWizard()
    app.run()


if __name__ == "__main__":
    main()
