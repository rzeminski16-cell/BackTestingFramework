"""
Backtesting Framework GUI (CustomTkinter Wizard)

A modern step-by-step graphical user interface for running backtests with a dark/blue theme.

Steps:
1. Mode & Securities - Choose single/portfolio mode and select securities
2. Strategy - Select and configure strategy parameters
3. Capital & Costs - Set initial capital, commission, and slippage
4. Time Period - Optional date range filtering
5. Review & Run - Verify all settings and run backtest

Features:
- Modern CustomTkinter dark/blue theme
- Clear wizard-based flow for easy configuration
- Reusable components (SecuritySelector, StrategySelector, etc.)
- Selection summary visible at each step
- Review step before running to verify settings
- Real-time validation with helpful messages
"""

import customtkinter as ctk
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import yaml
import threading
import queue

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes, ask_yes_no, show_error, ProgressPanel
from Classes.GUI.ctk_wizard_base import CTkWizardBase, CTkWizardStep, CTkReviewStep, CTkProgressDialog
from Classes.GUI.ctk_components import SecuritySelector, StrategySelector, DatePeriodSelector, CapitalCostsPanel

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
from Classes.Analysis.trade_logger import TradeLogger, PortfolioTradeLogger
from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
from Classes.Analysis.portfolio_report_generator import PortfolioReportGenerator

# Import available strategies
from strategies.alphatrend_strategy import AlphaTrendStrategy
from strategies.random_base_strategy import RandomBaseStrategy


# =============================================================================
# CTK VULNERABILITY SCORE CONFIG DIALOG
# =============================================================================

class CTkVulnerabilityConfigDialog(ctk.CTkToplevel):
    """CustomTkinter dialog for configuring vulnerability score parameters."""

    def __init__(self, parent, current_config: VulnerabilityScoreConfig = None, on_save=None):
        super().__init__(parent)

        self.current_config = current_config or VulnerabilityScoreConfig()
        self.on_save = on_save

        # Window setup
        self.title("Vulnerability Score Configuration")
        self.geometry("500x500")
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        x = parent_x + (parent_w - 500) // 2
        y = parent_y + (parent_h - 500) // 2
        self.geometry(f"+{x}+{y}")

        self.configure(fg_color=Colors.BG_DARK)
        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        Theme.create_header(main_frame, "Vulnerability Score Parameters", size="m").pack(pady=(0, 15))

        # Parameters frame
        params_card = Theme.create_card(main_frame)
        params_card.pack(fill="x", pady=(0, 15))

        params_frame = Theme.create_frame(params_card)
        params_frame.pack(fill="x", padx=15, pady=15)

        # Store entry variables
        self.vars = {}

        # Immunity Days
        self._add_param_row(params_frame, "Immunity Days:", "immunity_days",
                          str(self.current_config.immunity_days), "1-30 days (new trades protected)")

        # Min Profit Threshold
        self._add_param_row(params_frame, "Min Profit Threshold:", "min_profit_threshold",
                          str(self.current_config.min_profit_threshold), "0.0-0.20 (0.02 = 2%)")

        # Decay Rate Fast
        self._add_param_row(params_frame, "Decay Rate (Fast):", "decay_rate_fast",
                          str(self.current_config.decay_rate_fast), "Points/day for stagnant trades")

        # Decay Rate Slow
        self._add_param_row(params_frame, "Decay Rate (Slow):", "decay_rate_slow",
                          str(self.current_config.decay_rate_slow), "Points/day for performing trades")

        # Swap Threshold
        self._add_param_row(params_frame, "Swap Threshold:", "swap_threshold",
                          str(self.current_config.swap_threshold), "0-100 (score below = vulnerable)")

        # Description card
        desc_card = Theme.create_card(main_frame)
        desc_card.pack(fill="x", pady=(0, 15))

        desc_frame = Theme.create_frame(desc_card)
        desc_frame.pack(fill="x", padx=15, pady=15)

        Theme.create_label(desc_frame, "How It Works:", font=Fonts.LABEL_BOLD).pack(anchor="w")

        desc_text = (
            "When a new BUY signal arrives with no capital:\n"
            "1. Each open position gets a vulnerability score (0-100)\n"
            "2. New trades start at 100, protected during immunity period\n"
            "3. After immunity, stagnant trades (low P/L) decay faster\n"
            "4. If weakest position < swap threshold, it's closed for new signal"
        )
        Theme.create_label(desc_frame, desc_text, font=Fonts.BODY_S,
                          text_color=Colors.TEXT_SECONDARY).pack(anchor="w", pady=(5, 0))

        # Buttons
        btn_frame = Theme.create_frame(main_frame)
        btn_frame.pack(fill="x")

        Theme.create_button(btn_frame, "Reset to Defaults",
                           command=self._reset_defaults, style="ghost").pack(side="left")

        Theme.create_button(btn_frame, "Cancel",
                           command=self.destroy, style="secondary").pack(side="right", padx=(10, 0))

        Theme.create_button(btn_frame, "Save",
                           command=self._save).pack(side="right")

    def _add_param_row(self, parent, label: str, key: str, value: str, hint: str):
        """Add a parameter input row."""
        row = Theme.create_frame(parent)
        row.pack(fill="x", pady=5)

        Theme.create_label(row, label, font=Fonts.BODY_S, width=150).pack(side="left")

        var = ctk.StringVar(value=value)
        self.vars[key] = var

        entry = Theme.create_entry(row, width=100)
        entry.configure(textvariable=var)
        entry.pack(side="left", padx=(10, 10))

        Theme.create_label(row, hint, font=Fonts.BODY_XS,
                          text_color=Colors.TEXT_MUTED).pack(side="left")

    def _reset_defaults(self):
        """Reset to default values."""
        defaults = VulnerabilityScoreConfig()
        self.vars['immunity_days'].set(str(defaults.immunity_days))
        self.vars['min_profit_threshold'].set(str(defaults.min_profit_threshold))
        self.vars['decay_rate_fast'].set(str(defaults.decay_rate_fast))
        self.vars['decay_rate_slow'].set(str(defaults.decay_rate_slow))
        self.vars['swap_threshold'].set(str(defaults.swap_threshold))

    def _save(self):
        """Save configuration and close dialog."""
        try:
            config = VulnerabilityScoreConfig(
                immunity_days=int(self.vars['immunity_days'].get()),
                min_profit_threshold=float(self.vars['min_profit_threshold'].get()),
                decay_rate_fast=float(self.vars['decay_rate_fast'].get()),
                decay_rate_slow=float(self.vars['decay_rate_slow'].get()),
                swap_threshold=float(self.vars['swap_threshold'].get())
            )

            if self.on_save:
                self.on_save(config)

            self.destroy()

        except ValueError as e:
            show_error(self, "Invalid Input", str(e))


# =============================================================================
# WIZARD STEPS
# =============================================================================

class CTkModeSecuritiesStep(CTkWizardStep):
    """Step 1: Select backtest mode and securities."""

    def __init__(self, wizard: 'CTkBacktestWizard'):
        super().__init__(wizard, "Mode & Securities")
        self.wizard: CTkBacktestWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)

        # Mode Selection Card
        mode_card = Theme.create_card(parent)
        mode_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        mode_content = Theme.create_frame(mode_card)
        mode_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(mode_content, "Backtest Mode", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

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
            "Test strategy on one security at a time"
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
            "Test with shared capital across multiple securities"
        ).pack(anchor="w", padx=(Sizes.PAD_XL, 0))

        # Securities Selection
        securities_card = Theme.create_card(parent)
        securities_card.pack(fill="both", expand=True, pady=(0, Sizes.PAD_M))

        securities_content = Theme.create_frame(securities_card)
        securities_content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        self.security_selector = SecuritySelector(
            securities_content,
            available_securities=self.wizard.available_securities,
            title="Select Securities",
            multi_select=False,  # Start with single mode
            show_search=True,
            show_buttons=True,
            height=250,
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

        Theme.create_button(
            basket_frame, "Manage Baskets...",
            command=self._open_basket_manager,
            style="secondary",
            width=130
        ).pack(side="left")

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
        ).pack(side="left", padx=(0, Sizes.PAD_M))

        self.vuln_config_btn = Theme.create_button(
            contention_frame, "Configure...",
            command=self._open_vulnerability_config,
            style="ghost",
            width=100
        )
        # Hidden initially

    def _on_mode_change(self):
        """Handle mode change between single and portfolio."""
        mode = self.wizard.mode_var.get()
        if mode == "portfolio":
            self.security_selector.multi_select = True
            self.security_selector.update_securities(self.wizard.available_securities)
            self.portfolio_card.pack(fill="x", pady=(0, Sizes.PAD_M))
        else:
            self.security_selector.multi_select = False
            # Keep only first selection if multiple were selected
            selected = self.security_selector.get_selected()
            self.security_selector.update_securities(self.wizard.available_securities)
            if selected:
                self.security_selector.set_selected([selected[0]])
            self.portfolio_card.pack_forget()

    def _on_selection_change(self, selected: List[str]):
        """Handle security selection change."""
        pass  # Count is updated by SecuritySelector

    def _refresh_basket_list(self):
        baskets = self.wizard.basket_manager.list_baskets()
        self.basket_combo.configure(values=["(Select securities manually)"] + baskets)

    def _on_basket_selected(self, basket_name: str):
        if basket_name == "(Select securities manually)":
            self.wizard.selected_basket = None
            return

        basket = self.wizard.basket_manager.load(basket_name)
        if basket:
            self.wizard.selected_basket = basket
            self.security_selector.set_selected(basket.securities)

            # Load basket's capital contention settings
            if basket.default_capital_contention:
                if basket.default_capital_contention.mode == CapitalContentionMode.VULNERABILITY_SCORE:
                    self.wizard.contention_mode_var.set("vulnerability")
                    self.wizard.vulnerability_config = basket.default_capital_contention.vulnerability_config
                else:
                    self.wizard.contention_mode_var.set("default")
                self._on_contention_mode_change()

    def _open_basket_manager(self):
        """Open basket manager dialog (placeholder - needs CTK conversion)."""
        # TODO: Convert BasketManagerDialog to CTK
        from Classes.GUI.basket_manager_dialog import BasketManagerDialog
        import tkinter as tk

        def on_basket_selected(basket):
            if basket:
                self.wizard.selected_basket = basket
                self._refresh_basket_list()
                self.wizard.basket_var.set(basket.name)
                self._on_basket_selected(basket.name)

        # Create a temporary tk root for the old dialog
        temp_root = tk.Toplevel()
        temp_root.withdraw()
        BasketManagerDialog(
            temp_root,
            self.wizard.available_securities,
            on_basket_selected=on_basket_selected
        )

    def _on_contention_mode_change(self):
        mode = self.wizard.contention_mode_var.get()
        if mode == "vulnerability":
            self.vuln_config_btn.pack(side="left", padx=Sizes.PAD_S)
            self.wizard.capital_contention_config = CapitalContentionConfig(
                mode=CapitalContentionMode.VULNERABILITY_SCORE,
                vulnerability_config=self.wizard.vulnerability_config
            )
        else:
            self.vuln_config_btn.pack_forget()
            self.wizard.capital_contention_config = CapitalContentionConfig.default_mode()

    def _open_vulnerability_config(self):
        """Open vulnerability config dialog using CustomTkinter."""
        def on_save(config):
            self.wizard.vulnerability_config = config
            self.wizard.capital_contention_config = CapitalContentionConfig(
                mode=CapitalContentionMode.VULNERABILITY_SCORE,
                vulnerability_config=config
            )

        CTkVulnerabilityConfigDialog(
            self.wizard.root,
            current_config=self.wizard.vulnerability_config,
            on_save=on_save
        )

    def on_enter(self):
        """Restore security selections when returning to this step."""
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

        # Store selected securities
        self.wizard.selected_securities = selections
        return True


class CTkStrategyStep(CTkWizardStep):
    """Step 2: Select and configure strategy."""

    def __init__(self, wizard: 'CTkBacktestWizard'):
        super().__init__(wizard, "Strategy")
        self.wizard: CTkBacktestWizard = wizard
        self.param_entries: Dict[str, ctk.StringVar] = {}

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)

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

        # Preset management
        preset_frame = Theme.create_frame(selection_content)
        preset_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        Theme.create_label(preset_frame, "Load Preset:").pack(side="left", padx=(0, Sizes.PAD_S))
        self.preset_var = ctk.StringVar()
        self.preset_combo = Theme.create_combobox(
            preset_frame,
            values=[],
            variable=self.preset_var,
            width=180
        )
        self.preset_combo.pack(side="left", padx=(0, Sizes.PAD_S))

        Theme.create_button(preset_frame, "Load", command=self._load_preset, style="secondary", width=70).pack(side="left", padx=2)
        Theme.create_button(preset_frame, "Save As...", command=self._save_preset, style="secondary", width=85).pack(side="left", padx=2)
        Theme.create_button(preset_frame, "Delete", command=self._delete_preset, style="ghost", width=70).pack(side="left", padx=2)

        # Parameters Card
        params_card = Theme.create_card(parent)
        params_card.pack(fill="both", expand=True)

        params_header = Theme.create_frame(params_card)
        params_header.pack(fill="x", padx=Sizes.PAD_L, pady=(Sizes.PAD_M, 0))
        Theme.create_header(params_header, "Strategy Parameters", size="s").pack(anchor="w")

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

    def _on_strategy_change(self, strategy_name: str):
        """Handle strategy selection change."""
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
                'risk_percent': 2.0,
                'atr_multiplier': 1.0,
                'smoothing_length': 3,
                'percentile_period': 100
            }
        elif strategy_name == 'RandomBaseStrategy':
            self.wizard.strategy_params[strategy_name] = {
                'entry_probability': 0.10,
                'exit_probability': 0.10,
                'position_size_pct': 5.0,
                'stop_loss_atr_multiple': 2.0
            }

        self._refresh_presets()
        self._build_param_ui()

    def _refresh_presets(self):
        strategy_name = self.wizard.strategy_var.get()
        if strategy_name:
            presets = self.wizard.preset_manager.list_presets(strategy_name)
            preset_names = [p['preset_name'] for p in presets]
            self.preset_combo.configure(values=preset_names)

    def _build_param_ui(self):
        """Build the parameter editing UI."""
        # Clear existing
        for widget in self.params_scroll.winfo_children():
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

            # Parameters
            for param_name, param_value in param_list:
                spec = param_spec.get(param_name, {})
                min_val = spec.get('min', '')
                max_val = spec.get('max', '')
                param_type = spec.get('type', 'float')

                param_frame = Theme.create_frame(self.params_scroll)
                param_frame.pack(fill="x", pady=Sizes.PAD_XS, padx=(Sizes.PAD_M, 0))

                Theme.create_label(param_frame, f"{param_name}:", width=180).pack(side="left")

                var = ctk.StringVar(value=str(param_value))
                self.param_entries[param_name] = var
                entry = Theme.create_entry(param_frame, width=100)
                entry.configure(textvariable=var)
                entry.pack(side="left", padx=(0, Sizes.PAD_S))

                if min_val != '' and max_val != '':
                    if param_type == 'int':
                        range_str = f"[{int(min_val)} - {int(max_val)}]"
                    else:
                        range_str = f"[{min_val:.2f} - {max_val:.2f}]"
                    Theme.create_hint(param_frame, range_str).pack(side="left")

    def _load_preset(self):
        preset_name = self.preset_var.get()
        strategy_name = self.wizard.strategy_var.get()
        if not preset_name:
            show_error(self.wizard.root, "No Preset", "Please select a preset to load.")
            return

        params = self.wizard.preset_manager.get_preset_parameters(strategy_name, preset_name)
        if params:
            for name, value in params.items():
                if name in self.param_entries:
                    self.param_entries[name].set(str(value))

    def _save_preset(self):
        # Simple input dialog
        dialog = CTkInputDialog(
            self.wizard.root,
            title="Save Preset",
            prompt="Enter preset name:"
        )
        preset_name = dialog.get_input()

        if not preset_name:
            return

        strategy_name = self.wizard.strategy_var.get()
        try:
            params = self._get_current_params()
            self.wizard.preset_manager.save_preset(strategy_name, preset_name, params, "")
            self._refresh_presets()
        except ValueError as e:
            show_error(self.wizard.root, "Error", f"Invalid parameter value: {e}")

    def _delete_preset(self):
        preset_name = self.preset_var.get()
        strategy_name = self.wizard.strategy_var.get()
        if not preset_name:
            show_error(self.wizard.root, "No Preset", "Please select a preset to delete.")
            return

        if ask_yes_no(self.wizard.root, "Confirm Delete", f"Delete preset '{preset_name}'?"):
            if self.wizard.preset_manager.delete_preset(strategy_name, preset_name):
                self._refresh_presets()
                self.preset_var.set('')

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
            pass

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


class CTkCapitalCostsStep(CTkWizardStep):
    """Step 3: Configure capital and costs."""

    def __init__(self, wizard: 'CTkBacktestWizard'):
        super().__init__(wizard, "Capital & Costs")
        self.wizard: CTkBacktestWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)

        # Initial Capital Card
        capital_card = Theme.create_card(parent)
        capital_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        capital_content = Theme.create_frame(capital_card)
        capital_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(capital_content, "Initial Capital", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        cap_frame = Theme.create_frame(capital_content)
        cap_frame.pack(fill="x")

        Theme.create_label(cap_frame, "Amount:").pack(side="left", padx=(0, Sizes.PAD_S))
        self.wizard.capital_var = ctk.StringVar(value="100000")
        capital_entry = Theme.create_entry(cap_frame, width=150)
        capital_entry.configure(textvariable=self.wizard.capital_var)
        capital_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        Theme.create_hint(cap_frame, "GBP").pack(side="left")

        # Commission Card
        commission_card = Theme.create_card(parent)
        commission_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        commission_content = Theme.create_frame(commission_card)
        commission_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(commission_content, "Commission", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Commission Mode
        mode_frame = Theme.create_frame(commission_content)
        mode_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_label(mode_frame, "Mode:").pack(side="left", padx=(0, Sizes.PAD_S))
        self.wizard.commission_mode_var = ctk.StringVar(value="percentage")
        Theme.create_radiobutton(
            mode_frame, "Percentage",
            variable=self.wizard.commission_mode_var, value="percentage",
            command=self._update_commission_hint
        ).pack(side="left", padx=(0, Sizes.PAD_M))
        Theme.create_radiobutton(
            mode_frame, "Fixed Amount",
            variable=self.wizard.commission_mode_var, value="fixed",
            command=self._update_commission_hint
        ).pack(side="left")

        # Commission Value
        value_frame = Theme.create_frame(commission_content)
        value_frame.pack(fill="x")

        Theme.create_label(value_frame, "Value:").pack(side="left", padx=(0, Sizes.PAD_S))
        self.wizard.commission_value_var = ctk.StringVar(value="0.001")
        comm_entry = Theme.create_entry(value_frame, width=100)
        comm_entry.configure(textvariable=self.wizard.commission_value_var)
        comm_entry.pack(side="left", padx=(0, Sizes.PAD_S))

        self.commission_hint_label = Theme.create_hint(value_frame, "(0.1% per trade)")
        self.commission_hint_label.pack(side="left")

        # Slippage Card
        slippage_card = Theme.create_card(parent)
        slippage_card.pack(fill="x")

        slippage_content = Theme.create_frame(slippage_card)
        slippage_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(slippage_content, "Slippage", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        slip_frame = Theme.create_frame(slippage_content)
        slip_frame.pack(fill="x")

        Theme.create_label(slip_frame, "Slippage (%):").pack(side="left", padx=(0, Sizes.PAD_S))
        self.wizard.slippage_var = ctk.StringVar(value="0.1")
        slip_entry = Theme.create_entry(slip_frame, width=100)
        slip_entry.configure(textvariable=self.wizard.slippage_var)
        slip_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        Theme.create_hint(slip_frame, "(Simulates price impact when entering/exiting positions)").pack(side="left")

    def _update_commission_hint(self):
        mode = self.wizard.commission_mode_var.get()
        try:
            value = float(self.wizard.commission_value_var.get())
            if mode == "percentage":
                self.commission_hint_label.configure(text=f"({value * 100:.2f}% per trade)")
            else:
                self.commission_hint_label.configure(text=f"(${value:.2f} per trade)")
        except ValueError:
            self.commission_hint_label.configure(text="(enter a valid number)")

    def get_summary(self) -> Dict[str, str]:
        try:
            capital = float(self.wizard.capital_var.get())
            capital_str = f"${capital:,.2f}"
        except ValueError:
            capital_str = self.wizard.capital_var.get()

        mode = self.wizard.commission_mode_var.get()
        try:
            comm_val = float(self.wizard.commission_value_var.get())
            if mode == "percentage":
                comm_str = f"{comm_val * 100:.2f}%"
            else:
                comm_str = f"${comm_val:.2f}"
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


class CTkTimePeriodStep(CTkWizardStep):
    """Step 4: Configure time period."""

    def __init__(self, wizard: 'CTkBacktestWizard'):
        super().__init__(wizard, "Time Period")
        self.wizard: CTkBacktestWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)

        # Info text
        Theme.create_label(
            parent,
            "Optionally filter the backtest to a specific date range.\nLeave blank to use all available data.",
            text_color=Colors.TEXT_SECONDARY,
            wraplength=600
        ).pack(anchor="w", pady=(0, Sizes.PAD_L))

        # Date Range Card
        date_card = Theme.create_card(parent)
        date_card.pack(fill="x")

        date_content = Theme.create_frame(date_card)
        date_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_header(date_content, "Date Range", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Start Date
        start_frame = Theme.create_frame(date_content)
        start_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(start_frame, "Start Date:", width=100).pack(side="left")
        self.wizard.start_date_var = ctk.StringVar(value="")
        start_entry = Theme.create_entry(start_frame, placeholder="YYYY-MM-DD", width=150)
        start_entry.configure(textvariable=self.wizard.start_date_var)
        start_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        Theme.create_hint(start_frame, "(or leave blank)").pack(side="left")

        # End Date
        end_frame = Theme.create_frame(date_content)
        end_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(end_frame, "End Date:", width=100).pack(side="left")
        self.wizard.end_date_var = ctk.StringVar(value="")
        end_entry = Theme.create_entry(end_frame, placeholder="YYYY-MM-DD", width=150)
        end_entry.configure(textvariable=self.wizard.end_date_var)
        end_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        Theme.create_hint(end_frame, "(or leave blank)").pack(side="left")

        # Data range info
        self.data_range_label = Theme.create_label(
            date_content, "",
            text_color=Colors.PRIMARY_LIGHT
        )
        self.data_range_label.pack(anchor="w", pady=(Sizes.PAD_L, 0))

    def on_enter(self):
        """Update data range info when entering this step."""
        if hasattr(self.wizard, 'selected_securities') and self.wizard.selected_securities:
            try:
                symbol = self.wizard.selected_securities[0]
                data = self.wizard.data_loader.load_csv(symbol)
                start = data['date'].min().strftime('%Y-%m-%d')
                end = data['date'].max().strftime('%Y-%m-%d')
                self.data_range_label.configure(
                    text=f"Available data for {symbol}: {start} to {end}"
                )
            except Exception:
                self.data_range_label.configure(text="")

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


class CTkBacktestReviewStep(CTkReviewStep):
    """Step 5: Review and run backtest."""

    def __init__(self, wizard: 'CTkBacktestWizard'):
        super().__init__(wizard, "Review & Run")
        self.wizard: CTkBacktestWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        super().create_widgets(parent)

        # Add backtest options at the bottom
        options_card = Theme.create_card(parent)
        options_card.pack(fill="x", pady=(Sizes.PAD_M, 0), side="bottom")

        options_content = Theme.create_frame(options_card)
        options_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(options_content, "Backtest Options", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Backtest name
        name_frame = Theme.create_frame(options_content)
        name_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(name_frame, "Backtest Name:", width=130).pack(side="left")
        self.wizard.backtest_name_var = ctk.StringVar(value="test_1")
        name_entry = Theme.create_entry(name_frame, width=250)
        name_entry.configure(textvariable=self.wizard.backtest_name_var)
        name_entry.pack(side="left")

        # Excel report option
        self.wizard.generate_excel_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            options_content, "Generate Excel Report",
            variable=self.wizard.generate_excel_var
        ).pack(anchor="w", pady=Sizes.PAD_S)


# =============================================================================
# INPUT DIALOG
# =============================================================================

class CTkInputDialog(ctk.CTkToplevel):
    """Simple input dialog."""

    def __init__(self, parent, title: str, prompt: str):
        super().__init__(parent)

        self.title(title)
        self.geometry("350x150")
        self.resizable(False, False)
        self.configure(fg_color=Colors.BG_DARK)

        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 350) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 150) // 2
        self.geometry(f"+{x}+{y}")

        self._input_value = None

        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_label(content, prompt).pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.entry = Theme.create_entry(content)
        self.entry.pack(fill="x", pady=Sizes.PAD_S)
        self.entry.focus()
        self.entry.bind("<Return>", lambda e: self._on_ok())

        btn_frame = Theme.create_frame(content)
        btn_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        Theme.create_button(btn_frame, "Cancel", command=self._on_cancel, style="secondary", width=80).pack(side="right")
        Theme.create_button(btn_frame, "OK", command=self._on_ok, width=80).pack(side="right", padx=(0, Sizes.PAD_S))

        self.wait_window()

    def _on_ok(self):
        self._input_value = self.entry.get().strip()
        self.destroy()

    def _on_cancel(self):
        self._input_value = None
        self.destroy()

    def get_input(self) -> Optional[str]:
        return self._input_value


# =============================================================================
# RESULTS WINDOW
# =============================================================================

class CTkResultsWindow(ctk.CTkToplevel):
    """Modern results window for displaying backtest results."""

    def __init__(self, parent, title: str, settings: Dict[str, Dict[str, str]]):
        super().__init__(parent)

        self.title(title)
        self.geometry("900x700")
        self.configure(fg_color=Colors.BG_DARK)

        # Create layout
        self._create_layout(settings)

    def _create_layout(self, settings: Dict[str, Dict[str, str]]):
        # Main container
        main_frame = Theme.create_frame(self)
        main_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Configuration summary (collapsible)
        config_card = Theme.create_card(main_frame)
        config_card.grid(row=0, column=0, sticky="ew", pady=(0, Sizes.PAD_M))

        config_header = Theme.create_frame(config_card)
        config_header.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)
        Theme.create_header(config_header, "Configuration", size="s").pack(side="left")

        config_content = Theme.create_frame(config_card)
        config_content.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Display settings in compact format
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

        self.progress_panel = ProgressPanel(progress_content, show_log=True, log_height=400)
        self.progress_panel.pack(fill="both", expand=True)

        # Close button
        btn_frame = Theme.create_frame(main_frame)
        btn_frame.grid(row=2, column=0, sticky="e", pady=(Sizes.PAD_M, 0))

        Theme.create_button(btn_frame, "Close", command=self.destroy, style="secondary", width=100).pack(side="right")

    def log(self, message: str):
        """Add a message to the log."""
        self.progress_panel.log(message)

    def update_progress(self, current: int, total: int, detail: str = ""):
        """Update progress bar."""
        progress = current / total if total > 0 else 0
        self.progress_panel.update_progress(progress, detail)

    def on_complete(self):
        """Mark as complete."""
        self.progress_panel.set_complete("Backtest Complete")

    def on_error(self, error: str):
        """Show error."""
        self.progress_panel.log(f"ERROR: {error}", level="error")
        self.progress_panel.status_label.configure(text="Error", text_color=Colors.ERROR)


# =============================================================================
# MAIN WIZARD CLASS
# =============================================================================

class CTkBacktestWizard(CTkWizardBase):
    """Main Backtest Wizard application using CustomTkinter."""

    STRATEGIES = {
        'AlphaTrendStrategy': AlphaTrendStrategy,
        'RandomBaseStrategy': RandomBaseStrategy
    }

    def __init__(self):
        super().__init__(
            title="Backtesting Framework",
            width=1100,
            height=750
        )

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/daily'))
        self.available_securities = self.data_loader.get_available_symbols()

        # Currency support
        self.security_registry = SecurityRegistry(Path('config/security_metadata.json'))
        self.currency_converter = CurrencyConverter(base_currency='GBP')
        try:
            self.currency_converter.load_rates_directory(Path('currency_rates/'))
        except Exception as e:
            print(f"Warning: Could not load currency rates: {e}")

        # Strategy parameters cache
        self.strategy_params: Dict[str, Dict[str, Any]] = {}

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

        # Variables (will be set by steps)
        self.mode_var: Optional[ctk.StringVar] = None
        self.strategy_var: Optional[ctk.StringVar] = None
        self.capital_var: Optional[ctk.StringVar] = None
        self.commission_mode_var: Optional[ctk.StringVar] = None
        self.commission_value_var: Optional[ctk.StringVar] = None
        self.slippage_var: Optional[ctk.StringVar] = None
        self.start_date_var: Optional[ctk.StringVar] = None
        self.end_date_var: Optional[ctk.StringVar] = None
        self.backtest_name_var: Optional[ctk.StringVar] = None
        self.generate_excel_var: Optional[ctk.BooleanVar] = None
        self.basket_var: Optional[ctk.StringVar] = None
        self.contention_mode_var: Optional[ctk.StringVar] = None
        self.selected_securities: List[str] = []

        # Add wizard steps
        self.add_step(CTkModeSecuritiesStep(self))
        self.add_step(CTkStrategyStep(self))
        self.add_step(CTkCapitalCostsStep(self))
        self.add_step(CTkTimePeriodStep(self))
        self.add_step(CTkBacktestReviewStep(self))

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
        if ask_yes_no(self.root, "Cancel", "Are you sure you want to cancel?"):
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
            results_window = CTkResultsWindow(
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
                            return
                        elif msg_type == "error":
                            results_window.on_error(data)
                            return
                except queue.Empty:
                    pass
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
            show_error(self.root, "Error", f"Backtest failed: {str(e)}")
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
        msg_queue.put(("log", f"Capital: ${capital:,.2f}\n"))

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
        msg_queue.put(("log", f"Shared Capital: ${capital:,.2f}"))
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

    def _display_single_results_threaded(self, msg_queue: queue.Queue, symbol: str,
                                          result, backtest_name: str, strategy_params: Dict):
        """Display single backtest results via message queue."""
        metrics = PerformanceMetrics.calculate_metrics(result)

        msg_queue.put(("log", "\n" + "=" * 60))
        msg_queue.put(("log", f"RESULTS: {symbol}"))
        msg_queue.put(("log", "=" * 60))
        msg_queue.put(("log", f"Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)"))
        msg_queue.put(("log", f"Final Equity:        ${metrics['final_equity']:,.2f}"))
        msg_queue.put(("log", f"Number of Trades:    {metrics['num_trades']}"))
        msg_queue.put(("log", f"Win Rate:            {metrics['win_rate']*100:.2f}%"))
        msg_queue.put(("log", f"Winning Trades:      {metrics['num_wins']}"))
        msg_queue.put(("log", f"Losing Trades:       {metrics['num_losses']}"))
        msg_queue.put(("log", f"Average Win:         ${metrics['avg_win']:,.2f}"))
        msg_queue.put(("log", f"Average Loss:        ${metrics['avg_loss']:,.2f}"))
        msg_queue.put(("log", f"Largest Win:         ${metrics['largest_win']:,.2f}"))
        msg_queue.put(("log", f"Largest Loss:        ${metrics['largest_loss']:,.2f}"))
        msg_queue.put(("log", f"Profit Factor:       {metrics['profit_factor']:.2f}"))
        msg_queue.put(("log", f"Avg Trade Duration:  {metrics['avg_trade_duration']:.1f} days"))
        msg_queue.put(("log", f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}"))
        msg_queue.put(("log", f"Max Drawdown:        ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)"))
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

    def _display_portfolio_results_threaded(self, msg_queue: queue.Queue,
                                             result: PortfolioBacktestResult, backtest_name: str,
                                             strategy_params: Dict):
        """Display portfolio backtest results via message queue."""
        msg_queue.put(("log", "\n" + "=" * 60))
        msg_queue.put(("log", "PORTFOLIO RESULTS"))
        msg_queue.put(("log", "=" * 60))

        msg_queue.put(("log", f"\nInitial Capital:     ${result.config.initial_capital:,.2f}"))
        msg_queue.put(("log", f"Final Equity:        ${result.final_equity:,.2f}"))
        msg_queue.put(("log", f"Total Return:        ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)"))

        msg_queue.put(("log", "\n--- Per-Security Performance ---"))
        total_trades = 0
        for symbol, sym_result in result.symbol_results.items():
            num_trades = len(sym_result.trades)
            total_trades += num_trades
            wins = len([t for t in sym_result.trades if t.pl > 0])
            win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
            msg_queue.put(("log", f"  {symbol}: {num_trades} trades, Win Rate: {win_rate:.1f}%, P/L: ${sym_result.total_return:,.2f}"))

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


def main():
    """Main entry point for GUI."""
    app = CTkBacktestWizard()
    app.run()


if __name__ == "__main__":
    main()
