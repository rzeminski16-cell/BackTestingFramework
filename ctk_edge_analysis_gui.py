"""
Edge Analysis GUI (CustomTkinter Wizard)

A specialized backtesting GUI for analyzing entry edge and R-multiples.

Key Features:
- Unlimited capital mode (infinite concurrent trades)
- Position sizing based on £100k account
- No Excel reports generated
- Two visualization tabs:
  1. E-ratio (Edge Ratio) over different time horizons
  2. R-multiple distribution for winning and losing trades

E-ratio Formula:
    E-ratio = Avg(MFE / ATR) / Avg(MAE / ATR)
    Where:
    - MFE = Maximum Favorable Excursion (max gain from entry)
    - MAE = Maximum Adverse Excursion (max loss from entry)
    - ATR = Average True Range at entry (volatility normalization)

R-multiple Formula:
    R-multiple = (Exit Price - Entry Price) / (Entry Price - Stop Loss)
    Expresses profit/loss as a multiple of initial risk (R)
"""

import customtkinter as ctk
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes, ask_yes_no, show_error, ProgressPanel
from Classes.GUI.ctk_wizard_base import CTkWizardBase, CTkWizardStep, CTkReviewStep
from Classes.GUI.ctk_components import SecuritySelector, DatePeriodSelector

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Config.strategy_preset import StrategyParameterPreset
from Classes.Config.basket import BasketManager
from Classes.Data.data_loader import DataLoader
from Classes.Data.security_registry import SecurityRegistry
from Classes.Data.currency_converter import CurrencyConverter

from strategies.alphatrend_strategy import AlphaTrendStrategy


# =============================================================================
# EDGE ANALYSIS ENGINE
# =============================================================================

class EdgeAnalysisResult:
    """Container for edge analysis results."""

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.e_ratios: Dict[int, float] = {}  # n_days -> e_ratio
        self.r_multiples: List[float] = []
        self.winning_r_multiples: List[float] = []
        self.losing_r_multiples: List[float] = []
        self.mfe_mae_data: List[Dict[str, Any]] = []  # Per-trade MFE/MAE data


class UnlimitedCapitalEngine:
    """
    Simplified backtesting engine with unlimited capital.

    Runs all trades regardless of capital constraints.
    Position sizing is calculated using a fixed £100k account.
    """

    ACCOUNT_SIZE = 100000.0  # £100k for position sizing

    def __init__(self, data_loader: DataLoader,
                 security_registry: Optional[SecurityRegistry] = None,
                 currency_converter: Optional[CurrencyConverter] = None):
        self.data_loader = data_loader
        self.security_registry = security_registry
        self.currency_converter = currency_converter

    def run_backtest(self, symbols: List[str], strategy,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     max_e_ratio_days: int = 20,
                     progress_callback: Optional[callable] = None) -> EdgeAnalysisResult:
        """
        Run backtest with unlimited capital and collect edge analysis data.

        Args:
            symbols: List of security symbols
            strategy: Trading strategy instance
            start_date: Optional start date filter
            end_date: Optional end date filter
            max_e_ratio_days: Maximum number of days for E-ratio calculation
            progress_callback: Optional callback(current, total) for progress

        Returns:
            EdgeAnalysisResult with all analysis data
        """
        result = EdgeAnalysisResult()
        all_trades = []

        total_symbols = len(symbols)

        for sym_idx, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(sym_idx, total_symbols)

            try:
                # Load data
                data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())

                # Filter by date range
                if start_date:
                    data = data[data['date'] >= start_date]
                if end_date:
                    data = data[data['date'] <= end_date]

                if len(data) < 50:  # Need enough data
                    continue

                # Prepare data with indicators
                data = strategy.prepare_data(data)

                # Find entry signals and collect trade data
                trades = self._find_trades(symbol, data, strategy, max_e_ratio_days)
                all_trades.extend(trades)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        if progress_callback:
            progress_callback(total_symbols, total_symbols)

        result.trades = all_trades

        # Calculate E-ratios for each time horizon
        for n_days in range(1, max_e_ratio_days + 1):
            e_ratio = self._calculate_e_ratio(all_trades, n_days)
            result.e_ratios[n_days] = e_ratio

        # Calculate R-multiples
        for trade in all_trades:
            if trade.get('r_multiple') is not None:
                r_mult = trade['r_multiple']
                result.r_multiples.append(r_mult)
                if r_mult >= 0:
                    result.winning_r_multiples.append(r_mult)
                else:
                    result.losing_r_multiples.append(r_mult)

        # Store MFE/MAE data
        result.mfe_mae_data = [{
            'symbol': t['symbol'],
            'entry_date': t['entry_date'],
            'entry_price': t['entry_price'],
            'atr_at_entry': t['atr_at_entry'],
            'mfe': t.get('mfe', {}),
            'mae': t.get('mae', {}),
        } for t in all_trades if t.get('atr_at_entry')]

        return result

    def _find_trades(self, symbol: str, data: pd.DataFrame, strategy,
                     max_lookforward: int) -> List[Dict[str, Any]]:
        """Find all entry signals and collect forward-looking price data."""
        trades = []

        from Classes.Strategy.strategy_context import StrategyContext
        from Classes.Models.signal import SignalType
        from Classes.Data.historical_data_view import HistoricalDataView

        # Iterate through data looking for entry signals
        for i in range(len(data) - max_lookforward - 1):
            current_bar = data.iloc[i]
            current_date = current_bar['date']
            current_price = current_bar['close']

            # Create historical data view (no look-ahead)
            historical_data = HistoricalDataView(data, valid_end_index=i)

            # Create context
            context = StrategyContext(
                data=historical_data,
                current_index=i,
                current_price=current_price,
                current_date=current_date,
                position=None,
                available_capital=self.ACCOUNT_SIZE,
                total_equity=self.ACCOUNT_SIZE,
                symbol=symbol,
                fx_rate=1.0
            )

            # Generate signal
            signal = strategy.generate_signal(context)

            if signal.type != SignalType.BUY:
                continue

            # Get ATR at entry for volatility normalization
            atr_col = None
            for col in ['atr_14_atr', 'atr_14', 'atr']:
                if col in data.columns:
                    atr_col = col
                    break

            atr_at_entry = data.iloc[i][atr_col] if atr_col else None

            if atr_at_entry is None or pd.isna(atr_at_entry) or atr_at_entry <= 0:
                continue

            # Calculate stop loss price for R-multiple
            stop_loss = signal.stop_loss
            if stop_loss is None:
                continue

            # Calculate position size for this trade
            quantity = strategy.position_size(context, signal)
            if quantity <= 0:
                continue

            # Calculate initial risk (R)
            initial_risk = (current_price - stop_loss) * quantity
            if initial_risk <= 0:
                continue

            # Collect forward-looking data for MFE/MAE
            forward_data = data.iloc[i+1:i+1+max_lookforward]

            mfe_by_day = {}  # day -> max favorable excursion
            mae_by_day = {}  # day -> max adverse excursion

            for day_offset in range(1, min(max_lookforward + 1, len(forward_data) + 1)):
                day_data = data.iloc[i+1:i+1+day_offset]
                if len(day_data) == 0:
                    break

                # MFE: highest high minus entry price
                max_high = day_data['high'].max()
                mfe = max(0, max_high - current_price)
                mfe_by_day[day_offset] = mfe

                # MAE: entry price minus lowest low
                min_low = day_data['low'].min()
                mae = max(0, current_price - min_low)
                mae_by_day[day_offset] = mae

            # Find the actual exit to calculate R-multiple
            # Simulate holding until strategy exit or end of data
            exit_price = None
            exit_date = None
            exit_reason = None

            for j in range(i + 1, len(data)):
                bar = data.iloc[j]

                # Check stop loss
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_date = bar['date']
                    exit_reason = 'stop_loss'
                    break

                # Create context for exit signal check
                hist_data = HistoricalDataView(data, valid_end_index=j)

                from Classes.Models.position import Position
                position = Position(
                    symbol=symbol,
                    entry_date=current_date,
                    entry_price=current_price,
                    initial_quantity=quantity,
                    current_quantity=quantity,
                    stop_loss=stop_loss
                )

                exit_context = StrategyContext(
                    data=hist_data,
                    current_index=j,
                    current_price=bar['close'],
                    current_date=bar['date'],
                    position=position,
                    available_capital=self.ACCOUNT_SIZE,
                    total_equity=self.ACCOUNT_SIZE,
                    symbol=symbol,
                    fx_rate=1.0
                )

                exit_signal = strategy.generate_signal(exit_context)
                if exit_signal.type == SignalType.SELL:
                    exit_price = bar['close']
                    exit_date = bar['date']
                    exit_reason = 'strategy_exit'
                    break

            # If no exit found, use last bar
            if exit_price is None:
                exit_price = data.iloc[-1]['close']
                exit_date = data.iloc[-1]['date']
                exit_reason = 'end_of_data'

            # Calculate R-multiple
            pl = (exit_price - current_price) * quantity
            r_multiple = pl / initial_risk if initial_risk > 0 else 0

            trade = {
                'symbol': symbol,
                'entry_date': current_date,
                'entry_price': current_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'stop_loss': stop_loss,
                'quantity': quantity,
                'atr_at_entry': atr_at_entry,
                'initial_risk': initial_risk,
                'pl': pl,
                'r_multiple': r_multiple,
                'mfe': mfe_by_day,
                'mae': mae_by_day,
            }

            trades.append(trade)

        return trades

    def _calculate_e_ratio(self, trades: List[Dict], n_days: int) -> float:
        """
        Calculate E-ratio for a specific time horizon.

        E-ratio = Avg(MFE/ATR) / Avg(MAE/ATR)

        Args:
            trades: List of trade dictionaries with MFE/MAE data
            n_days: Number of days for the time horizon

        Returns:
            E-ratio value (or 0 if cannot calculate)
        """
        mfe_atr_values = []
        mae_atr_values = []

        for trade in trades:
            atr = trade.get('atr_at_entry')
            mfe_dict = trade.get('mfe', {})
            mae_dict = trade.get('mae', {})

            if atr is None or atr <= 0:
                continue

            mfe = mfe_dict.get(n_days)
            mae = mae_dict.get(n_days)

            if mfe is not None and mae is not None:
                mfe_atr_values.append(mfe / atr)
                mae_atr_values.append(mae / atr)

        if not mfe_atr_values or not mae_atr_values:
            return 0.0

        avg_mfe_atr = sum(mfe_atr_values) / len(mfe_atr_values)
        avg_mae_atr = sum(mae_atr_values) / len(mae_atr_values)

        if avg_mae_atr <= 0:
            return 0.0

        return avg_mfe_atr / avg_mae_atr


# =============================================================================
# WIZARD STEPS
# =============================================================================

class CTkSecuritiesStep(CTkWizardStep):
    """Step 1: Select securities (portfolio mode only)."""

    def __init__(self, wizard: 'CTkEdgeAnalysisWizard'):
        super().__init__(wizard, "Select Securities")
        self.wizard: CTkEdgeAnalysisWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)

        # Info text
        info_card = Theme.create_card(parent)
        info_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        info_content = Theme.create_frame(info_card)
        info_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_label(
            info_content,
            "Edge Analysis Mode",
            font=Fonts.HEADER_M
        ).pack(anchor="w")

        Theme.create_label(
            info_content,
            "This analysis runs with unlimited capital - all entry signals will be taken.\n"
            "Position sizing uses a £100,000 account for R-multiple calculations.\n"
            "Select the securities to include in the analysis.",
            text_color=Colors.TEXT_SECONDARY,
            wraplength=600
        ).pack(anchor="w", pady=(Sizes.PAD_S, 0))

        # Securities Selection Card
        securities_card = Theme.create_card(parent)
        securities_card.pack(fill="both", expand=True, pady=(0, Sizes.PAD_M))

        securities_content = Theme.create_frame(securities_card)
        securities_content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        self.security_selector = SecuritySelector(
            securities_content,
            available_securities=self.wizard.available_securities,
            title="Select Securities",
            multi_select=True,
            show_search=True,
            show_buttons=True,
            height=300,
            on_selection_change=self._on_selection_change
        )
        self.security_selector.pack(fill="both", expand=True)

        # Basket selection
        basket_frame = Theme.create_frame(parent)
        basket_frame.pack(fill="x")

        Theme.create_label(basket_frame, "Load Basket:").pack(side="left", padx=(0, Sizes.PAD_S))

        self.wizard.basket_var = ctk.StringVar(value="(Select securities manually)")
        basket_values = ["(Select securities manually)"] + self.wizard.basket_manager.list_baskets()
        self.basket_combo = Theme.create_combobox(
            basket_frame,
            values=basket_values,
            variable=self.wizard.basket_var,
            command=self._on_basket_selected,
            width=250
        )
        self.basket_combo.pack(side="left")

    def _on_selection_change(self, selected: List[str]):
        pass

    def _on_basket_selected(self, basket_name: str):
        if basket_name == "(Select securities manually)":
            return

        basket = self.wizard.basket_manager.load(basket_name)
        if basket:
            self.security_selector.set_selected(basket.securities)

    def get_summary(self) -> Dict[str, str]:
        securities = self.security_selector.get_selected()
        if securities:
            if len(securities) <= 3:
                sec_str = ", ".join(securities)
            else:
                sec_str = f"{', '.join(securities[:3])}... ({len(securities)} total)"
        else:
            sec_str = "None selected"

        return {"Securities": sec_str}

    def validate(self) -> bool:
        self.validation_errors = []
        selections = self.security_selector.get_selected()

        if not selections:
            self.validation_errors.append("Please select at least one security.")
            return False

        self.wizard.selected_securities = selections
        return True


class CTkStrategyStep(CTkWizardStep):
    """Step 2: Select and configure strategy."""

    def __init__(self, wizard: 'CTkEdgeAnalysisWizard'):
        super().__init__(wizard, "Strategy")
        self.wizard: CTkEdgeAnalysisWizard = wizard
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

        # Parameters Card
        params_card = Theme.create_card(parent)
        params_card.pack(fill="both", expand=True)

        params_header = Theme.create_frame(params_card)
        params_header.pack(fill="x", padx=Sizes.PAD_L, pady=(Sizes.PAD_M, 0))
        Theme.create_header(params_header, "Strategy Parameters", size="s").pack(anchor="w")

        # Scrollable parameters area
        self.params_scroll = ctk.CTkScrollableFrame(params_card, fg_color="transparent")
        self.params_scroll.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        # Initialize with first strategy
        if strategy_names:
            self.wizard.strategy_var.set(strategy_names[0])
            self._on_strategy_change(strategy_names[0])

    def _on_strategy_change(self, strategy_name: str):
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
            preset_names = [p['preset_name'] for p in presets]
            self.preset_combo.configure(values=preset_names)

    def _build_param_ui(self):
        for widget in self.params_scroll.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        strategy_name = self.wizard.strategy_var.get()
        if strategy_name not in self.wizard.strategy_params:
            return

        params = self.wizard.strategy_params[strategy_name]

        for param_name, param_value in params.items():
            param_frame = Theme.create_frame(self.params_scroll)
            param_frame.pack(fill="x", pady=Sizes.PAD_XS)

            Theme.create_label(param_frame, f"{param_name}:", width=200).pack(side="left")

            var = ctk.StringVar(value=str(param_value))
            self.param_entries[param_name] = var
            entry = Theme.create_entry(param_frame, width=100)
            entry.configure(textvariable=var)
            entry.pack(side="left")

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

    def _get_current_params(self) -> Dict[str, Any]:
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
        try:
            strategy_name = self.wizard.strategy_var.get()
            self.wizard.strategy_params[strategy_name] = self._get_current_params()
        except (ValueError, KeyError):
            pass

    def get_summary(self) -> Dict[str, str]:
        strategy_name = self.wizard.strategy_var.get()
        return {"Strategy": strategy_name if strategy_name else "None selected"}

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


class CTkTimePeriodStep(CTkWizardStep):
    """Step 3: Configure time period and E-ratio settings."""

    def __init__(self, wizard: 'CTkEdgeAnalysisWizard'):
        super().__init__(wizard, "Time Period & Settings")
        self.wizard: CTkEdgeAnalysisWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)

        # Date Range Card
        date_card = Theme.create_card(parent)
        date_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        date_content = Theme.create_frame(date_card)
        date_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(date_content, "Date Range", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(
            date_content,
            "Optionally filter the backtest to a specific date range.\nLeave blank to use all available data.",
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Start Date
        start_frame = Theme.create_frame(date_content)
        start_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(start_frame, "Start Date:", width=100).pack(side="left")
        self.wizard.start_date_var = ctk.StringVar(value="")
        start_entry = Theme.create_entry(start_frame, placeholder="YYYY-MM-DD", width=150)
        start_entry.configure(textvariable=self.wizard.start_date_var)
        start_entry.pack(side="left")

        # End Date
        end_frame = Theme.create_frame(date_content)
        end_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(end_frame, "End Date:", width=100).pack(side="left")
        self.wizard.end_date_var = ctk.StringVar(value="")
        end_entry = Theme.create_entry(end_frame, placeholder="YYYY-MM-DD", width=150)
        end_entry.configure(textvariable=self.wizard.end_date_var)
        end_entry.pack(side="left")

        # E-Ratio Settings Card
        eratio_card = Theme.create_card(parent)
        eratio_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        eratio_content = Theme.create_frame(eratio_card)
        eratio_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(eratio_content, "E-Ratio Settings", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(
            eratio_content,
            "The E-ratio measures entry edge by comparing maximum favorable vs adverse\n"
            "price excursions normalized by ATR. Set the maximum lookforward period.",
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        max_days_frame = Theme.create_frame(eratio_content)
        max_days_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(max_days_frame, "Max E-Ratio Days:", width=150).pack(side="left")
        self.wizard.max_eratio_days_var = ctk.StringVar(value="20")
        days_entry = Theme.create_entry(max_days_frame, width=80)
        days_entry.configure(textvariable=self.wizard.max_eratio_days_var)
        days_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        Theme.create_hint(max_days_frame, "(E-ratio will be calculated for days 1 to this value)").pack(side="left")

    def get_summary(self) -> Dict[str, str]:
        start = self.wizard.start_date_var.get().strip()
        end = self.wizard.end_date_var.get().strip()

        if start or end:
            date_range = f"{start or 'earliest'} to {end or 'latest'}"
        else:
            date_range = "All available data"

        max_days = self.wizard.max_eratio_days_var.get()

        return {
            "Date Range": date_range,
            "Max E-Ratio Days": max_days
        }

    def validate(self) -> bool:
        self.validation_errors = []

        start_str = self.wizard.start_date_var.get().strip()
        end_str = self.wizard.end_date_var.get().strip()

        if start_str:
            try:
                datetime.strptime(start_str, '%Y-%m-%d')
            except ValueError:
                self.validation_errors.append("Start date must be in YYYY-MM-DD format.")
                return False

        if end_str:
            try:
                datetime.strptime(end_str, '%Y-%m-%d')
            except ValueError:
                self.validation_errors.append("End date must be in YYYY-MM-DD format.")
                return False

        try:
            max_days = int(self.wizard.max_eratio_days_var.get())
            if max_days < 1 or max_days > 100:
                self.validation_errors.append("Max E-Ratio days must be between 1 and 100.")
                return False
        except ValueError:
            self.validation_errors.append("Max E-Ratio days must be a valid integer.")
            return False

        return True


class CTkEdgeAnalysisReviewStep(CTkReviewStep):
    """Step 4: Review and run analysis."""

    def __init__(self, wizard: 'CTkEdgeAnalysisWizard'):
        super().__init__(wizard, "Review & Run")
        self.wizard: CTkEdgeAnalysisWizard = wizard

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        super().create_widgets(parent)

        # Analysis info at the bottom
        info_card = Theme.create_card(parent)
        info_card.pack(fill="x", pady=(Sizes.PAD_M, 0), side="bottom")

        info_content = Theme.create_frame(info_card)
        info_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(info_content, "Analysis Details", size="s").pack(anchor="w", pady=(0, Sizes.PAD_S))

        Theme.create_label(
            info_content,
            "This analysis will:\n"
            "1. Run all entry signals with unlimited capital\n"
            "2. Calculate E-ratio (MFE/MAE normalized by ATR) for each time horizon\n"
            "3. Calculate R-multiple distribution for all completed trades\n"
            "4. Generate interactive visualizations for both metrics",
            text_color=Colors.TEXT_SECONDARY,
            justify="left"
        ).pack(anchor="w")


# =============================================================================
# RESULTS WINDOW WITH VISUALIZATIONS
# =============================================================================

class CTkEdgeResultsWindow(ctk.CTkToplevel):
    """Results window with E-ratio and R-multiple visualizations."""

    def __init__(self, parent, title: str, settings: Dict[str, Dict[str, str]]):
        super().__init__(parent)

        self.title(title)
        self.geometry("1200x800")
        self.configure(fg_color=Colors.BG_DARK)

        self.result: Optional[EdgeAnalysisResult] = None

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

        for step_name, step_settings in settings.items():
            step_frame = Theme.create_frame(config_content)
            step_frame.pack(fill="x", pady=Sizes.PAD_XS)

            Theme.create_label(step_frame, f"{step_name}: ", font=Fonts.LABEL_BOLD).pack(side="left")
            summary_parts = [f"{k}: {v}" for k, v in step_settings.items()]
            Theme.create_label(step_frame, " | ".join(summary_parts), text_color=Colors.TEXT_SECONDARY).pack(side="left")

        # Progress/Results section with tabs
        self.results_card = Theme.create_card(main_frame)
        self.results_card.grid(row=1, column=0, sticky="nsew")

        # Progress panel (shown initially)
        self.progress_frame = Theme.create_frame(self.results_card)
        self.progress_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        self.progress_panel = ProgressPanel(self.progress_frame, show_log=True, log_height=300)
        self.progress_panel.pack(fill="both", expand=True)

        # Results tabs (hidden initially)
        self.tabs_frame = Theme.create_frame(self.results_card)

        # Close button
        btn_frame = Theme.create_frame(main_frame)
        btn_frame.grid(row=2, column=0, sticky="e", pady=(Sizes.PAD_M, 0))

        Theme.create_button(btn_frame, "Close", command=self.destroy, style="secondary", width=100).pack(side="right")

    def log(self, message: str):
        self.progress_panel.log(message)

    def update_progress(self, current: int, total: int, detail: str = ""):
        progress = current / total if total > 0 else 0
        self.progress_panel.update_progress(progress, detail)

    def on_complete(self, result: EdgeAnalysisResult):
        """Show results with visualizations."""
        self.result = result

        self.progress_panel.set_complete(f"Analysis Complete - {len(result.trades)} trades analyzed")

        # Hide progress, show tabs
        self.progress_frame.pack_forget()
        self.tabs_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Create tabview
        self.tabview = Theme.create_tabview(self.tabs_frame)
        self.tabview.pack(fill="both", expand=True)

        # Create tabs
        self.tabview.add("E-Ratio")
        self.tabview.add("R-Multiple")
        self.tabview.add("Summary")

        # Populate E-Ratio tab
        self._create_eratio_tab(self.tabview.tab("E-Ratio"))

        # Populate R-Multiple tab
        self._create_rmultiple_tab(self.tabview.tab("R-Multiple"))

        # Populate Summary tab
        self._create_summary_tab(self.tabview.tab("Summary"))

    def _create_eratio_tab(self, parent):
        """Create E-ratio visualization."""
        if not MATPLOTLIB_AVAILABLE:
            Theme.create_label(
                parent,
                "Matplotlib not available. Install it with: pip install matplotlib",
                text_color=Colors.ERROR
            ).pack(pady=Sizes.PAD_XL)
            return

        if not self.result or not self.result.e_ratios:
            Theme.create_label(
                parent,
                "No E-ratio data available.",
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_XL)
            return

        # Create figure
        fig = Figure(figsize=(10, 6), facecolor=Colors.BG_DARK)
        ax = fig.add_subplot(111)

        # Style the plot
        ax.set_facecolor(Colors.SURFACE)
        ax.tick_params(colors=Colors.TEXT_PRIMARY)
        ax.spines['bottom'].set_color(Colors.BORDER)
        ax.spines['top'].set_color(Colors.BORDER)
        ax.spines['left'].set_color(Colors.BORDER)
        ax.spines['right'].set_color(Colors.BORDER)

        # Plot E-ratio data
        days = sorted(self.result.e_ratios.keys())
        ratios = [self.result.e_ratios[d] for d in days]

        ax.plot(days, ratios, color=Colors.PRIMARY_LIGHT, linewidth=2, marker='o', markersize=4)
        ax.axhline(y=1.0, color=Colors.TEXT_MUTED, linestyle='--', alpha=0.7, label='E-ratio = 1.0 (no edge)')

        ax.set_xlabel('Days After Entry', color=Colors.TEXT_PRIMARY, fontsize=11)
        ax.set_ylabel('E-Ratio', color=Colors.TEXT_PRIMARY, fontsize=11)
        ax.set_title('E-Ratio by Time Horizon\n(Higher = Better Entry Edge)',
                     color=Colors.TEXT_PRIMARY, fontsize=13, pad=15)

        ax.legend(facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                  labelcolor=Colors.TEXT_PRIMARY)
        ax.grid(True, alpha=0.3, color=Colors.BORDER)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        stats_frame = Theme.create_frame(parent)
        stats_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        max_eratio = max(ratios) if ratios else 0
        max_day = days[ratios.index(max_eratio)] if ratios else 0
        avg_eratio = sum(ratios) / len(ratios) if ratios else 0

        Theme.create_label(
            stats_frame,
            f"Max E-Ratio: {max_eratio:.3f} (at day {max_day}) | "
            f"Average E-Ratio: {avg_eratio:.3f} | "
            f"Total Trades: {len(self.result.trades)}",
            text_color=Colors.TEXT_SECONDARY
        ).pack()

    def _create_rmultiple_tab(self, parent):
        """Create R-multiple distribution visualization."""
        if not MATPLOTLIB_AVAILABLE:
            Theme.create_label(
                parent,
                "Matplotlib not available. Install it with: pip install matplotlib",
                text_color=Colors.ERROR
            ).pack(pady=Sizes.PAD_XL)
            return

        if not self.result or not self.result.r_multiples:
            Theme.create_label(
                parent,
                "No R-multiple data available.",
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_XL)
            return

        # Create figure with two subplots (back to back histograms)
        fig = Figure(figsize=(10, 6), facecolor=Colors.BG_DARK)

        # Create subplots
        ax1 = fig.add_subplot(121)  # Losing trades
        ax2 = fig.add_subplot(122)  # Winning trades

        for ax in [ax1, ax2]:
            ax.set_facecolor(Colors.SURFACE)
            ax.tick_params(colors=Colors.TEXT_PRIMARY)
            ax.spines['bottom'].set_color(Colors.BORDER)
            ax.spines['top'].set_color(Colors.BORDER)
            ax.spines['left'].set_color(Colors.BORDER)
            ax.spines['right'].set_color(Colors.BORDER)

        # Plot losing trades histogram (left)
        losing = self.result.losing_r_multiples
        if losing:
            # Create bins from min to 0
            min_r = min(losing)
            bins = np.arange(min_r - 0.5, 0.5, 0.25)
            ax1.hist(losing, bins=bins, color=Colors.CHART_NEGATIVE, edgecolor=Colors.BG_DARK, alpha=0.8)

        ax1.set_xlabel('R-Multiple', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_ylabel('Number of Trades', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_title(f'Losing Trades\n({len(losing)} trades)',
                      color=Colors.ERROR, fontsize=11, pad=10)
        ax1.axvline(x=0, color=Colors.TEXT_MUTED, linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3, color=Colors.BORDER)

        # Plot winning trades histogram (right)
        winning = self.result.winning_r_multiples
        if winning:
            # Create bins from 0 to max
            max_r = max(winning)
            bins = np.arange(-0.25, max_r + 0.5, 0.25)
            ax2.hist(winning, bins=bins, color=Colors.CHART_POSITIVE, edgecolor=Colors.BG_DARK, alpha=0.8)

        ax2.set_xlabel('R-Multiple', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_ylabel('Number of Trades', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_title(f'Winning Trades\n({len(winning)} trades)',
                      color=Colors.SUCCESS, fontsize=11, pad=10)
        ax2.axvline(x=0, color=Colors.TEXT_MUTED, linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3, color=Colors.BORDER)

        fig.suptitle('R-Multiple Distribution', color=Colors.TEXT_PRIMARY, fontsize=13, y=0.98)
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        stats_frame = Theme.create_frame(parent)
        stats_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        all_r = self.result.r_multiples
        avg_r = sum(all_r) / len(all_r) if all_r else 0
        avg_win_r = sum(winning) / len(winning) if winning else 0
        avg_loss_r = sum(losing) / len(losing) if losing else 0
        win_rate = len(winning) / len(all_r) * 100 if all_r else 0

        Theme.create_label(
            stats_frame,
            f"Win Rate: {win_rate:.1f}% | "
            f"Avg R: {avg_r:.2f} | "
            f"Avg Win: {avg_win_r:.2f}R | "
            f"Avg Loss: {avg_loss_r:.2f}R",
            text_color=Colors.TEXT_SECONDARY
        ).pack()

    def _create_summary_tab(self, parent):
        """Create summary statistics tab."""
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        if not self.result:
            return

        # Summary stats
        stats_card = Theme.create_card(scroll)
        stats_card.pack(fill="x", pady=Sizes.PAD_S)

        stats_content = Theme.create_frame(stats_card)
        stats_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(stats_content, "Summary Statistics", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Calculate stats
        total_trades = len(self.result.trades)
        winning = len(self.result.winning_r_multiples)
        losing = len(self.result.losing_r_multiples)
        win_rate = winning / total_trades * 100 if total_trades > 0 else 0

        all_r = self.result.r_multiples
        avg_r = sum(all_r) / len(all_r) if all_r else 0
        expectancy_r = avg_r  # Average R-multiple is the expectancy

        # E-ratio stats
        if self.result.e_ratios:
            days = sorted(self.result.e_ratios.keys())
            ratios = [self.result.e_ratios[d] for d in days]
            max_eratio = max(ratios)
            max_day = days[ratios.index(max_eratio)]
        else:
            max_eratio = 0
            max_day = 0

        stats = [
            ("Total Trades Analyzed", str(total_trades)),
            ("Winning Trades", str(winning)),
            ("Losing Trades", str(losing)),
            ("Win Rate", f"{win_rate:.1f}%"),
            ("Average R-Multiple", f"{avg_r:.2f}R"),
            ("Expectancy per Trade", f"{expectancy_r:.2f}R"),
            ("", ""),
            ("Peak E-Ratio", f"{max_eratio:.3f}"),
            ("Peak E-Ratio Day", str(max_day)),
        ]

        for label, value in stats:
            if not label:
                ctk.CTkFrame(stats_content, fg_color=Colors.BORDER, height=1).pack(fill="x", pady=Sizes.PAD_S)
                continue

            row = Theme.create_frame(stats_content)
            row.pack(fill="x", pady=Sizes.PAD_XS)

            Theme.create_label(row, f"{label}:", width=200).pack(side="left")
            Theme.create_label(row, value, text_color=Colors.PRIMARY_LIGHT).pack(side="left")

        # E-ratio table
        if self.result.e_ratios:
            eratio_card = Theme.create_card(scroll)
            eratio_card.pack(fill="x", pady=Sizes.PAD_S)

            eratio_content = Theme.create_frame(eratio_card)
            eratio_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

            Theme.create_header(eratio_content, "E-Ratio by Day", size="s").pack(anchor="w", pady=(0, Sizes.PAD_M))

            # Create table header
            header_frame = Theme.create_frame(eratio_content)
            header_frame.pack(fill="x")

            for col in ["Day", "E-Ratio", "Interpretation"]:
                Theme.create_label(header_frame, col, font=Fonts.LABEL_BOLD, width=120).pack(side="left")

            ctk.CTkFrame(eratio_content, fg_color=Colors.BORDER, height=1).pack(fill="x", pady=Sizes.PAD_XS)

            # Table rows
            for day in sorted(self.result.e_ratios.keys()):
                e_ratio = self.result.e_ratios[day]

                if e_ratio > 1.5:
                    interp = "Strong Edge"
                    color = Colors.SUCCESS
                elif e_ratio > 1.0:
                    interp = "Positive Edge"
                    color = Colors.PRIMARY_LIGHT
                elif e_ratio > 0.7:
                    interp = "Weak Edge"
                    color = Colors.WARNING
                else:
                    interp = "No Edge"
                    color = Colors.ERROR

                row = Theme.create_frame(eratio_content)
                row.pack(fill="x", pady=Sizes.PAD_XS)

                Theme.create_label(row, str(day), width=120).pack(side="left")
                Theme.create_label(row, f"{e_ratio:.3f}", width=120).pack(side="left")
                Theme.create_label(row, interp, width=120, text_color=color).pack(side="left")

    def on_error(self, error: str):
        self.progress_panel.log(f"ERROR: {error}", level="error")
        self.progress_panel.status_label.configure(text="Error", text_color=Colors.ERROR)


# =============================================================================
# MAIN WIZARD CLASS
# =============================================================================

class CTkEdgeAnalysisWizard(CTkWizardBase):
    """Main Edge Analysis Wizard application."""

    STRATEGIES = {
        'AlphaTrendStrategy': AlphaTrendStrategy
    }

    def __init__(self):
        super().__init__(
            title="Edge Analysis Framework",
            width=1100,
            height=750
        )

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/daily'))
        self.available_securities = self.data_loader.get_available_symbols()

        # Managers
        self.security_registry = SecurityRegistry(Path('config/security_metadata.json'))
        self.currency_converter = CurrencyConverter(base_currency='GBP')
        try:
            self.currency_converter.load_rates_directory(Path('currency_rates/'))
        except Exception as e:
            print(f"Warning: Could not load currency rates: {e}")

        self.preset_manager = StrategyParameterPreset()
        self.basket_manager = BasketManager()

        # Strategy parameters cache
        self.strategy_params: Dict[str, Dict[str, Any]] = {}

        # Variables
        self.strategy_var: Optional[ctk.StringVar] = None
        self.start_date_var: Optional[ctk.StringVar] = None
        self.end_date_var: Optional[ctk.StringVar] = None
        self.max_eratio_days_var: Optional[ctk.StringVar] = None
        self.basket_var: Optional[ctk.StringVar] = None
        self.selected_securities: List[str] = []

        # Add wizard steps
        self.add_step(CTkSecuritiesStep(self))
        self.add_step(CTkStrategyStep(self))
        self.add_step(CTkTimePeriodStep(self))
        self.add_step(CTkEdgeAnalysisReviewStep(self))

        # Set callbacks
        self.on_complete = self._run_analysis
        self.on_cancel = self._on_cancel

        # Start wizard
        self.start()

    def _get_final_button_text(self) -> str:
        return "Run Analysis"

    def _on_cancel(self):
        if ask_yes_no(self.root, "Cancel", "Are you sure you want to cancel?"):
            self.root.destroy()

    def _run_analysis(self):
        """Run the edge analysis with configured settings."""
        try:
            settings = self.get_all_summaries()

            # Parse configuration
            securities = self.selected_securities
            strategy_name = self.strategy_var.get()

            start_date = self._parse_date(self.start_date_var.get())
            end_date = self._parse_date(self.end_date_var.get())
            max_eratio_days = int(self.max_eratio_days_var.get())

            # Create strategy instance
            strategy_class = self.STRATEGIES[strategy_name]
            strategy_params = self.strategy_params.get(strategy_name, {})
            strategy = strategy_class(**strategy_params)

            # Create results window
            results_window = CTkEdgeResultsWindow(
                self.root,
                f"Edge Analysis Results: {strategy_name}",
                settings
            )

            # Create message queue for thread-safe UI updates
            msg_queue = queue.Queue()

            def update_ui():
                try:
                    while True:
                        msg_type, data = msg_queue.get_nowait()
                        if msg_type == "log":
                            results_window.log(data)
                        elif msg_type == "progress":
                            current, total, detail = data
                            results_window.update_progress(current, total, detail)
                        elif msg_type == "complete":
                            results_window.on_complete(data)
                            return
                        elif msg_type == "error":
                            results_window.on_error(data)
                            return
                except queue.Empty:
                    pass
                self.root.after(100, update_ui)

            def run_in_thread():
                try:
                    msg_queue.put(("log", f"Starting edge analysis..."))
                    msg_queue.put(("log", f"Strategy: {strategy_name}"))
                    msg_queue.put(("log", f"Securities: {len(securities)} symbols"))
                    msg_queue.put(("log", f"Max E-Ratio Days: {max_eratio_days}"))
                    msg_queue.put(("log", "=" * 50))

                    engine = UnlimitedCapitalEngine(
                        self.data_loader,
                        self.security_registry,
                        self.currency_converter
                    )

                    def progress_cb(current, total):
                        msg_queue.put(("progress", (current, total, f"Processing {securities[min(current, len(securities)-1)]}")))

                    result = engine.run_backtest(
                        symbols=securities,
                        strategy=strategy,
                        start_date=start_date,
                        end_date=end_date,
                        max_e_ratio_days=max_eratio_days,
                        progress_callback=progress_cb
                    )

                    msg_queue.put(("log", ""))
                    msg_queue.put(("log", f"Analysis complete! Found {len(result.trades)} trades."))
                    msg_queue.put(("complete", result))

                except Exception as e:
                    import traceback
                    msg_queue.put(("error", f"{str(e)}\n{traceback.format_exc()}"))

            # Start background thread
            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()

            # Start UI update polling
            self.root.after(100, update_ui)

        except Exception as e:
            show_error(self.root, "Error", f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str.strip():
            return None
        try:
            return datetime.strptime(date_str.strip(), '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    """Main entry point for GUI."""
    app = CTkEdgeAnalysisWizard()
    app.run()


if __name__ == "__main__":
    main()
