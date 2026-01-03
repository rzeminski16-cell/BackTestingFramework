"""
Edge Analysis GUI (CustomTkinter)

A specialized GUI for analyzing entry edge and R-multiples from existing trade logs.

Key Features:
- Load existing trade log CSV files (no backtesting required)
- Select individual trades to examine their E-ratio over time
- Two visualization tabs:
  1. E-ratio: Entry edge for selected trade over different time horizons
  2. R-multiple: Distribution of all trade outcomes

E-ratio Formula (for a single trade):
    E-ratio(n) = (MFE_n / ATR) / (MAE_n / ATR)
    Where:
    - MFE_n = Maximum Favorable Excursion over n days from entry
    - MAE_n = Maximum Adverse Excursion over n days from entry
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

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info
from Classes.Data.data_loader import DataLoader


# =============================================================================
# TRADE LOG LOADER
# =============================================================================

class TradeLogEntry:
    """Represents a single trade from a trade log."""

    def __init__(self, row: pd.Series):
        self.trade_id = row.get('trade_id', '')
        self.symbol = row.get('symbol', '')
        self.entry_date = self._parse_date(row.get('entry_date'))
        self.entry_price = float(row.get('entry_price', 0))
        self.exit_date = self._parse_date(row.get('exit_date'))
        self.exit_price = float(row.get('exit_price', 0))
        self.quantity = float(row.get('quantity', 0))
        self.initial_stop_loss = self._parse_float(row.get('initial_stop_loss'))
        self.final_stop_loss = self._parse_float(row.get('final_stop_loss'))
        self.pl = float(row.get('pl', 0))
        self.pl_pct = float(row.get('pl_pct', 0))
        self.entry_reason = row.get('entry_reason', '')
        self.exit_reason = row.get('exit_reason', '')

        # Calculate R-multiple if we have stop loss
        self.r_multiple = self._calculate_r_multiple()

    def _parse_date(self, val) -> Optional[datetime]:
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, datetime):
            return val
        try:
            return pd.to_datetime(val)
        except:
            return None

    def _parse_float(self, val) -> Optional[float]:
        if pd.isna(val) or val is None:
            return None
        try:
            return float(val)
        except:
            return None

    def _calculate_r_multiple(self) -> Optional[float]:
        """Calculate R-multiple based on entry, exit, and stop loss."""
        if self.initial_stop_loss is None or self.initial_stop_loss <= 0:
            return None

        # Initial risk (R) = entry price - stop loss
        initial_risk = self.entry_price - self.initial_stop_loss
        if initial_risk <= 0:
            return None

        # P/L per unit
        pl_per_unit = self.exit_price - self.entry_price

        # R-multiple = P/L per unit / R
        return pl_per_unit / initial_risk

    @property
    def display_name(self) -> str:
        """Display name for trade selection list."""
        date_str = self.entry_date.strftime('%Y-%m-%d') if self.entry_date else 'N/A'
        r_str = f"{self.r_multiple:.2f}R" if self.r_multiple is not None else "N/A"
        return f"{self.trade_id} | {self.symbol} | {date_str} | {r_str}"

    def __str__(self):
        return self.display_name


class TradeLogLoader:
    """Loads and parses trade log CSV files."""

    def __init__(self):
        self.trades: List[TradeLogEntry] = []
        self.source_file: Optional[Path] = None

    def load(self, filepath: Path) -> List[TradeLogEntry]:
        """Load trades from a CSV file."""
        df = pd.read_csv(filepath)

        self.trades = []
        for _, row in df.iterrows():
            try:
                trade = TradeLogEntry(row)
                if trade.symbol and trade.entry_date:
                    self.trades.append(trade)
            except Exception as e:
                print(f"Error parsing trade row: {e}")
                continue

        self.source_file = filepath
        return self.trades

    def get_symbols(self) -> List[str]:
        """Get unique symbols in loaded trades."""
        return list(set(t.symbol for t in self.trades))


# =============================================================================
# E-RATIO CALCULATOR
# =============================================================================

class ERatioCalculator:
    """Calculates E-ratio for individual trades using historical price data."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price data for a symbol, using cache if available."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        try:
            data = self.data_loader.load_csv(symbol)
            self._price_cache[symbol] = data
            return data
        except Exception as e:
            print(f"Error loading price data for {symbol}: {e}")
            return None

    def calculate_for_trade(self, trade: TradeLogEntry, max_days: int = 30) -> Dict[str, Any]:
        """
        Calculate E-ratio curve for a single trade.

        Args:
            trade: Trade log entry
            max_days: Maximum days to calculate forward

        Returns:
            Dict containing:
            - e_ratios: Dict[int, float] mapping day -> e_ratio
            - mfe_atr: Dict[int, float] mapping day -> MFE/ATR
            - mae_atr: Dict[int, float] mapping day -> MAE/ATR
            - atr_at_entry: ATR value at entry
            - error: Optional error message
        """
        result = {
            'e_ratios': {},
            'mfe_atr': {},
            'mae_atr': {},
            'atr_at_entry': None,
            'error': None
        }

        # Get price data
        price_data = self._get_price_data(trade.symbol)
        if price_data is None:
            result['error'] = f"Could not load price data for {trade.symbol}"
            return result

        # Find entry date in data
        entry_date = trade.entry_date
        if entry_date is None:
            result['error'] = "Trade has no entry date"
            return result

        # Convert entry_date to same type as data
        if 'date' not in price_data.columns:
            result['error'] = "Price data missing 'date' column"
            return result

        # Find the entry bar index
        price_data = price_data.copy()
        price_data['date'] = pd.to_datetime(price_data['date'])

        # Find closest date on or after entry
        mask = price_data['date'] >= entry_date
        if not mask.any():
            result['error'] = f"No price data after entry date {entry_date}"
            return result

        entry_idx = price_data[mask].index[0]
        entry_bar = price_data.loc[entry_idx]
        entry_price = trade.entry_price

        # Get ATR at entry
        atr_col = None
        for col in ['atr_14_atr', 'atr_14', 'atr']:
            if col in price_data.columns:
                atr_col = col
                break

        if atr_col is None:
            result['error'] = "No ATR column found in price data"
            return result

        atr_at_entry = entry_bar[atr_col]
        if pd.isna(atr_at_entry) or atr_at_entry <= 0:
            result['error'] = f"Invalid ATR at entry: {atr_at_entry}"
            return result

        result['atr_at_entry'] = float(atr_at_entry)

        # Get position in the dataframe for slicing
        entry_pos = price_data.index.get_loc(entry_idx)

        # Calculate MFE and MAE for each day from 1 to max_days
        for n_days in range(1, max_days + 1):
            end_pos = entry_pos + n_days
            if end_pos >= len(price_data):
                break

            # Get forward data (bars after entry, up to n_days)
            forward_data = price_data.iloc[entry_pos + 1:end_pos + 1]

            if len(forward_data) == 0:
                continue

            # MFE: Maximum gain from entry (highest high - entry price)
            max_high = forward_data['high'].max()
            mfe = max(0, max_high - entry_price)

            # MAE: Maximum loss from entry (entry price - lowest low)
            min_low = forward_data['low'].min()
            mae = max(0, entry_price - min_low)

            # Normalize by ATR
            mfe_atr = mfe / atr_at_entry
            mae_atr = mae / atr_at_entry

            result['mfe_atr'][n_days] = mfe_atr
            result['mae_atr'][n_days] = mae_atr

            # E-ratio = MFE/ATR divided by MAE/ATR
            # If MAE is 0, the E-ratio would be infinite (perfect entry)
            if mae_atr > 0:
                e_ratio = mfe_atr / mae_atr
            else:
                e_ratio = float('inf') if mfe_atr > 0 else 1.0

            result['e_ratios'][n_days] = e_ratio

        return result


# =============================================================================
# MAIN GUI
# =============================================================================

class CTkEdgeAnalysisGUI(ctk.CTk):
    """Main Edge Analysis GUI application."""

    def __init__(self):
        super().__init__()

        # Initialize theme
        Theme.setup()

        self.title("Edge Analysis - Trade Log Analyzer")
        self.geometry("1400x900")
        self.configure(fg_color=Colors.BG_DARK)

        # Data
        self.trade_loader = TradeLogLoader()
        self.data_loader = DataLoader(Path('raw_data/daily'))
        self.eratio_calculator = ERatioCalculator(self.data_loader)

        # State
        self.current_trades: List[TradeLogEntry] = []
        self.selected_trade: Optional[TradeLogEntry] = None
        self.max_eratio_days = 30

        self._create_layout()

    def _create_layout(self):
        """Create the main layout."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top bar - Load and Settings
        self._create_top_bar()

        # Main content area
        self._create_main_content()

    def _create_top_bar(self):
        """Create top bar with load button and settings."""
        top_frame = Theme.create_card(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        content = Theme.create_frame(top_frame)
        content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        # Title
        Theme.create_header(content, "Edge Analysis", size="l").pack(side="left")

        # Load button
        Theme.create_button(
            content, "Load Trade Log...",
            command=self._load_trade_log,
            style="primary",
            width=150
        ).pack(side="left", padx=(Sizes.PAD_XL, Sizes.PAD_M))

        # File path display
        self.file_label = Theme.create_label(
            content, "No file loaded",
            text_color=Colors.TEXT_MUTED
        )
        self.file_label.pack(side="left", padx=Sizes.PAD_M)

        # Max E-ratio days setting
        settings_frame = Theme.create_frame(content)
        settings_frame.pack(side="right")

        Theme.create_label(settings_frame, "Max E-Ratio Days:").pack(side="left", padx=(0, Sizes.PAD_S))

        self.max_days_var = ctk.StringVar(value="30")
        days_entry = Theme.create_entry(settings_frame, width=60)
        days_entry.configure(textvariable=self.max_days_var)
        days_entry.pack(side="left")

    def _create_main_content(self):
        """Create main content area with trade list and visualization tabs."""
        main_frame = Theme.create_frame(self)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Left panel - Trade list
        self._create_trade_list_panel(main_frame)

        # Right panel - Visualization tabs
        self._create_visualization_panel(main_frame)

    def _create_trade_list_panel(self, parent):
        """Create trade list panel on the left."""
        list_card = Theme.create_card(parent)
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_M))
        list_card.configure(width=400)
        list_card.grid_propagate(False)

        list_content = Theme.create_frame(list_card)
        list_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Header
        header_frame = Theme.create_frame(list_content)
        header_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_header(header_frame, "Trades", size="s").pack(side="left")

        self.trade_count_label = Theme.create_label(
            header_frame, "(0 trades)",
            text_color=Colors.TEXT_MUTED,
            font=Fonts.BODY_S
        )
        self.trade_count_label.pack(side="right")

        # Search/filter
        search_frame = Theme.create_frame(list_content)
        search_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self._on_search_change)

        search_entry = Theme.create_entry(search_frame, placeholder="Filter trades...")
        search_entry.configure(textvariable=self.search_var)
        search_entry.pack(fill="x")

        # Trade listbox
        self.trade_listbox = ctk.CTkScrollableFrame(
            list_content,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_M
        )
        self.trade_listbox.pack(fill="both", expand=True)

        # Instructions
        self.instructions_label = Theme.create_label(
            list_content,
            "Load a trade log CSV file to begin.\nSelect a trade to view its E-ratio.",
            text_color=Colors.TEXT_MUTED,
            justify="left"
        )
        self.instructions_label.pack(pady=Sizes.PAD_M)

    def _create_visualization_panel(self, parent):
        """Create visualization tabs panel on the right."""
        viz_card = Theme.create_card(parent)
        viz_card.grid(row=0, column=1, sticky="nsew")

        # Create tabview
        self.tabview = Theme.create_tabview(viz_card)
        self.tabview.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Create tabs
        self.tabview.add("E-Ratio")
        self.tabview.add("R-Multiple")
        self.tabview.add("Trade Details")

        # Initialize tab contents
        self._create_eratio_tab(self.tabview.tab("E-Ratio"))
        self._create_rmultiple_tab(self.tabview.tab("R-Multiple"))
        self._create_details_tab(self.tabview.tab("Trade Details"))

    def _create_eratio_tab(self, parent):
        """Create E-ratio tab content."""
        self.eratio_frame = Theme.create_frame(parent)
        self.eratio_frame.pack(fill="both", expand=True)

        # Placeholder message
        self.eratio_placeholder = Theme.create_label(
            self.eratio_frame,
            "Select a trade from the list to view its E-ratio curve.\n\n"
            "E-ratio measures entry edge by comparing:\n"
            "• MFE (Maximum Favorable Excursion) - how far price moved in your favor\n"
            "• MAE (Maximum Adverse Excursion) - how far price moved against you\n\n"
            "Both are normalized by ATR for volatility adjustment.\n"
            "E-ratio > 1.0 indicates price moved more favorably than adversely.",
            text_color=Colors.TEXT_MUTED,
            wraplength=500,
            justify="left"
        )
        self.eratio_placeholder.pack(expand=True)

    def _create_rmultiple_tab(self, parent):
        """Create R-multiple tab content."""
        self.rmultiple_frame = Theme.create_frame(parent)
        self.rmultiple_frame.pack(fill="both", expand=True)

        # Placeholder message
        self.rmultiple_placeholder = Theme.create_label(
            self.rmultiple_frame,
            "Load a trade log to view R-multiple distribution.\n\n"
            "R-multiple expresses profit/loss as a multiple of initial risk:\n"
            "• R = Entry Price - Stop Loss (your initial risk)\n"
            "• R-multiple = Profit/Loss ÷ R\n\n"
            "Example: +2R means you made twice your initial risk\n"
            "Example: -0.5R means you lost half your initial risk",
            text_color=Colors.TEXT_MUTED,
            wraplength=500,
            justify="left"
        )
        self.rmultiple_placeholder.pack(expand=True)

    def _create_details_tab(self, parent):
        """Create trade details tab content."""
        self.details_frame = Theme.create_frame(parent)
        self.details_frame.pack(fill="both", expand=True)

        # Scrollable frame for details
        self.details_scroll = ctk.CTkScrollableFrame(
            self.details_frame,
            fg_color="transparent"
        )
        self.details_scroll.pack(fill="both", expand=True)

        # Placeholder
        self.details_placeholder = Theme.create_label(
            self.details_scroll,
            "Select a trade to view its details.",
            text_color=Colors.TEXT_MUTED
        )
        self.details_placeholder.pack(expand=True, pady=Sizes.PAD_XL)

    def _load_trade_log(self):
        """Open file dialog to load a trade log."""
        from tkinter import filedialog

        filepath = filedialog.askopenfilename(
            title="Select Trade Log CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=Path("logs")
        )

        if not filepath:
            return

        try:
            trades = self.trade_loader.load(Path(filepath))
            self.current_trades = trades

            # Update UI
            self.file_label.configure(text=Path(filepath).name)
            self.trade_count_label.configure(text=f"({len(trades)} trades)")
            self.instructions_label.pack_forget()

            # Populate trade list
            self._populate_trade_list()

            # Update R-multiple chart
            self._update_rmultiple_chart()

        except Exception as e:
            show_error(self, "Error Loading File", str(e))

    def _populate_trade_list(self, filter_text: str = ""):
        """Populate the trade list with loaded trades."""
        # Clear existing
        for widget in self.trade_listbox.winfo_children():
            widget.destroy()

        filter_text = filter_text.lower()

        for trade in self.current_trades:
            # Apply filter
            if filter_text:
                if filter_text not in trade.display_name.lower():
                    continue

            # Create trade row
            row = ctk.CTkFrame(
                self.trade_listbox,
                fg_color="transparent",
                corner_radius=Sizes.RADIUS_S
            )
            row.pack(fill="x", pady=1)

            # Make clickable
            row.bind("<Button-1>", lambda e, t=trade: self._on_trade_selected(t))
            row.bind("<Enter>", lambda e, r=row: r.configure(fg_color=Colors.SURFACE_HOVER))
            row.bind("<Leave>", lambda e, r=row: r.configure(fg_color="transparent"))

            # Trade info
            info_frame = Theme.create_frame(row)
            info_frame.pack(fill="x", padx=Sizes.PAD_S, pady=Sizes.PAD_XS)
            info_frame.bind("<Button-1>", lambda e, t=trade: self._on_trade_selected(t))

            # Symbol and date
            date_str = trade.entry_date.strftime('%Y-%m-%d') if trade.entry_date else 'N/A'
            header_text = f"{trade.symbol} - {date_str}"

            header_label = Theme.create_label(
                info_frame, header_text,
                font=Fonts.LABEL_BOLD
            )
            header_label.pack(anchor="w")
            header_label.bind("<Button-1>", lambda e, t=trade: self._on_trade_selected(t))

            # R-multiple and P/L
            r_str = f"{trade.r_multiple:.2f}R" if trade.r_multiple is not None else "N/A"
            pl_color = Colors.SUCCESS if trade.pl > 0 else Colors.ERROR if trade.pl < 0 else Colors.TEXT_MUTED

            detail_frame = Theme.create_frame(info_frame)
            detail_frame.pack(anchor="w")
            detail_frame.bind("<Button-1>", lambda e, t=trade: self._on_trade_selected(t))

            Theme.create_label(
                detail_frame, f"ID: {trade.trade_id}",
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            ).pack(side="left", padx=(0, Sizes.PAD_M))

            Theme.create_label(
                detail_frame, r_str,
                font=Fonts.BODY_S,
                text_color=pl_color
            ).pack(side="left", padx=(0, Sizes.PAD_M))

            Theme.create_label(
                detail_frame, f"${trade.pl:,.2f}",
                font=Fonts.BODY_S,
                text_color=pl_color
            ).pack(side="left")

    def _on_search_change(self, *args):
        """Handle search text change."""
        self._populate_trade_list(self.search_var.get())

    def _on_trade_selected(self, trade: TradeLogEntry):
        """Handle trade selection."""
        self.selected_trade = trade

        # Update E-ratio chart
        self._update_eratio_chart(trade)

        # Update details tab
        self._update_details_tab(trade)

        # Switch to E-ratio tab
        self.tabview.set("E-Ratio")

    def _update_eratio_chart(self, trade: TradeLogEntry):
        """Update E-ratio chart for selected trade."""
        # Clear existing content
        for widget in self.eratio_frame.winfo_children():
            widget.destroy()

        if not MATPLOTLIB_AVAILABLE:
            Theme.create_label(
                self.eratio_frame,
                "Matplotlib not available. Install it with: pip install matplotlib",
                text_color=Colors.ERROR
            ).pack(expand=True)
            return

        # Get max days setting
        try:
            max_days = int(self.max_days_var.get())
        except ValueError:
            max_days = 30

        # Calculate E-ratio
        result = self.eratio_calculator.calculate_for_trade(trade, max_days)

        if result['error']:
            Theme.create_label(
                self.eratio_frame,
                f"Error calculating E-ratio:\n{result['error']}",
                text_color=Colors.ERROR,
                wraplength=400
            ).pack(expand=True)
            return

        if not result['e_ratios']:
            Theme.create_label(
                self.eratio_frame,
                "No E-ratio data available for this trade.",
                text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            return

        # Create figure
        fig = Figure(figsize=(10, 6), facecolor=Colors.BG_DARK)

        # Create two subplots - E-ratio on top, MFE/MAE on bottom
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for ax in [ax1, ax2]:
            ax.set_facecolor(Colors.SURFACE)
            ax.tick_params(colors=Colors.TEXT_PRIMARY)
            for spine in ax.spines.values():
                spine.set_color(Colors.BORDER)

        # Plot E-ratio
        days = sorted(result['e_ratios'].keys())
        ratios = [result['e_ratios'][d] for d in days]

        # Cap infinite values for display
        display_ratios = [min(r, 10) for r in ratios]

        ax1.plot(days, display_ratios, color=Colors.PRIMARY_LIGHT, linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=1.0, color=Colors.TEXT_MUTED, linestyle='--', alpha=0.7, label='E-ratio = 1.0 (no edge)')
        ax1.fill_between(days, 1.0, display_ratios, where=[r > 1 for r in display_ratios],
                         color=Colors.SUCCESS, alpha=0.3, label='Favorable')
        ax1.fill_between(days, 1.0, display_ratios, where=[r < 1 for r in display_ratios],
                         color=Colors.ERROR, alpha=0.3, label='Unfavorable')

        ax1.set_ylabel('E-Ratio', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_title(f'E-Ratio for {trade.symbol} (Entry: {trade.entry_date.strftime("%Y-%m-%d")})',
                      color=Colors.TEXT_PRIMARY, fontsize=12, pad=10)
        ax1.legend(facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                   labelcolor=Colors.TEXT_PRIMARY, fontsize=8)
        ax1.grid(True, alpha=0.3, color=Colors.BORDER)
        ax1.set_xlim(0, max(days) + 1)

        # Plot MFE and MAE
        mfe_values = [result['mfe_atr'].get(d, 0) for d in days]
        mae_values = [result['mae_atr'].get(d, 0) for d in days]

        ax2.plot(days, mfe_values, color=Colors.SUCCESS, linewidth=2, marker='o', markersize=4, label='MFE/ATR')
        ax2.plot(days, mae_values, color=Colors.ERROR, linewidth=2, marker='o', markersize=4, label='MAE/ATR')

        ax2.set_xlabel('Days After Entry', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_ylabel('ATR Multiples', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_title('Maximum Favorable vs Adverse Excursion (ATR-normalized)',
                      color=Colors.TEXT_PRIMARY, fontsize=11, pad=10)
        ax2.legend(facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                   labelcolor=Colors.TEXT_PRIMARY, fontsize=8)
        ax2.grid(True, alpha=0.3, color=Colors.BORDER)
        ax2.set_xlim(0, max(days) + 1)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.eratio_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        stats_frame = Theme.create_frame(self.eratio_frame)
        stats_frame.pack(fill="x", pady=(Sizes.PAD_S, 0))

        # Find peak E-ratio
        max_eratio = max(ratios)
        max_day = days[ratios.index(max_eratio)]
        max_eratio_display = f"{max_eratio:.2f}" if max_eratio < 100 else "inf"

        Theme.create_label(
            stats_frame,
            f"ATR at Entry: {result['atr_at_entry']:.4f} | "
            f"Peak E-Ratio: {max_eratio_display} (day {max_day}) | "
            f"Entry Price: ${trade.entry_price:.2f} | "
            f"Stop Loss: ${trade.initial_stop_loss:.2f}" if trade.initial_stop_loss else "",
            text_color=Colors.TEXT_SECONDARY
        ).pack()

    def _update_rmultiple_chart(self):
        """Update R-multiple distribution chart."""
        # Clear existing content
        for widget in self.rmultiple_frame.winfo_children():
            widget.destroy()

        if not MATPLOTLIB_AVAILABLE:
            Theme.create_label(
                self.rmultiple_frame,
                "Matplotlib not available. Install it with: pip install matplotlib",
                text_color=Colors.ERROR
            ).pack(expand=True)
            return

        # Get R-multiples from all trades
        r_multiples = [t.r_multiple for t in self.current_trades if t.r_multiple is not None]

        if not r_multiples:
            Theme.create_label(
                self.rmultiple_frame,
                "No R-multiple data available.\nTrades may be missing stop loss information.",
                text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            return

        winning = [r for r in r_multiples if r >= 0]
        losing = [r for r in r_multiples if r < 0]

        # Create figure with two subplots (back to back histograms)
        fig = Figure(figsize=(10, 6), facecolor=Colors.BG_DARK)

        ax1 = fig.add_subplot(121)  # Losing trades
        ax2 = fig.add_subplot(122)  # Winning trades

        for ax in [ax1, ax2]:
            ax.set_facecolor(Colors.SURFACE)
            ax.tick_params(colors=Colors.TEXT_PRIMARY)
            for spine in ax.spines.values():
                spine.set_color(Colors.BORDER)

        # Plot losing trades histogram (left)
        if losing:
            min_r = min(losing)
            bins = np.arange(min_r - 0.25, 0.25, 0.25)
            if len(bins) < 2:
                bins = np.linspace(min_r, 0, 10)
            ax1.hist(losing, bins=bins, color=Colors.CHART_NEGATIVE, edgecolor=Colors.BG_DARK, alpha=0.8)

        ax1.set_xlabel('R-Multiple', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_ylabel('Number of Trades', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_title(f'Losing Trades\n({len(losing)} trades)',
                      color=Colors.ERROR, fontsize=11, pad=10)
        ax1.axvline(x=0, color=Colors.TEXT_MUTED, linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3, color=Colors.BORDER)

        # Plot winning trades histogram (right)
        if winning:
            max_r = max(winning)
            bins = np.arange(-0.25, max_r + 0.5, 0.25)
            if len(bins) < 2:
                bins = np.linspace(0, max_r, 10)
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
        canvas = FigureCanvasTkAgg(fig, self.rmultiple_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        stats_frame = Theme.create_frame(self.rmultiple_frame)
        stats_frame.pack(fill="x", pady=(Sizes.PAD_S, 0))

        avg_r = sum(r_multiples) / len(r_multiples)
        avg_win_r = sum(winning) / len(winning) if winning else 0
        avg_loss_r = sum(losing) / len(losing) if losing else 0
        win_rate = len(winning) / len(r_multiples) * 100

        Theme.create_label(
            stats_frame,
            f"Win Rate: {win_rate:.1f}% | "
            f"Avg R: {avg_r:.2f} | "
            f"Avg Win: {avg_win_r:.2f}R | "
            f"Avg Loss: {avg_loss_r:.2f}R | "
            f"Total Trades: {len(r_multiples)}",
            text_color=Colors.TEXT_SECONDARY
        ).pack()

    def _update_details_tab(self, trade: TradeLogEntry):
        """Update trade details tab."""
        # Clear existing content
        for widget in self.details_scroll.winfo_children():
            widget.destroy()

        # Trade header
        Theme.create_header(
            self.details_scroll,
            f"Trade {trade.trade_id}",
            size="m"
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Create details grid
        details = [
            ("Symbol", trade.symbol),
            ("Entry Date", trade.entry_date.strftime('%Y-%m-%d %H:%M') if trade.entry_date else "N/A"),
            ("Entry Price", f"${trade.entry_price:.4f}"),
            ("Exit Date", trade.exit_date.strftime('%Y-%m-%d %H:%M') if trade.exit_date else "N/A"),
            ("Exit Price", f"${trade.exit_price:.4f}"),
            ("Quantity", f"{trade.quantity:.4f}"),
            ("", ""),
            ("Initial Stop Loss", f"${trade.initial_stop_loss:.4f}" if trade.initial_stop_loss else "N/A"),
            ("Final Stop Loss", f"${trade.final_stop_loss:.4f}" if trade.final_stop_loss else "N/A"),
            ("", ""),
            ("P/L", f"${trade.pl:,.2f}"),
            ("P/L %", f"{trade.pl_pct:.2f}%"),
            ("R-Multiple", f"{trade.r_multiple:.2f}R" if trade.r_multiple is not None else "N/A"),
            ("", ""),
            ("Entry Reason", trade.entry_reason or "N/A"),
            ("Exit Reason", trade.exit_reason or "N/A"),
        ]

        for label, value in details:
            if not label:
                ctk.CTkFrame(self.details_scroll, fg_color=Colors.BORDER, height=1).pack(
                    fill="x", pady=Sizes.PAD_S
                )
                continue

            row = Theme.create_frame(self.details_scroll)
            row.pack(fill="x", pady=Sizes.PAD_XS)

            Theme.create_label(row, f"{label}:", width=150, font=Fonts.LABEL_BOLD).pack(side="left")

            # Color P/L values
            text_color = Colors.TEXT_PRIMARY
            if "P/L" in label or "R-Multiple" in label:
                if trade.pl > 0:
                    text_color = Colors.SUCCESS
                elif trade.pl < 0:
                    text_color = Colors.ERROR

            Theme.create_label(row, str(value), text_color=text_color).pack(side="left")


def main():
    """Main entry point for GUI."""
    app = CTkEdgeAnalysisGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
