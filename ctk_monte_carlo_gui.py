"""
Monte Carlo Simulation GUI for the BackTesting Framework.

Wizard-style flow:
    Step 1 - Load Data:        select one or more trade-log CSVs
    Step 2 - Data Review:      summary stats and return-source picker
    Step 3 - Simulation:       capital, sims, trades, risk, sizing, sampling
    Step 4 - Advanced:         drawdown logic, costs/slippage, seed
    Step 5 - Run:              progress + results window with tabs

Pattern matches the rest of the framework: CustomTkinter, Theme/Colors, the
shared CTkWizardBase, and a separate Toplevel results window with embedded
matplotlib charts.
"""
from __future__ import annotations

import json
import sys
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).parent))

from Classes.GUI.ctk_theme import (
    Theme, Colors, Fonts, Sizes,
    show_error, show_info, show_warning, show_success, ask_yes_no,
)
from Classes.GUI.ctk_wizard_base import CTkWizardBase, CTkWizardStep
from Classes.MonteCarloAnalysis import (
    SimulationConfig, SamplingMethod, SizingMethod,
    LoadedTradeLog, TradeLogReturnSource, load_trade_logs,
    SimulationResult, MonteCarloSimulator,
    SimulationMetrics, SimulationAnalyzer,
)


# ============================================================================
# Helpers
# ============================================================================

def _format_runtime(seconds: float) -> str:
    """Human readable runtime estimate."""
    if seconds < 1.0:
        return f"~{int(seconds * 1000)} ms"
    if seconds < 60.0:
        return f"~{seconds:.1f} s"
    minutes = seconds / 60.0
    return f"~{minutes:.1f} min"


def _apply_dark_matplotlib_style(fig):
    """Style a matplotlib figure to match the dark theme."""
    fig.patch.set_facecolor(Colors.CHART_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(Colors.CHART_BG)
        ax.tick_params(colors=Colors.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(Colors.BORDER)
        ax.xaxis.label.set_color(Colors.TEXT_PRIMARY)
        ax.yaxis.label.set_color(Colors.TEXT_PRIMARY)
        if ax.get_title():
            ax.title.set_color(Colors.TEXT_PRIMARY)
        ax.grid(True, color=Colors.CHART_GRID, alpha=0.4, linestyle="--")
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_color(Colors.TEXT_PRIMARY)
            leg.get_frame().set_facecolor(Colors.SURFACE)
            leg.get_frame().set_edgecolor(Colors.BORDER)


# ============================================================================
# Wizard steps
# ============================================================================

class LoadDataStep(CTkWizardStep):
    """Step 1: select one or more trade-log CSV files."""

    def __init__(self, wizard: "MonteCarloWizard"):
        super().__init__(wizard, "Load Trade Logs")
        self.mc_wizard = wizard
        self.selected_paths: List[Path] = []
        self.file_listbox: Optional[ctk.CTkScrollableFrame] = None
        self.file_count_label: Optional[ctk.CTkLabel] = None

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        # Description
        Theme.create_label(
            parent,
            "Select one or more trade-log CSV files. Multiple files will be "
            "concatenated into a single pool of trades to sample from.",
            wraplength=700,
            justify="left",
            text_color=Colors.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Buttons row
        btn_row = Theme.create_frame(parent)
        btn_row.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_button(
            btn_row, "Browse Files...", command=self._browse,
            style="primary", width=160,
        ).pack(side="left")

        Theme.create_button(
            btn_row, "Clear", command=self._clear,
            style="secondary", width=100,
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        self.file_count_label = Theme.create_label(
            btn_row, "No files selected", text_color=Colors.TEXT_MUTED,
        )
        self.file_count_label.pack(side="left", padx=(Sizes.PAD_L, 0))

        # File list
        Theme.create_header(parent, "Selected files", size="s").pack(
            anchor="w", pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        self.file_listbox = ctk.CTkScrollableFrame(
            parent, fg_color=Colors.SURFACE, height=180,
            corner_radius=Sizes.RADIUS_M, border_width=1, border_color=Colors.BORDER,
        )
        self.file_listbox.pack(fill="both", expand=True, pady=Sizes.PAD_S)

        # Hint about default folder
        Theme.create_hint(
            parent,
            "Tip: trade logs are typically under logs/single_security/<backtest>/ "
            "or logs/portfolio/<backtest>/. The framework's canonical schema is "
            "auto-detected.",
        ).pack(anchor="w", pady=(Sizes.PAD_S, 0))

    def _browse(self):
        from tkinter import filedialog
        initial = "logs" if Path("logs").exists() else "."
        paths = filedialog.askopenfilenames(
            title="Select Trade Log CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=initial,
        )
        if not paths:
            return
        new_paths = [Path(p) for p in paths]
        # Add only new entries
        seen = set(self.selected_paths)
        for p in new_paths:
            if p not in seen:
                self.selected_paths.append(p)
        self._refresh_list()

    def _clear(self):
        self.selected_paths.clear()
        self._refresh_list()

    def _refresh_list(self):
        for child in self.file_listbox.winfo_children():
            child.destroy()
        for i, p in enumerate(self.selected_paths):
            row = Theme.create_frame(self.file_listbox)
            row.pack(fill="x", padx=Sizes.PAD_XS, pady=1)
            Theme.create_label(
                row, f"{i + 1}.", width=30,
                text_color=Colors.TEXT_MUTED, font=Fonts.MONO_S,
            ).pack(side="left")
            Theme.create_label(
                row, str(p), font=Fonts.MONO_S, anchor="w", justify="left",
            ).pack(side="left", fill="x", expand=True)
            Theme.create_button(
                row, "X", style="ghost", width=24, height=24,
                command=lambda idx=i: self._remove(idx),
            ).pack(side="right")

        n = len(self.selected_paths)
        self.file_count_label.configure(
            text=f"{n} file{'s' if n != 1 else ''} selected" if n else "No files selected",
        )

    def _remove(self, idx: int):
        if 0 <= idx < len(self.selected_paths):
            del self.selected_paths[idx]
            self._refresh_list()

    def get_summary(self) -> Dict[str, str]:
        if not self.selected_paths:
            return {"Files": "(none)"}
        return {
            "Files": f"{len(self.selected_paths)} selected",
            "First": self.selected_paths[0].name,
        }

    def validate(self) -> bool:
        self.validation_errors = []
        if not self.selected_paths:
            self.validation_errors.append("Select at least one CSV file.")
            return False

        # Try to load now so we can validate the data
        try:
            self.mc_wizard.loaded_log = load_trade_logs(self.selected_paths)
        except Exception as e:
            self.validation_errors.append(f"Failed to load CSV(s): {e}")
            return False
        return True


class DataReviewStep(CTkWizardStep):
    """Step 2: review summary stats and pick the return source."""

    def __init__(self, wizard: "MonteCarloWizard"):
        super().__init__(wizard, "Data Review")
        self.mc_wizard = wizard
        self.return_source_var: Optional[ctk.StringVar] = None
        self.summary_pct_frame: Optional[ctk.CTkFrame] = None
        self.summary_r_frame: Optional[ctk.CTkFrame] = None
        self.warnings_frame: Optional[ctk.CTkFrame] = None

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        # Return source selection
        Theme.create_header(parent, "Return source", size="s").pack(
            anchor="w", pady=(0, Sizes.PAD_XS),
        )
        Theme.create_hint(
            parent,
            "% returns and R-multiples model different sizing schemes and will "
            "give different results:\n"
            "  - % returns: equity update = equity * (1 + risk * pct). Models "
            "fixed-fraction positions (1% of equity invested per trade).\n"
            "  - R-multiples: equity update = equity * (1 + risk * R). Models "
            "risk-based sizing (a fixed % of equity risked per trade), so a tight "
            "stop means a large position. This is the standard quant interpretation.\n"
            "R-multiples drop trades without a valid initial_stop_loss, so the "
            "trade count and win rate may differ from % returns.",
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.return_source_var = ctk.StringVar(value=TradeLogReturnSource.PCT_RETURN.value)

        radio_row = Theme.create_frame(parent)
        radio_row.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_radiobutton(
            radio_row, "Trade returns (%)", self.return_source_var,
            TradeLogReturnSource.PCT_RETURN.value,
            command=self._refresh,
        ).pack(side="left", padx=(0, Sizes.PAD_L))

        Theme.create_radiobutton(
            radio_row, "R-multiples (computed from price + stop)",
            self.return_source_var,
            TradeLogReturnSource.R_MULTIPLE.value,
            command=self._refresh,
        ).pack(side="left")

        # Side-by-side summary stats
        Theme.create_header(parent, "Summary statistics", size="s").pack(
            anchor="w", pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        stats_row = Theme.create_frame(parent)
        stats_row.pack(fill="x", pady=Sizes.PAD_XS)
        self.summary_pct_frame = Theme.create_card(stats_row)
        self.summary_pct_frame.pack(side="left", fill="both", expand=True,
                                    padx=(0, Sizes.PAD_S))
        self.summary_r_frame = Theme.create_card(stats_row)
        self.summary_r_frame.pack(side="left", fill="both", expand=True,
                                  padx=(Sizes.PAD_S, 0))

        # Warnings
        Theme.create_header(parent, "Validation", size="s").pack(
            anchor="w", pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        self.warnings_frame = Theme.create_card(parent)
        self.warnings_frame.pack(fill="x", pady=Sizes.PAD_XS)

    def on_enter(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        log = self.mc_wizard.loaded_log
        if log is None:
            return

        source = TradeLogReturnSource(self.return_source_var.get())
        self.mc_wizard.return_source = source
        self.mc_wizard.return_pool = log.returns_for(source)

        self._render_card(
            self.summary_pct_frame, "Trade returns (%)",
            log.summary_stats(TradeLogReturnSource.PCT_RETURN),
            unit="%", scale=100.0,
            is_selected=(source == TradeLogReturnSource.PCT_RETURN),
        )
        self._render_card(
            self.summary_r_frame, "R-multiples",
            log.summary_stats(TradeLogReturnSource.R_MULTIPLE),
            unit="R", scale=1.0,
            is_selected=(source == TradeLogReturnSource.R_MULTIPLE),
        )

        # Outlier advisory for R-multiples
        r_stats = log.summary_stats(TradeLogReturnSource.R_MULTIPLE)
        extras: List[str] = []
        if r_stats["count"] > 0 and r_stats["outliers_10"] > 0:
            extras.append(
                f"R-multiple pool contains {r_stats['outliers_10']} trade(s) "
                f"with |R| > 10 (max |R|={r_stats['abs_max']:.1f}). These "
                f"likely come from very tight stops and can dominate Monte "
                f"Carlo results; consider enabling R-multiple clipping in "
                f"the Advanced step."
            )
        # Win-rate / count mismatch advisory
        pct_stats = log.summary_stats(TradeLogReturnSource.PCT_RETURN)
        if (pct_stats["count"] > 0 and r_stats["count"] > 0
                and pct_stats["count"] != r_stats["count"]):
            diff = pct_stats["count"] - r_stats["count"]
            extras.append(
                f"R-multiples are computed on {r_stats['count']} of "
                f"{pct_stats['count']} trades (excluded {diff} without a "
                f"valid initial_stop_loss). This is why the two pools have "
                f"different win rates."
            )
        if self.mc_wizard.return_pool is None or self.mc_wizard.return_pool.size == 0:
            extras.append(
                "Selected return source has no usable data. Pick the other "
                "source or load CSVs with the missing columns."
            )
        self._refresh_warnings(extras)

    def _render_card(self, frame, title: str, stats: dict, *,
                     unit: str, scale: float, is_selected: bool) -> None:
        for child in frame.winfo_children():
            child.destroy()

        border_color = Colors.PRIMARY_LIGHT if is_selected else Colors.BORDER
        try:
            frame.configure(border_color=border_color, border_width=2)
        except Exception:
            pass

        header_row = Theme.create_frame(frame)
        header_row.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, 0))
        Theme.create_label(
            header_row, title, font=Fonts.LABEL_BOLD,
            text_color=Colors.PRIMARY_LIGHT if is_selected else Colors.TEXT_SECONDARY,
        ).pack(side="left")
        if is_selected:
            Theme.create_label(
                header_row, "* used for simulation", font=Fonts.HINT,
                text_color=Colors.PRIMARY_LIGHT,
            ).pack(side="left", padx=(Sizes.PAD_S, 0))

        if stats["count"] == 0:
            Theme.create_label(
                frame, "No usable data for this source.",
                text_color=Colors.WARNING, wraplength=350,
            ).pack(anchor="w", padx=Sizes.PAD_M, pady=Sizes.PAD_M)
            return

        rows = [
            ("Trades", f"{stats['count']:,}"),
            ("Win rate", f"{stats['win_rate'] * 100:.2f}%"),
            ("Mean", f"{stats['mean'] * scale:+.4f} {unit}"),
            ("Median", f"{stats['median'] * scale:+.4f} {unit}"),
            ("Std dev", f"{stats['std'] * scale:.4f} {unit}"),
            ("Min", f"{stats['min'] * scale:+.4f} {unit}"),
            ("Max", f"{stats['max'] * scale:+.4f} {unit}"),
            ("5th pct", f"{stats['p5'] * scale:+.4f} {unit}"),
            ("95th pct", f"{stats['p95'] * scale:+.4f} {unit}"),
            ("Skew", f"{stats['skew']:+.3f}"),
        ]
        # Outlier counts make sense for R-multiples (where |R| is unitless)
        if unit == "R":
            rows.extend([
                ("|R| > 5", f"{stats['outliers_5']}"),
                ("|R| > 10", f"{stats['outliers_10']}"),
                ("|R| > 20", f"{stats['outliers_20']}"),
            ])

        grid = Theme.create_frame(frame)
        grid.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        for i, (k, v) in enumerate(rows):
            Theme.create_label(
                grid, k + ":", font=Fonts.LABEL_BOLD,
                text_color=Colors.TEXT_SECONDARY,
            ).grid(row=i, column=0, sticky="w", padx=(0, Sizes.PAD_M), pady=2)

            # Outlier rows colored if non-zero
            value_color = Colors.TEXT_PRIMARY
            if unit == "R" and k.startswith("|R|"):
                try:
                    n = int(v)
                    if n > 0:
                        value_color = Colors.WARNING if k == "|R| > 5" else Colors.ERROR
                except ValueError:
                    pass
            Theme.create_label(
                grid, v, font=Fonts.MONO, text_color=value_color,
            ).grid(row=i, column=1, sticky="w", pady=2)

    def _refresh_warnings(self, extra: Optional[List[str]] = None) -> None:
        for child in self.warnings_frame.winfo_children():
            child.destroy()
        msgs = list(self.mc_wizard.loaded_log.warnings) if self.mc_wizard.loaded_log else []
        if extra:
            msgs.extend(extra)
        if not msgs:
            Theme.create_label(
                self.warnings_frame, "No issues detected.",
                text_color=Colors.SUCCESS,
            ).pack(anchor="w", padx=Sizes.PAD_M, pady=Sizes.PAD_M)
            return
        for m in msgs:
            Theme.create_label(
                self.warnings_frame, "* " + m,
                text_color=Colors.WARNING, wraplength=700, justify="left",
            ).pack(anchor="w", padx=Sizes.PAD_M, pady=Sizes.PAD_XS)

    def get_summary(self) -> Dict[str, str]:
        if self.mc_wizard.loaded_log is None:
            return {}
        log = self.mc_wizard.loaded_log
        source = self.mc_wizard.return_source
        arr = log.returns_for(source)
        return {
            "Source": "% returns" if source == TradeLogReturnSource.PCT_RETURN else "R-multiples",
            "Trades": str(arr.size),
        }

    def validate(self) -> bool:
        self.validation_errors = []
        if self.mc_wizard.return_pool is None or self.mc_wizard.return_pool.size == 0:
            self.validation_errors.append(
                "No usable returns for the selected source - pick a different "
                "source or load files that contain the required columns."
            )
            return False
        return True


class SimulationStep(CTkWizardStep):
    """Step 3: general settings, position sizing, sampling method."""

    def __init__(self, wizard: "MonteCarloWizard"):
        super().__init__(wizard, "Simulation Settings")
        self.mc_wizard = wizard

        # Variables
        self.num_sims_var: Optional[ctk.StringVar] = None
        self.num_trades_var: Optional[ctk.StringVar] = None
        self.initial_capital_var: Optional[ctk.StringVar] = None
        self.risk_var: Optional[ctk.StringVar] = None
        self.sizing_var: Optional[ctk.StringVar] = None
        self.sampling_var: Optional[ctk.StringVar] = None
        self.block_size_var: Optional[ctk.StringVar] = None
        self.runtime_label: Optional[ctk.CTkLabel] = None
        self.block_size_entry: Optional[ctk.CTkEntry] = None
        self.block_size_label: Optional[ctk.CTkLabel] = None

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        # General settings
        general = Theme.create_card(parent)
        general.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_header(general, "General", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        gframe = Theme.create_frame(general)
        gframe.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.num_sims_var = ctk.StringVar(value="5000")
        self.num_trades_var = ctk.StringVar(value="200")
        self.initial_capital_var = ctk.StringVar(value="10000")

        self._add_field(gframe, 0, "Number of simulations",
                        self.num_sims_var, "5000",
                        hint="Defaults to 5000. Up to 50,000 supported.",
                        on_change=self._update_runtime)
        self._add_field(gframe, 1, "Trades per simulation",
                        self.num_trades_var, "200",
                        on_change=self._update_runtime)
        self._add_field(gframe, 2, "Initial capital",
                        self.initial_capital_var, "10000")

        # Runtime estimate
        self.runtime_label = Theme.create_label(
            general, "", font=Fonts.LABEL_BOLD,
            text_color=Colors.PRIMARY_LIGHT,
        )
        self.runtime_label.pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Position sizing
        sizing = Theme.create_card(parent)
        sizing.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_header(sizing, "Position sizing", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        sframe = Theme.create_frame(sizing)
        sframe.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.risk_var = ctk.StringVar(value="1.0")
        self._add_field(sframe, 0, "Risk per trade (%)", self.risk_var, "1.0",
                        hint="% of equity risked per trade.")

        self.sizing_var = ctk.StringVar(value=SizingMethod.COMPOUNDING.value)
        Theme.create_label(
            sframe, "Sizing scheme:", font=Fonts.LABEL_BOLD,
        ).grid(row=1, column=0, sticky="w", pady=Sizes.PAD_S)
        radio_row = Theme.create_frame(sframe)
        radio_row.grid(row=1, column=1, sticky="w", pady=Sizes.PAD_S)
        Theme.create_radiobutton(
            radio_row, "Compounding", self.sizing_var,
            SizingMethod.COMPOUNDING.value,
        ).pack(side="left", padx=(0, Sizes.PAD_M))
        Theme.create_radiobutton(
            radio_row, "Fixed (on initial capital)", self.sizing_var,
            SizingMethod.FIXED.value,
        ).pack(side="left")

        # Sampling method
        sampling = Theme.create_card(parent)
        sampling.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_header(sampling, "Sampling method", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        smpframe = Theme.create_frame(sampling)
        smpframe.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.sampling_var = ctk.StringVar(value=SamplingMethod.SIMPLE_BOOTSTRAP.value)
        Theme.create_label(
            smpframe, "Method:", font=Fonts.LABEL_BOLD,
        ).grid(row=0, column=0, sticky="w", pady=Sizes.PAD_S)
        method_menu = Theme.create_optionmenu(
            smpframe, [SamplingMethod.SIMPLE_BOOTSTRAP.value, SamplingMethod.BLOCK_BOOTSTRAP.value],
            variable=self.sampling_var, command=self._on_sampling_changed,
        )
        method_menu.grid(row=0, column=1, sticky="w", pady=Sizes.PAD_S, padx=(Sizes.PAD_M, 0))

        self.block_size_var = ctk.StringVar(value="10")
        self.block_size_label = Theme.create_label(
            smpframe, "Block size:", font=Fonts.LABEL_BOLD,
        )
        self.block_size_label.grid(row=1, column=0, sticky="w", pady=Sizes.PAD_S)
        self.block_size_entry = Theme.create_entry(
            smpframe, placeholder="10", width=120,
        )
        self.block_size_entry.configure(textvariable=self.block_size_var)
        self.block_size_entry.grid(row=1, column=1, sticky="w", pady=Sizes.PAD_S, padx=(Sizes.PAD_M, 0))
        Theme.create_hint(
            smpframe,
            "Typical: 5-20. Larger blocks preserve more streak structure but "
            "reduce randomness.",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 0))

        self._on_sampling_changed(self.sampling_var.get())
        self._update_runtime()

    def _add_field(self, parent, row, label_text, var, placeholder, hint=None,
                   on_change=None) -> None:
        Theme.create_label(
            parent, label_text + ":", font=Fonts.LABEL_BOLD,
        ).grid(row=row, column=0, sticky="w", pady=Sizes.PAD_S)
        entry = Theme.create_entry(parent, placeholder=placeholder, width=160)
        entry.configure(textvariable=var)
        entry.grid(row=row, column=1, sticky="w", pady=Sizes.PAD_S, padx=(Sizes.PAD_M, 0))
        if hint:
            Theme.create_hint(parent, hint).grid(
                row=row, column=2, sticky="w", padx=(Sizes.PAD_M, 0),
            )
        if on_change is not None:
            var.trace_add("write", lambda *args: on_change())

    def _on_sampling_changed(self, value: str) -> None:
        is_block = (value == SamplingMethod.BLOCK_BOOTSTRAP.value)
        state = "normal" if is_block else "disabled"
        if self.block_size_entry:
            self.block_size_entry.configure(state=state)
        if self.block_size_label:
            color = Colors.TEXT_PRIMARY if is_block else Colors.TEXT_MUTED
            self.block_size_label.configure(text_color=color)

    def _update_runtime(self) -> None:
        if self.runtime_label is None:
            return
        try:
            n_sim = int(self.num_sims_var.get())
            n_trd = int(self.num_trades_var.get())
        except (ValueError, TypeError):
            self.runtime_label.configure(text="Estimated runtime: -")
            return
        if n_sim <= 0 or n_trd <= 0:
            self.runtime_label.configure(text="Estimated runtime: -")
            return
        # Use the heaviest path for the estimate (drawdown logic).
        with_dd = self.mc_wizard.has_drawdown_logic()
        secs = MonteCarloSimulator.estimate_runtime(n_sim, n_trd, with_dd)
        size = n_sim * n_trd
        self.runtime_label.configure(
            text=f"Estimated runtime: {_format_runtime(secs)}  "
                 f"({size:,} trade-updates)"
        )

    def get_summary(self) -> Dict[str, str]:
        s = {
            "Simulations": self.num_sims_var.get() if self.num_sims_var else "-",
            "Trades / sim": self.num_trades_var.get() if self.num_trades_var else "-",
            "Initial capital": self.initial_capital_var.get() if self.initial_capital_var else "-",
            "Risk per trade": (self.risk_var.get() + "%") if self.risk_var else "-",
            "Sizing": self.sizing_var.get() if self.sizing_var else "-",
            "Sampling": self.sampling_var.get() if self.sampling_var else "-",
        }
        if self.sampling_var and self.sampling_var.get() == SamplingMethod.BLOCK_BOOTSTRAP.value:
            s["Block size"] = self.block_size_var.get() if self.block_size_var else "-"
        return s

    def validate(self) -> bool:
        self.validation_errors = []

        def parse_int(name, var, minimum=1):
            try:
                v = int(var.get())
            except (ValueError, TypeError):
                self.validation_errors.append(f"{name} must be an integer.")
                return None
            if v < minimum:
                self.validation_errors.append(f"{name} must be >= {minimum}.")
                return None
            return v

        def parse_float(name, var, minimum=None, maximum=None):
            try:
                v = float(var.get())
            except (ValueError, TypeError):
                self.validation_errors.append(f"{name} must be numeric.")
                return None
            if minimum is not None and v < minimum:
                self.validation_errors.append(f"{name} must be >= {minimum}.")
                return None
            if maximum is not None and v > maximum:
                self.validation_errors.append(f"{name} must be <= {maximum}.")
                return None
            return v

        n_sim = parse_int("Number of simulations", self.num_sims_var)
        n_trd = parse_int("Trades per simulation", self.num_trades_var)
        ic = parse_float("Initial capital", self.initial_capital_var, minimum=0.01)
        risk = parse_float("Risk per trade (%)", self.risk_var, minimum=1e-6, maximum=100.0)

        if self.sampling_var.get() == SamplingMethod.BLOCK_BOOTSTRAP.value:
            block = parse_int("Block size", self.block_size_var)
            if block is not None and n_trd is not None and block > n_trd:
                self.validation_errors.append("Block size cannot exceed trades per simulation.")

        return not self.validation_errors


class AdvancedStep(CTkWizardStep):
    """Step 4: drawdown-based risk reduction, costs, slippage, seed."""

    def __init__(self, wizard: "MonteCarloWizard"):
        super().__init__(wizard, "Advanced")
        self.mc_wizard = wizard

        self.dd_enabled_var: Optional[ctk.BooleanVar] = None
        self.dd_threshold_var: Optional[ctk.StringVar] = None
        self.dd_reduced_risk_var: Optional[ctk.StringVar] = None
        self.commission_var: Optional[ctk.StringVar] = None
        self.slippage_var: Optional[ctk.StringVar] = None
        self.clip_enabled_var: Optional[ctk.BooleanVar] = None
        self.clip_value_var: Optional[ctk.StringVar] = None
        self.clip_entry: Optional[ctk.CTkEntry] = None
        self.seed_enabled_var: Optional[ctk.BooleanVar] = None
        self.seed_var: Optional[ctk.StringVar] = None

        self.dd_threshold_entry: Optional[ctk.CTkEntry] = None
        self.dd_reduced_risk_entry: Optional[ctk.CTkEntry] = None
        self.seed_entry: Optional[ctk.CTkEntry] = None

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        # --- Drawdown-based risk reduction ----
        dd_card = Theme.create_card(parent)
        dd_card.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_header(
            dd_card, "Drawdown-based risk reduction", size="s",
        ).pack(anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS))

        self.dd_enabled_var = ctk.BooleanVar(value=False)
        Theme.create_switch(
            dd_card, "Reduce risk during drawdowns",
            variable=self.dd_enabled_var, command=self._toggle_dd,
        ).pack(anchor="w", padx=Sizes.PAD_M)

        Theme.create_hint(
            dd_card,
            "When equity is below the running peak by more than the threshold, "
            "switch to the reduced-risk %.",
        ).pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        ddframe = Theme.create_frame(dd_card)
        ddframe.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.dd_threshold_var = ctk.StringVar(value="10")
        self.dd_reduced_risk_var = ctk.StringVar(value="0.5")

        Theme.create_label(
            ddframe, "Drawdown threshold (%):", font=Fonts.LABEL_BOLD,
        ).grid(row=0, column=0, sticky="w", pady=Sizes.PAD_S)
        self.dd_threshold_entry = Theme.create_entry(ddframe, placeholder="10", width=120)
        self.dd_threshold_entry.configure(textvariable=self.dd_threshold_var)
        self.dd_threshold_entry.grid(row=0, column=1, sticky="w", padx=(Sizes.PAD_M, 0))

        Theme.create_label(
            ddframe, "Reduced risk per trade (%):", font=Fonts.LABEL_BOLD,
        ).grid(row=1, column=0, sticky="w", pady=Sizes.PAD_S)
        self.dd_reduced_risk_entry = Theme.create_entry(ddframe, placeholder="0.5", width=120)
        self.dd_reduced_risk_entry.configure(textvariable=self.dd_reduced_risk_var)
        self.dd_reduced_risk_entry.grid(row=1, column=1, sticky="w", padx=(Sizes.PAD_M, 0))

        # --- Costs ----
        cost_card = Theme.create_card(parent)
        cost_card.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_header(cost_card, "Costs", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        Theme.create_hint(
            cost_card,
            "Subtracted from each sampled return before equity is updated. "
            "NOTE: framework trade logs already include commission in pl_pct, "
            "so default these to 0 unless your CSV contains gross returns.",
        ).pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        cframe = Theme.create_frame(cost_card)
        cframe.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.commission_var = ctk.StringVar(value="0")
        self.slippage_var = ctk.StringVar(value="0")

        Theme.create_label(
            cframe, "Commission per trade (%):", font=Fonts.LABEL_BOLD,
        ).grid(row=0, column=0, sticky="w", pady=Sizes.PAD_S)
        e1 = Theme.create_entry(cframe, placeholder="0", width=120)
        e1.configure(textvariable=self.commission_var)
        e1.grid(row=0, column=1, sticky="w", padx=(Sizes.PAD_M, 0))

        Theme.create_label(
            cframe, "Slippage per trade (%):", font=Fonts.LABEL_BOLD,
        ).grid(row=1, column=0, sticky="w", pady=Sizes.PAD_S)
        e2 = Theme.create_entry(cframe, placeholder="0", width=120)
        e2.configure(textvariable=self.slippage_var)
        e2.grid(row=1, column=1, sticky="w", padx=(Sizes.PAD_M, 0))

        # --- Outlier clipping ----
        clip_card = Theme.create_card(parent)
        clip_card.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_header(clip_card, "Return clipping", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )

        self.clip_enabled_var = ctk.BooleanVar(value=False)
        Theme.create_switch(
            clip_card, "Clip extreme sampled returns",
            variable=self.clip_enabled_var, command=self._toggle_clip,
        ).pack(anchor="w", padx=Sizes.PAD_M)

        Theme.create_hint(
            clip_card,
            "Caps every sampled return at +/- the value below before equity is "
            "updated. Most useful for R-multiples: trades with tight stops can "
            "produce R values of -50 or worse, which wipe equity in a single "
            "sample. A clip of 5-10 R is a common floor for risk-based sizing.",
        ).pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        clipframe = Theme.create_frame(clip_card)
        clipframe.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.clip_value_var = ctk.StringVar(value="10")
        Theme.create_label(
            clipframe, "Clip threshold:", font=Fonts.LABEL_BOLD,
        ).grid(row=0, column=0, sticky="w", pady=Sizes.PAD_S)
        self.clip_entry = Theme.create_entry(clipframe, placeholder="10", width=120)
        self.clip_entry.configure(textvariable=self.clip_value_var)
        self.clip_entry.grid(row=0, column=1, sticky="w", padx=(Sizes.PAD_M, 0))
        Theme.create_hint(
            clipframe,
            "Units match the selected return source (R for R-multiples, "
            "% for trade returns).",
        ).grid(row=0, column=2, sticky="w", padx=(Sizes.PAD_M, 0))

        # --- Reproducibility ----
        seed_card = Theme.create_card(parent)
        seed_card.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_header(seed_card, "Reproducibility", size="s").pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS),
        )
        self.seed_enabled_var = ctk.BooleanVar(value=False)
        Theme.create_switch(
            seed_card, "Use fixed random seed",
            variable=self.seed_enabled_var, command=self._toggle_seed,
        ).pack(anchor="w", padx=Sizes.PAD_M)

        sframe = Theme.create_frame(seed_card)
        sframe.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_S, Sizes.PAD_M))
        self.seed_var = ctk.StringVar(value="42")
        Theme.create_label(
            sframe, "Seed:", font=Fonts.LABEL_BOLD,
        ).grid(row=0, column=0, sticky="w", pady=Sizes.PAD_S)
        self.seed_entry = Theme.create_entry(sframe, placeholder="42", width=120)
        self.seed_entry.configure(textvariable=self.seed_var)
        self.seed_entry.grid(row=0, column=1, sticky="w", padx=(Sizes.PAD_M, 0))

        self._toggle_dd()
        self._toggle_seed()
        self._toggle_clip()

    def on_enter(self) -> None:
        # Update parent estimate now that the user can toggle DD logic.
        sim_step = self.mc_wizard.steps[2]
        if isinstance(sim_step, SimulationStep):
            sim_step._update_runtime()

    def _toggle_dd(self) -> None:
        enabled = bool(self.dd_enabled_var.get())
        state = "normal" if enabled else "disabled"
        for w in (self.dd_threshold_entry, self.dd_reduced_risk_entry):
            if w:
                w.configure(state=state)

    def _toggle_seed(self) -> None:
        enabled = bool(self.seed_enabled_var.get())
        state = "normal" if enabled else "disabled"
        if self.seed_entry:
            self.seed_entry.configure(state=state)

    def _toggle_clip(self) -> None:
        enabled = bool(self.clip_enabled_var.get()) if self.clip_enabled_var else False
        state = "normal" if enabled else "disabled"
        if self.clip_entry:
            self.clip_entry.configure(state=state)

    def get_summary(self) -> Dict[str, str]:
        s: Dict[str, str] = {}
        if self.dd_enabled_var and self.dd_enabled_var.get():
            s["DD threshold"] = (self.dd_threshold_var.get() or "-") + "%"
            s["Reduced risk"] = (self.dd_reduced_risk_var.get() or "-") + "%"
        else:
            s["DD reduction"] = "off"
        s["Commission"] = (self.commission_var.get() if self.commission_var else "-") + "%"
        s["Slippage"] = (self.slippage_var.get() if self.slippage_var else "-") + "%"
        if self.clip_enabled_var and self.clip_enabled_var.get():
            s["Clip"] = "+/- " + (self.clip_value_var.get() or "-")
        else:
            s["Clip"] = "off"
        if self.seed_enabled_var and self.seed_enabled_var.get():
            s["Seed"] = self.seed_var.get() if self.seed_var else "-"
        else:
            s["Seed"] = "random"
        return s

    def validate(self) -> bool:
        self.validation_errors = []

        if self.dd_enabled_var and self.dd_enabled_var.get():
            try:
                t = float(self.dd_threshold_var.get())
                if not (0 < t < 100):
                    raise ValueError
            except (ValueError, TypeError):
                self.validation_errors.append("Drawdown threshold must be in (0, 100)%.")
            try:
                rr = float(self.dd_reduced_risk_var.get())
                if rr < 0:
                    raise ValueError
            except (ValueError, TypeError):
                self.validation_errors.append("Reduced risk must be a non-negative number.")

        for name, var in (("Commission", self.commission_var), ("Slippage", self.slippage_var)):
            try:
                v = float(var.get())
                if v < 0:
                    raise ValueError
            except (ValueError, TypeError):
                self.validation_errors.append(f"{name} must be a non-negative number.")

        if self.clip_enabled_var and self.clip_enabled_var.get():
            try:
                c = float(self.clip_value_var.get())
                if c <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                self.validation_errors.append("Clip threshold must be a positive number.")

        if self.seed_enabled_var and self.seed_enabled_var.get():
            try:
                int(self.seed_var.get())
            except (ValueError, TypeError):
                self.validation_errors.append("Seed must be an integer.")

        return not self.validation_errors


# ============================================================================
# Results window
# ============================================================================

class MonteCarloResultsWindow(ctk.CTkToplevel):
    """Results window with tabs for charts, metrics, and exports."""

    MAX_PLOTTED_CURVES = 100

    def __init__(self, parent, sim_result: SimulationResult,
                 sim_metrics: SimulationMetrics,
                 loaded_log: Optional[LoadedTradeLog] = None,
                 source: Optional[TradeLogReturnSource] = None):
        super().__init__(parent)
        self.title("Monte Carlo Simulation Results")
        self.geometry("1200x800")
        self.configure(fg_color=Colors.BG_DARK)

        self.sim_result = sim_result
        self.sim_metrics = sim_metrics
        self.loaded_log = loaded_log
        self.source = source
        self.analyzer = SimulationAnalyzer(sim_result)
        self._figures: Dict[str, plt.Figure] = {}

        self._create_layout()
        self._populate_tabs()

    def _create_layout(self) -> None:
        # Header bar
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(
            header, "Monte Carlo Results", size="l",
        ).pack(side="left")

        # Export buttons
        Theme.create_button(
            header, "Export curves CSV", command=self._export_csv,
            style="secondary", width=160,
        ).pack(side="right", padx=(Sizes.PAD_S, 0))
        Theme.create_button(
            header, "Export charts PNG", command=self._export_png,
            style="secondary", width=160,
        ).pack(side="right")

        # Tab view
        self.tabs = Theme.create_tabview(self)
        self.tabs.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        for name in ("Equity Curves", "Final Distribution", "Drawdowns",
                     "Percentile Curves", "R-Multiple Distribution", "Metrics"):
            self.tabs.add(name)

    def _populate_tabs(self) -> None:
        self._build_equity_tab(self.tabs.tab("Equity Curves"))
        self._build_final_dist_tab(self.tabs.tab("Final Distribution"))
        self._build_drawdown_tab(self.tabs.tab("Drawdowns"))
        self._build_percentile_tab(self.tabs.tab("Percentile Curves"))
        self._build_r_multiple_tab(self.tabs.tab("R-Multiple Distribution"))
        self._build_metrics_tab(self.tabs.tab("Metrics"))

    # ---- chart builders ----

    def _embed_fig(self, fig: plt.Figure, parent, key: str) -> None:
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=Sizes.PAD_S, pady=Sizes.PAD_S)
        self._figures[key] = fig

    def _build_equity_tab(self, parent) -> None:
        ec = self.analyzer.equity_curves
        rng = np.random.default_rng(0)
        idx = self.analyzer.sample_curve_indices(self.MAX_PLOTTED_CURVES, rng)
        sample = ec[idx]

        fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
        x = np.arange(ec.shape[1])
        for curve in sample:
            ax.plot(x, curve, color=Colors.CHART_LINE, alpha=0.18, linewidth=0.8)
        # Median overlay
        median = np.percentile(ec, 50, axis=0)
        ax.plot(x, median, color=Colors.CHART_POSITIVE, linewidth=2.0, label="Median")

        ax.axhline(self.sim_result.config.initial_capital, color=Colors.TEXT_MUTED,
                   linestyle="--", linewidth=1, alpha=0.6, label="Initial capital")

        ax.set_xlabel("Trade index")
        ax.set_ylabel("Equity")
        ax.set_title(f"Equity curves ({len(sample)} of {ec.shape[0]:,} simulations)")
        ax.legend(loc="upper left")
        _apply_dark_matplotlib_style(fig)
        fig.tight_layout()
        self._embed_fig(fig, parent, "equity_curves")

    def _build_final_dist_tab(self, parent) -> None:
        final = self.analyzer.final_equity
        cfg = self.sim_result.config

        fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
        ax.hist(final, bins=60, color=Colors.CHART_LINE, alpha=0.85,
                edgecolor=Colors.BORDER)
        ax.axvline(cfg.initial_capital, color=Colors.TEXT_MUTED, linestyle="--",
                   linewidth=1.2, label="Initial capital")
        ax.axvline(np.median(final), color=Colors.CHART_POSITIVE, linewidth=2,
                   label="Median")
        ax.axvline(np.percentile(final, 5), color=Colors.WARNING, linewidth=1.5,
                   linestyle=":", label="5th pct")
        ax.axvline(np.percentile(final, 95), color=Colors.SUCCESS, linewidth=1.5,
                   linestyle=":", label="95th pct")
        ax.set_xlabel("Final equity")
        ax.set_ylabel("Count")
        ax.set_title(f"Final equity distribution ({final.size:,} simulations)")
        ax.legend(loc="upper right")
        _apply_dark_matplotlib_style(fig)
        fig.tight_layout()
        self._embed_fig(fig, parent, "final_distribution")

    def _build_drawdown_tab(self, parent) -> None:
        dd = self.analyzer.max_drawdowns() * 100.0  # to percent
        fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
        ax.hist(dd, bins=60, color=Colors.CHART_NEGATIVE, alpha=0.85,
                edgecolor=Colors.BORDER)
        ax.axvline(np.median(dd), color=Colors.CHART_POSITIVE, linewidth=2,
                   label="Median")
        ax.axvline(np.percentile(dd, 95), color=Colors.WARNING, linewidth=1.5,
                   linestyle=":", label="95th pct")
        ax.axvline(np.max(dd), color=Colors.ERROR, linewidth=1.5,
                   linestyle=":", label="Worst")
        ax.set_xlabel("Max drawdown (%)")
        ax.set_ylabel("Count")
        ax.set_title("Maximum drawdown distribution")
        ax.legend(loc="upper right")
        _apply_dark_matplotlib_style(fig)
        fig.tight_layout()
        self._embed_fig(fig, parent, "drawdown_distribution")

    def _build_percentile_tab(self, parent) -> None:
        curves = self.analyzer.percentile_curves((5.0, 50.0, 95.0))
        fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
        x = np.arange(self.analyzer.equity_curves.shape[1])

        ax.fill_between(x, curves[5.0], curves[95.0], color=Colors.CHART_LINE,
                        alpha=0.20, label="5th-95th pct band")
        ax.plot(x, curves[5.0], color=Colors.WARNING, linewidth=1.6, label="5th pct")
        ax.plot(x, curves[50.0], color=Colors.CHART_POSITIVE, linewidth=2.2, label="Median")
        ax.plot(x, curves[95.0], color=Colors.SUCCESS, linewidth=1.6, label="95th pct")
        ax.axhline(self.sim_result.config.initial_capital, color=Colors.TEXT_MUTED,
                   linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Trade index")
        ax.set_ylabel("Equity")
        ax.set_title("Equity-curve percentiles across simulations")
        ax.legend(loc="upper left")
        _apply_dark_matplotlib_style(fig)
        fig.tight_layout()
        self._embed_fig(fig, parent, "percentile_curves")

    def _build_r_multiple_tab(self, parent) -> None:
        """Histogram of R-multiples from the loaded trade log.

        Always plots the R-multiples computed from price + stop columns,
        independent of which return source was used for the simulation,
        so the user can inspect the underlying risk-normalised return
        distribution and spot outliers from tight stops.
        """
        if self.loaded_log is None:
            Theme.create_label(
                parent,
                "No loaded trade log was passed to the results window.",
                text_color=Colors.WARNING,
            ).pack(expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)
            return

        r = self.loaded_log.r_multiples
        if r.size == 0:
            Theme.create_label(
                parent,
                "No R-multiples could be computed from the input data. "
                "R-multiples require entry_price, exit_price and "
                "initial_stop_loss columns with valid stops.",
                wraplength=600, text_color=Colors.WARNING,
            ).pack(expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)
            return

        # Stats banner (above the chart)
        stats = self.loaded_log.summary_stats(TradeLogReturnSource.R_MULTIPLE)
        banner = Theme.create_frame(parent)
        banner.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_S, 0))
        rows = [
            ("Trades", f"{stats['count']:,}"),
            ("Win rate", f"{stats['win_rate'] * 100:.1f}%"),
            ("Mean", f"{stats['mean']:+.3f}R"),
            ("Median", f"{stats['median']:+.3f}R"),
            ("Min / Max", f"{stats['min']:+.2f} / {stats['max']:+.2f}R"),
            ("|R| > 5", f"{stats['outliers_5']}"),
            ("|R| > 10", f"{stats['outliers_10']}"),
            ("|R| > 20", f"{stats['outliers_20']}"),
        ]
        for i, (k, v) in enumerate(rows):
            Theme.create_label(
                banner, f"{k}: {v}",
                font=Fonts.MONO_S,
                text_color=Colors.TEXT_SECONDARY,
            ).grid(row=0, column=i, padx=Sizes.PAD_S, pady=Sizes.PAD_XS, sticky="w")

        # Histogram - main view clipped to a sensible range, plus a separate
        # "full range" line showing extremes
        p1, p99 = np.percentile(r, 1), np.percentile(r, 99)
        view_lo = min(p1, -1.0) - 0.5
        view_hi = max(p99, 1.0) + 0.5
        # Cap the view so a single extreme outlier doesn't squash the histogram
        view_lo = max(view_lo, -30.0)
        view_hi = min(view_hi, 30.0)
        in_view = (r >= view_lo) & (r <= view_hi)
        outside = int((~in_view).sum())

        fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
        ax.hist(r[in_view], bins=50,
                color=Colors.CHART_LINE, alpha=0.85, edgecolor=Colors.BORDER)
        ax.axvline(0, color=Colors.TEXT_MUTED, linewidth=1, alpha=0.6)
        ax.axvline(float(np.median(r)), color=Colors.CHART_POSITIVE,
                   linewidth=2, label=f"Median {np.median(r):+.2f}R")
        ax.axvline(float(np.mean(r)), color=Colors.PRIMARY_LIGHT,
                   linewidth=1.5, linestyle="--", label=f"Mean {np.mean(r):+.2f}R")
        # Mark clip threshold if used
        clip = self.sim_result.config.return_clip
        if clip > 0 and self.source == TradeLogReturnSource.R_MULTIPLE:
            ax.axvline(clip, color=Colors.WARNING, linewidth=1.5, linestyle=":",
                       label=f"Clip +/-{clip:.1f}R")
            ax.axvline(-clip, color=Colors.WARNING, linewidth=1.5, linestyle=":")

        title = "R-multiple distribution"
        if outside:
            title += f"  (view clipped to [{view_lo:.1f}, {view_hi:.1f}]R; " \
                     f"{outside} trade(s) outside)"
        ax.set_title(title)
        ax.set_xlabel("R-multiple")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left")
        _apply_dark_matplotlib_style(fig)
        fig.tight_layout()
        self._embed_fig(fig, parent, "r_multiple_distribution")

    def _build_metrics_tab(self, parent) -> None:
        m = self.sim_metrics
        cfg = self.sim_result.config

        wrapper = Theme.create_frame(parent)
        wrapper.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Two columns: outcomes | risk
        col_left = Theme.create_card(wrapper)
        col_left.pack(side="left", fill="both", expand=True, padx=(0, Sizes.PAD_S))
        col_right = Theme.create_card(wrapper)
        col_right.pack(side="left", fill="both", expand=True, padx=(Sizes.PAD_S, 0))

        def render_kv(card, title, rows):
            Theme.create_header(card, title, size="s").pack(
                anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S),
            )
            inner = Theme.create_frame(card)
            inner.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))
            for r, (k, v, color) in enumerate(rows):
                Theme.create_label(
                    inner, k + ":", font=Fonts.LABEL_BOLD,
                    text_color=Colors.TEXT_SECONDARY,
                ).grid(row=r, column=0, sticky="w", padx=(0, Sizes.PAD_M), pady=3)
                Theme.create_label(
                    inner, v, font=Fonts.MONO, text_color=color,
                ).grid(row=r, column=1, sticky="w", pady=3)

        ic = cfg.initial_capital

        outcome_rows = [
            ("Median final equity", f"{m.median_final_equity:,.2f}",
             Colors.SUCCESS if m.median_final_equity >= ic else Colors.ERROR),
            ("Mean final equity", f"{m.mean_final_equity:,.2f}",
             Colors.SUCCESS if m.mean_final_equity >= ic else Colors.ERROR),
            ("5th percentile final", f"{m.p5_final_equity:,.2f}", Colors.WARNING),
            ("95th percentile final", f"{m.p95_final_equity:,.2f}", Colors.TEXT_PRIMARY),
            ("Min final equity", f"{m.min_final_equity:,.2f}", Colors.ERROR),
            ("Max final equity", f"{m.max_final_equity:,.2f}", Colors.SUCCESS),
            ("Probability of loss", f"{m.probability_of_loss * 100:.2f}%",
             Colors.ERROR if m.probability_of_loss > 0.2 else Colors.TEXT_PRIMARY),
            ("Probability of ~ruin (50% loss)", f"{m.probability_of_ruin * 100:.2f}%",
             Colors.ERROR if m.probability_of_ruin > 0.05 else Colors.TEXT_PRIMARY),
        ]
        render_kv(col_left, "Outcomes", outcome_rows)

        risk_rows = [
            ("Median max drawdown", f"{m.median_max_drawdown * 100:.2f}%", Colors.WARNING),
            ("Mean max drawdown", f"{m.mean_max_drawdown * 100:.2f}%", Colors.WARNING),
            ("95th pct max drawdown", f"{m.p95_max_drawdown * 100:.2f}%", Colors.ERROR),
            ("Worst observed drawdown", f"{m.worst_max_drawdown * 100:.2f}%", Colors.ERROR),
            ("Median per-trade Sharpe", f"{m.median_sharpe:.3f}", Colors.TEXT_PRIMARY),
            ("Median per-trade growth", f"{m.median_cagr_equivalent * 100:+.4f}%",
             Colors.SUCCESS if m.median_cagr_equivalent >= 0 else Colors.ERROR),
            ("Simulations", f"{m.num_simulations:,}", Colors.TEXT_PRIMARY),
            ("Trades / sim", f"{m.num_trades:,}", Colors.TEXT_PRIMARY),
            ("Compute time", f"{m.elapsed_seconds:.2f}s", Colors.TEXT_PRIMARY),
        ]
        render_kv(col_right, "Risk", risk_rows)

    # ---- export ----

    def _export_csv(self) -> None:
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Export equity curves",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="monte_carlo_curves.csv",
        )
        if not path:
            return
        try:
            self.analyzer.to_csv(path)
            show_success(self, "Exported", f"Equity curves saved to:\n{path}")
        except Exception as e:
            show_error(self, "Export failed", str(e))

    def _export_png(self) -> None:
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Choose folder for PNG exports")
        if not folder:
            return
        try:
            folder_path = Path(folder)
            for key, fig in self._figures.items():
                fig.savefig(folder_path / f"monte_carlo_{key}.png",
                            dpi=150, facecolor=fig.get_facecolor())
            show_success(self, "Exported", f"Charts saved to:\n{folder}")
        except Exception as e:
            show_error(self, "Export failed", str(e))


# ============================================================================
# Run-progress dialog
# ============================================================================

class RunProgressDialog(ctk.CTkToplevel):
    """Modal progress dialog shown while the simulation is running."""

    def __init__(self, parent, on_cancel=None):
        super().__init__(parent)
        self.title("Running Monte Carlo...")
        self.geometry("520x180")
        self.resizable(False, False)
        self.configure(fg_color=Colors.BG_DARK)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.on_cancel = on_cancel

        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        self.message_label = Theme.create_label(
            content, "Initialising...", wraplength=460,
        )
        self.message_label.pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.progress = Theme.create_progressbar(content)
        self.progress.pack(fill="x", pady=Sizes.PAD_S)
        self.progress.set(0)

        self.percent_label = Theme.create_label(
            content, "0%", font=Fonts.BODY_S, text_color=Colors.TEXT_MUTED,
        )
        self.percent_label.pack(anchor="e", pady=(Sizes.PAD_XS, 0))

    def update_progress(self, frac: float, message: str) -> None:
        self.progress.set(max(0.0, min(1.0, frac)))
        self.percent_label.configure(text=f"{int(frac * 100)}%")
        if message:
            self.message_label.configure(text=message)
        self.update_idletasks()


# ============================================================================
# Wizard
# ============================================================================

class MonteCarloWizard(CTkWizardBase):
    """Top-level wizard tying the steps + simulator + results window together."""

    def __init__(self):
        super().__init__(title="Monte Carlo Simulation", width=1100, height=780)

        # Run-time state populated by steps
        self.loaded_log: Optional[LoadedTradeLog] = None
        self.return_source: Optional[TradeLogReturnSource] = None
        self.return_pool: Optional[np.ndarray] = None

        # Step references (we need direct access for cross-step signalling)
        self.load_step = LoadDataStep(self)
        self.review_step = DataReviewStep(self)
        self.sim_step = SimulationStep(self)
        self.advanced_step = AdvancedStep(self)

        for s in (self.load_step, self.review_step, self.sim_step, self.advanced_step):
            self.add_step(s)

        self.on_complete = self._start_run
        self.on_cancel = self._on_cancel

        # Background thread plumbing
        self._sim_thread: Optional[threading.Thread] = None
        self._progress_queue: queue.Queue = queue.Queue()
        self._progress_dialog: Optional[RunProgressDialog] = None
        self._sim_result: Optional[SimulationResult] = None
        self._sim_error: Optional[Exception] = None

        self.start()

    def has_drawdown_logic(self) -> bool:
        """Used by SimulationStep to compute the runtime estimate."""
        if self.advanced_step.dd_enabled_var is None:
            return False
        return bool(self.advanced_step.dd_enabled_var.get())

    def _get_final_button_text(self) -> str:
        return "Run Simulation"

    def _on_cancel(self) -> None:
        if ask_yes_no(self.root, "Cancel", "Cancel and close the wizard?"):
            self.root.destroy()

    # ---- run + progress -----

    def _resolve_clip_value(self) -> float:
        """Convert the user's clip threshold to simulator units.

        The simulator stores returns in fractional units for % returns (so a
        +1.5% trade is 0.015) and raw R for R-multiples. The user enters the
        threshold in display units (10 = "10%" for pct, 10 = "10R" for R).
        Return 0 to disable clipping.
        """
        if not (self.advanced_step.clip_enabled_var
                and self.advanced_step.clip_enabled_var.get()):
            return 0.0
        try:
            v = float(self.advanced_step.clip_value_var.get())
        except (ValueError, TypeError):
            return 0.0
        if v <= 0:
            return 0.0
        if self.return_source == TradeLogReturnSource.PCT_RETURN:
            return v / 100.0
        return v

    def _build_config(self) -> SimulationConfig:
        cfg = SimulationConfig(
            num_simulations=int(self.sim_step.num_sims_var.get()),
            num_trades=int(self.sim_step.num_trades_var.get()),
            initial_capital=float(self.sim_step.initial_capital_var.get()),
            sampling_method=SamplingMethod(self.sim_step.sampling_var.get()),
            block_size=int(self.sim_step.block_size_var.get())
            if self.sim_step.sampling_var.get() == SamplingMethod.BLOCK_BOOTSTRAP.value
            else 10,
            sizing_method=SizingMethod(self.sim_step.sizing_var.get()),
            risk_per_trade=float(self.sim_step.risk_var.get()) / 100.0,
            drawdown_risk_reduction=bool(self.advanced_step.dd_enabled_var.get()),
            drawdown_threshold=float(self.advanced_step.dd_threshold_var.get()) / 100.0
            if self.advanced_step.dd_enabled_var.get() else 0.10,
            reduced_risk=float(self.advanced_step.dd_reduced_risk_var.get()) / 100.0
            if self.advanced_step.dd_enabled_var.get() else 0.005,
            commission_pct=float(self.advanced_step.commission_var.get()) / 100.0,
            slippage_pct=float(self.advanced_step.slippage_var.get()) / 100.0,
            return_clip=self._resolve_clip_value(),
            random_seed=int(self.advanced_step.seed_var.get())
            if self.advanced_step.seed_enabled_var.get() else None,
        )
        errors = cfg.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return cfg

    def _start_run(self) -> None:
        try:
            cfg = self._build_config()
        except Exception as e:
            show_error(self.root, "Configuration error", str(e))
            return

        # Confirm long runs
        est = MonteCarloSimulator.estimate_runtime(
            cfg.num_simulations, cfg.num_trades, cfg.drawdown_risk_reduction,
        )
        if est > 30.0:
            ok = ask_yes_no(
                self.root, "Long run",
                f"Estimated runtime is {_format_runtime(est)}. Proceed?",
            )
            if not ok:
                return

        # Spin up progress dialog and worker thread
        self._sim_result = None
        self._sim_error = None
        self._progress_dialog = RunProgressDialog(self.root)

        self._sim_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(cfg,),
            daemon=True,
        )
        self._sim_thread.start()
        self.root.after(50, self._poll_progress)

    def _run_simulation_thread(self, cfg: SimulationConfig) -> None:
        def cb(frac: float, msg: str) -> None:
            self._progress_queue.put(("progress", frac, msg))
        try:
            sim = MonteCarloSimulator(cfg)
            result = sim.run(self.return_pool, progress_cb=cb)
            self._sim_result = result
        except Exception as e:
            self._sim_error = e
        finally:
            self._progress_queue.put(("done", None, None))

    def _poll_progress(self) -> None:
        try:
            while True:
                kind, frac, msg = self._progress_queue.get_nowait()
                if kind == "progress" and self._progress_dialog is not None:
                    self._progress_dialog.update_progress(frac, msg)
                elif kind == "done":
                    self._on_simulation_done()
                    return
        except queue.Empty:
            pass
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self.root.after(50, self._poll_progress)
        else:
            self._on_simulation_done()

    def _on_simulation_done(self) -> None:
        if self._progress_dialog is not None:
            try:
                self._progress_dialog.grab_release()
            except Exception:
                pass
            self._progress_dialog.destroy()
            self._progress_dialog = None

        if self._sim_error is not None:
            show_error(self.root, "Simulation failed", str(self._sim_error))
            return

        if self._sim_result is None:
            show_error(self.root, "Simulation failed", "No result produced.")
            return

        metrics = SimulationAnalyzer(self._sim_result).metrics()
        MonteCarloResultsWindow(
            self.root, self._sim_result, metrics,
            loaded_log=self.loaded_log, source=self.return_source,
        )


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    Theme.setup()
    wizard = MonteCarloWizard()
    wizard.run()


if __name__ == "__main__":
    main()
