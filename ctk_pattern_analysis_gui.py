"""
Pattern Analysis GUI (CustomTkinter).

Reads a trade log, detects MA-crossover buy/sell signals (same logic as the
AlphaTrend strategy: ``MA(L)`` shifted by ``offset`` crossing the unshifted
``MA(L)``) in the windows preceding each trade entry, and writes a multi-sheet
Excel report describing the density and pattern of those signals split by
trade outcome.

Layout
------
- Left: configuration card (trade log, data path, lookback windows, combo
  builder, output path, run button, log).
- Right: results summary card showing high-level analysis numbers once a run
  completes.

The actual analysis runs in a worker thread so the UI stays responsive.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog
from typing import List, Optional

import customtkinter as ctk

from Classes.Analysis.pattern_analysis_report import write_report
from Classes.Analysis.pattern_analyzer import (
    SUPPORTED_MA_LENGTHS,
    SUPPORTED_MA_TYPES,
    MAComboSpec,
    PatternAnalyzer,
    PatternAnalysisResult,
    default_combos,
    default_windows,
    load_combos_from_config,
    load_trade_log,
)
from Classes.GUI.ctk_theme import (
    Colors,
    Fonts,
    Sizes,
    Theme,
    show_error,
    show_info,
)


DEFAULT_DATA_PATH = Path("raw_data/daily")
DEFAULT_OUTPUT_PATH = Path("pattern_analysis_report.xlsx")


# ---------------------------------------------------------------------------
# Combo list widget
# ---------------------------------------------------------------------------

class ComboListPanel(ctk.CTkFrame):
    """Editable list of MA combos. Builder row + scrollable list of combos."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._combos: List[MAComboSpec] = []

        # Builder row
        builder = Theme.create_frame(self)
        builder.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_label(builder, "Type:").pack(side="left")
        self.type_var = ctk.StringVar(value="EMA")
        Theme.create_optionmenu(
            builder,
            values=list(SUPPORTED_MA_TYPES),
            variable=self.type_var,
            width=80,
        ).pack(side="left", padx=(Sizes.PAD_S, Sizes.PAD_M))

        Theme.create_label(builder, "Length:").pack(side="left")
        self.length_var = ctk.StringVar(value="20")
        Theme.create_optionmenu(
            builder,
            values=[str(n) for n in SUPPORTED_MA_LENGTHS],
            variable=self.length_var,
            width=70,
        ).pack(side="left", padx=(Sizes.PAD_S, Sizes.PAD_M))

        Theme.create_label(builder, "Offset:").pack(side="left")
        self.offset_var = ctk.StringVar(value="5")
        offset_entry = Theme.create_entry(builder, width=60)
        offset_entry.configure(textvariable=self.offset_var)
        offset_entry.pack(side="left", padx=(Sizes.PAD_S, Sizes.PAD_M))

        Theme.create_button(
            builder, "Add",
            command=self._add_from_builder,
            style="primary",
            width=80,
        ).pack(side="left")

        # Action row
        actions = Theme.create_frame(self)
        actions.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_button(
            actions, "Load Defaults",
            command=self.load_defaults,
            style="secondary",
            width=120,
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        Theme.create_button(
            actions, "Import Config...",
            command=self._import_config,
            style="secondary",
            width=130,
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        Theme.create_button(
            actions, "Clear",
            command=self.clear,
            style="ghost",
            width=70,
        ).pack(side="left")

        # Scrollable list of combos
        self.list_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_M,
            height=160,
        )
        self.list_frame.pack(fill="both", expand=True)

        self._render()

    # -- public API ----------------------------------------------------
    def get_combos(self) -> List[MAComboSpec]:
        return list(self._combos)

    def set_combos(self, combos: List[MAComboSpec]) -> None:
        # De-duplicate by label while preserving order.
        seen = set()
        unique = []
        for c in combos:
            if c.label not in seen:
                seen.add(c.label)
                unique.append(c)
        self._combos = unique
        self._render()

    def load_defaults(self) -> None:
        self.set_combos(default_combos())

    def clear(self) -> None:
        self._combos = []
        self._render()

    # -- internals -----------------------------------------------------
    def _add_from_builder(self) -> None:
        try:
            combo = MAComboSpec(
                ma_type=self.type_var.get(),
                ma_length=int(self.length_var.get()),
                ma_offset=int(self.offset_var.get()),
            )
        except (ValueError, TypeError) as exc:
            show_error(self, "Invalid Combo", str(exc))
            return
        self.set_combos(self._combos + [combo])

    def _import_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Import combo config",
            filetypes=[("Config files", "*.yaml *.yml *.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            combos, _ = load_combos_from_config(Path(path))
        except Exception as exc:
            show_error(self, "Failed to load config", str(exc))
            return
        self.set_combos(self._combos + combos)

    def _render(self) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()

        if not self._combos:
            Theme.create_hint(
                self.list_frame,
                "No combos selected. Use 'Load Defaults' or build one above.",
            ).pack(pady=Sizes.PAD_M)
            return

        for idx, combo in enumerate(self._combos):
            row = Theme.create_frame(self.list_frame)
            row.pack(fill="x", padx=Sizes.PAD_S, pady=2)

            Theme.create_label(
                row,
                f"{idx + 1}. {combo.label}  "
                f"({combo.ma_type} length={combo.ma_length}, offset={combo.ma_offset})",
                font=Fonts.BODY_M,
            ).pack(side="left", padx=(Sizes.PAD_S, 0))

            Theme.create_button(
                row, "Remove",
                command=lambda i=idx: self._remove(i),
                style="ghost",
                width=70,
            ).pack(side="right")

    def _remove(self, index: int) -> None:
        if 0 <= index < len(self._combos):
            del self._combos[index]
            self._render()


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class CTkPatternAnalysisGUI(ctk.CTk):
    """Main Pattern Analysis GUI."""

    def __init__(self):
        super().__init__()

        Theme.setup()

        self.title("Pattern Analysis - Pre-Entry Signal Density")
        self.geometry("1200x780")
        self.minsize(1000, 700)
        self.configure(fg_color=Colors.BG_DARK)

        self._trade_log_path: Optional[Path] = None
        self._data_path: Path = DEFAULT_DATA_PATH
        self._result: Optional[PatternAnalysisResult] = None
        self._worker: Optional[threading.Thread] = None
        self._msg_queue: "queue.Queue[tuple]" = queue.Queue()

        self._build_layout()
        self._poll_messages()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header_card = Theme.create_card(self)
        header_card.grid(
            row=0, column=0, columnspan=2,
            sticky="ew", padx=Sizes.PAD_M, pady=Sizes.PAD_M,
        )
        header_inner = Theme.create_frame(header_card)
        header_inner.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)
        Theme.create_header(header_inner, "Pattern Analysis", size="l").pack(side="left")
        Theme.create_label(
            header_inner,
            "MA-crossover signal density before each trade entry.",
            text_color=Colors.TEXT_SECONDARY,
        ).pack(side="left", padx=(Sizes.PAD_L, 0))

        # Left: configuration
        self._build_config_panel()
        # Right: results
        self._build_results_panel()

    def _build_config_panel(self) -> None:
        card = Theme.create_card(self)
        card.grid(row=1, column=0, sticky="nsew", padx=(Sizes.PAD_M, Sizes.PAD_S), pady=(0, Sizes.PAD_M))

        scroll = ctk.CTkScrollableFrame(
            card, fg_color="transparent", corner_radius=Sizes.RADIUS_M,
        )
        scroll.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(scroll, "Configuration", size="m").pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Trade log path
        Theme.create_label(scroll, "Trade log CSV").pack(anchor="w", pady=(Sizes.PAD_S, 2))
        log_row = Theme.create_frame(scroll)
        log_row.pack(fill="x")

        self.trade_log_var = ctk.StringVar(value="")
        log_entry = Theme.create_entry(log_row, placeholder="Path to trade log CSV...")
        log_entry.configure(textvariable=self.trade_log_var)
        log_entry.pack(side="left", fill="x", expand=True)
        Theme.create_button(
            log_row, "Browse...",
            command=self._browse_trade_log, style="secondary", width=90,
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Data path
        Theme.create_label(scroll, "Price data directory").pack(anchor="w", pady=(Sizes.PAD_M, 2))
        data_row = Theme.create_frame(scroll)
        data_row.pack(fill="x")
        self.data_path_var = ctk.StringVar(value=str(DEFAULT_DATA_PATH))
        data_entry = Theme.create_entry(data_row, placeholder="raw_data/daily")
        data_entry.configure(textvariable=self.data_path_var)
        data_entry.pack(side="left", fill="x", expand=True)
        Theme.create_button(
            data_row, "Browse...",
            command=self._browse_data_path, style="secondary", width=90,
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Lookback windows
        Theme.create_label(scroll, "Lookback windows (days, comma-separated)").pack(
            anchor="w", pady=(Sizes.PAD_M, 2)
        )
        self.windows_var = ctk.StringVar(value=", ".join(str(w) for w in default_windows()))
        windows_entry = Theme.create_entry(scroll, placeholder="30, 60, 90, 120")
        windows_entry.configure(textvariable=self.windows_var)
        windows_entry.pack(fill="x")

        # MA combos
        Theme.create_label(scroll, "MA combos to analyze").pack(anchor="w", pady=(Sizes.PAD_M, 2))
        self.combo_panel = ComboListPanel(scroll)
        self.combo_panel.pack(fill="both", expand=True)
        # Pre-populate with defaults so first-time users see something useful.
        self.combo_panel.load_defaults()

        # Output path
        Theme.create_label(scroll, "Output Excel file").pack(anchor="w", pady=(Sizes.PAD_M, 2))
        out_row = Theme.create_frame(scroll)
        out_row.pack(fill="x")
        self.output_var = ctk.StringVar(value=str(DEFAULT_OUTPUT_PATH))
        out_entry = Theme.create_entry(out_row, placeholder="pattern_analysis_report.xlsx")
        out_entry.configure(textvariable=self.output_var)
        out_entry.pack(side="left", fill="x", expand=True)
        Theme.create_button(
            out_row, "Save As...",
            command=self._browse_output, style="secondary", width=90,
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Run button
        run_frame = Theme.create_frame(scroll)
        run_frame.pack(fill="x", pady=(Sizes.PAD_L, 0))
        self.run_button = Theme.create_button(
            run_frame, "Run Analysis",
            command=self._run_clicked, style="success", width=180,
        )
        self.run_button.pack(side="left")
        self.status_label = Theme.create_label(
            run_frame, "Idle.", text_color=Colors.TEXT_SECONDARY,
        )
        self.status_label.pack(side="left", padx=(Sizes.PAD_M, 0))

        # Progress bar
        self.progress = Theme.create_progressbar(scroll)
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(Sizes.PAD_S, 0))

        # Log textbox
        Theme.create_label(scroll, "Log").pack(anchor="w", pady=(Sizes.PAD_M, 2))
        self.log_box = Theme.create_textbox(scroll, height=140)
        self.log_box.pack(fill="x")
        self.log_box.configure(state="disabled")

    def _build_results_panel(self) -> None:
        card = Theme.create_card(self)
        card.grid(row=1, column=1, sticky="nsew", padx=(Sizes.PAD_S, Sizes.PAD_M), pady=(0, Sizes.PAD_M))

        inner = Theme.create_frame(card)
        inner.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_header(inner, "Results", size="m").pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.results_box = Theme.create_textbox(inner)
        self.results_box.pack(fill="both", expand=True)
        self._set_results_text(
            "Run the analysis to see a summary here.\n\n"
            "What this tool does:\n"
            "  - For each trade in the log, looks 30-120 days before entry.\n"
            "  - Detects buy / sell signals when an MA(L) shifted by `offset`\n"
            "    bars crosses the unshifted MA(L) (matches AlphaTrend logic).\n"
            "  - Writes a multi-sheet Excel report with per-trade features,\n"
            "    win/loss splits, and feature-vs-PL correlations.\n"
        )

        self.open_button = Theme.create_button(
            inner, "Open Report",
            command=self._open_report,
            style="primary",
            width=160,
        )
        self.open_button.pack(side="right", pady=(Sizes.PAD_S, 0))
        self.open_button.configure(state="disabled")

    # ------------------------------------------------------------------
    # Browse handlers
    # ------------------------------------------------------------------
    def _browse_trade_log(self) -> None:
        path = filedialog.askopenfilename(
            title="Select trade log CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.trade_log_var.set(path)

    def _browse_data_path(self) -> None:
        path = filedialog.askdirectory(title="Select price data directory")
        if path:
            self.data_path_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save report as",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile="pattern_analysis_report.xlsx",
        )
        if path:
            self.output_var.set(path)

    # ------------------------------------------------------------------
    # Run / worker
    # ------------------------------------------------------------------
    def _run_clicked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            show_info(self, "Busy", "An analysis is already running.")
            return

        try:
            trade_log_path = Path(self.trade_log_var.get().strip())
            if not trade_log_path.exists():
                raise ValueError(f"Trade log not found: {trade_log_path}")

            data_path = Path(self.data_path_var.get().strip())
            if not data_path.exists():
                raise ValueError(f"Price data path not found: {data_path}")

            windows = _parse_windows(self.windows_var.get())

            combos = self.combo_panel.get_combos()
            if not combos:
                raise ValueError("Add at least one MA combo before running.")

            output_path = Path(self.output_var.get().strip() or DEFAULT_OUTPUT_PATH)
        except Exception as exc:
            show_error(self, "Invalid Configuration", str(exc))
            return

        self._trade_log_path = trade_log_path
        self._data_path = data_path

        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.progress.set(0)
        self._set_log_text("")
        self._append_log(f"Loading trade log {trade_log_path}...")

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(trade_log_path, data_path, combos, windows, output_path),
            daemon=True,
        )
        self._worker.start()
        self.status_label.configure(text="Running...", text_color=Colors.WARNING)

    def _run_worker(
        self,
        trade_log_path: Path,
        data_path: Path,
        combos: List[MAComboSpec],
        windows: List[int],
        output_path: Path,
    ) -> None:
        try:
            trades = load_trade_log(trade_log_path)
            self._msg_queue.put(("log", f"Loaded {len(trades)} trade(s)."))
            self._msg_queue.put(("log",
                f"Analyzing across {len(combos)} combo(s) and windows {windows}..."))

            analyzer = PatternAnalyzer(
                data_path=data_path, combos=combos, windows=windows,
            )

            def progress(done: int, total: int, message: str) -> None:
                ratio = done / total if total else 1.0
                self._msg_queue.put(("progress", ratio))
                if done == 1 or done == total or done % 5 == 0:
                    self._msg_queue.put(("log", f"  [{done}/{total}] {message}"))

            result = analyzer.analyze(trades, progress=progress)
            write_report(result, output_path, trade_log_path=trade_log_path)
            self._msg_queue.put(("done", (result, output_path)))
        except Exception as exc:
            tb = traceback.format_exc()
            self._msg_queue.put(("error", (str(exc), tb)))

    # ------------------------------------------------------------------
    # Message pump (main thread)
    # ------------------------------------------------------------------
    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    self.progress.set(payload)
                elif kind == "done":
                    result, output_path = payload
                    self._on_done(result, output_path)
                elif kind == "error":
                    msg, tb = payload
                    self._on_error(msg, tb)
        except queue.Empty:
            pass
        self.after(120, self._poll_messages)

    def _on_done(self, result: PatternAnalysisResult, output_path: Path) -> None:
        self._result = result
        self.progress.set(1.0)
        self.status_label.configure(text="Done.", text_color=Colors.SUCCESS)
        self.run_button.configure(state="normal")
        self.open_button.configure(state="normal")
        self._render_results(result, output_path)
        self._append_log(f"Report written: {output_path}")

    def _on_error(self, message: str, tb: str) -> None:
        self.progress.set(0)
        self.status_label.configure(text="Failed.", text_color=Colors.ERROR)
        self.run_button.configure(state="normal")
        self._append_log(f"ERROR: {message}")
        self._append_log(tb)
        show_error(self, "Analysis Failed", message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_results(self, result: PatternAnalysisResult, output_path: Path) -> None:
        lines: List[str] = []
        lines.append(f"Report: {output_path}")
        lines.append("")

        features = result.features
        if features.empty:
            lines.append("No features computed (all trades skipped?).")
        else:
            unique_trades = features[["trade_id", "symbol", "entry_date"]].drop_duplicates()
            lines.append(f"Trades analyzed: {len(unique_trades)}")
            outcomes = features.drop_duplicates(["trade_id", "symbol", "entry_date"])["is_winner"]
            lines.append(
                f"  Winners: {(outcomes == True).sum()}    "
                f"Losers: {(outcomes == False).sum()}    "
                f"Unknown: {outcomes.isna().sum()}"
            )
            lines.append(f"Combos: {', '.join(c.label for c in result.combos)}")
            lines.append(f"Windows (days): {result.windows}")
            lines.append("")

            grouped = features.groupby(["combo_label", "window_days"])
            lines.append("Avg buy / sell counts per (combo, window):")
            for (combo_label, window), group in grouped:
                avg_buys = group["buy_count"].mean()
                avg_sells = group["sell_count"].mean()
                lines.append(
                    f"  {combo_label:<14} {int(window)}d   "
                    f"buys={avg_buys:5.2f}   sells={avg_sells:5.2f}"
                )

        if not result.skipped_trades.empty:
            lines.append("")
            lines.append(f"Skipped: {len(result.skipped_trades)} trade(s) "
                         f"(see 'Skipped Trades' sheet).")

        self._set_results_text("\n".join(lines))

    def _open_report(self) -> None:
        path = self.output_var.get().strip()
        if not path or not Path(path).exists():
            show_error(self, "No Report", "Run the analysis first.")
            return
        try:
            import os
            import platform
            if platform.system() == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                import subprocess
                subprocess.Popen(["open", path])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            show_error(self, "Could not open", str(exc))

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_log_text(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        if text:
            self.log_box.insert("end", text + "\n")
        self.log_box.configure(state="disabled")

    def _set_results_text(self, text: str) -> None:
        self.results_box.configure(state="normal")
        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", text)
        self.results_box.configure(state="disabled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_windows(text: str) -> List[int]:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return default_windows()
    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise ValueError(f"Invalid window value: {p!r}") from exc
        if v <= 0:
            raise ValueError(f"Window days must be > 0, got {v}.")
        out.append(v)
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pattern Analysis GUI - pre-entry MA crossover signal density",
    )
    parser.parse_args()

    try:
        app = CTkPatternAnalysisGUI()
        app.mainloop()
    except Exception as exc:
        print(f"Error launching GUI: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
