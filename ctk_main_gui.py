"""
BackTesting Framework - Main Launcher GUI

A modern CustomTkinter main menu that provides access to all framework
modules: backtesting, walk-forward and univariate optimization, edge
analysis, data collection & preparation, rule testing, vulnerability
visualization, pattern analysis, Monte Carlo simulation, ML modelling &
evaluation, and the Streamlit results dashboard.
"""

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional

import customtkinter as ctk

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes
from Classes._version import __version__


def _missing_modules(module_names: List[str]) -> List[str]:
    """Return the subset of ``module_names`` that cannot be imported."""
    return [m for m in module_names if importlib.util.find_spec(m) is None]


class FeatureCard(ctk.CTkFrame):
    """A clickable card representing a framework feature."""

    def __init__(
        self,
        parent,
        title: str,
        description: str,
        icon: str,
        command: Callable,
        **kwargs
    ):
        super().__init__(
            parent,
            fg_color=Colors.SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=Colors.BORDER,
            **kwargs
        )

        self.command = command
        self._is_hovered = False

        # Make the whole card clickable
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

        # Icon (emoji or unicode symbol)
        self.icon_label = ctk.CTkLabel(
            self,
            text=icon,
            font=("", 48),
            text_color=Colors.PRIMARY_LIGHT
        )
        self.icon_label.pack(pady=(24, 12))
        self.icon_label.bind("<Button-1>", self._on_click)
        self.icon_label.bind("<Enter>", self._on_enter)
        self.icon_label.bind("<Leave>", self._on_leave)

        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=Fonts.HEADER_M,
            text_color=Colors.TEXT_PRIMARY
        )
        self.title_label.pack(pady=(0, 8))
        self.title_label.bind("<Button-1>", self._on_click)
        self.title_label.bind("<Enter>", self._on_enter)
        self.title_label.bind("<Leave>", self._on_leave)

        # Description
        self.desc_label = ctk.CTkLabel(
            self,
            text=description,
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY,
            wraplength=200,
            justify="center"
        )
        self.desc_label.pack(pady=(0, 24), padx=16)
        self.desc_label.bind("<Button-1>", self._on_click)
        self.desc_label.bind("<Enter>", self._on_enter)
        self.desc_label.bind("<Leave>", self._on_leave)

    def _on_click(self, event=None):
        """Handle card click."""
        if self.command:
            self.command()

    def _on_enter(self, event=None):
        """Handle mouse enter."""
        self._is_hovered = True
        self.configure(
            fg_color=Colors.SURFACE_HOVER,
            border_color=Colors.PRIMARY_LIGHT
        )

    def _on_leave(self, event=None):
        """Handle mouse leave."""
        self._is_hovered = False
        self.configure(
            fg_color=Colors.SURFACE,
            border_color=Colors.BORDER
        )


class CTkMainLauncher(ctk.CTk):
    """Main launcher GUI for the BackTesting Framework."""

    def __init__(self):
        super().__init__()

        # Apply theme
        Theme.setup()

        # Window configuration
        self.title("BackTesting Framework")
        self.geometry("900x700")
        self.minsize(800, 600)
        self.configure(fg_color=Colors.BG_DARK)

        # Center window on screen
        self.update_idletasks()
        width = 900
        height = 700
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        self._create_ui()

    def _create_ui(self):
        """Create the main UI layout."""
        # Main container with padding
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=40, pady=30)

        # Header section
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 30))

        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="BackTesting Framework",
            font=("", 32, "bold"),
            text_color=Colors.TEXT_PRIMARY
        )
        title_label.pack()

        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Select a module to get started",
            font=Fonts.BODY_L,
            text_color=Colors.TEXT_SECONDARY
        )
        subtitle_label.pack(pady=(8, 0))

        # Cards container - use grid for 3 columns
        cards_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        cards_frame.pack(fill="both", expand=True)

        # Configure grid columns to be equal width (3 columns, 4 rows of feature cards)
        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)
        cards_frame.grid_columnconfigure(2, weight=1)
        cards_frame.grid_rowconfigure(0, weight=1)
        cards_frame.grid_rowconfigure(1, weight=1)
        cards_frame.grid_rowconfigure(2, weight=1)
        cards_frame.grid_rowconfigure(3, weight=1)

        # Define features
        features = [
            {
                "title": "Backtesting",
                "description": "Run backtests on securities with configurable strategies and parameters",
                "icon": "\u23F1",  # Stopwatch
                "command": self._launch_backtest
            },
            {
                "title": "Optimization",
                "description": "Walk-forward parameter optimization with Bayesian search and sensitivity analysis",
                "icon": "\U0001F3AF",  # Direct hit
                "command": self._launch_optimization
            },
            {
                "title": "Univariate Optimization",
                "description": "Test one parameter at a time to analyze strategy sensitivity",
                "icon": "\u2699",  # Gear
                "command": self._launch_univariate_optimization
            },
            {
                "title": "Edge Analysis",
                "description": "Analyze entry edge with E-ratio and R-multiple distributions",
                "icon": "\u2197",  # Arrow pointing up-right
                "command": self._launch_edge_analysis
            },
            {
                "title": "Data Collection",
                "description": "Collect and prepare raw data from Alpha Vantage API",
                "icon": "\U0001F4E5",  # Inbox tray
                "command": self._launch_data_collection
            },
            {
                "title": "Data Preparation",
                "description": "Build a point-in-time research run package (trades + factors) for the modelling stage",
                "icon": "\U0001F9EA",  # Test tube
                "command": self._launch_data_prep
            },
            {
                "title": "Rule Tester",
                "description": "Test how entry/exit rules affect strategy performance with lookback support",
                "icon": "\U0001F50D",  # Magnifying glass
                "command": self._launch_rule_tester
            },
            {
                "title": "Vulnerability Visualizer",
                "description": "Explore how the vulnerability score evolves per trade and tune the parameters live",
                "icon": "\U0001F4C9",  # Chart with downwards trend
                "command": self._launch_vulnerability_visualizer
            },
            {
                "title": "Pattern Analysis",
                "description": "Analyze MA-crossover signal density before each trade entry to find predictive patterns",
                "icon": "\U0001F50E",  # Magnifying glass tilted right
                "command": self._launch_pattern_analysis
            },
            {
                "title": "Monte Carlo",
                "description": "Bootstrap trade-log returns to simulate equity curves, drawdowns and probability of loss",
                "icon": "\U0001F3B2",  # Game die
                "command": self._launch_monte_carlo
            },
            {
                "title": "Modelling & Evaluation",
                "description": "Train & evaluate models on a prepared run package to diagnose when the strategy works",
                "icon": "\U0001F916",  # Robot
                "command": self._launch_modelling_evaluation
            },
            {
                "title": "Results Dashboard",
                "description": (
                    "Interactive browser dashboard to explore modelling results: regimes, factors, overlays"
                    + (
                        "\n⚠ requires: " + ", ".join(_missing_modules(["streamlit", "plotly"]))
                        if _missing_modules(["streamlit", "plotly"]) else ""
                    )
                ),
                "icon": "\U0001F4CA",  # Bar chart
                "command": self._launch_modelling_dashboard
            }
        ]

        # Create feature cards in grid
        for i, feature in enumerate(features):
            row = i // 3
            col = i % 3

            card = FeatureCard(
                cards_frame,
                title=feature["title"],
                description=feature["description"],
                icon=feature["icon"],
                command=feature["command"],
                width=240,
                height=220
            )
            card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

        # Footer with version/info
        footer_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        footer_frame.pack(fill="x", pady=(20, 0))

        version_label = ctk.CTkLabel(
            footer_frame,
            text=f"v{__version__}",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        version_label.pack(side="right")

    def _launch_module(self, module_name: str):
        """Launch a module as a subprocess."""
        script_path = Path(__file__).parent / f"ctk_{module_name}_gui.py"
        if script_path.exists():
            self._spawn_watched(module_name, [sys.executable, str(script_path)])
        else:
            self._show_error(f"Module not found: {script_path}")

    def _spawn_watched(self, name: str, cmd: List[str]):
        """
        Launch a child process and surface a crash dialog if it dies with a
        non-zero exit code. Without this, a module that fails on startup
        (missing dependency, bad config) disappears silently.
        """
        try:
            proc = subprocess.Popen(
                cmd, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace"
            )
        except OSError as exc:
            self._show_error(f"Could not launch {name}: {exc}")
            return

        def watch():
            _, stderr = proc.communicate()
            if proc.returncode not in (0, None) and proc.returncode >= 0:
                tail = "\n".join((stderr or "").strip().splitlines()[-12:])
                message = (f"{name} exited with code {proc.returncode}."
                           + (f"\n\n{tail}" if tail else ""))
                # Marshal back onto the Tk main thread.
                try:
                    self.after(0, lambda: self._show_error(message))
                except Exception:
                    pass  # launcher already closed

        threading.Thread(target=watch, daemon=True,
                         name=f"watch-{name}").start()

    def _launch_backtest(self):
        """Launch the backtesting GUI."""
        self._launch_module("backtest")

    def _launch_optimization(self):
        """Launch the optimization GUI."""
        self._launch_module("optimization")

    def _launch_univariate_optimization(self):
        """Launch the univariate optimization GUI."""
        self._launch_module("univariate_optimization")

    def _launch_edge_analysis(self):
        """Launch the edge analysis GUI."""
        self._launch_module("edge_analysis")

    def _launch_data_collection(self):
        """Launch the data collection GUI."""
        script_path = Path(__file__).parent / "apps" / "data_collection_gui.py"
        if script_path.exists():
            self._spawn_watched("Data Collection", [sys.executable, str(script_path)])
        else:
            self._show_error(f"Data Collection GUI not found: {script_path}")

    def _launch_data_prep(self):
        """Launch the Data Preparation (run-package builder) GUI."""
        self._launch_module("data_prep")

    def _launch_rule_tester(self):
        """Launch the Rule Tester GUI."""
        self._launch_module("rule_tester")

    def _launch_vulnerability_visualizer(self):
        """Launch the Vulnerability Score Visualizer GUI."""
        self._launch_module("vulnerability_visualizer")

    def _launch_pattern_analysis(self):
        """Launch the Pattern Analysis GUI."""
        self._launch_module("pattern_analysis")

    def _launch_monte_carlo(self):
        """Launch the Monte Carlo simulation GUI."""
        self._launch_module("monte_carlo")

    def _launch_modelling_evaluation(self):
        """Launch the Modelling & Evaluation GUI."""
        self._launch_module("modelling_evaluation")

    def _launch_modelling_dashboard(self):
        """Launch the interactive Streamlit results dashboard."""
        missing = _missing_modules(["streamlit", "plotly"])
        if missing:
            self._show_error(
                "The Results Dashboard needs optional packages that are not "
                f"installed: {', '.join(missing)}.\n\n"
                f"Install them with:  pip install {' '.join(missing)}"
            )
            return
        script_path = Path(__file__).parent / "apps" / "modelling_dashboard.py"
        if script_path.exists():
            self._spawn_watched(
                "Results Dashboard",
                [sys.executable, "-m", "streamlit", "run", str(script_path)]
            )
        else:
            self._show_error(f"Dashboard not found: {script_path}")

    def _show_error(self, message: str):
        """Show an error dialog (sized to fit multi-line crash output)."""
        long_message = len(message) > 120 or "\n" in message
        width, height = (620, 380) if long_message else (420, 170)

        dialog = ctk.CTkToplevel(self)
        dialog.title("Error")
        dialog.geometry(f"{width}x{height}")
        dialog.transient(self)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - width // 2
        y = self.winfo_y() + (self.winfo_height() // 2) - height // 2
        dialog.geometry(f"+{x}+{y}")

        if long_message:
            box = ctk.CTkTextbox(dialog, wrap="word",
                                 text_color=Colors.ERROR,
                                 font=Fonts.BODY_S)
            box.insert("1.0", message)
            box.configure(state="disabled")
            box.pack(expand=True, fill="both", padx=16, pady=(16, 8))
        else:
            ctk.CTkLabel(
                dialog,
                text=message,
                font=Fonts.BODY_M,
                text_color=Colors.ERROR,
                wraplength=width - 60,
                justify="left"
            ).pack(expand=True, pady=20, padx=20)

        ctk.CTkButton(
            dialog,
            text="OK",
            command=dialog.destroy,
            width=100
        ).pack(pady=(0, 16))


def main():
    """Main entry point for the launcher."""
    from Classes.Core.logging_config import setup_logging
    setup_logging(app_name="launcher")
    app = CTkMainLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
