"""
BackTesting Framework - Main Launcher GUI

A modern CustomTkinter main menu that provides access to all framework features.

Features:
- Backtesting: Run backtests with various strategies
- Optimization: Walk-forward strategy optimization
- Edge Analysis: E-ratio and R-multiple analysis from trade logs
- Factor Analysis: Strategy performance factor analysis
- Vulnerability Modeler: Vulnerability score parameter optimization
"""

import customtkinter as ctk
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes


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

        # Configure grid columns to be equal width
        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)
        cards_frame.grid_columnconfigure(2, weight=1)
        cards_frame.grid_rowconfigure(0, weight=1)
        cards_frame.grid_rowconfigure(1, weight=1)

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
                "description": "Walk-forward optimization to find optimal strategy parameters",
                "icon": "\u2699",  # Gear
                "command": self._launch_optimization
            },
            {
                "title": "Edge Analysis",
                "description": "Analyze entry edge with E-ratio and R-multiple distributions",
                "icon": "\u2197",  # Arrow pointing up-right
                "command": self._launch_edge_analysis
            },
            {
                "title": "Factor Analysis",
                "description": "Multi-factor analysis of strategy performance drivers",
                "icon": "\u2630",  # Trigram for heaven (bars)
                "command": self._launch_factor_analysis
            },
            {
                "title": "Vulnerability Modeler",
                "description": "Model and optimize vulnerability score parameters",
                "icon": "\u26A0",  # Warning sign
                "command": self._launch_vulnerability
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
            text="v1.0.0",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        version_label.pack(side="right")

    def _launch_module(self, module_name: str):
        """Launch a module as a subprocess."""
        script_path = Path(__file__).parent / f"ctk_{module_name}_gui.py"
        if script_path.exists():
            subprocess.Popen([sys.executable, str(script_path)])
        else:
            self._show_error(f"Module not found: {script_path}")

    def _launch_backtest(self):
        """Launch the backtesting GUI."""
        self._launch_module("backtest")

    def _launch_optimization(self):
        """Launch the optimization GUI."""
        self._launch_module("optimization")

    def _launch_edge_analysis(self):
        """Launch the edge analysis GUI."""
        self._launch_module("edge_analysis")

    def _launch_factor_analysis(self):
        """Launch the factor analysis GUI."""
        self._launch_module("factor_analysis")

    def _launch_vulnerability(self):
        """Launch the vulnerability modeler GUI."""
        self._launch_module("vulnerability")

    def _show_error(self, message: str):
        """Show an error dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Error")
        dialog.geometry("400x150")
        dialog.transient(self)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 200
        y = self.winfo_y() + (self.winfo_height() // 2) - 75
        dialog.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dialog,
            text=message,
            font=Fonts.BODY_M,
            text_color=Colors.ERROR
        ).pack(expand=True, pady=20)

        ctk.CTkButton(
            dialog,
            text="OK",
            command=dialog.destroy,
            width=100
        ).pack(pady=(0, 20))


def main():
    """Main entry point for the launcher."""
    app = CTkMainLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
