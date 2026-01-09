"""
Main GUI Window for Backtesting Framework.

This is the main launcher that provides access to all features:
- Parameter Optimization
- Single Backtest
- Strategy Builder (future)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path


class MainWindow:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Backtesting Framework")
        self.root.geometry("600x500")
        self.root.minsize(500, 400)

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure colors
        self.root.configure(bg="#f0f0f0")

        self._create_widgets()
        self._center_window()

    def _center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Backtesting Framework",
            font=("Helvetica", 24, "bold"),
        )
        title_label.pack(pady=(0, 10))

        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Parameter Optimization & Strategy Testing",
            font=("Helvetica", 12),
            foreground="gray",
        )
        subtitle_label.pack(pady=(0, 30))

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid
        buttons_frame.columnconfigure(0, weight=1)

        # Optimization button
        self._create_feature_button(
            buttons_frame,
            title="Parameter Optimization",
            description="Run univariate parameter optimization.\nTest one parameter at a time while others\nstay at control values. Export results to Excel.",
            command=self._open_optimization,
            row=0,
        )

        # Single Backtest button
        self._create_feature_button(
            buttons_frame,
            title="Single Backtest",
            description="Run a single backtest with specific\nparameter values. View detailed results\nand performance metrics.",
            command=self._open_single_backtest,
            row=1,
        )

        # Data Loader button
        self._create_feature_button(
            buttons_frame,
            title="Data Manager",
            description="Load and manage market data from\nYahoo Finance, CSV files, or generate\nsynthetic data for testing.",
            command=self._open_data_manager,
            row=2,
        )

        # Footer
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(20, 0))

        version_label = ttk.Label(
            footer_frame,
            text="Version 1.0.0",
            font=("Helvetica", 9),
            foreground="gray",
        )
        version_label.pack(side=tk.LEFT)

        quit_btn = ttk.Button(
            footer_frame,
            text="Quit",
            command=self.root.quit,
        )
        quit_btn.pack(side=tk.RIGHT)

    def _create_feature_button(
        self,
        parent,
        title: str,
        description: str,
        command,
        row: int,
    ):
        """Create a feature button with description."""
        frame = ttk.Frame(parent, padding="10")
        frame.grid(row=row, column=0, sticky="ew", pady=5)

        # Make frame look like a button/card
        inner_frame = ttk.Frame(frame, relief="raised", borderwidth=1, padding="15")
        inner_frame.pack(fill=tk.X)

        # Title
        title_label = ttk.Label(
            inner_frame,
            text=title,
            font=("Helvetica", 14, "bold"),
            cursor="hand2",
        )
        title_label.pack(anchor="w")

        # Description
        desc_label = ttk.Label(
            inner_frame,
            text=description,
            font=("Helvetica", 10),
            foreground="gray",
            justify=tk.LEFT,
        )
        desc_label.pack(anchor="w", pady=(5, 10))

        # Launch button
        btn = ttk.Button(
            inner_frame,
            text=f"Open {title}",
            command=command,
        )
        btn.pack(anchor="e")

        # Make entire frame clickable
        for widget in [inner_frame, title_label, desc_label]:
            widget.bind("<Button-1>", lambda e, cmd=command: cmd())
            widget.bind("<Enter>", lambda e, f=inner_frame: f.configure(relief="groove"))
            widget.bind("<Leave>", lambda e, f=inner_frame: f.configure(relief="raised"))

    def _open_optimization(self):
        """Open the optimization window."""
        from backtesting.gui.optimization_gui import OptimizationWindow
        OptimizationWindow(self.root)

    def _open_single_backtest(self):
        """Open single backtest window."""
        from backtesting.gui.backtest_gui import BacktestWindow
        BacktestWindow(self.root)

    def _open_data_manager(self):
        """Open data manager window."""
        from backtesting.gui.data_gui import DataManagerWindow
        DataManagerWindow(self.root)

    def run(self):
        """Start the main event loop."""
        self.root.mainloop()


def main():
    """Entry point for the GUI application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
