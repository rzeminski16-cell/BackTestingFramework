"""
Results Window

A separate window for displaying backtest/optimization results.
Shows the settings used and the results, allowing users to verify their configuration.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, Optional, Callable
from datetime import datetime


class ResultsWindow:
    """
    A standalone window for displaying backtest or optimization results.

    Features:
    - Shows configuration summary at the top
    - Displays results in a scrollable text area
    - Progress bar for ongoing operations
    - Can be kept open while running multiple backtests
    """

    def __init__(self, parent: tk.Tk, title: str, settings: Dict[str, Dict[str, str]],
                 width: int = 900, height: int = 700):
        """
        Create a new results window.

        Args:
            parent: Parent window
            title: Window title
            settings: Dictionary of settings categories -> key-value pairs
            width: Window width
            height: Window height
        """
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry(f"{width}x{height}")

        # Maximize window (fullscreen)
        try:
            self.window.state('zoomed')  # Windows
        except tk.TclError:
            try:
                self.window.attributes('-zoomed', True)  # Linux
            except tk.TclError:
                pass  # Fallback to default size

        # Keep window on top initially
        self.window.transient(parent)

        self.settings = settings
        self._create_widgets()

    def _create_widgets(self):
        """Create the window widgets."""
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        # === Settings Summary (Collapsible) ===
        self.settings_frame = ttk.LabelFrame(self.window, text="Configuration Used", padding="10")
        self.settings_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.settings_frame.columnconfigure(0, weight=1)

        # Toggle button for expanding/collapsing settings
        self.settings_expanded = tk.BooleanVar(value=True)
        self.toggle_btn = ttk.Button(
            self.settings_frame,
            text="Hide Settings",
            command=self._toggle_settings,
            width=15
        )
        self.toggle_btn.grid(row=0, column=0, sticky="e", pady=(0, 5))

        # Settings content frame
        self.settings_content = ttk.Frame(self.settings_frame)
        self.settings_content.grid(row=1, column=0, sticky="ew")
        self.settings_content.columnconfigure(0, weight=1)
        self.settings_content.columnconfigure(1, weight=1)
        self.settings_content.columnconfigure(2, weight=1)

        # Populate settings in a multi-column layout
        self._populate_settings()

        # === Progress Section ===
        self.progress_frame = ttk.Frame(self.window)
        self.progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.progress_frame.columnconfigure(1, weight=1)

        ttk.Label(self.progress_frame, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.status_var = tk.StringVar(value="Running...")
        self.status_label = ttk.Label(self.progress_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=1, sticky="w")

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        self.progress_detail_var = tk.StringVar(value="")
        self.progress_detail_label = ttk.Label(
            self.progress_frame,
            textvariable=self.progress_detail_var,
            font=('TkDefaultFont', 8, 'italic')
        )
        self.progress_detail_label.grid(row=2, column=0, columnspan=2, sticky="w")

        # === Results Area ===
        results_label_frame = ttk.LabelFrame(self.window, text="Results", padding="5")
        results_label_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))
        results_label_frame.columnconfigure(0, weight=1)
        results_label_frame.rowconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            results_label_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            state=tk.NORMAL
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")

        # === Action Buttons ===
        button_frame = ttk.Frame(self.window)
        button_frame.grid(row=3, column=0, sticky="e", padx=10, pady=(0, 10))

        self.close_btn = ttk.Button(button_frame, text="Close", command=self.window.destroy)
        self.close_btn.pack(side=tk.RIGHT)

        self.copy_btn = ttk.Button(button_frame, text="Copy Results", command=self._copy_results)
        self.copy_btn.pack(side=tk.RIGHT, padx=(0, 5))

    def _populate_settings(self):
        """Populate the settings section with configuration details."""
        col = 0
        row = 0
        max_cols = 3

        for category, items in self.settings.items():
            if not items:
                continue

            # Category frame
            cat_frame = ttk.Frame(self.settings_content)
            cat_frame.grid(row=row, column=col, sticky="nw", padx=10, pady=5)

            # Category title
            ttk.Label(
                cat_frame,
                text=category,
                font=('TkDefaultFont', 9, 'bold'),
                foreground='#2E5994'
            ).pack(anchor=tk.W)

            # Items
            for key, value in items.items():
                item_frame = ttk.Frame(cat_frame)
                item_frame.pack(fill=tk.X, pady=1)

                ttk.Label(
                    item_frame,
                    text=f"{key}:",
                    font=('TkDefaultFont', 8),
                    width=20,
                    anchor=tk.W
                ).pack(side=tk.LEFT)

                # Truncate long values
                display_value = str(value)
                if len(display_value) > 40:
                    display_value = display_value[:37] + "..."

                ttk.Label(
                    item_frame,
                    text=display_value,
                    font=('TkDefaultFont', 8, 'italic'),
                    foreground='#555555'
                ).pack(side=tk.LEFT)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _toggle_settings(self):
        """Toggle the settings section visibility."""
        if self.settings_expanded.get():
            self.settings_content.grid_forget()
            self.toggle_btn.config(text="Show Settings")
            self.settings_expanded.set(False)
        else:
            self.settings_content.grid(row=1, column=0, sticky="ew")
            self.toggle_btn.config(text="Hide Settings")
            self.settings_expanded.set(True)

    def _copy_results(self):
        """Copy results text to clipboard."""
        self.window.clipboard_clear()
        self.window.clipboard_append(self.results_text.get(1.0, tk.END))

    def log(self, message: str):
        """Log a message to the results area."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.window.update_idletasks()

    def update_progress(self, current: int, total: int, detail: str = ""):
        """Update the progress bar."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_var.set(percentage)
            self.progress_detail_var.set(f"{current}/{total} ({percentage}%) {detail}")
        self.window.update_idletasks()

    def set_status(self, status: str, is_complete: bool = False):
        """Set the status label."""
        self.status_var.set(status)
        if is_complete:
            self.progress_bar.grid_forget()
            self.progress_detail_label.grid_forget()
        self.window.update_idletasks()

    def on_complete(self):
        """Called when the operation is complete."""
        self.set_status("Complete", is_complete=True)
        self.log("\n" + "=" * 60)
        self.log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)

    def on_error(self, error_message: str):
        """Called when an error occurs."""
        self.set_status("Error", is_complete=True)
        self.log(f"\nERROR: {error_message}")


class OptimizationResultsWindow(ResultsWindow):
    """
    Extended results window for optimization with cancel capability.
    """

    def __init__(self, parent: tk.Tk, title: str, settings: Dict[str, Dict[str, str]],
                 on_cancel: Optional[Callable] = None, width: int = 900, height: int = 750):
        self.on_cancel_callback = on_cancel
        super().__init__(parent, title, settings, width, height)

    def _create_widgets(self):
        """Create widgets with cancel button."""
        super()._create_widgets()

        # Add cancel button to progress frame
        self.cancel_btn = ttk.Button(
            self.progress_frame,
            text="Cancel",
            command=self._on_cancel
        )
        self.cancel_btn.grid(row=0, column=2, padx=(10, 0))

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.cancel_btn.config(state=tk.DISABLED)
        self.set_status("Cancelling...")

    def on_complete(self):
        """Called when optimization is complete."""
        super().on_complete()
        self.cancel_btn.config(state=tk.DISABLED)

    def disable_cancel(self):
        """Disable the cancel button."""
        self.cancel_btn.config(state=tk.DISABLED)
