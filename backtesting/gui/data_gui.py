"""
Data Manager GUI Window.

Provides interface for loading and managing market data:
- Yahoo Finance download
- CSV file import
- Sample data generation
- Data preview and validation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict
import pandas as pd

from backtesting.data import DataLoader


class DataManagerWindow:
    """Data management window."""

    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Data Manager")
        self.window.geometry("800x600")
        self.window.minsize(600, 400)

        self.data: Optional[pd.DataFrame] = None
        self.data_dict: Optional[Dict[str, pd.DataFrame]] = None

        self._create_widgets()
        self._center_window()

    def _center_window(self):
        """Center the window on screen."""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Notebook for different data sources
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Yahoo Finance tab
        yahoo_frame = ttk.Frame(notebook, padding="10")
        notebook.add(yahoo_frame, text="Yahoo Finance")
        self._create_yahoo_tab(yahoo_frame)

        # CSV tab
        csv_frame = ttk.Frame(notebook, padding="10")
        notebook.add(csv_frame, text="CSV File")
        self._create_csv_tab(csv_frame)

        # Sample Data tab
        sample_frame = ttk.Frame(notebook, padding="10")
        notebook.add(sample_frame, text="Sample Data")
        self._create_sample_tab(sample_frame)

        # Data preview section
        preview_frame = ttk.LabelFrame(self.window, text="Data Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Preview table
        columns = ("date", "open", "high", "low", "close", "volume")
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=10)

        for col in columns:
            self.preview_tree.heading(col, text=col.capitalize())
            self.preview_tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=scrollbar.set)

        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        self.status_var = tk.StringVar(value="No data loaded")
        status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _create_yahoo_tab(self, parent):
        """Create Yahoo Finance tab."""
        ttk.Label(parent, text="Download data from Yahoo Finance", font=("Helvetica", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Enter stock symbol(s) to download historical data.", foreground="gray").pack(anchor="w", pady=(0, 10))

        form_frame = ttk.Frame(parent)
        form_frame.pack(fill=tk.X)

        # Symbol
        ttk.Label(form_frame, text="Symbol(s):").grid(row=0, column=0, sticky="w", pady=5)
        self.yahoo_symbol = tk.StringVar(value="AAPL")
        ttk.Entry(form_frame, textvariable=self.yahoo_symbol, width=30).grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(form_frame, text="(comma-separated for multiple)", foreground="gray").grid(row=0, column=2)

        # Start date
        ttk.Label(form_frame, text="Start Date:").grid(row=1, column=0, sticky="w", pady=5)
        self.yahoo_start = tk.StringVar(value="2022-01-01")
        ttk.Entry(form_frame, textvariable=self.yahoo_start, width=15).grid(row=1, column=1, sticky="w", padx=10, pady=5)

        # End date
        ttk.Label(form_frame, text="End Date:").grid(row=2, column=0, sticky="w", pady=5)
        self.yahoo_end = tk.StringVar(value="2023-12-31")
        ttk.Entry(form_frame, textvariable=self.yahoo_end, width=15).grid(row=2, column=1, sticky="w", padx=10, pady=5)

        # Interval
        ttk.Label(form_frame, text="Interval:").grid(row=3, column=0, sticky="w", pady=5)
        self.yahoo_interval = tk.StringVar(value="1d")
        interval_combo = ttk.Combobox(
            form_frame,
            textvariable=self.yahoo_interval,
            values=["1d", "1wk", "1mo"],
            state="readonly",
            width=10,
        )
        interval_combo.grid(row=3, column=1, sticky="w", padx=10, pady=5)

        # Download button
        ttk.Button(parent, text="Download", command=self._download_yahoo).pack(pady=20)

    def _create_csv_tab(self, parent):
        """Create CSV tab."""
        ttk.Label(parent, text="Import data from CSV file", font=("Helvetica", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, text="CSV must have columns: date, open, high, low, close (volume optional)", foreground="gray").pack(anchor="w", pady=(0, 10))

        form_frame = ttk.Frame(parent)
        form_frame.pack(fill=tk.X)

        # File path
        ttk.Label(form_frame, text="File:").grid(row=0, column=0, sticky="w", pady=5)
        self.csv_path = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.csv_path, width=50).grid(row=0, column=1, padx=10, pady=5)
        ttk.Button(form_frame, text="Browse...", command=self._browse_csv).grid(row=0, column=2, pady=5)

        # Date column
        ttk.Label(form_frame, text="Date Column:").grid(row=1, column=0, sticky="w", pady=5)
        self.csv_date_col = tk.StringVar(value="date")
        ttk.Entry(form_frame, textvariable=self.csv_date_col, width=20).grid(row=1, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(form_frame, text="(leave blank for auto-detect)", foreground="gray").grid(row=1, column=2)

        # Load button
        ttk.Button(parent, text="Load CSV", command=self._load_csv).pack(pady=20)

    def _create_sample_tab(self, parent):
        """Create sample data tab."""
        ttk.Label(parent, text="Generate synthetic data", font=("Helvetica", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Generate random price data for testing strategies.", foreground="gray").pack(anchor="w", pady=(0, 10))

        form_frame = ttk.Frame(parent)
        form_frame.pack(fill=tk.X)

        # Days
        ttk.Label(form_frame, text="Number of Days:").grid(row=0, column=0, sticky="w", pady=5)
        self.sample_days = tk.StringVar(value="504")
        ttk.Entry(form_frame, textvariable=self.sample_days, width=10).grid(row=0, column=1, sticky="w", padx=10, pady=5)

        # Start price
        ttk.Label(form_frame, text="Start Price:").grid(row=1, column=0, sticky="w", pady=5)
        self.sample_price = tk.StringVar(value="100")
        ttk.Entry(form_frame, textvariable=self.sample_price, width=10).grid(row=1, column=1, sticky="w", padx=10, pady=5)

        # Volatility
        ttk.Label(form_frame, text="Daily Volatility:").grid(row=2, column=0, sticky="w", pady=5)
        self.sample_vol = tk.StringVar(value="0.015")
        ttk.Entry(form_frame, textvariable=self.sample_vol, width=10).grid(row=2, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(form_frame, text="(0.01 = 1% std dev)", foreground="gray").grid(row=2, column=2)

        # Drift
        ttk.Label(form_frame, text="Daily Drift:").grid(row=3, column=0, sticky="w", pady=5)
        self.sample_drift = tk.StringVar(value="0.0003")
        ttk.Entry(form_frame, textvariable=self.sample_drift, width=10).grid(row=3, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(form_frame, text="(0.0003 = ~8% annual return)", foreground="gray").grid(row=3, column=2)

        # Seed
        ttk.Label(form_frame, text="Random Seed:").grid(row=4, column=0, sticky="w", pady=5)
        self.sample_seed = tk.StringVar(value="42")
        ttk.Entry(form_frame, textvariable=self.sample_seed, width=10).grid(row=4, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(form_frame, text="(for reproducibility)", foreground="gray").grid(row=4, column=2)

        # Generate button
        ttk.Button(parent, text="Generate", command=self._generate_sample).pack(pady=20)

    def _browse_csv(self):
        """Browse for CSV file."""
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path.set(path)

    def _download_yahoo(self):
        """Download data from Yahoo Finance."""
        try:
            loader = DataLoader()
            symbols = [s.strip() for s in self.yahoo_symbol.get().split(",")]

            if len(symbols) == 1:
                self.data = loader.from_yahoo(
                    symbols[0],
                    start=self.yahoo_start.get(),
                    end=self.yahoo_end.get(),
                    interval=self.yahoo_interval.get(),
                )
                self.data_dict = None
            else:
                self.data_dict = loader.from_yahoo(
                    symbols,
                    start=self.yahoo_start.get(),
                    end=self.yahoo_end.get(),
                    interval=self.yahoo_interval.get(),
                )
                self.data = None

            self._update_preview()
            messagebox.showinfo("Success", "Data downloaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to download data:\n{str(e)}")

    def _load_csv(self):
        """Load data from CSV."""
        try:
            loader = DataLoader()
            date_col = self.csv_date_col.get() or None
            self.data = loader.from_csv(self.csv_path.get(), date_column=date_col)
            self.data_dict = None

            self._update_preview()
            messagebox.showinfo("Success", "CSV loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")

    def _generate_sample(self):
        """Generate sample data."""
        try:
            loader = DataLoader()
            seed = int(self.sample_seed.get()) if self.sample_seed.get() else None

            self.data = loader.generate_sample_data(
                days=int(self.sample_days.get()),
                start_price=float(self.sample_price.get()),
                volatility=float(self.sample_vol.get()),
                drift=float(self.sample_drift.get()),
                seed=seed,
            )
            self.data_dict = None

            self._update_preview()
            messagebox.showinfo("Success", "Sample data generated!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data:\n{str(e)}")

    def _update_preview(self):
        """Update the data preview table."""
        # Clear existing
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)

        if self.data is not None:
            df = self.data
            symbol = "default"
        elif self.data_dict is not None:
            symbol = list(self.data_dict.keys())[0]
            df = self.data_dict[symbol]
        else:
            return

        # Show last 50 rows
        for idx, row in df.tail(50).iterrows():
            values = (
                idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                f"{row.get('open', 0):.2f}",
                f"{row.get('high', 0):.2f}",
                f"{row.get('low', 0):.2f}",
                f"{row.get('close', 0):.2f}",
                f"{int(row.get('volume', 0)):,}" if 'volume' in row else "N/A",
            )
            self.preview_tree.insert("", 0, values=values)

        # Update status
        bars = len(df)
        start = df.index[0].strftime("%Y-%m-%d") if hasattr(df.index[0], "strftime") else str(df.index[0])
        end = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])

        if self.data_dict:
            self.status_var.set(f"Loaded {len(self.data_dict)} symbols | {bars} bars | {start} to {end}")
        else:
            self.status_var.set(f"Loaded {bars} bars | {start} to {end}")
