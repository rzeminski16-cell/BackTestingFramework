"""
Optimization GUI Window.

Provides a complete interface for running parameter optimization:
1. Data loading (securities selection)
2. Strategy selection
3. Control values configuration
4. Parameters to optimize with ranges
5. Metrics selection
6. Run optimization and export results
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Any, Optional, Type
import threading
from datetime import datetime
import os

import pandas as pd

from backtesting.strategies.base import Strategy, Parameter, ParameterType
from backtesting.strategies.examples import SMACrossover, RSIStrategy, BollingerBandsStrategy
from backtesting.optimization import Optimizer, OptimizationConfig, ParameterRange
from backtesting.metrics.performance import AVAILABLE_METRICS
from backtesting.data import DataLoader


# Available strategies
AVAILABLE_STRATEGIES: Dict[str, Type[Strategy]] = {
    "SMA Crossover": SMACrossover,
    "RSI Strategy": RSIStrategy,
    "Bollinger Bands": BollingerBandsStrategy,
}


class OptimizationWindow:
    """Optimization configuration and execution window."""

    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Parameter Optimization")
        self.window.geometry("900x700")
        self.window.minsize(800, 600)

        # State
        self.data: Optional[pd.DataFrame] = None
        self.data_dict: Optional[Dict[str, pd.DataFrame]] = None
        self.selected_strategy: Optional[Type[Strategy]] = None
        self.control_entries: Dict[str, tk.StringVar] = {}
        self.optimize_vars: Dict[str, tk.BooleanVar] = {}
        self.range_entries: Dict[str, tk.StringVar] = {}
        self.metric_vars: Dict[str, tk.BooleanVar] = {}
        self.results = None

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
        # Main container with notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Data & Strategy
        self.tab_data = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_data, text="1. Data & Strategy")

        # Tab 2: Parameters
        self.tab_params = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_params, text="2. Parameters")

        # Tab 3: Metrics
        self.tab_metrics = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_metrics, text="3. Metrics")

        # Tab 4: Run & Results
        self.tab_run = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_run, text="4. Run & Results")

        self._create_data_tab()
        self._create_params_tab()
        self._create_metrics_tab()
        self._create_run_tab()

    def _create_data_tab(self):
        """Create the data and strategy selection tab."""
        # Data section
        data_frame = ttk.LabelFrame(self.tab_data, text="Data Source", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))

        # Data source options
        self.data_source = tk.StringVar(value="sample")

        ttk.Radiobutton(
            data_frame,
            text="Generate Sample Data",
            variable=self.data_source,
            value="sample",
            command=self._update_data_options,
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Radiobutton(
            data_frame,
            text="Yahoo Finance",
            variable=self.data_source,
            value="yahoo",
            command=self._update_data_options,
        ).grid(row=1, column=0, sticky="w", pady=2)

        ttk.Radiobutton(
            data_frame,
            text="CSV File",
            variable=self.data_source,
            value="csv",
            command=self._update_data_options,
        ).grid(row=2, column=0, sticky="w", pady=2)

        # Data options frame
        self.data_options_frame = ttk.Frame(data_frame)
        self.data_options_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(20, 0))
        data_frame.columnconfigure(1, weight=1)

        self._create_sample_options()

        # Load data button
        load_btn = ttk.Button(
            data_frame,
            text="Load Data",
            command=self._load_data,
        )
        load_btn.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # Data status
        self.data_status = ttk.Label(
            data_frame,
            text="No data loaded",
            foreground="gray",
        )
        self.data_status.grid(row=4, column=0, columnspan=2, pady=(5, 0))

        # Strategy section
        strategy_frame = ttk.LabelFrame(self.tab_data, text="Strategy Selection", padding="10")
        strategy_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(strategy_frame, text="Strategy:").grid(row=0, column=0, sticky="w", pady=5)

        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(
            strategy_frame,
            textvariable=self.strategy_var,
            values=list(AVAILABLE_STRATEGIES.keys()),
            state="readonly",
            width=30,
        )
        strategy_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))
        strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy_selected)

        # Strategy description
        self.strategy_desc = ttk.Label(
            strategy_frame,
            text="Select a strategy to see its description",
            foreground="gray",
            wraplength=400,
        )
        self.strategy_desc.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Next button
        next_btn = ttk.Button(
            self.tab_data,
            text="Next: Configure Parameters →",
            command=lambda: self.notebook.select(1),
        )
        next_btn.pack(side=tk.BOTTOM, pady=10)

    def _create_sample_options(self):
        """Create sample data options."""
        for widget in self.data_options_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.data_options_frame, text="Days:").grid(row=0, column=0, sticky="w")
        self.sample_days = tk.StringVar(value="504")
        ttk.Entry(self.data_options_frame, textvariable=self.sample_days, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(self.data_options_frame, text="Start Price:").grid(row=1, column=0, sticky="w")
        self.sample_price = tk.StringVar(value="100")
        ttk.Entry(self.data_options_frame, textvariable=self.sample_price, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(self.data_options_frame, text="Volatility:").grid(row=2, column=0, sticky="w")
        self.sample_vol = tk.StringVar(value="0.015")
        ttk.Entry(self.data_options_frame, textvariable=self.sample_vol, width=10).grid(row=2, column=1, padx=5)

    def _create_yahoo_options(self):
        """Create Yahoo Finance options."""
        for widget in self.data_options_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.data_options_frame, text="Symbol(s):").grid(row=0, column=0, sticky="w")
        self.yahoo_symbols = tk.StringVar(value="AAPL")
        ttk.Entry(self.data_options_frame, textvariable=self.yahoo_symbols, width=20).grid(row=0, column=1, padx=5)
        ttk.Label(self.data_options_frame, text="(comma-separated)", foreground="gray").grid(row=0, column=2)

        ttk.Label(self.data_options_frame, text="Start Date:").grid(row=1, column=0, sticky="w")
        self.yahoo_start = tk.StringVar(value="2022-01-01")
        ttk.Entry(self.data_options_frame, textvariable=self.yahoo_start, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(self.data_options_frame, text="End Date:").grid(row=2, column=0, sticky="w")
        self.yahoo_end = tk.StringVar(value="2023-12-31")
        ttk.Entry(self.data_options_frame, textvariable=self.yahoo_end, width=15).grid(row=2, column=1, padx=5)

    def _create_csv_options(self):
        """Create CSV file options."""
        for widget in self.data_options_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.data_options_frame, text="File:").grid(row=0, column=0, sticky="w")
        self.csv_path = tk.StringVar()
        ttk.Entry(self.data_options_frame, textvariable=self.csv_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(self.data_options_frame, text="Browse...", command=self._browse_csv).grid(row=0, column=2)

    def _update_data_options(self):
        """Update data options based on selected source."""
        source = self.data_source.get()
        if source == "sample":
            self._create_sample_options()
        elif source == "yahoo":
            self._create_yahoo_options()
        elif source == "csv":
            self._create_csv_options()

    def _browse_csv(self):
        """Browse for CSV file."""
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path.set(path)

    def _load_data(self):
        """Load data from selected source."""
        try:
            loader = DataLoader()
            source = self.data_source.get()

            if source == "sample":
                self.data = loader.generate_sample_data(
                    days=int(self.sample_days.get()),
                    start_price=float(self.sample_price.get()),
                    volatility=float(self.sample_vol.get()),
                    seed=42,
                )
                self.data_dict = None

            elif source == "yahoo":
                symbols = [s.strip() for s in self.yahoo_symbols.get().split(",")]
                if len(symbols) == 1:
                    self.data = loader.from_yahoo(
                        symbols[0],
                        start=self.yahoo_start.get(),
                        end=self.yahoo_end.get(),
                    )
                    self.data_dict = None
                else:
                    self.data_dict = loader.from_yahoo(
                        symbols,
                        start=self.yahoo_start.get(),
                        end=self.yahoo_end.get(),
                    )
                    self.data = None

            elif source == "csv":
                self.data = loader.from_csv(self.csv_path.get())
                self.data_dict = None

            # Update status
            if self.data is not None:
                bars = len(self.data)
                start = self.data.index[0].strftime("%Y-%m-%d")
                end = self.data.index[-1].strftime("%Y-%m-%d")
                self.data_status.config(
                    text=f"✓ Loaded {bars} bars ({start} to {end})",
                    foreground="green",
                )
            elif self.data_dict is not None:
                symbols = list(self.data_dict.keys())
                bars = len(list(self.data_dict.values())[0])
                self.data_status.config(
                    text=f"✓ Loaded {len(symbols)} symbols, {bars} bars each",
                    foreground="green",
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.data_status.config(text=f"Error: {str(e)}", foreground="red")

    def _on_strategy_selected(self, event=None):
        """Handle strategy selection."""
        strategy_name = self.strategy_var.get()
        self.selected_strategy = AVAILABLE_STRATEGIES.get(strategy_name)

        if self.selected_strategy:
            # Update description
            doc = self.selected_strategy.__doc__ or "No description available"
            self.strategy_desc.config(text=doc.strip().split("\n")[0])

            # Update parameters tab
            self._update_params_tab()

    def _create_params_tab(self):
        """Create the parameters configuration tab."""
        # Instructions
        ttk.Label(
            self.tab_params,
            text="Configure control values and select parameters to optimize:",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 10))

        # Scrollable frame for parameters
        canvas = tk.Canvas(self.tab_params)
        scrollbar = ttk.Scrollbar(self.tab_params, orient="vertical", command=canvas.yview)
        self.params_frame = ttk.Frame(canvas)

        self.params_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.params_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Placeholder
        ttk.Label(
            self.params_frame,
            text="Please select a strategy first",
            foreground="gray",
        ).pack(pady=20)

        # Navigation buttons
        nav_frame = ttk.Frame(self.tab_params)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        ttk.Button(
            nav_frame,
            text="← Back",
            command=lambda: self.notebook.select(0),
        ).pack(side=tk.LEFT)

        ttk.Button(
            nav_frame,
            text="Next: Select Metrics →",
            command=lambda: self.notebook.select(2),
        ).pack(side=tk.RIGHT)

    def _update_params_tab(self):
        """Update parameters tab with strategy parameters."""
        # Clear existing
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        self.control_entries.clear()
        self.optimize_vars.clear()
        self.range_entries.clear()

        if not self.selected_strategy:
            ttk.Label(
                self.params_frame,
                text="Please select a strategy first",
                foreground="gray",
            ).pack(pady=20)
            return

        # Header
        header_frame = ttk.Frame(self.params_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="Parameter", font=("Helvetica", 10, "bold"), width=15).grid(row=0, column=0)
        ttk.Label(header_frame, text="Control Value", font=("Helvetica", 10, "bold"), width=15).grid(row=0, column=1)
        ttk.Label(header_frame, text="Optimize?", font=("Helvetica", 10, "bold"), width=10).grid(row=0, column=2)
        ttk.Label(header_frame, text="Test Values (comma-separated)", font=("Helvetica", 10, "bold"), width=30).grid(row=0, column=3)

        ttk.Separator(self.params_frame, orient="horizontal").pack(fill=tk.X, pady=5)

        # Parameter rows
        for i, param in enumerate(self.selected_strategy.parameters):
            self._create_param_row(param, i)

    def _create_param_row(self, param: Parameter, row: int):
        """Create a row for a parameter."""
        frame = ttk.Frame(self.params_frame)
        frame.pack(fill=tk.X, pady=5)

        # Parameter name and description
        name_frame = ttk.Frame(frame)
        name_frame.grid(row=0, column=0, sticky="w", padx=5)

        ttk.Label(name_frame, text=param.name, font=("Helvetica", 10, "bold")).pack(anchor="w")
        ttk.Label(name_frame, text=param.description, foreground="gray", font=("Helvetica", 8)).pack(anchor="w")

        # Control value entry
        control_var = tk.StringVar(value=str(param.default))
        self.control_entries[param.name] = control_var
        ttk.Entry(frame, textvariable=control_var, width=12).grid(row=0, column=1, padx=5)

        # Optimize checkbox
        optimize_var = tk.BooleanVar(value=True)
        self.optimize_vars[param.name] = optimize_var
        ttk.Checkbutton(frame, variable=optimize_var).grid(row=0, column=2, padx=5)

        # Range/values entry
        default_range = self._get_default_range(param)
        range_var = tk.StringVar(value=default_range)
        self.range_entries[param.name] = range_var
        ttk.Entry(frame, textvariable=range_var, width=40).grid(row=0, column=3, padx=5)

    def _get_default_range(self, param: Parameter) -> str:
        """Get default range string for a parameter."""
        if param.param_type == ParameterType.BOOLEAN:
            return "True, False"
        elif param.param_type == ParameterType.CATEGORICAL:
            return ", ".join(str(c) for c in param.choices)
        elif param.min_value is not None and param.max_value is not None:
            values = param.generate_values(num_values=7)
            return ", ".join(str(v) for v in values)
        return str(param.default)

    def _create_metrics_tab(self):
        """Create the metrics selection tab."""
        ttk.Label(
            self.tab_metrics,
            text="Select performance metrics to calculate:",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 10))

        # Metrics frame
        metrics_frame = ttk.Frame(self.tab_metrics)
        metrics_frame.pack(fill=tk.BOTH, expand=True)

        # Create checkboxes for each metric
        row = 0
        col = 0
        for metric_name, metric_def in AVAILABLE_METRICS.items():
            var = tk.BooleanVar(value=metric_name in [
                "total_return", "annual_return", "sharpe_ratio",
                "max_drawdown", "win_rate", "profit_factor", "total_trades"
            ])
            self.metric_vars[metric_name] = var

            frame = ttk.Frame(metrics_frame)
            frame.grid(row=row, column=col, sticky="w", padx=10, pady=5)

            cb = ttk.Checkbutton(frame, variable=var, text=metric_def.name)
            cb.pack(anchor="w")

            desc = ttk.Label(frame, text=metric_def.description, foreground="gray", font=("Helvetica", 8))
            desc.pack(anchor="w")

            col += 1
            if col >= 3:
                col = 0
                row += 1

        # Select all / none buttons
        btn_frame = ttk.Frame(self.tab_metrics)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Select All", command=self._select_all_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select None", command=self._select_no_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select Default", command=self._select_default_metrics).pack(side=tk.LEFT, padx=5)

        # Navigation buttons
        nav_frame = ttk.Frame(self.tab_metrics)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        ttk.Button(nav_frame, text="← Back", command=lambda: self.notebook.select(1)).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next: Run Optimization →", command=lambda: self.notebook.select(3)).pack(side=tk.RIGHT)

    def _select_all_metrics(self):
        """Select all metrics."""
        for var in self.metric_vars.values():
            var.set(True)

    def _select_no_metrics(self):
        """Deselect all metrics."""
        for var in self.metric_vars.values():
            var.set(False)

    def _select_default_metrics(self):
        """Select default metrics."""
        defaults = ["total_return", "annual_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor", "total_trades"]
        for name, var in self.metric_vars.items():
            var.set(name in defaults)

    def _create_run_tab(self):
        """Create the run and results tab."""
        # Configuration summary
        summary_frame = ttk.LabelFrame(self.tab_run, text="Configuration Summary", padding="10")
        summary_frame.pack(fill=tk.X, pady=(0, 10))

        self.summary_text = tk.Text(summary_frame, height=8, width=80, state="disabled")
        self.summary_text.pack(fill=tk.X)

        # Run button
        run_frame = ttk.Frame(self.tab_run)
        run_frame.pack(fill=tk.X, pady=10)

        self.run_btn = ttk.Button(
            run_frame,
            text="Run Optimization",
            command=self._run_optimization,
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(run_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT, padx=20)

        self.progress_bar = ttk.Progressbar(run_frame, mode="determinate", length=300)
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.tab_run, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.results_text = tk.Text(results_frame, height=15, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Export button
        export_frame = ttk.Frame(self.tab_run)
        export_frame.pack(fill=tk.X)

        self.export_btn = ttk.Button(
            export_frame,
            text="Export to Excel",
            command=self._export_results,
            state="disabled",
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Back button
        ttk.Button(export_frame, text="← Back", command=lambda: self.notebook.select(2)).pack(side=tk.RIGHT)

        # Update summary when tab is selected
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event):
        """Handle tab change events."""
        if self.notebook.index(self.notebook.select()) == 3:
            self._update_summary()

    def _update_summary(self):
        """Update the configuration summary."""
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", tk.END)

        lines = []

        # Data
        if self.data is not None:
            lines.append(f"Data: {len(self.data)} bars")
        elif self.data_dict is not None:
            lines.append(f"Data: {len(self.data_dict)} symbols")
        else:
            lines.append("Data: Not loaded")

        # Strategy
        if self.selected_strategy:
            lines.append(f"Strategy: {self.selected_strategy.name}")
        else:
            lines.append("Strategy: Not selected")

        # Control values
        lines.append("\nControl Values:")
        for name, var in self.control_entries.items():
            lines.append(f"  {name}: {var.get()}")

        # Parameters to optimize
        lines.append("\nParameters to Optimize:")
        for name, var in self.optimize_vars.items():
            if var.get():
                values = self.range_entries[name].get()
                lines.append(f"  {name}: {values}")

        # Metrics
        selected_metrics = [name for name, var in self.metric_vars.items() if var.get()]
        lines.append(f"\nMetrics: {len(selected_metrics)} selected")

        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.config(state="disabled")

    def _run_optimization(self):
        """Run the optimization."""
        # Validate
        if self.data is None and self.data_dict is None:
            messagebox.showerror("Error", "Please load data first")
            return

        if not self.selected_strategy:
            messagebox.showerror("Error", "Please select a strategy")
            return

        selected_metrics = [name for name, var in self.metric_vars.items() if var.get()]
        if not selected_metrics:
            messagebox.showerror("Error", "Please select at least one metric")
            return

        # Build configuration
        try:
            control_values = {}
            for name, var in self.control_entries.items():
                param_def = self.selected_strategy.get_parameter_definitions()[name]
                value = var.get()
                if param_def.param_type == ParameterType.INTEGER:
                    control_values[name] = int(value)
                elif param_def.param_type == ParameterType.FLOAT:
                    control_values[name] = float(value)
                elif param_def.param_type == ParameterType.BOOLEAN:
                    control_values[name] = value.lower() in ("true", "1", "yes")
                else:
                    control_values[name] = value

            parameters_to_optimize = []
            for name, optimize_var in self.optimize_vars.items():
                if optimize_var.get():
                    values_str = self.range_entries[name].get()
                    param_def = self.selected_strategy.get_parameter_definitions()[name]

                    # Parse values
                    values = []
                    for v in values_str.split(","):
                        v = v.strip()
                        if param_def.param_type == ParameterType.INTEGER:
                            values.append(int(v))
                        elif param_def.param_type == ParameterType.FLOAT:
                            values.append(float(v))
                        elif param_def.param_type == ParameterType.BOOLEAN:
                            values.append(v.lower() in ("true", "1", "yes"))
                        else:
                            values.append(v)

                    parameters_to_optimize.append(ParameterRange(name=name, values=values))

            if not parameters_to_optimize:
                messagebox.showerror("Error", "Please select at least one parameter to optimize")
                return

            config = OptimizationConfig(
                strategy_class=self.selected_strategy,
                parameters_to_optimize=parameters_to_optimize,
                control_values=control_values,
                metrics=selected_metrics,
            )

        except Exception as e:
            messagebox.showerror("Error", f"Invalid configuration:\n{str(e)}")
            return

        # Run in thread
        self.run_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.progress_bar["value"] = 0
        self.results_text.delete("1.0", tk.END)

        def run_thread():
            try:
                def progress_callback(param_name, current, total):
                    pct = (current / total) * 100
                    self.window.after(0, lambda: self._update_progress(param_name, pct))

                optimizer = Optimizer(config, progress_callback=progress_callback)
                data = self.data if self.data is not None else self.data_dict
                self.results = optimizer.run(data)

                self.window.after(0, self._on_optimization_complete)

            except Exception as e:
                self.window.after(0, lambda: self._on_optimization_error(str(e)))

        thread = threading.Thread(target=run_thread)
        thread.start()

    def _update_progress(self, param_name: str, pct: float):
        """Update progress display."""
        self.progress_var.set(f"Optimizing {param_name}...")
        self.progress_bar["value"] = pct

    def _on_optimization_complete(self):
        """Handle optimization completion."""
        self.progress_var.set("Complete!")
        self.progress_bar["value"] = 100
        self.run_btn.config(state="normal")
        self.export_btn.config(state="normal")

        # Display results
        self.results_text.delete("1.0", tk.END)

        lines = ["Optimization Results", "=" * 50, ""]

        for param_name, param_results in self.results.parameter_results.items():
            lines.append(f"Parameter: {param_name}")
            lines.append(f"Control Value: {param_results.control_value}")
            lines.append("")

            # Best values
            lines.append("Best Values by Metric:")
            for metric in self.results.config.metrics:
                best = param_results.get_best_value(metric)
                lines.append(f"  {metric}: {best}")
            lines.append("")

            # Results table
            df = param_results.to_dataframe()
            lines.append(df.to_string(index=False))
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        self.results_text.insert("1.0", "\n".join(lines))

    def _on_optimization_error(self, error: str):
        """Handle optimization error."""
        self.progress_var.set("Error!")
        self.run_btn.config(state="normal")
        messagebox.showerror("Error", f"Optimization failed:\n{error}")

    def _export_results(self):
        """Export results to Excel."""
        if not self.results:
            messagebox.showerror("Error", "No results to export")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfilename=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        )

        if filepath:
            try:
                from backtesting.optimization.report import ExcelReportGenerator
                generator = ExcelReportGenerator(self.results)
                generator.generate(filepath)
                messagebox.showinfo("Success", f"Results exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
