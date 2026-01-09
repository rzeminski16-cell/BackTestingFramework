"""
Single Backtest GUI Window.

Provides interface for running a single backtest:
- Data selection
- Strategy configuration
- Run backtest
- View results and metrics
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Type
import pandas as pd

from backtesting.core.engine import BacktestEngine, BacktestResult
from backtesting.strategies.base import Strategy, Parameter, ParameterType
from backtesting.strategies.examples import SMACrossover, RSIStrategy, BollingerBandsStrategy
from backtesting.metrics.performance import PerformanceMetrics, AVAILABLE_METRICS
from backtesting.data import DataLoader


# Available strategies
AVAILABLE_STRATEGIES: Dict[str, Type[Strategy]] = {
    "SMA Crossover": SMACrossover,
    "RSI Strategy": RSIStrategy,
    "Bollinger Bands": BollingerBandsStrategy,
}


class BacktestWindow:
    """Single backtest window."""

    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Single Backtest")
        self.window.geometry("900x700")
        self.window.minsize(700, 500)

        self.data: Optional[pd.DataFrame] = None
        self.selected_strategy: Optional[Type[Strategy]] = None
        self.param_entries: Dict[str, tk.StringVar] = {}
        self.result: Optional[BacktestResult] = None

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
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Configuration
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Data section
        data_frame = ttk.LabelFrame(left_frame, text="Data", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))

        self._create_data_section(data_frame)

        # Strategy section
        strategy_frame = ttk.LabelFrame(left_frame, text="Strategy", padding="10")
        strategy_frame.pack(fill=tk.X, pady=(0, 10))

        self._create_strategy_section(strategy_frame)

        # Parameters section
        self.params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding="10")
        self.params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(self.params_frame, text="Select a strategy first", foreground="gray").pack()

        # Settings section
        settings_frame = ttk.LabelFrame(left_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        self._create_settings_section(settings_frame)

        # Run button
        self.run_btn = ttk.Button(left_frame, text="Run Backtest", command=self._run_backtest)
        self.run_btn.pack(pady=10)

        # Right panel - Results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Results section
        results_frame = ttk.LabelFrame(right_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self._create_results_section(results_frame)

    def _create_data_section(self, parent):
        """Create data section."""
        # Quick load options
        ttk.Label(parent, text="Quick Load:").pack(anchor="w")

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Sample Data", command=self._load_sample).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Yahoo Finance", command=self._load_yahoo).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="CSV File", command=self._load_csv).pack(side=tk.LEFT, padx=2)

        # Data status
        self.data_status = ttk.Label(parent, text="No data loaded", foreground="gray")
        self.data_status.pack(anchor="w", pady=(5, 0))

    def _create_strategy_section(self, parent):
        """Create strategy section."""
        ttk.Label(parent, text="Select Strategy:").pack(anchor="w")

        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(
            parent,
            textvariable=self.strategy_var,
            values=list(AVAILABLE_STRATEGIES.keys()),
            state="readonly",
            width=25,
        )
        strategy_combo.pack(anchor="w", pady=5)
        strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy_selected)

    def _create_settings_section(self, parent):
        """Create settings section."""
        # Initial capital
        ttk.Label(parent, text="Initial Capital:").grid(row=0, column=0, sticky="w", pady=2)
        self.capital_var = tk.StringVar(value="100000")
        ttk.Entry(parent, textvariable=self.capital_var, width=15).grid(row=0, column=1, padx=5, pady=2)

        # Commission
        ttk.Label(parent, text="Commission (%):").grid(row=1, column=0, sticky="w", pady=2)
        self.commission_var = tk.StringVar(value="0.1")
        ttk.Entry(parent, textvariable=self.commission_var, width=15).grid(row=1, column=1, padx=5, pady=2)

        # Slippage
        ttk.Label(parent, text="Slippage (%):").grid(row=2, column=0, sticky="w", pady=2)
        self.slippage_var = tk.StringVar(value="0.05")
        ttk.Entry(parent, textvariable=self.slippage_var, width=15).grid(row=2, column=1, padx=5, pady=2)

    def _create_results_section(self, parent):
        """Create results section."""
        # Metrics display
        self.metrics_text = tk.Text(parent, height=20, width=40)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Trades section
        trades_frame = ttk.LabelFrame(parent, text="Recent Trades", padding="5")
        trades_frame.pack(fill=tk.X, pady=(10, 0))

        columns = ("entry", "exit", "side", "pnl")
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show="headings", height=5)

        self.trades_tree.heading("entry", text="Entry")
        self.trades_tree.heading("exit", text="Exit")
        self.trades_tree.heading("side", text="Side")
        self.trades_tree.heading("pnl", text="P&L %")

        for col in columns:
            self.trades_tree.column(col, width=80)

        self.trades_tree.pack(fill=tk.X)

    def _load_sample(self):
        """Load sample data."""
        try:
            loader = DataLoader()
            self.data = loader.generate_sample_data(days=504, seed=42)
            self._update_data_status()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _load_yahoo(self):
        """Load Yahoo Finance data with dialog."""
        dialog = tk.Toplevel(self.window)
        dialog.title("Yahoo Finance")
        dialog.geometry("300x200")
        dialog.transient(self.window)
        dialog.grab_set()

        ttk.Label(dialog, text="Symbol:").pack(pady=5)
        symbol_var = tk.StringVar(value="AAPL")
        ttk.Entry(dialog, textvariable=symbol_var).pack()

        ttk.Label(dialog, text="Start Date:").pack(pady=5)
        start_var = tk.StringVar(value="2022-01-01")
        ttk.Entry(dialog, textvariable=start_var).pack()

        ttk.Label(dialog, text="End Date:").pack(pady=5)
        end_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(dialog, textvariable=end_var).pack()

        def download():
            try:
                loader = DataLoader()
                self.data = loader.from_yahoo(symbol_var.get(), start=start_var.get(), end=end_var.get())
                self._update_data_status()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(dialog, text="Download", command=download).pack(pady=10)

    def _load_csv(self):
        """Load CSV file."""
        path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            try:
                loader = DataLoader()
                self.data = loader.from_csv(path)
                self._update_data_status()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _update_data_status(self):
        """Update data status label."""
        if self.data is not None:
            bars = len(self.data)
            start = self.data.index[0].strftime("%Y-%m-%d")
            end = self.data.index[-1].strftime("%Y-%m-%d")
            self.data_status.config(
                text=f"✓ {bars} bars ({start} to {end})",
                foreground="green",
            )

    def _on_strategy_selected(self, event=None):
        """Handle strategy selection."""
        strategy_name = self.strategy_var.get()
        self.selected_strategy = AVAILABLE_STRATEGIES.get(strategy_name)

        if self.selected_strategy:
            self._update_params_section()

    def _update_params_section(self):
        """Update parameters section."""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        self.param_entries.clear()

        if not self.selected_strategy:
            ttk.Label(self.params_frame, text="Select a strategy first", foreground="gray").pack()
            return

        for i, param in enumerate(self.selected_strategy.parameters):
            frame = ttk.Frame(self.params_frame)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=f"{param.name}:", width=15).pack(side=tk.LEFT)

            var = tk.StringVar(value=str(param.default))
            self.param_entries[param.name] = var
            ttk.Entry(frame, textvariable=var, width=15).pack(side=tk.LEFT, padx=5)

            ttk.Label(frame, text=param.description, foreground="gray", font=("Helvetica", 8)).pack(side=tk.LEFT)

    def _run_backtest(self):
        """Run the backtest."""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        if not self.selected_strategy:
            messagebox.showerror("Error", "Please select a strategy")
            return

        try:
            # Build parameters
            params = {}
            for name, var in self.param_entries.items():
                param_def = self.selected_strategy.get_parameter_definitions()[name]
                value = var.get()

                if param_def.param_type == ParameterType.INTEGER:
                    params[name] = int(value)
                elif param_def.param_type == ParameterType.FLOAT:
                    params[name] = float(value)
                elif param_def.param_type == ParameterType.BOOLEAN:
                    params[name] = value.lower() in ("true", "1", "yes")
                else:
                    params[name] = value

            # Create strategy and engine
            strategy = self.selected_strategy(**params)
            engine = BacktestEngine(
                initial_capital=float(self.capital_var.get()),
                commission=float(self.commission_var.get()) / 100,
                slippage=float(self.slippage_var.get()) / 100,
            )

            # Run backtest
            self.result = engine.run(strategy, self.data)

            # Display results
            self._display_results()

        except Exception as e:
            messagebox.showerror("Error", f"Backtest failed:\n{str(e)}")

    def _display_results(self):
        """Display backtest results."""
        if not self.result:
            return

        # Calculate metrics
        metrics = PerformanceMetrics(self.result)

        # Clear and update metrics text
        self.metrics_text.delete("1.0", tk.END)

        lines = [
            f"Strategy: {self.result.strategy_name}",
            f"Parameters: {self.result.parameters}",
            "",
            "=" * 40,
            "Performance Metrics",
            "=" * 40,
            "",
        ]

        for metric_name in PerformanceMetrics.DEFAULT_METRICS:
            value = metrics.get(metric_name)
            formatted = metrics.format_metric(metric_name, value)
            defn = AVAILABLE_METRICS[metric_name]
            lines.append(f"{defn.name:25s}: {formatted}")

        lines.extend([
            "",
            "=" * 40,
            "Trade Statistics",
            "=" * 40,
            "",
            f"{'Total Trades':25s}: {len(self.result.trades)}",
            f"{'Average Trade Return':25s}: {metrics.get('avg_trade_return'):.2f}%",
            f"{'Average Win':25s}: {metrics.get('avg_win'):.2f}%",
            f"{'Average Loss':25s}: {metrics.get('avg_loss'):.2f}%",
            f"{'Max Consecutive Wins':25s}: {int(metrics.get('max_consecutive_wins'))}",
            f"{'Max Consecutive Losses':25s}: {int(metrics.get('max_consecutive_losses'))}",
        ])

        self.metrics_text.insert("1.0", "\n".join(lines))

        # Update trades tree
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)

        for trade in self.result.trades[-10:]:  # Last 10 trades
            self.trades_tree.insert("", 0, values=(
                f"${trade.entry_price:.2f}",
                f"${trade.exit_price:.2f}",
                trade.side.value,
                f"{trade.pnl_percent:.2f}%",
            ))
