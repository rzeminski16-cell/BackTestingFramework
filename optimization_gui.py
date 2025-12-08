"""
Optimization GUI

A graphical interface for walk-forward optimization with Bayesian optimization.

Features:
- Select strategy to optimize
- Select securities (single or multiple)
- Choose which parameters to optimize
- Configure optimization settings
- Run walk-forward optimization
- View progress with real-time updates
- Generate comprehensive Excel reports
- Cancel optimization in progress
"""

import logging
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

from Classes.Data.data_loader import DataLoader
from Classes.Optimization.optimization_report_generator import \
    OptimizationReportGenerator
from Classes.Optimization.sensitivity_analyzer import SensitivityAnalyzer
from Classes.Optimization.walk_forward_optimizer import WalkForwardOptimizer

# Import available strategies
from strategies.alphatrend_strategy import AlphaTrendStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationGUI:
    """Main GUI application for walk-forward optimization."""

    # Available strategies
    STRATEGIES = {
        'AlphaTrendStrategy': AlphaTrendStrategy
    }

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Walk-Forward Optimization")
        self.root.geometry("1000x700")

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/processed_exports'))
        self.available_securities = self.data_loader.get_available_symbols()

        # Optimization components
        self.optimizer = WalkForwardOptimizer()
        self.sensitivity_analyzer = SensitivityAnalyzer(self.optimizer.config)
        self.report_generator = OptimizationReportGenerator(self.optimizer.config)

        # State
        self.optimization_thread = None
        self.is_running = False
        self.selected_parameters = {}  # Track which parameters to optimize

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ===== Left Panel: Configuration =====
        config_frame = ttk.LabelFrame(main_frame, text="Optimization Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        row = 0

        # Strategy Selection
        ttk.Label(config_frame, text="Strategy:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(
            config_frame,
            textvariable=self.strategy_var,
            values=list(self.STRATEGIES.keys()),
            state="readonly",
            width=28
        )
        self.strategy_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        if self.STRATEGIES:
            self.strategy_combo.current(0)
        self.strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        row += 1

        # Securities Selection
        ttk.Label(config_frame, text="Securities:").grid(row=row, column=0, sticky=tk.W, pady=5)
        securities_container = ttk.Frame(config_frame)
        securities_container.grid(row=row, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Listbox with scrollbar
        securities_frame = ttk.Frame(securities_container)
        securities_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(securities_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.securities_listbox = tk.Listbox(
            securities_frame,
            selectmode=tk.MULTIPLE,
            height=6,
            yscrollcommand=scrollbar.set
        )
        self.securities_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.securities_listbox.yview)

        # Populate securities
        for symbol in sorted(self.available_securities):
            self.securities_listbox.insert(tk.END, symbol)

        # Select/Deselect all buttons
        button_frame = ttk.Frame(securities_container)
        button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Select All", command=self.select_all_securities).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_securities).pack(side=tk.LEFT, padx=2)

        row += 1

        # Run Sensitivity Analysis checkbox
        self.run_sensitivity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_frame,
            text="Run Sensitivity Analysis",
            variable=self.run_sensitivity_var
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # Speed mode selection
        ttk.Label(config_frame, text="Speed Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.speed_mode_var = tk.StringVar(value="full")
        speed_frame = ttk.Frame(config_frame)
        speed_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(speed_frame, text="Full (100 iter)", variable=self.speed_mode_var,
                       value="full").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(speed_frame, text="Fast (50 iter)", variable=self.speed_mode_var,
                       value="fast").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(speed_frame, text="Quick (25 iter)", variable=self.speed_mode_var,
                       value="quick").pack(side=tk.LEFT, padx=5)
        row += 1

        # Parallel jobs setting
        ttk.Label(config_frame, text="CPU Cores:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_jobs_var = tk.IntVar(value=1)
        jobs_frame = ttk.Frame(config_frame)
        jobs_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(jobs_frame, text="1 (Slower)", variable=self.n_jobs_var,
                       value=1).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(jobs_frame, text="2", variable=self.n_jobs_var,
                       value=2).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(jobs_frame, text="4", variable=self.n_jobs_var,
                       value=4).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(jobs_frame, text="All", variable=self.n_jobs_var,
                       value=-1).pack(side=tk.LEFT, padx=5)
        row += 1

        # Config file path display
        ttk.Label(config_frame, text="Config File:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(
            config_frame,
            text="config/optimization_config.yaml",
            font=('TkDefaultFont', 8, 'italic')
        ).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)

        self.configure_params_button = ttk.Button(
            button_frame,
            text="Select Parameters...",
            command=self.configure_parameters
        )
        self.configure_params_button.pack(side=tk.LEFT, padx=5)

        # Parameter selection status label
        self.param_status_label = ttk.Label(
            config_frame,
            text="(All parameters will be optimized)",
            font=('TkDefaultFont', 8, 'italic'),
            foreground='gray'
        )
        self.param_status_label.grid(row=row + 1, column=0, columnspan=2, pady=(0, 5))

        self.optimize_button = ttk.Button(
            button_frame,
            text="Start Optimization",
            command=self.start_optimization
        )
        self.optimize_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_optimization,
            state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # ===== Right Panel: Results and Progress =====
        results_frame = ttk.LabelFrame(main_frame, text="Progress & Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Progress bar
        ttk.Label(results_frame, text="Progress:").pack(anchor=tk.W, pady=(0, 5))
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(results_frame, textvariable=self.progress_var)
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(results_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Results text area
        ttk.Label(results_frame, text="Results:").pack(anchor=tk.W, pady=(10, 5))
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            width=60,
            height=30,
            wrap=tk.WORD,
            font=('Courier', 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def select_all_securities(self):
        """Select all securities in the listbox."""
        self.securities_listbox.select_set(0, tk.END)

    def deselect_all_securities(self):
        """Deselect all securities in the listbox."""
        self.securities_listbox.select_clear(0, tk.END)

    def on_strategy_change(self, event=None):
        """Handle strategy selection change."""
        strategy_name = self.strategy_var.get()
        self.log_message(f"Selected strategy: {strategy_name}")
        # Reset parameter selection when strategy changes
        self.selected_parameters = {}
        self.param_status_label.config(text="(All parameters will be optimized)")

    def configure_parameters(self):
        """Open dialog to select which parameters to optimize."""
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            messagebox.showerror("Error", "Please select a strategy first")
            return

        # Get parameter configuration for this strategy
        strategy_config = self.optimizer.config['strategy_parameters'].get(strategy_name)
        if not strategy_config:
            messagebox.showerror("Error", f"No parameter configuration found for {strategy_name}")
            return

        # Get strategy class to read default values
        strategy_class = self.STRATEGIES[strategy_name]

        # Initialize selected_parameters if empty (first time for this strategy)
        if not self.selected_parameters:
            for param_name in strategy_config.keys():
                self.selected_parameters[param_name] = True

        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Select Parameters to Optimize - {strategy_name}")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Title
        title_label = ttk.Label(
            dialog,
            text="Select which parameters to optimize:",
            font=('TkDefaultFont', 11, 'bold')
        )
        title_label.pack(pady=10)

        # Instructions
        instr_label = ttk.Label(
            dialog,
            text="Checked = Optimize | Unchecked = Use default value",
            font=('TkDefaultFont', 9, 'italic')
        )
        instr_label.pack(pady=5)

        # Create scrollable frame for parameters
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Get default values from strategy
        try:
            default_strategy = strategy_class()
            default_params = {
                param: getattr(default_strategy, param, None)
                for param in strategy_config.keys()
            }
        except:
            # If can't instantiate, use config mins as defaults
            default_params = {
                param: spec.get('min', 0)
                for param, spec in strategy_config.items()
            }

        # Create checkboxes for each parameter
        param_vars = {}

        for idx, (param_name, param_spec) in enumerate(strategy_config.items()):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=20, pady=5)

            # Use existing selection state
            var = tk.BooleanVar(value=self.selected_parameters.get(param_name, True))
            param_vars[param_name] = var

            # Checkbox
            cb = ttk.Checkbutton(frame, variable=var, width=3)
            cb.pack(side=tk.LEFT)

            # Parameter name and info
            param_type = param_spec.get('type', 'float')
            min_val = param_spec.get('min', 'N/A')
            max_val = param_spec.get('max', 'N/A')
            default_val = default_params.get(param_name, 'N/A')

            if 'values' in param_spec:
                range_str = f"Values: {param_spec['values']}"
            else:
                if param_type == 'int':
                    range_str = f"Range: {int(min_val)} to {int(max_val)}"
                else:
                    range_str = f"Range: {min_val:.2f} to {max_val:.2f}"

            info_text = f"{param_name} ({param_type})\n  {range_str}, Default: {default_val}"

            label = ttk.Label(frame, text=info_text, font=('TkDefaultFont', 9))
            label.pack(side=tk.LEFT, padx=10)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def select_all():
            for var in param_vars.values():
                var.set(True)

        def deselect_all():
            for var in param_vars.values():
                var.set(False)

        def save_and_close():
            # Save selections
            for param_name, var in param_vars.items():
                self.selected_parameters[param_name] = var.get()

            # Count selected
            num_selected = sum(self.selected_parameters.values())
            num_total = len(self.selected_parameters)

            # Update status label
            if num_selected == num_total:
                self.param_status_label.config(text="(All parameters will be optimized)")
            elif num_selected == 0:
                self.param_status_label.config(text="(No parameters selected - using all defaults)")
            else:
                num_fixed = num_total - num_selected
                self.param_status_label.config(
                    text=f"(Optimizing {num_selected} params, fixing {num_fixed} at defaults)"
                )

            self.log_message(f"Parameter selection updated: {num_selected}/{num_total} parameters selected for optimization")
            dialog.destroy()

        ttk.Button(button_frame, text="Select All", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Deselect All", command=deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=save_and_close).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def log_message(self, message: str):
        """Log a message to the results text area."""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.results_text.update()

    def update_progress(self, stage: str, current: int, total: int):
        """Update progress bar and label."""
        # Cap current at total (Bayesian optimizer sometimes goes slightly over)
        current = min(current, total)
        percentage = (current / total * 100) if total > 0 else 0
        percentage = min(percentage, 100.0)  # Cap at 100%
        self.progress_var.set(f"{stage}: {current}/{total} ({percentage:.1f}%)")
        self.progress_bar['value'] = percentage
        self.root.update_idletasks()

    def start_optimization(self):
        """Start the optimization process."""
        # Validate inputs
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            messagebox.showerror("Error", "Please select a strategy")
            return

        selected_indices = self.securities_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one security")
            return

        selected_securities = [self.securities_listbox.get(i) for i in selected_indices]

        # Confirm
        confirm_msg = (
            f"Start optimization for:\n"
            f"Strategy: {strategy_name}\n"
            f"Securities: {', '.join(selected_securities)}\n\n"
            f"This may take a long time depending on data size and configuration."
        )

        if not messagebox.askyesno("Confirm Optimization", confirm_msg):
            return

        # Clear results
        self.results_text.delete('1.0', tk.END)

        # Disable optimize button, enable cancel
        self.optimize_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.is_running = True

        # Start optimization in background thread
        self.optimization_thread = threading.Thread(
            target=self.run_optimization,
            args=(strategy_name, selected_securities),
            daemon=True
        )
        self.optimization_thread.start()

    def cancel_optimization(self):
        """Cancel the running optimization."""
        if self.is_running:
            self.optimizer.cancel()
            self.sensitivity_analyzer.cancel()
            self.log_message("\n=== CANCELLATION REQUESTED ===")
            self.log_message("Waiting for current operation to complete...")

    def run_optimization(self, strategy_name: str, securities: List[str]):
        """
        Run optimization for selected strategy and securities.
        This runs in a background thread.
        """
        try:
            strategy_class = self.STRATEGIES[strategy_name]

            # Apply GUI settings to optimizer config
            speed_mode = self.speed_mode_var.get()
            n_jobs = self.n_jobs_var.get()

            self.optimizer.config['bayesian_optimization']['speed_mode'] = speed_mode
            self.optimizer.config['bayesian_optimization']['n_jobs'] = n_jobs

            self.log_message(f"Speed Mode: {speed_mode.upper()}")
            self.log_message(f"CPU Cores: {n_jobs if n_jobs > 0 else 'All available'}")

            if n_jobs != 1:
                self.log_message("Note: If parallel processing fails, will automatically fall back to serial processing")
                self.log_message("")

            for sec_idx, symbol in enumerate(securities):
                if not self.is_running:
                    self.log_message("Optimization cancelled by user")
                    break

                self.log_message(f"\n{'=' * 60}")
                self.log_message(f"OPTIMIZING: {symbol} ({sec_idx + 1}/{len(securities)})")
                self.log_message(f"{'=' * 60}\n")

                # Load data
                self.log_message(f"Loading data for {symbol}...")
                try:
                    data = self.data_loader.load_csv(symbol)
                    self.log_message(f"Loaded {len(data)} bars of data")
                except Exception as e:
                    self.log_message(f"ERROR: Failed to load data for {symbol}: {e}")
                    continue

                # Run walk-forward optimization
                self.log_message("\nStarting walk-forward optimization...")
                self.log_message(f"Optimizing {sum(self.selected_parameters.values())} parameters, " +
                               f"fixing {len(self.selected_parameters) - sum(self.selected_parameters.values())} parameters")

                try:
                    wf_results = self.optimizer.optimize(
                        strategy_class=strategy_class,
                        symbol=symbol,
                        data=data,
                        selected_params=self.selected_parameters if self.selected_parameters else None,
                        progress_callback=lambda stage, curr, total: self.root.after(
                            0, self.update_progress, stage, curr, total
                        )
                    )

                    if not self.is_running:
                        break

                    # Display results
                    self.log_message("\n" + "=" * 60)
                    self.log_message("WALK-FORWARD OPTIMIZATION RESULTS")
                    self.log_message("=" * 60)
                    self.log_message(f"Total Windows: {wf_results.total_windows}")
                    self.log_message(f"Windows Passed Constraints: {wf_results.windows_passed_constraints}")
                    self.log_message(f"Success Rate: {wf_results.windows_passed_constraints / wf_results.total_windows * 100:.1f}%")
                    self.log_message(f"\nAvg In-Sample Sortino: {wf_results.avg_in_sample_sortino:.4f}")
                    self.log_message(f"Avg Out-Sample Sortino: {wf_results.avg_out_sample_sortino:.4f}")
                    self.log_message(f"Sortino Degradation: {wf_results.avg_sortino_degradation_pct:.2f}%")
                    self.log_message(f"\nAvg In-Sample Sharpe: {wf_results.avg_in_sample_sharpe:.4f}")
                    self.log_message(f"Avg Out-Sample Sharpe: {wf_results.avg_out_sample_sharpe:.4f}")
                    self.log_message(f"Sharpe Degradation: {wf_results.avg_sharpe_degradation_pct:.2f}%")

                    self.log_message("\nRecommended Parameters:")
                    for param_name, param_value in wf_results.most_common_params.items():
                        min_val, max_val = wf_results.parameter_ranges[param_name]
                        self.log_message(f"  {param_name}: {param_value:.4f} (range: {min_val:.2f} - {max_val:.2f})")

                except Exception as e:
                    self.log_message(f"ERROR during walk-forward optimization: {e}")
                    logger.exception("Walk-forward optimization failed")
                    continue

                # Run sensitivity analysis
                sensitivity_results = None
                if self.run_sensitivity_var.get() and self.is_running:
                    self.log_message("\n\nStarting sensitivity analysis...")
                    try:
                        sensitivity_results = self.sensitivity_analyzer.analyze(
                            strategy_class=strategy_class,
                            base_parameters=wf_results.most_common_params,
                            data=data,
                            symbol=symbol,
                            progress_callback=lambda curr, total: self.root.after(
                                0, self.update_progress, "Sensitivity Analysis", curr, total
                            )
                        )

                        if not self.is_running:
                            break

                        # Display sensitivity results
                        self.log_message("\n" + "=" * 60)
                        self.log_message("SENSITIVITY ANALYSIS RESULTS")
                        self.log_message("=" * 60)
                        self.log_message(f"Total Tests: {len(sensitivity_results.all_tests)}")
                        self.log_message(f"Overall Sortino Range: {sensitivity_results.overall_sortino_range_pct:.2f}%")
                        self.log_message(f"Overall Assessment: {'ROBUST' if sensitivity_results.is_overall_robust else 'SENSITIVE'}")

                        self.log_message("\nMost Robust Parameters:")
                        for param in sensitivity_results.most_robust_params:
                            self.log_message(f"  - {param}")

                        self.log_message("\nLeast Robust Parameters (require monitoring):")
                        for param in sensitivity_results.least_robust_params:
                            self.log_message(f"  - {param}")

                    except Exception as e:
                        self.log_message(f"ERROR during sensitivity analysis: {e}")
                        logger.exception("Sensitivity analysis failed")

                # Generate Excel report
                if self.is_running:
                    self.log_message("\n\nGenerating Excel report...")
                    try:
                        report_path = self.report_generator.generate_report(
                            wf_results=wf_results,
                            sensitivity_results=sensitivity_results
                        )
                        self.log_message(f"Report saved to: {report_path}")
                        self.log_message("\nOPTIMIZATION COMPLETE!")

                    except Exception as e:
                        self.log_message(f"ERROR generating report: {e}")
                        logger.exception("Report generation failed")

            if self.is_running:
                self.log_message(f"\n\n{'=' * 60}")
                self.log_message("ALL OPTIMIZATIONS COMPLETE")
                self.log_message(f"{'=' * 60}")
                messagebox.showinfo("Complete", "Optimization completed successfully!")

        except Exception as e:
            self.log_message(f"\n\nFATAL ERROR: {e}")
            logger.exception("Optimization failed")
            messagebox.showerror("Error", f"Optimization failed: {e}")

        finally:
            # Re-enable buttons
            self.root.after(0, self._reset_ui)

    def _reset_ui(self):
        """Reset UI state after optimization completes or is cancelled."""
        self.is_running = False
        self.optimize_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_var.set("Ready")
        self.progress_bar['value'] = 0


def main():
    """Main entry point for the optimization GUI."""
    root = tk.Tk()
    app = OptimizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
