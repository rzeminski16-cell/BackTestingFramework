"""
Backtesting Framework GUI

A graphical user interface for running backtests with configurable parameters.

Features:
- Select securities (single or multiple)
- Choose backtest mode (single security or portfolio)
- Select and configure strategies
- Configure commission settings
- Set date ranges
- Name backtests
- View results
- Save trade logs
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from Classes.Config.config import (
    BacktestConfig, PortfolioConfig, CommissionConfig,
    CommissionMode, OptimizationConfig
)
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Analysis.trade_logger import TradeLogger
from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
from Classes.Optimization.optimizer import StrategyOptimizer

# Import available strategies
from strategies.examples import (
    AdvancedTrailingStopStrategy,
    PartialExitStrategy
)
from strategies.alphatrend_strategy import AlphaTrendStrategy


class BacktestGUI:
    """Main GUI application for backtesting."""

    # Available strategies
    STRATEGIES = {
        'AdvancedTrailingStopStrategy': AdvancedTrailingStopStrategy,
        'PartialExitStrategy': PartialExitStrategy,
        'AlphaTrendStrategy': AlphaTrendStrategy
    }

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Backtesting Framework")
        self.root.geometry("1200x800")

        # Data loader
        self.data_loader = DataLoader(Path('raw_data/processed_exports'))
        self.available_securities = self.data_loader.get_available_symbols()

        # Strategy parameters cache
        self.strategy_params = {}

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

        # ===== Left Panel: Configuration =====
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        row = 0

        # Backtest Name
        ttk.Label(config_frame, text="Backtest Name:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.backtest_name_var = tk.StringVar(value="test_1")
        ttk.Entry(config_frame, textvariable=self.backtest_name_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Backtest Mode
        ttk.Label(config_frame, text="Backtest Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="single")
        mode_frame = ttk.Frame(config_frame)
        mode_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(mode_frame, text="Single Security", variable=self.mode_var,
                       value="single", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Portfolio", variable=self.mode_var,
                       value="portfolio", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        row += 1

        # Securities Selection
        ttk.Label(config_frame, text="Securities:").grid(row=row, column=0, sticky=tk.W, pady=5)
        securities_frame = ttk.Frame(config_frame)
        securities_frame.grid(row=row, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Listbox with scrollbar
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
        for symbol in self.available_securities:
            self.securities_listbox.insert(tk.END, symbol)

        row += 1

        # Strategy Selection
        ttk.Label(config_frame, text="Strategy:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(
            config_frame,
            textvariable=self.strategy_var,
            values=list(self.STRATEGIES.keys()),
            state="readonly",
            width=30
        )
        strategy_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        if self.STRATEGIES:
            strategy_combo.current(0)
            self.on_strategy_change(None)
        row += 1

        # Strategy Parameters Button
        ttk.Button(config_frame, text="Configure Strategy Parameters",
                  command=self.open_strategy_params_window).grid(
            row=row, column=0, columnspan=2, pady=10
        )
        row += 1

        # Capital
        ttk.Label(config_frame, text="Initial Capital:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.capital_var = tk.StringVar(value="100000")
        ttk.Entry(config_frame, textvariable=self.capital_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Commission Mode
        ttk.Label(config_frame, text="Commission Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.commission_mode_var = tk.StringVar(value="percentage")
        comm_frame = ttk.Frame(config_frame)
        comm_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(comm_frame, text="Percentage", variable=self.commission_mode_var,
                       value="percentage").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(comm_frame, text="Fixed", variable=self.commission_mode_var,
                       value="fixed").pack(side=tk.LEFT, padx=5)
        row += 1

        # Commission Value
        ttk.Label(config_frame, text="Commission Value:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.commission_value_var = tk.StringVar(value="0.001")
        ttk.Entry(config_frame, textvariable=self.commission_value_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Label(config_frame, text="(e.g., 0.001 for 0.1% or 3.0 for £3)",
                 font=('Arial', 8)).grid(row=row+1, column=1, sticky=tk.W)
        row += 2

        # Date Range
        ttk.Label(config_frame, text="Start Date (optional):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.start_date_var = tk.StringVar(value="")
        ttk.Entry(config_frame, textvariable=self.start_date_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Label(config_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(
            row=row+1, column=1, sticky=tk.W
        )
        row += 2

        ttk.Label(config_frame, text="End Date (optional):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.end_date_var = tk.StringVar(value="")
        ttk.Entry(config_frame, textvariable=self.end_date_var, width=30).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Label(config_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(
            row=row+1, column=1, sticky=tk.W
        )
        row += 2

        # Excel Report Generation
        self.generate_excel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Generate Excel Report",
                       variable=self.generate_excel_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5
        )
        row += 1

        # Portfolio-specific settings (initially hidden)
        self.portfolio_frame = ttk.LabelFrame(config_frame, text="Portfolio Settings", padding="5")
        self.portfolio_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.portfolio_frame.grid_remove()  # Hide initially

        # Max Positions
        ttk.Label(self.portfolio_frame, text="Max Positions:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_positions_var = tk.StringVar(value="3")
        ttk.Entry(self.portfolio_frame, textvariable=self.max_positions_var, width=10).grid(
            row=0, column=1, sticky=tk.W, pady=2
        )

        # Position Size Limit
        ttk.Label(self.portfolio_frame, text="Position Size Limit:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.position_limit_var = tk.StringVar(value="0.3")
        ttk.Entry(self.portfolio_frame, textvariable=self.position_limit_var, width=10).grid(
            row=1, column=1, sticky=tk.W, pady=2
        )

        # Total Allocation Limit
        ttk.Label(self.portfolio_frame, text="Total Allocation Limit:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.total_allocation_var = tk.StringVar(value="0.9")
        ttk.Entry(self.portfolio_frame, textvariable=self.total_allocation_var, width=10).grid(
            row=2, column=1, sticky=tk.W, pady=2
        )

        row += 1

        # Run Button
        ttk.Button(config_frame, text="Run Backtest", command=self.run_backtest,
                  style='Accent.TButton').grid(
            row=row, column=0, columnspan=2, pady=20, ipadx=20, ipady=10
        )

        # ===== Right Panel: Results =====
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=70,
            height=40,
            font=('Courier', 9)
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.grid(row=0, column=1, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def on_mode_change(self):
        """Handle backtest mode change."""
        mode = self.mode_var.get()
        if mode == "portfolio":
            self.portfolio_frame.grid()
            self.securities_listbox.config(selectmode=tk.MULTIPLE)
        else:
            self.portfolio_frame.grid_remove()
            self.securities_listbox.config(selectmode=tk.SINGLE)

    def on_strategy_change(self, event):
        """Handle strategy selection change."""
        strategy_name = self.strategy_var.get()
        if strategy_name not in self.STRATEGIES:
            return

        # Initialize default parameters for this strategy
        strategy_class = self.STRATEGIES[strategy_name]

        # Get default parameters from strategy
        if strategy_name == 'AdvancedTrailingStopStrategy':
            self.strategy_params[strategy_name] = {
                'sma_period': 200,
                'ema_period': 14,
                'rsi_period': 14,
                'rsi_threshold': 70,
                'position_size': 0.25,
                'initial_stop_pct': 0.08,
                'breakeven_profit_pct': 0.05
            }
        elif strategy_name == 'PartialExitStrategy':
            self.strategy_params[strategy_name] = {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'position_size': 0.3,
                'first_target_pct': 0.10,
                'second_target_pct': 0.20,
                'stop_loss_pct': 0.06
            }

    def open_strategy_params_window(self):
        """Open window to configure strategy parameters."""
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            messagebox.showwarning("No Strategy", "Please select a strategy first.")
            return

        if strategy_name not in self.strategy_params:
            self.on_strategy_change(None)

        # Create parameter window
        param_window = tk.Toplevel(self.root)
        param_window.title(f"Configure {strategy_name} Parameters")
        param_window.geometry("400x500")

        # Parameter entries
        params = self.strategy_params[strategy_name]
        param_vars = {}

        row = 0
        for param_name, param_value in params.items():
            ttk.Label(param_window, text=f"{param_name}:").grid(
                row=row, column=0, sticky=tk.W, padx=10, pady=5
            )
            var = tk.StringVar(value=str(param_value))
            param_vars[param_name] = var
            ttk.Entry(param_window, textvariable=var, width=20).grid(
                row=row, column=1, sticky=(tk.W, tk.E), padx=10, pady=5
            )
            row += 1

        # Save button
        def save_params():
            try:
                for param_name, var in param_vars.items():
                    value = var.get()
                    # Try to convert to appropriate type
                    if '.' in value:
                        self.strategy_params[strategy_name][param_name] = float(value)
                    else:
                        self.strategy_params[strategy_name][param_name] = int(value)
                messagebox.showinfo("Success", "Parameters saved!")
                param_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid parameter value: {e}")

        ttk.Button(param_window, text="Save Parameters", command=save_params).grid(
            row=row, column=0, columnspan=2, pady=20
        )

    def get_selected_securities(self) -> List[str]:
        """Get list of selected securities."""
        selection = self.securities_listbox.curselection()
        return [self.securities_listbox.get(i) for i in selection]

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str.strip():
            return None
        try:
            return datetime.strptime(date_str.strip(), '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

    def run_backtest(self):
        """Run the backtest with configured parameters."""
        try:
            # Validate inputs
            securities = self.get_selected_securities()
            if not securities:
                messagebox.showerror("Error", "Please select at least one security.")
                return

            strategy_name = self.strategy_var.get()
            if not strategy_name:
                messagebox.showerror("Error", "Please select a strategy.")
                return

            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.status_var.set("Running backtest...")
            self.root.update()

            # Get configuration
            mode = self.mode_var.get()
            capital = float(self.capital_var.get())
            commission_mode = CommissionMode.PERCENTAGE if self.commission_mode_var.get() == "percentage" else CommissionMode.FIXED
            commission_value = float(self.commission_value_var.get())

            commission = CommissionConfig(mode=commission_mode, value=commission_value)

            start_date = self.parse_date(self.start_date_var.get())
            end_date = self.parse_date(self.end_date_var.get())

            # Create strategy instance
            strategy_class = self.STRATEGIES[strategy_name]
            strategy_params = self.strategy_params.get(strategy_name, {})
            strategy = strategy_class(**strategy_params)

            # Generate backtest name
            user_name = self.backtest_name_var.get().strip()
            full_backtest_name = f"{strategy_name}_{user_name}"

            # Run backtest based on mode
            if mode == "single":
                self.run_single_backtest(
                    securities[0], strategy, capital, commission,
                    start_date, end_date, full_backtest_name
                )
            else:
                self.run_portfolio_backtest(
                    securities, strategy, capital, commission,
                    start_date, end_date, full_backtest_name
                )

            self.status_var.set("Backtest completed!")

        except Exception as e:
            messagebox.showerror("Error", f"Backtest failed: {str(e)}")
            self.status_var.set("Error!")
            import traceback
            traceback.print_exc()

    def update_progress(self, current: int, total: int):
        """Update progress bar and label."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_var.set(percentage)
            self.progress_label.config(text=f"{current}/{total} ({percentage}%)")
            self.root.update_idletasks()  # Force GUI update

    def reset_progress(self):
        """Reset progress bar."""
        self.progress_var.set(0)
        self.progress_label.config(text="")
        self.root.update_idletasks()

    def run_single_backtest(self, symbol: str, strategy, capital: float,
                          commission: CommissionConfig, start_date, end_date,
                          backtest_name: str):
        """Run single security backtest."""
        # Configure backtest
        config = BacktestConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date
        )

        # Load data
        self.log_result(f"Loading data for {symbol}...")
        data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
        self.log_result(f"Loaded {len(data)} bars\n")

        # Run backtest
        self.log_result(f"Running backtest: {backtest_name}")
        self.log_result(f"Strategy: {strategy}")
        self.log_result(f"Security: {symbol}")
        self.log_result(f"Capital: ${capital:,.2f}\n")

        # Reset progress bar
        self.reset_progress()
        self.status_var.set(f"Processing {len(data)} bars...")

        engine = SingleSecurityEngine(config)
        result = engine.run(symbol, data, strategy, progress_callback=self.update_progress)

        # Reset progress bar
        self.reset_progress()

        # Display results
        self.display_result(symbol, result, backtest_name)

    def run_portfolio_backtest(self, symbols: List[str], strategy, capital: float,
                              commission: CommissionConfig, start_date, end_date,
                              backtest_name: str):
        """Run portfolio backtest."""
        # Get portfolio settings
        max_positions = int(self.max_positions_var.get()) if self.max_positions_var.get() else None
        position_limit = float(self.position_limit_var.get())
        total_allocation = float(self.total_allocation_var.get())

        # Configure portfolio
        config = PortfolioConfig(
            initial_capital=capital,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            max_positions=max_positions,
            position_size_limit=position_limit,
            total_allocation_limit=total_allocation
        )

        # Load data
        self.log_result(f"Loading data for {len(symbols)} securities...")
        data_dict = {}
        for symbol in symbols:
            try:
                data = self.data_loader.load_csv(symbol, required_columns=strategy.required_columns())
                data_dict[symbol] = data
                self.log_result(f"  {symbol}: {len(data)} bars")
            except Exception as e:
                self.log_result(f"  {symbol}: FAILED - {e}")

        if not data_dict:
            raise ValueError("No data loaded successfully")

        self.log_result("")

        # Run backtest
        self.log_result(f"Running portfolio backtest: {backtest_name}")
        self.log_result(f"Strategy: {strategy}")
        self.log_result(f"Securities: {', '.join(data_dict.keys())}")
        self.log_result(f"Capital: ${capital:,.2f}\n")

        # Reset progress bar
        self.reset_progress()
        self.status_var.set(f"Processing portfolio backtest...")

        engine = PortfolioEngine(config)
        results = engine.run(data_dict, strategy, progress_callback=self.update_progress)

        # Reset progress bar
        self.reset_progress()

        # Display results
        self.display_portfolio_results(results, backtest_name)

    def display_result(self, symbol: str, result, backtest_name: str):
        """Display single backtest result."""
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_metrics(result)

        # Display metrics
        self.log_result("=" * 70)
        self.log_result(f"RESULTS: {symbol}")
        self.log_result("=" * 70)
        self.log_result(f"Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        self.log_result(f"Final Equity:        ${metrics['final_equity']:,.2f}")
        self.log_result(f"Number of Trades:    {metrics['num_trades']}")
        self.log_result(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
        self.log_result(f"Winning Trades:      {metrics['num_wins']}")
        self.log_result(f"Losing Trades:       {metrics['num_losses']}")
        self.log_result(f"Average Win:         ${metrics['avg_win']:,.2f}")
        self.log_result(f"Average Loss:        ${metrics['avg_loss']:,.2f}")
        self.log_result(f"Largest Win:         ${metrics['largest_win']:,.2f}")
        self.log_result(f"Largest Loss:        ${metrics['largest_loss']:,.2f}")
        self.log_result(f"Profit Factor:       {metrics['profit_factor']:.2f}")
        self.log_result(f"Avg Trade Duration:  {metrics['avg_trade_duration']:.1f} days")
        self.log_result(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        self.log_result(f"Max Drawdown:        ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        self.log_result("=" * 70)

        # Save trade log
        logger = TradeLogger(Path('logs') / backtest_name)
        logger.log_trades(symbol, backtest_name, result.trades)
        self.log_result(f"\nTrade log saved to: logs/{backtest_name}/{backtest_name}_{symbol}_trades.csv")

        # Generate Excel report if enabled
        if self.generate_excel_var.get():
            try:
                self.log_result("\nGenerating Excel report...")
                excel_generator = ExcelReportGenerator(
                    output_directory=Path('logs') / backtest_name / 'reports',
                    initial_capital=float(self.capital_var.get()),
                    risk_free_rate=0.02,
                    benchmark_name="S&P 500"
                )
                report_path = excel_generator.generate_report(
                    result=result,
                    filename=f"{backtest_name}_{symbol}_report.xlsx"
                )
                self.log_result(f"✓ Excel report saved to: {report_path}")
            except Exception as e:
                self.log_result(f"⚠ Excel report generation failed: {str(e)}")

    def display_portfolio_results(self, results: Dict, backtest_name: str):
        """Display portfolio backtest results."""
        self.log_result("=" * 70)
        self.log_result("PORTFOLIO RESULTS")
        self.log_result("=" * 70)

        total_pl = 0
        total_trades = 0

        # Table header
        self.log_result(f"{'Symbol':<10} {'Trades':<10} {'P/L':<15} {'Return %':<12}")
        self.log_result("-" * 70)

        # Results per symbol
        for symbol, result in results.items():
            total_pl += result.total_return
            total_trades += result.num_trades
            self.log_result(
                f"{symbol:<10} {result.num_trades:<10} "
                f"${result.total_return:>12,.2f} {result.total_return_pct:>10.2f}%"
            )

        self.log_result("-" * 70)

        # Get initial capital from first result
        initial_capital = list(results.values())[0].equity_curve.iloc[0]['equity']
        total_return_pct = (total_pl / initial_capital) * 100

        self.log_result(
            f"{'TOTAL':<10} {total_trades:<10} "
            f"${total_pl:>12,.2f} {total_return_pct:>10.2f}%"
        )
        self.log_result("=" * 70)

        # Save trade logs
        logger = TradeLogger(Path('logs') / backtest_name)
        for symbol, result in results.items():
            logger.log_trades(symbol, backtest_name, result.trades)

        self.log_result(f"\nTrade logs saved to: logs/{backtest_name}/")

        # Generate Excel reports if enabled
        if self.generate_excel_var.get():
            try:
                self.log_result("\nGenerating Excel reports for portfolio...")
                excel_generator = ExcelReportGenerator(
                    output_directory=Path('logs') / backtest_name / 'reports',
                    initial_capital=float(self.capital_var.get()),
                    risk_free_rate=0.02,
                    benchmark_name="S&P 500"
                )

                for symbol, result in results.items():
                    report_path = excel_generator.generate_report(
                        result=result,
                        filename=f"{backtest_name}_{symbol}_report.xlsx"
                    )
                    self.log_result(f"  ✓ {symbol}: {report_path.name}")

                self.log_result(f"\n✓ All Excel reports saved to: logs/{backtest_name}/reports/")
            except Exception as e:
                self.log_result(f"⚠ Excel report generation failed: {str(e)}")

    def log_result(self, message: str):
        """Log message to results text area."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update()


def main():
    """Main entry point for GUI."""
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    style.theme_use('clam')

    app = BacktestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
