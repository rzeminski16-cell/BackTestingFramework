"""
Main entry point for backtesting framework.

This script demonstrates how to use the framework for:
1. Single security backtesting
2. Portfolio backtesting
3. Parameter optimization
4. Trade logging and performance analysis
"""
from pathlib import Path
from datetime import datetime

from Classes.Config.config import (
    BacktestConfig, PortfolioConfig, OptimizationConfig,
    CommissionConfig, CommissionMode
)
from Classes.Data.data_loader import DataLoader
from Classes.Data.security_registry import SecurityRegistry
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Optimization.optimizer import StrategyOptimizer
from Classes.Analysis.trade_logger import TradeLogger
from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Analysis.excel_report_generator import ExcelReportGenerator


def example_single_security_backtest():
    """
    Example: Backtest a single security with a strategy.

    NOTE: This example has been removed as it used PartialExitStrategy which has been deprecated.
    Use backtest_gui.py for running backtests with available strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Security Backtest")
    print("="*80)
    print("This example has been removed. Use backtest_gui.py to run backtests.")
    return None


def example_portfolio_backtest():
    """
    Example: Backtest multiple securities with shared capital.

    NOTE: This example has been removed as it used PartialExitStrategy which has been deprecated.
    Use backtest_gui.py for running backtests with available strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Portfolio Backtest")
    print("="*80)
    print("This example has been removed. Use backtest_gui.py to run backtests.")
    return None


def example_parameter_optimization():
    """
    Example: Optimize strategy parameters for best performance.

    NOTE: This example has been removed as it used PartialExitStrategy which has been deprecated.
    Use backtest_gui.py for running backtests with available strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Parameter Optimization")
    print("="*80)
    print("This example has been removed. Use backtest_gui.py to run backtests.")
    return None


def example_advanced_strategy():
    """
    Example: Use advanced strategy with trailing stops.

    NOTE: This example has been removed as it used AdvancedTrailingStopStrategy which has been deprecated.
    Use backtest_gui.py for running backtests with available strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Advanced Strategy with Trailing Stops")
    print("="*80)
    print("This example has been removed. Use backtest_gui.py to run backtests.")
    return None


def example_partial_exits():
    """
    Example: Strategy with partial position exits.

    NOTE: This example has been removed as it used PartialExitStrategy which has been deprecated.
    Use backtest_gui.py for running backtests with available strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Strategy with Partial Exits")
    print("="*80)
    print("This example has been removed. Use backtest_gui.py to run backtests.")
    return None


def example_excel_report_generation():
    """
    Example: Generate comprehensive Excel report for backtest results.

    NOTE: This example has been removed as it used PartialExitStrategy which has been deprecated.
    Use backtest_gui.py for running backtests with available strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Excel Report Generation")
    print("="*80)
    print("This example has been removed. Use backtest_gui.py to run backtests.")
    return None


def main():
    """
    Main function - run all examples.
    """
    print("\n" + "="*80)
    print("BACKTESTING FRAMEWORK - EXAMPLES")
    print("="*80)

    try:
        # Example 1: Single security backtest
        example_single_security_backtest()

        # Example 2: Portfolio backtest
        example_portfolio_backtest()

        # Example 3: Parameter optimization
        # example_parameter_optimization()  # Commented out by default (takes time)

        # Example 4: Advanced strategy
        # example_advanced_strategy()  # Uncomment if you have required indicators

        # Example 5: Partial exits
        # example_partial_exits()  # Uncomment to run

        # Example 6: Excel report generation
        # example_excel_report_generation()  # Uncomment to generate Excel reports

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Examples completed!")
    print("  - Trade logs: Check 'logs' directory for CSV files")
    print("  - Excel reports: Check 'logs/reports' directory for .xlsx files")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
