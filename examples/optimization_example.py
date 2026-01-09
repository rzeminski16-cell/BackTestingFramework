#!/usr/bin/env python3
"""
Example: Parameter Optimization with Excel Report Generation

This example demonstrates the new optimization system:
1. Select securities (single or portfolio)
2. Select a strategy
3. Configure parameters:
   - Control values (baseline for all parameters)
   - Parameters to optimize with value ranges
   - Metrics to examine
4. Run optimization and generate Excel report

The optimization tests ONE parameter at a time while keeping others at control values.
This enables sensitivity analysis to understand each parameter's impact.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting import BacktestEngine, Optimizer
from backtesting.optimization import OptimizationConfig, ParameterRange
from backtesting.strategies.examples import SMACrossover, RSIStrategy, BollingerBandsStrategy
from backtesting.data import DataLoader
from backtesting.metrics import PerformanceMetrics


def progress_callback(param_name: str, current: int, total: int) -> None:
    """Callback to show optimization progress."""
    pct = (current / total) * 100
    print(f"\rOptimizing: {param_name} [{current}/{total}] {pct:.1f}%", end="", flush=True)


def main():
    print("=" * 60)
    print("Backtesting Framework - Parameter Optimization Example")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 1: Load Data (Single Security or Portfolio)
    # =========================================================================
    print("Step 1: Loading data...")

    # Option A: Generate sample data for testing
    loader = DataLoader()
    data = loader.generate_sample_data(
        days=504,  # ~2 years of trading days
        start_price=100.0,
        volatility=0.015,
        drift=0.0003,
        seed=42,
    )

    # Option B: Load from Yahoo Finance (uncomment to use)
    # data = loader.from_yahoo("AAPL", start="2022-01-01", end="2023-12-31")

    # Option C: Load portfolio of securities
    # data = loader.from_yahoo(
    #     ["AAPL", "GOOGL", "MSFT"],
    #     start="2022-01-01",
    #     end="2023-12-31"
    # )

    print(f"  Loaded {len(data)} bars of data")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print()

    # =========================================================================
    # STEP 2: Select Strategy
    # =========================================================================
    print("Step 2: Selecting strategy...")

    # Available strategies:
    # - SMACrossover: Simple moving average crossover
    # - RSIStrategy: RSI overbought/oversold
    # - BollingerBandsStrategy: Bollinger Bands mean reversion

    strategy_class = SMACrossover

    print(f"  Strategy: {strategy_class.name}")
    print(f"  Parameters:")
    for param in strategy_class.parameters:
        print(f"    - {param.name}: {param.description} (default: {param.default})")
    print()

    # =========================================================================
    # STEP 3: Configure Parameters
    # =========================================================================
    print("Step 3: Configuring parameters...")

    # 3a. Set CONTROL values for all parameters
    # These are the baseline values used when NOT testing a parameter
    control_values = {
        "fast_period": 10,      # Control value for fast SMA
        "slow_period": 30,      # Control value for slow SMA
        "position_size": 0.95,  # Control value for position sizing
    }

    # 3b. Select which parameters to optimize and their test ranges
    parameters_to_optimize = [
        # Test fast_period from 5 to 25 in steps of 2
        ParameterRange(
            name="fast_period",
            values=[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
        ),
        # Test slow_period from 20 to 60 in steps of 5
        ParameterRange(
            name="slow_period",
            values=[20, 25, 30, 35, 40, 45, 50, 55, 60],
        ),
        # Can also auto-generate values based on parameter definition
        # ParameterRange(name="position_size", num_values=5),
    ]

    # 3c. Select performance metrics to examine
    metrics_to_examine = [
        "total_return",      # Total percentage return
        "annual_return",     # Annualized return
        "sharpe_ratio",      # Risk-adjusted return
        "max_drawdown",      # Maximum peak-to-trough decline
        "win_rate",          # Percentage of winning trades
        "profit_factor",     # Gross profit / gross loss
        "total_trades",      # Number of trades
    ]

    print(f"  Control values: {control_values}")
    print(f"  Parameters to optimize: {[p.name for p in parameters_to_optimize]}")
    print(f"  Metrics to examine: {metrics_to_examine}")
    print()

    # =========================================================================
    # STEP 4: Create Optimization Configuration
    # =========================================================================
    print("Step 4: Creating optimization configuration...")

    config = OptimizationConfig(
        strategy_class=strategy_class,
        parameters_to_optimize=parameters_to_optimize,
        control_values=control_values,
        metrics=metrics_to_examine,
        initial_capital=100000.0,
        commission=0.001,   # 0.1% commission
        slippage=0.0005,    # 0.05% slippage
    )

    print("  Configuration created successfully")
    print()

    # =========================================================================
    # STEP 5: Run Optimization
    # =========================================================================
    print("Step 5: Running optimization...")
    print()

    optimizer = Optimizer(config, progress_callback=progress_callback)
    results = optimizer.run(data)

    print()  # New line after progress
    print()
    print("  Optimization complete!")
    print()

    # =========================================================================
    # STEP 6: View Results
    # =========================================================================
    print("Step 6: Viewing results...")
    print()

    # Print summary for each parameter
    for param_name, param_results in results.parameter_results.items():
        print(f"  Parameter: {param_name}")
        print(f"  Control value: {param_results.control_value}")
        print(f"  Values tested: {param_results.values_tested}")
        print()

        # Show best values for each metric
        print("  Best values by metric:")
        for metric in metrics_to_examine:
            best_value = param_results.get_best_value(metric)
            print(f"    {metric}: {best_value}")
        print()

        # Show data table
        df = param_results.to_dataframe()
        print("  Results table:")
        print(df.to_string(index=False))
        print()
        print("-" * 40)
        print()

    # =========================================================================
    # STEP 7: Export to Excel with Charts
    # =========================================================================
    print("Step 7: Exporting to Excel...")

    output_file = "optimization_results.xlsx"
    optimizer.export_to_excel(results, output_file)

    print(f"  Report saved to: {output_file}")
    print()
    print("  The Excel file contains:")
    print("    - Summary tab: Control values, configuration, best values")
    print("    - One tab per parameter: Data table + line charts for each metric")
    print()

    print("=" * 60)
    print("Optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
