#!/usr/bin/env python3
"""
Example: Basic Backtesting

This example demonstrates basic usage of the backtesting framework:
1. Load market data
2. Create a strategy
3. Run a backtest
4. Analyze results
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting import BacktestEngine
from backtesting.strategies.examples import SMACrossover, RSIStrategy
from backtesting.data import DataLoader
from backtesting.metrics import PerformanceMetrics


def main():
    print("=" * 60)
    print("Backtesting Framework - Basic Example")
    print("=" * 60)
    print()

    # Load sample data
    print("Loading data...")
    loader = DataLoader()
    data = loader.generate_sample_data(
        days=504,
        start_price=100.0,
        volatility=0.015,
        drift=0.0003,
        seed=42,
    )
    print(f"Loaded {len(data)} bars")
    print()

    # Create strategy
    print("Creating strategy...")
    strategy = SMACrossover(
        fast_period=10,
        slow_period=30,
        position_size=0.95,
    )
    print(f"Strategy: {strategy}")
    print()

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
    )

    # Run backtest
    print("Running backtest...")
    result = engine.run(strategy, data)
    print()

    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = PerformanceMetrics(result)

    print()
    print(metrics.summary())
    print()

    # Show trades
    print(f"Total trades: {len(result.trades)}")
    if result.trades:
        print("\nFirst 5 trades:")
        for i, trade in enumerate(result.trades[:5]):
            print(f"  {i+1}. {trade.side.value.upper()} @ ${trade.entry_price:.2f} -> "
                  f"${trade.exit_price:.2f} | P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")

    print()
    print("=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
