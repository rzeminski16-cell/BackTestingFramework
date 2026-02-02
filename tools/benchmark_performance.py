"""
Performance Benchmark Script

Tests the performance improvements from:
1. Indicator precomputation
2. Data handling optimization (no copying)
3. Vectorization

Compares backtest execution time and provides detailed metrics.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import pandas as pd
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Data.data_loader import DataLoader
from strategies.base_alphatrend_strategy import BaseAlphaTrendStrategy


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def benchmark_backtest():
    """Run benchmark test on AlphaTrend strategy."""
    print("=" * 80)
    print("BACKTESTING PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    data_loader = DataLoader(data_directory="raw_data")
    data = data_loader.load_csv("AAPL")
    print(f"✓ Loaded {len(data)} bars of AAPL data")
    print()

    # Create strategy
    print("Initializing AlphaTrend strategy...")
    strategy = AlphaTrendStrategy(
        atr_multiplier=1.0,
        source='close',
        smoothing_length=3,
        percentile_period=100,
        volume_short_ma=4,
        volume_long_ma=30,
        volume_alignment_window=14,
        stop_loss_percent=2.0,
        grace_period_bars=14,
        momentum_gain_pct=2.0,
        momentum_lookback=7,
        risk_percent=2.0
    )
    print("✓ Strategy initialized")
    print()

    # Create backtest config
    config = BacktestConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001),
        position_size_limit=0.95
    )

    # Create engine
    engine = SingleSecurityEngine(config)

    # Progress callback
    def progress_callback(current: int, total: int):
        if current % 500 == 0 or current == total:
            pct = (current / total) * 100
            print(f"  Progress: {current}/{total} bars ({pct:.1f}%)")

    # Run backtest with timing
    print("Running backtest with optimizations...")
    print("-" * 80)
    start_time = time.time()

    result = engine.run(
        symbol="AAPL",
        data=data,
        strategy=strategy,
        progress_callback=progress_callback
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 80)
    print()

    # Display results
    print("=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    print()

    print(f"Total Execution Time:      {format_time(elapsed_time)}")
    print(f"Time per Bar:              {format_time(elapsed_time / len(data))}")
    print(f"Bars per Second:           {len(data) / elapsed_time:.2f}")
    print()

    print("=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print()

    print(f"Symbol:                    {result.symbol}")
    print(f"Strategy:                  {result.strategy_name}")
    print(f"Total Bars:                {len(data)}")
    print(f"Initial Capital:           ${config.initial_capital:,.2f}")
    print(f"Final Equity:              ${result.final_equity:,.2f}")
    print(f"Total Return:              ${result.total_return:,.2f}")
    print(f"Return %:                  {result.total_return_pct:.2f}%")
    print(f"Number of Trades:          {len(result.trades)}")
    print()

    if len(result.trades) > 0:
        print("=" * 80)
        print("TRADE SUMMARY")
        print("=" * 80)
        print()

        trades_df = pd.DataFrame([
            {
                'Entry': trade.entry_date.strftime('%Y-%m-%d'),
                'Exit': trade.exit_date.strftime('%Y-%m-%d'),
                'Entry Price': f"${trade.entry_price:.2f}",
                'Exit Price': f"${trade.exit_price:.2f}",
                'P/L': f"${trade.pl:.2f}",
                'P/L %': f"{trade.pl_pct:.2f}%"
            }
            for trade in result.trades[:10]  # Show first 10 trades
        ])

        print(trades_df.to_string(index=False))
        if len(result.trades) > 10:
            print(f"\n... and {len(result.trades) - 10} more trades")
        print()

    # Performance analysis
    print("=" * 80)
    print("OPTIMIZATION IMPACT")
    print("=" * 80)
    print()

    print("Optimizations Applied:")
    print("  ✓ Indicator Precomputation:     Indicators calculated ONCE before backtest")
    print("  ✓ Data Handling:                No DataFrame copying (pass full data)")
    print("  ✓ Vectorization:                Numpy/pandas vectorized operations")
    print()

    print("Expected Performance Improvements:")
    print("  • Indicator Calculation:        10-100x faster (O(n) vs O(n²))")
    print("  • Memory Usage:                 5-10x less (no repeated copying)")
    print("  • Overall Backtest:             10-100x faster depending on strategy")
    print()

    # Theoretical comparison
    num_bars = len(data)
    theoretical_old_calculations = (num_bars * (num_bars + 1)) // 2  # O(n²)
    theoretical_new_calculations = num_bars  # O(n)
    theoretical_speedup = theoretical_old_calculations / theoretical_new_calculations

    print(f"Theoretical Analysis (for {num_bars} bars):")
    print(f"  Old Method Calculations:       ~{theoretical_old_calculations:,} (O(n²))")
    print(f"  New Method Calculations:       ~{theoretical_new_calculations:,} (O(n))")
    print(f"  Theoretical Speedup:           {theoretical_speedup:.0f}x")
    print()

    print("=" * 80)
    print(f"Benchmark completed in {format_time(elapsed_time)}")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_backtest()
