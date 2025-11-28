"""
Debug script to trace signal generation in detail.
"""
from pathlib import Path
import pandas as pd

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Strategy.strategy_context import StrategyContext
from strategies.base_new_highs_strategy import BaseNewHighsStrategy


def debug_signals():
    """Debug signal generation."""

    # Load data
    data_loader = DataLoader(Path('raw_data/processed_exports'))
    data = data_loader.load_csv('AAPL', required_columns=['date', 'open', 'high', 'low', 'close'])

    print(f"Loaded {len(data)} bars for AAPL")

    # Create strategy
    strategy = BaseNewHighsStrategy()

    # Calculate indicators
    strategy._calculate_indicators(data)

    # Create mock contexts and test signal generation
    print("\n" + "="*100)
    print("Testing signal generation on bars where all rules should be TRUE...")
    print("="*100)

    # Test on bar 345 where we know all rules are TRUE
    test_indices = [345, 350, 351, 354, 643]

    for idx in test_indices:
        row = data.iloc[idx]

        # Create strategy context
        context = StrategyContext(
            data=data[:idx+1],  # Historical data up to current bar
            current_index=idx,
            current_price=row['close'],
            current_date=row['date'],
            position=None,  # No position
            available_capital=100000.0,
            total_equity=100000.0
        )

        # Generate signal
        signal = strategy.generate_signal(context)

        print(f"\nBar {idx} - {row['date'].strftime('%Y-%m-%d')} - Close: ${row['close']:.2f}")
        print(f"  Signal Type: {signal.type}")
        print(f"  Signal Reason: {signal.reason if signal.reason else 'N/A'}")

        if signal.type == 'BUY':
            print(f"  Signal Size: {signal.size}")
            print(f"  Stop Loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "  Stop Loss: None")

            # Test position sizing
            shares = strategy.position_size(context, signal)
            print(f"  Position Size: {shares:.2f} shares (${shares * row['close']:.2f})")

    print("\n" + "="*100)
    print("Checking min_required_bars threshold...")
    print("="*100)

    min_required_bars = max(
        strategy.sma_length,
        strategy.new_high_n,
        strategy.ma_lookback_k,
        strategy.atr_length
    )
    print(f"min_required_bars = max({strategy.sma_length}, {strategy.new_high_n}, {strategy.ma_lookback_k}, {strategy.atr_length}) = {min_required_bars}")
    print(f"\nBars where signals were found: {test_indices}")
    print(f"All test bars > {min_required_bars}? {all(idx >= min_required_bars for idx in test_indices)}")


if __name__ == "__main__":
    debug_signals()
