"""
Debug script to understand why BaseNewHighs strategy doesn't generate trades.
"""
from pathlib import Path
import pandas as pd

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from strategies.base_new_highs_strategy import BaseNewHighsStrategy


def debug_strategy():
    """Debug the strategy to see which rules are passing/failing."""

    # Load data
    data_loader = DataLoader(Path('raw_data/processed_exports'))
    data = data_loader.load_csv('AAPL', required_columns=['date', 'open', 'high', 'low', 'close'])

    print(f"Loaded {len(data)} bars for AAPL")

    # Create strategy
    strategy = BaseNewHighsStrategy()

    # Calculate indicators
    strategy._calculate_indicators(data)

    print("\nChecking indicators...")
    print(f"SMA 200: {strategy._sma_200.notna().sum()} non-null values")
    print(f"EMA Fast: {strategy._ema_fast.notna().sum()} non-null values")
    print(f"EMA Slow: {strategy._ema_slow.notna().sum()} non-null values")
    print(f"EMA Sell: {strategy._ema_sell.notna().sum()} non-null values")
    print(f"ATR: {strategy._atr.notna().sum()} non-null values")
    print(f"SAR: {strategy._sar.notna().sum()} non-null values")
    print(f"MA Crossovers detected: {len(strategy._ma_crossover_dates)}")

    # Check a few sample bars to see rule status
    print("\n" + "="*100)
    print("Sampling bars to check rule status:")
    print("="*100)

    # Sample every 100 bars starting from bar 220 (after warmup)
    sample_indices = list(range(220, len(data), 500))[:10]

    for idx in sample_indices:
        row = data.iloc[idx]
        date = row['date']
        close = row['close']

        print(f"\nBar {idx} - {date.strftime('%Y-%m-%d')} - Close: ${close:.2f}")

        # New High Rule
        historical_highs = data['high'].iloc[max(0, idx - strategy.new_high_n):idx]
        highest_high = historical_highs.max()
        new_high_rule = close > highest_high
        print(f"  New High Rule: {new_high_rule} (close ${close:.2f} vs highest ${highest_high:.2f})")

        # SMA 200 Rule
        sma_value = strategy._sma_200.iloc[idx]
        sma_200_rule = close >= sma_value if pd.notna(sma_value) else False
        print(f"  SMA 200 Rule: {sma_200_rule} (close ${close:.2f} vs SMA ${sma_value:.2f})")

        # MA Crossover Rule
        start_idx = max(0, idx - strategy.ma_lookback_k + 1)
        lookback_dates = data['date'].iloc[start_idx:idx+1].values
        ma_rule = any(date in strategy._ma_crossover_dates for date in lookback_dates)
        print(f"  MA Crossover Rule: {ma_rule}")

        # SAR Rule
        sar_value = strategy._sar.iloc[idx]
        sar_buy_rule = sar_value < close if pd.notna(sar_value) else False
        print(f"  SAR Buy Rule: {sar_buy_rule} (SAR ${sar_value:.2f} vs close ${close:.2f})")

        # All rules
        all_rules = new_high_rule and sma_200_rule and ma_rule and sar_buy_rule
        print(f"  --> ALL RULES: {all_rules}")

    # Find bars where ALL rules are true
    print("\n" + "="*100)
    print("Searching for bars where ALL entry rules are TRUE...")
    print("="*100)

    signals_found = 0
    for idx in range(220, len(data)):
        row = data.iloc[idx]
        close = row['close']

        # Check all rules
        historical_highs = data['high'].iloc[max(0, idx - strategy.new_high_n):idx]
        highest_high = historical_highs.max()
        new_high_rule = close > highest_high

        sma_value = strategy._sma_200.iloc[idx]
        sma_200_rule = close >= sma_value if pd.notna(sma_value) else False

        start_idx = max(0, idx - strategy.ma_lookback_k + 1)
        lookback_dates = data['date'].iloc[start_idx:idx+1].values
        ma_rule = any(date in strategy._ma_crossover_dates for date in lookback_dates)

        sar_value = strategy._sar.iloc[idx]
        sar_buy_rule = sar_value < close if pd.notna(sar_value) else False

        all_rules = new_high_rule and sma_200_rule and ma_rule and sar_buy_rule

        if all_rules:
            signals_found += 1
            if signals_found <= 10:  # Print first 10
                print(f"Bar {idx} - {row['date'].strftime('%Y-%m-%d')} - Close: ${close:.2f} - ALL RULES TRUE ✓")

    print(f"\nTotal bars where all entry rules are TRUE: {signals_found}")

    if signals_found == 0:
        print("\n⚠ WARNING: No bars found where all entry rules are satisfied!")
        print("   The strategy may be too restrictive. Consider:")
        print("   1. Reducing the number of required rules")
        print("   2. Adjusting parameter thresholds")
        print("   3. Testing on different securities or time periods")


if __name__ == "__main__":
    debug_strategy()
