"""Quick test to debug a single bar."""
from pathlib import Path
import pandas as pd

from Classes.Data.data_loader import DataLoader
from Classes.Strategy.strategy_context import StrategyContext
from strategies.base_new_highs_strategy import BaseNewHighsStrategy

# Load data
data_loader = DataLoader(Path('raw_data/processed_exports'))
data = data_loader.load_csv('AAPL')

# Create strategy
strategy = BaseNewHighsStrategy()

# Test on bar 345 where we know all rules should pass
test_idx = 345
historical_data = data.iloc[:test_idx+1]

context = StrategyContext(
    data=historical_data,
    current_index=test_idx,
    current_price=historical_data['close'].iloc[-1],
    current_date=historical_data['date'].iloc[-1],
    position=None,
    available_capital=100000.0,
    total_equity=100000.0
)

print(f"Testing bar {test_idx}")
print(f"Date: {context.current_date}")
print(f"Close: ${context.current_price:.2f}")
print(f"Data length: {len(context.data)}")

# Calculate indicators
indicators = strategy._calculate_current_indicators(context.data)

print(f"\nIndicators:")
for key, value in indicators.items():
    print(f"  {key}: {value}")

# Check rules
print(f"\nRule checks:")

# New High
historical_highs = context.data['high'].iloc[-(strategy.new_high_n+1):-1]
new_high_rule = context.current_price > historical_highs.max()
print(f"  New High: {new_high_rule} (${context.current_price:.2f} > ${historical_highs.max():.2f})")

# SMA 200
sma_200_rule = not pd.isna(indicators['sma_200']) and context.current_price >= indicators['sma_200']
print(f"  SMA 200: {sma_200_rule} (${context.current_price:.2f} >= ${indicators['sma_200']:.2f})")

# MA Crossover
print(f"  MA Crossover: {indicators['ma_crossed_recently']}")

# SAR
sar_buy_rule = not pd.isna(indicators['sar']) and indicators['sar'] < context.current_price
print(f"  SAR Buy: {sar_buy_rule} (SAR ${indicators['sar']:.2f} < ${context.current_price:.2f})")

# Generate signal
signal = strategy.generate_signal(context)
print(f"\nGenerated Signal: {signal.type}")
if signal.reason:
    print(f"Reason: {signal.reason}")
