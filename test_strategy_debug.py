"""
Quick test to debug why AlphaTrend strategy isn't generating trades
"""
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.alphatrend_strategy import AlphaTrendStrategy
from Classes.Engine.single_security_engine import SingleSecurityBacktest
from Classes.Config.config import BacktestConfig

def main():
    print("=" * 80)
    print("DEBUG TEST: AlphaTrend Strategy")
    print("=" * 80)
    print()

    # Load data
    data_file = "raw_data/AAPL.csv"
    print(f"Loading data from {data_file}...")

    try:
        data = pd.read_csv(data_file)
        print(f"✓ Loaded {len(data)} bars")
        print(f"  Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
        print()
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Check required columns
    required = ['date', 'open', 'high', 'low', 'close', 'volume', 'atr_14', 'ema_50']
    missing = [col for col in required if col not in data.columns]
    if missing:
        print(f"✗ Missing required columns: {missing}")
        print(f"  Available columns: {list(data.columns)}")
        return

    print(f"✓ All required columns present")
    print()

    # Create strategy with default parameters
    print("Creating strategy...")
    strategy = AlphaTrendStrategy(
        volume_alignment_window=14,
        stop_loss_percent=2.0,
        grace_period_bars=14
    )
    print(f"✓ Strategy created with volume_alignment_window={strategy.volume_alignment_window}")
    print()

    # Create backtest config
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_pct=0.1
    )

    # Run backtest
    print("Running backtest with DEBUG logging...")
    print("=" * 80)
    print()

    try:
        engine = SingleSecurityBacktest(strategy, config)
        result = engine.run(data, symbol="AAPL")

        print()
        print("=" * 80)
        print("BACKTEST COMPLETE")
        print("=" * 80)
        print(f"Total Trades: {len(result.trades)}")
        print(f"Final Equity: ${result.final_equity:,.2f}")

        if len(result.trades) > 0:
            print(f"\nFirst 5 trades:")
            for i, trade in enumerate(result.trades[:5]):
                print(f"  {i+1}. Entry: {trade.entry_date}, Exit: {trade.exit_date}, P/L: ${trade.pl:,.2f}")
        else:
            print("\n⚠️  NO TRADES EXECUTED!")
            print("Check the debug output above to see why.")

    except Exception as e:
        print(f"\n✗ Error running backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
