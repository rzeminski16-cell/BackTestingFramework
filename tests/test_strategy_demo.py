"""
Demonstration test script for strategy testing.

This script demonstrates how to test key strategy functionality and
identifies potential problems in the strategy system.

Run with: python tests/test_strategy_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from Classes
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from Classes.Strategy.base_strategy import BaseStrategy, StrategyValidationError
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal, SignalType
from Classes.Models.trade_direction import TradeDirection
from strategies.base_alphatrend_strategy import BaseAlphaTrendStrategy


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def generate_test_ohlcv(bars: int = 200, start_price: float = 100.0,
                        volatility: float = 0.02) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Args:
        bars: Number of bars to generate
        start_price: Starting price
        volatility: Daily volatility (as fraction)

    Returns:
        DataFrame with OHLCV data
    """
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(bars)]
    np.random.seed(42)  # Reproducible

    # Generate prices with random walk
    returns = np.random.normal(0, volatility, bars)
    prices = start_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    data = {
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, bars)),
        'high': prices * (1 + np.random.uniform(0.001, 0.01, bars)),
        'low': prices * (1 + np.random.uniform(-0.01, -0.001, bars)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, bars)
    }

    return pd.DataFrame(data)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add required indicators for AlphaTrend strategy.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with indicators added
    """
    # Calculate ATR (simplified)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14_atr'] = true_range.rolling(window=14).mean()

    # Calculate MFI (simplified - using volume as proxy)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']

    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

    positive_mf = pd.Series(positive_flow).rolling(window=14).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=14).sum()

    mfi_ratio = positive_mf / (negative_mf + 1e-10)  # Avoid division by zero
    df['mfi_14_mfi'] = 100 - (100 / (1 + mfi_ratio))

    return df


# ============================================================================
# TEST CASES
# ============================================================================

def test_strategy_validation():
    """Test that strategy validation catches missing implementations."""
    print("\n" + "="*70)
    print("TEST: Strategy Validation")
    print("="*70)

    class IncompleteStrategy(BaseStrategy):
        """Strategy missing required methods."""
        @property
        def trade_direction(self):
            return TradeDirection.LONG

        def required_columns(self):
            return ['date', 'close']

        # Missing: generate_entry_signal, calculate_initial_stop_loss,
        #          position_size, generate_exit_signal

    try:
        strategy = IncompleteStrategy()
        print("❌ FAIL: Should have raised StrategyValidationError")
    except (NotImplementedError, TypeError) as e:
        print("✅ PASS: Correctly caught incomplete strategy")
        print(f"   Error: {type(e).__name__}")


def test_parameter_validation():
    """Test that invalid parameters are handled."""
    print("\n" + "="*70)
    print("TEST: Parameter Validation")
    print("="*70)

    test_cases = [
        ("Negative ATR multiplier", {'atr_multiplier': -1.0}),
        ("Zero ATR multiplier", {'atr_multiplier': 0.0}),
        ("Risk > 100%", {'risk_percent': 150.0}),
        ("Negative risk", {'risk_percent': -5.0}),
        ("Zero max hold days", {'max_hold_days': 0}),
        ("Negative max hold days", {'max_hold_days': -5}),
    ]

    for test_name, params in test_cases:
        try:
            strategy = BaseAlphaTrendStrategy(**params)
            print(f"❌ FAIL: {test_name} - should have raised ValueError")
            print(f"   Created strategy with params: {params}")
        except (ValueError, AssertionError) as e:
            print(f"✅ PASS: {test_name} - correctly rejected")
        except Exception as e:
            print(f"⚠️  WARN: {test_name} - unexpected error: {type(e).__name__}")


def test_warmup_period():
    """Test strategy behavior with insufficient warmup data."""
    print("\n" + "="*70)
    print("TEST: Warmup Period Handling")
    print("="*70)

    # Create data with only 50 bars (less than percentile_period=100)
    short_data = generate_test_ohlcv(bars=50)
    short_data = add_indicators(short_data)

    strategy = BaseAlphaTrendStrategy(percentile_period=100)

    # Prepare data
    try:
        prepared_data = strategy.prepare_data(short_data)
        print(f"✅ prepare_data() succeeded with {len(short_data)} bars")
    except Exception as e:
        print(f"❌ FAIL: prepare_data() raised error: {e}")
        return

    # Test entry signal generation at various indices
    signals_generated = []
    for i in range(len(prepared_data)):
        context = StrategyContext(
            data=prepared_data.iloc[:i+1],
            current_index=i,
            current_price=prepared_data.iloc[i]['close'],
            current_date=prepared_data.iloc[i]['date'],
            position=None,
            available_capital=10000,
            total_equity=10000,
            symbol='TEST'
        )

        signal = strategy.generate_entry_signal(context)
        if signal is not None and signal.type == SignalType.BUY:
            signals_generated.append(i)

    if len(signals_generated) == 0:
        print(f"✅ PASS: No signals generated with insufficient warmup")
        print(f"   (percentile_period=100, data_length={len(short_data)})")
    else:
        print(f"⚠️  WARN: Signals generated despite insufficient warmup")
        print(f"   Signals at bars: {signals_generated}")


def test_stop_loss_calculation():
    """Test stop loss calculation and validation."""
    print("\n" + "="*70)
    print("TEST: Stop Loss Calculation")
    print("="*70)

    data = generate_test_ohlcv(bars=200)
    data = add_indicators(data)

    strategy = BaseAlphaTrendStrategy(atr_multiplier=2.0)
    prepared_data = strategy.prepare_data(data)

    # Test at bar 150 (after warmup)
    bar_idx = 150
    context = StrategyContext(
        data=prepared_data.iloc[:bar_idx+1],
        current_index=bar_idx,
        current_price=prepared_data.iloc[bar_idx]['close'],
        current_date=prepared_data.iloc[bar_idx]['date'],
        position=None,
        available_capital=10000,
        total_equity=10000,
        symbol='TEST'
    )

    try:
        stop_loss = strategy.calculate_initial_stop_loss(context)
        entry_price = context.current_price
        atr = context.get_indicator_value('atr_14')

        print(f"Entry Price: ${entry_price:.2f}")
        print(f"ATR(14): ${atr:.2f}")
        print(f"Stop Loss: ${stop_loss:.2f}")

        # Validate stop loss is below entry (LONG strategy)
        if stop_loss < entry_price:
            stop_distance = entry_price - stop_loss
            expected_distance = atr * strategy.atr_multiplier
            if abs(stop_distance - expected_distance) < 0.01:
                print(f"✅ PASS: Stop loss correctly calculated")
                print(f"   Distance: ${stop_distance:.2f} (expected ${expected_distance:.2f})")
            else:
                print(f"❌ FAIL: Stop distance incorrect")
                print(f"   Got ${stop_distance:.2f}, expected ${expected_distance:.2f}")
        else:
            print(f"❌ FAIL: Stop loss above entry price for LONG strategy")

    except Exception as e:
        print(f"❌ FAIL: Stop loss calculation raised error: {e}")


def test_position_sizing():
    """Test risk-based position sizing calculation."""
    print("\n" + "="*70)
    print("TEST: Position Sizing")
    print("="*70)

    data = generate_test_ohlcv(bars=200)
    data = add_indicators(data)

    strategy = BaseAlphaTrendStrategy(
        atr_multiplier=2.0,
        risk_percent=2.0  # Risk 2% of equity per trade
    )
    prepared_data = strategy.prepare_data(data)

    bar_idx = 150
    equity = 10000.0
    context = StrategyContext(
        data=prepared_data.iloc[:bar_idx+1],
        current_index=bar_idx,
        current_price=prepared_data.iloc[bar_idx]['close'],
        current_date=prepared_data.iloc[bar_idx]['date'],
        position=None,
        available_capital=equity,
        total_equity=equity,
        symbol='TEST',
        fx_rate=1.0
    )

    entry_price = context.current_price
    stop_loss = strategy.calculate_initial_stop_loss(context)
    signal = Signal.buy(size=1.0, stop_loss=stop_loss, direction=TradeDirection.LONG)

    try:
        position_size = strategy.position_size(context, signal)

        print(f"Equity: ${equity:.2f}")
        print(f"Risk%: {strategy.risk_percent}%")
        print(f"Entry: ${entry_price:.2f}")
        print(f"Stop: ${stop_loss:.2f}")
        print(f"Position Size: {position_size:.2f} shares")

        # Calculate expected size
        risk_amount = equity * (strategy.risk_percent / 100)  # $200
        stop_distance = entry_price - stop_loss
        expected_size = risk_amount / stop_distance

        print(f"Risk Amount: ${risk_amount:.2f}")
        print(f"Stop Distance: ${stop_distance:.2f}")
        print(f"Expected Size: {expected_size:.2f} shares")

        if abs(position_size - expected_size) < 0.01:
            print(f"✅ PASS: Position size correctly calculated")
        else:
            print(f"❌ FAIL: Position size mismatch")

    except Exception as e:
        print(f"❌ FAIL: Position sizing raised error: {e}")


def test_invalid_stop_loss():
    """Test handling of invalid stop loss (stop above entry for LONG)."""
    print("\n" + "="*70)
    print("TEST: Invalid Stop Loss Handling")
    print("="*70)

    data = generate_test_ohlcv(bars=200)
    data = add_indicators(data)

    strategy = BaseAlphaTrendStrategy()
    prepared_data = strategy.prepare_data(data)

    bar_idx = 150
    entry_price = prepared_data.iloc[bar_idx]['close']

    context = StrategyContext(
        data=prepared_data.iloc[:bar_idx+1],
        current_index=bar_idx,
        current_price=entry_price,
        current_date=prepared_data.iloc[bar_idx]['date'],
        position=None,
        available_capital=10000,
        total_equity=10000,
        symbol='TEST'
    )

    # Create signal with invalid stop loss (above entry for LONG)
    invalid_stop = entry_price * 1.1  # 10% above entry (invalid for LONG)
    signal = Signal.buy(size=1.0, stop_loss=invalid_stop, direction=TradeDirection.LONG)

    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Invalid Stop: ${invalid_stop:.2f} (above entry)")

    try:
        position_size = strategy.position_size(context, signal)
        print(f"⚠️  WARN: Position sizing accepted invalid stop loss")
        print(f"   Returned size: {position_size:.2f} shares")
        print(f"   Should have raised ValueError or logged critical warning")
    except ValueError as e:
        print(f"✅ PASS: Correctly rejected invalid stop loss")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {type(e).__name__}: {e}")


def test_required_columns_validation():
    """Test that missing required columns are detected."""
    print("\n" + "="*70)
    print("TEST: Required Columns Validation")
    print("="*70)

    # Create data missing required indicators
    incomplete_data = generate_test_ohlcv(bars=200)
    # Missing: atr_14_atr, mfi_14_mfi

    strategy = BaseAlphaTrendStrategy()

    required = strategy.required_columns()
    available = list(incomplete_data.columns)
    missing = set(required) - set(available)

    print(f"Required columns: {required}")
    print(f"Available columns: {available}")
    print(f"Missing columns: {list(missing)}")

    try:
        prepared_data = strategy.prepare_data(incomplete_data)
        print(f"❌ FAIL: Should have raised ValueError for missing columns")
    except ValueError as e:
        if 'Missing required columns' in str(e):
            print(f"✅ PASS: Correctly detected missing columns")
            print(f"   Error message: {str(e)[:100]}...")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {type(e).__name__}: {e}")


def test_signal_generation_full_flow():
    """Test full signal generation flow from entry to exit."""
    print("\n" + "="*70)
    print("TEST: Full Signal Generation Flow")
    print("="*70)

    data = generate_test_ohlcv(bars=200)
    data = add_indicators(data)

    strategy = BaseAlphaTrendStrategy(
        atr_multiplier=2.0,
        risk_percent=2.0,
        max_hold_days=10
    )
    prepared_data = strategy.prepare_data(data)

    print(f"Testing with {len(prepared_data)} bars of data")
    print(f"Strategy: {strategy}")

    entry_signals = []
    exit_signals = []

    # Simulate bar-by-bar
    for i in range(len(prepared_data)):
        context = StrategyContext(
            data=prepared_data.iloc[:i+1],
            current_index=i,
            current_price=prepared_data.iloc[i]['close'],
            current_date=prepared_data.iloc[i]['date'],
            position=None,
            available_capital=10000,
            total_equity=10000,
            symbol='TEST'
        )

        signal = strategy.generate_signal(context)

        if signal.type == SignalType.BUY:
            entry_signals.append((i, signal))
        elif signal.type == SignalType.SELL:
            exit_signals.append((i, signal))

    print(f"\nEntry Signals: {len(entry_signals)}")
    if len(entry_signals) > 0:
        print(f"  First entry at bar {entry_signals[0][0]}: {entry_signals[0][1].reason}")
        if len(entry_signals) > 1:
            print(f"  Second entry at bar {entry_signals[1][0]}: {entry_signals[1][1].reason}")

    print(f"Exit Signals: {len(exit_signals)}")

    if len(entry_signals) > 0:
        print(f"✅ PASS: Strategy generated entry signals")
    else:
        print(f"⚠️  WARN: No entry signals generated (could be valid if no setup)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all demonstration tests."""
    print("\n" + "="*70)
    print("STRATEGY TESTING DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates key test cases for the strategy system.")
    print("It identifies potential problems and edge cases.\n")

    tests = [
        test_strategy_validation,
        test_parameter_validation,
        test_warmup_period,
        test_stop_loss_calculation,
        test_position_sizing,
        test_invalid_stop_loss,
        test_required_columns_validation,
        test_signal_generation_full_flow,
    ]

    passed = 0
    failed = 0
    warnings = 0

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {test_func.__name__}")
            print(f"   Error: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests completed. See individual test output for PASS/FAIL status.")
    print(f"\nReview the output above to identify:")
    print(f"  ✅ PASS: Working correctly")
    print(f"  ❌ FAIL: Broken functionality")
    print(f"  ⚠️  WARN: Potential issues or missing validation")
    print("\nFor detailed analysis, see docs/STRATEGY_TESTING_ANALYSIS.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
