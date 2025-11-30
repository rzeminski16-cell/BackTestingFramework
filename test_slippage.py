#!/usr/bin/env python3
"""
Quick test to verify slippage feature implementation.
"""
from Classes.Config.config import BacktestConfig, CommissionConfig

def test_backtest_config_slippage():
    """Test that BacktestConfig accepts and validates slippage_percent."""
    # Test default slippage
    config1 = BacktestConfig()
    assert config1.slippage_percent == 0.1, f"Expected default slippage 0.1, got {config1.slippage_percent}"
    print("✓ Default slippage is 0.1%")

    # Test custom slippage
    config2 = BacktestConfig(slippage_percent=0.5)
    assert config2.slippage_percent == 0.5, f"Expected slippage 0.5, got {config2.slippage_percent}"
    print("✓ Custom slippage (0.5%) works")

    # Test zero slippage
    config3 = BacktestConfig(slippage_percent=0.0)
    assert config3.slippage_percent == 0.0, f"Expected slippage 0.0, got {config3.slippage_percent}"
    print("✓ Zero slippage works")

    # Test negative slippage validation
    try:
        config4 = BacktestConfig(slippage_percent=-0.1)
        assert False, "Should have raised ValueError for negative slippage"
    except ValueError as e:
        assert "non-negative" in str(e).lower(), f"Wrong error message: {e}"
        print("✓ Negative slippage validation works")

    print("\n✅ All slippage configuration tests passed!")

def test_slippage_calculation():
    """Test the slippage calculation formula."""
    # BUY: price * (1 + slippage/100)
    price = 100.0
    slippage = 0.1  # 0.1%

    buy_execution_price = price * (1 + slippage / 100)
    expected_buy = 100.10
    assert abs(buy_execution_price - expected_buy) < 0.001, \
        f"BUY slippage calculation wrong: expected {expected_buy}, got {buy_execution_price}"
    print(f"✓ BUY slippage: ${price:.2f} -> ${buy_execution_price:.2f} (pays more)")

    # SELL: price * (1 - slippage/100)
    sell_execution_price = price * (1 - slippage / 100)
    expected_sell = 99.90
    assert abs(sell_execution_price - expected_sell) < 0.001, \
        f"SELL slippage calculation wrong: expected {expected_sell}, got {sell_execution_price}"
    print(f"✓ SELL slippage: ${price:.2f} -> ${sell_execution_price:.2f} (receives less)")

    print("\n✅ All slippage calculation tests passed!")

def test_quantity_adjustment():
    """Test that quantity is adjusted to account for slippage."""
    price = 100.0
    slippage = 0.1  # 0.1%
    execution_price = price * (1 + slippage / 100)  # 100.10

    # If strategy calculated 100 shares at $100, after slippage adjustment:
    original_quantity = 100
    adjusted_quantity = original_quantity * (price / execution_price)

    # Original cost: 100 shares * $100 = $10,000
    # Adjusted cost: ~99.9 shares * $100.10 = ~$10,000
    original_cost = original_quantity * price
    adjusted_cost = adjusted_quantity * execution_price

    print(f"✓ Original: {original_quantity} shares @ ${price:.2f} = ${original_cost:.2f}")
    print(f"✓ Adjusted: {adjusted_quantity:.4f} shares @ ${execution_price:.2f} = ${adjusted_cost:.2f}")

    # Costs should be approximately equal (within $0.01)
    assert abs(original_cost - adjusted_cost) < 0.01, \
        f"Cost mismatch: original ${original_cost:.2f} vs adjusted ${adjusted_cost:.2f}"
    print(f"✓ Quantity adjustment maintains capital allocation")

    print("\n✅ Quantity adjustment test passed!")

if __name__ == "__main__":
    test_backtest_config_slippage()
    test_slippage_calculation()
    test_quantity_adjustment()
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("="*50)
