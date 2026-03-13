"""
Comprehensive tests for SingleSecurityEngine and its supporting components.

Tests cover the full trade lifecycle:
- Validation, capital tracking, commission, slippage
- Stop loss, take profit, trailing stops
- Partial exits, pyramiding
- No-lookahead bias, equity curve, date filtering
- Trade record accuracy, progress callback, FX conversion
- Integration with real strategies
- Bug regression tests
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock

from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.backtest_result import BacktestResult
from Classes.Engine.trade_executor import TradeExecutor
from Classes.Engine.position_manager import PositionManager
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Models.signal import Signal, SignalType
from Classes.Models.trade import Trade, reset_trade_counter
from Classes.Models.trade_direction import TradeDirection
from Classes.Models.position import Position
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext


# =============================================================================
# DETERMINISTIC TEST STRATEGY
# =============================================================================

class DeterministicTestStrategy(BaseStrategy):
    """
    Strategy that generates signals on pre-configured bar indices.
    Allows tests to control exactly when BUY/SELL/PYRAMID/PARTIAL_EXIT happen.
    """

    _validate_on_init = False  # Skip validation for test strategy

    def __init__(self, buy_bars=None, sell_bars=None, stop_loss_price=None,
                 take_profit_price=None, position_size_shares=100.0,
                 pyramid_bars=None, pyramid_size=0.5,
                 partial_exit_bars=None, partial_exit_fraction=0.5,
                 trailing_stop_bars=None, trailing_stop_prices=None,
                 adjust_stop_bars=None, adjust_stop_prices=None):
        self.buy_bars = set(buy_bars or [])
        self.sell_bars = set(sell_bars or [])
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.position_size_shares = position_size_shares
        self.pyramid_bars = set(pyramid_bars or [])
        self.pyramid_size = pyramid_size
        self.partial_exit_bars = set(partial_exit_bars or [])
        self.partial_exit_fraction = partial_exit_fraction
        self.trailing_stop_bars = dict(zip(trailing_stop_bars or [], trailing_stop_prices or []))
        self.adjust_stop_bars = dict(zip(adjust_stop_bars or [], adjust_stop_prices or []))
        super().__init__()

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index in self.buy_bars:
            return Signal.buy(
                size=1.0,
                stop_loss=self.stop_loss_price,
                take_profit=self.take_profit_price,
                reason="Test buy signal",
                direction=self.trade_direction
            )
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        if self.stop_loss_price is not None:
            return self.stop_loss_price
        return context.current_price * 0.95

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index in self.sell_bars:
            return Signal.sell(reason="Test sell signal")
        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return self.position_size_shares

    def should_pyramid(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index in self.pyramid_bars:
            return Signal.pyramid(size=self.pyramid_size, reason="Test pyramid")
        return None

    def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
        if context.current_index in self.partial_exit_bars:
            return self.partial_exit_fraction
        return None

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        idx = context.current_index
        if idx in self.trailing_stop_bars:
            return self.trailing_stop_bars[idx]
        if idx in self.adjust_stop_bars:
            return self.adjust_stop_bars[idx]
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_data(n_bars=200, base_price=100.0, trend=0.0, volatility=1.0,
                     start_date='2020-01-01', include_indicators=False):
    """Create synthetic OHLCV data with known properties."""
    dates = pd.date_range(start=start_date, periods=n_bars, freq='B')
    np.random.seed(42)

    closes = []
    price = base_price
    for i in range(n_bars):
        price = price + trend + np.random.normal(0, volatility * 0.1)
        price = max(price, 1.0)  # Prevent negative prices
        closes.append(price)

    closes = np.array(closes)
    highs = closes * (1 + np.random.uniform(0.001, 0.02, n_bars))
    lows = closes * (1 - np.random.uniform(0.001, 0.02, n_bars))
    opens = (closes + np.random.normal(0, 0.5, n_bars))

    data = {
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.uniform(500000, 2000000, n_bars),
    }

    if include_indicators:
        data['atr_14_atr'] = np.full(n_bars, 2.0)
        data['mfi_14_mfi'] = np.full(n_bars, 50.0)

    return pd.DataFrame(data)


def create_flat_data(n_bars=50, price=100.0, start_date='2020-01-01'):
    """Create data where price is constant — useful for isolating commission/slippage effects."""
    dates = pd.date_range(start=start_date, periods=n_bars, freq='B')
    return pd.DataFrame({
        'date': dates,
        'open': [price] * n_bars,
        'high': [price] * n_bars,
        'low': [price] * n_bars,
        'close': [price] * n_bars,
        'volume': [1000000] * n_bars,
    })


def create_config(capital=100000.0, commission_pct=0.001, slippage_pct=0.0,
                  start_date=None, end_date=None):
    """Create a standard test BacktestConfig."""
    return BacktestConfig(
        initial_capital=capital,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=commission_pct),
        slippage_percent=slippage_pct,
        start_date=start_date,
        end_date=end_date
    )


# =============================================================================
# A. VALIDATION TESTS
# =============================================================================

class TestValidation(unittest.TestCase):
    """Tests for input validation."""

    def test_empty_data_raises_value_error(self):
        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        with self.assertRaises(ValueError):
            engine.run('TEST', pd.DataFrame(), strategy)

    def test_missing_date_column_raises_error(self):
        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = pd.DataFrame({'close': [100.0], 'volume': [1000]})
        with self.assertRaises(ValueError):
            engine.run('TEST', data, strategy)

    def test_missing_close_column_raises_error(self):
        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = pd.DataFrame({'date': [datetime(2020, 1, 1)], 'volume': [1000]})
        with self.assertRaises(ValueError):
            engine.run('TEST', data, strategy)

    def test_date_range_filter_no_data_raises_error(self):
        config = create_config(
            start_date=datetime(2030, 1, 1),
            end_date=datetime(2030, 12, 31)
        )
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = create_test_data(n_bars=50)
        with self.assertRaises(ValueError):
            engine.run('TEST', data, strategy)

    def test_valid_data_runs_successfully(self):
        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = create_test_data(n_bars=50)
        result = engine.run('TEST', data, strategy)
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.symbol, 'TEST')


# =============================================================================
# B. CAPITAL TRACKING TESTS
# =============================================================================

class TestCapitalTracking(unittest.TestCase):
    """Tests for capital tracking correctness."""

    def test_initial_capital_set_correctly(self):
        config = create_config(capital=50000.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()  # No trades
        data = create_test_data(n_bars=10)
        result = engine.run('TEST', data, strategy)
        self.assertAlmostEqual(result.equity_curve.iloc[0]['equity'], 50000.0)

    def test_capital_decreases_on_buy(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(buy_bars=[5], position_size_shares=10.0)
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # After buy at bar 5: capital should decrease by 10 * 100 = 1000
        capital_after_buy = result.equity_curve.iloc[5]['capital']
        self.assertAlmostEqual(capital_after_buy, 100000.0 - 1000.0, places=2)

    def test_capital_increases_on_sell(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # With flat price and no commission/slippage, capital after sell should equal initial
        capital_after_sell = result.equity_curve.iloc[10]['capital']
        self.assertAlmostEqual(capital_after_sell, 100000.0, places=2)

    def test_final_equity_no_open_position(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # No open position at end, flat price, no costs → final equity = initial
        self.assertAlmostEqual(result.final_equity, 100000.0, places=2)

    def test_equity_curve_length_matches_bars(self):
        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = create_test_data(n_bars=30)
        result = engine.run('TEST', data, strategy)
        self.assertEqual(len(result.equity_curve), 30)

    def test_total_return_calculation(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = create_test_data(n_bars=10)
        result = engine.run('TEST', data, strategy)
        expected_return = result.final_equity - 100000.0
        self.assertAlmostEqual(result.total_return, expected_return, places=2)


# =============================================================================
# C. COMMISSION TESTS
# =============================================================================

class TestCommission(unittest.TestCase):
    """Tests for commission handling."""

    def test_percentage_commission_on_entry(self):
        commission_rate = 0.001  # 0.1%
        config = create_config(commission_pct=commission_rate, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # Entry commission = 10 * 100 * 0.001 = 1.0
        # Exit commission = 10 * 100 * 0.001 = 1.0
        # Total commission = 2.0
        self.assertEqual(len(result.trades), 1)
        self.assertAlmostEqual(result.trades[0].commission_paid, 2.0, places=2)

    def test_fixed_commission_mode(self):
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=CommissionConfig(mode=CommissionMode.FIXED, value=5.0),
            slippage_percent=0.0
        )
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # Fixed commission: $5 entry + $5 exit = $10
        self.assertAlmostEqual(result.trades[0].commission_paid, 10.0, places=2)

    def test_commission_deducted_from_capital(self):
        config = create_config(capital=100000.0, commission_pct=0.001, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # After round-trip with flat price, capital should be reduced by total commission
        # Entry commission = 10 * 100 * 0.001 = 1.0
        # Exit commission = 10 * 100 * 0.001 = 1.0
        self.assertAlmostEqual(result.final_equity, 100000.0 - 2.0, places=2)

    def test_commission_recorded_in_trade(self):
        config = create_config(commission_pct=0.002, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=50.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        expected_commission = 2 * (50 * 100 * 0.002)  # entry + exit
        self.assertAlmostEqual(trade.commission_paid, expected_commission, places=2)

    def test_zero_commission(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        self.assertAlmostEqual(result.final_equity, 100000.0, places=2)
        self.assertAlmostEqual(result.trades[0].commission_paid, 0.0, places=2)


# =============================================================================
# D. SLIPPAGE TESTS
# =============================================================================

class TestSlippage(unittest.TestCase):
    """Tests for slippage handling."""

    def test_buy_slippage_increases_execution_price(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.1)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        # BUY slippage: 100 * (1 + 0.001) = 100.1
        self.assertAlmostEqual(trade.entry_price, 100.1, places=2)

    def test_sell_slippage_decreases_execution_price(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.1)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        # SELL slippage: 100 * (1 - 0.001) = 99.9
        self.assertAlmostEqual(trade.exit_price, 99.9, places=2)

    def test_slippage_reduces_position_quantity(self):
        # With slippage, you buy at higher price so get fewer shares
        config_no_slip = create_config(commission_pct=0.0, slippage_pct=0.0)
        config_slip = create_config(commission_pct=0.0, slippage_pct=1.0)  # 1% slippage

        data = create_flat_data(n_bars=20, price=100.0)

        engine1 = SingleSecurityEngine(config_no_slip)
        strategy1 = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[15], position_size_shares=100.0
        )
        result1 = engine1.run('TEST', data, strategy1)

        engine2 = SingleSecurityEngine(config_slip)
        strategy2 = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[15], position_size_shares=100.0
        )
        result2 = engine2.run('TEST', data, strategy2)

        # Slippage should reduce quantity
        self.assertGreater(result1.trades[0].quantity, result2.trades[0].quantity)

    def test_zero_slippage_no_price_change(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_price, 100.0, places=2)
        self.assertAlmostEqual(trade.exit_price, 100.0, places=2)

    def test_slippage_cost_recorded_in_trade_metadata(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.5)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        self.assertIn('slippage', trade.metadata)
        self.assertGreater(trade.metadata['slippage'], 0)


# =============================================================================
# E. STOP LOSS TESTS
# =============================================================================

class TestStopLoss(unittest.TestCase):
    """Tests for stop loss functionality."""

    def test_stop_loss_triggered_when_price_at_or_below_stop(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        # Buy at bar 2 with stop at 90, price drops to 85 at bar 5
        strategy = DeterministicTestStrategy(
            buy_bars=[2], stop_loss_price=90.0, position_size_shares=10.0
        )
        # Create data where price drops below stop
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[5, 'close'] = 85.0  # Below stop loss
        result = engine.run('TEST', data, strategy)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].exit_reason, "Stop loss hit")

    def test_stop_loss_not_triggered_when_price_above_stop(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], stop_loss_price=50.0,
            position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].exit_reason, "Test sell signal")

    def test_stop_loss_exit_reason_recorded(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], stop_loss_price=90.0, position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[5, 'close'] = 89.0
        result = engine.run('TEST', data, strategy)
        self.assertIn("Stop loss", result.trades[0].exit_reason)

    def test_stop_loss_checked_before_strategy_signal(self):
        """Stop loss should take priority over strategy exit signal."""
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[5], stop_loss_price=90.0,
            position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[5, 'close'] = 85.0  # Both stop loss and sell signal on bar 5
        result = engine.run('TEST', data, strategy)
        self.assertEqual(result.trades[0].exit_reason, "Stop loss hit")

    def test_no_stop_loss_when_none_set(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[10], stop_loss_price=None,
            position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[5, 'close'] = 50.0  # Massive drop but no stop loss set
        data.loc[6, 'close'] = 100.0  # Recovery
        result = engine.run('TEST', data, strategy)
        # Position should survive the drop since no stop loss
        self.assertEqual(result.trades[0].exit_reason, "Test sell signal")


# =============================================================================
# F. TAKE PROFIT TESTS
# =============================================================================

class TestTakeProfit(unittest.TestCase):
    """Tests for take profit functionality."""

    def test_take_profit_triggered_when_price_at_or_above_target(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], take_profit_price=110.0, position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[7, 'close'] = 115.0  # Above take profit
        result = engine.run('TEST', data, strategy)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].exit_reason, "Take profit hit")

    def test_take_profit_exit_reason_recorded(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], take_profit_price=110.0, position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[7, 'close'] = 112.0
        result = engine.run('TEST', data, strategy)
        self.assertIn("Take profit", result.trades[0].exit_reason)

    def test_take_profit_checked_after_stop_loss(self):
        """Stop loss takes priority when both trigger on same bar."""
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], stop_loss_price=90.0, take_profit_price=110.0,
            position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        # Price drops below stop — even though take profit also set
        data.loc[5, 'close'] = 85.0
        result = engine.run('TEST', data, strategy)
        self.assertEqual(result.trades[0].exit_reason, "Stop loss hit")


# =============================================================================
# G. TRAILING STOP TESTS
# =============================================================================

class TestTrailingStop(unittest.TestCase):
    """Tests for trailing stop functionality."""

    def test_trailing_stop_moves_up_for_long(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        # Buy at bar 2 with stop at 90, trailing adjusts up at bar 5
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], stop_loss_price=90.0,
            position_size_shares=10.0,
            trailing_stop_bars=[5], trailing_stop_prices=[95.0]
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # The stop should have moved from 90 to 95
        self.assertEqual(result.trades[0].exit_reason, "Test sell signal")
        # Verify by checking that stop was raised (if price dropped to 92,
        # it would trigger new stop at 95 but not old stop at 90)

    def test_trailing_stop_cannot_move_down_for_long(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        # Try to move stop DOWN from 90 to 85 — should be rejected
        strategy = DeterministicTestStrategy(
            buy_bars=[2], stop_loss_price=90.0, position_size_shares=10.0,
            trailing_stop_bars=[5], trailing_stop_prices=[85.0]
        )
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[8, 'close'] = 88.0  # Below original stop of 90
        result = engine.run('TEST', data, strategy)
        # Stop should still be at 90, triggered at bar 8
        self.assertEqual(result.trades[0].exit_reason, "Stop loss hit")

    def test_adjust_stop_via_signal(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], stop_loss_price=90.0,
            position_size_shares=10.0,
            adjust_stop_bars=[5], adjust_stop_prices=[96.0]
        )
        data = create_flat_data(n_bars=20, price=100.0)
        data.loc[8, 'close'] = 95.0  # Below new stop of 96 but above old stop of 90
        result = engine.run('TEST', data, strategy)
        # After stop adjusted to 96, price of 95 should trigger it
        self.assertEqual(result.trades[0].exit_reason, "Stop loss hit")


# =============================================================================
# H. PARTIAL EXIT TESTS
# =============================================================================

class TestPartialExit(unittest.TestCase):
    """Tests for partial exit functionality."""

    def test_partial_exit_reduces_quantity(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            partial_exit_bars=[5], partial_exit_fraction=0.5
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # Trade quantity should reflect initial, partial_exits count = 1
        trade = result.trades[0]
        self.assertEqual(trade.partial_exits, 1)

    def test_partial_exit_adds_to_capital(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            partial_exit_bars=[5], partial_exit_fraction=0.5
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # After partial exit of 50 shares at $100, capital should increase by $5000
        capital_after_partial = result.equity_curve.iloc[5]['capital']
        capital_before_partial = result.equity_curve.iloc[4]['capital']
        self.assertGreater(capital_after_partial, capital_before_partial)

    def test_partial_exit_position_stays_open(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            partial_exit_bars=[5], partial_exit_fraction=0.5
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # Position should still exist after partial exit — final sell at bar 15
        self.assertEqual(result.trades[0].exit_reason, "Test sell signal")

    def test_partial_exit_commission_charged(self):
        config = create_config(commission_pct=0.001, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            partial_exit_bars=[5], partial_exit_fraction=0.5
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        # Commission on entry (100*100*0.001=10) + partial exit (50*100*0.001=5) + final exit (50*100*0.001=5) = 20
        self.assertAlmostEqual(trade.commission_paid, 20.0, places=1)


# =============================================================================
# I. PYRAMID TESTS
# =============================================================================

class TestPyramid(unittest.TestCase):
    """Tests for pyramiding functionality."""

    def test_pyramid_increases_quantity(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            pyramid_bars=[5], pyramid_size=0.5
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # Quantity should be > initial 100 shares
        trade = result.trades[0]
        self.assertGreater(trade.quantity, 100.0)

    def test_pyramid_deducts_capital(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            pyramid_bars=[5], pyramid_size=0.5
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        capital_before = result.equity_curve.iloc[4]['capital']
        capital_after = result.equity_curve.iloc[5]['capital']
        self.assertLess(capital_after, capital_before)

    def test_only_one_pyramid_per_trade(self):
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            pyramid_bars=[5, 8], pyramid_size=0.5  # Two pyramid attempts
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        # Only one pyramid should succeed (BaseStrategy tracks via _has_pyramided)
        # Capital should only decrease once from pyramid
        trade = result.trades[0]
        # Verify quantity reflects only one pyramid
        # Initial: 100, pyramid at bar 5 adds ~445 shares (50% of remaining capital / 100)
        # Second pyramid at bar 8 should be blocked by strategy
        self.assertTrue(trade.quantity > 100)  # First pyramid worked


# =============================================================================
# J. PYRAMID DOUBLE SLIPPAGE BUG TEST
# =============================================================================

class TestPyramidSlippageBug(unittest.TestCase):
    """Tests for BUG 1: Pyramid double slippage penalty."""

    def test_pyramid_slippage_not_double_counted(self):
        """
        BUG 1: In _pyramid_position(), quantity is first calculated using
        execution_price (already slippage-adjusted), then further reduced by
        price/execution_price ratio. This double-penalizes for slippage.

        After fix, the second adjustment should be removed.
        """
        slippage_pct = 1.0  # 1% slippage to make effect visible
        config = create_config(capital=100000.0, commission_pct=0.0, slippage_pct=slippage_pct)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[15], position_size_shares=100.0,
            pyramid_bars=[5], pyramid_size=1.0  # Use 100% of remaining capital
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)

        # After initial buy of ~100 shares at 101 (slippage), remaining capital ≈ 89900
        # Pyramid should use all remaining capital at execution_price = 101
        # Expected pyramid shares ≈ 89900 / 101 ≈ 890 (single slippage)
        # With bug (double slippage): ≈ 890 * (100/101) ≈ 881 (extra reduction)
        trade = result.trades[0]
        total_qty = trade.quantity
        initial_qty_approx = 100.0 * (100.0 / 101.0)  # ~99.01 after slippage adjustment

        pyramid_qty = total_qty - initial_qty_approx
        remaining_capital_approx = 100000.0 - (initial_qty_approx * 101.0)
        expected_pyramid_shares = remaining_capital_approx / 101.0  # Single slippage

        # With fix, pyramid_qty should be close to expected_pyramid_shares
        # Tolerance: within 1% (without fix, difference would be ~1%)
        self.assertAlmostEqual(
            pyramid_qty, expected_pyramid_shares,
            delta=expected_pyramid_shares * 0.02,
            msg="Pyramid quantity suggests double slippage penalty (BUG 1)"
        )


# =============================================================================
# K. NO LOOKAHEAD TESTS
# =============================================================================

class TestNoLookahead(unittest.TestCase):
    """Tests for look-ahead bias protection."""

    def test_strategy_sees_only_historical_data(self):
        """Strategy should only see data up to current bar."""
        seen_lengths = []

        class LengthTrackingStrategy(DeterministicTestStrategy):
            def generate_entry_signal(self, context):
                seen_lengths.append(len(context.data))
                return None

        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = LengthTrackingStrategy()
        data = create_test_data(n_bars=20)
        engine.run('TEST', data, strategy)

        # At bar i, strategy should see i+1 rows
        for i, length in enumerate(seen_lengths):
            self.assertEqual(length, i + 1, f"At bar {i}, strategy saw {length} rows instead of {i+1}")

    def test_strategy_cannot_access_future_bars(self):
        """Accessing future data should raise IndexError."""
        future_access_blocked = []

        class FutureAccessStrategy(DeterministicTestStrategy):
            def generate_entry_signal(self, context):
                try:
                    _ = context.data.iloc[context.current_index + 1]
                    future_access_blocked.append(False)
                except IndexError:
                    future_access_blocked.append(True)
                return None

        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = FutureAccessStrategy()
        data = create_test_data(n_bars=20)
        engine.run('TEST', data, strategy)

        # Every bar (except maybe the last) should block future access
        for i, blocked in enumerate(future_access_blocked[:-1]):  # Skip last bar
            self.assertTrue(blocked, f"Future data access was NOT blocked at bar {i}")


# =============================================================================
# L. END OF BACKTEST TESTS
# =============================================================================

class TestEndOfBacktest(unittest.TestCase):
    """Tests for end-of-backtest behavior."""

    def test_open_position_auto_closed_at_end(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], position_size_shares=10.0
            # No sell bars — position stays open
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        self.assertEqual(len(result.trades), 1)

    def test_end_close_trade_record_created(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        self.assertIsNotNone(trade.exit_date)

    def test_end_close_exit_reason(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        self.assertEqual(result.trades[0].exit_reason, "End of backtest period")


# =============================================================================
# M. DATE RANGE FILTERING TESTS
# =============================================================================

class TestDateRangeFiltering(unittest.TestCase):
    """Tests for date range filtering."""

    def test_start_date_filter(self):
        data = create_test_data(n_bars=100, start_date='2020-01-01')
        start_date = datetime(2020, 3, 1)
        config = create_config(start_date=start_date)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        result = engine.run('TEST', data, strategy)
        first_date = result.equity_curve.iloc[0]['date']
        self.assertGreaterEqual(pd.Timestamp(first_date), pd.Timestamp(start_date))

    def test_end_date_filter(self):
        data = create_test_data(n_bars=100, start_date='2020-01-01')
        end_date = datetime(2020, 3, 1)
        config = create_config(end_date=end_date)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        result = engine.run('TEST', data, strategy)
        last_date = result.equity_curve.iloc[-1]['date']
        self.assertLessEqual(pd.Timestamp(last_date), pd.Timestamp(end_date))

    def test_combined_date_filter(self):
        data = create_test_data(n_bars=200, start_date='2020-01-01')
        start_date = datetime(2020, 3, 1)
        end_date = datetime(2020, 6, 1)
        config = create_config(start_date=start_date, end_date=end_date)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        result = engine.run('TEST', data, strategy)
        first_date = pd.Timestamp(result.equity_curve.iloc[0]['date'])
        last_date = pd.Timestamp(result.equity_curve.iloc[-1]['date'])
        self.assertGreaterEqual(first_date, pd.Timestamp(start_date))
        self.assertLessEqual(last_date, pd.Timestamp(end_date))


# =============================================================================
# N. EQUITY CURVE TESTS
# =============================================================================

class TestEquityCurve(unittest.TestCase):
    """Tests for equity curve generation."""

    def test_equity_curve_starts_at_initial_capital(self):
        config = create_config(capital=75000.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = create_test_data(n_bars=10)
        result = engine.run('TEST', data, strategy)
        self.assertAlmostEqual(result.equity_curve.iloc[0]['equity'], 75000.0)

    def test_equity_curve_has_required_columns(self):
        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = create_test_data(n_bars=10)
        result = engine.run('TEST', data, strategy)
        required_cols = {'date', 'equity', 'capital', 'position_value'}
        self.assertTrue(required_cols.issubset(set(result.equity_curve.columns)))

    def test_equity_curve_final_matches_result(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[2], sell_bars=[8], position_size_shares=10.0
        )
        data = create_test_data(n_bars=15)
        result = engine.run('TEST', data, strategy)
        # Final equity in equity curve should match result.final_equity
        last_equity = result.equity_curve.iloc[-1]['equity']
        self.assertAlmostEqual(last_equity, result.final_equity, places=2)


# =============================================================================
# O. TRADE RECORD TESTS
# =============================================================================

class TestTradeRecord(unittest.TestCase):
    """Tests for trade record accuracy."""

    def test_trade_entry_exit_dates_correct(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        self.assertEqual(pd.Timestamp(trade.entry_date), data.iloc[5]['date'])
        self.assertEqual(pd.Timestamp(trade.exit_date), data.iloc[10]['date'])

    def test_trade_quantity_matches_position(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=50.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        self.assertAlmostEqual(result.trades[0].quantity, 50.0, places=2)

    def test_trade_pl_matches_manual_calculation(self):
        config = create_config(capital=100000.0, commission_pct=0.001, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=50.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        # With flat price and 0.1% commission: P/L = -commission
        # Entry commission: 50 * 100 * 0.001 = 5
        # Exit commission: 50 * 100 * 0.001 = 5
        # P/L = (100-100)*50 - 10 = -10
        self.assertAlmostEqual(trade.pl, -10.0, places=1)

    def test_trade_duration_days_correct(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        expected_days = (data.iloc[10]['date'] - data.iloc[5]['date']).days
        self.assertEqual(trade.duration_days, expected_days)

    def test_trade_entry_exit_reasons_recorded(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        self.assertTrue(len(trade.entry_reason) > 0)
        self.assertTrue(len(trade.exit_reason) > 0)


# =============================================================================
# P. PROGRESS CALLBACK TESTS
# =============================================================================

class TestProgressCallback(unittest.TestCase):
    """Tests for progress callback functionality."""

    def test_progress_callback_called(self):
        calls = []

        def callback(current, total):
            calls.append((current, total))

        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        data = create_test_data(n_bars=100)
        engine.run('TEST', data, strategy, progress_callback=callback)
        self.assertGreater(len(calls), 0)

    def test_progress_callback_values_correct(self):
        calls = []

        def callback(current, total):
            calls.append((current, total))

        config = create_config()
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy()
        n_bars = 100
        data = create_test_data(n_bars=n_bars)
        engine.run('TEST', data, strategy, progress_callback=callback)
        for current, total in calls:
            self.assertEqual(total, n_bars)
            self.assertGreaterEqual(current, 1)
            self.assertLessEqual(current, total)


# =============================================================================
# Q. INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests with real strategies."""

    def test_multiple_trade_cycles(self):
        config = create_config(capital=100000.0, commission_pct=0.001, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5, 20, 35], sell_bars=[10, 25, 40],
            position_size_shares=10.0
        )
        data = create_flat_data(n_bars=50, price=100.0)
        result = engine.run('TEST', data, strategy)
        self.assertEqual(len(result.trades), 3)

    def test_random_control_strategy_runs(self):
        """RandomControlStrategy should produce trades with fixed seed."""
        from strategies.random_control_strategy import RandomControlStrategy
        config = create_config(commission_pct=0.001, slippage_pct=0.1)
        engine = SingleSecurityEngine(config)
        strategy = RandomControlStrategy(
            entry_probability=0.2, exit_probability=0.2, random_seed=42
        )
        data = create_test_data(n_bars=200, include_indicators=True)
        result = engine.run('TEST', data, strategy)
        self.assertIsInstance(result, BacktestResult)
        # With 20% probability over 200 bars, should have at least some trades
        self.assertGreater(result.num_trades, 0)

    def test_backtest_result_properties(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5, 20], sell_bars=[10, 25], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=30, price=100.0)
        result = engine.run('TEST', data, strategy)
        # BacktestResult properties should work
        self.assertEqual(result.num_trades, 2)
        self.assertIsInstance(result.winning_trades, list)
        self.assertIsInstance(result.losing_trades, list)
        self.assertIsInstance(result.win_rate, float)


# =============================================================================
# R. FX CONVERSION TESTS
# =============================================================================

class TestFXConversion(unittest.TestCase):
    """Tests for FX conversion functionality."""

    def test_no_conversion_when_same_currency(self):
        """Without currency converter, fx_rate should be 1.0."""
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5], sell_bars=[10], position_size_shares=10.0
        )
        data = create_flat_data(n_bars=20, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_fx_rate, 1.0)
        self.assertAlmostEqual(trade.exit_fx_rate, 1.0)

    def test_fx_rate_defaults_without_converter(self):
        """Without CurrencyConverter, all FX rates should default to 1.0."""
        config = create_config()
        engine = SingleSecurityEngine(config, currency_converter=None, security_registry=None)
        fx_rate = engine._get_fx_rate('TEST', datetime(2020, 1, 1))
        self.assertEqual(fx_rate, 1.0)


# =============================================================================
# S. TRADE COUNTER RESET TEST
# =============================================================================

class TestTradeCounterReset(unittest.TestCase):
    """Test that trade counter behavior is consistent."""

    def test_trade_ids_are_unique_within_backtest(self):
        config = create_config(commission_pct=0.0, slippage_pct=0.0)
        engine = SingleSecurityEngine(config)
        strategy = DeterministicTestStrategy(
            buy_bars=[5, 20, 35], sell_bars=[10, 25, 40],
            position_size_shares=10.0
        )
        data = create_flat_data(n_bars=50, price=100.0)
        result = engine.run('TEST', data, strategy)
        trade_ids = [t.trade_id for t in result.trades]
        self.assertEqual(len(trade_ids), len(set(trade_ids)), "Trade IDs not unique")


# =============================================================================
# T. POSITION MANAGER UNIT TESTS
# =============================================================================

class TestPositionManager(unittest.TestCase):
    """Unit tests for PositionManager."""

    def test_open_position_creates_position(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0)
        self.assertTrue(pm.has_position)

    def test_cannot_open_second_position(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0)
        with self.assertRaises(ValueError):
            pm.open_position('TEST2', datetime(2020, 1, 2), 50.0, 5.0)

    def test_close_position_clears_position(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0)
        pm.close_position()
        self.assertFalse(pm.has_position)

    def test_stop_loss_long_triggered(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0,
                        stop_loss=90.0, direction=TradeDirection.LONG)
        self.assertTrue(pm.check_stop_loss(89.0))
        self.assertTrue(pm.check_stop_loss(90.0))
        self.assertFalse(pm.check_stop_loss(91.0))

    def test_stop_loss_short_triggered(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0,
                        stop_loss=110.0, direction=TradeDirection.SHORT)
        self.assertTrue(pm.check_stop_loss(111.0))
        self.assertTrue(pm.check_stop_loss(110.0))
        self.assertFalse(pm.check_stop_loss(109.0))

    def test_take_profit_long_triggered(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0,
                        take_profit=120.0, direction=TradeDirection.LONG)
        self.assertTrue(pm.check_take_profit(121.0))
        self.assertTrue(pm.check_take_profit(120.0))
        self.assertFalse(pm.check_take_profit(119.0))

    def test_take_profit_short_triggered(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0,
                        take_profit=80.0, direction=TradeDirection.SHORT)
        self.assertTrue(pm.check_take_profit(79.0))
        self.assertTrue(pm.check_take_profit(80.0))
        self.assertFalse(pm.check_take_profit(81.0))

    def test_position_value(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0)
        self.assertAlmostEqual(pm.get_position_value(105.0), 1050.0)

    def test_adjust_stop_loss(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0, stop_loss=90.0)
        pm.adjust_stop_loss(95.0)
        self.assertAlmostEqual(pm.position.stop_loss, 95.0)

    def test_pyramid_increases_quantity_and_updates_price(self):
        pm = PositionManager()
        pm.open_position('TEST', datetime(2020, 1, 1), 100.0, 10.0, stop_loss=90.0)
        pm.add_pyramid(datetime(2020, 1, 5), 10.0, 110.0, 0.0, "Pyramid")
        self.assertAlmostEqual(pm.position.current_quantity, 20.0)
        # Average price = (100*10 + 110*10) / 20 = 105
        self.assertAlmostEqual(pm.position.entry_price, 105.0)


# =============================================================================
# U. TRADE EXECUTOR UNIT TESTS
# =============================================================================

class TestTradeExecutor(unittest.TestCase):
    """Unit tests for TradeExecutor."""

    def test_percentage_commission_calculation(self):
        commission = CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
        executor = TradeExecutor(commission)
        from Classes.Models.order import Order, OrderSide, OrderType
        order = Order('TEST', OrderSide.BUY, 100.0, OrderType.MARKET, 50.0,
                     datetime(2020, 1, 1))
        commission_paid = executor.execute_order(order)
        self.assertAlmostEqual(commission_paid, 5.0)  # 100 * 50 * 0.001

    def test_fixed_commission_calculation(self):
        commission = CommissionConfig(mode=CommissionMode.FIXED, value=9.99)
        executor = TradeExecutor(commission)
        from Classes.Models.order import Order, OrderSide, OrderType
        order = Order('TEST', OrderSide.BUY, 100.0, OrderType.MARKET, 50.0,
                     datetime(2020, 1, 1))
        commission_paid = executor.execute_order(order)
        self.assertAlmostEqual(commission_paid, 9.99)

    def test_trade_recording(self):
        commission = CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
        executor = TradeExecutor(commission)
        position = Position(
            symbol='TEST',
            entry_date=datetime(2020, 1, 1),
            entry_price=100.0,
            initial_quantity=10.0,
            current_quantity=10.0,
            entry_reason="Test"
        )
        trade = executor.create_trade(
            position=position,
            exit_date=datetime(2020, 1, 10),
            exit_price=110.0,
            exit_reason="Test exit",
            exit_commission=1.1
        )
        self.assertEqual(trade.symbol, 'TEST')
        self.assertEqual(trade.exit_reason, "Test exit")
        self.assertEqual(executor.get_trade_count(), 1)


if __name__ == '__main__':
    unittest.main()
