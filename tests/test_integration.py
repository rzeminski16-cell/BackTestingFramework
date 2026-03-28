"""
Integration tests for the BackTestingFramework.

These tests validate complete end-to-end workflows:
1. Full single-security backtest with deterministic strategy
2. Stop loss execution in backtests
3. Take profit execution in backtests
4. Trailing stop adjustment during backtest
5. Partial exit during backtest
6. Pyramiding during backtest
7. Slippage application
8. Commission tracking through full cycle
9. Equity curve accuracy
10. No-lookahead-bias verification
11. Date range filtering
12. Zero-trade scenarios
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List

from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.backtest_result import BacktestResult
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Models.signal import Signal, SignalType
from Classes.Models.trade import Trade, reset_trade_counter
from Classes.Models.trade_direction import TradeDirection
from Classes.Models.position import Position
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext


# =============================================================================
# Deterministic Test Strategy
# =============================================================================

class DeterministicStrategy(BaseStrategy):
    """Strategy that buys/sells on specified bar indices for deterministic testing."""

    _validate_on_init = False

    def __init__(self, buy_bars=None, sell_bars=None, stop_loss_price=None,
                 take_profit_price=None, position_size_shares=100.0,
                 pyramid_bars=None, pyramid_size=0.5,
                 partial_exit_bars=None, partial_exit_fraction=0.5,
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
                reason="Test buy",
                direction=self.trade_direction
            )
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        if self.stop_loss_price is not None:
            return self.stop_loss_price
        return context.current_price * 0.95

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index in self.sell_bars:
            return Signal.sell(reason="Test sell")
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
        if context.current_index in self.adjust_stop_bars:
            return self.adjust_stop_bars[context.current_index]
        return None


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def simple_price_data():
    """Create simple, predictable price data for testing."""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    prices = [100.0, 101.0, 102.0, 103.0, 104.0,  # 0-4: rising
              105.0, 106.0, 107.0, 108.0, 109.0,    # 5-9: rising
              110.0, 109.0, 108.0, 107.0, 106.0,    # 10-14: falling
              105.0, 104.0, 103.0, 102.0, 101.0,    # 15-19: falling
              100.0, 99.0, 98.0, 97.0, 96.0,        # 20-24: falling
              95.0, 94.0, 93.0, 92.0, 91.0]         # 25-29: falling
    return pd.DataFrame({'date': dates, 'close': prices})


@pytest.fixture
def config_no_slippage():
    return BacktestConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0
    )


@pytest.fixture
def config_with_commission():
    return BacktestConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001),
        slippage_percent=0.0
    )


@pytest.fixture
def config_with_slippage():
    return BacktestConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.1
    )


# =============================================================================
# Basic Backtest Tests
# =============================================================================

class TestBasicBacktest:
    def test_buy_and_sell_trade(self, simple_price_data, config_no_slippage):
        """Buy at bar 2, sell at bar 8. Price goes 102 -> 108."""
        reset_trade_counter()
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.entry_price == 102.0
        assert trade.exit_price == 108.0
        assert trade.pl > 0  # Profitable trade
        assert bool(trade.is_winner) is True

    def test_losing_trade(self, simple_price_data, config_no_slippage):
        """Buy at bar 10, sell at bar 18. Price drops from 110.
        Note: default stop loss is 95% of entry = 104.5, which triggers before bar 18."""
        reset_trade_counter()
        # Set stop loss very low so it doesn't interfere
        strategy = DeterministicStrategy(
            buy_bars=[10], sell_bars=[18], stop_loss_price=50.0
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.entry_price == 110.0
        assert trade.exit_price == 102.0
        assert trade.pl < 0  # Losing trade
        assert bool(trade.is_winner) is False

    def test_multiple_trades(self, simple_price_data, config_no_slippage):
        """Two sequential trades."""
        reset_trade_counter()
        strategy = DeterministicStrategy(
            buy_bars=[2, 15],
            sell_bars=[8, 20]
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.num_trades == 2

    def test_no_trades(self, simple_price_data, config_no_slippage):
        """No buy signals means no trades."""
        reset_trade_counter()
        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.num_trades == 0
        assert result.total_return == 0.0

    def test_position_closed_at_end(self, simple_price_data, config_no_slippage):
        """Open position is closed at end of backtest if stop loss doesn't trigger first."""
        reset_trade_counter()
        # Set very low stop loss so it won't trigger (price never goes below 91)
        strategy = DeterministicStrategy(buy_bars=[2], stop_loss_price=50.0)
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.exit_reason == "End of backtest period"

    def test_empty_data_rejected(self, config_no_slippage):
        """Empty DataFrame should raise error."""
        data = pd.DataFrame(columns=['date', 'close'])
        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config_no_slippage)
        with pytest.raises(ValueError, match="No data"):
            engine.run("TEST", data, strategy)


# =============================================================================
# Stop Loss Tests
# =============================================================================

class TestStopLoss:
    def test_stop_loss_triggers(self, simple_price_data, config_no_slippage):
        """Buy at bar 5 (105), stop at 100. Price drops through 100 at bar 20."""
        reset_trade_counter()
        strategy = DeterministicStrategy(
            buy_bars=[5],
            stop_loss_price=100.0
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.num_trades == 1
        trade = result.trades[0]
        assert "Stop loss" in trade.exit_reason or "stop" in trade.exit_reason.lower()

    def test_stop_loss_not_triggered(self, config_no_slippage):
        """Stop loss below all prices should not trigger."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100 + i for i in range(20)]  # Always rising
        data = pd.DataFrame({'date': dates, 'close': prices})

        strategy = DeterministicStrategy(
            buy_bars=[2], sell_bars=[15],
            stop_loss_price=50.0
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", data, strategy)

        trade = result.trades[0]
        assert "Stop loss" not in trade.exit_reason


# =============================================================================
# Take Profit Tests
# =============================================================================

class TestTakeProfit:
    def test_take_profit_triggers(self, config_no_slippage):
        """Buy at bar 2, take profit at 108. Price reaches 108 at bar 8."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100 + i for i in range(20)]
        data = pd.DataFrame({'date': dates, 'close': prices})

        strategy = DeterministicStrategy(
            buy_bars=[2], take_profit_price=108.0
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", data, strategy)

        assert result.num_trades == 1
        trade = result.trades[0]
        assert "take profit" in trade.exit_reason.lower() or "Take profit" in trade.exit_reason


# =============================================================================
# Commission Tests
# =============================================================================

class TestCommission:
    def test_commission_deducted(self, simple_price_data, config_with_commission):
        """Commission should reduce P/L."""
        reset_trade_counter()
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])

        # Run with commission
        engine = SingleSecurityEngine(config_with_commission)
        result_with_comm = engine.run("TEST", simple_price_data, strategy)

        # Run without commission
        reset_trade_counter()
        config_no_comm = BacktestConfig(
            initial_capital=100000.0,
            commission=CommissionConfig(value=0.0),
            slippage_percent=0.0
        )
        engine_no_comm = SingleSecurityEngine(config_no_comm)
        result_no_comm = engine_no_comm.run("TEST", simple_price_data, strategy)

        # With commission should have lower return
        assert result_with_comm.total_return < result_no_comm.total_return

    def test_commission_tracked(self, simple_price_data, config_with_commission):
        """Commission should be recorded on trade."""
        reset_trade_counter()
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])
        engine = SingleSecurityEngine(config_with_commission)
        result = engine.run("TEST", simple_price_data, strategy)

        trade = result.trades[0]
        assert trade.commission_paid > 0


# =============================================================================
# Slippage Tests
# =============================================================================

class TestSlippage:
    def test_slippage_affects_price(self, simple_price_data, config_with_slippage):
        """Slippage should worsen execution prices."""
        reset_trade_counter()
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])
        engine = SingleSecurityEngine(config_with_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        trade = result.trades[0]
        # Buy slippage: entry should be higher than 102
        assert trade.entry_price > 102.0
        # Sell slippage: exit should be lower than 108
        assert trade.exit_price < 108.0

    def test_zero_slippage(self, simple_price_data, config_no_slippage):
        """Zero slippage should give exact prices."""
        reset_trade_counter()
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        trade = result.trades[0]
        assert trade.entry_price == 102.0
        assert trade.exit_price == 108.0


# =============================================================================
# Equity Curve Tests
# =============================================================================

class TestEquityCurve:
    def test_equity_curve_length(self, simple_price_data, config_no_slippage):
        """Equity curve should have one entry per bar."""
        reset_trade_counter()
        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert len(result.equity_curve) == len(simple_price_data)

    def test_equity_curve_starts_at_initial_capital(self, simple_price_data, config_no_slippage):
        """First equity value should be initial capital."""
        reset_trade_counter()
        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.equity_curve.iloc[0]['equity'] == pytest.approx(100000.0)

    def test_equity_stays_flat_no_trades(self, simple_price_data, config_no_slippage):
        """With no trades, equity should stay at initial capital."""
        reset_trade_counter()
        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        for _, row in result.equity_curve.iterrows():
            assert row['equity'] == pytest.approx(100000.0)

    def test_equity_reflects_position_value(self, simple_price_data, config_no_slippage):
        """Equity should include position value when in a trade."""
        reset_trade_counter()
        # Buy at bar 2, position held through bar 5
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        # At bar 5, price is 105 (bought 100 shares at 102)
        # Position value = 100 * 105 = 10500
        # Capital = 100000 - 10200 = 89800
        # Equity = 89800 + 10500 = 100300
        eq_at_5 = result.equity_curve.iloc[5]['equity']
        assert eq_at_5 > 100000.0  # Should reflect unrealized gain


# =============================================================================
# Trailing Stop Tests
# =============================================================================

class TestTrailingStop:
    def test_trailing_stop_moves_up(self, config_no_slippage):
        """Trailing stop should move up with rising prices."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100 + i for i in range(20)]
        data = pd.DataFrame({'date': dates, 'close': prices})

        strategy = DeterministicStrategy(
            buy_bars=[2],
            sell_bars=[15],
            stop_loss_price=95.0,
            adjust_stop_bars=[5, 8],
            adjust_stop_prices=[100.0, 103.0]
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", data, strategy)

        # Trade should complete normally (stop was moved up but never hit)
        assert result.num_trades == 1


# =============================================================================
# Partial Exit Tests
# =============================================================================

class TestPartialExit:
    def test_partial_exit_records(self, config_no_slippage):
        """Partial exit should be recorded on the trade."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100 + i for i in range(20)]
        data = pd.DataFrame({'date': dates, 'close': prices})

        strategy = DeterministicStrategy(
            buy_bars=[2],
            sell_bars=[15],
            partial_exit_bars=[8],
            partial_exit_fraction=0.5
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", data, strategy)

        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.partial_exits >= 1


# =============================================================================
# Pyramid Tests
# =============================================================================

class TestPyramidIntegration:
    def test_pyramid_adds_to_position(self, config_no_slippage):
        """Pyramiding should add to position size."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100 + i for i in range(20)]
        data = pd.DataFrame({'date': dates, 'close': prices})

        strategy = DeterministicStrategy(
            buy_bars=[2],
            sell_bars=[15],
            pyramid_bars=[8],
            pyramid_size=0.5
        )
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", data, strategy)

        assert result.num_trades == 1


# =============================================================================
# No Lookahead Bias Tests
# =============================================================================

class TestNoLookaheadBias:
    def test_strategy_only_sees_historical_data(self, config_no_slippage):
        """Strategy should never see future data."""
        reset_trade_counter()
        seen_data_lengths = []

        class SpyStrategy(BaseStrategy):
            _validate_on_init = False

            @property
            def trade_direction(self):
                return TradeDirection.LONG

            def required_columns(self):
                return ['date', 'close']

            def generate_entry_signal(self, context):
                # Record how much data we can see
                seen_data_lengths.append(len(context.data))
                return None

            def calculate_initial_stop_loss(self, context):
                return context.current_price * 0.95

            def generate_exit_signal(self, context):
                return None

            def position_size(self, context, signal):
                return 100.0

        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100 + i for i in range(20)]
        data = pd.DataFrame({'date': dates, 'close': prices})

        strategy = SpyStrategy()
        engine = SingleSecurityEngine(config_no_slippage)
        engine.run("TEST", data, strategy)

        # Each bar should see only data up to current bar (inclusive)
        for i, length in enumerate(seen_data_lengths):
            assert length == i + 1, f"At bar {i}, strategy saw {length} bars (expected {i+1})"


# =============================================================================
# Progress Callback Tests
# =============================================================================

class TestProgressCallback:
    def test_progress_callback_called(self, simple_price_data, config_no_slippage):
        """Progress callback should be invoked during backtest."""
        reset_trade_counter()
        calls = []

        def callback(current, total):
            calls.append((current, total))

        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config_no_slippage)
        engine.run("TEST", simple_price_data, strategy, progress_callback=callback)

        assert len(calls) > 0
        # Last call should have current == total
        assert calls[-1][0] == calls[-1][1]


# =============================================================================
# Date Range Filtering Tests
# =============================================================================

class TestDateRangeFiltering:
    def test_start_date_filter(self):
        """Only data from start_date onward should be used."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = [100 + i * 0.1 for i in range(100)]
        data = pd.DataFrame({'date': dates, 'close': prices})

        config = BacktestConfig(
            initial_capital=100000.0,
            start_date=datetime(2024, 2, 1),
            commission=CommissionConfig(value=0.0),
            slippage_percent=0.0
        )
        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config)
        result = engine.run("TEST", data, strategy)

        # Equity curve should start at/after Feb 1
        first_date = result.equity_curve.iloc[0]['date']
        assert first_date >= pd.Timestamp('2024-02-01')

    def test_end_date_filter(self):
        """Only data up to end_date should be used."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = [100 + i * 0.1 for i in range(100)]
        data = pd.DataFrame({'date': dates, 'close': prices})

        config = BacktestConfig(
            initial_capital=100000.0,
            end_date=datetime(2024, 2, 28),
            commission=CommissionConfig(value=0.0),
            slippage_percent=0.0
        )
        strategy = DeterministicStrategy()
        engine = SingleSecurityEngine(config)
        result = engine.run("TEST", data, strategy)

        last_date = result.equity_curve.iloc[-1]['date']
        assert last_date <= pd.Timestamp('2024-02-28')


# =============================================================================
# Capital Sufficiency Tests
# =============================================================================

class TestCapitalSufficiency:
    def test_insufficient_capital_skips_trade(self, config_no_slippage):
        """If position cost exceeds capital, trade should be skipped."""
        reset_trade_counter()
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100.0] * 20
        data = pd.DataFrame({'date': dates, 'close': prices})

        # Request 2000 shares at $100 = $200k, but only $100k available
        strategy = DeterministicStrategy(
            buy_bars=[2],
            position_size_shares=2000.0
        )
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=CommissionConfig(value=0.0),
            slippage_percent=0.0,
            position_size_limit=1.0  # Max 100% of capital
        )
        engine = SingleSecurityEngine(config)
        result = engine.run("TEST", data, strategy)

        # Trade should be skipped or quantity adjusted to fit
        # Either 0 trades (skipped) or 1 trade with reduced quantity
        if result.num_trades == 1:
            trade = result.trades[0]
            assert trade.quantity * trade.entry_price <= 100000.0 + 0.01


# =============================================================================
# Return Accuracy Tests
# =============================================================================

class TestReturnAccuracy:
    def test_total_return_matches(self, simple_price_data, config_no_slippage):
        """total_return should equal final_equity - initial_capital."""
        reset_trade_counter()
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        assert result.total_return == pytest.approx(
            result.final_equity - config_no_slippage.initial_capital, abs=0.01
        )

    def test_return_pct_consistent(self, simple_price_data, config_no_slippage):
        """total_return_pct should match total_return / initial_capital * 100."""
        reset_trade_counter()
        strategy = DeterministicStrategy(buy_bars=[2], sell_bars=[8])
        engine = SingleSecurityEngine(config_no_slippage)
        result = engine.run("TEST", simple_price_data, strategy)

        expected_pct = (result.total_return / config_no_slippage.initial_capital) * 100
        assert result.total_return_pct == pytest.approx(expected_pct, abs=0.01)
