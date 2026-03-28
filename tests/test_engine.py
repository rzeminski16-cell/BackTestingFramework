"""
Comprehensive tests for Engine components:
- PositionManager (open/close, stop loss, take profit, pyramiding, partial exits)
- TradeExecutor (commission, trade creation, FX P&L)
- BacktestResult (properties, str)
"""
import pytest
import pandas as pd
from datetime import datetime
from typing import Optional, List

from Classes.Engine.position_manager import PositionManager
from Classes.Engine.trade_executor import TradeExecutor
from Classes.Engine.backtest_result import BacktestResult
from Classes.Config.config import CommissionConfig, CommissionMode
from Classes.Models.position import Position, PartialExit
from Classes.Models.order import Order, OrderSide, OrderType
from Classes.Models.trade import Trade, reset_trade_counter
from Classes.Models.trade_direction import TradeDirection
from Classes.Models.signal import Signal, SignalType


# =============================================================================
# PositionManager Tests
# =============================================================================

class TestPositionManager:
    def setup_method(self):
        self.pm = PositionManager()

    def test_no_position_initially(self):
        assert self.pm.has_position is False
        assert self.pm.get_position() is None

    def test_open_position(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=140.0,
            take_profit=170.0, entry_reason="test"
        )
        assert self.pm.has_position is True
        pos = self.pm.get_position()
        assert pos.symbol == "AAPL"
        assert pos.entry_price == 150.0
        assert pos.current_quantity == 100.0

    def test_cannot_open_two_positions(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0
        )
        with pytest.raises(ValueError, match="already in position"):
            self.pm.open_position(
                symbol="MSFT", entry_date=datetime(2024, 1, 2),
                entry_price=300.0, quantity=50.0
            )

    def test_close_position(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0
        )
        closed = self.pm.close_position()
        assert closed.symbol == "AAPL"
        assert self.pm.has_position is False

    def test_close_no_position(self):
        with pytest.raises(ValueError, match="No position to close"):
            self.pm.close_position()

    # --- Stop Loss ---

    def test_check_stop_loss_long_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=140.0
        )
        assert self.pm.check_stop_loss(140.0) is True
        assert self.pm.check_stop_loss(135.0) is True

    def test_check_stop_loss_long_not_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=140.0
        )
        assert self.pm.check_stop_loss(145.0) is False

    def test_check_stop_loss_short_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=160.0,
            direction=TradeDirection.SHORT
        )
        assert self.pm.check_stop_loss(160.0) is True
        assert self.pm.check_stop_loss(165.0) is True

    def test_check_stop_loss_short_not_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=160.0,
            direction=TradeDirection.SHORT
        )
        assert self.pm.check_stop_loss(155.0) is False

    def test_check_stop_loss_no_position(self):
        assert self.pm.check_stop_loss(100.0) is False

    def test_check_stop_loss_no_stop_set(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=None
        )
        assert self.pm.check_stop_loss(100.0) is False

    # --- Take Profit ---

    def test_check_take_profit_long_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, take_profit=170.0
        )
        assert self.pm.check_take_profit(170.0) is True
        assert self.pm.check_take_profit(175.0) is True

    def test_check_take_profit_long_not_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, take_profit=170.0
        )
        assert self.pm.check_take_profit(165.0) is False

    def test_check_take_profit_short_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, take_profit=130.0,
            direction=TradeDirection.SHORT
        )
        assert self.pm.check_take_profit(130.0) is True
        assert self.pm.check_take_profit(125.0) is True

    def test_check_take_profit_short_not_hit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, take_profit=130.0,
            direction=TradeDirection.SHORT
        )
        assert self.pm.check_take_profit(135.0) is False

    def test_check_take_profit_no_position(self):
        assert self.pm.check_take_profit(200.0) is False

    def test_check_take_profit_no_tp_set(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, take_profit=None
        )
        assert self.pm.check_take_profit(200.0) is False

    # --- Adjust Stop Loss ---

    def test_adjust_stop_loss(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=140.0
        )
        self.pm.adjust_stop_loss(145.0)
        assert self.pm.get_position().stop_loss == 145.0

    def test_adjust_stop_no_position(self):
        with pytest.raises(ValueError, match="No position"):
            self.pm.adjust_stop_loss(145.0)

    # --- Partial Exit ---

    def test_partial_exit(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0
        )
        self.pm.add_partial_exit(
            exit_date=datetime(2024, 1, 15),
            quantity=30.0, price=160.0, reason="partial profit",
            commission_paid=5.0
        )
        pos = self.pm.get_position()
        assert pos.current_quantity == 70.0
        assert pos.total_commission_paid == 5.0
        assert len(pos.partial_exits) == 1

    def test_partial_exit_no_position(self):
        with pytest.raises(ValueError, match="No position"):
            self.pm.add_partial_exit(
                exit_date=datetime(2024, 1, 15),
                quantity=30.0, price=160.0
            )

    # --- Pyramiding ---

    def test_pyramid(self):
        self.pm.commission_rate = 0.001
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, quantity=100.0, stop_loss=90.0
        )
        breakeven = self.pm.add_pyramid(
            pyramid_date=datetime(2024, 1, 10),
            quantity=50.0, price=110.0, commission=5.0,
            reason="momentum"
        )
        pos = self.pm.get_position()
        assert pos.current_quantity == 150.0
        assert pos.has_pyramided is True
        # Break-even stop should be set
        assert pos.stop_loss == breakeven
        assert breakeven > 0

    def test_pyramid_no_position(self):
        with pytest.raises(ValueError, match="No position to pyramid"):
            self.pm.add_pyramid(
                pyramid_date=datetime(2024, 1, 10),
                quantity=50.0, price=110.0, commission=5.0
            )

    def test_pyramid_already_pyramided(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, quantity=100.0, stop_loss=90.0
        )
        self.pm.add_pyramid(
            pyramid_date=datetime(2024, 1, 10),
            quantity=50.0, price=110.0, commission=5.0
        )
        with pytest.raises(ValueError, match="already been pyramided"):
            self.pm.add_pyramid(
                pyramid_date=datetime(2024, 1, 20),
                quantity=25.0, price=115.0, commission=3.0
            )

    # --- Position Value ---

    def test_get_position_value(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, quantity=100.0
        )
        assert self.pm.get_position_value(110.0) == 11000.0

    def test_get_position_value_no_position(self):
        assert self.pm.get_position_value(100.0) == 0.0

    # --- Direction-aware position opening ---

    def test_open_short_position(self):
        self.pm.open_position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=150.0, quantity=100.0, stop_loss=160.0,
            direction=TradeDirection.SHORT
        )
        pos = self.pm.get_position()
        assert pos.direction == TradeDirection.SHORT


# =============================================================================
# TradeExecutor Tests
# =============================================================================

class TestTradeExecutor:
    def setup_method(self):
        reset_trade_counter()
        self.commission = CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
        self.executor = TradeExecutor(self.commission)

    def test_execute_order_commission(self):
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100.0,
            order_type=OrderType.MARKET, price=150.0,
            timestamp=datetime(2024, 1, 1)
        )
        commission = self.executor.execute_order(order)
        # 0.1% of 15000 = 15
        assert commission == pytest.approx(15.0)

    def test_create_trade(self):
        pos = Position(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=90.0,
            entry_reason="test entry"
        )
        trade = self.executor.create_trade(
            position=pos,
            exit_date=datetime(2024, 1, 15),
            exit_price=110.0,
            exit_reason="take profit",
            exit_commission=10.0
        )
        assert trade.symbol == "AAPL"
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.exit_reason == "take profit"
        assert len(self.executor.trades) == 1

    def test_create_trade_fx_conversion(self):
        """Trade created with different entry/exit FX rates."""
        pos = Position(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
        )
        trade = self.executor.create_trade(
            position=pos,
            exit_date=datetime(2024, 1, 15),
            exit_price=110.0,
            exit_reason="exit",
            exit_commission=0.0,
            entry_fx_rate=0.8,
            exit_fx_rate=0.85,
            security_currency="USD"
        )
        assert trade.entry_fx_rate == 0.8
        assert trade.exit_fx_rate == 0.85
        assert trade.security_currency == "USD"
        # Security PL uses entry_fx_rate
        # FX PL should be non-zero since rates differ
        assert trade.fx_pl != 0

    def test_get_trades(self):
        assert self.executor.get_trades() == []
        # Create a trade
        pos = Position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, initial_quantity=100.0, current_quantity=100.0,
            direction=TradeDirection.LONG
        )
        self.executor.create_trade(
            position=pos, exit_date=datetime(2024, 1, 15),
            exit_price=110.0, exit_reason="exit", exit_commission=0.0
        )
        trades = self.executor.get_trades()
        assert len(trades) == 1
        # Should be a copy
        trades.append(None)
        assert len(self.executor.trades) == 1

    def test_get_trade_count(self):
        assert self.executor.get_trade_count() == 0

    def test_get_total_pl(self):
        assert self.executor.get_total_pl() == 0.0

    def test_get_winning_losing_trades(self):
        pos = Position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, initial_quantity=100.0, current_quantity=100.0,
            direction=TradeDirection.LONG
        )
        # Winning trade
        self.executor.create_trade(
            position=pos, exit_date=datetime(2024, 1, 15),
            exit_price=110.0, exit_reason="win", exit_commission=0.0
        )
        reset_trade_counter()
        # Losing trade
        pos2 = Position(
            symbol="MSFT", entry_date=datetime(2024, 2, 1),
            entry_price=300.0, initial_quantity=50.0, current_quantity=50.0,
            direction=TradeDirection.LONG
        )
        self.executor.create_trade(
            position=pos2, exit_date=datetime(2024, 2, 15),
            exit_price=290.0, exit_reason="loss", exit_commission=0.0
        )
        assert len(self.executor.get_winning_trades()) == 1
        assert len(self.executor.get_losing_trades()) == 1

    def test_reset(self):
        pos = Position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, initial_quantity=100.0, current_quantity=100.0,
            direction=TradeDirection.LONG
        )
        self.executor.create_trade(
            position=pos, exit_date=datetime(2024, 1, 15),
            exit_price=110.0, exit_reason="exit", exit_commission=0.0
        )
        self.executor.reset()
        assert self.executor.get_trade_count() == 0


# =============================================================================
# BacktestResult Tests
# =============================================================================

class TestBacktestResult:
    def _make_result(self, trades=None, equity=None):
        reset_trade_counter()
        if trades is None:
            trades = [
                Trade(symbol="AAPL", pl=100.0, entry_date=datetime(2024, 1, 1),
                      exit_date=datetime(2024, 1, 15)),
                Trade(symbol="AAPL", pl=-30.0, entry_date=datetime(2024, 2, 1),
                      exit_date=datetime(2024, 2, 15)),
                Trade(symbol="AAPL", pl=50.0, entry_date=datetime(2024, 3, 1),
                      exit_date=datetime(2024, 3, 15)),
            ]
        if equity is None:
            equity = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=90, freq='D'),
                'equity': [100000 + i * 10 for i in range(90)]
            })
        return BacktestResult(
            symbol="AAPL",
            strategy_name="TestStrategy",
            trades=trades,
            equity_curve=equity,
            final_equity=100120.0,
            total_return=120.0,
            total_return_pct=0.12
        )

    def test_num_trades(self):
        result = self._make_result()
        assert result.num_trades == 3

    def test_winning_trades(self):
        result = self._make_result()
        assert len(result.winning_trades) == 2

    def test_losing_trades(self):
        result = self._make_result()
        assert len(result.losing_trades) == 1

    def test_win_rate(self):
        result = self._make_result()
        assert result.win_rate == pytest.approx(2/3)

    def test_win_rate_no_trades(self):
        result = self._make_result(trades=[])
        assert result.win_rate == 0.0

    def test_str(self):
        result = self._make_result()
        s = str(result)
        assert "AAPL" in s
        assert "TestStrategy" in s
        assert "3 trades" in s
