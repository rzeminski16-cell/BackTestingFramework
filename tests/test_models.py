"""
Comprehensive tests for all Model classes:
- TradeDirection
- Signal / SignalType
- Order / OrderSide / OrderType
- Position / PartialExit / PyramidEntry
- Trade (including from_position factory)
"""
import pytest
from datetime import datetime, timedelta

from Classes.Models.trade_direction import TradeDirection
from Classes.Models.signal import Signal, SignalType
from Classes.Models.order import Order, OrderSide, OrderType
from Classes.Models.position import Position, PartialExit, PyramidEntry
from Classes.Models.trade import Trade, generate_trade_id, reset_trade_counter


# =============================================================================
# TradeDirection Tests
# =============================================================================

class TestTradeDirection:
    def test_long_value(self):
        assert TradeDirection.LONG.value == "LONG"

    def test_short_value(self):
        assert TradeDirection.SHORT.value == "SHORT"

    def test_is_long(self):
        assert TradeDirection.LONG.is_long is True
        assert TradeDirection.SHORT.is_long is False

    def test_is_short(self):
        assert TradeDirection.SHORT.is_short is True
        assert TradeDirection.LONG.is_short is False

    def test_str(self):
        assert str(TradeDirection.LONG) == "LONG"
        assert str(TradeDirection.SHORT) == "SHORT"


# =============================================================================
# Signal Tests
# =============================================================================

class TestSignal:
    def test_buy_signal(self):
        sig = Signal.buy(size=1.0, stop_loss=95.0, take_profit=110.0,
                         reason="test", direction=TradeDirection.LONG)
        assert sig.type == SignalType.BUY
        assert sig.size == 1.0
        assert sig.stop_loss == 95.0
        assert sig.take_profit == 110.0
        assert sig.reason == "test"
        assert sig.direction == TradeDirection.LONG

    def test_sell_signal(self):
        sig = Signal.sell(reason="exit test")
        assert sig.type == SignalType.SELL
        assert sig.reason == "exit test"

    def test_hold_signal(self):
        sig = Signal.hold(reason="no action")
        assert sig.type == SignalType.HOLD
        assert sig.reason == "no action"

    def test_pyramid_signal(self):
        sig = Signal.pyramid(size=0.5, reason="add to position")
        assert sig.type == SignalType.PYRAMID
        assert sig.size == 0.5

    def test_partial_exit_signal(self):
        sig = Signal.partial_exit(fraction=0.3, reason="take profit")
        assert sig.type == SignalType.PARTIAL_EXIT
        assert sig.size == 0.3

    def test_partial_exit_invalid_fraction_zero(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Signal.partial_exit(fraction=0.0)

    def test_partial_exit_invalid_fraction_negative(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Signal.partial_exit(fraction=-0.5)

    def test_partial_exit_invalid_fraction_over_one(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Signal.partial_exit(fraction=1.5)

    def test_partial_exit_exactly_one(self):
        sig = Signal.partial_exit(fraction=1.0)
        assert sig.size == 1.0

    def test_adjust_stop_signal(self):
        sig = Signal.adjust_stop(new_stop_loss=98.0, reason="trail")
        assert sig.type == SignalType.ADJUST_STOP
        assert sig.new_stop_loss == 98.0

    def test_signal_metadata_defaults_to_empty_dict(self):
        sig = Signal(type=SignalType.HOLD)
        assert sig.metadata == {}

    def test_signal_metadata_preserved(self):
        sig = Signal(type=SignalType.HOLD, metadata={"key": "val"})
        assert sig.metadata == {"key": "val"}

    def test_buy_default_direction_is_long(self):
        sig = Signal.buy(size=1.0)
        assert sig.direction == TradeDirection.LONG

    def test_buy_short_direction(self):
        sig = Signal.buy(size=1.0, direction=TradeDirection.SHORT)
        assert sig.direction == TradeDirection.SHORT


# =============================================================================
# Order Tests
# =============================================================================

class TestOrder:
    def test_order_creation(self):
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
            price=150.0,
            timestamp=datetime(2024, 1, 1),
            stop_loss=140.0,
            take_profit=170.0,
            reason="test buy"
        )
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100.0
        assert order.price == 150.0

    def test_total_value(self):
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100.0,
            order_type=OrderType.MARKET, price=150.0,
            timestamp=datetime(2024, 1, 1)
        )
        assert order.total_value() == 15000.0

    def test_str_representation(self):
        order = Order(
            symbol="AAPL", side=OrderSide.SELL, quantity=50.0,
            order_type=OrderType.MARKET, price=160.0,
            timestamp=datetime(2024, 1, 1), reason="take profit"
        )
        s = str(order)
        assert "SELL" in s
        assert "AAPL" in s
        assert "take profit" in s

    def test_order_side_enum(self):
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_type_enum(self):
        assert OrderType.MARKET.value == "MARKET"


# =============================================================================
# Position Tests
# =============================================================================

class TestPosition:
    def _make_position(self, **kwargs):
        defaults = dict(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=90.0,
            take_profit=120.0,
        )
        defaults.update(kwargs)
        # Flip stop / take-profit defaults to be valid for SHORT so the
        # entry-time validation in Position.__post_init__ does not reject
        # the test setup. Caller can still override explicitly.
        if (defaults["direction"] == TradeDirection.SHORT
                and "stop_loss" not in kwargs
                and "take_profit" not in kwargs):
            defaults["stop_loss"] = 110.0
            defaults["take_profit"] = 80.0
        return Position(**defaults)

    def test_is_open(self):
        pos = self._make_position()
        assert pos.is_open is True

    def test_is_closed(self):
        pos = self._make_position(current_quantity=0.0)
        assert pos.is_open is False

    def test_calculate_value(self):
        pos = self._make_position()
        assert pos.calculate_value(110.0) == 11000.0

    def test_calculate_pl_long_profit(self):
        pos = self._make_position()
        pl = pos.calculate_pl(110.0)
        assert pl == 1000.0  # (110-100)*100

    def test_calculate_pl_long_loss(self):
        pos = self._make_position()
        pl = pos.calculate_pl(90.0)
        assert pl == -1000.0

    def test_calculate_pl_short_profit(self):
        pos = self._make_position(direction=TradeDirection.SHORT)
        pl = pos.calculate_pl(90.0)
        assert pl == 1000.0  # (100-90)*100

    def test_calculate_pl_short_loss(self):
        pos = self._make_position(direction=TradeDirection.SHORT)
        pl = pos.calculate_pl(110.0)
        assert pl == -1000.0

    def test_calculate_pl_with_commission(self):
        pos = self._make_position(total_commission_paid=50.0)
        pl = pos.calculate_pl(110.0)
        assert pl == 950.0  # 1000 - 50

    def test_calculate_pl_pct(self):
        pos = self._make_position()
        pct = pos.calculate_pl_pct(110.0)
        assert pct == pytest.approx(10.0, abs=0.01)  # 1000/10000 * 100

    def test_calculate_pl_pct_zero_investment(self):
        pos = self._make_position(current_quantity=0.0)
        pct = pos.calculate_pl_pct(110.0)
        assert pct == 0.0

    def test_total_quantity_no_pyramids(self):
        pos = self._make_position()
        assert pos.total_quantity == 100.0

    def test_add_pyramid(self):
        pos = self._make_position()
        pyramid = PyramidEntry(
            date=datetime(2024, 1, 10),
            quantity=50.0,
            price=110.0,
            commission=5.0,
            reason="test pyramid"
        )
        pos.add_pyramid(pyramid)

        assert pos.has_pyramided is True
        assert pos.current_quantity == 150.0
        # Weighted avg: (100*100 + 50*110) / 150 = 15500/150 = 103.333
        assert pos.entry_price == pytest.approx(103.333, abs=0.01)
        assert pos.total_commission_paid == 5.0
        assert len(pos.pyramid_entries) == 1

    def test_add_pyramid_only_once(self):
        pos = self._make_position()
        pyramid1 = PyramidEntry(date=datetime(2024, 1, 10), quantity=50.0,
                                 price=110.0, commission=5.0)
        pos.add_pyramid(pyramid1)

        pyramid2 = PyramidEntry(date=datetime(2024, 1, 20), quantity=25.0,
                                 price=115.0, commission=3.0)
        with pytest.raises(ValueError, match="already been pyramided"):
            pos.add_pyramid(pyramid2)

    def test_add_partial_exit(self):
        pos = self._make_position()
        exit_record = PartialExit(
            date=datetime(2024, 1, 15),
            quantity=30.0,
            price=110.0,
            reason="partial profit"
        )
        pos.add_partial_exit(exit_record)

        assert pos.current_quantity == 70.0
        assert len(pos.partial_exits) == 1

    def test_partial_exit_exceeds_quantity(self):
        pos = self._make_position()
        exit_record = PartialExit(
            date=datetime(2024, 1, 15),
            quantity=150.0,
            price=110.0
        )
        with pytest.raises(ValueError, match="Cannot exit"):
            pos.add_partial_exit(exit_record)

    def test_calculate_pl_with_partial_exits_long(self):
        pos = self._make_position()
        exit_record = PartialExit(
            date=datetime(2024, 1, 15),
            quantity=40.0,
            price=110.0
        )
        pos.add_partial_exit(exit_record)
        # Remaining 60 shares, current price 105
        pl = pos.calculate_pl(105.0)
        # Partial PL: (110-100)*40 = 400
        # Current PL: (105-100)*60 = 300
        # Total: 700
        assert pl == 700.0

    def test_calculate_pl_with_partial_exits_short(self):
        pos = self._make_position(direction=TradeDirection.SHORT)
        exit_record = PartialExit(
            date=datetime(2024, 1, 15),
            quantity=40.0,
            price=90.0
        )
        pos.add_partial_exit(exit_record)
        pl = pos.calculate_pl(95.0)
        # Partial PL: (100-90)*40 = 400
        # Current PL: (100-95)*60 = 300
        # Total: 700
        assert pl == 700.0

    def test_update_stop_loss(self):
        pos = self._make_position()
        pos.update_stop_loss(95.0)
        assert pos.stop_loss == 95.0

    def test_duration_days(self):
        pos = self._make_position(entry_date=datetime(2024, 1, 1))
        days = pos.duration_days(datetime(2024, 1, 11))
        assert days == 10

    def test_calculate_breakeven_stop(self):
        pos = self._make_position(total_commission_paid=100.0)
        # total cost = 100*100 + 100 = 10100
        # breakeven = 10100 / 100 = 101.0
        breakeven = pos.calculate_breakeven_stop(exit_commission_rate=0.0)
        assert breakeven == pytest.approx(101.0, abs=0.01)

    def test_calculate_breakeven_stop_with_exit_commission(self):
        pos = self._make_position()
        # estimated exit value = 100 * 100 = 10000
        # exit commission = 10000 * 0.001 = 10
        # total cost = 10000 + 0 + 10 = 10010
        # breakeven = 10010 / 100 = 100.10
        breakeven = pos.calculate_breakeven_stop(exit_commission_rate=0.001)
        assert breakeven == pytest.approx(100.10, abs=0.01)

    def test_str_representation(self):
        pos = self._make_position()
        s = str(pos)
        assert "AAPL" in s
        assert "LONG" in s

    def test_str_pyramided(self):
        pos = self._make_position()
        pyramid = PyramidEntry(date=datetime(2024, 1, 10), quantity=50.0,
                                price=110.0, commission=5.0)
        pos.add_pyramid(pyramid)
        s = str(pos)
        assert "PYRAMIDED" in s

    def test_current_value_raises(self):
        pos = self._make_position()
        with pytest.raises(NotImplementedError):
            _ = pos.current_value


# =============================================================================
# Position stop-loss invariant tests (entry-time validation + immutability)
# =============================================================================

class TestPositionInitialStopLoss:
    """Regression tests for the bug where initial_stop_loss was lost across
    pyramiding / trailing-stop updates and could be recorded above the entry
    price on the resulting Trade record."""

    def _long(self, **overrides):
        kw = dict(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, initial_quantity=100.0, current_quantity=100.0,
            direction=TradeDirection.LONG, stop_loss=95.0,
        )
        kw.update(overrides)
        return Position(**kw)

    def test_initial_stop_loss_auto_populates_from_stop_loss(self):
        pos = self._long(stop_loss=95.0)
        assert pos.initial_stop_loss == 95.0

    def test_initial_stop_loss_none_when_no_stop_provided(self):
        pos = self._long(stop_loss=None)
        assert pos.initial_stop_loss is None

    def test_long_with_stop_above_entry_raises(self):
        with pytest.raises(ValueError, match="LONG"):
            self._long(stop_loss=110.0)

    def test_long_with_stop_equal_to_entry_raises(self):
        with pytest.raises(ValueError, match="LONG"):
            self._long(stop_loss=100.0)

    def test_short_with_stop_below_entry_raises(self):
        with pytest.raises(ValueError, match="SHORT"):
            Position(
                symbol="AAPL", entry_date=datetime(2024, 1, 1),
                entry_price=100.0, initial_quantity=100.0, current_quantity=100.0,
                direction=TradeDirection.SHORT, stop_loss=95.0,
            )

    def test_short_with_stop_above_entry_is_valid(self):
        pos = Position(
            symbol="AAPL", entry_date=datetime(2024, 1, 1),
            entry_price=100.0, initial_quantity=100.0, current_quantity=100.0,
            direction=TradeDirection.SHORT, stop_loss=105.0,
        )
        assert pos.initial_stop_loss == 105.0

    def test_non_positive_stop_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self._long(stop_loss=-1.0)

    def test_update_stop_loss_does_not_change_initial_stop_loss(self):
        pos = self._long(stop_loss=95.0)
        pos.update_stop_loss(98.0)
        assert pos.stop_loss == 98.0
        assert pos.initial_stop_loss == 95.0

    def test_trailing_stop_above_entry_does_not_change_initial(self):
        # On a profitable LONG, the trailing stop eventually moves above
        # entry. The original entry stop must be preserved.
        pos = self._long(stop_loss=95.0)
        pos.update_stop_loss(105.0)  # trailing past entry
        assert pos.stop_loss == 105.0
        assert pos.initial_stop_loss == 95.0

    def test_pyramid_breakeven_does_not_change_initial(self):
        # Simulate what PositionManager.add_pyramid does: replace
        # position.stop_loss with the break-even price (which is above
        # entry for a LONG). initial_stop_loss must survive.
        pos = self._long(stop_loss=95.0)
        breakeven = 100.5  # any value > entry; mirrors break-even calc
        pos.stop_loss = breakeven
        assert pos.initial_stop_loss == 95.0

    def test_trade_from_position_preserves_initial_stop_loss(self):
        # The original bug: stop_loss gets trailed/break-evened, then
        # Trade.from_position copied position.stop_loss into BOTH
        # initial_stop_loss and final_stop_loss on the trade record.
        from Classes.Models.trade import Trade
        pos = self._long(stop_loss=95.0)
        # Simulate trailing past entry
        pos.update_stop_loss(105.0)
        # Build a trade as the engine does
        trade = Trade.from_position(
            position=pos, exit_date=datetime(2024, 1, 15),
            exit_price=110.0, exit_reason="take profit", commission_paid=10.0,
        )
        assert trade.initial_stop_loss == 95.0  # the bug we fixed
        assert trade.final_stop_loss == 105.0


# =============================================================================
# Trade Tests
# =============================================================================

class TestTrade:
    def setup_method(self):
        reset_trade_counter()

    def test_generate_trade_id_sequential(self):
        reset_trade_counter()
        id1 = generate_trade_id()
        id2 = generate_trade_id()
        assert id1 == "T000001"
        assert id2 == "T000002"

    def test_reset_trade_counter(self):
        reset_trade_counter()
        id1 = generate_trade_id()
        assert id1 == "T000001"
        reset_trade_counter()
        id2 = generate_trade_id()
        assert id2 == "T000001"

    def test_trade_defaults(self):
        trade = Trade()
        assert trade.symbol == ""
        assert trade.pl == 0.0
        assert trade.side == "LONG"
        assert trade.metadata == {}

    def test_is_winner(self):
        trade = Trade(pl=100.0)
        assert trade.is_winner is True

    def test_is_loser(self):
        trade = Trade(pl=-50.0)
        assert trade.is_winner is False

    def test_is_breakeven(self):
        trade = Trade(pl=0.0)
        assert trade.is_winner is False

    def test_entry_value(self):
        trade = Trade(entry_price=100.0, quantity=50.0)
        assert trade.entry_value == 5000.0

    def test_exit_value(self):
        trade = Trade(exit_price=110.0, quantity=50.0)
        assert trade.exit_value == 5500.0

    def test_to_dict(self):
        trade = Trade(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 15),
            entry_price=100.0,
            exit_price=110.0,
            quantity=50.0,
            pl=500.0,
            pl_pct=10.0,
        )
        d = trade.to_dict()
        assert d['symbol'] == "AAPL"
        assert d['entry_price'] == 100.0
        assert d['exit_price'] == 110.0
        assert d['quantity'] == 50.0
        assert d['pl'] == 500.0
        assert d['entry_date'] == "2024-01-01"
        assert d['exit_date'] == "2024-01-15"

    def test_from_position_long(self):
        pos = Position(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=90.0,
            entry_reason="test"
        )
        trade = Trade.from_position(
            position=pos,
            exit_date=datetime(2024, 1, 15),
            exit_price=110.0,
            exit_reason="take profit",
            commission_paid=20.0
        )
        assert trade.symbol == "AAPL"
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.quantity == 100.0
        # PL: (110-100)*100 - 20 = 980
        assert trade.pl == 980.0
        assert trade.duration_days == 14
        assert trade.side == "LONG"
        assert trade.exit_reason == "take profit"

    def test_from_position_short(self):
        pos = Position(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.SHORT,
            stop_loss=110.0,
            entry_reason="short test"
        )
        trade = Trade.from_position(
            position=pos,
            exit_date=datetime(2024, 1, 15),
            exit_price=90.0,
            exit_reason="cover",
            commission_paid=20.0
        )
        # Short PL: (100-90)*100 - 20 = 980
        assert trade.pl == 980.0
        assert trade.side == "SHORT"

    def test_from_position_with_partial_exits(self):
        pos = Position(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=60.0,
            direction=TradeDirection.LONG,
        )
        pos.partial_exits = [
            PartialExit(date=datetime(2024, 1, 10), quantity=40.0, price=110.0)
        ]
        trade = Trade.from_position(
            position=pos,
            exit_date=datetime(2024, 1, 20),
            exit_price=105.0,
            exit_reason="exit",
            commission_paid=10.0
        )
        # Partial PL: (110-100)*40 = 400
        # Final PL: (105-100)*60 = 300
        # Total: 700 - 10 = 690
        assert trade.pl == 690.0
        assert trade.partial_exits == 1

    def test_str_representation(self):
        trade = Trade(
            symbol="AAPL",
            side="LONG",
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 15),
            pl=500.0,
            pl_pct=5.0
        )
        s = str(trade)
        assert "AAPL" in s
        assert "LONG" in s
        assert "500.00" in s

    def test_str_with_fx_pl(self):
        trade = Trade(
            symbol="AAPL",
            side="LONG",
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 15),
            pl=500.0,
            pl_pct=5.0,
            fx_pl=25.0
        )
        s = str(trade)
        assert "FX P/L" in s
