"""
Tests for the opt-in execution realism options:

- intrabar_stops: stop/take-profit trigger on the bar's high/low with
  gap-aware fills (gap through the level fills at the open).
- execution_timing=NEXT_BAR_OPEN: signals computed on a close fill at the
  next bar's open (single-security engine; the portfolio engine rejects it).

Defaults (SAME_BAR_CLOSE + close-only stops) must be bit-identical to the
historical behaviour — that is covered by the unchanged existing suite.
"""
import pandas as pd
import pytest

from Classes.Config.config import (
    BacktestConfig, CommissionConfig, CommissionMode, ExecutionTiming,
    PortfolioConfig,
)
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Models.position import Position
from Classes.Engine.position_manager import PositionManager
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy


def _config(**kwargs):
    return BacktestConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0,
        **kwargs,
    )


class EnterOnceStrategy(BaseStrategy):
    """Enter LONG on bar 1 with a fixed stop / take-profit; never exits."""

    _validate_on_init = False

    def __init__(self, stop=90.0, take_profit=None, **params):
        self._stop = stop
        self._tp = take_profit
        super().__init__(**params)

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index == 1:
            return Signal.buy(size=1.0, stop_loss=self._stop,
                              take_profit=self._tp,
                              direction=TradeDirection.LONG, reason="entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return self._stop

    def generate_exit_signal(self, context):
        return None

    def position_size(self, context, signal):
        return (context.available_capital * 0.5) / context.current_price


def _bars(closes, highs=None, lows=None, opens=None):
    n = len(closes)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "close": closes,
        "open": opens or closes,
        "high": highs or closes,
        "low": lows or closes,
        "volume": [1000] * n,
    })


# ---------------------------------------------------------------------------
# PositionManager.check_exits_intrabar unit behaviour
# ---------------------------------------------------------------------------

class TestIntrabarChecker:
    def _pm(self, direction=TradeDirection.LONG, stop=90.0, tp=None):
        pm = PositionManager()
        pm.position = Position(
            symbol="T", entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0, initial_quantity=10.0, current_quantity=10.0,
            direction=direction, stop_loss=stop, take_profit=tp)
        return pm

    def test_long_stop_touched_fills_at_stop(self):
        pm = self._pm()
        fill = pm.check_exits_intrabar(bar_open=99.0, bar_high=100.0, bar_low=89.0)
        assert fill == (90.0, "Stop loss hit")

    def test_long_stop_gap_fills_at_open(self):
        pm = self._pm()
        fill = pm.check_exits_intrabar(bar_open=85.0, bar_high=88.0, bar_low=84.0)
        assert fill == (85.0, "Stop loss hit (gap)")

    def test_long_tp_touched_fills_at_tp(self):
        pm = self._pm(stop=50.0, tp=110.0)
        fill = pm.check_exits_intrabar(bar_open=105.0, bar_high=112.0, bar_low=104.0)
        assert fill == (110.0, "Take profit hit")

    def test_long_tp_gap_fills_at_open(self):
        pm = self._pm(stop=50.0, tp=110.0)
        fill = pm.check_exits_intrabar(bar_open=115.0, bar_high=118.0, bar_low=114.0)
        assert fill == (115.0, "Take profit hit (gap)")

    def test_stop_beats_tp_when_both_in_range(self):
        pm = self._pm(stop=95.0, tp=105.0)
        fill = pm.check_exits_intrabar(bar_open=100.0, bar_high=106.0, bar_low=94.0)
        assert fill[1].startswith("Stop loss")

    def test_short_mirrored(self):
        pm = self._pm(direction=TradeDirection.SHORT, stop=110.0, tp=90.0)
        # Stop touched intrabar
        assert pm.check_exits_intrabar(100.0, 111.0, 99.0) == (110.0, "Stop loss hit")
        # Stop gapped at open
        assert pm.check_exits_intrabar(115.0, 116.0, 114.0) == (115.0, "Stop loss hit (gap)")
        # TP touched
        assert pm.check_exits_intrabar(95.0, 96.0, 89.0) == (90.0, "Take profit hit")

    def test_no_trigger_returns_none(self):
        pm = self._pm(stop=90.0, tp=120.0)
        assert pm.check_exits_intrabar(100.0, 105.0, 95.0) is None


# ---------------------------------------------------------------------------
# Engine integration: intrabar stops
# ---------------------------------------------------------------------------

class TestIntrabarEngine:
    def test_close_only_misses_wick_intrabar_catches_it(self):
        # Bar 3 wicks to 88 (through the 90 stop) but closes back at 100.
        closes = [100, 100, 100, 100, 100, 100]
        lows = [100, 100, 100, 88, 100, 100]
        data = _bars(closes, lows=lows)

        default_result = SingleSecurityEngine(_config()).run(
            "T", data.copy(), EnterOnceStrategy(stop=90.0))
        # Close never <= 90, so the close-only engine never stops out.
        assert all("Stop loss" not in t.exit_reason for t in default_result.trades)

        intrabar_result = SingleSecurityEngine(_config(intrabar_stops=True)).run(
            "T", data.copy(), EnterOnceStrategy(stop=90.0))
        trade = intrabar_result.trades[0]
        assert trade.exit_reason == "Stop loss hit"
        assert trade.exit_price == pytest.approx(90.0)
        assert trade.exit_date == data["date"].iloc[3]

    def test_gap_through_stop_fills_at_open(self):
        closes = [100, 100, 100, 95, 100, 100]
        opens = [100, 100, 100, 84, 100, 100]   # bar 3 gaps to 84 < stop 90
        lows = [100, 100, 100, 83, 100, 100]
        data = _bars(closes, opens=opens, lows=lows)

        result = SingleSecurityEngine(_config(intrabar_stops=True)).run(
            "T", data.copy(), EnterOnceStrategy(stop=90.0))
        trade = result.trades[0]
        assert trade.exit_reason == "Stop loss hit (gap)"
        assert trade.exit_price == pytest.approx(84.0)

    def test_portfolio_engine_intrabar(self):
        closes = [100, 100, 100, 100, 100, 100]
        lows = [100, 100, 100, 88, 100, 100]
        data = _bars(closes, lows=lows)
        cfg = PortfolioConfig(
            initial_capital=100_000.0,
            commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
            slippage_percent=0.0,
            intrabar_stops=True,
        )
        result = PortfolioEngine(cfg).run({"T": data}, EnterOnceStrategy(stop=90.0))
        trades = [t for r in result.symbol_results.values() for t in r.trades]
        assert trades[0].exit_reason == "Stop loss hit"
        assert trades[0].exit_price == pytest.approx(90.0)

    def test_ledger_invariant_holds_with_intrabar(self):
        closes = [100, 100, 100, 100, 100, 100]
        lows = [100, 100, 100, 88, 100, 100]
        data = _bars(closes, lows=lows)
        cfg = BacktestConfig(
            initial_capital=100_000.0,
            commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001),
            slippage_percent=0.1,
            intrabar_stops=True,
        )
        result = SingleSecurityEngine(cfg).run("T", data, EnterOnceStrategy(stop=90.0))
        assert sum(t.pl for t in result.trades) == pytest.approx(
            result.final_equity - 100_000.0, abs=0.01)


# ---------------------------------------------------------------------------
# Engine integration: next-bar-open execution
# ---------------------------------------------------------------------------

class EntryExitStrategy(EnterOnceStrategy):
    """Enter on bar 1, exit signal on bar 3."""

    def generate_exit_signal(self, context):
        if context.current_index == 3:
            return Signal.sell(reason="scripted exit")
        return None


class TestNextBarOpen:
    def _data(self):
        closes = [100, 102, 104, 106, 108, 110]
        opens = [99, 101, 103, 105, 107, 109]
        return _bars(closes, opens=opens)

    def test_fills_move_to_next_open(self):
        data = self._data()
        strat = EntryExitStrategy(stop=50.0)

        same_bar = SingleSecurityEngine(_config()).run("T", data.copy(), strat)
        t0 = same_bar.trades[0]
        assert t0.entry_price == pytest.approx(102.0)  # bar-1 close
        assert t0.exit_price == pytest.approx(106.0)   # bar-3 close

        nxt = SingleSecurityEngine(
            _config(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)
        ).run("T", data.copy(), EntryExitStrategy(stop=50.0))
        t1 = nxt.trades[0]
        assert t1.entry_price == pytest.approx(103.0)  # bar-2 open
        assert t1.entry_date == data["date"].iloc[2]
        assert t1.exit_price == pytest.approx(107.0)   # bar-4 open
        assert t1.exit_date == data["date"].iloc[4]

    def test_signal_on_final_bar_is_dropped(self):
        closes = [100, 100, 100]
        data = _bars(closes)

        class LateEntry(EnterOnceStrategy):
            def generate_entry_signal(self, context):
                if context.current_index == 2:  # final bar
                    return Signal.buy(size=1.0, stop_loss=50.0,
                                      direction=TradeDirection.LONG, reason="late")
                return None

        result = SingleSecurityEngine(
            _config(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)
        ).run("T", data, LateEntry(stop=50.0))
        assert len(result.trades) == 0

    def test_entry_gapped_across_stop_is_skipped(self):
        closes = [100, 100, 80, 80, 80]
        opens = [100, 100, 80, 80, 80]  # bar-2 open 80 < stop 90
        data = _bars(closes, opens=opens)
        result = SingleSecurityEngine(
            _config(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)
        ).run("T", data, EnterOnceStrategy(stop=90.0))
        assert len(result.trades) == 0

    def test_ledger_invariant_holds_next_bar_open(self):
        data = self._data()
        cfg = BacktestConfig(
            initial_capital=100_000.0,
            commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001),
            slippage_percent=0.1,
            execution_timing=ExecutionTiming.NEXT_BAR_OPEN,
        )
        result = SingleSecurityEngine(cfg).run("T", data, EntryExitStrategy(stop=50.0))
        assert len(result.trades) == 1
        assert sum(t.pl for t in result.trades) == pytest.approx(
            result.final_equity - 100_000.0, abs=0.01)

    def test_portfolio_engine_rejects_next_bar_open(self):
        cfg = PortfolioConfig(
            initial_capital=100_000.0,
            execution_timing=ExecutionTiming.NEXT_BAR_OPEN,
        )
        engine = PortfolioEngine(cfg)
        with pytest.raises(ValueError, match="next_bar_open"):
            engine.run({"T": self._data()}, EntryExitStrategy(stop=50.0))

    def test_string_config_coerced(self):
        cfg = _config(execution_timing="next_bar_open")
        assert cfg.execution_timing == ExecutionTiming.NEXT_BAR_OPEN
