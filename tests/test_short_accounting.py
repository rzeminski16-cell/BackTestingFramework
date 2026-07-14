"""
Ledger-vs-trade-log invariant tests for LONG and SHORT accounting.

These tests pin the contract that the cash ledger, the equity curve, and the
trade records must tell the same story in both directions:

    final_equity - initial_capital == sum(trade.pl)      (no-FX runs)

and that mark-to-market equity moves in the *direction-aware* sense while a
position is open (a profitable short raises equity as price falls).

Regression coverage for the short-accounting fix: previously both engines
booked shorts with long semantics (buy at entry / sell at exit), so a short
that the trade log recorded as +1,000 profit showed up as -1,000 in the
equity curve and final capital.
"""
import pandas as pd
import pytest

from Classes.Config.config import (
    BacktestConfig, CommissionConfig, CommissionMode, PortfolioConfig,
)
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Models.position import Position
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext


# ---------------------------------------------------------------------------
# Deterministic one-trade strategies
# ---------------------------------------------------------------------------

class OneTradeStrategy(BaseStrategy):
    """Enter on ``entry_bar``, exit on ``exit_bar``; optional partial exit."""

    _validate_on_init = False

    def __init__(self, direction, entry_bar=1, exit_bar=8,
                 partial_bar=None, partial_fraction=0.5, **params):
        self._direction = direction
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar
        self.partial_bar = partial_bar
        self.partial_fraction = partial_fraction
        super().__init__(**params)

    @property
    def trade_direction(self) -> TradeDirection:
        return self._direction

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context: StrategyContext):
        if context.current_index == self.entry_bar:
            return Signal.buy(size=1.0,
                              stop_loss=self.calculate_initial_stop_loss(context),
                              direction=self._direction, reason="test entry")
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        # Wide stop on the correct side so it never triggers in these tests.
        if self._direction == TradeDirection.SHORT:
            return context.current_price * 2.0
        return context.current_price * 0.5

    def generate_exit_signal(self, context: StrategyContext):
        if context.current_index == self.exit_bar:
            return Signal.sell(reason="test exit")
        return None

    def should_partial_exit(self, context: StrategyContext):
        if self.partial_bar is not None and context.current_index == self.partial_bar:
            self.partial_bar = None  # fire once
            return self.partial_fraction
        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return (context.available_capital * 0.5) / context.current_price


def _falling_data(num_bars=12, start=100.0, step=-2.0):
    dates = pd.date_range("2024-01-01", periods=num_bars, freq="D")
    prices = [start + i * step for i in range(num_bars)]
    return pd.DataFrame({"date": dates, "close": prices, "open": prices,
                         "high": [p + 0.5 for p in prices],
                         "low": [p - 0.5 for p in prices],
                         "volume": [1000] * num_bars})


def _rising_data(num_bars=12, start=100.0, step=2.0):
    return _falling_data(num_bars=num_bars, start=start, step=step)


def _config(commission=0.001, slippage=0.1):
    return BacktestConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=commission),
        slippage_percent=slippage,
    )


def _run(data, strategy, commission=0.001, slippage=0.1):
    engine = SingleSecurityEngine(_config(commission, slippage))
    return engine.run("TEST", data, strategy)


def _assert_ledger_matches_trades(result, initial_capital=100_000.0):
    ledger_pl = result.final_equity - initial_capital
    trades_pl = sum(t.pl for t in result.trades)
    assert trades_pl == pytest.approx(ledger_pl, abs=0.01), (
        f"cash ledger P/L ({ledger_pl:.2f}) != trade-log P/L ({trades_pl:.2f})"
    )


# ---------------------------------------------------------------------------
# Single-security engine invariants
# ---------------------------------------------------------------------------

class TestSingleSecurityLedgerInvariant:
    def test_long_ledger_matches_trades(self):
        result = _run(_rising_data(), OneTradeStrategy(TradeDirection.LONG))
        assert len(result.trades) == 1
        assert result.trades[0].pl > 0
        _assert_ledger_matches_trades(result)

    def test_short_ledger_matches_trades(self):
        result = _run(_falling_data(), OneTradeStrategy(TradeDirection.SHORT))
        assert len(result.trades) == 1
        assert result.trades[0].pl > 0, "short into a falling market must profit"
        _assert_ledger_matches_trades(result)

    def test_short_losing_ledger_matches_trades(self):
        result = _run(_rising_data(), OneTradeStrategy(TradeDirection.SHORT))
        assert len(result.trades) == 1
        assert result.trades[0].pl < 0, "short into a rising market must lose"
        _assert_ledger_matches_trades(result)

    def test_long_with_partial_exit(self):
        result = _run(_rising_data(),
                      OneTradeStrategy(TradeDirection.LONG, partial_bar=4))
        assert len(result.trades) == 1
        assert result.trades[0].partial_exits == 1
        _assert_ledger_matches_trades(result)

    def test_short_with_partial_exit(self):
        result = _run(_falling_data(),
                      OneTradeStrategy(TradeDirection.SHORT, partial_bar=4))
        assert len(result.trades) == 1
        assert result.trades[0].partial_exits == 1
        _assert_ledger_matches_trades(result)

    def test_short_equity_rises_as_price_falls(self):
        """Mark-to-market must be direction-aware, not just the trade log."""
        result = _run(_falling_data(), OneTradeStrategy(TradeDirection.SHORT),
                      commission=0.0, slippage=0.0)
        eq = result.equity_curve["equity"]
        # In the middle of the trade the short is in profit.
        assert eq.iloc[5] > eq.iloc[0], (
            "equity should rise while an open short is profitable "
            f"(got {eq.iloc[5]:.2f} vs {eq.iloc[0]:.2f})"
        )

    def test_short_equity_unchanged_at_entry_bar(self):
        """Posting collateral must not change total equity."""
        result = _run(_falling_data(), OneTradeStrategy(TradeDirection.SHORT),
                      commission=0.0, slippage=0.0)
        eq = result.equity_curve["equity"]
        assert eq.iloc[1] == pytest.approx(eq.iloc[0], abs=0.01)


# ---------------------------------------------------------------------------
# Portfolio engine invariants
# ---------------------------------------------------------------------------

def _portfolio_config():
    return PortfolioConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001),
        slippage_percent=0.1,
    )


class TestPortfolioLedgerInvariant:
    def test_long_ledger_matches_trades(self):
        engine = PortfolioEngine(_portfolio_config())
        result = engine.run({"AAA": _rising_data()},
                            OneTradeStrategy(TradeDirection.LONG))
        trades = [t for r in result.symbol_results.values() for t in r.trades]
        assert len(trades) == 1
        assert sum(t.pl for t in trades) == pytest.approx(
            result.final_equity - 100_000.0, abs=0.01)

    def test_short_ledger_matches_trades(self):
        engine = PortfolioEngine(_portfolio_config())
        result = engine.run({"AAA": _falling_data()},
                            OneTradeStrategy(TradeDirection.SHORT))
        trades = [t for r in result.symbol_results.values() for t in r.trades]
        assert len(trades) == 1
        assert trades[0].pl > 0
        assert sum(t.pl for t in trades) == pytest.approx(
            result.final_equity - 100_000.0, abs=0.01)

    def test_short_portfolio_equity_direction(self):
        cfg = PortfolioConfig(
            initial_capital=100_000.0,
            commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
            slippage_percent=0.0,
        )
        engine = PortfolioEngine(cfg)
        result = engine.run({"AAA": _falling_data()},
                            OneTradeStrategy(TradeDirection.SHORT))
        eq = result.portfolio_equity_curve["equity"]
        assert eq.iloc[5] > eq.iloc[0]


# ---------------------------------------------------------------------------
# Position-level direction awareness
# ---------------------------------------------------------------------------

class TestPositionDirectionAware:
    def _pos(self, direction, entry=100.0, qty=10.0):
        stop = 150.0 if direction == TradeDirection.SHORT else 50.0
        return Position(symbol="T", entry_date=pd.Timestamp("2024-01-01"),
                        entry_price=entry, initial_quantity=qty,
                        current_quantity=qty, direction=direction,
                        stop_loss=stop)

    def test_long_value_tracks_price(self):
        pos = self._pos(TradeDirection.LONG)
        assert pos.calculate_value(110.0) == pytest.approx(1100.0)

    def test_short_value_rises_as_price_falls(self):
        pos = self._pos(TradeDirection.SHORT)
        # collateral 1000 + P/L 100 = 1100
        assert pos.calculate_value(90.0) == pytest.approx(1100.0)
        # collateral 1000 - P/L 100 = 900
        assert pos.calculate_value(110.0) == pytest.approx(900.0)

    def test_short_value_at_entry_equals_collateral(self):
        pos = self._pos(TradeDirection.SHORT)
        assert pos.calculate_value(100.0) == pytest.approx(1000.0)

    def test_close_out_value_matches_calculate_value_full_close(self):
        for direction in (TradeDirection.LONG, TradeDirection.SHORT):
            pos = self._pos(direction)
            assert pos.close_out_value(93.0) == pytest.approx(pos.calculate_value(93.0))

    def test_breakeven_stop_sides(self):
        commission_rate = 0.001
        long_pos = self._pos(TradeDirection.LONG)
        long_pos.total_commission_paid = 5.0
        assert long_pos.calculate_breakeven_stop(commission_rate) > long_pos.entry_price

        short_pos = self._pos(TradeDirection.SHORT)
        short_pos.total_commission_paid = 5.0
        assert short_pos.calculate_breakeven_stop(commission_rate) < short_pos.entry_price
