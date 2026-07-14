"""
Cross-engine parity: the single-security engine and the portfolio engine
must produce the same trades and the same equity path when given the same
strategy, data, and costs on a single symbol.

The two engines are separate implementations of the same execution model;
this test is the drift alarm between them. Any intentional divergence must
be documented here.
"""
import pandas as pd
import pytest

from Classes.Config.config import (
    BacktestConfig, CommissionConfig, CommissionMode, PortfolioConfig,
)
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext


class ScriptedStrategy(BaseStrategy):
    """Deterministic multi-trade strategy driven by bar indices."""

    _validate_on_init = False

    def __init__(self, direction=TradeDirection.LONG, **params):
        self._direction = direction
        # Two round trips, a trailing-stop bump, and a partial exit.
        self.entry_bars = {2, 14}
        self.exit_bars = {8, 22}
        self.partial_bars = {5}
        super().__init__(**params)

    @property
    def trade_direction(self):
        return self._direction

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context: StrategyContext):
        if context.current_index in self.entry_bars:
            return Signal.buy(size=1.0,
                              stop_loss=self.calculate_initial_stop_loss(context),
                              direction=self._direction, reason="scripted entry")
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext):
        if self._direction == TradeDirection.SHORT:
            return context.current_price * 1.8
        return context.current_price * 0.6

    def generate_exit_signal(self, context: StrategyContext):
        if context.current_index in self.exit_bars:
            return Signal.sell(reason="scripted exit")
        return None

    def should_partial_exit(self, context: StrategyContext):
        if context.current_index in self.partial_bars:
            return 0.25
        return None

    def should_adjust_stop(self, context: StrategyContext):
        # Trail: keep the stop 30% below (long) the running close.
        if self._direction == TradeDirection.LONG:
            return context.current_price * 0.7
        return context.current_price * 1.3

    def position_size(self, context: StrategyContext, signal: Signal):
        return (context.available_capital * 0.4) / context.current_price


def _data(num_bars=28):
    dates = pd.date_range("2024-01-01", periods=num_bars, freq="D")
    # A wiggly but deterministic path.
    prices = [100 + 3 * ((i * 7) % 5) + i * 0.8 for i in range(num_bars)]
    return pd.DataFrame({"date": dates, "close": prices, "open": prices,
                         "high": [p * 1.01 for p in prices],
                         "low": [p * 0.99 for p in prices],
                         "volume": [5000] * num_bars})


def _run_both(direction=TradeDirection.LONG, commission=0.001, slippage=0.1):
    data = _data()

    single = SingleSecurityEngine(BacktestConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=commission),
        slippage_percent=slippage,
    ))
    single_result = single.run("AAA", data.copy(), ScriptedStrategy(direction))

    portfolio = PortfolioEngine(PortfolioConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=commission),
        slippage_percent=slippage,
    ))
    portfolio_result = portfolio.run({"AAA": data.copy()}, ScriptedStrategy(direction))
    return single_result, portfolio_result


class TestEngineParity:
    @pytest.mark.parametrize("direction", [TradeDirection.LONG, TradeDirection.SHORT])
    def test_trades_match(self, direction):
        s, p = _run_both(direction)
        p_trades = [t for r in p.symbol_results.values() for t in r.trades]
        assert len(s.trades) == len(p_trades) > 0

        for st, pt in zip(s.trades, p_trades):
            assert st.entry_date == pt.entry_date
            assert st.exit_date == pt.exit_date
            assert st.entry_price == pytest.approx(pt.entry_price, rel=1e-9)
            assert st.exit_price == pytest.approx(pt.exit_price, rel=1e-9)
            assert st.quantity == pytest.approx(pt.quantity, rel=1e-9)
            assert st.pl == pytest.approx(pt.pl, rel=1e-9, abs=1e-6)
            assert st.partial_exits == pt.partial_exits

    @pytest.mark.parametrize("direction", [TradeDirection.LONG, TradeDirection.SHORT])
    def test_final_equity_matches(self, direction):
        s, p = _run_both(direction)
        assert s.final_equity == pytest.approx(p.final_equity, rel=1e-9, abs=1e-4)

    def test_equity_curves_match(self):
        s, p = _run_both(TradeDirection.LONG)
        se = s.equity_curve[["date", "equity"]].reset_index(drop=True)
        pe = p.portfolio_equity_curve[["date", "equity"]].reset_index(drop=True)
        assert len(se) == len(pe)
        merged = se.merge(pe, on="date", suffixes=("_s", "_p"))
        assert len(merged) == len(se)
        diffs = (merged["equity_s"] - merged["equity_p"]).abs()
        assert float(diffs.max()) == pytest.approx(0.0, abs=1e-6), (
            f"equity paths diverge, max abs diff {diffs.max():.6f} on "
            f"{merged.loc[diffs.idxmax(), 'date']}"
        )
