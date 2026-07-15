"""
Interactive-mode integration tests for the portfolio engine.

Covers: accept-all parity, per-signal prompting order, capital
contingency (reduce size / free capital by closing or trimming positions
/ reject), full_isolation rejection, and continuity of the existing
capital-allocation logs.
"""
import pandas as pd
import pytest

from Classes.Config.config import (
    CommissionConfig, CommissionMode, PortfolioConfig,
)
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Interactive.decision_provider import ScriptedDecisionProvider
from Classes.Interactive.models import (
    BacktestRunManifest,
    DecisionAction,
    DecisionResponse,
)
from Classes.Interactive.session import InteractiveSession
from Classes.Interactive.store import InteractiveRunStore
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy


def _config(**kwargs):
    return PortfolioConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0,
        **kwargs,
    )


class PortfolioTestStrategy(BaseStrategy):
    """Per-symbol scripted entries/exits with equity-fraction sizing."""

    _validate_on_init = False

    def __init__(self, entry_bars=None, exit_bars=None, size_frac=0.5,
                 stop_frac=0.5, **params):
        self._entry_bars = entry_bars or {}
        self._exit_bars = exit_bars or {}
        self._size_frac = size_frac
        self._stop_frac = stop_frac
        super().__init__(**params)

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index == self._entry_bars.get(context.symbol, -1):
            return Signal.buy(size=1.0,
                              stop_loss=context.current_price * self._stop_frac,
                              direction=TradeDirection.LONG, reason="entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return context.current_price * self._stop_frac

    def generate_exit_signal(self, context):
        if context.current_index == self._exit_bars.get(context.symbol, -1):
            return Signal.sell(reason="exit")
        return None

    def position_size(self, context, signal):
        # Sized off total equity (like the real risk-based strategies), so
        # a signal can require more than the available cash.
        return (context.total_equity * self._size_frac) / context.current_price


def _data_dict(symbols=("AAA", "BBB"), n=12, price=100.0):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return {
        sym: pd.DataFrame({
            "date": dates,
            "close": [price] * n,
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "volume": [1000.0] * n,
        })
        for sym in symbols
    }


def make_session(tmp_path, script=None, cooldown_bars=21):
    store = InteractiveRunStore(tmp_path / "run")
    manifest = BacktestRunManifest(
        run_id="p", backtest_name="p", mode="interactive",
        engine_type="portfolio", cooldown_bars=cooldown_bars)
    store.write_manifest(manifest)
    provider = ScriptedDecisionProvider(script)
    return InteractiveSession(store, provider, manifest), store, provider


class TestAcceptAllParity:
    def test_accept_all_equals_rules_only(self, tmp_path):
        strategy_args = dict(entry_bars={"AAA": 2, "BBB": 3},
                             exit_bars={"AAA": 8, "BBB": 9},
                             size_frac=0.3)
        baseline = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(**strategy_args))

        session, store, _ = make_session(tmp_path)
        interactive = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(**strategy_args),
            decision_session=session)

        base_trades = sorted(
            (t for r in baseline.symbol_results.values() for t in r.trades),
            key=lambda t: (str(t.entry_date), t.symbol))
        int_trades = sorted(
            (t for r in interactive.symbol_results.values() for t in r.trades),
            key=lambda t: (str(t.entry_date), t.symbol))
        assert len(base_trades) == len(int_trades) == 2
        for bt, it in zip(base_trades, int_trades):
            assert it.symbol == bt.symbol
            assert it.entry_price == bt.entry_price
            assert it.quantity == bt.quantity
            assert it.pl == bt.pl
        assert interactive.final_equity == baseline.final_equity
        pd.testing.assert_frame_equal(interactive.portfolio_equity_curve,
                                      baseline.portfolio_equity_curve)

    def test_full_isolation_with_interactive_raises(self, tmp_path):
        session, _, _ = make_session(tmp_path)
        with pytest.raises(ValueError, match="full_isolation"):
            PortfolioEngine(_config(full_isolation=True)).run(
                _data_dict(), PortfolioTestStrategy(), decision_session=session)


class TestPromptingOrder:
    def test_same_day_buys_prompt_in_symbol_order(self, tmp_path):
        session, _, provider = make_session(tmp_path)
        PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2, "BBB": 2}, size_frac=0.2),
            decision_session=session)
        buy_requests = [r for r in provider.requests if r.kind == "SIGNAL"]
        assert [r.event.symbol for r in buy_requests] == ["AAA", "BBB"]
        assert all(r.event.bar_date == "2024-01-03" for r in buy_requests)

    def test_reject_buy_records_capital_event_and_rejection(self, tmp_path):
        session, _, _ = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.REJECT,
                             rationale="weak fundamentals"),
        ])
        result = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(entry_bars={"AAA": 2}),
            decision_session=session)
        assert all(not r.trades for r in result.symbol_results.values())
        rejected = [e for e in result.capital_allocation_events
                    if e.signal_type == "REJECTED"]
        assert len(rejected) == 1 and rejected[0].symbol == "AAA"
        assert len(result.signal_rejections) == 1
        assert "weak fundamentals" in result.signal_rejections[0].reason

    def test_exit_signal_reject_keeps_position(self, tmp_path):
        session, store, _ = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),      # BUY
            DecisionResponse(action=DecisionAction.REJECT,       # SELL
                             rationale="thesis intact"),
        ])
        result = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2}, exit_bars={"AAA": 5}),
            decision_session=session)
        trades = result.symbol_results["AAA"].trades
        # Only the end-of-backtest close, not the rejected bar-5 exit.
        assert len(trades) == 1
        assert trades[0].exit_reason == "End of backtest period"
        sell_events = [e for e in store.load_events()
                       if e.signal_type == "SELL"
                       and e.event_kind == "STRATEGY_SIGNAL"]
        assert len(sell_events) == 1


class TestCapitalContingency:
    def _engine_run(self, tmp_path, responses, entry_bars=None, size_frac=0.8,
                    exit_bars=None):
        session, store, provider = make_session(tmp_path, responses)
        result = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars=entry_bars or {"AAA": 2, "BBB": 5},
                exit_bars=exit_bars or {}, size_frac=size_frac),
            decision_session=session)
        return result, session, store, provider

    def test_capital_options_shipped_with_first_request(self, tmp_path):
        _, _, _, provider = self._engine_run(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),   # AAA fits
            DecisionResponse(action=DecisionAction.ACCEPT),   # BBB doesn't
            DecisionResponse(action=DecisionAction.REJECT,
                             capital_resolution={'choice': 'reject'}),
        ])
        bbb_request = [r for r in provider.requests
                       if r.kind == "SIGNAL" and r.event.symbol == "BBB"][0]
        assert bbb_request.capital_options is not None
        opts = bbb_request.capital_options
        assert opts.required_capital > opts.available_capital
        assert 0 < opts.affordable_fraction < 1
        assert opts.positions[0]['symbol'] == "AAA"
        assert 'est_capital_freed_close' in opts.positions[0]

    def test_reduce_size(self, tmp_path):
        result, _, _, provider = self._engine_run(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),   # AAA: 80k of 100k
            DecisionResponse(action=DecisionAction.ACCEPT),   # BBB wants ~80k
            DecisionResponse(action=DecisionAction.MODIFY,
                             capital_resolution={'choice': 'reduce_size'},
                             rationale="take what fits"),
        ])
        bbb_trades = result.symbol_results["BBB"].trades
        assert len(bbb_trades) == 1
        # AAA consumed 80k; BBB opened with the remaining ~20k at price 100.
        assert bbb_trades[0].quantity == pytest.approx(200.0, rel=1e-3)
        capital_request = [r for r in provider.requests
                           if r.kind == "CAPITAL_RESOLUTION"]
        assert len(capital_request) == 1

    def test_free_capital_full_close(self, tmp_path):
        result, _, store, _ = self._engine_run(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),   # AAA
            DecisionResponse(action=DecisionAction.ACCEPT),   # BBB
            DecisionResponse(action=DecisionAction.MODIFY,
                             capital_resolution={
                                 'choice': 'free_capital',
                                 'free_actions': [
                                     {'symbol': 'AAA', 'action': 'close'}]},
                             rationale="B is the better idea"),
        ])
        aaa_trades = result.symbol_results["AAA"].trades
        assert len(aaa_trades) == 1
        assert "Discretionary close to fund BBB" in aaa_trades[0].exit_reason
        bbb_trades = result.symbol_results["BBB"].trades
        assert len(bbb_trades) == 1
        # Full close freed ~80k, total ~100k available: full-size entry fits.
        assert bbb_trades[0].quantity * 100.0 == pytest.approx(80_000.0, rel=1e-2)
        # The funding leg is logged as a linked child event.
        funding = [e for e in store.load_events()
                   if e.event_kind == "DISCRETIONARY_FUNDING"]
        assert len(funding) == 1
        assert funding[0].symbol == "AAA"
        # Capital events include the discretionary close.
        kinds = {e.signal_type for e in result.capital_allocation_events}
        assert "DISCRETIONARY_CLOSE" in kinds

    def test_free_capital_trim_then_clamp(self, tmp_path):
        result, _, store, _ = self._engine_run(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.MODIFY,
                             capital_resolution={
                                 'choice': 'free_capital',
                                 'free_actions': [
                                     {'symbol': 'AAA', 'action': 'trim',
                                      'fraction': 0.5}]},
                             rationale="half out of A"),
        ])
        # AAA trimmed to half.
        aaa_positions = [e for e in store.load_events()
                         if e.event_kind == "DISCRETIONARY_FUNDING"]
        assert len(aaa_positions) == 1
        assert aaa_positions[0].signal_type == "PARTIAL_EXIT"
        # Freed ~40k + 20k cash = 60k < 80k wanted: clamped to affordable.
        bbb_trades = result.symbol_results["BBB"].trades
        assert len(bbb_trades) == 1
        assert bbb_trades[0].quantity * 100.0 == pytest.approx(60_000.0, rel=1e-2)
        kinds = {e.signal_type for e in result.capital_allocation_events}
        assert "DISCRETIONARY_TRIM" in kinds

    def test_capital_reject(self, tmp_path):
        result, _, store, _ = self._engine_run(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.REJECT,
                             capital_resolution={'choice': 'reject'},
                             rationale="won't sell A for this"),
        ])
        assert result.symbol_results["BBB"].trades == []
        # Parent BUY outcome shows the capital-stage rejection.
        events = store.load_events()
        buy_event = [e for e in events if e.symbol == "BBB"
                     and e.event_kind == "STRATEGY_SIGNAL"][0]
        outcomes = {o.event_id: o for o in store.load_outcomes()}
        assert outcomes[buy_event.event_id].executed is False
        assert "insufficient capital" in outcomes[buy_event.event_id].reason
        # The capital-resolution child event exists and links back.
        cap_events = [e for e in events if e.event_kind == "CAPITAL_RESOLUTION"]
        assert len(cap_events) == 1
        assert cap_events[0].portfolio_snapshot['parent_event_id'] == buy_event.event_id


class TestEngineExitsPortfolio:
    def test_stop_loss_logged_and_suppression_cleared(self, tmp_path):
        session, store, _ = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),
        ])
        data = _data_dict(symbols=("AAA",))
        data["AAA"].loc[5, ["close", "open", "high", "low"]] = 40.0  # below 50 stop
        result = PortfolioEngine(_config()).run(
            data, PortfolioTestStrategy(entry_bars={"AAA": 2}, size_frac=0.3,
                                        stop_frac=0.5),
            decision_session=session)
        trades = result.symbol_results["AAA"].trades
        assert len(trades) == 1
        assert trades[0].exit_reason == "Stop loss hit"
        engine_exits = [e for e in store.load_events()
                        if e.event_kind == "ENGINE_EXIT"]
        assert len(engine_exits) == 1
        assert engine_exits[0].signal_reason == "Stop loss hit"
