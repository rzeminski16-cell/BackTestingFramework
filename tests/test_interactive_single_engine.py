"""
Interactive-mode integration tests for the single-security engine.

Covers: accept-all parity with rules-only runs, reject/modify/defer,
NEXT_BAR_OPEN decide-at-signal / fill-at-open, final-bar handling,
engine-exit auto records, and outcome rows.
"""
import pandas as pd
import pytest

from Classes.Config.config import (
    BacktestConfig, CommissionConfig, CommissionMode, ExecutionTiming,
)
from Classes.Engine.single_security_engine import SingleSecurityEngine
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
    return BacktestConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0,
        **kwargs,
    )


class RoundTripStrategy(BaseStrategy):
    """Enter LONG on a fixed bar, exit via signal on a later fixed bar."""

    _validate_on_init = False

    def __init__(self, entry_bar=2, exit_bar=8, stop=90.0, **params):
        self._entry_bar = entry_bar
        self._exit_bar = exit_bar
        self._stop = stop
        super().__init__(**params)

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index == self._entry_bar:
            return Signal.buy(size=1.0, stop_loss=self._stop,
                              direction=TradeDirection.LONG, reason="entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return self._stop

    def generate_exit_signal(self, context):
        if context.current_index == self._exit_bar:
            return Signal.sell(reason="exit")
        return None

    def position_size(self, context, signal):
        return (context.available_capital * 0.5) / context.current_price


def _bars(closes, opens=None, lows=None, highs=None):
    n = len(closes)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "close": [float(c) for c in closes],
        "open": [float(o) for o in (opens or closes)],
        "high": [float(h) for h in (highs or closes)],
        "low": [float(l) for l in (lows or closes)],
        "volume": [1000.0] * n,
    })


def make_session(tmp_path, script=None, cooldown_bars=21):
    store = InteractiveRunStore(tmp_path / "run")
    manifest = BacktestRunManifest(
        run_id="t", backtest_name="t", mode="interactive",
        engine_type="single", cooldown_bars=cooldown_bars)
    store.write_manifest(manifest)
    provider = ScriptedDecisionProvider(script)
    return InteractiveSession(store, provider, manifest), store, provider


DATA = _bars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])


class TestAcceptAllParity:
    def test_accept_all_equals_rules_only(self, tmp_path):
        baseline = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy())

        session, store, _ = make_session(tmp_path)  # scripted default: accept
        interactive = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy(), decision_session=session)

        assert len(interactive.trades) == len(baseline.trades) == 1
        bt, it = baseline.trades[0], interactive.trades[0]
        assert it.entry_price == bt.entry_price
        assert it.quantity == bt.quantity
        assert it.pl == bt.pl
        assert interactive.final_equity == baseline.final_equity
        pd.testing.assert_frame_equal(interactive.equity_curve,
                                      baseline.equity_curve)
        # Both the BUY and the SELL were logged with outcomes.
        events = store.load_events()
        assert [e.signal_type for e in events] == ["BUY", "SELL"]
        outcomes = {o.event_id: o for o in store.load_outcomes()}
        assert all(outcomes[e.event_id].executed for e in events)

    def test_none_session_signature_unchanged(self):
        result = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy(), decision_session=None)
        assert len(result.trades) == 1


class TestDecisions:
    def test_reject_drops_trade(self, tmp_path):
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.REJECT,
                                        rationale="not convinced")])
        result = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy(), decision_session=session)
        assert result.trades == []
        assert result.final_equity == 100_000.0
        outcome = store.load_outcomes()[0]
        assert outcome.executed is False
        assert "Rejected" in outcome.reason

    def test_modify_halves_quantity(self, tmp_path):
        baseline = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy())
        session, _, _ = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.MODIFY, size_factor=0.5,
                             rationale="half size"),
            DecisionResponse(action=DecisionAction.ACCEPT),
        ])
        result = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy(), decision_session=session)
        assert len(result.trades) == 1
        assert result.trades[0].quantity == pytest.approx(
            baseline.trades[0].quantity * 0.5)

    def test_modify_stop_override(self, tmp_path):
        session, store, _ = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.MODIFY,
                             modified_stop_loss=95.0, rationale="tighter stop"),
        ])
        # Price dips to 94 on bar 5: the modified 95 stop triggers, the
        # original 90 stop would not.
        data = _bars([100, 101, 102, 103, 104, 94, 106, 107, 108, 109, 110, 111])
        result = SingleSecurityEngine(_config()).run(
            "T", data, RoundTripStrategy(), decision_session=session)
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "Stop loss hit"
        # The engine stop-out was logged as an auto-applied ENGINE_EXIT.
        events = store.load_events()
        assert any(e.event_kind == "ENGINE_EXIT" for e in events)

    def test_invalid_modified_stop_raises(self, tmp_path):
        session, _, _ = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.MODIFY,
                             modified_stop_loss=150.0),
        ])
        with pytest.raises(ValueError, match="must be below"):
            SingleSecurityEngine(_config()).run(
                "T", DATA.copy(), RoundTripStrategy(), decision_session=session)

    def test_defer_skips_today_reprompts_next_signal(self, tmp_path):
        class AlwaysEnterStrategy(RoundTripStrategy):
            def generate_entry_signal(self, context):
                if context.current_index >= 2:
                    return Signal.buy(size=1.0, stop_loss=self._stop,
                                      direction=TradeDirection.LONG,
                                      reason="entry")
                return None

        session, _, provider = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.DEFER, rationale="researching"),
            DecisionResponse(action=DecisionAction.ACCEPT),
        ])
        result = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), AlwaysEnterStrategy(exit_bar=99),
            decision_session=session)
        # Deferred on bar 2, accepted on bar 3.
        assert len(result.trades) == 1
        assert result.trades[0].entry_date == DATA['date'].iloc[3]
        assert len(provider.requests) == 2


class TestEngineExits:
    def test_stop_loss_hit_logged_as_engine_exit(self, tmp_path):
        session, store, provider = make_session(tmp_path)
        data = _bars([100, 101, 102, 103, 85, 105, 106, 107, 108, 109, 110, 111])
        result = SingleSecurityEngine(_config()).run(
            "T", data, RoundTripStrategy(stop=90.0), decision_session=session)
        assert result.trades[0].exit_reason == "Stop loss hit"
        engine_exits = [e for e in store.load_events()
                        if e.event_kind == "ENGINE_EXIT"]
        assert len(engine_exits) == 1
        assert engine_exits[0].signal_reason == "Stop loss hit"
        # Stop exits never prompt: the only prompt was the entry.
        assert len(provider.requests) == 1
        outcome = [o for o in store.load_outcomes()
                   if o.event_id == engine_exits[0].event_id][0]
        assert outcome.executed is True

    def test_end_of_backtest_close_logged(self, tmp_path):
        session, store, _ = make_session(tmp_path)
        result = SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy(exit_bar=99),
            decision_session=session)
        assert result.trades[0].exit_reason == "End of backtest period"
        engine_exits = [e for e in store.load_events()
                        if e.event_kind == "ENGINE_EXIT"]
        assert len(engine_exits) == 1
        assert engine_exits[0].signal_reason == "End of backtest period"


class TestNextBarOpen:
    def test_decision_at_signal_fill_at_next_open(self, tmp_path):
        session, store, _ = make_session(tmp_path)
        opens = [100, 101, 102, 250, 104, 105, 106, 107, 108, 109, 110, 111]
        data = _bars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                     opens=opens)
        result = SingleSecurityEngine(
            _config(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)).run(
            "T", data, RoundTripStrategy(entry_bar=2, exit_bar=8),
            decision_session=session)
        assert len(result.trades) == 1
        # Filled at bar 3's open (250), decided on bar 2.
        assert result.trades[0].entry_price == 250.0
        event = store.load_events()[0]
        assert event.bar_date == "2024-01-03"  # signal bar, not fill bar
        outcome = store.load_outcomes()[0]
        assert outcome.executed is True
        assert outcome.executed_price == pytest.approx(250.0)

    def test_rejected_signal_never_defers(self, tmp_path):
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.REJECT)])
        result = SingleSecurityEngine(
            _config(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)).run(
            "T", DATA.copy(), RoundTripStrategy(), decision_session=session)
        assert result.trades == []
        assert store.load_outcomes()[0].executed is False

    def test_final_bar_signal_logged_not_prompted(self, tmp_path):
        session, store, provider = make_session(tmp_path)
        result = SingleSecurityEngine(
            _config(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)).run(
            "T", DATA.copy(), RoundTripStrategy(entry_bar=len(DATA) - 1),
            decision_session=session)
        assert result.trades == []
        assert provider.requests == []  # never prompted
        events = store.load_events()
        assert len(events) == 1
        outcome = store.load_outcomes()[0]
        assert outcome.executed is False
        assert "Final-bar" in outcome.reason

    def test_gap_across_stop_outcome_recorded(self, tmp_path):
        session, store, _ = make_session(tmp_path)
        # Signal on bar 2 with stop 90; bar 3 opens at 80 (gap across stop).
        opens = [100, 101, 102, 80, 104, 105, 106, 107, 108, 109, 110, 111]
        data = _bars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                     opens=opens)
        result = SingleSecurityEngine(
            _config(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)).run(
            "T", data, RoundTripStrategy(entry_bar=2, exit_bar=99, stop=90.0),
            decision_session=session)
        assert result.trades == []
        outcome = store.load_outcomes()[0]
        assert outcome.executed is False
        assert "gapped across" in outcome.reason


class TestEventContents:
    def test_buy_event_snapshot_and_prompt_flow(self, tmp_path):
        session, store, provider = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT,
                             prompt_text="Assume today is 2024-01-03...",
                             response_summary="ok"),
            DecisionResponse(action=DecisionAction.ACCEPT),
        ])
        SingleSecurityEngine(_config()).run(
            "T", DATA.copy(), RoundTripStrategy(), decision_session=session)
        event = store.load_events()[0]
        assert event.symbol == "T"
        assert event.signal_type == "BUY"
        assert event.proposed_stop_loss == 90.0
        assert event.technical_snapshot['close'] == 102.0
        assert event.portfolio_snapshot['available_capital'] == 100_000.0
        assert event.portfolio_snapshot['num_positions'] == 0
        prompts = store.load_prompts()
        assert len(prompts) == 1 and prompts[0].response_summary == "ok"
        # SELL event carries the open position in its snapshot.
        sell_event = store.load_events()[1]
        assert sell_event.signal_type == "SELL"
        assert sell_event.portfolio_snapshot['num_positions'] == 1
        assert sell_event.portfolio_snapshot['positions'][0]['symbol'] == "T"
