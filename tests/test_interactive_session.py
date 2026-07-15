"""
Tests for InteractiveSession: decision flow, suppression, capital
resolution, abort, persistence, and basic replay behaviour.
"""
import numpy as np
import pandas as pd
import pytest

from Classes.Data.historical_data_view import HistoricalDataView
from Classes.Interactive.decision_provider import ScriptedDecisionProvider
from Classes.Interactive.models import (
    BacktestRunManifest,
    CapitalOptions,
    DecisionAction,
    DecisionResponse,
    DecisionSource,
    ReplayMismatchError,
    RunAbortedByUser,
)
from Classes.Interactive.session import InteractiveSession
from Classes.Interactive.store import InteractiveRunStore
from Classes.Models.signal import Signal, SignalType
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.strategy_context import StrategyContext


def make_data(n=300, start_price=100.0):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = start_price + np.linspace(0, 20, n)
    return pd.DataFrame({
        'date': dates,
        'open': prices - 0.5,
        'high': prices + 1.0,
        'low': prices - 1.0,
        'close': prices,
        'volume': np.full(n, 1000.0),
        'ema_20_ema': prices - 0.2,
    })


def make_context(data, index=200, symbol="AAPL", position=None,
                 capital=100000.0, equity=100000.0):
    return StrategyContext(
        data=HistoricalDataView(data, valid_end_index=index),
        current_index=index,
        current_price=float(data['close'].iloc[index]),
        current_date=data['date'].iloc[index],
        position=position,
        available_capital=capital,
        total_equity=equity,
        symbol=symbol,
    )


def make_session(tmp_path, script=None, cooldown_bars=21, resume_records=None):
    store = InteractiveRunStore(tmp_path / "run")
    manifest = BacktestRunManifest(
        run_id="test_run", backtest_name="test", mode="interactive",
        engine_type="single", cooldown_bars=cooldown_bars,
    )
    store.write_manifest(manifest)
    provider = ScriptedDecisionProvider(script)
    session = InteractiveSession(store, provider, manifest,
                                 resume_records=resume_records)
    return session, store, provider


def buy_signal(size=0.5, stop=95.0):
    return Signal.buy(size=size, stop_loss=stop, reason="EMA crossover")


class TestResolveSignal:
    def test_accept_returns_signal_unchanged(self, tmp_path):
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.ACCEPT)])
        data = make_data()
        resolved = session.resolve_signal(buy_signal(), make_context(data))
        assert resolved.effective_signal.type == SignalType.BUY
        assert resolved.effective_signal.size == 0.5
        assert resolved.size_factor == 1.0
        assert resolved.record.action == DecisionAction.ACCEPT
        assert len(store.load_events()) == 1
        assert len(store.load_decisions()) == 1

    def test_modify_scales_size_and_overrides_stop(self, tmp_path):
        session, _, _ = make_session(tmp_path, [DecisionResponse(
            action=DecisionAction.MODIFY, size_factor=0.5,
            modified_stop_loss=90.0, rationale="half size")])
        resolved = session.resolve_signal(buy_signal(size=0.8, stop=95.0),
                                          make_context(make_data()))
        assert resolved.effective_signal.size == pytest.approx(0.4)
        assert resolved.effective_signal.stop_loss == 90.0
        assert resolved.size_factor == 0.5

    def test_reject_returns_hold(self, tmp_path):
        session, store, _ = make_session(tmp_path, [DecisionResponse(
            action=DecisionAction.REJECT, rationale="earnings tomorrow")])
        resolved = session.resolve_signal(buy_signal(), make_context(make_data()))
        assert resolved.effective_signal.type == SignalType.HOLD
        assert "earnings tomorrow" in store.load_decisions()[0].rationale

    def test_defer_returns_hold_and_no_suppression(self, tmp_path):
        session, _, provider = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.DEFER),
            DecisionResponse(action=DecisionAction.ACCEPT),
        ])
        data = make_data()
        r1 = session.resolve_signal(buy_signal(), make_context(data, index=200))
        assert r1.effective_signal.type == SignalType.HOLD
        # Signal fires again the very next bar: prompts again immediately.
        r2 = session.resolve_signal(buy_signal(), make_context(data, index=201))
        assert r2.effective_signal.type == SignalType.BUY
        assert len(provider.requests) == 2

    def test_event_snapshot_contents(self, tmp_path):
        session, store, provider = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.ACCEPT)])
        data = make_data()
        session.resolve_signal(buy_signal(), make_context(data, index=260),
                               portfolio_snapshot={'available_capital': 5000.0},
                               technical_columns=['ema_20_ema'])
        event = store.load_events()[0]
        assert event.symbol == "AAPL"
        assert event.bar_index == 260
        assert event.technical_snapshot['close'] > 0
        assert 'ema_20_ema' in event.technical_snapshot
        assert 'pct_change_21d' in event.technical_snapshot
        assert 'dist_from_252d_high_pct' in event.technical_snapshot
        assert event.portfolio_snapshot['available_capital'] == 5000.0
        # Chart slice went to the provider, capped and look-ahead-safe.
        chart = provider.requests[0].chart_data
        assert len(chart) == 150
        assert chart['close'].iloc[-1] == pytest.approx(
            float(data['close'].iloc[260]))

    def test_abort_raises_and_leaves_dangling_event(self, tmp_path):
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.ABORT)])
        with pytest.raises(RunAbortedByUser):
            session.resolve_signal(buy_signal(), make_context(make_data()))
        assert len(store.load_events()) == 1
        assert len(store.load_decisions()) == 0
        assert store.load_for_resume() == []

    def test_prompt_persisted_and_linked(self, tmp_path):
        session, store, _ = make_session(tmp_path, [DecisionResponse(
            action=DecisionAction.ACCEPT,
            prompt_text="Assume today is 2020-10-06...",
            response_summary="Fundamentals fine.")])
        session.resolve_signal(buy_signal(), make_context(make_data()))
        prompts = store.load_prompts()
        assert len(prompts) == 1
        assert prompts[0].response_summary == "Fundamentals fine."
        assert store.load_decisions()[0].prompt_id == prompts[0].prompt_id


class TestSuppression:
    def test_reject_cooldown_suppresses_continuous_firing(self, tmp_path):
        session, store, provider = make_session(
            tmp_path,
            [DecisionResponse(action=DecisionAction.REJECT, rationale="no")],
            cooldown_bars=3)
        data = make_data()
        session.resolve_signal(buy_signal(), make_context(data, index=200))
        # Three continuous firings are auto-suppressed.
        for i in (201, 202, 203):
            resolved = session.resolve_signal(buy_signal(), make_context(data, index=i))
            assert resolved.record.action == DecisionAction.AUTO_SUPPRESSED
            assert resolved.record.suppressed_by == 1
        # Fourth continuous firing re-prompts (cooldown exhausted).
        resolved = session.resolve_signal(buy_signal(), make_context(data, index=204))
        assert resolved.record.action == DecisionAction.ACCEPT  # script exhausted -> default
        assert len(provider.requests) == 2
        # Every firing logged.
        assert len(store.load_decisions()) == 5

    def test_gap_in_firing_resets_suppression(self, tmp_path):
        session, _, provider = make_session(
            tmp_path,
            [DecisionResponse(action=DecisionAction.REJECT)],
            cooldown_bars=21)
        data = make_data()
        session.resolve_signal(buy_signal(), make_context(data, index=200))
        r = session.resolve_signal(buy_signal(), make_context(data, index=201))
        assert r.record.action == DecisionAction.AUTO_SUPPRESSED
        # Signal skips bar 202 and fires at 203: treated as fresh, prompts.
        r = session.resolve_signal(buy_signal(), make_context(data, index=203))
        assert r.record.action != DecisionAction.AUTO_SUPPRESSED
        assert len(provider.requests) == 2

    def test_position_open_clears_buy_suppression(self, tmp_path):
        session, _, provider = make_session(
            tmp_path,
            [DecisionResponse(action=DecisionAction.REJECT)],
            cooldown_bars=21)
        data = make_data()
        session.resolve_signal(buy_signal(), make_context(data, index=200))
        session.on_position_opened("AAPL")
        r = session.resolve_signal(buy_signal(), make_context(data, index=201))
        assert r.record.action != DecisionAction.AUTO_SUPPRESSED
        assert len(provider.requests) == 2


class TestCapitalResolution:
    def _options(self):
        return CapitalOptions(
            required_capital=20000.0, available_capital=8000.0,
            affordable_fraction=0.4,
            positions=[{'symbol': 'MSFT', 'unrealized_pl_pct': -2.0}])

    def test_reduce_size(self, tmp_path):
        session, store, _ = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.MODIFY,
                             capital_resolution={'choice': 'reduce_size'}),
        ])
        resolved = session.resolve_signal(buy_signal(), make_context(make_data()))
        resolution = session.resolve_capital(resolved, self._options())
        assert resolution == {'choice': 'reduce_size'}
        events = store.load_events()
        assert events[1].event_kind == "CAPITAL_RESOLUTION"
        assert events[1].portfolio_snapshot['parent_event_id'] == events[0].event_id

    def test_capital_reject_starts_cooldown(self, tmp_path):
        session, _, provider = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.REJECT,
                             capital_resolution={'choice': 'reject'}),
        ], cooldown_bars=21)
        data = make_data()
        resolved = session.resolve_signal(buy_signal(), make_context(data, index=200))
        resolution = session.resolve_capital(resolved, self._options())
        assert resolution['choice'] == 'reject'
        # Next continuous firing is suppressed without prompting.
        r = session.resolve_signal(buy_signal(), make_context(data, index=201))
        assert r.record.action == DecisionAction.AUTO_SUPPRESSED
        assert len(provider.requests) == 2


class TestOutcomesAndAuto:
    def test_record_outcome(self, tmp_path):
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.ACCEPT)])
        resolved = session.resolve_signal(buy_signal(), make_context(make_data()))
        session.record_outcome(resolved.event.event_id, executed=True,
                               quantity=42.0, price=101.0)
        outcome = store.load_outcomes()[0]
        assert outcome.executed and outcome.executed_quantity == 42.0

    def test_log_auto_engine_exit(self, tmp_path):
        session, store, provider = make_session(tmp_path)
        event_id = session.log_auto(
            "ENGINE_EXIT", symbol="AAPL", bar_date="2020-06-01", bar_index=99,
            signal_type="SELL", reason="Stop loss hit")
        assert provider.requests == []  # never prompts
        decisions = store.load_decisions()
        assert decisions[0].action == DecisionAction.AUTO_APPLIED
        assert decisions[0].event_id == event_id

    def test_finalize_updates_manifest(self, tmp_path):
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.ACCEPT)])
        session.resolve_signal(buy_signal(), make_context(make_data()))
        session.finalize("completed")
        manifest = store.read_manifest()
        assert manifest.status == "completed"
        assert manifest.counts['events'] == 1
        assert manifest.counts['decisions'] == 1


class TestReplay:
    def _run_live(self, tmp_path, responses, indices):
        session, store, _ = make_session(tmp_path, responses)
        data = make_data()
        results = [session.resolve_signal(buy_signal(), make_context(data, index=i))
                   for i in indices]
        for r in results:
            session.record_outcome(r.event.event_id,
                                   executed=r.record.action == DecisionAction.ACCEPT)
        return store, results

    def test_replay_reproduces_decisions_without_prompting(self, tmp_path):
        store, live = self._run_live(tmp_path, [
            DecisionResponse(action=DecisionAction.REJECT, rationale="no"),
            DecisionResponse(action=DecisionAction.ACCEPT),
        ], indices=[200, 205])

        manifest = store.read_manifest()
        provider = ScriptedDecisionProvider()
        session = InteractiveSession(store, provider, manifest,
                                     resume_records=store.load_for_resume())
        data = make_data()
        r1 = session.resolve_signal(buy_signal(), make_context(data, index=200))
        r2 = session.resolve_signal(buy_signal(), make_context(data, index=205))
        assert provider.requests == []  # nothing prompted
        assert r1.record.action == DecisionAction.REJECT
        assert r2.record.action == DecisionAction.ACCEPT
        assert r1.effective_signal.type == SignalType.HOLD
        assert r2.effective_signal.type == SignalType.BUY
        # No duplicate rows were appended.
        assert len(store.load_events()) == 2
        assert len(store.load_decisions()) == 2
        # Post-replay the session goes live again and appends.
        r3 = session.resolve_signal(buy_signal(), make_context(data, index=210))
        assert r3.record.action == DecisionAction.ACCEPT  # scripted default
        assert len(store.load_decisions()) == 3
        assert r3.record.decision_id == 3

    def test_replay_outcomes_not_duplicated(self, tmp_path):
        store, _ = self._run_live(
            tmp_path, [DecisionResponse(action=DecisionAction.ACCEPT)],
            indices=[200])
        manifest = store.read_manifest()
        session = InteractiveSession(store, ScriptedDecisionProvider(), manifest,
                                     resume_records=store.load_for_resume())
        r = session.resolve_signal(buy_signal(), make_context(make_data(), index=200))
        session.record_outcome(r.event.event_id, executed=True)
        assert len(store.load_outcomes()) == 1  # not re-appended

    def test_replay_mismatch_raises(self, tmp_path):
        store, _ = self._run_live(
            tmp_path, [DecisionResponse(action=DecisionAction.ACCEPT)],
            indices=[200])
        manifest = store.read_manifest()
        session = InteractiveSession(store, ScriptedDecisionProvider(), manifest,
                                     resume_records=store.load_for_resume())
        # Different signal facts (size changed) -> fingerprint mismatch.
        with pytest.raises(ReplayMismatchError):
            session.resolve_signal(buy_signal(size=0.9),
                                   make_context(make_data(), index=200))

    def test_replay_rebuilds_suppression_state(self, tmp_path):
        # Live: reject at 200, suppressed at 201.
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.REJECT)],
            cooldown_bars=21)
        data = make_data()
        session.resolve_signal(buy_signal(), make_context(data, index=200))
        session.resolve_signal(buy_signal(), make_context(data, index=201))

        manifest = store.read_manifest()
        provider = ScriptedDecisionProvider()
        resumed = InteractiveSession(store, provider, manifest,
                                     resume_records=store.load_for_resume())
        resumed.resolve_signal(buy_signal(), make_context(data, index=200))
        resumed.resolve_signal(buy_signal(), make_context(data, index=201))
        # Live continuation: bar 202 is still inside the rebuilt cooldown.
        r = resumed.resolve_signal(buy_signal(), make_context(data, index=202))
        assert r.record.action == DecisionAction.AUTO_SUPPRESSED
        assert provider.requests == []

    def test_dangling_event_regenerates_live_without_duplicate_row(self, tmp_path):
        # Live run aborts mid-decision: event logged, no decision.
        session, store, _ = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.ABORT)])
        data = make_data()
        with pytest.raises(RunAbortedByUser):
            session.resolve_signal(buy_signal(), make_context(data, index=200))
        assert len(store.load_events()) == 1

        manifest = store.read_manifest()
        # resume_records is [] (the dangling event has no decision), but
        # passing it non-None marks the session as resuming so the
        # already-logged event id is not appended twice.
        resumed = InteractiveSession(store, ScriptedDecisionProvider(), manifest,
                                     resume_records=store.load_for_resume())
        r = resumed.resolve_signal(buy_signal(), make_context(data, index=200))
        assert r.record.action == DecisionAction.ACCEPT
        assert len(store.load_events()) == 1  # no duplicate
        assert len(store.load_decisions()) == 1
