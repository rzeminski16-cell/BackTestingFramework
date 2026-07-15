"""
Engine-level tests for reject-cooldown suppression: a rejected,
continuously-firing signal auto-suppresses for cooldown_bars firings,
then re-prompts; a gap in firing resets immediately.
"""
import pandas as pd
import pytest

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Interactive.decision_provider import ScriptedDecisionProvider
from Classes.Interactive.models import (
    BacktestRunManifest,
    DecisionAction,
    DecisionResponse,
)
from Classes.Interactive.session import InteractiveSession
from Classes.Interactive.store import InteractiveRunStore
from Classes.Interactive.suppression import RejectCooldownTracker
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy


def _config():
    return BacktestConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0)


class FiringPatternStrategy(BaseStrategy):
    """Fires an entry signal on every bar index in ``firing_bars`` when flat."""

    _validate_on_init = False

    def __init__(self, firing_bars=None, **params):
        self._firing_bars = set(firing_bars or [])
        super().__init__(**params)

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index in self._firing_bars:
            return Signal.buy(size=1.0, stop_loss=50.0,
                              direction=TradeDirection.LONG, reason="entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return 50.0

    def generate_exit_signal(self, context):
        return None

    def position_size(self, context, signal):
        return (context.available_capital * 0.5) / context.current_price


def _data(n=30):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "close": [100.0] * n,
        "open": [100.0] * n,
        "high": [100.0] * n,
        "low": [100.0] * n,
        "volume": [1000.0] * n,
    })


def make_session(tmp_path, script, cooldown_bars):
    store = InteractiveRunStore(tmp_path / "run")
    manifest = BacktestRunManifest(
        run_id="c", backtest_name="c", mode="interactive",
        engine_type="single", cooldown_bars=cooldown_bars)
    store.write_manifest(manifest)
    provider = ScriptedDecisionProvider(script)
    return InteractiveSession(store, provider, manifest), store, provider


class TestEngineCooldown:
    def test_sustained_signal_reprompts_after_cooldown(self, tmp_path):
        # Fires continuously from bar 2. Reject at 2 -> bars 3,4,5
        # suppressed (cooldown 3) -> re-prompt at 6 (default accept).
        session, store, provider = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.REJECT,
                                        rationale="no")],
            cooldown_bars=3)
        result = SingleSecurityEngine(_config()).run(
            "T", _data(), FiringPatternStrategy(firing_bars=range(2, 30)),
            decision_session=session)

        assert len(result.trades) == 1
        assert result.trades[0].entry_date == _data()['date'].iloc[6]
        assert len(provider.requests) == 2  # bar 2 and bar 6

        decisions = store.load_decisions()
        actions = [d.action for d in decisions
                   if d.action in (DecisionAction.REJECT,
                                   DecisionAction.AUTO_SUPPRESSED,
                                   DecisionAction.ACCEPT)]
        assert actions[:5] == [DecisionAction.REJECT,
                               DecisionAction.AUTO_SUPPRESSED,
                               DecisionAction.AUTO_SUPPRESSED,
                               DecisionAction.AUTO_SUPPRESSED,
                               DecisionAction.ACCEPT]
        # Suppressed records link back to the originating reject.
        reject_id = decisions[0].decision_id
        suppressed = [d for d in decisions
                      if d.action == DecisionAction.AUTO_SUPPRESSED]
        assert all(d.suppressed_by == reject_id for d in suppressed)
        # Every firing has an outcome row (executed=False while suppressed).
        outcomes = store.load_outcomes()
        assert sum(1 for o in outcomes if not o.executed) == 4

    def test_gap_in_firing_reprompts_immediately(self, tmp_path):
        # Fires at bars 2,3 then stops; fires again at 6.
        session, _, provider = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.REJECT)],
            cooldown_bars=21)
        result = SingleSecurityEngine(_config()).run(
            "T", _data(), FiringPatternStrategy(firing_bars=[2, 3, 6]),
            decision_session=session)
        # Rejected at 2, suppressed at 3, fresh prompt at 6 (accepted).
        assert len(provider.requests) == 2
        assert len(result.trades) == 1
        assert result.trades[0].entry_date == _data()['date'].iloc[6]

    def test_defer_never_suppresses(self, tmp_path):
        session, _, provider = make_session(
            tmp_path, [DecisionResponse(action=DecisionAction.DEFER)] * 3,
            cooldown_bars=21)
        result = SingleSecurityEngine(_config()).run(
            "T", _data(), FiringPatternStrategy(firing_bars=[2, 3, 4, 5]),
            decision_session=session)
        # Deferred three times, accepted on the fourth firing.
        assert len(provider.requests) == 4
        assert len(result.trades) == 1
        assert result.trades[0].entry_date == _data()['date'].iloc[5]


class TestTrackerUnit:
    def test_default_cooldown_is_21(self):
        tracker = RejectCooldownTracker()
        assert tracker.cooldown_bars == 21
        tracker.on_reject("A", "BUY", decision_id=1, bar_index=100)
        # 21 continuous firings suppressed, the 22nd prompts.
        for i in range(1, 22):
            assert tracker.check("A", "BUY", 100 + i) == 1
        assert tracker.check("A", "BUY", 122) is None

    def test_negative_cooldown_rejected(self):
        with pytest.raises(ValueError):
            RejectCooldownTracker(cooldown_bars=-1)

    def test_zero_cooldown_always_prompts(self):
        tracker = RejectCooldownTracker(cooldown_bars=0)
        tracker.on_reject("A", "BUY", decision_id=1, bar_index=5)
        assert tracker.check("A", "BUY", 6) is None
