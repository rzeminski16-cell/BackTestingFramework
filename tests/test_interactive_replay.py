"""
End-to-end resume/replay tests: abort an interactive run mid-decision,
resume it by re-running the engine with the logged decisions, and verify
the final result is identical to an uninterrupted run.
"""
import pandas as pd
import pytest

from Classes.Config.config import (
    BacktestConfig, CommissionConfig, CommissionMode, PortfolioConfig,
)
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Interactive.decision_provider import ScriptedDecisionProvider
from Classes.Interactive.models import (
    BacktestRunManifest,
    DecisionAction,
    DecisionResponse,
    ReplayMismatchError,
    RunAbortedByUser,
    build_data_fingerprints,
    verify_data_fingerprints,
)
from Classes.Interactive.session import InteractiveSession
from Classes.Interactive.store import InteractiveRunStore
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy


def _single_config():
    return BacktestConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0)


def _portfolio_config():
    return PortfolioConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0)


class CyclingStrategy(BaseStrategy):
    """Enters whenever flat, exits after holding for three days."""

    _validate_on_init = False

    def __init__(self, **params):
        super().__init__(**params)

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index >= 2:
            return Signal.buy(size=1.0,
                              stop_loss=context.current_price * 0.5,
                              direction=TradeDirection.LONG, reason="entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return context.current_price * 0.5

    def generate_exit_signal(self, context):
        if context.position is not None:
            days_held = (context.current_date - context.position.entry_date).days
            if days_held >= 3:
                return Signal.sell(reason="time exit")
        return None

    def position_size(self, context, signal):
        return (context.available_capital * 0.5) / context.current_price


def _data(n=20, price=100.0):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "close": [price + i for i in range(n)],
        "open": [price + i for i in range(n)],
        "high": [price + i for i in range(n)],
        "low": [price + i for i in range(n)],
        "volume": [1000.0] * n,
    })


def _make_session(tmp_path, name, script, resume_records=None,
                  engine_type="single", data_by_symbol=None):
    store = InteractiveRunStore(tmp_path / name)
    manifest = store.read_manifest()
    if manifest is None:
        manifest = BacktestRunManifest(
            run_id=name, backtest_name=name, mode="interactive",
            engine_type=engine_type,
            data_fingerprints=build_data_fingerprints(data_by_symbol or {}))
        store.write_manifest(manifest)
    provider = ScriptedDecisionProvider(script)
    return InteractiveSession(store, provider, manifest,
                              resume_records=resume_records), store, provider


def _assert_results_equal(a, b):
    assert a.final_equity == b.final_equity
    assert len(a.trades) == len(b.trades)
    for ta, tb in zip(a.trades, b.trades):
        assert (ta.symbol, str(ta.entry_date), ta.entry_price, ta.quantity,
                str(ta.exit_date), ta.exit_price, ta.pl) == \
               (tb.symbol, str(tb.entry_date), tb.entry_price, tb.quantity,
                str(tb.exit_date), tb.exit_price, tb.pl)
    pd.testing.assert_frame_equal(a.equity_curve, b.equity_curve)


FULL_SCRIPT = [
    DecisionResponse(action=DecisionAction.ACCEPT),   # entry 1
    DecisionResponse(action=DecisionAction.ACCEPT),   # exit 1
    DecisionResponse(action=DecisionAction.REJECT,    # entry 2 rejected
                     rationale="skip this one"),
    # remainder: scripted default = accept
]


class TestSingleEngineResume:
    def test_abort_and_resume_matches_uninterrupted(self, tmp_path):
        data = _data()

        # Uninterrupted reference run.
        ref_session, _, _ = _make_session(
            tmp_path, "ref", list(FULL_SCRIPT), data_by_symbol={"T": data})
        reference = SingleSecurityEngine(_single_config()).run(
            "T", data.copy(), CyclingStrategy(), decision_session=ref_session)

        # Run that gets aborted at the third prompt.
        abort_script = [
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.ACCEPT),
            DecisionResponse(action=DecisionAction.ABORT),
        ]
        session, store, _ = _make_session(
            tmp_path, "paused", abort_script, data_by_symbol={"T": data})
        with pytest.raises(RunAbortedByUser):
            SingleSecurityEngine(_single_config()).run(
                "T", data.copy(), CyclingStrategy(), decision_session=session)
        session.finalize("paused")
        assert store.read_manifest().status == "paused"

        # Resume: replay the two logged decisions, then continue live with
        # the same remaining choices as the reference run.
        resume_session, store2, provider = _make_session(
            tmp_path, "paused",
            [DecisionResponse(action=DecisionAction.REJECT,
                              rationale="skip this one")],
            resume_records=store.load_for_resume())
        resumed = SingleSecurityEngine(_single_config()).run(
            "T", data.copy(), CyclingStrategy(),
            decision_session=resume_session)
        resume_session.finalize("completed")

        _assert_results_equal(resumed, reference)
        # The replayed prefix never prompted again: the first live request
        # was the third decision.
        first_ids = [r.event.event_id for r in provider.requests]
        assert min(first_ids) > 2
        # Log contains each decision exactly once.
        decisions = store2.load_decisions()
        assert len(decisions) == len({d.decision_id for d in decisions})

    def test_resume_with_changed_data_raises(self, tmp_path):
        data = _data()
        session, store, _ = _make_session(
            tmp_path, "run", [DecisionResponse(action=DecisionAction.ACCEPT),
                              DecisionResponse(action=DecisionAction.ABORT)],
            data_by_symbol={"T": data})
        with pytest.raises(RunAbortedByUser):
            SingleSecurityEngine(_single_config()).run(
                "T", data.copy(), CyclingStrategy(), decision_session=session)

        # Data changed: the entry price on the signal bar differs, so the
        # regenerated event's fingerprint (stop level) diverges.
        changed = data.copy()
        changed.loc[2, "close"] = 500.0
        resume_session, _, _ = _make_session(
            tmp_path, "run", None, resume_records=store.load_for_resume())
        with pytest.raises(ReplayMismatchError):
            SingleSecurityEngine(_single_config()).run(
                "T", changed, CyclingStrategy(),
                decision_session=resume_session)

    def test_verify_data_fingerprints(self, tmp_path):
        data = _data()
        manifest = BacktestRunManifest(
            run_id="r", backtest_name="r", mode="interactive",
            engine_type="single",
            data_fingerprints=build_data_fingerprints({"T": data}))
        verify_data_fingerprints(manifest, {"T": data})  # no raise

        changed = data.copy()
        changed.loc[len(changed)] = changed.iloc[-1]  # extra row
        with pytest.raises(ReplayMismatchError, match="rows changed"):
            verify_data_fingerprints(manifest, {"T": changed})
        with pytest.raises(ReplayMismatchError, match="no longer available"):
            verify_data_fingerprints(manifest, {})


class TestPortfolioEngineResume:
    def _data_dict(self):
        return {"AAA": _data(), "BBB": _data(price=50.0)}

    def test_abort_and_resume_matches_uninterrupted(self, tmp_path):
        # Reference run.
        ref_session, _, _ = _make_session(
            tmp_path, "ref", list(FULL_SCRIPT), engine_type="portfolio",
            data_by_symbol=self._data_dict())
        reference = PortfolioEngine(_portfolio_config()).run(
            self._data_dict(), CyclingStrategy(), decision_session=ref_session)

        # Aborted run.
        session, store, _ = _make_session(
            tmp_path, "paused", [
                DecisionResponse(action=DecisionAction.ACCEPT),
                DecisionResponse(action=DecisionAction.ACCEPT),
                DecisionResponse(action=DecisionAction.ABORT),
            ], engine_type="portfolio", data_by_symbol=self._data_dict())
        with pytest.raises(RunAbortedByUser):
            PortfolioEngine(_portfolio_config()).run(
                self._data_dict(), CyclingStrategy(), decision_session=session)
        session.finalize("paused")

        # Resume.
        resume_session, _, provider = _make_session(
            tmp_path, "paused",
            [DecisionResponse(action=DecisionAction.REJECT,
                              rationale="skip this one")],
            resume_records=store.load_for_resume(), engine_type="portfolio")
        resumed = PortfolioEngine(_portfolio_config()).run(
            self._data_dict(), CyclingStrategy(),
            decision_session=resume_session)

        assert resumed.final_equity == reference.final_equity
        pd.testing.assert_frame_equal(resumed.portfolio_equity_curve,
                                      reference.portfolio_equity_curve)
        ref_trades = sorted(
            (t for r in reference.symbol_results.values() for t in r.trades),
            key=lambda t: (str(t.entry_date), t.symbol))
        res_trades = sorted(
            (t for r in resumed.symbol_results.values() for t in r.trades),
            key=lambda t: (str(t.entry_date), t.symbol))
        assert [(t.symbol, str(t.entry_date), t.quantity, t.pl)
                for t in ref_trades] == \
               [(t.symbol, str(t.entry_date), t.quantity, t.pl)
                for t in res_trades]
