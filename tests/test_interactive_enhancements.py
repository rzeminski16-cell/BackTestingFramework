"""
Tests for the interactive-mode enhancements:

- Perplexity-oriented research prompt (bullet-list output, strict as-of-date
  policy, 12-quarter financial trend, past-only SWOT, intrinsic value,
  catalysts/recent news).
- Same-day signal batches shipped with each DecisionRequest (day_batch /
  batch_index) so the panel can show all of today's signals at once.
- Seeded shuffle of same-day BUY processing order
  (PortfolioConfig.randomize_signal_order / signal_seed).
- Random auto-completion: RandomDecisionProvider, the mid-run
  hand_off_random hand-off, and session.activate_random_completion.
"""
import pandas as pd
import pytest

from Classes.Config.config import (
    CommissionConfig, CommissionMode, PortfolioConfig,
)
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Interactive.decision_provider import (
    RandomDecisionProvider,
    ScriptedDecisionProvider,
)
from Classes.Interactive.models import (
    BacktestRunManifest,
    DecisionAction,
    DecisionRequest,
    DecisionResponse,
    DecisionSource,
    SignalEvent,
)
from Classes.Interactive.prompt_generator import generate_research_prompt
from Classes.Interactive.session import InteractiveSession
from Classes.Interactive.store import InteractiveRunStore
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy


# --------------------------------------------------------------------------
# Shared fixtures (mirrors test_interactive_portfolio_engine.py)
# --------------------------------------------------------------------------

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


def make_session(tmp_path, script=None, cooldown_bars=21, name="run"):
    store = InteractiveRunStore(tmp_path / name)
    manifest = BacktestRunManifest(
        run_id=name, backtest_name=name, mode="interactive",
        engine_type="portfolio", cooldown_bars=cooldown_bars)
    store.write_manifest(manifest)
    provider = ScriptedDecisionProvider(script)
    return InteractiveSession(store, provider, manifest), store, provider


def make_event(**overrides):
    kwargs = dict(
        event_id=1, run_id="r", event_kind="STRATEGY_SIGNAL",
        symbol="AAPL", bar_date="2021-03-15", bar_index=100,
        signal_type="BUY", direction="LONG", proposed_size=0.5,
    )
    kwargs.update(overrides)
    return SignalEvent(**kwargs)


def _signal_request(**event_overrides):
    return DecisionRequest(kind="SIGNAL", event=make_event(**event_overrides))


def _all_trades(result):
    return sorted(
        (t for r in result.symbol_results.values() for t in r.trades),
        key=lambda t: (str(t.entry_date), t.symbol))


# --------------------------------------------------------------------------
# Research prompt: Perplexity spec
# --------------------------------------------------------------------------

class TestPromptPerplexitySpec:
    def test_asks_for_bullet_list_not_paragraphs(self):
        prompt = generate_research_prompt(make_event())
        assert "organised bullet list" in prompt
        assert "Do not write paragraphs" in prompt

    def test_answers_as_of_signal_date(self):
        prompt = generate_research_prompt(make_event())
        assert "Answer exactly as if it is 2021-03-15" in prompt
        assert "Assume today is 2021-03-15" in prompt
        assert "NO LOOK-AHEAD" in prompt

    def test_business_overview_section(self):
        prompt = generate_research_prompt(make_event())
        assert "business model" in prompt
        assert "current operations" in prompt

    def test_twelve_quarter_financial_trend(self):
        prompt = generate_research_prompt(make_event())
        assert "previous 12 reported quarters" in prompt
        assert "trend" in prompt
        for metric in ("revenue", "margins", "cash flow"):
            assert metric in prompt

    def test_swot_past_data_only(self):
        prompt = generate_research_prompt(make_event())
        assert "SWOT analysis" in prompt
        assert "no forward-looking speculation" in prompt

    def test_intrinsic_value_estimate(self):
        prompt = generate_research_prompt(make_event())
        assert "intrinsic value" in prompt
        assert "assumptions" in prompt

    def test_catalysts_and_recent_news(self):
        prompt = generate_research_prompt(make_event())
        assert "Catalysts & recent news" in prompt
        assert "recent relative to 2021-03-15" in prompt

    def test_final_rating_is_a_bullet(self):
        prompt = generate_research_prompt(make_event())
        assert "ATTRACTIVE / NEUTRAL / UNATTRACTIVE" in prompt
        assert "one-paragraph summary" not in prompt


# --------------------------------------------------------------------------
# RandomDecisionProvider
# --------------------------------------------------------------------------

class TestRandomDecisionProvider:
    def _actions(self, seed, n=32, approve_probability=0.5):
        provider = RandomDecisionProvider(
            seed=seed, approve_probability=approve_probability)
        return [provider.decide(_signal_request(event_id=i)).action
                for i in range(1, n + 1)]

    def test_seed_reproducible(self):
        assert self._actions(seed=42) == self._actions(seed=42)

    def test_different_seeds_differ(self):
        assert self._actions(seed=1) != self._actions(seed=2)

    def test_probability_extremes(self):
        assert set(self._actions(seed=0, approve_probability=1.0)) \
            == {DecisionAction.ACCEPT}
        assert set(self._actions(seed=0, approve_probability=0.0)) \
            == {DecisionAction.REJECT}

    def test_exits_always_accepted(self):
        provider = RandomDecisionProvider(seed=0, approve_probability=0.0)
        for signal_type in ("SELL", "PARTIAL_EXIT"):
            response = provider.decide(_signal_request(signal_type=signal_type))
            assert response.action == DecisionAction.ACCEPT

    def test_capital_resolution_reduces_size(self):
        provider = RandomDecisionProvider(seed=0)
        response = provider.decide(
            DecisionRequest(kind="CAPITAL_RESOLUTION", event=make_event()))
        assert response.action == DecisionAction.MODIFY
        assert response.capital_resolution == {'choice': 'reduce_size'}

    def test_decisions_are_auto_sourced(self):
        provider = RandomDecisionProvider(seed=0)
        assert provider.decide(_signal_request()).source == DecisionSource.AUTO

    def test_invalid_probability_rejected(self):
        with pytest.raises(ValueError, match="approve_probability"):
            RandomDecisionProvider(approve_probability=1.5)


# --------------------------------------------------------------------------
# Same-day batch shipped with each request
# --------------------------------------------------------------------------

class TestDayBatch:
    def test_same_day_signals_share_the_batch(self, tmp_path):
        session, _, provider = make_session(tmp_path)
        PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2, "BBB": 2}, size_frac=0.2),
            decision_session=session)

        buy_requests = [r for r in provider.requests if r.kind == "SIGNAL"]
        assert len(buy_requests) == 2
        for index, request in enumerate(buy_requests):
            assert request.batch_index == index
            assert [b['symbol'] for b in request.day_batch] == ["AAA", "BBB"]
        batch = buy_requests[0].day_batch
        assert batch[0]['signal_type'] == "BUY"
        assert batch[0]['price'] == 100.0
        assert batch[0]['required_capital'] == pytest.approx(20_000.0)
        assert batch[0]['reason'] == "entry"

    def test_single_signal_day_has_batch_of_one(self, tmp_path):
        session, _, provider = make_session(tmp_path)
        PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2}, size_frac=0.2),
            decision_session=session)
        buy_requests = [r for r in provider.requests if r.kind == "SIGNAL"]
        assert len(buy_requests) == 1
        assert len(buy_requests[0].day_batch) == 1
        assert buy_requests[0].batch_index == 0


# --------------------------------------------------------------------------
# Randomised same-day signal order (automatic runs)
# --------------------------------------------------------------------------

class TestRandomizeSignalOrder:
    """Two same-day signals each wanting 80% of equity: only one wins."""

    STRATEGY_ARGS = dict(entry_bars={"AAA": 2, "BBB": 2}, size_frac=0.8)

    def _winner(self, seed):
        result = PortfolioEngine(
            _config(randomize_signal_order=True, signal_seed=seed)).run(
            _data_dict(), PortfolioTestStrategy(**self.STRATEGY_ARGS))
        trades = _all_trades(result)
        assert len(trades) == 1  # loser is below the 50% reduction floor
        return trades[0].symbol

    def test_default_order_is_deterministic_first_symbol(self):
        result = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(**self.STRATEGY_ARGS))
        trades = _all_trades(result)
        assert len(trades) == 1
        assert trades[0].symbol == "AAA"

    def test_same_seed_reproduces_the_run(self):
        assert self._winner(seed=123) == self._winner(seed=123)

    def test_winner_rotates_across_seeds(self):
        winners = {self._winner(seed=s) for s in range(12)}
        assert winners == {"AAA", "BBB"}

    def test_config_round_trip(self):
        config = _config(randomize_signal_order=True, signal_seed=99)
        restored = PortfolioConfig.from_dict(config.to_dict())
        assert restored.randomize_signal_order is True
        assert restored.signal_seed == 99
        defaults = PortfolioConfig.from_dict(_config().to_dict())
        assert defaults.randomize_signal_order is False
        assert defaults.signal_seed is None

    def test_interactive_requires_seed(self, tmp_path):
        session, _, _ = make_session(tmp_path)
        with pytest.raises(ValueError, match="signal_seed"):
            PortfolioEngine(
                _config(randomize_signal_order=True)).run(
                _data_dict(), PortfolioTestStrategy(**self.STRATEGY_ARGS),
                decision_session=session)

    def test_interactive_with_seed_prompts_in_shuffled_order(self, tmp_path):
        # Find a seed whose shuffled order differs from symbol order, then
        # check the prompts arrive in exactly that order.
        flipped_seed = next(s for s in range(50) if self._winner(seed=s) == "BBB")
        session, _, provider = make_session(tmp_path)
        PortfolioEngine(
            _config(randomize_signal_order=True, signal_seed=flipped_seed)).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2, "BBB": 2}, size_frac=0.2),
            decision_session=session)
        buy_requests = [r for r in provider.requests if r.kind == "SIGNAL"]
        assert [r.event.symbol for r in buy_requests] == ["BBB", "AAA"]
        assert [b['symbol'] for b in buy_requests[0].day_batch] == ["BBB", "AAA"]


# --------------------------------------------------------------------------
# Random auto-completion
# --------------------------------------------------------------------------

class TestRandomAutoCompletion:
    def test_hand_off_switches_provider_permanently(self, tmp_path):
        # First (and only) scripted answer hands the run off; the second
        # day's signal must never reach the scripted provider.
        session, store, provider = make_session(tmp_path, [
            DecisionResponse(action=DecisionAction.ACCEPT,
                             source=DecisionSource.USER,
                             rationale="Handed off",
                             hand_off_random=True),
        ])
        PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2, "BBB": 5}, size_frac=0.2),
            decision_session=session)

        assert session.random_completion_active
        assert len(provider.requests) == 1  # only the first signal
        strategy_decisions = [
            d for d in store.load_decisions()
            if d.action in (DecisionAction.ACCEPT, DecisionAction.REJECT)]
        assert len(strategy_decisions) == 2
        # The hand-off event itself is decided randomly too.
        for decision in strategy_decisions:
            assert decision.source == DecisionSource.AUTO
            assert decision.rationale.startswith("Random auto-completion")

    def test_activate_up_front_runs_headless(self, tmp_path):
        session, store, provider = make_session(tmp_path)
        session.activate_random_completion(seed=7, approve_probability=1.0)
        result = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2, "BBB": 3}, exit_bars={"AAA": 8},
                size_frac=0.3),
            decision_session=session)

        assert provider.requests == []  # scripted provider never consulted
        trades = _all_trades(result)
        assert {t.symbol for t in trades} == {"AAA", "BBB"}
        aaa = [t for t in trades if t.symbol == "AAA"][0]
        assert aaa.exit_reason == "exit"  # exit signal accepted

    def test_seeded_completion_is_reproducible(self, tmp_path):
        def run(name):
            session, _, _ = make_session(tmp_path, name=name)
            session.activate_random_completion(seed=11, approve_probability=0.5)
            result = PortfolioEngine(_config()).run(
                _data_dict(), PortfolioTestStrategy(
                    entry_bars={"AAA": 2, "BBB": 4}, size_frac=0.3),
                decision_session=session)
            return [(t.symbol, t.quantity, t.pl) for t in _all_trades(result)]

        assert run("first") == run("second")

    def test_capital_contingency_resolved_without_prompting(self, tmp_path):
        # Both entries want 80% of equity: the second only fits reduced.
        # Random completion (approve all) must finish without touching the
        # scripted provider, taking the reduce-size branch.
        session, _, provider = make_session(tmp_path)
        session.activate_random_completion(seed=3, approve_probability=1.0)
        result = PortfolioEngine(_config()).run(
            _data_dict(), PortfolioTestStrategy(
                entry_bars={"AAA": 2, "BBB": 5}, size_frac=0.8),
            decision_session=session)
        assert provider.requests == []
        bbb_trades = result.symbol_results["BBB"].trades
        assert len(bbb_trades) == 1
        # AAA consumed 80k of 100k: BBB reduced to the remaining ~20k.
        assert bbb_trades[0].quantity == pytest.approx(200.0, rel=1e-3)
