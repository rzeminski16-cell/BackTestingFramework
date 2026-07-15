"""
Tests for the deterministic, time-bounded research prompt generator.
"""
from Classes.Interactive.models import SignalEvent
from Classes.Interactive.prompt_generator import (
    default_horizon_days,
    generate_research_prompt,
)


def make_event(**overrides):
    kwargs = dict(
        event_id=1, run_id="r", event_kind="STRATEGY_SIGNAL",
        symbol="AAPL", bar_date="2021-03-15", bar_index=100,
        signal_type="BUY", direction="LONG", proposed_size=0.5,
        technical_snapshot={
            'pct_change_5d': 2.5, 'pct_change_21d': -4.0,
            'dist_from_252d_high_pct': -12.0, 'dist_from_252d_low_pct': 30.0,
        },
    )
    kwargs.update(overrides)
    return SignalEvent(**kwargs)


class TestGenerateResearchPrompt:
    def test_deterministic(self):
        assert (generate_research_prompt(make_event())
                == generate_research_prompt(make_event()))

    def test_contains_hard_time_bound(self):
        prompt = generate_research_prompt(make_event())
        assert "Assume today is 2021-03-15" in prompt
        assert "on or before 2021-03-15" in prompt
        assert "NO LOOK-AHEAD" in prompt
        assert "official, reputable sources" in prompt

    def test_mentions_symbol_action_and_horizon(self):
        prompt = generate_research_prompt(make_event(), horizon_days=45)
        assert "AAPL" in prompt
        assert "opening a new long (buy) position" in prompt
        assert "45 days" in prompt

    def test_exit_signal_wording(self):
        prompt = generate_research_prompt(make_event(signal_type="SELL"))
        assert "closing an existing long (buy) position" in prompt

    def test_short_direction_wording(self):
        prompt = generate_research_prompt(make_event(direction="SHORT"))
        assert "short (sell)" in prompt

    def test_price_action_from_snapshot_only(self):
        prompt = generate_research_prompt(make_event())
        assert "+2.5% over the last 1 week" in prompt
        assert "-4.0% over the last 1 month" in prompt
        assert "12.0% below its 52-week high" in prompt

    def test_no_price_action_section_when_snapshot_empty(self):
        prompt = generate_research_prompt(make_event(technical_snapshot={}))
        assert "Recent price action" not in prompt

    def test_currency_note(self):
        prompt = generate_research_prompt(make_event(), currency="GBP")
        assert "denominated in GBP" in prompt


class TestDefaultHorizonDays:
    def test_default(self):
        assert default_horizon_days("AnyStrategy", {}) == 90

    def test_uses_holding_period_param(self):
        assert default_horizon_days("S", {'max_holding_days': 30}) == 30
        assert default_horizon_days("S", {'horizon_bars': 15}) == 15

    def test_ignores_non_numeric_and_non_positive(self):
        assert default_horizon_days("S", {'holding_mode': 'long'}) == 90
        assert default_horizon_days("S", {'hold_days': 0}) == 90
