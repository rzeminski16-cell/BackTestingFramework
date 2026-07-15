"""
Tests for the interactive-layer data model: serialization round-trips,
fingerprint stability, and the JSONL store.
"""
import json

import pytest

from Classes.Interactive.models import (
    BacktestRunManifest,
    CapitalOptions,
    DecisionAction,
    DecisionRecord,
    DecisionSource,
    OutcomeRecord,
    PromptRecord,
    SignalEvent,
    compute_fingerprint,
)
from Classes.Interactive.store import InteractiveRunStore, find_resumable_runs


def make_event(event_id=1, **overrides):
    kwargs = dict(
        event_id=event_id,
        run_id="run_x",
        event_kind="STRATEGY_SIGNAL",
        symbol="AAPL",
        bar_date="2021-03-15",
        bar_index=120,
        signal_type="BUY",
        direction="LONG",
        proposed_size=0.25,
        proposed_stop_loss=95.5,
        signal_reason="EMA crossover",
        technical_snapshot={'close': 100.0, 'pct_change_21d': 4.2},
        portfolio_snapshot={'available_capital': 50000.0, 'total_equity': 100000.0},
    )
    kwargs.update(overrides)
    return SignalEvent(**kwargs)


class TestSignalEvent:
    def test_roundtrip(self):
        event = make_event()
        restored = SignalEvent.from_dict(json.loads(json.dumps(event.to_dict())))
        assert restored == event

    def test_fingerprint_autofilled_and_stable(self):
        a = make_event()
        b = make_event()
        assert a.fingerprint and a.fingerprint == b.fingerprint

    def test_fingerprint_changes_with_signal_facts(self):
        base = make_event()
        assert make_event(symbol="MSFT").fingerprint != base.fingerprint
        assert make_event(bar_date="2021-03-16").fingerprint != base.fingerprint
        assert make_event(signal_type="SELL").fingerprint != base.fingerprint
        assert make_event(proposed_size=0.5).fingerprint != base.fingerprint
        assert make_event(proposed_stop_loss=None).fingerprint != base.fingerprint

    def test_fingerprint_ignores_snapshots_and_ids(self):
        base = make_event()
        other = make_event(event_id=99, technical_snapshot={}, portfolio_snapshot={})
        assert other.fingerprint == base.fingerprint

    def test_compute_fingerprint_handles_none(self):
        fp = compute_fingerprint("STRATEGY_SIGNAL", "A", "2020-01-01",
                                 "SELL", "LONG", None, None)
        assert len(fp) == 16


class TestDecisionRecord:
    def test_roundtrip_with_enums(self):
        record = DecisionRecord(
            decision_id=3, event_id=1, run_id="run_x",
            action=DecisionAction.MODIFY, source=DecisionSource.USER,
            size_factor=0.5, modified_stop_loss=90.0,
            rationale="Earnings tomorrow; halving size",
            capital_resolution={'choice': 'reduce_size'},
            decided_at="2026-07-14T10:00:00",
        )
        restored = DecisionRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored == record
        assert restored.action is DecisionAction.MODIFY
        assert restored.source is DecisionSource.USER


class TestPromptAndOutcome:
    def test_prompt_roundtrip(self):
        prompt = PromptRecord(prompt_id=1, event_id=2, run_id="run_x",
                              prompt_text="Assume today is 2021-03-15...",
                              response_summary="Valuation stretched.")
        assert PromptRecord.from_dict(prompt.to_dict()) == prompt

    def test_outcome_roundtrip(self):
        outcome = OutcomeRecord(event_id=4, executed=True,
                                executed_quantity=10.0, executed_price=101.5)
        assert OutcomeRecord.from_dict(outcome.to_dict()) == outcome


class TestManifest:
    def test_roundtrip(self):
        manifest = BacktestRunManifest(
            run_id="alpha_test_20260714_120000",
            backtest_name="Alpha_test",
            mode="interactive",
            engine_type="portfolio",
            symbols=["AAPL", "MSFT"],
            strategy_class="AlphaTrendV1Strategy",
            strategy_params={'ema_fast': 20},
            config={'initial_capital': 100000.0},
            data_fingerprints={'AAPL': {'rows': 500, 'last_close': 150.0}},
        )
        restored = BacktestRunManifest.from_dict(
            json.loads(json.dumps(manifest.to_dict())))
        assert restored == manifest

    def test_from_dict_tolerates_missing_keys(self):
        manifest = BacktestRunManifest.from_dict({
            'run_id': 'r', 'backtest_name': 'b',
            'mode': 'interactive', 'engine_type': 'single',
        })
        assert manifest.status == "in_progress"
        assert manifest.cooldown_bars == 21


class TestStore:
    def test_append_and_load(self, tmp_path):
        store = InteractiveRunStore(tmp_path / "run")
        event = make_event()
        record = DecisionRecord(decision_id=1, event_id=1, run_id="run_x",
                                action=DecisionAction.ACCEPT,
                                source=DecisionSource.QUICK)
        store.append_event(event)
        store.append_decision(record)
        store.append_outcome(OutcomeRecord(event_id=1, executed=True,
                                           executed_quantity=5, executed_price=100.0))
        store.append_prompt(PromptRecord(prompt_id=1, event_id=1, run_id="run_x",
                                         prompt_text="p"))

        assert store.load_events() == [event]
        assert store.load_decisions() == [record]
        assert store.load_outcomes()[0].executed is True
        assert store.load_prompts()[0].prompt_text == "p"

    def test_load_for_resume_drops_dangling_event(self, tmp_path):
        store = InteractiveRunStore(tmp_path / "run")
        store.append_event(make_event(event_id=1))
        store.append_decision(DecisionRecord(
            decision_id=1, event_id=1, run_id="run_x",
            action=DecisionAction.ACCEPT, source=DecisionSource.USER))
        store.append_event(make_event(event_id=2))  # decision never landed
        pairs = store.load_for_resume()
        assert len(pairs) == 1
        assert pairs[0][0].event_id == 1

    def test_torn_final_line_is_dropped(self, tmp_path):
        store = InteractiveRunStore(tmp_path / "run")
        store.append_event(make_event(event_id=1))
        events_file = store.interactive_dir / "signal_events.jsonl"
        with open(events_file, "a") as f:
            f.write('{"event_id": 2, "run_id": "run_x", "truncat')
        assert [e.event_id for e in store.load_events()] == [1]

    def test_manifest_write_read(self, tmp_path):
        store = InteractiveRunStore(tmp_path / "run")
        assert store.read_manifest() is None
        manifest = BacktestRunManifest(run_id="r", backtest_name="b",
                                       mode="interactive", engine_type="single")
        store.write_manifest(manifest)
        assert store.read_manifest() == manifest

    def test_export_flat(self, tmp_path):
        store = InteractiveRunStore(tmp_path / "run")
        store.append_event(make_event(event_id=1))
        store.append_decision(DecisionRecord(
            decision_id=1, event_id=1, run_id="run_x",
            action=DecisionAction.REJECT, source=DecisionSource.USER,
            rationale="valuation stretched"))
        store.append_outcome(OutcomeRecord(event_id=1, executed=False,
                                           reason="Rejected by user"))
        paths = store.export_flat()
        csv_path = [p for p in paths if p.suffix == ".csv"][0]
        content = csv_path.read_text()
        assert "valuation stretched" in content
        assert "AAPL" in content
        df = store.build_flat_table()
        assert df.loc[0, 'action'] == "reject"
        assert bool(df.loc[0, 'executed']) is False


class TestFindResumableRuns:
    def test_finds_only_resumable_interactive_runs(self, tmp_path):
        logs = tmp_path / "logs"
        for name, mode, status in [
            ("a", "interactive", "paused"),
            ("b", "interactive", "completed"),
            ("c", "auto_baseline", "in_progress"),
            ("d", "interactive", "in_progress"),
        ]:
            store = InteractiveRunStore(logs / "portfolio" / name)
            store.write_manifest(BacktestRunManifest(
                run_id=name, backtest_name=name, mode=mode,
                engine_type="portfolio", status=status,
                created_at=f"2026-07-0{ord(name[0]) - ord('a') + 1}T00:00:00"))
        found = find_resumable_runs(logs)
        assert sorted(r['manifest'].run_id for r in found) == ["a", "d"]

    def test_missing_root_is_empty(self, tmp_path):
        assert find_resumable_runs(tmp_path / "nope") == []
