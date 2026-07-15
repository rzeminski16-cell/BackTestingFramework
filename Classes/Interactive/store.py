"""
Append-only JSONL persistence for interactive runs.

Layout inside an existing run folder (logs/<engine>/<backtest_name>/):

    interactive/
    ├── run_manifest.json      # written at start, rewritten on finalize
    ├── signal_events.jsonl    # appended (and flushed) at prompt time
    ├── decisions.jsonl        # appended (and flushed) at decision time
    ├── outcomes.jsonl         # appended after dispatch, keyed by event_id
    ├── prompts.jsonl
    └── exports/               # flat CSV/JSON/XLSX join written on finalize

Every append is flushed and fsync'd so a decision is durable the moment
it is made. The resume loader is tolerant: a torn final line is dropped,
and a trailing event without a matching decision is dropped too (the
engine regenerates it identically on replay).
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    BacktestRunManifest,
    DecisionRecord,
    OutcomeRecord,
    PromptRecord,
    SignalEvent,
)

EVENTS_FILE = "signal_events.jsonl"
DECISIONS_FILE = "decisions.jsonl"
OUTCOMES_FILE = "outcomes.jsonl"
PROMPTS_FILE = "prompts.jsonl"
MANIFEST_FILE = "run_manifest.json"
EXPORTS_DIR = "exports"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object per line, durable immediately."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, silently dropping a torn final line."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn final line from a crash mid-write
            raise
    return rows


class InteractiveRunStore:
    """Owns the interactive/ folder of one run."""

    def __init__(self, run_dir: Path):
        """
        Args:
            run_dir: The run folder (e.g. logs/portfolio/<backtest_name>).
                     The interactive/ subfolder is created inside it.
        """
        self.run_dir = Path(run_dir)
        self.interactive_dir = self.run_dir / "interactive"
        self.interactive_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    @property
    def manifest_path(self) -> Path:
        return self.interactive_dir / MANIFEST_FILE

    @property
    def exports_dir(self) -> Path:
        path = self.interactive_dir / EXPORTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, name: str) -> Path:
        return self.interactive_dir / name

    # --------------------------------------------------------------- manifest
    def write_manifest(self, manifest: BacktestRunManifest) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

    def read_manifest(self) -> Optional[BacktestRunManifest]:
        if not self.manifest_path.exists():
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return BacktestRunManifest.from_dict(json.load(f))

    # ---------------------------------------------------------------- appends
    def append_event(self, event: SignalEvent) -> None:
        _append_jsonl(self._path(EVENTS_FILE), event.to_dict())

    def append_decision(self, record: DecisionRecord) -> None:
        _append_jsonl(self._path(DECISIONS_FILE), record.to_dict())

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        _append_jsonl(self._path(OUTCOMES_FILE), outcome.to_dict())

    def append_prompt(self, prompt: PromptRecord) -> None:
        _append_jsonl(self._path(PROMPTS_FILE), prompt.to_dict())

    # ----------------------------------------------------------------- loads
    def load_events(self) -> List[SignalEvent]:
        return [SignalEvent.from_dict(r) for r in _read_jsonl(self._path(EVENTS_FILE))]

    def load_decisions(self) -> List[DecisionRecord]:
        return [DecisionRecord.from_dict(r) for r in _read_jsonl(self._path(DECISIONS_FILE))]

    def load_outcomes(self) -> List[OutcomeRecord]:
        return [OutcomeRecord.from_dict(r) for r in _read_jsonl(self._path(OUTCOMES_FILE))]

    def load_prompts(self) -> List[PromptRecord]:
        return [PromptRecord.from_dict(r) for r in _read_jsonl(self._path(PROMPTS_FILE))]

    def load_for_resume(self) -> List[Tuple[SignalEvent, DecisionRecord]]:
        """
        Return the replayable (event, decision) pairs in event order.

        An event whose decision never landed (crash/abort mid-decision)
        is dropped — the engine will regenerate it identically and
        prompt live.
        """
        events = {e.event_id: e for e in self.load_events()}
        pairs: List[Tuple[SignalEvent, DecisionRecord]] = []
        for decision in self.load_decisions():
            event = events.get(decision.event_id)
            if event is not None:
                pairs.append((event, decision))
        pairs.sort(key=lambda p: p[0].event_id)
        return pairs

    # ---------------------------------------------------------------- exports
    def build_flat_table(self):
        """
        Join events + decisions + outcomes + prompts into one flat
        pandas DataFrame (one row per decision) for exports/analysis.
        """
        import pandas as pd

        events = {e.event_id: e for e in self.load_events()}
        outcomes = {o.event_id: o for o in self.load_outcomes()}
        prompts = {p.prompt_id: p for p in self.load_prompts()}

        rows = []
        for decision in self.load_decisions():
            event = events.get(decision.event_id)
            outcome = outcomes.get(decision.event_id)
            prompt = prompts.get(decision.prompt_id) if decision.prompt_id else None
            row: Dict[str, Any] = {
                'event_id': decision.event_id,
                'decision_id': decision.decision_id,
                'symbol': event.symbol if event else None,
                'bar_date': event.bar_date if event else None,
                'event_kind': event.event_kind if event else None,
                'signal_type': event.signal_type if event else None,
                'direction': event.direction if event else None,
                'signal_reason': event.signal_reason if event else None,
                'proposed_size': event.proposed_size if event else None,
                'proposed_stop_loss': event.proposed_stop_loss if event else None,
                'proposed_take_profit': event.proposed_take_profit if event else None,
                'action': decision.action.value,
                'source': decision.source.value,
                'size_factor': decision.size_factor,
                'modified_stop_loss': decision.modified_stop_loss,
                'modified_take_profit': decision.modified_take_profit,
                'rationale': decision.rationale,
                'capital_resolution': (json.dumps(decision.capital_resolution)
                                       if decision.capital_resolution else None),
                'suppressed_by': decision.suppressed_by,
                'decided_at': decision.decided_at,
                'decision_duration_secs': decision.decision_duration_secs,
                'executed': outcome.executed if outcome else None,
                'executed_quantity': outcome.executed_quantity if outcome else None,
                'executed_price': outcome.executed_price if outcome else None,
                'outcome_reason': outcome.reason if outcome else None,
                'prompt_id': decision.prompt_id,
                'prompt_response_summary': prompt.response_summary if prompt else None,
            }
            if event:
                for key, value in event.technical_snapshot.items():
                    row[f'tech_{key}'] = value
                row['portfolio_cash'] = event.portfolio_snapshot.get('available_capital')
                row['portfolio_equity'] = event.portfolio_snapshot.get('total_equity')
            rows.append(row)
        return pd.DataFrame(rows)

    def export_flat(self) -> List[Path]:
        """Write decisions.csv/.json/.xlsx into exports/. Returns paths."""
        df = self.build_flat_table()
        written: List[Path] = []

        csv_path = self.exports_dir / "decisions.csv"
        df.to_csv(csv_path, index=False)
        written.append(csv_path)

        json_path = self.exports_dir / "decisions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
        written.append(json_path)

        try:
            xlsx_path = self.exports_dir / "decisions.xlsx"
            df.to_excel(xlsx_path, index=False, sheet_name="Decisions")
            written.append(xlsx_path)
        except Exception:
            # Excel export is best-effort; JSONL remains the source of truth.
            pass
        return written


def find_resumable_runs(logs_root: Path = Path("logs")) -> List[Dict[str, Any]]:
    """
    Scan logs/*/*/interactive/run_manifest.json for interactive runs that
    can be resumed (status in_progress or paused). Returns dicts with the
    manifest and the run folder path, newest first.
    """
    results: List[Dict[str, Any]] = []
    logs_root = Path(logs_root)
    if not logs_root.exists():
        return results
    for manifest_path in sorted(logs_root.glob("*/*/interactive/" + MANIFEST_FILE)):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = BacktestRunManifest.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, TypeError):
            continue
        if manifest.mode == "interactive" and manifest.status in ("in_progress", "paused"):
            results.append({
                'manifest': manifest,
                'run_dir': manifest_path.parent.parent,
            })
    results.sort(key=lambda r: r['manifest'].created_at, reverse=True)
    return results
