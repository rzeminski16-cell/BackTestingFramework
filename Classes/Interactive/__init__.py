"""
Interactive (discretionary) decision layer for backtests.

This package implements INTERACTIVE mode: the engine pauses at each
actionable technical signal, asks a DecisionProvider (GUI panel, scripted
test double, or resume-replay) for a human decision, and logs every
signal event, decision, and research prompt to append-only JSONL files
so runs are auditable, resumable, and comparable against a rules-only
baseline of the identical configuration.

Prompting policy (v1):
- Strategy-generated BUY / SELL / PYRAMID / PARTIAL_EXIT signals prompt,
  but only when actionable (BUY if flat, the others if positioned).
- ADJUST_STOP signals and engine-generated protective stop-loss /
  take-profit exits auto-execute (they model resting orders) and are
  logged as AUTO_APPLIED decision records.
- HOLD and non-actionable signals are not logged.
"""
from .models import (
    DecisionAction,
    DecisionSource,
    SignalEvent,
    DecisionRecord,
    PromptRecord,
    OutcomeRecord,
    BacktestRunManifest,
    CapitalOptions,
    CapitalResolutionChoice,
    DecisionRequest,
    DecisionResponse,
    ResolvedDecision,
    RunAbortedByUser,
    ReplayMismatchError,
    build_data_fingerprints,
    verify_data_fingerprints,
)
from .decision_provider import (
    DecisionProvider,
    QueueDecisionProvider,
    ScriptedDecisionProvider,
)
from .suppression import RejectCooldownTracker
from .store import InteractiveRunStore, find_resumable_runs
from .session import InteractiveSession
from .prompt_generator import generate_research_prompt, default_horizon_days

__all__ = [
    'DecisionAction', 'DecisionSource', 'SignalEvent', 'DecisionRecord',
    'PromptRecord', 'OutcomeRecord', 'BacktestRunManifest', 'CapitalOptions',
    'CapitalResolutionChoice', 'DecisionRequest', 'DecisionResponse',
    'ResolvedDecision', 'RunAbortedByUser', 'ReplayMismatchError',
    'build_data_fingerprints', 'verify_data_fingerprints',
    'DecisionProvider', 'QueueDecisionProvider', 'ScriptedDecisionProvider',
    'RejectCooldownTracker', 'InteractiveRunStore', 'find_resumable_runs',
    'InteractiveSession', 'generate_research_prompt', 'default_horizon_days',
]
