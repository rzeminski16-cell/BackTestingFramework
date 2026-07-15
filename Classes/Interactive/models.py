"""
Data model for the interactive (discretionary) decision layer.

Four persisted entities describe an interactive run:

- BacktestRunManifest: run identity, mode, config, and data fingerprints.
- SignalEvent: an actionable signal with technical/portfolio snapshots
  frozen at signal time (no look-ahead: built only from the engine's
  HistoricalDataView / StrategyContext).
- DecisionRecord: what was decided, by whom (user/quick/auto/replay),
  and why (rationale text).
- PromptRecord: the generated time-bounded research prompt plus the
  user's optionally pasted response summary.

OutcomeRecord rows (what actually executed) live in a separate stream so
decision records stay append-only — that is what makes resume-replay and
crash tolerance simple.

Serialization follows the house style: dataclass to_dict()/from_dict()
pairs, JSON-safe values only (enums as their .value).
"""
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


class RunAbortedByUser(Exception):
    """Raised inside the engine when the user pauses/abandons an interactive run."""


class ReplayMismatchError(Exception):
    """Raised when a resumed run's regenerated events diverge from the logged ones."""


class DecisionAction(Enum):
    """What was decided for a signal event."""
    ACCEPT = "accept"
    MODIFY = "modify"
    REJECT = "reject"
    DEFER = "defer"
    AUTO_SUPPRESSED = "auto_suppressed"  # reject-cooldown suppression
    AUTO_APPLIED = "auto_applied"        # ADJUST_STOP / engine stop/TP exits
    ABORT = "abort"                      # user paused/closed the run


class DecisionSource(Enum):
    """Who/what produced the decision."""
    USER = "user"      # full panel interaction
    QUICK = "quick"    # one-click quick-action button
    AUTO = "auto"      # engine/session generated (suppression, auto-applied)
    REPLAY = "replay"  # replayed from the log during resume


class CapitalResolutionChoice(Enum):
    """How an accepted BUY with insufficient capital was resolved."""
    REDUCE_SIZE = "reduce_size"
    FREE_CAPITAL = "free_capital"
    REJECT = "reject"


def _round_token(value: Optional[float]) -> str:
    """Stable string token for a float that may be None."""
    if value is None:
        return "None"
    return f"{float(value):.8f}"


def compute_fingerprint(event_kind: str, symbol: str, bar_date: str,
                        signal_type: str, direction: str,
                        proposed_size: Optional[float],
                        proposed_stop_loss: Optional[float]) -> str:
    """
    Deterministic short hash identifying a signal event for replay
    verification. Depends only on signal-time facts, never on wall-clock
    or decision data.
    """
    payload = "|".join([
        event_kind, symbol, bar_date, signal_type, direction,
        _round_token(proposed_size), _round_token(proposed_stop_loss),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class SignalEvent:
    """An actionable signal with context frozen at signal time."""
    event_id: int
    run_id: str
    event_kind: str          # "STRATEGY_SIGNAL" | "ENGINE_EXIT" | "CAPITAL_RESOLUTION" | "DISCRETIONARY_FUNDING"
    symbol: str
    bar_date: str            # ISO date of the backtest bar
    bar_index: int           # per-symbol bar index (cooldown continuity key)
    signal_type: str         # SignalType.value
    direction: str           # TradeDirection.value
    proposed_size: float = 0.0
    proposed_stop_loss: Optional[float] = None
    proposed_take_profit: Optional[float] = None
    new_stop_loss: Optional[float] = None
    signal_reason: str = ""
    technical_snapshot: Dict[str, Any] = field(default_factory=dict)
    portfolio_snapshot: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = compute_fingerprint(
                self.event_kind, self.symbol, self.bar_date, self.signal_type,
                self.direction, self.proposed_size, self.proposed_stop_loss,
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalEvent':
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class DecisionRecord:
    """The decision taken for one SignalEvent."""
    decision_id: int
    event_id: int
    run_id: str
    action: DecisionAction
    source: DecisionSource
    size_factor: Optional[float] = None          # MODIFY: scale on proposed size/quantity
    modified_stop_loss: Optional[float] = None   # MODIFY: absolute price override
    modified_take_profit: Optional[float] = None
    rationale: str = ""
    capital_resolution: Optional[Dict[str, Any]] = None
    suppressed_by: Optional[int] = None          # decision_id of originating REJECT
    prompt_id: Optional[int] = None
    decided_at: str = ""                         # wall clock; excluded from replay checks
    decision_duration_secs: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['action'] = self.action.value
        d['source'] = self.source.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionRecord':
        kwargs = {k: data.get(k) for k in cls.__dataclass_fields__}
        kwargs['action'] = DecisionAction(data['action'])
        kwargs['source'] = DecisionSource(data['source'])
        return cls(**kwargs)


@dataclass
class PromptRecord:
    """A generated research prompt and (optionally) its pasted findings."""
    prompt_id: int
    event_id: int
    run_id: str
    generated_at: str = ""
    horizon_days: int = 90
    prompt_text: str = ""
    response_summary: str = ""
    response_pasted_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptRecord':
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class OutcomeRecord:
    """What actually executed for one SignalEvent (post-dispatch)."""
    event_id: int
    executed: bool
    executed_quantity: Optional[float] = None
    executed_price: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutcomeRecord':
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class BacktestRunManifest:
    """
    Self-describing manifest for an interactive run (or its AUTO baseline).

    Written to <run folder>/interactive/run_manifest.json at run start
    (status=in_progress) and rewritten on finalize. The interactive and
    baseline runs share the identical config dict — mode lives here, not
    in BacktestConfig/PortfolioConfig.
    """
    run_id: str
    backtest_name: str
    mode: str                       # "interactive" | "auto_baseline"
    engine_type: str                # "single" | "portfolio"
    status: str = "in_progress"     # in_progress | completed | paused | corrupt | aborted
    created_at: str = ""
    finalized_at: str = ""
    symbols: List[str] = field(default_factory=list)   # ordered; replay depends on order
    strategy_class: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    cooldown_bars: int = 21
    data_fingerprints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    baseline_dir: str = ""
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestRunManifest':
        kwargs = {k: data.get(k) for k in cls.__dataclass_fields__ if k in data}
        return cls(**kwargs)


def data_fingerprint(df) -> Dict[str, Any]:
    """
    Cheap fingerprint of a symbol's price data used to refuse unsafe
    resumes: if rows/date span/last close changed, replay is not valid.
    """
    fp: Dict[str, Any] = {'rows': int(len(df))}
    if len(df) > 0:
        if 'date' in df.columns:
            fp['first_date'] = str(df['date'].iloc[0])
            fp['last_date'] = str(df['date'].iloc[-1])
        if 'close' in df.columns:
            fp['last_close'] = round(float(df['close'].iloc[-1]), 8)
    return fp


def build_data_fingerprints(data_by_symbol: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Fingerprint every symbol's DataFrame for the run manifest."""
    return {symbol: data_fingerprint(df) for symbol, df in data_by_symbol.items()}


def verify_data_fingerprints(manifest: 'BacktestRunManifest',
                             data_by_symbol: Dict[str, Any]) -> None:
    """
    Refuse to resume when the underlying data changed since the run was
    paused — replaying decisions against different data would silently
    corrupt the results.

    Raises:
        ReplayMismatchError: on any symbol/fingerprint difference.
    """
    problems = []
    for symbol, expected in (manifest.data_fingerprints or {}).items():
        if symbol not in data_by_symbol:
            problems.append(f"{symbol}: data file no longer available")
            continue
        actual = data_fingerprint(data_by_symbol[symbol])
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                problems.append(
                    f"{symbol}: {key} changed "
                    f"({expected_value!r} -> {actual.get(key)!r})")
    if problems:
        raise ReplayMismatchError(
            "The underlying data has changed since this run was paused; "
            "resuming is unsafe:\n  - " + "\n  - ".join(problems))


@dataclass
class CapitalOptions:
    """
    Presented when an accepted BUY does not fit in available capital
    (portfolio mode). ``positions`` rows describe each open position the
    user could trim or close to free capital.
    """
    required_capital: float
    available_capital: float
    affordable_fraction: float          # available / required (clipped to [0, 1])
    positions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionRequest:
    """
    Payload sent to a DecisionProvider. ``chart_data`` is a small
    trailing slice (<=150 bars) of the look-ahead-safe context data for
    the GUI chart; it is never serialized.

    ``day_batch`` lists summaries of every same-day BUY signal (this one
    included) so the panel can show all of today's signals at once;
    ``batch_index`` is this event's position in that list. Decisions are
    still taken one at a time, in order, because each accept/reject
    changes the capital available to the signals after it.
    """
    kind: str                       # "SIGNAL" | "CAPITAL_RESOLUTION"
    event: SignalEvent
    chart_data: Any = None          # pd.DataFrame or None
    capital_options: Optional[CapitalOptions] = None
    warning: str = ""               # e.g. hint shown when rejecting a SELL
    day_batch: List[Dict[str, Any]] = field(default_factory=list)
    batch_index: int = 0


@dataclass
class DecisionResponse:
    """A DecisionProvider's answer to a DecisionRequest."""
    action: DecisionAction
    size_factor: Optional[float] = None
    modified_stop_loss: Optional[float] = None
    modified_take_profit: Optional[float] = None
    rationale: str = ""
    capital_resolution: Optional[Dict[str, Any]] = None
    prompt_text: str = ""           # generated research prompt, if any
    prompt_horizon_days: int = 90
    response_summary: str = ""      # pasted findings, if any
    source: DecisionSource = DecisionSource.USER
    hand_off_random: bool = False   # finish the run with random decisions


@dataclass
class ResolvedDecision:
    """What the engine receives back from InteractiveSession.resolve_signal."""
    effective_signal: Any           # Classes.Models.signal.Signal
    event: SignalEvent
    record: DecisionRecord

    @property
    def size_factor(self) -> float:
        """Quantity scale the engine applies to the strategy-computed size."""
        if self.record.action == DecisionAction.MODIFY and self.record.size_factor:
            return float(self.record.size_factor)
        return 1.0
