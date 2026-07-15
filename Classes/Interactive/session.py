"""
InteractiveSession: the engine-facing API of the discretionary layer.

The engines call:

- resolve_signal(...)   for each actionable strategy signal. The session
  builds a SignalEvent snapshot, applies reject-cooldown suppression,
  asks the DecisionProvider (or replays the logged decision on resume),
  persists everything, and returns the effective Signal to dispatch.
- resolve_capital(...)  (portfolio only) when an accepted BUY does not
  fit in available capital.
- log_auto(...)         for auto-applied events (ADJUST_STOP, engine
  stop/take-profit exits, final-bar drops) so the audit trail is complete.
- record_outcome(...)   after dispatch, with what actually executed.
- on_position_opened/closed(...) to clear moot suppressions.
- finalize(status)      to rewrite the manifest at the end.

Resume: constructed with ``resume_records`` (from
InteractiveRunStore.load_for_resume()), the session verifies each
regenerated event against the logged fingerprint and replays the logged
decision without re-appending or prompting; when the records run out it
switches to live prompting seamlessly. Any divergence raises
ReplayMismatchError — a resumed run must be byte-identical up to the
last logged decision.
"""
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..Models.signal import Signal, SignalType
from .decision_provider import DecisionProvider
from .models import (
    BacktestRunManifest,
    CapitalOptions,
    CapitalResolutionChoice,
    DecisionAction,
    DecisionRecord,
    DecisionRequest,
    DecisionResponse,
    DecisionSource,
    OutcomeRecord,
    PromptRecord,
    ReplayMismatchError,
    ResolvedDecision,
    RunAbortedByUser,
    SignalEvent,
)
from .store import InteractiveRunStore
from .suppression import RejectCooldownTracker

# Strategy signal types that prompt the user (when actionable).
PROMPTING_TYPES = (SignalType.BUY, SignalType.SELL,
                   SignalType.PARTIAL_EXIT, SignalType.PYRAMID)

CHART_BARS = 150

SELL_REJECT_WARNING = ("Rejecting this exit keeps the position open, protected "
                       "only by its resting stop-loss/take-profit.")


class InteractiveSession:
    """Coordinates decisions, suppression, persistence, and replay for one run."""

    def __init__(self,
                 store: InteractiveRunStore,
                 provider: DecisionProvider,
                 manifest: BacktestRunManifest,
                 resume_records: Optional[List[Tuple[SignalEvent, DecisionRecord]]] = None):
        self.store = store
        self.provider = provider
        self.manifest = manifest
        self.tracker = RejectCooldownTracker(manifest.cooldown_bars)

        self._event_counter = 0
        self._decision_counter = 0
        resuming = resume_records is not None
        self._replay: Deque[Tuple[SignalEvent, DecisionRecord]] = deque(resume_records or [])
        self._resume_last_event_id = (resume_records[-1][0].event_id
                                      if resume_records else 0)
        # Event ids already on disk (covers dangling events whose decision
        # never landed — do not append them twice on resume).
        self._events_already_logged = (
            {e.event_id for e in store.load_events()} if resuming else set()
        )
        existing_prompts = store.load_prompts() if resuming else []
        self._prompt_counter = max((p.prompt_id for p in existing_prompts), default=0)

    # ------------------------------------------------------------------ ids
    def _next_event_id(self) -> int:
        self._event_counter += 1
        return self._event_counter

    def _next_decision_id(self) -> int:
        self._decision_counter += 1
        return self._decision_counter

    @property
    def replaying(self) -> bool:
        return bool(self._replay)

    # -------------------------------------------------------------- snapshots
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return None if f != f else f  # NaN -> None

    @classmethod
    def build_technical_snapshot(cls, context,
                                 technical_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        OHLCV + strategy-required indicator values at the signal bar, plus
        derived recent-price-action features. Uses only the look-ahead-safe
        context data (bars 0..current_index).
        """
        snapshot: Dict[str, Any] = {}
        bar = context.current_bar
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col in bar.index:
                snapshot[col] = cls._safe_float(bar[col])
        for col in technical_columns or []:
            if col in bar.index and col not in snapshot:
                snapshot[col] = cls._safe_float(bar[col])

        idx = context.current_index
        if 'close' in context.data:
            closes = context.data['close']
            c_now = cls._safe_float(closes.iloc[idx])
            if c_now:
                for n in (5, 21, 63):
                    if idx - n >= 0:
                        prev = cls._safe_float(closes.iloc[idx - n])
                        if prev:
                            snapshot[f'pct_change_{n}d'] = round((c_now / prev - 1) * 100, 4)
                window = closes.iloc[max(0, idx - 251):idx + 1]
                hi, lo = cls._safe_float(window.max()), cls._safe_float(window.min())
                if hi:
                    snapshot['dist_from_252d_high_pct'] = round((c_now / hi - 1) * 100, 4)
                if lo:
                    snapshot['dist_from_252d_low_pct'] = round((c_now / lo - 1) * 100, 4)
        return snapshot

    @staticmethod
    def _bar_date(context) -> str:
        date = context.current_date
        return date.date().isoformat() if hasattr(date, 'date') else str(date)

    def _build_event(self, event_kind: str, symbol: str, bar_date: str,
                     bar_index: int, signal: Optional[Signal],
                     technical_snapshot: Optional[Dict[str, Any]] = None,
                     portfolio_snapshot: Optional[Dict[str, Any]] = None,
                     signal_type: Optional[str] = None,
                     direction: Optional[str] = None,
                     reason: str = "") -> SignalEvent:
        return SignalEvent(
            event_id=self._next_event_id(),
            run_id=self.manifest.run_id,
            event_kind=event_kind,
            symbol=symbol,
            bar_date=bar_date,
            bar_index=bar_index,
            signal_type=(signal_type if signal_type is not None
                         else signal.type.value if signal else "HOLD"),
            direction=(direction if direction is not None
                       else signal.direction.value if signal else "LONG"),
            proposed_size=float(signal.size) if signal else 0.0,
            proposed_stop_loss=signal.stop_loss if signal else None,
            proposed_take_profit=signal.take_profit if signal else None,
            new_stop_loss=signal.new_stop_loss if signal else None,
            signal_reason=(reason or (signal.reason if signal else "")),
            technical_snapshot=technical_snapshot or {},
            portfolio_snapshot=portfolio_snapshot or {},
        )

    # ----------------------------------------------------------------- replay
    def _pop_replay(self, event: SignalEvent) -> Optional[DecisionRecord]:
        """
        During resume-replay, verify the regenerated event against the
        next logged pair and return the logged decision. Returns None in
        live mode.
        """
        if not self._replay:
            return None
        logged_event, logged_decision = self._replay.popleft()
        if (logged_event.event_id != event.event_id
                or logged_event.fingerprint != event.fingerprint):
            raise ReplayMismatchError(
                f"Replay diverged at event {event.event_id}: regenerated "
                f"{event.event_kind} {event.symbol} {event.signal_type} on "
                f"{event.bar_date} (fingerprint {event.fingerprint}) does not "
                f"match logged event {logged_event.event_id} "
                f"{logged_event.event_kind} {logged_event.symbol} "
                f"{logged_event.signal_type} on {logged_event.bar_date} "
                f"(fingerprint {logged_event.fingerprint}). The underlying "
                f"data, strategy, or config has changed; this run cannot be "
                f"resumed safely."
            )
        # Keep counters aligned with the log.
        self._decision_counter = max(self._decision_counter, logged_decision.decision_id)
        return logged_decision

    def _persist(self, event: SignalEvent, record: DecisionRecord,
                 prompt: Optional[PromptRecord] = None) -> None:
        if event.event_id not in self._events_already_logged:
            self.store.append_event(event)
            self._events_already_logged.add(event.event_id)
        if prompt is not None:
            self.store.append_prompt(prompt)
        self.store.append_decision(record)

    # ------------------------------------------------------------- decisions
    def _record_from_response(self, event: SignalEvent,
                              response: DecisionResponse,
                              duration_secs: float,
                              prompt_id: Optional[int]) -> DecisionRecord:
        return DecisionRecord(
            decision_id=self._next_decision_id(),
            event_id=event.event_id,
            run_id=self.manifest.run_id,
            action=response.action,
            source=response.source,
            size_factor=response.size_factor,
            modified_stop_loss=response.modified_stop_loss,
            modified_take_profit=response.modified_take_profit,
            rationale=response.rationale,
            capital_resolution=response.capital_resolution,
            prompt_id=prompt_id,
            decided_at=datetime.now().isoformat(timespec='seconds'),
            decision_duration_secs=round(duration_secs, 3),
        )

    @staticmethod
    def _effective_signal(signal: Signal, record: DecisionRecord) -> Signal:
        """Translate a decision into the Signal the engine dispatches."""
        action = record.action
        if action == DecisionAction.ACCEPT:
            return signal
        if action == DecisionAction.MODIFY:
            new_size = signal.size
            if record.size_factor:
                new_size = max(0.0, min(1.0, signal.size * float(record.size_factor)))
            return replace(
                signal,
                size=new_size,
                stop_loss=(record.modified_stop_loss
                           if record.modified_stop_loss is not None else signal.stop_loss),
                take_profit=(record.modified_take_profit
                             if record.modified_take_profit is not None else signal.take_profit),
            )
        if action == DecisionAction.REJECT:
            return Signal.hold(reason=f"Rejected by user: {record.rationale}".strip(': '))
        if action == DecisionAction.DEFER:
            return Signal.hold(reason=f"Deferred by user: {record.rationale}".strip(': '))
        if action == DecisionAction.AUTO_SUPPRESSED:
            return Signal.hold(reason="Suppressed (rejected signal cooldown)")
        return signal

    def resolve_signal(self, signal: Signal, context,
                       portfolio_snapshot: Optional[Dict[str, Any]] = None,
                       technical_columns: Optional[List[str]] = None,
                       capital_options: Optional[CapitalOptions] = None) -> ResolvedDecision:
        """
        Resolve one actionable strategy signal into an effective Signal.

        Raises RunAbortedByUser if the user pauses the run instead of
        deciding, and ReplayMismatchError on resume divergence.
        """
        event = self._build_event(
            event_kind="STRATEGY_SIGNAL",
            symbol=context.symbol,
            bar_date=self._bar_date(context),
            bar_index=int(context.current_index),
            signal=signal,
            technical_snapshot=self.build_technical_snapshot(context, technical_columns),
            portfolio_snapshot=portfolio_snapshot or {},
        )

        suppressed_by = self.tracker.check(event.symbol, event.signal_type,
                                           event.bar_index)

        logged = self._pop_replay(event)
        if logged is not None:
            if suppressed_by is not None and logged.action != DecisionAction.AUTO_SUPPRESSED:
                raise ReplayMismatchError(
                    f"Replay diverged at event {event.event_id}: suppression "
                    f"state expects AUTO_SUPPRESSED but the log recorded "
                    f"{logged.action.value}."
                )
            if suppressed_by is None and logged.action == DecisionAction.AUTO_SUPPRESSED:
                raise ReplayMismatchError(
                    f"Replay diverged at event {event.event_id}: the log "
                    f"recorded AUTO_SUPPRESSED but suppression state does "
                    f"not expect it."
                )
            if logged.action == DecisionAction.REJECT:
                self.tracker.on_reject(event.symbol, event.signal_type,
                                       logged.decision_id, event.bar_index)
            return ResolvedDecision(
                effective_signal=self._effective_signal(signal, logged),
                event=event, record=logged,
            )

        if suppressed_by is not None:
            record = DecisionRecord(
                decision_id=self._next_decision_id(),
                event_id=event.event_id,
                run_id=self.manifest.run_id,
                action=DecisionAction.AUTO_SUPPRESSED,
                source=DecisionSource.AUTO,
                suppressed_by=suppressed_by,
                decided_at=datetime.now().isoformat(timespec='seconds'),
            )
            self._persist(event, record)
            return ResolvedDecision(
                effective_signal=self._effective_signal(signal, record),
                event=event, record=record,
            )

        request = DecisionRequest(
            kind="SIGNAL",
            event=event,
            chart_data=self._chart_slice(context),
            capital_options=capital_options,
            warning=(SELL_REJECT_WARNING
                     if event.signal_type == SignalType.SELL.value else ""),
        )
        current_price = float(context.current_price)
        # The event is durable before the (possibly long) human decision;
        # if the run is abandoned here, the resume loader drops the
        # dangling event and it regenerates identically.
        if event.event_id not in self._events_already_logged:
            self.store.append_event(event)
            self._events_already_logged.add(event.event_id)

        started = time.monotonic()
        response = self.provider.decide(request)
        if response.action == DecisionAction.ABORT:
            raise RunAbortedByUser(
                f"Run paused by user at {event.symbol} {event.signal_type} "
                f"on {event.bar_date}."
            )

        prompt = self._build_prompt(event, response)
        record = self._record_from_response(
            event, response, time.monotonic() - started,
            prompt.prompt_id if prompt else None,
        )
        if record.action == DecisionAction.REJECT:
            self.tracker.on_reject(event.symbol, event.signal_type,
                                   record.decision_id, event.bar_index)
        effective = self._effective_signal(signal, record)
        self._validate_modified_stop(effective, current_price)
        self._persist(event, record, prompt)
        return ResolvedDecision(
            effective_signal=effective, event=event, record=record,
        )

    @staticmethod
    def _validate_modified_stop(signal: Signal, current_price: float) -> None:
        """
        A user-modified stop on the wrong side of the price would trigger
        immediately (and fails Position validation deep in the engine) —
        fail loudly here with an actionable message instead.
        """
        if signal.type != SignalType.BUY or signal.stop_loss is None:
            return
        is_short = getattr(signal.direction, 'value', str(signal.direction)) == "SHORT"
        if not is_short and signal.stop_loss >= current_price:
            raise ValueError(
                f"Modified stop loss {signal.stop_loss} must be below the "
                f"current price {current_price} for a LONG entry.")
        if is_short and signal.stop_loss <= current_price:
            raise ValueError(
                f"Modified stop loss {signal.stop_loss} must be above the "
                f"current price {current_price} for a SHORT entry.")

    def _build_prompt(self, event: SignalEvent,
                      response: DecisionResponse) -> Optional[PromptRecord]:
        if not response.prompt_text:
            return None
        self._prompt_counter += 1
        return PromptRecord(
            prompt_id=self._prompt_counter,
            event_id=event.event_id,
            run_id=self.manifest.run_id,
            generated_at=datetime.now().isoformat(timespec='seconds'),
            horizon_days=response.prompt_horizon_days,
            prompt_text=response.prompt_text,
            response_summary=response.response_summary,
            response_pasted_at=(datetime.now().isoformat(timespec='seconds')
                                if response.response_summary else ""),
        )

    @staticmethod
    def _chart_slice(context):
        try:
            return context.data.tail(CHART_BARS)
        except Exception:
            return None

    # -------------------------------------------------- capital contingency
    def resolve_capital(self, parent: ResolvedDecision,
                        options: CapitalOptions) -> Dict[str, Any]:
        """
        Ask how to resolve an accepted BUY that does not fit in available
        capital (portfolio mode). Returns a capital_resolution dict:

            {'choice': 'reduce_size'}                      # open at affordable size
            {'choice': 'free_capital', 'free_actions': [
                {'symbol': ..., 'action': 'close'},
                {'symbol': ..., 'action': 'trim', 'fraction': 0.5}, ...]}
            {'choice': 'reject'}

        Logged as its own child SignalEvent (event_kind
        CAPITAL_RESOLUTION) + DecisionRecord so it replays exactly.
        A capital-stage reject starts the same reject cooldown as a
        signal-stage reject.
        """
        parent_event = parent.event
        event = self._build_event(
            event_kind="CAPITAL_RESOLUTION",
            symbol=parent_event.symbol,
            bar_date=parent_event.bar_date,
            bar_index=parent_event.bar_index,
            signal=None,
            signal_type=parent_event.signal_type,
            direction=parent_event.direction,
            reason=(f"Insufficient capital: required "
                    f"{options.required_capital:.2f}, available "
                    f"{options.available_capital:.2f}"),
            portfolio_snapshot={
                'required_capital': options.required_capital,
                'available_capital': options.available_capital,
                'affordable_fraction': options.affordable_fraction,
                'parent_event_id': parent_event.event_id,
            },
        )

        logged = self._pop_replay(event)
        if logged is not None:
            resolution = logged.capital_resolution or {'choice': 'reject'}
            if resolution.get('choice') == CapitalResolutionChoice.REJECT.value:
                self.tracker.on_reject(event.symbol, parent_event.signal_type,
                                       logged.decision_id, event.bar_index)
            return resolution

        if event.event_id not in self._events_already_logged:
            self.store.append_event(event)
            self._events_already_logged.add(event.event_id)

        request = DecisionRequest(kind="CAPITAL_RESOLUTION", event=event,
                                  capital_options=options)
        started = time.monotonic()
        response = self.provider.decide(request)
        if response.action == DecisionAction.ABORT:
            raise RunAbortedByUser(
                f"Run paused by user during capital resolution for "
                f"{event.symbol} on {event.bar_date}."
            )

        resolution = response.capital_resolution or {'choice': 'reject'}
        action = (DecisionAction.REJECT
                  if resolution.get('choice') == CapitalResolutionChoice.REJECT.value
                  else DecisionAction.MODIFY)
        record = DecisionRecord(
            decision_id=self._next_decision_id(),
            event_id=event.event_id,
            run_id=self.manifest.run_id,
            action=action,
            source=response.source,
            rationale=response.rationale,
            capital_resolution=resolution,
            decided_at=datetime.now().isoformat(timespec='seconds'),
            decision_duration_secs=round(time.monotonic() - started, 3),
        )
        if action == DecisionAction.REJECT:
            self.tracker.on_reject(event.symbol, parent_event.signal_type,
                                   record.decision_id, event.bar_index)
        self._persist(event, record)
        return resolution

    # ------------------------------------------------------------- auto logs
    def log_auto(self, event_kind: str, symbol: str, bar_date: str,
                 bar_index: int, signal_type: str, direction: str = "LONG",
                 reason: str = "", signal: Optional[Signal] = None,
                 portfolio_snapshot: Optional[Dict[str, Any]] = None) -> int:
        """
        Log an auto-applied event (ADJUST_STOP, engine stop/TP exit,
        final-bar drop, discretionary funding leg) with an AUTO_APPLIED
        decision record. Returns the event_id for outcome linking.
        """
        event = self._build_event(
            event_kind=event_kind, symbol=symbol, bar_date=bar_date,
            bar_index=bar_index, signal=signal, signal_type=signal_type,
            direction=direction, reason=reason,
            portfolio_snapshot=portfolio_snapshot or {},
        )
        logged = self._pop_replay(event)
        if logged is not None:
            return event.event_id
        record = DecisionRecord(
            decision_id=self._next_decision_id(),
            event_id=event.event_id,
            run_id=self.manifest.run_id,
            action=DecisionAction.AUTO_APPLIED,
            source=DecisionSource.AUTO,
            rationale=reason,
            decided_at=datetime.now().isoformat(timespec='seconds'),
        )
        self._persist(event, record)
        return event.event_id

    # -------------------------------------------------------------- outcomes
    def record_outcome(self, event_id: int, executed: bool,
                       quantity: Optional[float] = None,
                       price: Optional[float] = None,
                       reason: str = "") -> None:
        """Persist what actually executed for an event (post-dispatch)."""
        if event_id <= self._resume_last_event_id:
            return  # outcome already on disk from the original run
        self.store.append_outcome(OutcomeRecord(
            event_id=event_id, executed=executed,
            executed_quantity=quantity, executed_price=price, reason=reason,
        ))

    # ---------------------------------------------------- position lifecycle
    def on_position_opened(self, symbol: str) -> None:
        self.tracker.on_position_opened(symbol)

    def on_position_closed(self, symbol: str) -> None:
        self.tracker.on_position_closed(symbol)

    # -------------------------------------------------------------- finalize
    def finalize(self, status: str) -> None:
        """Rewrite the manifest with the final status and record counts."""
        self.manifest.status = status
        self.manifest.finalized_at = datetime.now().isoformat(timespec='seconds')
        self.manifest.counts = {
            'events': len(self.store.load_events()),
            'decisions': len(self.store.load_decisions()),
            'outcomes': len(self.store.load_outcomes()),
            'prompts': len(self.store.load_prompts()),
        }
        self.store.write_manifest(self.manifest)
