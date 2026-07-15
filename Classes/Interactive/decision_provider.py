"""
Decision providers: pluggable sources of answers to DecisionRequests.

The engine never talks to a GUI directly — it calls
InteractiveSession.resolve_signal(), which delegates the human-facing
part to a DecisionProvider:

- QueueDecisionProvider bridges to the Tk GUI: it posts the request on
  the existing worker->UI message queue and blocks the worker thread on
  a per-request reply queue until the panel submits.
- ScriptedDecisionProvider answers from a script (tests / headless).
- RandomDecisionProvider coin-flips entries (random-completion mode and
  random-control baselines).
- Resume-replay does not need a provider: the session replays logged
  decisions itself before falling through to the live provider.
"""
import queue
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

from .models import DecisionAction, DecisionRequest, DecisionResponse, DecisionSource


class DecisionProvider(ABC):
    """Source of decisions for interactive signal events."""

    @abstractmethod
    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Return the decision for one request. May block."""


class QueueDecisionProvider(DecisionProvider):
    """
    GUI bridge. Runs on the engine's worker thread: posts
    ("decision_request", (request, reply_queue)) on ``msg_queue`` (the
    queue the Tk main thread already polls with .after) and blocks until
    the panel puts a DecisionResponse on the reply queue. The GUI can
    unblock an abandoned run by posting an ABORT response.
    """

    MSG_TYPE = "decision_request"

    def __init__(self, msg_queue: "queue.Queue"):
        self.msg_queue = msg_queue

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        reply_q: "queue.Queue" = queue.Queue(maxsize=1)
        self.msg_queue.put((self.MSG_TYPE, (request, reply_q)))
        return reply_q.get()


class ScriptedDecisionProvider(DecisionProvider):
    """
    Deterministic provider for tests and headless runs.

    Accepts either a callable ``(request) -> DecisionResponse`` or a list
    of DecisionResponses consumed in order. When a list runs out,
    ``default_action`` is used (accept by default).
    """

    def __init__(self,
                 script: Union[Callable[[DecisionRequest], DecisionResponse],
                               List[DecisionResponse], None] = None,
                 default_action: DecisionAction = DecisionAction.ACCEPT):
        self._callable = script if callable(script) else None
        self._responses = list(script) if isinstance(script, list) else []
        self._index = 0
        self.default_action = default_action
        self.requests: List[DecisionRequest] = []  # captured for assertions

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        if self._callable is not None:
            return self._callable(request)
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response
        return DecisionResponse(action=self.default_action,
                                source=DecisionSource.AUTO)


class RandomDecisionProvider(DecisionProvider):
    """
    Coin-flip provider: each entry signal (BUY / PYRAMID) is ACCEPTed
    with ``approve_probability``, otherwise REJECTed. Exit signals are
    always accepted (randomly refusing exits would leave positions
    dangling on their stops, which is never the intent of a random
    control). An approved entry that does not fit in capital is opened
    at the affordable reduced size.

    Seeded, so a given seed reproduces the same decision sequence.
    Used both for the in-run "decide the rest randomly" hand-off and as
    a standalone random-decision baseline.
    """

    ENTRY_TYPES = ("BUY", "PYRAMID")

    def __init__(self, seed: Optional[int] = None,
                 approve_probability: float = 0.5):
        if not 0.0 <= approve_probability <= 1.0:
            raise ValueError("approve_probability must be within [0, 1]")
        self._rng = random.Random(seed)
        self.approve_probability = approve_probability

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if request.kind == "CAPITAL_RESOLUTION":
            # The entry itself was already (randomly) approved — complete
            # it at whatever size fits rather than flipping a second coin.
            return DecisionResponse(
                action=DecisionAction.MODIFY,
                rationale="Random auto-completion: reduced to affordable size",
                capital_resolution={'choice': 'reduce_size'},
                source=DecisionSource.AUTO,
            )
        if request.event.signal_type not in self.ENTRY_TYPES:
            return DecisionResponse(
                action=DecisionAction.ACCEPT,
                rationale="Random auto-completion: exits always accepted",
                source=DecisionSource.AUTO,
            )
        approved = self._rng.random() < self.approve_probability
        action = DecisionAction.ACCEPT if approved else DecisionAction.REJECT
        return DecisionResponse(
            action=action,
            rationale=f"Random auto-completion: {action.value}",
            source=DecisionSource.AUTO,
        )
