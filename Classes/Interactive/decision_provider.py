"""
Decision providers: pluggable sources of answers to DecisionRequests.

The engine never talks to a GUI directly — it calls
InteractiveSession.resolve_signal(), which delegates the human-facing
part to a DecisionProvider:

- QueueDecisionProvider bridges to the Tk GUI: it posts the request on
  the existing worker->UI message queue and blocks the worker thread on
  a per-request reply queue until the panel submits.
- ScriptedDecisionProvider answers from a script (tests / headless).
- Resume-replay does not need a provider: the session replays logged
  decisions itself before falling through to the live provider.
"""
import queue
from abc import ABC, abstractmethod
from typing import Callable, List, Union

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
