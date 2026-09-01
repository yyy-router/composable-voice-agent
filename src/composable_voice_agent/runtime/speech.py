"""Speech selection policy."""

from collections.abc import Callable

from ..contracts import AgentEvent, AgentQuestion, AgentTextDelta


class SpeechPolicy:
    """Select user-facing Agent events that should be synthesized."""

    def __init__(self, predicate: Callable[[AgentEvent], bool] | None = None) -> None:
        self._predicate = predicate

    def should_speak(self, event: AgentEvent) -> bool:
        if self._predicate is not None:
            return self._predicate(event)
        return isinstance(event, (AgentTextDelta, AgentQuestion))
