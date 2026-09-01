"""Session state for independent voice conversations."""

from dataclasses import dataclass, field

from ..contracts import AgentContext
from .agent import Conversation


@dataclass(slots=True)
class Session:
    session_id: str
    context: AgentContext
    conversation: Conversation = field(default_factory=Conversation)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, context: AgentContext) -> Session:
        session = self._sessions.get(context.session_id)
        if session is None:
            session = Session(context.session_id, context)
            self._sessions[context.session_id] = session
        return session

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


__all__ = ["Session", "SessionStore"]
