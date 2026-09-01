"""Agent output contracts."""

from dataclasses import dataclass
from typing import TypeAlias

from .tools import ToolResult


@dataclass(frozen=True, slots=True)
class AgentTurnStarted:
    """The runtime accepted a new user turn."""


@dataclass(frozen=True, slots=True)
class AgentTextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class AgentQuestion:
    question_id: str
    text: str
    expected_type: str | None = None
    options: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallCompleted:
    call_id: str
    name: str
    result: ToolResult


@dataclass(frozen=True, slots=True)
class AgentCompleted:
    pass


@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str


AgentEvent: TypeAlias = (
    AgentTurnStarted
    | AgentTextDelta
    | AgentQuestion
    | ToolCallStarted
    | ToolCallCompleted
    | AgentCompleted
    | AgentError
)
