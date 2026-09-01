"""Agent output contracts."""

from dataclasses import dataclass
from typing import TypeAlias


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
class AgentCompleted:
    pass


@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str


AgentEvent: TypeAlias = AgentTextDelta | AgentQuestion | AgentCompleted | AgentError
