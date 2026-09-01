"""LLM provider contracts."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class LlmUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class LlmCompleted:
    usage: LlmUsage | None = None


LlmEvent: TypeAlias = TextDelta | ToolCallDelta | LlmCompleted


class LlmProvider(Protocol):
    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[object],
        context: Mapping[str, object] | None = None,
    ) -> AsyncIterator[LlmEvent]: ...
