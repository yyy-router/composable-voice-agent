"""LLM provider contracts."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from .tools import ToolDefinition


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class LlmContext:
    model: str | None = None
    temperature: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


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
        tools: Sequence[ToolDefinition],
        context: LlmContext | None = None,
    ) -> AsyncIterator[LlmEvent]: ...


__all__ = [
    "ChatMessage",
    "LlmCompleted",
    "LlmContext",
    "LlmEvent",
    "LlmProvider",
    "LlmUsage",
    "TextDelta",
    "ToolCall",
    "ToolCallDelta",
]
