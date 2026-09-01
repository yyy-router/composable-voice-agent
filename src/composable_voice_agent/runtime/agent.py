"""Provider-neutral agent runtime."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from ..contracts import (
    AgentCompleted,
    AgentContext,
    AgentError,
    AgentEvent,
    AgentTextDelta,
    ChatMessage,
    LlmCompleted,
    LlmProvider,
    TextDelta,
    ToolCallDelta,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


@dataclass(slots=True)
class Conversation:
    messages: list[ChatMessage] = field(default_factory=list)


@dataclass(slots=True)
class AgentRuntime:
    llm: LlmProvider
    tools: ToolRegistry
    system_prompt: str | None = None
    max_tool_rounds: int = 4

    def __post_init__(self) -> None:
        if self.max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")

    def run_turn(
        self,
        conversation: Conversation,
        user_text: str,
        context: AgentContext,
    ) -> AsyncIterator[AgentEvent]:
        return self._run_turn(conversation, user_text, context)

    async def _run_turn(
        self,
        conversation: Conversation,
        user_text: str,
        context: AgentContext,
    ) -> AsyncIterator[AgentEvent]:
        if not user_text.strip():
            yield AgentError("invalid_input", "user text must be non-empty")
            return
        if not conversation.messages and (context.system_prompt or self.system_prompt):
            conversation.messages.append(
                ChatMessage(
                    role="system", content=context.system_prompt or self.system_prompt or ""
                )
            )
        conversation.messages.append(ChatMessage(role="user", content=user_text))
        for round_number in range(self.max_tool_rounds + 1):
            text_parts: list[str] = []
            calls: dict[int, _Call] = {}
            completed: LlmCompleted | None = None
            try:
                async for event in self.llm.stream(
                    conversation.messages,
                    self.tools.definitions(),
                    {"session_id": context.session_id},
                ):
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                        if round_number > 0:
                            yield AgentTextDelta(event.text)
                    elif isinstance(event, ToolCallDelta):
                        calls.setdefault(event.index, _Call()).add(event)
                    elif isinstance(event, LlmCompleted):
                        completed = event
                    else:
                        raise TypeError("unsupported LLM event")
            except Exception:
                yield AgentError("llm_error", "the language model provider failed")
                return
            if completed is None:
                yield AgentError(
                    "llm_protocol_error", "the language model ended without completion"
                )
                return
            if not calls:
                final_text = "".join(text_parts)
                conversation.messages.append(ChatMessage(role="assistant", content=final_text))
                if round_number == 0 and final_text:
                    yield AgentTextDelta(final_text)
                yield AgentCompleted()
                return
            if round_number >= self.max_tool_rounds:
                yield AgentError("tool_round_limit", "maximum tool rounds exceeded")
                return
            conversation.messages.append(
                ChatMessage(role="assistant", content="", name="tool_calls")
            )
            for call in calls.values():
                if not call.name or call.call_id is None:
                    yield AgentError(
                        "invalid_tool_call", "the model returned an incomplete tool call"
                    )
                    return
                try:
                    arguments = json.loads(call.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                    tool = self.tools.get(call.name)
                    result = await tool.execute(
                        arguments,
                        ToolContext(context.session_id, context.identity, context.metadata),
                    )
                    if not isinstance(result, ToolResult):
                        raise TypeError("tool must return ToolResult")
                    content = result.content
                except Exception:
                    content = "The tool failed to execute."
                conversation.messages.append(
                    ChatMessage(
                        role="tool", content=content, tool_call_id=call.call_id, name=call.name
                    )
                )


@dataclass(slots=True)
class _Call:
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""

    def add(self, event: ToolCallDelta) -> None:
        self.call_id = event.call_id or self.call_id or f"call_{uuid4().hex}"
        self.name = event.name or self.name
        self.arguments += event.arguments


__all__ = ["AgentRuntime", "Conversation"]
