"""OpenAI-compatible streaming chat provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ..contracts import (
    ChatMessage,
    LlmCompleted,
    LlmContext,
    LlmEvent,
    LlmProvider,
    LlmUsage,
    TextDelta,
    ToolCallDelta,
    ToolDefinition,
)
from .errors import ProviderProtocolError, ProviderRequestError


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("base_url and model must be non-empty")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class OpenAICompatibleLlm(LlmProvider):
    """Adapt an OpenAI Chat Completions-compatible SSE endpoint to LlmProvider."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
        )
        self._owns_client = client is None

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        context: LlmContext | None = None,
    ) -> AsyncIterator[LlmEvent]:
        return self._stream(messages, tools, context)

    async def _stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        context: LlmContext | None,
    ) -> AsyncIterator[LlmEvent]:
        body = self._request_body(messages, tools, context)
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                json=body,
            ) as response:
                if response.status_code >= 400:
                    raise ProviderRequestError(
                        "provider_http_error",
                        "the language model provider rejected the request",
                        retryable=response.status_code >= 500,
                    )
                usage: LlmUsage | None = None
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        raise ProviderProtocolError(
                            "provider_sse_error", "the language model returned invalid stream data"
                        )
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ProviderProtocolError(
                            "provider_json_error", "the language model returned invalid JSON"
                        ) from exc
                    if not isinstance(payload, Mapping):
                        raise ProviderProtocolError(
                            "provider_json_error", "the language model returned an invalid event"
                        )
                    parsed_usage = _parse_usage(payload.get("usage"))
                    if parsed_usage is not None:
                        usage = parsed_usage
                    for event in _events_from_payload(payload):
                        yield event
                yield LlmCompleted(usage)
        except ProviderProtocolError:
            raise
        except ProviderRequestError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderRequestError(
                "provider_unavailable", "the language model provider is unavailable", retryable=True
            ) from exc
        except Exception as exc:
            raise ProviderRequestError(
                "provider_request_error", "the language model request failed"
            ) from exc

    def _request_body(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        context: LlmContext | None,
    ) -> dict[str, Any]:
        config = self._config
        body: dict[str, Any] = {
            "model": context.model if context and context.model else config.model,
            "messages": [
                {
                    key: value
                    for key, value in {
                        "role": message.role,
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                        "name": message.name,
                    }.items()
                    if value is not None
                }
                for message in messages
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        temperature = context.temperature if context else config.temperature
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in tools
            ]
        return body

    async def aclose(self) -> None:
        await self._client.aclose()


def _events_from_payload(payload: Mapping[str, Any]) -> tuple[LlmEvent, ...]:
    choices = payload.get("choices", [])
    if not isinstance(choices, list):
        raise ProviderProtocolError(
            "provider_protocol_error", "the language model returned invalid choices"
        )
    events: list[LlmEvent] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            raise ProviderProtocolError(
                "provider_protocol_error", "the language model returned invalid choice"
            )
        delta = choice.get("delta", {})
        if not isinstance(delta, Mapping):
            raise ProviderProtocolError(
                "provider_protocol_error", "the language model returned invalid delta"
            )
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise ProviderProtocolError(
                    "provider_protocol_error", "the language model returned invalid text"
                )
            events.append(TextDelta(content))
        calls = delta.get("tool_calls", [])
        if calls is None:
            continue
        if not isinstance(calls, list):
            raise ProviderProtocolError(
                "provider_protocol_error", "the language model returned invalid tool calls"
            )
        for call in calls:
            if not isinstance(call, Mapping) or not isinstance(call.get("index"), int):
                raise ProviderProtocolError(
                    "provider_protocol_error", "the language model returned invalid tool call"
                )
            function = call.get("function", {})
            if not isinstance(function, Mapping):
                raise ProviderProtocolError(
                    "provider_protocol_error", "the language model returned invalid function"
                )
            arguments = function.get("arguments", "")
            events.append(
                ToolCallDelta(
                    index=call["index"],
                    call_id=call.get("id") if isinstance(call.get("id"), str) else None,
                    name=function.get("name") if isinstance(function.get("name"), str) else None,
                    arguments=arguments if isinstance(arguments, str) else "",
                )
            )
    return tuple(events)


def _parse_usage(value: object) -> LlmUsage | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProviderProtocolError(
            "provider_protocol_error", "the language model returned invalid usage"
        )
    prompt = value.get("prompt_tokens")
    completion = value.get("completion_tokens")
    if (prompt is not None and not isinstance(prompt, int)) or (
        completion is not None and not isinstance(completion, int)
    ):
        raise ProviderProtocolError(
            "provider_protocol_error", "the language model returned invalid usage"
        )
    return LlmUsage(prompt, completion)


__all__ = ["OpenAICompatibleConfig", "OpenAICompatibleLlm"]
