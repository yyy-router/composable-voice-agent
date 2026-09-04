"""OpenAI-compatible provider tests."""

from __future__ import annotations

import json

import httpx
import pytest

from composable_voice_agent import ChatMessage, LlmContext, ToolDefinition
from composable_voice_agent.contracts import LlmCompleted, LlmUsage, TextDelta, ToolCallDelta
from composable_voice_agent.providers import (
    OpenAICompatibleConfig,
    OpenAICompatibleLlm,
    ProviderProtocolError,
    ProviderRequestError,
)


def sse(*payloads: object) -> bytes:
    lines = [f"data: {json.dumps(payload)}\n" for payload in payloads]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def provider_response(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert str(request.url) == "https://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=sse(
            {"choices": [{"delta": {"content": "hel"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "lookup", "arguments": '{"a":'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "1}"},
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ),
        request=request,
    )


@pytest.mark.asyncio
async def test_provider_normalizes_text_tool_calls_and_usage() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(provider_response),
        base_url="https://llm.example/v1",
    )
    provider = OpenAICompatibleLlm(
        OpenAICompatibleConfig("https://llm.example/v1", "secret", "model"),
        client=client,
    )
    tools = [ToolDefinition("lookup", "Look up", {"type": "object", "properties": {}})]

    events = [
        event
        async for event in provider.stream(
            [ChatMessage("user", "hello")],
            tools,
            LlmContext(temperature=0.2),
        )
    ]

    assert events == [
        TextDelta("hel"),
        ToolCallDelta(0, "call-1", "lookup", '{"a":'),
        ToolCallDelta(0, None, None, "1}"),
        LlmCompleted(LlmUsage(prompt_tokens=3, completion_tokens=2)),
    ]
    await provider.aclose()


@pytest.mark.asyncio
async def test_provider_maps_http_error_without_leaking_secret() -> None:
    def error_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"secret backend details", request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(error_response),
        base_url="https://llm.example",
    )
    provider = OpenAICompatibleLlm(
        OpenAICompatibleConfig("https://llm.example", "secret", "model"), client=client
    )

    with pytest.raises(ProviderRequestError) as caught:
        _ = [event async for event in provider.stream([], [])]

    assert caught.value.code == "provider_http_error"
    assert "secret" not in str(caught.value)
    await provider.aclose()


@pytest.mark.asyncio
async def test_provider_rejects_invalid_sse_payload() -> None:
    def invalid_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: nope\n\ndata: [DONE]\n\n",
            request=request,
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(invalid_response),
        base_url="https://llm.example",
    )
    provider = OpenAICompatibleLlm(
        OpenAICompatibleConfig("https://llm.example", "secret", "model"), client=client
    )

    with pytest.raises(ProviderProtocolError):
        _ = [event async for event in provider.stream([], [])]
    await provider.aclose()


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleConfig("", "secret", "model")
    with pytest.raises(ValueError):
        OpenAICompatibleConfig("https://llm.example", "", "model")


__all__ = []
