"""End-to-end reference example using only local deterministic providers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from composable_voice_agent import (
    AgentRuntime,
    AudioFormat,
    ToolRegistry,
    VoiceRuntime,
    VoiceSessionContext,
)
from composable_voice_agent.providers import FakeAsr, FakeLlm, FakeTts, TextResponse
from composable_voice_agent.transport import AuthRequest, AuthResult
from composable_voice_agent.transport.websocket.server import WebSocketVoiceServer


class FakeAuthenticator:
    async def authenticate(self, request: AuthRequest) -> AuthResult:
        del request
        return AuthResult(accepted=True)


class FakeWebSocket:
    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = list(frames)
        self.sent_json: list[Mapping[str, object]] = []
        self.sent_bytes: list[bytes] = []
        self.headers: Mapping[str, str] = {}
        self.query_params: Mapping[str, str] = {}

    async def accept(self) -> None:
        pass

    async def receive(self) -> Mapping[str, Any]:
        return self.frames.pop(0)

    async def send_json(self, data: Mapping[str, object]) -> None:
        self.sent_json.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int = 1000) -> None:
        del code


def text_frame(message: dict[str, object]) -> dict[str, Any]:
    return {"type": "websocket.receive", "text": json.dumps(message)}


@pytest.mark.asyncio
async def test_reference_providers_complete_voice_runtime_flow() -> None:
    voice = VoiceRuntime(
        asr=FakeAsr("hello"),
        agent=AgentRuntime(FakeLlm([TextResponse("ready")]), ToolRegistry()),
        tts=FakeTts(),
    )

    async def audio() -> AsyncIterator[bytes]:
        yield b"pcm"

    events = [
        event
        async for event in voice.handle_audio(
            audio(), VoiceSessionContext("session-1", audio_format=AudioFormat())
        )
    ]
    assert len(events) == 7
    assert events[0].text == "hello"
    assert events[2].event.text == "ready"
    assert events[3].event.__class__.__name__ == "AgentCompleted"
    assert events[4].text == "ready"
    assert events[5].data == b"ready"


@pytest.mark.asyncio
async def test_reference_server_can_run_without_external_credentials() -> None:
    runtime = VoiceRuntime(
        asr=FakeAsr("hello"),
        agent=AgentRuntime(FakeLlm([TextResponse("ready")]), ToolRegistry()),
        tts=FakeTts(),
    )
    websocket = FakeWebSocket(
        [
            text_frame({"type": "session.start"}),
            text_frame({"type": "audio.start"}),
            {"type": "websocket.receive", "bytes": b"pcm"},
            text_frame({"type": "audio.end"}),
            text_frame({"type": "session.close"}),
        ]
    )

    await WebSocketVoiceServer(runtime, FakeAuthenticator()).serve(websocket)

    assert websocket.sent_bytes == [b"ready"]
    assert websocket.sent_json[-1]["type"] == "session.completed"
