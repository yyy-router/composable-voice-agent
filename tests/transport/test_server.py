from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from composable_voice_agent.contracts import (
    AgentCompleted,
    AgentTextDelta,
    Identity,
    VoiceAgentEvent,
    VoiceAudioChunk,
    VoiceAudioCompleted,
    VoiceAudioStarted,
    VoiceEvent,
    VoiceSessionContext,
    VoiceTranscriptCompleted,
)
from composable_voice_agent.transport import AuthRequest, AuthResult
from composable_voice_agent.transport.websocket.server import AudioLimits, WebSocketVoiceServer


class FakeAuthenticator:
    def __init__(self, result: AuthResult) -> None:
        self.result = result
        self.request: AuthRequest | None = None

    async def authenticate(self, request: AuthRequest) -> AuthResult:
        self.request = request
        return self.result


class FakeRuntime:
    def __init__(self) -> None:
        self.contexts: list[VoiceSessionContext] = []
        self.closed: list[str] = []

    def handle_audio(
        self, audio_chunks: AsyncIterator[bytes], context: VoiceSessionContext
    ) -> AsyncIterator[VoiceEvent]:
        async def events() -> AsyncIterator[VoiceEvent]:
            self.contexts.append(context)
            received = b"".join([chunk async for chunk in audio_chunks])
            yield VoiceTranscriptCompleted(received.decode())
            yield VoiceAgentEvent(AgentTextDelta("ready"))
            yield VoiceAgentEvent(AgentCompleted())
            yield VoiceAudioStarted("audio-1", "ready")
            yield VoiceAudioChunk("audio-1", b"pcm")
            yield VoiceAudioCompleted("audio-1")

        return events()

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)


class FakeWebSocket:
    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = list(frames)
        self.sent_json: list[Mapping[str, object]] = []
        self.sent_bytes: list[bytes] = []
        self.headers = {"authorization": "Bearer test"}
        self.query_params = {"client": "test"}
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> Mapping[str, Any]:
        return self.frames.pop(0)

    async def send_json(self, data: Mapping[str, object]) -> None:
        self.sent_json.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int = 1000) -> None:
        del code


def text_frame(message: dict[str, object]) -> dict[str, Any]:
    import json

    return {"type": "websocket.receive", "text": json.dumps(message)}


@pytest.mark.asyncio
async def test_server_authenticates_and_streams_json_and_binary_events() -> None:
    runtime = FakeRuntime()
    authenticator = FakeAuthenticator(AuthResult(True, Identity("user-1")))
    websocket = FakeWebSocket(
        [
            text_frame({"version": 1, "type": "session.start", "request_id": "r1"}),
            text_frame(
                {
                    "version": 1,
                    "type": "audio.start",
                    "payload": {
                        "stream_id": "stream-1",
                        "audio_format": {
                            "encoding": "pcm_s16le",
                            "sample_rate_hz": 16000,
                            "channels": 1,
                        },
                    },
                }
            ),
            {"type": "websocket.receive", "bytes": b"hello"},
            text_frame({"version": 1, "type": "audio.end"}),
            text_frame({"version": 1, "type": "session.close"}),
        ]
    )

    await WebSocketVoiceServer(runtime, authenticator).serve(websocket)

    assert websocket.accepted
    assert authenticator.request is not None
    assert authenticator.request.headers["authorization"] == "Bearer test"
    assert runtime.contexts[0].identity == Identity("user-1")
    session_id = websocket.sent_json[0]["payload"]["session_id"]
    assert runtime.closed == [session_id]
    assert websocket.sent_bytes == [b"pcm"]
    assert websocket.sent_json[0]["type"] == "session.ready"
    assert websocket.sent_json[-1]["type"] == "session.completed"
    assert any(item["type"] == "transcript.completed" for item in websocket.sent_json)


@pytest.mark.asyncio
async def test_server_rejects_audio_before_audio_start() -> None:
    runtime = FakeRuntime()
    websocket = FakeWebSocket(
        [
            text_frame({"type": "session.start"}),
            {"type": "websocket.receive", "bytes": b"audio"},
            text_frame({"type": "session.close"}),
        ]
    )

    await WebSocketVoiceServer(
        runtime,
        FakeAuthenticator(AuthResult(True, Identity("user-1"))),
    ).serve(websocket)

    assert any(
        message["payload"]["code"] == "audio_not_started"
        for message in websocket.sent_json
        if message["type"] == "session.error"
    )


def test_audio_limits_validate_values() -> None:
    with pytest.raises(ValueError):
        AudioLimits(max_chunk_bytes=2, max_stream_bytes=1)

    with pytest.raises(ValueError):
        AudioLimits(max_queue_chunks=0)
