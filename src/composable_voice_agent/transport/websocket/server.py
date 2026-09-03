"""WebSocket server adapter for the provider-neutral voice runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from ...contracts import (
    AgentCompleted,
    AgentError,
    AgentQuestion,
    AgentTextDelta,
    AudioFormat,
    ToolCallCompleted,
    ToolCallStarted,
    VoiceAgentEvent,
    VoiceAudioCanceled,
    VoiceAudioChunk,
    VoiceAudioCompleted,
    VoiceAudioStarted,
    VoiceEvent,
    VoiceSessionContext,
    VoiceTranscriptCompleted,
)
from ..ports import Authenticator, AuthRequest
from .protocol import (
    CLIENT_AUDIO_END,
    CLIENT_AUDIO_START,
    CLIENT_SESSION_CLOSE,
    CLIENT_SESSION_START,
    SERVER_AGENT_EVENT,
    SERVER_AUDIO_END,
    SERVER_AUDIO_STARTED,
    SERVER_SESSION_COMPLETED,
    SERVER_SESSION_ERROR,
    SERVER_SESSION_READY,
    SERVER_TRANSCRIPT_COMPLETED,
    ControlMessage,
    ProtocolError,
    event_message,
    parse_control_message,
)


class WebSocketConnection(Protocol):
    """The small WebSocket surface required by this transport."""

    async def accept(self) -> None: ...

    async def receive(self) -> Mapping[str, Any]: ...

    async def send_json(self, data: Mapping[str, object]) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...

    async def close(self, code: int = 1000) -> None: ...


class VoiceRuntimePort(Protocol):
    """The runtime surface consumed by the transport."""

    def handle_audio(
        self,
        audio_chunks: AsyncIterator[bytes],
        context: VoiceSessionContext,
    ) -> AsyncIterator[VoiceEvent]: ...

    async def close_session(self, session_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AudioLimits:
    max_chunk_bytes: int = 512 * 1024
    max_stream_bytes: int = 16 * 1024 * 1024
    max_queue_chunks: int = 32

    def __post_init__(self) -> None:
        if min(self.max_chunk_bytes, self.max_stream_bytes, self.max_queue_chunks) <= 0:
            raise ValueError("audio limits must be positive")
        if self.max_chunk_bytes > self.max_stream_bytes:
            raise ValueError("max_chunk_bytes cannot exceed max_stream_bytes")


@dataclass(slots=True)
class _AudioStream:
    stream_id: str
    context: VoiceSessionContext
    queue: asyncio.Queue[bytes | None]
    total_bytes: int = 0
    task: asyncio.Task[None] | None = None


class WebSocketVoiceServer:
    """Serve one provider-neutral voice session over a WebSocket connection."""

    def __init__(
        self,
        runtime: VoiceRuntimePort,
        authenticator: Authenticator,
        *,
        limits: AudioLimits | None = None,
        handshake_timeout_seconds: float = 10.0,
    ) -> None:
        if handshake_timeout_seconds <= 0:
            raise ValueError("handshake_timeout_seconds must be positive")
        self._runtime = runtime
        self._authenticator = authenticator
        self._limits = limits or AudioLimits()
        self._handshake_timeout = handshake_timeout_seconds

    async def serve(self, websocket: WebSocketConnection) -> None:
        await websocket.accept()
        session_id: str | None = None
        stream: _AudioStream | None = None
        try:
            first = await asyncio.wait_for(websocket.receive(), self._handshake_timeout)
            try:
                message = self._parse_json_frame(first)
                if message.type != CLIENT_SESSION_START:
                    raise ProtocolError("first message must be session.start")
            except (ProtocolError, json.JSONDecodeError, TypeError, ValueError):
                await self._error(
                    websocket,
                    None,
                    "invalid_session_start",
                    "invalid session.start message",
                )
                return

            try:
                auth = await self._authenticator.authenticate(
                    AuthRequest(
                        headers=websocket_headers(websocket),
                        query=websocket_query(websocket),
                    )
                )
            except Exception:
                await self._error(
                    websocket,
                    message.request_id,
                    "authentication_error",
                    "authentication service failed",
                )
                return
            if not auth.accepted:
                await self._error(
                    websocket,
                    message.request_id,
                    "authentication_failed",
                    "authentication failed",
                )
                return

            requested_session_id = message.payload.get("session_id")
            session_id = (
                requested_session_id
                if isinstance(requested_session_id, str) and requested_session_id.strip()
                else f"session_{uuid4().hex}"
            )
            await websocket.send_json(
                event_message(
                    SERVER_SESSION_READY,
                    {"session_id": session_id},
                    request_id=message.request_id,
                )
            )

            while True:
                frame = await websocket.receive()
                if frame.get("type") == "websocket.disconnect":
                    break
                if isinstance(frame.get("bytes"), bytes):
                    if stream is None:
                        await self._error(
                            websocket,
                            None,
                            "audio_not_started",
                            "send audio.start first",
                        )
                        continue
                    chunk = frame["bytes"]
                    if len(chunk) > self._limits.max_chunk_bytes:
                        await self._error(
                            websocket,
                            None,
                            "audio_chunk_too_large",
                            "audio chunk exceeds limit",
                        )
                        await self._stop_stream(stream)
                        stream = None
                        continue
                    if stream.total_bytes + len(chunk) > self._limits.max_stream_bytes:
                        await self._error(
                            websocket,
                            None,
                            "audio_stream_too_large",
                            "audio stream exceeds limit",
                        )
                        await self._stop_stream(stream)
                        stream = None
                        continue
                    stream.total_bytes += len(chunk)
                    await stream.queue.put(chunk)
                    continue

                try:
                    message = self._parse_json_frame(frame)
                except (ProtocolError, json.JSONDecodeError, TypeError, ValueError):
                    await self._error(
                        websocket,
                        None,
                        "invalid_message",
                        "invalid control message",
                    )
                    continue

                if message.type == CLIENT_AUDIO_START:
                    if stream is not None:
                        await self._error(
                            websocket,
                            message.request_id,
                            "audio_already_started",
                            "audio stream is already active",
                        )
                        continue
                    try:
                        stream = self._new_stream(
                            message,
                            session_id,
                            auth.identity,
                            auth.metadata,
                        )
                    except (TypeError, ValueError):
                        await self._error(
                            websocket,
                            message.request_id,
                            "invalid_audio_start",
                            "invalid audio.start message",
                        )
                        continue
                    stream.task = asyncio.create_task(self._run_stream(websocket, stream))
                elif message.type == CLIENT_AUDIO_END:
                    if stream is None:
                        await self._error(
                            websocket,
                            message.request_id,
                            "audio_not_started",
                            "no active audio stream",
                        )
                    else:
                        await self._stop_stream(stream)
                        stream = None
                elif message.type == CLIENT_SESSION_CLOSE:
                    break
                else:
                    await self._error(
                        websocket,
                        message.request_id,
                        "unsupported_message",
                        "unsupported message type",
                    )
        except TimeoutError:
            await self._error(
                websocket,
                None,
                "handshake_timeout",
                "session.start timed out",
            )
        finally:
            if stream is not None:
                await self._stop_stream(stream)
            if session_id is not None:
                await self._runtime.close_session(session_id)
                await websocket.send_json(
                    event_message(SERVER_SESSION_COMPLETED, {"session_id": session_id})
                )

    @staticmethod
    def _parse_json_frame(frame: Mapping[str, Any]) -> ControlMessage:
        text = frame.get("text")
        if frame.get("type") != "websocket.receive" or not isinstance(text, str):
            raise ProtocolError("expected a JSON text frame")
        return parse_control_message(json.loads(text))

    def _new_stream(
        self,
        message: Any,
        session_id: str,
        identity: Any,
        metadata: Mapping[str, object],
    ) -> _AudioStream:
        payload = message.payload
        raw_format = payload.get("audio_format", {})
        if not isinstance(raw_format, Mapping):
            raise ValueError("audio_format must be an object")
        audio_format = AudioFormat(
            encoding=str(raw_format.get("encoding", "pcm_s16le")),
            sample_rate_hz=int(raw_format.get("sample_rate_hz", 16000)),
            channels=int(raw_format.get("channels", 1)),
        )
        stream_id = payload.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id.strip():
            stream_id = f"stream_{uuid4().hex}"
        return _AudioStream(
            stream_id,
            VoiceSessionContext(
                session_id=session_id,
                identity=identity,
                metadata=metadata,
                voice_mode=str(payload.get("voice_mode", "push_to_talk")),
                audio_format=audio_format,
            ),
            asyncio.Queue(self._limits.max_queue_chunks),
        )

    async def _run_stream(self, websocket: WebSocketConnection, stream: _AudioStream) -> None:
        async def chunks() -> AsyncIterator[bytes]:
            while True:
                chunk = await stream.queue.get()
                if chunk is None:
                    return
                yield chunk

        try:
            async for event in self._runtime.handle_audio(chunks(), stream.context):
                await self._send_event(websocket, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._error(websocket, stream.stream_id, "runtime_error", "voice runtime failed")

    async def _stop_stream(self, stream: _AudioStream) -> None:
        await stream.queue.put(None)
        if stream.task is not None:
            await asyncio.gather(stream.task, return_exceptions=True)

    async def _send_event(self, websocket: WebSocketConnection, event: VoiceEvent) -> None:
        if isinstance(event, VoiceTranscriptCompleted):
            await websocket.send_json(
                event_message(SERVER_TRANSCRIPT_COMPLETED, {"text": event.text})
            )
        elif isinstance(event, VoiceAgentEvent):
            inner = event.event
            if isinstance(inner, AgentTextDelta):
                payload: dict[str, object] = {"event": "text.delta", "text": inner.text}
            elif isinstance(inner, AgentQuestion):
                payload = {
                    "event": "question",
                    "question_id": inner.question_id,
                    "text": inner.text,
                }
            elif isinstance(inner, AgentCompleted):
                payload = {"event": "completed"}
            elif isinstance(inner, AgentError):
                payload = {"event": "error", "code": inner.code, "message": inner.message}
            elif isinstance(inner, ToolCallStarted):
                payload = {
                    "event": "tool.started",
                    "call_id": inner.call_id,
                    "name": inner.name,
                }
            elif isinstance(inner, ToolCallCompleted):
                payload = {
                    "event": "tool.completed",
                    "call_id": inner.call_id,
                    "name": inner.name,
                    "success": inner.result.success,
                }
            else:
                payload = {"event": "agent.event"}
            await websocket.send_json(event_message(SERVER_AGENT_EVENT, payload))
        elif isinstance(event, VoiceAudioStarted):
            await websocket.send_json(
                event_message(
                    SERVER_AUDIO_STARTED,
                    {"audio_id": event.audio_id, "text": event.text},
                )
            )
        elif isinstance(event, VoiceAudioChunk):
            await websocket.send_bytes(event.data)
        elif isinstance(event, VoiceAudioCompleted):
            await websocket.send_json(event_message(SERVER_AUDIO_END, {"audio_id": event.audio_id}))
        elif isinstance(event, VoiceAudioCanceled):
            await self._error(
                websocket,
                event.audio_id,
                "audio_canceled",
                "audio output was canceled",
            )

    async def _error(
        self,
        websocket: WebSocketConnection,
        request_id: str | None,
        code: str,
        message: str,
    ) -> None:
        await websocket.send_json(
            event_message(
                SERVER_SESSION_ERROR,
                {"code": code, "message": message},
                request_id=request_id,
            )
        )


def websocket_headers(websocket: WebSocketConnection) -> Mapping[str, str]:
    headers = getattr(websocket, "headers", {})
    return headers if isinstance(headers, Mapping) else {}


def websocket_query(websocket: WebSocketConnection) -> Mapping[str, str]:
    query = getattr(websocket, "query_params", {})
    return query if isinstance(query, Mapping) else {}


__all__ = ["AudioLimits", "WebSocketVoiceServer"]
