"""Versioned WebSocket message protocol."""

from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1

CLIENT_SESSION_START = "session.start"
CLIENT_AUDIO_START = "audio.start"
CLIENT_AUDIO_END = "audio.end"
CLIENT_SESSION_CLOSE = "session.close"

SERVER_SESSION_READY = "session.ready"
SERVER_TRANSCRIPT_COMPLETED = "transcript.completed"
SERVER_AGENT_EVENT = "agent.event"
SERVER_AUDIO_STARTED = "audio.started"
SERVER_AUDIO_END = "audio.completed"
SERVER_SESSION_COMPLETED = "session.completed"
SERVER_SESSION_ERROR = "session.error"


class ProtocolError(ValueError):
    """A control frame is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class ControlMessage:
    type: str
    payload: dict[str, Any]
    request_id: str | None = None
    version: int = PROTOCOL_VERSION


def parse_control_message(raw: object) -> ControlMessage:
    if not isinstance(raw, dict):
        raise ProtocolError("control message must be an object")
    message_type = raw.get("type")
    if not isinstance(message_type, str) or not message_type.strip():
        raise ProtocolError("control message type must be a non-empty string")
    version = raw.get("version", PROTOCOL_VERSION)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version!r}")
    payload = raw.get("payload", {})
    if not isinstance(payload, dict):
        raise ProtocolError("control message payload must be an object")
    request_id = raw.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise ProtocolError("request_id must be a string")
    return ControlMessage(message_type, payload, request_id, version)


def event_message(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    if not event_type.strip():
        raise ValueError("event type must be non-empty")
    message: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "type": event_type,
        "payload": payload or {},
    }
    if request_id is not None:
        message["request_id"] = request_id
    return message
