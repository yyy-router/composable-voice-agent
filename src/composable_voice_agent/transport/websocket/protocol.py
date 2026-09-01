"""A small, provider-neutral WebSocket message protocol."""

from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ControlMessage:
    type: str
    payload: dict[str, Any]


def parse_control_message(raw: object) -> ControlMessage:
    if not isinstance(raw, dict):
        raise ProtocolError("control message must be an object")
    message_type = raw.get("type")
    if not isinstance(message_type, str) or not message_type.strip():
        raise ProtocolError("control message type must be a non-empty string")
    payload = raw.get("payload", {})
    if not isinstance(payload, dict):
        raise ProtocolError("control message payload must be an object")
    return ControlMessage(message_type, payload)


def event_message(event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not event_type.strip():
        raise ValueError("event type must be non-empty")
    return {"type": event_type, "payload": payload or {}}
