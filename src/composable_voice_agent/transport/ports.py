"""Transport contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import Identity


@dataclass(frozen=True, slots=True)
class AuthRequest:
    headers: Mapping[str, str] = field(default_factory=dict)
    query: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuthResult:
    accepted: bool
    identity: Identity | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None


class Authenticator(Protocol):
    async def authenticate(self, request: AuthRequest) -> AuthResult: ...


class EventSink(Protocol):
    async def send_json(self, message: Mapping[str, object]) -> None: ...

    async def send_audio(self, data: bytes) -> None: ...
