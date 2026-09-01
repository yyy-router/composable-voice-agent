"""Identity and host-provided context contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .audio import AudioFormat


@dataclass(frozen=True, slots=True)
class Identity:
    user_id: str
    tenant_id: str | None = None
    permissions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id must be non-empty")


@dataclass(frozen=True, slots=True)
class AgentContext:
    session_id: str
    identity: Identity | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    identity: Identity | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceSessionContext(AgentContext):
    voice_mode: str = "push_to_talk"
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    timezone: str | None = None
