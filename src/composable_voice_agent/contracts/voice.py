"""Voice-level events produced by the ASR, Agent, and TTS pipeline."""

from dataclasses import dataclass
from typing import TypeAlias

from .agent import AgentEvent


@dataclass(frozen=True, slots=True)
class VoiceTranscriptCompleted:
    text: str


@dataclass(frozen=True, slots=True)
class VoiceAgentEvent:
    event: AgentEvent


@dataclass(frozen=True, slots=True)
class VoiceAudioStarted:
    audio_id: str
    text: str


@dataclass(frozen=True, slots=True)
class VoiceAudioChunk:
    audio_id: str
    data: bytes


@dataclass(frozen=True, slots=True)
class VoiceAudioCompleted:
    audio_id: str


@dataclass(frozen=True, slots=True)
class VoiceAudioCanceled:
    audio_id: str


VoiceEvent: TypeAlias = (
    VoiceTranscriptCompleted
    | VoiceAgentEvent
    | VoiceAudioStarted
    | VoiceAudioChunk
    | VoiceAudioCompleted
    | VoiceAudioCanceled
)


__all__ = [
    "VoiceAgentEvent",
    "VoiceAudioCanceled",
    "VoiceAudioChunk",
    "VoiceAudioCompleted",
    "VoiceAudioStarted",
    "VoiceEvent",
    "VoiceTranscriptCompleted",
]
