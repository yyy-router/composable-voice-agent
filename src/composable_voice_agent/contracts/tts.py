"""TTS provider contracts."""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    index: int
    text: str
    purpose: str = "dialogue"


@dataclass(frozen=True, slots=True)
class TtsAudioChunk:
    data: bytes


@dataclass(frozen=True, slots=True)
class TtsCompleted:
    characters: int | None = None


TtsEvent: TypeAlias = TtsAudioChunk | TtsCompleted


class TtsProvider(Protocol):
    def stream(self, segments: AsyncIterable[SpeechSegment]) -> AsyncIterator[TtsEvent]: ...
