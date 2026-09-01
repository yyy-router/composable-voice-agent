"""ASR provider contracts."""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from .audio import AudioFormat


@dataclass(frozen=True, slots=True)
class AsrContext:
    audio_format: AudioFormat
    language: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptPreview:
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptCompleted:
    text: str


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    pass


@dataclass(frozen=True, slots=True)
class SpeechStopped:
    pass


AsrEvent: TypeAlias = TranscriptPreview | TranscriptCompleted | SpeechStarted | SpeechStopped


class AsrProvider(Protocol):
    def stream(
        self, audio_chunks: AsyncIterable[bytes], context: AsrContext
    ) -> AsyncIterator[AsrEvent]: ...


class AsrError(Exception):
    """Base class for ASR failures."""
