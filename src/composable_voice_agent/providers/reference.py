"""Deterministic providers for local development and tests.

These providers deliberately do not emulate a vendor protocol. They implement the
public contracts directly so an application can exercise the runtime without API keys.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

from ..contracts import (
    AsrContext,
    AsrEvent,
    AsrProvider,
    ChatMessage,
    LlmCompleted,
    LlmContext,
    LlmEvent,
    LlmProvider,
    SpeechSegment,
    TextDelta,
    ToolCallDelta,
    ToolDefinition,
    TranscriptCompleted,
    TtsAudioChunk,
    TtsCompleted,
    TtsEvent,
    TtsProvider,
)


class FakeAsr(AsrProvider):
    """Consume all audio and emit a configured final transcript."""

    def __init__(self, transcripts: str | Sequence[str] = "hello") -> None:
        if isinstance(transcripts, str):
            values = [transcripts]
        else:
            values = list(transcripts)
        if not values or any(not value.strip() for value in values):
            raise ValueError("transcripts must contain at least one non-empty string")
        self._transcripts = deque(values)
        self._last_transcript = values[-1]

    def stream(
        self,
        audio_chunks: AsyncIterable[bytes],
        context: AsrContext,
    ) -> AsyncIterator[AsrEvent]:
        del context
        return self._stream(audio_chunks)

    async def _stream(self, audio_chunks: AsyncIterable[bytes]) -> AsyncIterator[AsrEvent]:
        async for _ in audio_chunks:
            pass
        if self._transcripts:
            self._last_transcript = self._transcripts.popleft()
        yield TranscriptCompleted(self._last_transcript)


@dataclass(frozen=True, slots=True)
class TextResponse:
    """One scripted LLM response containing final text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolResponse:
    """One scripted LLM response containing one function call."""

    name: str
    arguments: Mapping[str, object] | None = None
    call_id: str | None = None
    preamble: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool response name must be non-empty")
        if self.arguments is None:
            object.__setattr__(self, "arguments", {})


LlmResponse = TextResponse | ToolResponse


class FakeLlm(LlmProvider):
    """Return scripted text or tool-call responses one model round at a time."""

    def __init__(self, responses: Sequence[LlmResponse] = ()) -> None:
        self._responses = deque(responses)
        self.calls = 0

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        context: LlmContext | None = None,
    ) -> AsyncIterator[LlmEvent]:
        del messages, tools, context
        self.calls += 1
        response = self._responses.popleft() if self._responses else TextResponse("OK")
        return self._stream(response)

    async def _stream(self, response: LlmResponse) -> AsyncIterator[LlmEvent]:
        if isinstance(response, TextResponse):
            if response.text:
                yield TextDelta(response.text)
        else:
            if response.preamble:
                yield TextDelta(response.preamble)
            call_id = response.call_id or f"call_{self.calls}"
            yield ToolCallDelta(
                index=0,
                call_id=call_id,
                name=response.name,
                arguments=json.dumps(dict(response.arguments or {}), ensure_ascii=False),
            )
        yield LlmCompleted()


class FakeTts(TtsProvider):
    """Encode each text segment as deterministic UTF-8 test audio bytes."""

    def stream(self, segments: AsyncIterable[SpeechSegment]) -> AsyncIterator[TtsEvent]:
        return self._stream(segments)

    async def _stream(self, segments: AsyncIterable[SpeechSegment]) -> AsyncIterator[TtsEvent]:
        characters = 0
        async for segment in segments:
            characters += len(segment.text)
            yield TtsAudioChunk(segment.text.encode("utf-8"))
        yield TtsCompleted(characters=characters)


__all__ = ["FakeAsr", "FakeLlm", "FakeTts", "LlmResponse", "TextResponse", "ToolResponse"]
