"""Voice runtime orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from ..contracts import (
    AgentContext,
    AgentQuestion,
    AsrContext,
    AsrProvider,
    SpeechSegment,
    TranscriptCompleted,
    TtsAudioChunk,
    TtsCompleted,
    TtsProvider,
    VoiceAgentEvent,
    VoiceAudioCanceled,
    VoiceAudioChunk,
    VoiceAudioCompleted,
    VoiceAudioStarted,
    VoiceEvent,
    VoiceSessionContext,
    VoiceTranscriptCompleted,
)
from .agent import AgentRuntime
from .session import SessionStore
from .speech import SpeechPolicy


@dataclass(slots=True)
class VoiceRuntime:
    asr: AsrProvider
    agent: AgentRuntime
    tts: TtsProvider | None = None
    speech_policy: SpeechPolicy | None = None
    sessions: SessionStore | None = None
    _active: dict[str, asyncio.Task[object]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = SessionStore()
        if self.speech_policy is None:
            self.speech_policy = SpeechPolicy()

    async def handle_audio(
        self,
        audio_chunks: AsyncIterable[bytes],
        context: VoiceSessionContext,
    ) -> AsyncIterator[VoiceEvent]:
        task = asyncio.current_task()
        if task is not None:
            self._active[context.session_id] = task
        audio_id: str | None = None
        try:
            assert self.sessions is not None
            session = self.sessions.get_or_create(
                AgentContext(
                    session_id=context.session_id,
                    identity=context.identity,
                    metadata=context.metadata,
                    system_prompt=context.system_prompt,
                )
            )
            transcript: str | None = None
            async for asr_event in self.asr.stream(audio_chunks, AsrContext(context.audio_format)):
                if isinstance(asr_event, TranscriptCompleted):
                    transcript = asr_event.text
            if transcript is None or not transcript.strip():
                return
            yield VoiceTranscriptCompleted(transcript)
            speakable: list[str] = []
            async for event in self.agent.run_turn(session.conversation, transcript, context):
                yield VoiceAgentEvent(event)
                if self.speech_policy is None or not self.speech_policy.should_speak(event):
                    continue
                if isinstance(event, AgentQuestion):
                    speakable.append(event.text)
                else:
                    text = getattr(event, "text", None)
                    if isinstance(text, str) and text.strip():
                        speakable.append(text)
            if self.tts is None or not speakable:
                return
            text = "".join(speakable)
            audio_id = f"audio_{uuid4().hex}"
            yield VoiceAudioStarted(audio_id, text)

            async def segments() -> AsyncIterator[SpeechSegment]:
                yield SpeechSegment(0, text)

            async for tts_event in self.tts.stream(segments()):
                if isinstance(tts_event, TtsAudioChunk):
                    yield VoiceAudioChunk(audio_id, tts_event.data)
                elif isinstance(tts_event, TtsCompleted):
                    yield VoiceAudioCompleted(audio_id)
        except asyncio.CancelledError:
            if audio_id is not None:
                yield VoiceAudioCanceled(audio_id)
            raise
        finally:
            if task is not None and self._active.get(context.session_id) is task:
                self._active.pop(context.session_id, None)

    async def interrupt(self, session_id: str, reason: str = "interrupted") -> None:
        del reason
        task = self._active.get(session_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def close_session(self, session_id: str) -> None:
        await self.interrupt(session_id, "session_closed")
        assert self.sessions is not None
        self.sessions.remove(session_id)


__all__ = ["VoiceRuntime"]
