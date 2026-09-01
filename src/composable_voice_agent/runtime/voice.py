"""Voice runtime orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

from ..contracts import (
    AgentContext,
    AgentEvent,
    AsrContext,
    AsrProvider,
    TranscriptCompleted,
    TtsProvider,
    VoiceSessionContext,
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

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = SessionStore()

    async def handle_audio(
        self,
        audio_chunks: AsyncIterable[bytes],
        context: VoiceSessionContext,
    ) -> AsyncIterator[AgentEvent]:
        assert self.sessions is not None
        session = self.sessions.get_or_create(
            AgentContext(
                session_id=context.session_id,
                identity=context.identity,
                metadata=context.metadata,
                system_prompt=context.system_prompt,
            )
        )
        transcript = None
        async for event in self.asr.stream(
            audio_chunks, AsrContext(context.audio_format)
        ):
            if isinstance(event, TranscriptCompleted):
                transcript = event.text
        if transcript is None:
            return
        async for event in self.agent.run_turn(session.conversation, transcript, context):
            yield event

    async def interrupt(self, session_id: str, reason: str = "interrupted") -> None:
        del session_id, reason

    async def close_session(self, session_id: str) -> None:
        assert self.sessions is not None
        self.sessions.remove(session_id)


__all__ = ["VoiceRuntime"]
