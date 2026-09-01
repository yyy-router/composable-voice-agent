from collections.abc import AsyncIterator, Sequence

import pytest

from composable_voice_agent.contracts import (
    AgentCompleted,
    AgentContext,
    AgentTextDelta,
    AgentTurnStarted,
    AsrProvider,
    AudioFormat,
    ChatMessage,
    LlmCompleted,
    TextDelta,
    ToolCallCompleted,
    ToolCallDelta,
    ToolCallStarted,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    TranscriptCompleted,
    TtsAudioChunk,
    TtsCompleted,
    VoiceAgentEvent,
    VoiceAudioChunk,
    VoiceAudioCompleted,
    VoiceAudioStarted,
    VoiceSessionContext,
)
from composable_voice_agent.runtime.agent import AgentRuntime, Conversation
from composable_voice_agent.runtime.voice import VoiceRuntime


class FakeLlm:
    def __init__(self) -> None:
        self.calls = 0

    def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[object], context=None
    ) -> AsyncIterator[object]:
        del messages, tools, context
        self.calls += 1

        async def events():
            if self.calls == 1:
                yield ToolCallDelta(0, "call-1", "status", "{}")
            else:
                yield TextDelta("done")
            yield LlmCompleted()

        return events()


class StatusTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition("status", "", {"type": "object", "properties": {}})

    async def execute(self, arguments, context) -> ToolResult:
        assert arguments == {}
        assert context.session_id == "s1"
        return ToolResult("ready", data={"state": "ready"})


@pytest.mark.asyncio
async def test_agent_executes_tool_then_streams_final_text() -> None:
    runtime = AgentRuntime(FakeLlm(), ToolRegistry([StatusTool()]))
    events = [event async for event in runtime.run_turn(Conversation(), "go", AgentContext("s1"))]
    assert [type(event) for event in events] == [
        AgentTurnStarted,
        ToolCallStarted,
        ToolCallCompleted,
        AgentTextDelta,
        AgentCompleted,
    ]
    assert "".join(event.text for event in events if isinstance(event, AgentTextDelta)) == "done"


class FakeAsr(AsrProvider):
    def stream(self, audio_chunks, context):
        async def events():
            async for _ in audio_chunks:
                pass
            yield TranscriptCompleted("hello")

        return events()


@pytest.mark.asyncio
async def test_voice_runtime_composes_asr_and_agent() -> None:
    runtime = VoiceRuntime(FakeAsr(), AgentRuntime(FakeLlm(), ToolRegistry()))

    async def audio():
        yield b"audio"

    events = [
        event
        async for event in runtime.handle_audio(
            audio(), VoiceSessionContext("s1", audio_format=AudioFormat())
        )
    ]
    assert any(
        isinstance(event, VoiceAgentEvent) and isinstance(event.event, AgentTextDelta)
        for event in events
    )


class FakeTts:
    def stream(self, segments):
        async def events():
            async for segment in segments:
                yield TtsAudioChunk(segment.text.encode())
            yield TtsCompleted()

        return events()


@pytest.mark.asyncio
async def test_voice_runtime_includes_tts_events() -> None:
    runtime = VoiceRuntime(FakeAsr(), AgentRuntime(FakeLlm(), ToolRegistry()), tts=FakeTts())

    async def audio():
        yield b"audio"

    events = [
        event
        async for event in runtime.handle_audio(
            audio(), VoiceSessionContext("s1", audio_format=AudioFormat())
        )
    ]
    assert any(isinstance(event, VoiceAudioStarted) for event in events)
    assert any(isinstance(event, VoiceAudioChunk) for event in events)
    assert any(isinstance(event, VoiceAudioCompleted) for event in events)
