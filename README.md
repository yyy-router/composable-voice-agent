# Composable Voice Agent

A provider-neutral runtime for building composable voice agents.

This project provides contracts and orchestration only. It does not contain a business domain, built-in business tools, authentication implementation, database integration, or vendor credentials.

## Architecture

```text
WebSocket transport
        │
        ▼
VoiceRuntime
   ┌────┼────┐
   ▼    ▼    ▼
  ASR  Agent  TTS
       │
       ▼
   ToolRegistry
       │
       ▼
  Your business tools
```

The host application supplies an ASR provider, an LLM provider, an optional TTS provider, an authenticator, and its own tools. Identity and permissions are injected by the host; they are never model-generated arguments.

## Provider adapters

The runtime never calls a vendor API directly. Each provider adapter owns its vendor's request and response format and translates it into the public contracts:

```text
vendor request/response
        ↕ adapter
public provider contract
        ↕ runtime
Agent / Voice events
```

For example, an `AsrProvider` may use HTTP, WebSocket, an SDK, or a local model. It must translate vendor partial and final transcripts into `TranscriptPreview` and `TranscriptCompleted`. An `LlmProvider` translates `ChatMessage` and `ToolDefinition` into the model's request format, then yields `TextDelta`, `ToolCallDelta`, and `LlmCompleted`. A `TtsProvider` receives ordered `SpeechSegment` values and yields `TtsAudioChunk` values. No vendor-specific event names or credentials appear in the runtime.

```python
class MyLlm:
    def stream(self, messages, tools, context=None):
        return self._stream(messages, tools, context)

    async def _stream(self, messages, tools, context):
        request = convert_to_my_provider_format(messages, tools, context)
        async for response in call_my_provider(request):
            event = convert_from_my_provider_format(response)
            if event is not None:
                yield event
```

## Public contracts

```python
from composable_voice_agent import AgentRuntime, ToolRegistry, VoiceRuntime
from composable_voice_agent.contracts import (
    AgentTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)


class LookupStatus:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="lookup_status",
            description="Look up the current status.",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments, context: ToolContext) -> ToolResult:
        del arguments
        return ToolResult(content="The status is ready.", data={"state": "ready"})


registry = ToolRegistry([LookupStatus()])
agent = AgentRuntime(llm=my_llm, tools=registry, system_prompt=my_prompt)
voice = VoiceRuntime(asr=my_asr, agent=agent, tts=my_tts)
```

### Tool execution

A tool owns its business behavior. The runtime supplies a host-controlled `ToolContext`, parses the model's JSON arguments, enforces the configured timeout, and returns a safe failure result when execution fails. It never interprets the structure of `ToolResult.data`.

```python
from composable_voice_agent import ToolExecutionPolicy

agent = AgentRuntime(
    llm=my_llm,
    tools=registry,
    tool_policy=ToolExecutionPolicy(timeout_seconds=15),
)
```

### Provider choices

Implementations of `AsrProvider`, `LlmProvider`, and `TtsProvider` are intentionally not bundled. A provider can use any API style or local runtime. The core package can also be used without voice: construct `AgentRuntime` directly when ASR and TTS are not needed.

## Design principles

- Business-neutral: no domain models or built-in tools.
- Provider-neutral: ASR, LLM, and TTS are injectable ports.
- Host-controlled identity: authentication and authorization stay outside the model.
- Streaming first: providers use async iterators for incremental results.
- Safe failures: tool and provider failures are represented without leaking internal details.
- Transport-independent core: WebSocket is an adapter, not a requirement of the Agent runtime.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The repository currently contains contracts, a tested Agent/Voice runtime foundation, and a small WebSocket message parser. Provider adapters and a complete WebSocket server are separate integration concerns.

## Scope

This project deliberately does not provide a business scenario. Application authors define their own tools, system prompt, identity model, authorization policy, provider adapters, and transport authentication.
