# Composable Voice Agent

A provider-neutral runtime for building composable voice agents.

This project provides contracts, orchestration, deterministic reference providers, an OpenAI-compatible LLM adapter, and an optional WebSocket transport. It does not contain a business domain, built-in business tools, authentication implementation, database integration, or vendor credentials.

## Architecture

```text
WebSocket transport
        │ JSON + binary audio
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

An `AsrProvider` may use HTTP, WebSocket, an SDK, or a local model. It translates vendor partial and final transcripts into `TranscriptPreview` and `TranscriptCompleted`. An `LlmProvider` translates `ChatMessage` and `ToolDefinition` into the model's request format, then yields `TextDelta`, `ToolCallDelta`, and `LlmCompleted`. A `TtsProvider` receives ordered `SpeechSegment` values and yields `TtsAudioChunk` values. No vendor-specific event names or credentials appear in the runtime.

## Public contracts

```python
from composable_voice_agent import AgentRuntime, ToolRegistry, VoiceRuntime
from composable_voice_agent.contracts import (
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

### OpenAI-compatible LLM

The optional adapter supports the OpenAI Chat Completions streaming shape and works with OpenAI-compatible deployments. It uses `httpx`, so it does not force the OpenAI SDK into the core package:

```python
import os

from composable_voice_agent.providers import OpenAICompatibleConfig, OpenAICompatibleLlm

llm = OpenAICompatibleLlm(
    OpenAICompatibleConfig(
        base_url="https://api.example.com/v1",
        api_key=os.environ["LLM_API_KEY"],
        model="my-model",
    )
)
```

The adapter normalizes text deltas, fragmented tool-call arguments, usage, HTTP failures, timeouts, and malformed SSE events. Tests use an in-memory HTTP transport and never contact a real provider.

### WebSocket transport

The WebSocket adapter accepts one authenticated session per connection. JSON control frames use a versioned envelope; audio is sent as binary frames rather than base64:

```text
client: session.start  → server: session.ready
client: audio.start
client: binary PCM chunks
client: audio.end
server: transcript.completed
server: agent.event
server: audio.started
server: binary PCM chunks
server: audio.completed
server: session.completed
```

Authentication remains an application concern:

```python
from composable_voice_agent.transport import WebSocketVoiceServer

server = WebSocketVoiceServer(runtime=voice_runtime, authenticator=my_authenticator)
# Pass `server.serve` to the WebSocket framework's connection handler.
```

`AudioLimits` provides bounded chunk size, stream size, and queue size. Runtime and authentication failures sent over the wire use safe public messages; provider-specific errors stay inside the adapter.

### Local reference providers

The package includes deterministic providers that require no API keys. They are intended for tests and local protocol exploration, not production speech recognition or synthesis:

```python
from composable_voice_agent.providers import FakeAsr, FakeLlm, FakeTts, TextResponse

voice_runtime = VoiceRuntime(
    asr=FakeAsr("hello"),
    agent=AgentRuntime(FakeLlm([TextResponse("ready")]), ToolRegistry()),
    tts=FakeTts(),
)
```

A minimal FastAPI application is available in `examples/reference_server.py`:

```bash
uv sync --extra dev --extra server
uv run uvicorn examples.reference_server:app --reload
```

The reference server uses an allow-all authenticator for local development only. Replace it before exposing a deployment.

## Design principles

- Business-neutral: no domain models or built-in tools.
- Provider-neutral: ASR, LLM, and TTS are injectable ports.
- Host-controlled identity: authentication and authorization stay outside the model.
- Streaming first: providers use async iterators for incremental results.
- Safe failures: tool and provider failures are represented without leaking internal details.
- Transport-independent core: WebSocket is an adapter, not a requirement of the Agent runtime.

## Development

```bash
uv sync --extra dev --extra openai-compatible --extra server
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The repository contains contracts, a tested Agent/Voice runtime, deterministic reference providers, an OpenAI-compatible LLM adapter, and a provider-neutral WebSocket transport adapter. Other real provider implementations remain separate integration concerns.

## Scope

This project deliberately does not provide a business scenario. Application authors define their own tools, system prompt, identity model, authorization policy, provider adapters, and transport authentication.

## Protocol reference

Every JSON control message has this shape:

```json
{
  "version": 1,
  "type": "audio.start",
  "request_id": "optional-request-id",
  "payload": {}
}
```

See `composable_voice_agent.transport.websocket.protocol` for message constants and parsing helpers.

## License

Apache-2.0
    