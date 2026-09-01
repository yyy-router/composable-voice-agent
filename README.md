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

Implementations of `AsrProvider`, `LlmProvider`, and `TtsProvider` are intentionally not bundled. This keeps the core independent from vendors and lets applications choose their own providers.

## Design principles

- Business-neutral: no domain models or built-in tools.
- Provider-neutral: ASR, LLM, and TTS are injectable ports.
- Host-controlled identity: authentication and authorization stay outside the model.
- Streaming first: providers use async iterators for incremental results.
- Safe failures: tool and provider failures are represented without leaking internal details.
- Transport-independent core: WebSocket is an adapter, not a requirement of the Agent runtime.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy
```

The repository currently contains contracts and a minimal runtime foundation. Provider adapters and a complete WebSocket server will be added only after the public protocol is stable.
