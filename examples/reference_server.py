"""A runnable reference WebSocket application."""

from fastapi import FastAPI, WebSocket

from composable_voice_agent import ToolRegistry
from composable_voice_agent.providers import FakeAsr, FakeLlm, FakeTts, TextResponse
from composable_voice_agent.runtime import AgentRuntime, VoiceRuntime
from composable_voice_agent.transport import AuthRequest, AuthResult
from composable_voice_agent.transport.websocket import WebSocketVoiceServer


class AllowAllAuthenticator:
    """Development-only authenticator; replace it in a real application."""

    async def authenticate(self, request: AuthRequest) -> AuthResult:
        del request
        return AuthResult(accepted=True)


voice_runtime = VoiceRuntime(
    asr=FakeAsr("Hello from the reference server"),
    agent=AgentRuntime(
        llm=FakeLlm([TextResponse("The reference server is ready.")]),
        tools=ToolRegistry(),
    ),
    tts=FakeTts(),
)
transport = WebSocketVoiceServer(voice_runtime, AllowAllAuthenticator())
app = FastAPI(title="Composable Voice Agent reference")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await transport.serve(websocket)  # type: ignore[arg-type]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.reference_server:app", host="127.0.0.1", port=8000, reload=False)


__all__ = ["app"]
