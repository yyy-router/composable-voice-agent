"""Minimal extension point for application-owned tools."""

from collections.abc import Mapping

from composable_voice_agent.contracts import ToolContext, ToolDefinition, ToolResult


class EchoTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Return text supplied by the caller.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        del context
        text = arguments.get("text")
        if not isinstance(text, str):
            return ToolResult("text is required", success=False, error_code="invalid_input")
        return ToolResult(text, data={"text": text})
