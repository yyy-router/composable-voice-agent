"""Tool contracts and registry."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .identity import ToolContext


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must be non-empty")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be an object schema")


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    success: bool = True
    data: Mapping[str, object] | None = None
    error_code: str | None = None


class AgentTool(Protocol):
    @property
    def definition(self) -> ToolDefinition:
        ...

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        ...


class ToolRegistry:
    def __init__(self, tools: Sequence[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"duplicate tool name: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())
