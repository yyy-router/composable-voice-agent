import pytest

from composable_voice_agent.contracts import ToolDefinition, ToolRegistry


def test_registry_rejects_duplicate_names() -> None:
    class Tool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition("same", "", {"type": "object"})

        async def execute(self, arguments, context):
            del arguments, context
            raise AssertionError

    registry = ToolRegistry([Tool()])
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(Tool())


def test_definition_requires_object_schema() -> None:
    with pytest.raises(ValueError, match="object"):
        ToolDefinition("x", "", {"type": "string"})


def test_registry_preserves_definition_order() -> None:
    class Tool:
        def __init__(self, name: str) -> None:
            self._definition = ToolDefinition(name, "", {"type": "object"})

        @property
        def definition(self) -> ToolDefinition:
            return self._definition

        async def execute(self, arguments, context):
            del arguments, context
            raise AssertionError

    registry = ToolRegistry([Tool("first"), Tool("second")])
    assert [item.name for item in registry.definitions()] == ["first", "second"]
