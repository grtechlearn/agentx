"""
AgentX - Tool system for agents.
Tools give agents the ability to interact with external systems.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result returned by a tool execution."""

    success: bool = True
    data: Any = None
    error: str | None = None

    @classmethod
    def ok(cls, data: Any = None) -> ToolResult:
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(success=False, error=error)


class BaseTool(ABC):
    """Base class for all tools available to agents."""

    name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__.lower().replace("tool", "")

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""

    def get_schema(self) -> dict[str, Any]:
        """Return JSON schema for this tool's parameters."""
        sig = inspect.signature(self.execute)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs"):
                continue
            prop: dict[str, str] = {"type": "string"}
            if param.annotation != inspect.Parameter.empty:
                type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
                prop["type"] = type_map.get(param.annotation, "string")
            properties[param_name] = prop
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }


class FunctionTool(BaseTool):
    """Wrap a plain function as a tool."""

    def __init__(self, fn: Callable[..., Any], name: str = "", description: str = ""):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__ or ""

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = self.fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(str(e))


def tool(name: str = "", description: str = "") -> Callable[..., Any]:
    """Decorator to convert a function into a Tool."""
    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(fn, name=name or fn.__name__, description=description or fn.__doc__ or "")
    return decorator
