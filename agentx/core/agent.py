"""
AgentX - Base Agent class.
All agents inherit from this. Lightweight, no magic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .context import AgentContext
from .llm import BaseLLMProvider, LLMResponse, StreamChunk, create_llm, LLMConfig
from .message import AgentMessage, MessageType
from .tool import BaseTool, ToolResult

logger = logging.getLogger("agentx")


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str
    role: str = ""
    system_prompt: str = ""
    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 2
    tools_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Runtime state of an agent."""

    status: str = "idle"  # idle, running, waiting, error, completed
    current_task: str = ""
    messages_processed: int = 0
    errors: list[str] = Field(default_factory=list)
    results: list[Any] = Field(default_factory=list)


class BaseAgent(ABC):
    """
    Base agent class. Every agent in AgentX extends this.

    Features:
    - LLM integration (Claude, OpenAI, custom)
    - Tool execution
    - Inter-agent messaging
    - Shared context
    - Lifecycle hooks (before/after run)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: BaseLLMProvider | None = None,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ):
        if config is None:
            name = kwargs.pop("name", self.__class__.__name__)
            config = AgentConfig(name=name, **kwargs)
        self.config = config
        self.state = AgentState()
        self.tools: dict[str, BaseTool] = {}
        self._llm = llm

        if tools:
            for t in tools:
                self.register_tool(t)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def llm(self) -> BaseLLMProvider:
        if self._llm is None:
            self._llm = create_llm(LLMConfig(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ))
        return self._llm

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [t.get_schema() for t in self.tools.values()]

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult.fail(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' error: {e}")
            return ToolResult.fail(str(e))

    async def think(
        self,
        prompt: str,
        context: AgentContext | None = None,
        system: str | None = None,
        use_tools: bool = False,
    ) -> LLMResponse:
        """Send a prompt to the LLM and get a response."""
        messages: list[dict[str, str]] = []
        if context:
            messages.extend(context.conversation_history)
        messages.append({"role": "user", "content": prompt})

        sys_prompt = system or self.config.system_prompt

        kwargs: dict[str, Any] = {
            "messages": messages,
            "system": sys_prompt,
        }
        if use_tools and self.tools:
            kwargs["tools"] = self._get_anthropic_tools()

        response = await self.llm.generate(**kwargs)

        # Handle tool calls if any
        if response.tool_calls and self.config.tools_enabled:
            response = await self._handle_tool_calls(response, messages, sys_prompt)

        return response

    async def think_stream(
        self,
        prompt: str,
        context: AgentContext | None = None,
        system: str | None = None,
    ):
        """Stream a response from the LLM token by token.

        Usage:
            async for chunk in agent.think_stream("Hello"):
                print(chunk.content, end="")
        """
        messages: list[dict[str, str]] = []
        if context:
            messages.extend(context.conversation_history)
        messages.append({"role": "user", "content": prompt})
        sys_prompt = system or self.config.system_prompt

        async for chunk in self.llm.stream(messages=messages, system=sys_prompt):
            yield chunk

    async def think_with_callback(
        self,
        prompt: str,
        context: AgentContext | None = None,
        system: str | None = None,
        on_chunk: Any = None,
    ) -> LLMResponse:
        """Stream with a callback per token, returns final LLMResponse.

        Usage:
            response = await agent.think_with_callback(
                "Hello", on_chunk=lambda c: print(c.content, end="")
            )
        """
        messages: list[dict[str, str]] = []
        if context:
            messages.extend(context.conversation_history)
        messages.append({"role": "user", "content": prompt})
        sys_prompt = system or self.config.system_prompt

        return await self.llm.generate_or_stream(
            messages=messages, system=sys_prompt,
            stream=True, on_chunk=on_chunk,
        )

    async def think_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        context: AgentContext | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Get a structured JSON response from the LLM."""
        messages: list[dict[str, str]] = []
        if context:
            messages.extend(context.conversation_history)
        messages.append({"role": "user", "content": prompt})
        sys_prompt = system or self.config.system_prompt
        return await self.llm.generate_json(messages=messages, system=sys_prompt, schema=schema)

    def _get_anthropic_tools(self) -> list[dict[str, Any]]:
        """Convert tools to Anthropic tool format."""
        result = []
        for t in self.tools.values():
            schema = t.get_schema()
            result.append({
                "name": schema["name"],
                "description": schema["description"],
                "input_schema": schema["parameters"],
            })
        return result

    async def _handle_tool_calls(
        self,
        response: LLMResponse,
        messages: list[dict[str, str]],
        system: str,
    ) -> LLMResponse:
        """Process tool calls and get final response."""
        tool_results = []
        for tc in response.tool_calls:
            result = await self.execute_tool(tc["name"], **tc["input"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": str(result.data) if result.success else f"Error: {result.error}",
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": str(tool_results)})

        return await self.llm.generate(messages=messages, system=system)

    # --- Lifecycle hooks ---

    async def on_start(self, context: AgentContext) -> None:
        """Called before the agent starts processing."""

    async def on_complete(self, context: AgentContext, result: Any) -> None:
        """Called after the agent completes processing."""

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Called when an error occurs during processing."""
        logger.error(f"Agent '{self.name}' error: {error}")
        self.state.errors.append(str(error))

    # --- Main execution ---

    @abstractmethod
    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        """
        Process a message and return a response.
        This is the main method each agent must implement.
        """

    async def run(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        """Execute the agent with lifecycle hooks."""
        self.state.status = "running"
        self.state.current_task = message.content[:100]

        try:
            await self.on_start(context)
            result = await self.process(message, context)
            self.state.status = "completed"
            self.state.messages_processed += 1
            context.store_result(self.name, result.data)
            await self.on_complete(context, result)
            return result
        except Exception as e:
            self.state.status = "error"
            await self.on_error(context, e)
            return message.error(f"Agent '{self.name}' failed: {str(e)}")


class SimpleAgent(BaseAgent):
    """
    A simple agent that just sends prompts to the LLM.
    Good for quick prototyping or simple tasks.
    """

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=message.content,
            context=context,
            use_tools=bool(self.tools),
        )
        return message.reply(content=response.content, data={"usage": response.usage})
