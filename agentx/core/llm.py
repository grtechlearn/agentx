"""
AgentX - LLM Provider abstraction.
Supports Claude (Anthropic), OpenAI, and custom providers.
Includes streaming support for real-time token delivery.

Usage:
    # Standard generation
    response = await llm.generate(messages=[...])

    # Streaming generation
    async for chunk in llm.stream(messages=[...]):
        print(chunk.content, end="", flush=True)

    # JSON generation
    data = await llm.generate_json(messages=[...], schema={...})
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


class LLMResponse(BaseModel):
    """Standardized LLM response."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    raw: Any = None


class StreamChunk(BaseModel):
    """A single chunk from a streaming response."""

    content: str = ""             # Text delta for this chunk
    done: bool = False            # True when stream is complete
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    accumulated: str = ""         # Full text so far (running total)


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""

    @abstractmethod
    async def generate_json(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON response."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response token by token."""
        yield StreamChunk()  # pragma: no cover

    async def generate_or_stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        on_chunk: Any = None,
    ) -> LLMResponse:
        """
        Unified method: generate or stream based on flag.
        If stream=True and on_chunk callback provided, streams and calls
        on_chunk(chunk: StreamChunk) for each token, then returns final LLMResponse.
        """
        if not stream:
            return await self.generate(
                messages=messages, system=system, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )

        # Stream and collect
        full_content = ""
        final_usage: dict[str, int] = {}
        final_model = ""
        final_tool_calls: list[dict[str, Any]] = []

        async for chunk in self.stream(
            messages=messages, system=system, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        ):
            full_content = chunk.accumulated
            if chunk.usage:
                final_usage = chunk.usage
            if chunk.model:
                final_model = chunk.model
            if chunk.tool_calls:
                final_tool_calls = chunk.tool_calls

            if on_chunk:
                result = on_chunk(chunk)
                if asyncio.iscoroutine(result):
                    await result

        return LLMResponse(
            content=full_content,
            tool_calls=final_tool_calls,
            usage=final_usage,
            model=final_model,
        )


class AnthropicProvider(BaseLLMProvider):
    """Claude (Anthropic) LLM provider with streaming support."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build common kwargs for Anthropic API calls."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return kwargs

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(messages, system, tools, temperature, max_tokens)
        response = await client.messages.create(**kwargs)

        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            model=response.model,
            raw=response,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from Claude token by token."""
        client = self._get_client()
        kwargs = self._build_kwargs(messages, system, tools, temperature, max_tokens)

        accumulated = ""
        model_name = ""
        usage_data: dict[str, int] = {}
        tool_calls: list[dict[str, Any]] = []
        current_tool: dict[str, Any] = {}
        current_tool_json = ""

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                # Handle different event types
                if hasattr(event, 'type'):
                    event_type = event.type

                    if event_type == "message_start":
                        if hasattr(event, 'message'):
                            model_name = getattr(event.message, 'model', '')
                            msg_usage = getattr(event.message, 'usage', None)
                            if msg_usage:
                                usage_data["input_tokens"] = getattr(msg_usage, 'input_tokens', 0)

                    elif event_type == "content_block_start":
                        block = getattr(event, 'content_block', None)
                        if block and getattr(block, 'type', '') == 'tool_use':
                            current_tool = {
                                "id": getattr(block, 'id', ''),
                                "name": getattr(block, 'name', ''),
                                "input": {},
                            }
                            current_tool_json = ""

                    elif event_type == "content_block_delta":
                        delta = getattr(event, 'delta', None)
                        if delta:
                            delta_type = getattr(delta, 'type', '')
                            if delta_type == "text_delta":
                                text = getattr(delta, 'text', '')
                                accumulated += text
                                yield StreamChunk(
                                    content=text,
                                    accumulated=accumulated,
                                    model=model_name,
                                )
                            elif delta_type == "input_json_delta":
                                current_tool_json += getattr(delta, 'partial_json', '')

                    elif event_type == "content_block_stop":
                        if current_tool:
                            try:
                                current_tool["input"] = json.loads(current_tool_json) if current_tool_json else {}
                            except json.JSONDecodeError:
                                current_tool["input"] = {}
                            tool_calls.append(current_tool)
                            current_tool = {}
                            current_tool_json = ""

                    elif event_type == "message_delta":
                        delta = getattr(event, 'delta', None)
                        msg_usage = getattr(event, 'usage', None)
                        if msg_usage:
                            usage_data["output_tokens"] = getattr(msg_usage, 'output_tokens', 0)

                # For simple text events (some SDK versions)
                elif isinstance(event, str):
                    accumulated += event
                    yield StreamChunk(
                        content=event,
                        accumulated=accumulated,
                        model=model_name,
                    )

        # Final chunk with done=True
        yield StreamChunk(
            content="",
            done=True,
            accumulated=accumulated,
            usage=usage_data,
            model=model_name,
            tool_calls=tool_calls,
        )

    async def generate_json(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_instruction = "Respond ONLY with valid JSON. No markdown, no explanation, no code fences."
        if schema:
            json_instruction += f"\n\nExpected JSON schema:\n{json.dumps(schema, indent=2)}"

        full_system = f"{system}\n\n{json_instruction}" if system else json_instruction
        response = await self.generate(messages=messages, system=full_system, temperature=0.1)

        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider with streaming support."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    def _build_messages(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> list[dict[str, str]]:
        """Build message list with system prompt."""
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        return all_messages

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        all_messages = self._build_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": all_messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            model=response.model,
            raw=response,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from OpenAI token by token."""
        client = self._get_client()
        all_messages = self._build_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": all_messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        accumulated = ""
        model_name = ""
        usage_data: dict[str, int] = {}

        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            if chunk.model:
                model_name = chunk.model

            # Usage comes in the final chunk
            if chunk.usage:
                usage_data = {
                    "input_tokens": chunk.usage.prompt_tokens or 0,
                    "output_tokens": chunk.usage.completion_tokens or 0,
                }

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    accumulated += delta.content
                    yield StreamChunk(
                        content=delta.content,
                        accumulated=accumulated,
                        model=model_name,
                    )

                # Check for finish
                if chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        content="",
                        done=True,
                        accumulated=accumulated,
                        usage=usage_data,
                        model=model_name,
                    )

    async def generate_json(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_instruction = "Respond ONLY with valid JSON."
        if schema:
            json_instruction += f"\n\nExpected schema:\n{json.dumps(schema, indent=2)}"
        full_system = f"{system}\n\n{json_instruction}" if system else json_instruction
        response = await self.generate(messages=messages, system=full_system, temperature=0.1)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)


def create_llm(config: LLMConfig | None = None, **kwargs: Any) -> BaseLLMProvider:
    """Factory to create LLM provider."""
    if config is None:
        config = LLMConfig(**kwargs)
    providers = {
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "openai": OpenAIProvider,
        "gpt": OpenAIProvider,
    }
    provider_cls = providers.get(config.provider.lower())
    if not provider_cls:
        raise ValueError(f"Unknown provider: {config.provider}. Supported: {list(providers.keys())}")
    return provider_cls(config)
