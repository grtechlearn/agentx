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


class LLMRegistry:
    """
    Dynamic LLM provider and model registry.

    Register custom providers and model aliases at runtime.
    Projects can add their own LLM backends without modifying AgentX source.

    Usage:
        # Register a custom provider
        registry = LLMRegistry.default()
        registry.register_provider("my-llm", MyCustomProvider)

        # Register model aliases (shortcuts)
        registry.register_model("fast", provider="groq", model="llama-3.3-70b-versatile")
        registry.register_model("local", provider="ollama", model="llama3.2")
        registry.register_model("cheap", provider="anthropic", model="claude-haiku-4-5-20251001")
        registry.register_model("smart", provider="anthropic", model="claude-opus-4-6")

        # Use aliases
        llm = registry.create("fast")         # -> GroqProvider with llama-3.3-70b
        llm = registry.create("local")        # -> OllamaProvider with llama3.2
        llm = registry.create("cheap")        # -> AnthropicProvider with haiku

        # Register an OpenAI-compatible endpoint
        registry.register_openai_compatible(
            name="together",
            base_url="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )
        llm = registry.create("together")

        # List everything
        registry.list_providers()   # ['anthropic', 'openai', 'ollama', 'groq', 'gemini', ...]
        registry.list_models()      # ['fast', 'local', 'cheap', 'smart', ...]
    """

    def __init__(self):
        self._providers: dict[str, type[BaseLLMProvider]] = {}
        self._model_aliases: dict[str, dict[str, Any]] = {}
        self._openai_compatible: dict[str, dict[str, str]] = {}

    @classmethod
    def default(cls) -> LLMRegistry:
        """Create a registry with all built-in providers."""
        from .providers import OllamaProvider, GroqProvider, GeminiProvider

        reg = cls()
        # Built-in providers
        reg._providers = {
            "anthropic": AnthropicProvider,
            "claude": AnthropicProvider,
            "openai": OpenAIProvider,
            "gpt": OpenAIProvider,
            "ollama": OllamaProvider,
            "groq": GroqProvider,
            "gemini": GeminiProvider,
            "google": GeminiProvider,
        }
        # Built-in model aliases
        reg._model_aliases = {
            "opus": {"provider": "anthropic", "model": "claude-opus-4-6"},
            "sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            "haiku": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            "gpt4o": {"provider": "openai", "model": "gpt-4o"},
            "gpt4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
            "llama3": {"provider": "ollama", "model": "llama3.2"},
            "mistral": {"provider": "ollama", "model": "mistral"},
            "deepseek": {"provider": "ollama", "model": "deepseek-r1"},
            "qwen": {"provider": "ollama", "model": "qwen2.5"},
            "groq-llama": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            "groq-mixtral": {"provider": "groq", "model": "mixtral-8x7b-32768"},
            "gemini-flash": {"provider": "gemini", "model": "gemini-2.0-flash"},
            "gemini-pro": {"provider": "gemini", "model": "gemini-1.5-pro"},
        }
        return reg

    # --- Provider Registration ---

    def register_provider(
        self,
        name: str,
        provider_class: type[BaseLLMProvider],
    ) -> None:
        """Register a custom LLM provider class."""
        self._providers[name.lower()] = provider_class
        logger.info(f"Registered LLM provider: {name}")

    def register_openai_compatible(
        self,
        name: str,
        base_url: str,
        api_key_env: str = "",
        api_key: str = "",
        default_model: str = "",
    ) -> None:
        """
        Register any OpenAI-compatible API endpoint as a provider.

        Works with: Together AI, Fireworks, Anyscale, vLLM, LiteLLM,
        Azure OpenAI, DeepInfra, Perplexity, and any OpenAI-compatible server.
        """
        import os
        self._openai_compatible[name.lower()] = {
            "base_url": base_url,
            "api_key": api_key or os.environ.get(api_key_env, ""),
            "default_model": default_model,
        }
        # Also register as a provider alias
        self._providers[name.lower()] = OpenAIProvider
        logger.info(f"Registered OpenAI-compatible provider: {name} -> {base_url}")

    # --- Model Aliases ---

    def register_model(
        self,
        alias: str,
        provider: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str = "",
        base_url: str = "",
        **extra: Any,
    ) -> None:
        """
        Register a model alias (shortcut name).

        Usage:
            registry.register_model("fast", provider="groq", model="llama-3.3-70b-versatile")
            llm = registry.create("fast")
        """
        alias_config: dict[str, Any] = {"provider": provider, "model": model}
        if temperature is not None:
            alias_config["temperature"] = temperature
        if max_tokens is not None:
            alias_config["max_tokens"] = max_tokens
        if api_key:
            alias_config["api_key"] = api_key
        if base_url:
            alias_config["base_url"] = base_url
        alias_config.update(extra)
        self._model_aliases[alias.lower()] = alias_config
        logger.info(f"Registered model alias: {alias} -> {provider}:{model}")

    def unregister_model(self, alias: str) -> bool:
        """Remove a model alias."""
        return self._model_aliases.pop(alias.lower(), None) is not None

    # --- Create LLM ---

    def create(
        self,
        provider_or_alias: str = "anthropic",
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Can be called with:
        - Provider name: create("anthropic", model="claude-sonnet-4-6")
        - Model alias: create("fast") or create("haiku")
        - OpenAI-compatible name: create("together", model="...")
        """
        key = provider_or_alias.lower()

        # 1. Check model aliases first
        if key in self._model_aliases:
            alias_cfg = self._model_aliases[key]
            return self.create(
                provider_or_alias=alias_cfg["provider"],
                model=alias_cfg.get("model", model),
                api_key=alias_cfg.get("api_key", api_key),
                base_url=alias_cfg.get("base_url", base_url),
                temperature=alias_cfg.get("temperature", temperature),
                max_tokens=alias_cfg.get("max_tokens", max_tokens),
            )

        # 2. Check OpenAI-compatible endpoints
        if key in self._openai_compatible:
            compat = self._openai_compatible[key]
            config = LLMConfig(
                provider=key,
                model=model or compat.get("default_model", ""),
                api_key=api_key or compat.get("api_key", ""),
                base_url=base_url or compat.get("base_url", ""),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return OpenAIProvider(config)

        # 3. Check registered providers
        provider_cls = self._providers.get(key)
        if not provider_cls:
            raise ValueError(
                f"Unknown provider or alias: '{provider_or_alias}'. "
                f"Available providers: {self.list_providers()}. "
                f"Available aliases: {self.list_models()}"
            )

        config = LLMConfig(
            provider=key,
            model=model or "claude-sonnet-4-6",
            api_key=api_key,
            base_url=base_url or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return provider_cls(config)

    def create_from_config(self, config: LLMConfig) -> BaseLLMProvider:
        """Create an LLM from an LLMConfig object."""
        return self.create(
            provider_or_alias=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or "",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    # --- Info ---

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return sorted(set(self._providers.keys()))

    def list_models(self) -> list[dict[str, str]]:
        """List all registered model aliases with their provider and model."""
        return [
            {"alias": alias, "provider": cfg["provider"], "model": cfg["model"]}
            for alias, cfg in sorted(self._model_aliases.items())
        ]

    def list_openai_compatible(self) -> list[dict[str, str]]:
        """List all OpenAI-compatible endpoints."""
        return [
            {"name": name, "base_url": cfg["base_url"], "default_model": cfg.get("default_model", "")}
            for name, cfg in sorted(self._openai_compatible.items())
        ]

    def get_model_config(self, alias: str) -> dict[str, Any] | None:
        """Get the config for a model alias."""
        return self._model_aliases.get(alias.lower())

    def summary(self) -> dict[str, Any]:
        """Full registry summary."""
        return {
            "providers": self.list_providers(),
            "model_aliases": self.list_models(),
            "openai_compatible": self.list_openai_compatible(),
            "total_providers": len(set(self._providers.values())),
            "total_aliases": len(self._model_aliases),
        }


# Global default registry (singleton)
_default_registry: LLMRegistry | None = None


def get_registry() -> LLMRegistry:
    """Get the global LLM registry (creates on first call)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = LLMRegistry.default()
    return _default_registry


def register_provider(name: str, provider_class: type[BaseLLMProvider]) -> None:
    """Register a custom LLM provider globally."""
    get_registry().register_provider(name, provider_class)


def register_model(alias: str, provider: str, model: str, **kwargs: Any) -> None:
    """Register a model alias globally."""
    get_registry().register_model(alias, provider, model, **kwargs)


def register_openai_compatible(name: str, base_url: str, **kwargs: Any) -> None:
    """Register an OpenAI-compatible endpoint globally."""
    get_registry().register_openai_compatible(name, base_url, **kwargs)


def create_llm(config: LLMConfig | None = None, **kwargs: Any) -> BaseLLMProvider:
    """
    Factory to create LLM provider.

    Supports:
    - Provider names: "anthropic", "openai", "ollama", "groq", "gemini"
    - Model aliases: "haiku", "sonnet", "opus", "llama3", "gemini-flash"
    - Custom registered providers and aliases
    - OpenAI-compatible endpoints

    Usage:
        # By provider
        llm = create_llm(provider="anthropic", model="claude-sonnet-4-6")

        # By alias
        llm = create_llm(provider="haiku")
        llm = create_llm(provider="groq-llama")

        # From config
        llm = create_llm(LLMConfig(provider="ollama", model="llama3.2"))
    """
    registry = get_registry()

    if config is not None:
        return registry.create_from_config(config)

    provider = kwargs.pop("provider", "anthropic")
    return registry.create(provider_or_alias=provider, **kwargs)
