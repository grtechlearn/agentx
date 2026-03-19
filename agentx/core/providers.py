"""
AgentX - Additional LLM Providers.

Provides OllamaProvider (local inference), GroqProvider (Groq cloud),
and GeminiProvider (Google Gemini) without requiring extra SDK dependencies.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

from .llm import BaseLLMProvider, LLMConfig, LLMResponse, StreamChunk

logger = logging.getLogger("agentx")


# ---------------------------------------------------------------------------
# HTTP helper: prefer aiohttp, fall back to httpx, then urllib
# ---------------------------------------------------------------------------

async def _http_post(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    stream: bool = False,
) -> Any:
    """
    Perform an async HTTP POST.
    Returns the parsed JSON response (non-stream) or an async iterator of raw
    bytes lines (stream).
    """
    headers = headers or {}
    data = json.dumps(json_body).encode() if json_body else b""
    headers.setdefault("Content-Type", "application/json")

    # --- aiohttp ---
    try:
        import aiohttp  # type: ignore[import-untyped]

        if stream:
            return _aiohttp_stream(url, headers, data)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as resp:
                resp.raise_for_status()
                return await resp.json()
    except ImportError:
        pass

    # --- httpx ---
    try:
        import httpx  # type: ignore[import-untyped]

        if stream:
            return _httpx_stream(url, headers, data)
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, content=data)
            resp.raise_for_status()
            return resp.json()
    except ImportError:
        pass

    # --- urllib (sync fallback, wrapped in asyncio) ---
    import asyncio
    import urllib.request

    def _sync_post() -> Any:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    if stream:
        raise RuntimeError(
            "Streaming requires aiohttp or httpx. Install one: pip install aiohttp"
        )
    return await asyncio.get_event_loop().run_in_executor(None, _sync_post)


async def _aiohttp_stream(
    url: str, headers: dict[str, str], data: bytes
) -> AsyncIterator[bytes]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                if line:
                    yield line


async def _httpx_stream(
    url: str, headers: dict[str, str], data: bytes
) -> AsyncIterator[bytes]:
    import httpx

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, content=data) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield line.encode()


# ===========================================================================
# 1. OllamaProvider
# ===========================================================================

class OllamaProvider(BaseLLMProvider):
    """Local LLM inference via Ollama HTTP API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    def _chat_url(self) -> str:
        return f"{self._base_url}/api/chat"

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float | None = None,
        stream: bool = False,
        json_format: bool = False,
    ) -> dict[str, Any]:
        all_messages: list[dict[str, str]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": all_messages,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if json_format:
            payload["format"] = "json"
        return payload

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(messages, system, temperature, stream=False)
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        data = await _http_post(self._chat_url(), json_body=payload)

        message = data.get("message", {})
        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", self.config.model),
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(messages, system, temperature, stream=True)
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        accumulated = ""
        model_name = self.config.model
        usage_data: dict[str, int] = {}

        line_iter = await _http_post(self._chat_url(), json_body=payload, stream=True)

        async for raw_line in line_iter:
            line_str = raw_line.decode("utf-8").strip() if isinstance(raw_line, bytes) else raw_line.strip()
            if not line_str:
                continue
            try:
                chunk_data = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            model_name = chunk_data.get("model", model_name)
            msg = chunk_data.get("message", {})
            content = msg.get("content", "")

            if chunk_data.get("done", False):
                usage_data = {
                    "input_tokens": chunk_data.get("prompt_eval_count", 0),
                    "output_tokens": chunk_data.get("eval_count", 0),
                }
                yield StreamChunk(
                    content=content,
                    done=True,
                    accumulated=accumulated + content,
                    usage=usage_data,
                    model=model_name,
                )
                return

            accumulated += content
            yield StreamChunk(
                content=content,
                accumulated=accumulated,
                model=model_name,
            )

        # Safety: if loop ends without done=True
        yield StreamChunk(content="", done=True, accumulated=accumulated, model=model_name)

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

        payload = self._build_payload(messages, full_system, stream=False, json_format=True)
        data = await _http_post(self._chat_url(), json_body=payload)

        text = data.get("message", {}).get("content", "").strip()
        return json.loads(text)


# ===========================================================================
# 2. GroqProvider
# ===========================================================================

class GroqProvider(BaseLLMProvider):
    """Groq cloud LLM provider (OpenAI-compatible API, uses openai SDK)."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: LLMConfig):
        if not config.model or config.model in ("claude-sonnet-4-6",):
            config = config.model_copy(update={"model": self.DEFAULT_MODEL})
        if not config.base_url:
            config = config.model_copy(update={"base_url": self.BASE_URL})
        if not config.api_key:
            config = config.model_copy(update={"api_key": os.environ.get("GROQ_API_KEY", "")})
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai  # type: ignore[import-untyped]
            kwargs: dict[str, Any] = {
                "base_url": self.config.base_url,
            }
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    def _build_messages(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> list[dict[str, str]]:
        all_messages: list[dict[str, str]] = []
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


# ===========================================================================
# 3. GeminiProvider
# ===========================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider via HTTP API (no SDK dependency)."""

    DEFAULT_MODEL = "gemini-2.0-flash"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: LLMConfig):
        if not config.model or config.model in ("claude-sonnet-4-6",):
            config = config.model_copy(update={"model": self.DEFAULT_MODEL})
        if not config.base_url:
            config = config.model_copy(update={"base_url": self.BASE_URL})
        if not config.api_key:
            config = config.model_copy(update={"api_key": os.environ.get("GOOGLE_API_KEY", "")})
        super().__init__(config)

    def _generate_url(self) -> str:
        return (
            f"{self.config.base_url}/models/{self.config.model}"
            f":generateContent?key={self.config.api_key}"
        )

    def _stream_url(self) -> str:
        return (
            f"{self.config.base_url}/models/{self.config.model}"
            f":streamGenerateContent?alt=sse&key={self.config.api_key}"
        )

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Convert standard messages to Gemini's contents format."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini uses "user" and "model" roles
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": msg.get("content", "")}],
            })

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "maxOutputTokens": max_tokens or self.config.max_tokens,
                "topP": self.config.top_p,
            },
        }

        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}],
            }

        return payload

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(messages, system, temperature, max_tokens)
        data = await _http_post(self._generate_url(), json_body=payload)

        # Extract text from Gemini response
        content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)

        usage_meta = data.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            model=self.config.model,
            usage={
                "input_tokens": usage_meta.get("promptTokenCount", 0),
                "output_tokens": usage_meta.get("candidatesTokenCount", 0),
            },
            raw=data,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from Gemini via SSE."""
        payload = self._build_payload(messages, system, temperature, max_tokens)
        accumulated = ""
        model_name = self.config.model
        usage_data: dict[str, int] = {}

        line_iter = await _http_post(self._stream_url(), json_body=payload, stream=True)

        async for raw_line in line_iter:
            line_str = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line_str = line_str.strip()

            # SSE format: lines starting with "data: "
            if not line_str.startswith("data: "):
                continue

            json_str = line_str[6:]  # strip "data: "
            if not json_str or json_str == "[DONE]":
                continue

            try:
                chunk_data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            # Extract text
            candidates = chunk_data.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)

            # Extract usage if present
            usage_meta = chunk_data.get("usageMetadata", {})
            if usage_meta:
                usage_data = {
                    "input_tokens": usage_meta.get("promptTokenCount", 0),
                    "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                }

            accumulated += text
            yield StreamChunk(
                content=text,
                accumulated=accumulated,
                model=model_name,
            )

        # Final done chunk
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
