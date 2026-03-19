"""
Tests for OllamaProvider, GroqProvider, and GeminiProvider.
All HTTP/SDK calls are mocked -- no real network or API keys needed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentx.core.llm import LLMConfig, LLMResponse, StreamChunk, create_llm
from agentx.core.providers import OllamaProvider, GroqProvider, GeminiProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(provider: str, **kw: Any) -> LLMConfig:
    return LLMConfig(provider=provider, api_key="test-key", **kw)


# ---------------------------------------------------------------------------
# create_llm factory tests
# ---------------------------------------------------------------------------

class TestCreateLLMFactory:
    def test_ollama(self):
        llm = create_llm(_cfg("ollama", model="llama3"))
        assert isinstance(llm, OllamaProvider)

    def test_groq(self):
        llm = create_llm(_cfg("groq"))
        assert isinstance(llm, GroqProvider)

    def test_gemini(self):
        llm = create_llm(_cfg("gemini"))
        assert isinstance(llm, GeminiProvider)

    def test_google_alias(self):
        llm = create_llm(_cfg("google"))
        assert isinstance(llm, GeminiProvider)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm(_cfg("nonexistent"))


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def _provider(self, model: str = "llama3") -> OllamaProvider:
        return OllamaProvider(LLMConfig(provider="ollama", model=model))

    @pytest.mark.asyncio
    async def test_generate(self):
        provider = self._provider()
        mock_response = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "Hello from Ollama!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch("agentx.core.providers._http_post", new_callable=AsyncMock, return_value=mock_response):
            resp = await provider.generate(messages=[{"role": "user", "content": "Hi"}])

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello from Ollama!"
        assert resp.model == "llama3"
        assert resp.usage["input_tokens"] == 10
        assert resp.usage["output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_stream(self):
        provider = self._provider()

        # Simulate NDJSON stream lines
        chunks_data = [
            {"model": "llama3", "message": {"content": "Hello"}, "done": False},
            {"model": "llama3", "message": {"content": " world"}, "done": False},
            {
                "model": "llama3",
                "message": {"content": "!"},
                "done": True,
                "prompt_eval_count": 8,
                "eval_count": 3,
            },
        ]

        async def fake_stream_iter() -> AsyncIterator[bytes]:
            for c in chunks_data:
                yield json.dumps(c).encode()

        with patch("agentx.core.providers._http_post", new_callable=AsyncMock, return_value=fake_stream_iter()):
            collected: list[StreamChunk] = []
            async for chunk in provider.stream(messages=[{"role": "user", "content": "Hi"}]):
                collected.append(chunk)

        assert len(collected) == 3
        assert collected[0].content == "Hello"
        assert collected[1].content == " world"
        assert collected[2].done is True
        assert collected[2].accumulated == "Hello world!"
        assert collected[2].usage["output_tokens"] == 3

    @pytest.mark.asyncio
    async def test_generate_json(self):
        provider = self._provider()
        mock_response = {
            "model": "llama3",
            "message": {"role": "assistant", "content": '{"name": "test", "value": 42}'},
        }

        with patch("agentx.core.providers._http_post", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate_json(
                messages=[{"role": "user", "content": "Give me JSON"}],
                schema={"type": "object"},
            )

        assert result == {"name": "test", "value": 42}

    def test_default_base_url(self):
        provider = self._provider()
        assert provider._base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        cfg = LLMConfig(provider="ollama", model="llama3", base_url="http://myhost:1234")
        provider = OllamaProvider(cfg)
        assert provider._base_url == "http://myhost:1234"


# ---------------------------------------------------------------------------
# GroqProvider
# ---------------------------------------------------------------------------

class TestGroqProvider:
    def _provider(self) -> GroqProvider:
        return GroqProvider(LLMConfig(provider="groq", api_key="groq-test-key"))

    def test_default_model(self):
        provider = self._provider()
        assert provider.config.model == "llama-3.3-70b-versatile"

    def test_default_base_url(self):
        provider = self._provider()
        assert provider.config.base_url == "https://api.groq.com/openai/v1"

    def test_custom_model_preserved(self):
        cfg = LLMConfig(provider="groq", api_key="k", model="mixtral-8x7b-32768")
        provider = GroqProvider(cfg)
        assert provider.config.model == "mixtral-8x7b-32768"

    @pytest.mark.asyncio
    async def test_generate(self):
        provider = self._provider()

        # Mock OpenAI SDK response
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Groq!"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 12
        mock_usage.completion_tokens = 6
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "llama-3.3-70b-versatile"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            resp = await provider.generate(messages=[{"role": "user", "content": "Hi"}])

        assert resp.content == "Hello from Groq!"
        assert resp.usage["input_tokens"] == 12
        assert resp.usage["output_tokens"] == 6

    @pytest.mark.asyncio
    async def test_stream(self):
        provider = self._provider()

        # Build mock stream chunks
        def _make_chunk(content: str | None, finish: str | None = None, model: str = "llama-3.3-70b-versatile"):
            c = MagicMock()
            c.model = model
            c.usage = None
            if content is not None:
                delta = MagicMock()
                delta.content = content
                choice = MagicMock()
                choice.delta = delta
                choice.finish_reason = finish
                c.choices = [choice]
            else:
                c.choices = []
            return c

        async def fake_stream():
            yield _make_chunk("Hello")
            yield _make_chunk(" Groq")
            final = _make_chunk("", finish="stop")
            final.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
            yield final

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream())

        with patch.object(provider, "_get_client", return_value=mock_client):
            collected: list[StreamChunk] = []
            async for chunk in provider.stream(messages=[{"role": "user", "content": "Hi"}]):
                collected.append(chunk)

        assert any(c.content == "Hello" for c in collected)
        assert any(c.done for c in collected)

    @pytest.mark.asyncio
    async def test_generate_json(self):
        provider = self._provider()

        mock_choice = MagicMock()
        mock_choice.message.content = '{"answer": 42}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)
        mock_response.model = "llama-3.3-70b-versatile"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.generate_json(
                messages=[{"role": "user", "content": "JSON please"}],
            )

        assert result == {"answer": 42}


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    def _provider(self) -> GeminiProvider:
        return GeminiProvider(LLMConfig(provider="gemini", api_key="gemini-test-key"))

    def test_default_model(self):
        provider = self._provider()
        assert provider.config.model == "gemini-2.0-flash"

    def test_default_base_url(self):
        provider = self._provider()
        assert provider.config.base_url == "https://generativelanguage.googleapis.com/v1beta"

    def test_custom_model_preserved(self):
        cfg = LLMConfig(provider="gemini", api_key="k", model="gemini-1.5-pro")
        provider = GeminiProvider(cfg)
        assert provider.config.model == "gemini-1.5-pro"

    @pytest.mark.asyncio
    async def test_generate(self):
        provider = self._provider()
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello from Gemini!"}],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 7,
            },
        }

        with patch("agentx.core.providers._http_post", new_callable=AsyncMock, return_value=mock_response):
            resp = await provider.generate(messages=[{"role": "user", "content": "Hi"}])

        assert resp.content == "Hello from Gemini!"
        assert resp.model == "gemini-2.0-flash"
        assert resp.usage["input_tokens"] == 15
        assert resp.usage["output_tokens"] == 7

    @pytest.mark.asyncio
    async def test_stream(self):
        provider = self._provider()

        # Simulate SSE lines
        sse_chunks = [
            {
                "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1},
            },
            {
                "candidates": [{"content": {"parts": [{"text": " Gemini"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3},
            },
        ]

        async def fake_sse_stream() -> AsyncIterator[bytes]:
            for chunk in sse_chunks:
                yield f"data: {json.dumps(chunk)}".encode()

        with patch("agentx.core.providers._http_post", new_callable=AsyncMock, return_value=fake_sse_stream()):
            collected: list[StreamChunk] = []
            async for chunk in provider.stream(messages=[{"role": "user", "content": "Hi"}]):
                collected.append(chunk)

        # 2 content chunks + 1 final done chunk
        assert len(collected) == 3
        assert collected[0].content == "Hello"
        assert collected[1].content == " Gemini"
        assert collected[2].done is True
        assert collected[2].accumulated == "Hello Gemini"

    @pytest.mark.asyncio
    async def test_generate_json(self):
        provider = self._provider()
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"status": "ok"}'}],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }

        with patch("agentx.core.providers._http_post", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate_json(
                messages=[{"role": "user", "content": "JSON please"}],
                schema={"type": "object"},
            )

        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_generate_with_system(self):
        provider = self._provider()
        mock_response = {
            "candidates": [{"content": {"parts": [{"text": "System works"}]}}],
            "usageMetadata": {},
        }

        with patch("agentx.core.providers._http_post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await provider.generate(
                messages=[{"role": "user", "content": "Hi"}],
                system="You are helpful",
            )

            # Verify systemInstruction was included in payload
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json_body") or call_kwargs[1].get("json_body")
            assert "systemInstruction" in payload
            assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_ollama_uses_provided_model(self):
        cfg = LLMConfig(provider="ollama", model="mistral")
        p = OllamaProvider(cfg)
        assert p.config.model == "mistral"

    def test_groq_env_key(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-groq-key"}):
            cfg = LLMConfig(provider="groq")
            p = GroqProvider(cfg)
            assert p.config.api_key == "env-groq-key"

    def test_gemini_env_key(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-google-key"}):
            cfg = LLMConfig(provider="gemini")
            p = GeminiProvider(cfg)
            assert p.config.api_key == "env-google-key"
