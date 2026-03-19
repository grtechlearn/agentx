"""
Tests for agentx.prompts module.
Covers: PromptTemplate, PromptManager, ContextManager, ResponseCache.
"""

import time

import pytest

from agentx.prompts.manager import (
    PromptTemplate, PromptManager,
    ContextManager, ResponseCache,
)


# ─────────────────────────────────────────────
# PromptTemplate
# ─────────────────────────────────────────────

class TestPromptTemplate:
    def test_render_basic(self):
        t = PromptTemplate(
            name="greet",
            template="Hello, {{name}}! You are a {{role}}.",
            variables=["name", "role"],
        )
        result = t.render(name="Alice", role="developer")
        assert result == "Hello, Alice! You are a developer."

    def test_render_missing_variable_warning(self):
        t = PromptTemplate(
            name="test",
            template="Hello {{name}}, your {{missing}} is ready.",
            variables=["name", "missing"],
        )
        result = t.render(name="Bob")
        assert "Bob" in result
        # Missing variable placeholder should remain
        assert "{{missing}}" in result

    def test_render_no_variables(self):
        t = PromptTemplate(name="static", template="No variables here.")
        result = t.render()
        assert result == "No variables here."

    def test_render_extra_kwargs_ignored(self):
        t = PromptTemplate(name="t", template="Hello {{name}}.", variables=["name"])
        result = t.render(name="Alice", extra="ignored")
        assert result == "Hello Alice."

    def test_template_metadata(self):
        t = PromptTemplate(
            name="t", template="...", version="2.0",
            description="Test template", model_hint="claude-sonnet",
            max_tokens_hint=1000, temperature_hint=0.5,
            tags=["test", "demo"],
            metadata={"author": "test"},
        )
        assert t.version == "2.0"
        assert t.description == "Test template"
        assert t.model_hint == "claude-sonnet"
        assert t.max_tokens_hint == 1000
        assert t.temperature_hint == 0.5
        assert t.tags == ["test", "demo"]
        assert t.metadata["author"] == "test"


# ─────────────────────────────────────────────
# PromptManager
# ─────────────────────────────────────────────

class TestPromptManager:
    def test_register_and_get(self):
        pm = PromptManager()
        t = PromptTemplate(name="greet", template="Hello {{name}}", variables=["name"])
        pm.register(t)
        retrieved = pm.get("greet")
        assert retrieved is not None
        assert retrieved.name == "greet"

    def test_get_by_version(self):
        pm = PromptManager()
        t1 = PromptTemplate(name="greet", template="V1: Hello {{name}}", version="1.0")
        t2 = PromptTemplate(name="greet", template="V2: Hello {{name}}", version="2.0")
        pm.register(t1)
        pm.register(t2)
        # Get by version
        v1 = pm.get("greet", version="1.0")
        v2 = pm.get("greet", version="2.0")
        assert v1.template.startswith("V1")
        assert v2.template.startswith("V2")
        # Latest (no version) should be v2
        latest = pm.get("greet")
        assert latest.template.startswith("V2")

    def test_get_nonexistent(self):
        pm = PromptManager()
        assert pm.get("nonexistent") is None

    def test_render(self):
        pm = PromptManager()
        pm.register(PromptTemplate(
            name="greet", template="Hello {{user}}!", variables=["user"],
        ))
        result = pm.render("greet", user="Alice")
        assert result == "Hello Alice!"

    def test_render_not_found(self):
        pm = PromptManager()
        with pytest.raises(ValueError, match="not found"):
            pm.render("nonexistent")

    def test_list_templates(self):
        pm = PromptManager()
        pm.register(PromptTemplate(name="a", template="...", tags=["system"]))
        pm.register(PromptTemplate(name="b", template="...", tags=["user"]))
        pm.register(PromptTemplate(name="c", template="...", tags=["system"]))

        all_templates = pm.list_templates()
        assert len(all_templates) == 3

        system_templates = pm.list_templates(tag="system")
        assert len(system_templates) == 2
        assert "a" in system_templates
        assert "c" in system_templates

    def test_track_performance(self):
        pm = PromptManager()
        pm.register(PromptTemplate(name="t", template="..."))
        pm.track_performance("t", score=0.9, latency_ms=100)
        pm.track_performance("t", score=0.8, latency_ms=200)
        perf = pm.get_performance("t")
        assert perf["uses"] == 2
        assert perf["avg_score"] == pytest.approx(0.85)
        assert perf["avg_latency_ms"] == pytest.approx(150)

    def test_get_performance_empty(self):
        pm = PromptManager()
        assert pm.get_performance("nonexistent") == {}


# ─────────────────────────────────────────────
# ContextManager
# ─────────────────────────────────────────────

class TestContextManager:
    def test_estimate_tokens(self):
        cm = ContextManager(max_context_tokens=10000)
        tokens = cm.estimate_tokens("a" * 100)
        # Should return a positive integer (exact value depends on tokenizer availability)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_fit_context_system_prompt_only(self):
        cm = ContextManager(max_context_tokens=10000)
        result = cm.fit_context(system_prompt="You are helpful.")
        assert result["system"] == "You are helpful."
        assert result["truncated"] is False

    def test_fit_context_with_rag(self):
        cm = ContextManager(max_context_tokens=100000)
        result = cm.fit_context(
            system_prompt="System prompt.",
            rag_context="Here is some context about the topic.",
        )
        assert result["system"] != ""
        assert result["rag_context"] != ""

    def test_fit_context_with_conversation(self):
        cm = ContextManager(max_context_tokens=100000)
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = cm.fit_context(
            system_prompt="System.",
            conversation=conversation,
        )
        assert len(result["messages"]) == 3

    def test_fit_context_truncation(self):
        cm = ContextManager(max_context_tokens=50)
        # Very limited budget
        long_system = "x" * 1000
        result = cm.fit_context(system_prompt=long_system, max_response_tokens=10)
        assert result["truncated"] is True

    def test_build_messages(self):
        cm = ContextManager(max_context_tokens=100000)
        system, messages = cm.build_messages(
            system_prompt="You are helpful.",
            user_query="What is Python?",
            rag_context="Python is a language.",
            conversation=[{"role": "user", "content": "prev"}],
        )
        assert "You are helpful" in system
        assert "Reference Material" in system
        assert "Python is a language" in system
        assert messages[-1]["content"] == "What is Python?"
        assert messages[-1]["role"] == "user"

    def test_build_messages_no_rag(self):
        cm = ContextManager(max_context_tokens=100000)
        system, messages = cm.build_messages(
            system_prompt="System.",
            user_query="Hi",
        )
        assert "Reference Material" not in system
        assert messages[-1]["content"] == "Hi"


# ─────────────────────────────────────────────
# ResponseCache
# ─────────────────────────────────────────────

class TestResponseCache:
    def test_set_and_get(self):
        cache = ResponseCache()
        cache.set("What is Python?", "A programming language.")
        result = cache.get("What is Python?")
        assert result == "A programming language."

    def test_get_miss(self):
        cache = ResponseCache()
        assert cache.get("unknown query") is None

    def test_get_with_system(self):
        cache = ResponseCache()
        cache.set("query", "response1", system="sys1")
        cache.set("query", "response2", system="sys2")
        r1 = cache.get("query", system="sys1")
        r2 = cache.get("query", system="sys2")
        assert r1 == "response1"
        assert r2 == "response2"

    def test_stats(self):
        cache = ResponseCache()
        cache.set("q1", "r1")
        cache.get("q1")
        cache.get("q1")
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["total_hits"] == 2

    def test_clear(self):
        cache = ResponseCache()
        cache.set("q", "r")
        cache.clear()
        assert cache.get("q") is None
        stats = cache.stats()
        assert stats["size"] == 0

    def test_eviction_on_max_size(self):
        cache = ResponseCache(max_size=2)
        cache.set("q1", "r1")
        cache.set("q2", "r2")
        cache.set("q3", "r3")
        # One entry should have been evicted
        assert len(cache._cache) == 2

    def test_ttl_conceptual(self):
        # The cache stores TTL but doesn't enforce it on get.
        # This test verifies TTL is stored.
        cache = ResponseCache()
        cache.set("q", "r", ttl=60)
        key = cache._make_key("q")
        assert cache._cache[key]["ttl"] == 60
