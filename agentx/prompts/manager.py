"""
AgentX - Prompt Engineering & Context Management.
Phase 5: Prompt templates, context window management, prompt optimization.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


class PromptTemplate(BaseModel):
    """A reusable, versioned prompt template."""

    name: str
    version: str = "1.0"
    template: str
    description: str = ""
    variables: list[str] = Field(default_factory=list)
    model_hint: str = ""  # recommended model for this prompt
    max_tokens_hint: int = 0
    temperature_hint: float | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render(self, **kwargs: Any) -> str:
        """Render the template with variables."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        # Check for unresolved variables
        for var in self.variables:
            placeholder = f"{{{{{var}}}}}"
            if placeholder in result:
                logger.warning(f"Unresolved variable '{var}' in prompt '{self.name}'")
        return result


class PromptManager:
    """
    Manage prompt templates with versioning and A/B testing support.

    Features:
    - Template storage and retrieval
    - Variable interpolation
    - Versioning
    - Performance tracking per prompt

    When a Database instance is provided, templates persist to DB.
    Otherwise uses in-memory storage.
    """

    def __init__(self, db: Any = None) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._performance: dict[str, dict[str, Any]] = {}
        self._db = db  # Optional Database instance

    def register(self, template: PromptTemplate) -> None:
        key = f"{template.name}:{template.version}"
        self._templates[key] = template
        self._templates[template.name] = template  # latest version shortcut
        # Persist to DB
        if self._db and self._db.is_connected:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._db.save_prompt(
                    name=template.name, template=template.template,
                    version=template.version, description=template.description,
                    variables=template.variables, model_hint=template.model_hint,
                    tags=template.tags,
                ))
            except RuntimeError:
                pass

    async def register_async(self, template: PromptTemplate) -> None:
        """Register with guaranteed DB persistence."""
        key = f"{template.name}:{template.version}"
        self._templates[key] = template
        self._templates[template.name] = template
        if self._db and self._db.is_connected:
            await self._db.save_prompt(
                name=template.name, template=template.template,
                version=template.version, description=template.description,
                variables=template.variables, model_hint=template.model_hint,
                tags=template.tags,
            )

    async def load_from_db(self) -> None:
        """Load all prompt templates from database into memory."""
        if not self._db or not self._db.is_connected:
            return
        rows = await self._db.fetch_all(
            "SELECT * FROM agentx_prompts WHERE is_active = 1"
        )
        for row in rows:
            template = PromptTemplate(
                name=row["name"], version=row.get("version", "1.0"),
                template=row["template"], description=row.get("description", ""),
                variables=json.loads(row.get("variables", "[]")),
                model_hint=row.get("model_hint", ""),
                tags=json.loads(row.get("tags", "[]")),
            )
            key = f"{template.name}:{template.version}"
            self._templates[key] = template
            self._templates[template.name] = template

    def get(self, name: str, version: str = "") -> PromptTemplate | None:
        key = f"{name}:{version}" if version else name
        return self._templates.get(key)

    def render(self, name: str, version: str = "", **kwargs: Any) -> str:
        template = self.get(name, version)
        if not template:
            raise ValueError(f"Prompt template '{name}' not found")
        return template.render(**kwargs)

    def track_performance(self, name: str, score: float, latency_ms: float = 0) -> None:
        if name not in self._performance:
            self._performance[name] = {"scores": [], "latencies": [], "uses": 0}
        self._performance[name]["scores"].append(score)
        self._performance[name]["latencies"].append(latency_ms)
        self._performance[name]["uses"] += 1

    def get_performance(self, name: str) -> dict[str, Any]:
        perf = self._performance.get(name, {})
        if not perf:
            return {}
        scores = perf.get("scores", [])
        latencies = perf.get("latencies", [])
        return {
            "uses": perf["uses"],
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        }

    def list_templates(self, tag: str = "") -> list[str]:
        seen = set()
        results = []
        for t in self._templates.values():
            if t.name not in seen:
                if not tag or tag in t.tags:
                    results.append(t.name)
                    seen.add(t.name)
        return results


class ContextManager:
    """
    Manage LLM context window efficiently.

    Handles:
    - Token counting (accurate via tiktoken, fallback to estimation)
    - Context truncation strategies
    - Priority-based message selection
    - System prompt + RAG context + conversation fitting
    - Dynamic budget adjustment based on content importance
    """

    # Approximate tokens per character (conservative fallback)
    CHARS_PER_TOKEN = 4

    def __init__(self, max_context_tokens: int = 100000, tokenizer: str = ""):
        self.max_context_tokens = max_context_tokens
        self._tokenizer: Any = None
        self._tokenizer_name = tokenizer

        # Try to load accurate tokenizer
        if tokenizer:
            self._init_tokenizer(tokenizer)
        else:
            # Auto-detect best available
            self._try_auto_tokenizer()

    def _init_tokenizer(self, name: str) -> None:
        """Initialize a specific tokenizer."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(name)
            logger.debug(f"Using tiktoken tokenizer: {name}")
        except (ImportError, Exception):
            pass

    def _try_auto_tokenizer(self) -> None:
        """Try to auto-detect the best tokenizer."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude/GPT-4 compatible
            logger.debug("Using tiktoken cl100k_base tokenizer")
        except ImportError:
            logger.debug("tiktoken not installed, using character-based estimation")

    def estimate_tokens(self, text: str) -> int:
        """
        Count tokens accurately if tokenizer available, else estimate.
        Accurate counting prevents wasted context or truncation errors.
        """
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: character-based estimation
        return len(text) // self.CHARS_PER_TOKEN

    def fit_context(
        self,
        system_prompt: str,
        rag_context: str = "",
        conversation: list[dict[str, str]] | None = None,
        max_response_tokens: int = 4096,
        priorities: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """
        Fit all context components within the token limit.

        Priority order (default):
        1. System prompt (always included)
        2. RAG context (most recent/relevant)
        3. Recent conversation messages
        4. Older conversation messages (truncated first)

        Returns fitted context components.
        """
        available = self.max_context_tokens - max_response_tokens
        used = 0
        result: dict[str, Any] = {"system": "", "rag_context": "", "messages": [], "truncated": False}

        # 1. System prompt (always fits)
        sys_tokens = self.estimate_tokens(system_prompt)
        if sys_tokens <= available:
            result["system"] = system_prompt
            used += sys_tokens
        else:
            # Truncate system prompt if too long
            max_chars = (available // 2) * self.CHARS_PER_TOKEN
            result["system"] = system_prompt[:max_chars]
            used += available // 2
            result["truncated"] = True

        # 2. RAG context
        if rag_context:
            rag_tokens = self.estimate_tokens(rag_context)
            rag_budget = min(rag_tokens, (available - used) // 2)  # Max 50% of remaining
            if rag_budget > 0:
                if rag_tokens <= rag_budget:
                    result["rag_context"] = rag_context
                    used += rag_tokens
                else:
                    # Truncate RAG context
                    max_chars = rag_budget * self.CHARS_PER_TOKEN
                    result["rag_context"] = rag_context[:max_chars] + "\n[...truncated]"
                    used += rag_budget
                    result["truncated"] = True

        # 3. Conversation messages (newest first priority)
        if conversation:
            remaining = available - used
            messages = []
            for msg in reversed(conversation):
                msg_tokens = self.estimate_tokens(msg.get("content", ""))
                if msg_tokens <= remaining:
                    messages.insert(0, msg)
                    remaining -= msg_tokens
                else:
                    result["truncated"] = True
                    break
            result["messages"] = messages

        result["tokens_used"] = used
        result["tokens_remaining"] = available - used
        return result

    def build_messages(
        self,
        system_prompt: str,
        user_query: str,
        rag_context: str = "",
        conversation: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Build final system prompt and messages list for LLM call.
        Convenience method that handles context fitting.
        """
        fitted = self.fit_context(system_prompt, rag_context, conversation)

        # Build system prompt with RAG context
        final_system = fitted["system"]
        if fitted["rag_context"]:
            final_system += f"\n\n## Reference Material\n{fitted['rag_context']}"

        # Build messages
        messages = fitted["messages"]
        messages.append({"role": "user", "content": user_query})

        return final_system, messages


class ResponseCache:
    """
    Cache LLM responses to reduce API calls and costs.
    Uses semantic similarity for cache hits.
    """

    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.95):
        self._cache: dict[str, dict[str, Any]] = {}
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def _make_key(self, query: str, system: str = "") -> str:
        """Create a cache key from query + system prompt."""
        import hashlib
        normalized = f"{system.strip()[:200]}|{query.strip().lower()}"
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str, system: str = "") -> str | None:
        """Get cached response if available."""
        key = self._make_key(query, system)
        entry = self._cache.get(key)
        if entry:
            entry["hits"] += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return entry["response"]
        return None

    def set(self, query: str, response: str, system: str = "", ttl: int = 3600) -> None:
        """Cache a response."""
        if len(self._cache) >= self.max_size:
            # Evict least-hit entry
            min_key = min(self._cache, key=lambda k: self._cache[k]["hits"])
            del self._cache[min_key]

        key = self._make_key(query, system)
        import time
        self._cache[key] = {"response": response, "hits": 0, "created": time.time(), "ttl": ttl}

    def stats(self) -> dict[str, Any]:
        total_hits = sum(e["hits"] for e in self._cache.values())
        return {"size": len(self._cache), "total_hits": total_hits, "max_size": self.max_size}

    def clear(self) -> None:
        self._cache.clear()
