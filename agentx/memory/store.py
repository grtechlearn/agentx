"""
AgentX - Memory system.
Short-term (conversation) + Long-term (persistent) memory for agents.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory entry."""

    key: str
    value: Any
    memory_type: str = "general"  # general, fact, preference, conversation, skill
    agent: str = ""
    user_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    ttl: float | None = None  # Time-to-live in seconds (None = forever)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5  # 0.0 to 1.0

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


class BaseMemoryStore(ABC):
    """Abstract memory store."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""

    @abstractmethod
    async def retrieve(self, key: str) -> MemoryEntry | None:
        """Retrieve a memory by key."""

    @abstractmethod
    async def search(self, query: str, limit: int = 10, **filters: Any) -> list[MemoryEntry]:
        """Search memories by query."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a memory entry."""

    @abstractmethod
    async def list_all(self, **filters: Any) -> list[MemoryEntry]:
        """List all memories matching filters."""


class ShortTermMemory(BaseMemoryStore):
    """
    In-memory store for current session/conversation.
    Fast, ephemeral — cleared when session ends.
    """

    def __init__(self, max_entries: int = 1000):
        self._store: dict[str, MemoryEntry] = {}
        self.max_entries = max_entries

    async def store(self, entry: MemoryEntry) -> None:
        self._cleanup_expired()
        if len(self._store) >= self.max_entries:
            oldest = min(self._store.values(), key=lambda e: e.timestamp)
            del self._store[oldest.key]
        self._store[entry.key] = entry

    async def retrieve(self, key: str) -> MemoryEntry | None:
        entry = self._store.get(key)
        if entry and entry.is_expired():
            del self._store[key]
            return None
        return entry

    async def search(self, query: str, limit: int = 10, **filters: Any) -> list[MemoryEntry]:
        self._cleanup_expired()
        query_lower = query.lower()
        results = []
        for entry in self._store.values():
            if self._matches_filters(entry, filters):
                score = self._simple_relevance(entry, query_lower)
                if score > 0:
                    results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    async def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    async def list_all(self, **filters: Any) -> list[MemoryEntry]:
        self._cleanup_expired()
        return [e for e in self._store.values() if self._matches_filters(e, filters)]

    def clear(self) -> None:
        self._store.clear()

    def _cleanup_expired(self) -> None:
        expired = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired:
            del self._store[k]

    @staticmethod
    def _matches_filters(entry: MemoryEntry, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if hasattr(entry, key) and getattr(entry, key) != value:
                return False
        return True

    @staticmethod
    def _simple_relevance(entry: MemoryEntry, query: str) -> float:
        score = 0.0
        text = f"{entry.key} {json.dumps(entry.value) if isinstance(entry.value, (dict, list)) else str(entry.value)}".lower()
        words = query.split()
        for word in words:
            if word in text:
                score += 1.0
        return score / max(len(words), 1)


class LongTermMemory(BaseMemoryStore):
    """
    File-based persistent memory.
    Survives restarts, good for user preferences, learned facts, etc.
    """

    def __init__(self, storage_path: str = "./data/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, MemoryEntry] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            for file in self.storage_path.glob("*.json"):
                try:
                    data = json.loads(file.read_text())
                    entry = MemoryEntry(**data)
                    if not entry.is_expired():
                        self._cache[entry.key] = entry
                except Exception:
                    continue
            self._loaded = True

    async def store(self, entry: MemoryEntry) -> None:
        await self._ensure_loaded()
        self._cache[entry.key] = entry
        file_path = self.storage_path / f"{self._safe_filename(entry.key)}.json"
        file_path.write_text(entry.model_dump_json(indent=2))

    async def retrieve(self, key: str) -> MemoryEntry | None:
        await self._ensure_loaded()
        entry = self._cache.get(key)
        if entry and entry.is_expired():
            await self.delete(key)
            return None
        return entry

    async def search(self, query: str, limit: int = 10, **filters: Any) -> list[MemoryEntry]:
        await self._ensure_loaded()
        query_lower = query.lower()
        results = []
        for entry in self._cache.values():
            if not entry.is_expired() and self._matches_filters(entry, filters):
                score = self._relevance_score(entry, query_lower)
                if score > 0:
                    results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    async def delete(self, key: str) -> bool:
        await self._ensure_loaded()
        if key in self._cache:
            del self._cache[key]
            file_path = self.storage_path / f"{self._safe_filename(key)}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    async def list_all(self, **filters: Any) -> list[MemoryEntry]:
        await self._ensure_loaded()
        return [e for e in self._cache.values() if not e.is_expired() and self._matches_filters(e, filters)]

    @staticmethod
    def _safe_filename(key: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in key)

    @staticmethod
    def _matches_filters(entry: MemoryEntry, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if hasattr(entry, key) and getattr(entry, key) != value:
                return False
        return True

    @staticmethod
    def _relevance_score(entry: MemoryEntry, query: str) -> float:
        text = f"{entry.key} {json.dumps(entry.value) if isinstance(entry.value, (dict, list)) else str(entry.value)}".lower()
        words = query.split()
        matched = sum(1 for w in words if w in text)
        base_score = matched / max(len(words), 1)
        return base_score * (0.5 + entry.importance * 0.5)


class AgentMemory:
    """
    Combined memory system for an agent.
    Provides unified interface to short-term and long-term memory.
    """

    def __init__(self, storage_path: str = "./data/memory"):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(storage_path)

    async def remember(self, key: str, value: Any, long_term: bool = False, **kwargs: Any) -> None:
        entry = MemoryEntry(key=key, value=value, **kwargs)
        if long_term:
            await self.long_term.store(entry)
        else:
            await self.short_term.store(entry)

    async def recall(self, key: str) -> Any:
        entry = await self.short_term.retrieve(key)
        if entry is None:
            entry = await self.long_term.retrieve(key)
        return entry.value if entry else None

    async def search(self, query: str, limit: int = 10, **filters: Any) -> list[MemoryEntry]:
        short = await self.short_term.search(query, limit, **filters)
        long = await self.long_term.search(query, limit, **filters)
        combined = {e.key: e for e in short + long}
        entries = sorted(combined.values(), key=lambda e: e.importance, reverse=True)
        return entries[:limit]

    async def forget(self, key: str) -> None:
        await self.short_term.delete(key)
        await self.long_term.delete(key)

    def clear_session(self) -> None:
        self.short_term.clear()
