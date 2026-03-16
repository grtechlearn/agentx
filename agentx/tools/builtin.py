"""
AgentX - Built-in tools for agents.
Database, HTTP, file, and RAG tools.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.tool import BaseTool, ToolResult


class DatabaseTool(BaseTool):
    """Execute SQL queries against PostgreSQL."""

    name = "database"
    description = "Execute SQL queries against the database. Use for reading/writing user data, progress, goals, etc."

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.dsn)
        return self._pool

    async def execute(self, query: str, params: list[Any] | None = None) -> ToolResult:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                if query.strip().upper().startswith("SELECT"):
                    rows = await conn.fetch(query, *(params or []))
                    return ToolResult.ok([dict(r) for r in rows])
                else:
                    result = await conn.execute(query, *(params or []))
                    return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(str(e))


class HTTPTool(BaseTool):
    """Make HTTP requests."""

    name = "http"
    description = "Make HTTP requests to external APIs."

    async def execute(self, url: str, method: str = "GET", headers: dict[str, str] | None = None, body: str = "") -> ToolResult:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                kwargs: dict[str, Any] = {"headers": headers or {}}
                if body and method.upper() in ("POST", "PUT", "PATCH"):
                    kwargs["data"] = body
                async with session.request(method, url, **kwargs) as resp:
                    text = await resp.text()
                    return ToolResult.ok({"status": resp.status, "body": text})
        except Exception as e:
            return ToolResult.fail(str(e))


class RAGSearchTool(BaseTool):
    """Search the RAG knowledge base."""

    name = "rag_search"
    description = "Search the knowledge base for relevant technical information. Returns relevant documents and context."

    def __init__(self, rag_engine: Any):
        self.rag_engine = rag_engine

    async def execute(self, query: str, limit: int = 5, strategy: str = "hybrid", technology: str = "") -> ToolResult:
        try:
            filters = {}
            if technology:
                filters["technology"] = technology
            context = await self.rag_engine.get_context(
                query=query,
                limit=int(limit) if isinstance(limit, str) else limit,
                strategy=strategy,
                filters=filters if filters else None,
            )
            return ToolResult.ok(context)
        except Exception as e:
            return ToolResult.fail(str(e))


class RedisTool(BaseTool):
    """Interact with Redis for caching and pub/sub."""

    name = "redis"
    description = "Store and retrieve cached data from Redis."

    def __init__(self, url: str = "redis://localhost:6379"):
        self.url = url
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self.url)
        return self._client

    async def execute(self, action: str = "get", key: str = "", value: str = "", ttl: int = 0) -> ToolResult:
        try:
            client = await self._get_client()
            if action == "get":
                result = await client.get(key)
                return ToolResult.ok(result.decode() if result else None)
            elif action == "set":
                if ttl > 0:
                    await client.setex(key, ttl, value)
                else:
                    await client.set(key, value)
                return ToolResult.ok("OK")
            elif action == "delete":
                await client.delete(key)
                return ToolResult.ok("OK")
            else:
                return ToolResult.fail(f"Unknown action: {action}")
        except Exception as e:
            return ToolResult.fail(str(e))


class JSONParserTool(BaseTool):
    """Parse and validate JSON data."""

    name = "json_parser"
    description = "Parse, validate, and extract data from JSON strings."

    async def execute(self, data: str, path: str = "") -> ToolResult:
        try:
            parsed = json.loads(data)
            if path:
                for key in path.split("."):
                    if isinstance(parsed, dict):
                        parsed = parsed[key]
                    elif isinstance(parsed, list):
                        parsed = parsed[int(key)]
                    else:
                        return ToolResult.fail(f"Cannot navigate path '{path}' at key '{key}'")
            return ToolResult.ok(parsed)
        except Exception as e:
            return ToolResult.fail(str(e))
