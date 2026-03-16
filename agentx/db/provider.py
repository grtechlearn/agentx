"""
AgentX - Database Provider.
Unified async database interface for SQLite (dev) and PostgreSQL (production).
Zero-config by default — just works with SQLite, upgrade to PostgreSQL when ready.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import SCHEMA, SCHEMA_VERSION

logger = logging.getLogger("agentx")


class DatabaseProvider(ABC):
    """Abstract database provider."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def execute(self, query: str, params: tuple | list = ()) -> None: ...

    @abstractmethod
    async def execute_many(self, query: str, params_list: list[tuple | list]) -> None: ...

    @abstractmethod
    async def fetch_one(self, query: str, params: tuple | list = ()) -> dict | None: ...

    @abstractmethod
    async def fetch_all(self, query: str, params: tuple | list = ()) -> list[dict]: ...

    @abstractmethod
    async def fetch_val(self, query: str, params: tuple | list = ()) -> Any: ...

    async def init_schema(self) -> None:
        """Run schema creation if not already applied."""
        # Check if schema version table exists and has current version
        try:
            row = await self.fetch_one(
                "SELECT version FROM agentx_schema WHERE version = ?",
                (SCHEMA_VERSION,),
            )
            if row:
                logger.debug(f"Schema v{SCHEMA_VERSION} already applied")
                return
        except Exception:
            pass  # Table doesn't exist yet

        # Apply schema — split on semicolons but handle multi-line statements
        # by stripping comment-only lines first
        clean_lines = []
        for line in SCHEMA.split("\n"):
            stripped = line.strip()
            if stripped.startswith("--") or not stripped:
                continue
            clean_lines.append(line)
        clean_sql = "\n".join(clean_lines)

        for statement in clean_sql.split(";"):
            statement = statement.strip()
            if statement:
                await self.execute(statement)

        # Record schema version
        await self.execute(
            "INSERT OR REPLACE INTO agentx_schema (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, time.time()),
        )
        logger.info(f"Schema v{SCHEMA_VERSION} applied")


# ---------------------------------------------------------------------------
# SQLite Provider (zero-config, great for dev/small deployments)
# ---------------------------------------------------------------------------

class SQLiteProvider(DatabaseProvider):
    """
    SQLite-based database provider. Zero config — just works.

    Usage:
        db = SQLiteProvider()                          # ./data/agentx.db
        db = SQLiteProvider("path/to/my.db")           # custom path
        db = SQLiteProvider(":memory:")                 # in-memory (tests)
    """

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.environ.get("AGENTX_DB_PATH", "./data/agentx.db")
        self.db_path = db_path
        self._conn: Any = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        import aiosqlite

        # Create directory if needed
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        # Enable WAL mode for better concurrency
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info(f"SQLite connected: {self.db_path}")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def execute(self, query: str, params: tuple | list = ()) -> None:
        async with self._lock:
            await self._conn.execute(query, params)
            await self._conn.commit()

    async def execute_many(self, query: str, params_list: list[tuple | list]) -> None:
        async with self._lock:
            await self._conn.executemany(query, params_list)
            await self._conn.commit()

    async def fetch_one(self, query: str, params: tuple | list = ()) -> dict | None:
        async with self._lock:
            cursor = await self._conn.execute(query, params)
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    async def fetch_all(self, query: str, params: tuple | list = ()) -> list[dict]:
        async with self._lock:
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def fetch_val(self, query: str, params: tuple | list = ()) -> Any:
        row = await self.fetch_one(query, params)
        if row:
            return next(iter(row.values()))
        return None


# ---------------------------------------------------------------------------
# PostgreSQL Provider (production)
# ---------------------------------------------------------------------------

class PostgreSQLProvider(DatabaseProvider):
    """
    PostgreSQL-based database provider for production.

    Usage:
        db = PostgreSQLProvider("postgresql://user:pass@localhost/agentx")
        db = PostgreSQLProvider()  # reads AGENTX_DATABASE_URL env var
    """

    def __init__(self, dsn: str = ""):
        self.dsn = dsn or os.environ.get("AGENTX_DATABASE_URL", "")
        self._pool: Any = None

    async def connect(self) -> None:
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg not installed. Install with: pip install asyncpg\n"
                "Or: pip install agentx[postgres]"
            )

        if not self.dsn:
            raise ValueError(
                "PostgreSQL DSN required. Set AGENTX_DATABASE_URL env var or pass dsn= parameter.\n"
                "Example: postgresql://user:pass@localhost:5432/agentx"
            )

        self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
        logger.info("PostgreSQL connected")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _adapt_query(self, query: str) -> str:
        """Convert ? placeholders to $1, $2, ... for asyncpg."""
        parts = query.split("?")
        if len(parts) == 1:
            return query
        result = parts[0]
        for i, part in enumerate(parts[1:], 1):
            result += f"${i}" + part
        # Adapt SQLite-specific syntax
        result = result.replace("INSERT OR REPLACE", "INSERT ... ON CONFLICT DO UPDATE SET")
        result = result.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        return result

    async def execute(self, query: str, params: tuple | list = ()) -> None:
        query = self._adapt_query(query)
        async with self._pool.acquire() as conn:
            await conn.execute(query, *params)

    async def execute_many(self, query: str, params_list: list[tuple | list]) -> None:
        query = self._adapt_query(query)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for params in params_list:
                    await conn.execute(query, *params)

    async def fetch_one(self, query: str, params: tuple | list = ()) -> dict | None:
        query = self._adapt_query(query)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            if row is None:
                return None
            return dict(row)

    async def fetch_all(self, query: str, params: tuple | list = ()) -> list[dict]:
        query = self._adapt_query(query)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(r) for r in rows]

    async def fetch_val(self, query: str, params: tuple | list = ()) -> Any:
        query = self._adapt_query(query)
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *params)

    async def init_schema(self) -> None:
        """PostgreSQL schema init — adapt SQL syntax."""
        try:
            row = await self.fetch_one(
                "SELECT version FROM agentx_schema WHERE version = $1",
                (SCHEMA_VERSION,),
            )
            if row:
                return
        except Exception:
            pass

        # Adapt schema for PostgreSQL
        pg_schema = SCHEMA
        pg_schema = pg_schema.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        pg_schema = pg_schema.replace("REAL", "DOUBLE PRECISION")

        async with self._pool.acquire() as conn:
            await conn.execute(pg_schema)
            await conn.execute(
                "INSERT INTO agentx_schema (version, applied_at) VALUES ($1, $2) "
                "ON CONFLICT (version) DO NOTHING",
                SCHEMA_VERSION, time.time(),
            )
        logger.info(f"PostgreSQL schema v{SCHEMA_VERSION} applied")


# ---------------------------------------------------------------------------
# Database — High-level interface with CRUD helpers
# ---------------------------------------------------------------------------

class Database:
    """
    High-level database interface for AgentX.

    Usage:
        # Zero-config (SQLite)
        db = Database()
        await db.connect()

        # PostgreSQL
        db = Database(provider="postgres", dsn="postgresql://...")
        await db.connect()

        # CRUD
        await db.save_user(user_dict)
        user = await db.get_user("user-123")
        sessions = await db.get_user_sessions("user-123")
    """

    def __init__(
        self,
        provider: str = "",
        db_path: str = "",
        dsn: str = "",
    ):
        provider = provider or os.environ.get("AGENTX_DB_PROVIDER", "sqlite")

        if provider == "postgres" or provider == "postgresql":
            self._provider = PostgreSQLProvider(dsn=dsn)
        else:
            self._provider = SQLiteProvider(db_path=db_path)

        self._connected = False

    async def connect(self) -> None:
        """Connect and initialize schema."""
        await self._provider.connect()
        await self._provider.init_schema()
        self._connected = True

    async def close(self) -> None:
        await self._provider.close()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- Raw access --

    async def execute(self, query: str, params: tuple | list = ()) -> None:
        return await self._provider.execute(query, params)

    async def fetch_one(self, query: str, params: tuple | list = ()) -> dict | None:
        return await self._provider.fetch_one(query, params)

    async def fetch_all(self, query: str, params: tuple | list = ()) -> list[dict]:
        return await self._provider.fetch_all(query, params)

    # -----------------------------------------------------------------------
    # Users
    # -----------------------------------------------------------------------

    async def save_user(
        self,
        id: str = "",
        name: str = "",
        email: str = "",
        role: str = "user",
        organization_id: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Create or update a user. Returns user ID."""
        user_id = id or str(uuid.uuid4())
        now = time.time()
        existing = await self._provider.fetch_one(
            "SELECT id FROM agentx_users WHERE id = ?", (user_id,)
        )
        if existing:
            await self._provider.execute(
                "UPDATE agentx_users SET name=?, email=?, role=?, organization_id=?, "
                "metadata=?, updated_at=? WHERE id=?",
                (name, email, role, organization_id, json.dumps(metadata or {}), now, user_id),
            )
        else:
            await self._provider.execute(
                "INSERT INTO agentx_users (id, name, email, role, organization_id, metadata, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, name, email, role, organization_id, json.dumps(metadata or {}), now, now),
            )
        return user_id

    async def get_user(self, user_id: str) -> dict | None:
        row = await self._provider.fetch_one(
            "SELECT * FROM agentx_users WHERE id = ?", (user_id,)
        )
        if row:
            row["metadata"] = json.loads(row.get("metadata", "{}"))
            row["custom_permissions"] = json.loads(row.get("custom_permissions", "[]"))
            row["denied_permissions"] = json.loads(row.get("denied_permissions", "[]"))
        return row

    async def list_users(self, active_only: bool = True) -> list[dict]:
        q = "SELECT * FROM agentx_users"
        if active_only:
            q += " WHERE is_active = 1"
        q += " ORDER BY created_at DESC"
        return await self._provider.fetch_all(q)

    # -----------------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------------

    async def create_session(self, user_id: str, metadata: dict | None = None) -> str:
        session_id = str(uuid.uuid4())
        now = time.time()
        await self._provider.execute(
            "INSERT INTO agentx_sessions (id, user_id, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, json.dumps(metadata or {}), now, now),
        )
        return session_id

    async def get_session(self, session_id: str) -> dict | None:
        row = await self._provider.fetch_one(
            "SELECT * FROM agentx_sessions WHERE id = ?", (session_id,)
        )
        if row:
            row["conversation_history"] = json.loads(row.get("conversation_history", "[]"))
            row["shared_state"] = json.loads(row.get("shared_state", "{}"))
            row["agent_results"] = json.loads(row.get("agent_results", "{}"))
            row["metadata"] = json.loads(row.get("metadata", "{}"))
        return row

    async def update_session(self, session_id: str, **kwargs: Any) -> None:
        sets = []
        params = []
        for key in ("conversation_history", "shared_state", "agent_results", "metadata"):
            if key in kwargs:
                sets.append(f"{key} = ?")
                val = kwargs[key]
                params.append(json.dumps(val) if isinstance(val, (dict, list)) else val)
        if sets:
            sets.append("updated_at = ?")
            params.append(time.time())
            params.append(session_id)
            await self._provider.execute(
                f"UPDATE agentx_sessions SET {', '.join(sets)} WHERE id = ?",
                tuple(params),
            )

    async def get_user_sessions(self, user_id: str, limit: int = 20) -> list[dict]:
        return await self._provider.fetch_all(
            "SELECT * FROM agentx_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )

    # -----------------------------------------------------------------------
    # Conversations
    # -----------------------------------------------------------------------

    async def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        agent_name: str = "",
        data: dict | None = None,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> int:
        now = time.time()
        await self._provider.execute(
            "INSERT INTO agentx_conversations "
            "(session_id, user_id, agent_name, role, content, data, tokens_used, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, user_id, agent_name, role, content, json.dumps(data or {}),
             tokens_used, cost_usd, now),
        )
        row = await self._provider.fetch_one("SELECT last_insert_rowid() as id")
        return row["id"] if row else 0

    async def get_conversation(self, session_id: str, limit: int = 100) -> list[dict]:
        rows = await self._provider.fetch_all(
            "SELECT * FROM agentx_conversations WHERE session_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        )
        for r in rows:
            r["data"] = json.loads(r.get("data", "{}"))
        return rows

    # -----------------------------------------------------------------------
    # Memory
    # -----------------------------------------------------------------------

    async def save_memory(
        self,
        key: str,
        value: str,
        memory_type: str = "general",
        agent: str = "",
        user_id: str = "",
        importance: float = 0.5,
        ttl: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        now = time.time()
        existing = await self._provider.fetch_one(
            "SELECT key FROM agentx_memory WHERE key = ? AND user_id = ?",
            (key, user_id),
        )
        if existing:
            await self._provider.execute(
                "UPDATE agentx_memory SET value=?, memory_type=?, agent=?, importance=?, "
                "ttl=?, metadata=?, updated_at=? WHERE key=? AND user_id=?",
                (value, memory_type, agent, importance, ttl,
                 json.dumps(metadata or {}), now, key, user_id),
            )
        else:
            await self._provider.execute(
                "INSERT INTO agentx_memory (key, value, memory_type, agent, user_id, importance, "
                "ttl, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, value, memory_type, agent, user_id, importance, ttl,
                 json.dumps(metadata or {}), now, now),
            )

    async def get_memory(self, key: str, user_id: str = "") -> dict | None:
        row = await self._provider.fetch_one(
            "SELECT * FROM agentx_memory WHERE key = ? AND user_id = ?",
            (key, user_id),
        )
        if row:
            row["metadata"] = json.loads(row.get("metadata", "{}"))
            # Check TTL
            if row.get("ttl") and time.time() > row["created_at"] + row["ttl"]:
                await self._provider.execute(
                    "DELETE FROM agentx_memory WHERE key = ? AND user_id = ?",
                    (key, user_id),
                )
                return None
        return row

    async def search_memory(
        self,
        user_id: str = "",
        memory_type: str = "",
        agent: str = "",
        min_importance: float = 0.0,
    ) -> list[dict]:
        conditions = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = await self._provider.fetch_all(
            f"SELECT * FROM agentx_memory WHERE {where} ORDER BY importance DESC",
            tuple(params),
        )
        for r in rows:
            r["metadata"] = json.loads(r.get("metadata", "{}"))
        return rows

    async def delete_memory(self, key: str, user_id: str = "") -> None:
        await self._provider.execute(
            "DELETE FROM agentx_memory WHERE key = ? AND user_id = ?",
            (key, user_id),
        )

    # -----------------------------------------------------------------------
    # Agents
    # -----------------------------------------------------------------------

    async def save_agent(
        self,
        name: str,
        role: str = "",
        system_prompt: str = "",
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        now = time.time()
        existing = await self._provider.fetch_one(
            "SELECT name FROM agentx_agents WHERE name = ?", (name,)
        )
        if existing:
            await self._provider.execute(
                "UPDATE agentx_agents SET role=?, system_prompt=?, model=?, provider=?, "
                "temperature=?, max_tokens=?, tools=?, metadata=?, updated_at=? WHERE name=?",
                (role, system_prompt, model, provider, temperature, max_tokens,
                 json.dumps(tools or []), json.dumps(metadata or {}), now, name),
            )
        else:
            await self._provider.execute(
                "INSERT INTO agentx_agents (name, role, system_prompt, model, provider, "
                "temperature, max_tokens, tools, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, role, system_prompt, model, provider, temperature, max_tokens,
                 json.dumps(tools or []), json.dumps(metadata or {}), now, now),
            )

    async def get_agent(self, name: str) -> dict | None:
        row = await self._provider.fetch_one(
            "SELECT * FROM agentx_agents WHERE name = ?", (name,)
        )
        if row:
            row["tools"] = json.loads(row.get("tools", "[]"))
            row["metadata"] = json.loads(row.get("metadata", "{}"))
        return row

    async def list_agents(self, active_only: bool = True) -> list[dict]:
        q = "SELECT * FROM agentx_agents"
        if active_only:
            q += " WHERE is_active = 1"
        return await self._provider.fetch_all(q)

    # -----------------------------------------------------------------------
    # Evaluations
    # -----------------------------------------------------------------------

    async def save_evaluation(
        self,
        session_id: str = "",
        user_id: str = "",
        agent_name: str = "",
        query: str = "",
        response: str = "",
        score: float = 0.0,
        faithfulness: float = 0.0,
        hallucination_detected: bool = False,
        evaluation_data: dict | None = None,
    ) -> int:
        now = time.time()
        await self._provider.execute(
            "INSERT INTO agentx_evaluations "
            "(session_id, user_id, agent_name, query, response, score, faithfulness, "
            "hallucination_detected, evaluation_data, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, user_id, agent_name, query, response, score, faithfulness,
             1 if hallucination_detected else 0, json.dumps(evaluation_data or {}), now),
        )
        row = await self._provider.fetch_one("SELECT last_insert_rowid() as id")
        return row["id"] if row else 0

    async def get_evaluations(
        self, user_id: str = "", agent_name: str = "", limit: int = 50
    ) -> list[dict]:
        conditions = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        rows = await self._provider.fetch_all(
            f"SELECT * FROM agentx_evaluations WHERE {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )
        for r in rows:
            r["evaluation_data"] = json.loads(r.get("evaluation_data", "{}"))
        return rows

    # -----------------------------------------------------------------------
    # Cost Tracking
    # -----------------------------------------------------------------------

    async def track_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        user_id: str = "",
        agent_name: str = "",
        session_id: str = "",
    ) -> None:
        await self._provider.execute(
            "INSERT INTO agentx_costs "
            "(user_id, agent_name, model, input_tokens, output_tokens, cost_usd, "
            "session_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, agent_name, model, input_tokens, output_tokens, cost_usd,
             session_id, time.time()),
        )

    async def get_cost_summary(
        self, user_id: str = "", days: int = 30
    ) -> dict:
        since = time.time() - (days * 86400)
        conditions = ["created_at >= ?"]
        params: list[Any] = [since]
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        where = " AND ".join(conditions)

        row = await self._provider.fetch_one(
            f"SELECT COALESCE(SUM(cost_usd), 0) as total_cost, "
            f"COALESCE(SUM(input_tokens), 0) as total_input, "
            f"COALESCE(SUM(output_tokens), 0) as total_output, "
            f"COUNT(*) as total_calls FROM agentx_costs WHERE {where}",
            tuple(params),
        )
        return row or {"total_cost": 0, "total_input": 0, "total_output": 0, "total_calls": 0}

    # -----------------------------------------------------------------------
    # Audit Log
    # -----------------------------------------------------------------------

    async def audit(
        self,
        action: str,
        user_id: str = "",
        resource: str = "",
        details: dict | None = None,
        ip_address: str = "",
        success: bool = True,
        reason: str = "",
    ) -> None:
        await self._provider.execute(
            "INSERT INTO agentx_audit "
            "(user_id, action, resource, details, ip_address, success, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, action, resource, json.dumps(details or {}), ip_address,
             1 if success else 0, reason, time.time()),
        )

    async def get_audit_log(
        self, user_id: str = "", action: str = "", limit: int = 100
    ) -> list[dict]:
        conditions = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        rows = await self._provider.fetch_all(
            f"SELECT * FROM agentx_audit WHERE {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )
        for r in rows:
            r["details"] = json.loads(r.get("details", "{}"))
        return rows

    # -----------------------------------------------------------------------
    # Goals
    # -----------------------------------------------------------------------

    async def save_goal(
        self,
        user_id: str,
        title: str,
        description: str = "",
        target_data: dict | None = None,
        status: str = "active",
        target_date: float | None = None,
        id: str = "",
    ) -> str:
        goal_id = id or str(uuid.uuid4())
        now = time.time()
        existing = await self._provider.fetch_one(
            "SELECT id FROM agentx_goals WHERE id = ?", (goal_id,)
        )
        if existing:
            await self._provider.execute(
                "UPDATE agentx_goals SET title=?, description=?, target_data=?, "
                "status=?, target_date=?, updated_at=? WHERE id=?",
                (title, description, json.dumps(target_data or {}), status, target_date, now, goal_id),
            )
        else:
            await self._provider.execute(
                "INSERT INTO agentx_goals (id, user_id, title, description, target_data, "
                "status, target_date, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (goal_id, user_id, title, description, json.dumps(target_data or {}),
                 status, target_date, now, now),
            )
        return goal_id

    async def get_user_goals(self, user_id: str, status: str = "") -> list[dict]:
        if status:
            rows = await self._provider.fetch_all(
                "SELECT * FROM agentx_goals WHERE user_id = ? AND status = ? ORDER BY created_at DESC",
                (user_id, status),
            )
        else:
            rows = await self._provider.fetch_all(
                "SELECT * FROM agentx_goals WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        for r in rows:
            r["target_data"] = json.loads(r.get("target_data", "{}"))
            r["progress_data"] = json.loads(r.get("progress_data", "{}"))
        return rows

    async def update_goal_progress(
        self, goal_id: str, progress_data: dict, streak_days: int = 0
    ) -> None:
        await self._provider.execute(
            "UPDATE agentx_goals SET progress_data=?, streak_days=?, updated_at=? WHERE id=?",
            (json.dumps(progress_data), streak_days, time.time(), goal_id),
        )

    # -----------------------------------------------------------------------
    # Learned Rules (Self-Learning)
    # -----------------------------------------------------------------------

    async def save_rule(
        self, pattern: str, response: str, confidence: float = 0.0, source: str = "auto"
    ) -> None:
        now = time.time()
        existing = await self._provider.fetch_one(
            "SELECT id FROM agentx_learned_rules WHERE pattern = ?", (pattern,)
        )
        if existing:
            await self._provider.execute(
                "UPDATE agentx_learned_rules SET response=?, confidence=?, "
                "source=?, updated_at=? WHERE pattern=?",
                (response, confidence, source, now, pattern),
            )
        else:
            await self._provider.execute(
                "INSERT INTO agentx_learned_rules "
                "(pattern, response, confidence, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (pattern, response, confidence, source, now, now),
            )

    async def find_rule(self, pattern: str, min_confidence: float = 0.5) -> dict | None:
        return await self._provider.fetch_one(
            "SELECT * FROM agentx_learned_rules "
            "WHERE pattern = ? AND confidence >= ? AND is_active = 1",
            (pattern, min_confidence),
        )

    async def increment_rule_usage(self, pattern: str) -> None:
        await self._provider.execute(
            "UPDATE agentx_learned_rules SET times_used = times_used + 1, "
            "updated_at = ? WHERE pattern = ?",
            (time.time(), pattern),
        )

    # -----------------------------------------------------------------------
    # Prompts
    # -----------------------------------------------------------------------

    async def save_prompt(
        self,
        name: str,
        template: str,
        version: str = "1.0",
        description: str = "",
        variables: list[str] | None = None,
        model_hint: str = "",
        tags: list[str] | None = None,
    ) -> None:
        now = time.time()
        existing = await self._provider.fetch_one(
            "SELECT name FROM agentx_prompts WHERE name = ? AND version = ?",
            (name, version),
        )
        if existing:
            await self._provider.execute(
                "UPDATE agentx_prompts SET template=?, description=?, variables=?, "
                "model_hint=?, tags=?, updated_at=? WHERE name=? AND version=?",
                (template, description, json.dumps(variables or []),
                 model_hint, json.dumps(tags or []), now, name, version),
            )
        else:
            await self._provider.execute(
                "INSERT INTO agentx_prompts (name, version, template, description, variables, "
                "model_hint, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, version, template, description, json.dumps(variables or []),
                 model_hint, json.dumps(tags or []), now, now),
            )

    async def get_prompt(self, name: str, version: str = "") -> dict | None:
        if version:
            row = await self._provider.fetch_one(
                "SELECT * FROM agentx_prompts WHERE name = ? AND version = ? AND is_active = 1",
                (name, version),
            )
        else:
            row = await self._provider.fetch_one(
                "SELECT * FROM agentx_prompts WHERE name = ? AND is_active = 1 "
                "ORDER BY version DESC LIMIT 1",
                (name,),
            )
        if row:
            row["variables"] = json.loads(row.get("variables", "[]"))
            row["tags"] = json.loads(row.get("tags", "[]"))
            row["performance_data"] = json.loads(row.get("performance_data", "{}"))
        return row


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_database(
    provider: str = "",
    db_path: str = "",
    dsn: str = "",
) -> Database:
    """
    Create a Database instance.

    Usage:
        db = create_database()                                      # SQLite default
        db = create_database(db_path="./my_app.db")                 # SQLite custom path
        db = create_database(provider="postgres", dsn="postgresql://...")  # PostgreSQL
    """
    return Database(provider=provider, db_path=db_path, dsn=dsn)
