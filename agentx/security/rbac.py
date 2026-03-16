"""
AgentX - Security & RBAC (Role-Based Access Control).
Phase 6: Security, authentication, authorization, and audit logging.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


class Role(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"
    API = "api"


class Permission(str, Enum):
    # Agent permissions
    AGENT_RUN = "agent:run"
    AGENT_CREATE = "agent:create"
    AGENT_DELETE = "agent:delete"
    AGENT_CONFIG = "agent:config"

    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # RAG permissions
    RAG_SEARCH = "rag:search"
    RAG_INGEST = "rag:ingest"
    RAG_DELETE = "rag:delete"

    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_BILLING = "admin:billing"
    ADMIN_AUDIT = "admin:audit"

    # Analytics
    ANALYTICS_SELF = "analytics:self"
    ANALYTICS_TEAM = "analytics:team"
    ANALYTICS_ALL = "analytics:all"


# Default role-permission mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.MANAGER: {
        Permission.AGENT_RUN, Permission.AGENT_CREATE, Permission.AGENT_CONFIG,
        Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_EXPORT,
        Permission.RAG_SEARCH, Permission.RAG_INGEST,
        Permission.ANALYTICS_SELF, Permission.ANALYTICS_TEAM,
    },
    Role.USER: {
        Permission.AGENT_RUN,
        Permission.DATA_READ,
        Permission.RAG_SEARCH,
        Permission.ANALYTICS_SELF,
    },
    Role.VIEWER: {
        Permission.DATA_READ,
        Permission.RAG_SEARCH,
        Permission.ANALYTICS_SELF,
    },
    Role.API: {
        Permission.AGENT_RUN,
        Permission.DATA_READ,
        Permission.RAG_SEARCH,
    },
}


class User(BaseModel):
    """User with role-based permissions."""

    id: str
    name: str = ""
    email: str = ""
    role: Role = Role.USER
    organization_id: str = ""
    custom_permissions: set[Permission] = Field(default_factory=set)
    denied_permissions: set[Permission] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: float = Field(default_factory=time.time)

    def has_permission(self, permission: Permission) -> bool:
        if not self.is_active:
            return False
        if permission in self.denied_permissions:
            return False
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        return permission in role_perms or permission in self.custom_permissions

    def get_permissions(self) -> set[Permission]:
        if not self.is_active:
            return set()
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        return (role_perms | self.custom_permissions) - self.denied_permissions


class AuditEntry(BaseModel):
    """Audit log entry for tracking actions."""

    timestamp: float = Field(default_factory=time.time)
    user_id: str = ""
    action: str = ""
    resource: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str = ""
    success: bool = True
    reason: str = ""


class RBACManager:
    """
    Role-Based Access Control manager.
    Handles authentication, authorization, and audit logging.

    When a Database instance is provided, users and audit logs persist to DB.
    Otherwise uses in-memory storage (lost on restart).
    """

    def __init__(self, secret_key: str = "agentx-secret", db: Any = None):
        self._users: dict[str, User] = {}
        self._audit_log: list[AuditEntry] = []
        self._secret_key = secret_key
        self._rate_limits: dict[str, list[float]] = {}
        self._db = db  # Optional Database instance

    # --- User Management ---

    def add_user(self, user: User) -> None:
        self._users[user.id] = user
        self._audit("system", "user_created", f"user:{user.id}", {"role": user.role.value})
        # Persist to DB
        if self._db and self._db.is_connected:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._db.save_user(
                    id=user.id, name=user.name, email=user.email,
                    role=user.role.value, organization_id=user.organization_id,
                    metadata=user.metadata,
                ))
            except RuntimeError:
                pass  # No event loop, skip async persistence

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def remove_user(self, user_id: str) -> bool:
        if user_id in self._users:
            del self._users[user_id]
            self._audit("system", "user_removed", f"user:{user_id}")
            return True
        return False

    def update_role(self, user_id: str, new_role: Role) -> bool:
        user = self._users.get(user_id)
        if user:
            old_role = user.role
            user.role = new_role
            self._audit("system", "role_changed", f"user:{user_id}", {"old": old_role.value, "new": new_role.value})
            if self._db and self._db.is_connected:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._db.save_user(
                        id=user.id, name=user.name, email=user.email,
                        role=new_role.value, organization_id=user.organization_id,
                        metadata=user.metadata,
                    ))
                except RuntimeError:
                    pass
            return True
        return False

    # --- Authorization ---

    def authorize(self, user_id: str, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        user = self._users.get(user_id)
        if not user:
            self._audit(user_id, "auth_failed", str(permission), {"reason": "user_not_found"}, success=False)
            return False

        allowed = user.has_permission(permission)
        if not allowed:
            self._audit(user_id, "auth_denied", str(permission), {"role": user.role.value}, success=False)
            logger.warning(f"Permission denied: user={user_id} permission={permission}")
        return allowed

    def require(self, user_id: str, permission: Permission) -> None:
        """Raise error if user doesn't have permission."""
        if not self.authorize(user_id, permission):
            raise PermissionError(f"User '{user_id}' lacks permission: {permission.value}")

    # --- Rate Limiting ---

    def check_rate_limit(self, user_id: str, max_requests: int = 60, window_seconds: int = 60) -> bool:
        """Check if user is within rate limits."""
        now = time.time()
        if user_id not in self._rate_limits:
            self._rate_limits[user_id] = []

        # Remove old entries
        self._rate_limits[user_id] = [t for t in self._rate_limits[user_id] if now - t < window_seconds]

        if len(self._rate_limits[user_id]) >= max_requests:
            self._audit(user_id, "rate_limited", "api", {"count": len(self._rate_limits[user_id])}, success=False)
            return False

        self._rate_limits[user_id].append(now)
        return True

    # --- API Key Management ---

    def generate_api_key(self, user_id: str) -> str:
        """Generate an API key for a user."""
        raw = f"{user_id}:{self._secret_key}:{time.time()}"
        key = hashlib.sha256(raw.encode()).hexdigest()
        user = self._users.get(user_id)
        if user:
            user.metadata["api_key"] = key
        self._audit(user_id, "api_key_generated", f"user:{user_id}")
        return f"agx_{key[:40]}"

    def validate_api_key(self, api_key: str) -> User | None:
        """Validate an API key and return the associated user."""
        for user in self._users.values():
            stored_key = user.metadata.get("api_key", "")
            if api_key == f"agx_{stored_key[:40]}":
                return user
        return None

    # --- Async User Management (for DB-backed operations) ---

    async def load_users(self) -> None:
        """Load users from database into memory cache."""
        if not self._db or not self._db.is_connected:
            return
        rows = await self._db.list_users()
        for row in rows:
            user = User(
                id=row["id"], name=row.get("name", ""), email=row.get("email", ""),
                role=Role(row.get("role", "user")),
                organization_id=row.get("organization_id", ""),
                is_active=bool(row.get("is_active", 1)),
                created_at=row.get("created_at", time.time()),
            )
            self._users[user.id] = user

    async def add_user_async(self, user: User) -> None:
        """Add user with async DB persistence."""
        self._users[user.id] = user
        self._audit("system", "user_created", f"user:{user.id}", {"role": user.role.value})
        if self._db and self._db.is_connected:
            await self._db.save_user(
                id=user.id, name=user.name, email=user.email,
                role=user.role.value, organization_id=user.organization_id,
                metadata=user.metadata,
            )

    # --- Audit ---

    def _audit(self, user_id: str, action: str, resource: str, details: dict[str, Any] | None = None, success: bool = True) -> None:
        entry = AuditEntry(user_id=user_id, action=action, resource=resource, details=details or {}, success=success)
        self._audit_log.append(entry)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]
        # Write-through to DB
        if self._db and self._db.is_connected:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._db.audit(
                    action=action, user_id=user_id, resource=resource,
                    details=details or {}, success=success, reason=entry.reason,
                ))
            except RuntimeError:
                pass

    def get_audit_log(self, user_id: str = "", action: str = "", limit: int = 100) -> list[AuditEntry]:
        results = self._audit_log
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if action:
            results = [e for e in results if e.action == action]
        return results[-limit:]

    async def get_audit_log_async(self, user_id: str = "", action: str = "", limit: int = 100) -> list[dict]:
        """Get audit log from database (full history, not just in-memory)."""
        if self._db and self._db.is_connected:
            return await self._db.get_audit_log(user_id=user_id, action=action, limit=limit)
        # Fallback to in-memory
        return [e.model_dump() for e in self.get_audit_log(user_id, action, limit)]

    def audit_summary(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._audit_log),
            "total_users": len(self._users),
            "active_users": sum(1 for u in self._users.values() if u.is_active),
            "failed_auths": sum(1 for e in self._audit_log if not e.success),
        }
