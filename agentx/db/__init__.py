"""
AgentX - Database Layer.
Built-in persistence for SQLite (zero-config) and PostgreSQL (production).
"""

from .provider import (
    Database,
    DatabaseProvider,
    SQLiteProvider,
    PostgreSQLProvider,
    create_database,
)
from .models import SCHEMA, SCHEMA_VERSION

__all__ = [
    "Database",
    "DatabaseProvider",
    "SQLiteProvider",
    "PostgreSQLProvider",
    "create_database",
    "SCHEMA",
    "SCHEMA_VERSION",
]
