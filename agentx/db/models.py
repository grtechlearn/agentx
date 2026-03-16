"""
AgentX - Database Models & Schema.
Defines all tables used by the framework.
Works with SQLite (zero-config) and PostgreSQL (production).
"""

from __future__ import annotations

# SQL schema that works for both SQLite and PostgreSQL
SCHEMA_VERSION = 1

SCHEMA = """
-- AgentX Schema v1

-- Users & Authentication
CREATE TABLE IF NOT EXISTS agentx_users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    email TEXT DEFAULT '',
    role TEXT NOT NULL DEFAULT 'user',
    organization_id TEXT DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 1,
    custom_permissions TEXT DEFAULT '[]',
    denied_permissions TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    api_key TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Sessions
CREATE TABLE IF NOT EXISTS agentx_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_history TEXT DEFAULT '[]',
    shared_state TEXT DEFAULT '{}',
    agent_results TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL DEFAULT NULL,
    FOREIGN KEY (user_id) REFERENCES agentx_users(id)
);

-- Agent Configurations (stored configs for registered agents)
CREATE TABLE IF NOT EXISTS agentx_agents (
    name TEXT PRIMARY KEY,
    role TEXT DEFAULT '',
    system_prompt TEXT DEFAULT '',
    model TEXT DEFAULT 'claude-sonnet-4-6',
    provider TEXT DEFAULT 'anthropic',
    temperature REAL DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 4096,
    tools TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Conversation History
CREATE TABLE IF NOT EXISTS agentx_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_name TEXT DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    data TEXT DEFAULT '{}',
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES agentx_sessions(id)
);

-- Long-term Memory
CREATE TABLE IF NOT EXISTS agentx_memory (
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    memory_type TEXT DEFAULT 'general',
    agent TEXT DEFAULT '',
    user_id TEXT DEFAULT '',
    importance REAL DEFAULT 0.5,
    ttl REAL DEFAULT NULL,
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (key, user_id)
);

-- RAG Documents (metadata only — vectors stored in vector DB)
CREATE TABLE IF NOT EXISTS agentx_documents (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    source TEXT DEFAULT '',
    technology TEXT DEFAULT '',
    topic TEXT DEFAULT '',
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    metadata TEXT DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Evaluation Results
CREATE TABLE IF NOT EXISTS agentx_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    user_id TEXT DEFAULT '',
    agent_name TEXT DEFAULT '',
    query TEXT DEFAULT '',
    response TEXT DEFAULT '',
    score REAL DEFAULT 0.0,
    faithfulness REAL DEFAULT 0.0,
    hallucination_detected INTEGER DEFAULT 0,
    evaluation_data TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);

-- Self-Learning Rules
CREATE TABLE IF NOT EXISTS agentx_learned_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL UNIQUE,
    response TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    times_used INTEGER DEFAULT 0,
    times_validated INTEGER DEFAULT 0,
    source TEXT DEFAULT 'auto',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Cost Tracking
CREATE TABLE IF NOT EXISTS agentx_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT DEFAULT '',
    agent_name TEXT DEFAULT '',
    model TEXT DEFAULT '',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    session_id TEXT DEFAULT '',
    created_at REAL NOT NULL
);

-- Audit Log
CREATE TABLE IF NOT EXISTS agentx_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT DEFAULT '',
    action TEXT NOT NULL,
    resource TEXT DEFAULT '',
    details TEXT DEFAULT '{}',
    ip_address TEXT DEFAULT '',
    success INTEGER NOT NULL DEFAULT 1,
    reason TEXT DEFAULT '',
    created_at REAL NOT NULL
);

-- Goals & Progress
CREATE TABLE IF NOT EXISTS agentx_goals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    target_data TEXT DEFAULT '{}',
    progress_data TEXT DEFAULT '{}',
    status TEXT DEFAULT 'active',
    streak_days INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    target_date REAL DEFAULT NULL,
    FOREIGN KEY (user_id) REFERENCES agentx_users(id)
);

-- Prompt Templates
CREATE TABLE IF NOT EXISTS agentx_prompts (
    name TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0',
    template TEXT NOT NULL,
    description TEXT DEFAULT '',
    variables TEXT DEFAULT '[]',
    model_hint TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    performance_data TEXT DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (name, version)
);

-- Schema Version Tracking
CREATE TABLE IF NOT EXISTS agentx_schema (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_user ON agentx_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON agentx_conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON agentx_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_user ON agentx_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_type ON agentx_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_evaluations_user ON agentx_evaluations(user_id);
CREATE INDEX IF NOT EXISTS idx_costs_user ON agentx_costs(user_id);
CREATE INDEX IF NOT EXISTS idx_costs_date ON agentx_costs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_user ON agentx_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON agentx_audit(action);
CREATE INDEX IF NOT EXISTS idx_goals_user ON agentx_goals(user_id);
"""
