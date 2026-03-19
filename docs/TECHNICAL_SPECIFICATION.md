# AgentX Framework - Full Technical Engineering Specification

**Document Version**: 1.0
**Framework Version**: 0.6.0
**Date**: 2026-03-16
**Author**: GR Tech Learn (contact@aimediahub.in)
**License**: MIT
**Repository**: https://github.com/grtechlearn/agentx
**Python**: >=3.11

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Core Module](#3-core-module)
4. [Configuration System](#4-configuration-system)
5. [Database Layer](#5-database-layer)
6. [Memory System](#6-memory-system)
7. [RAG System](#7-rag-system)
8. [Pipeline & Ingestion](#8-pipeline--ingestion)
9. [Security Layer](#9-security-layer)
10. [Evaluation & Guardrails](#10-evaluation--guardrails)
11. [Scaling & Operations](#11-scaling--operations)
12. [Prompt Engineering](#12-prompt-engineering)
13. [Tools & MCP Integration](#13-tools--mcp-integration)
14. [Autonomous Daemon](#14-autonomous-daemon)
15. [Pre-built Agent Patterns](#15-pre-built-agent-patterns)
16. [Application Bootstrap](#16-application-bootstrap)
17. [API Reference](#17-api-reference)
18. [Deployment](#18-deployment)
19. [Dependencies](#19-dependencies)
20. [Project Use Cases](#20-project-use-cases)

---

## 1. Executive Summary

### 1.1 What is AgentX

AgentX is an open-source, enterprise-grade Multi-Agent System (MAS) framework for building production-ready AI agent applications. It provides a complete infrastructure layer so developers focus on domain-specific logic, not plumbing.

### 1.2 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Zero-config startup** | `AgentXApp()` + `await app.start()` — works immediately |
| **Progressive disclosure** | Simple by default, every component deeply configurable |
| **Plugin architecture** | Daemon, RAG, Security are independent layers |
| **Provider-agnostic** | Swap LLM providers, vector stores, databases without code changes |
| **Async-first** | All I/O is async (asyncio), sync wrappers available |
| **Production-ready** | Auth, RBAC, PII masking, cost caps, circuit breakers from day one |

### 1.3 Module Overview

```
agentx/
  core/           # Agent, Orchestrator, Message, Tool, LLM abstraction
  config/         # Centralized configuration (Pydantic models)
  db/             # SQLite + PostgreSQL (async, auto-migrate)
  memory/         # Short-term (in-memory) + Long-term (DB-backed)
  rag/            # Hybrid search, BM25, cross-encoder, semantic cache
  pipeline/       # Ingestion, PII, validation, knowledge freshness
  security/       # JWT auth, RBAC, injection guard, content moderation
  evaluation/     # RAGAS metrics, hallucination guard, cost tracking
  scaling/        # Model router, circuit breaker, task queue, tracing
  prompts/        # Template manager, context fitting, response cache
  tools/          # MCP integration, built-in tools
  agents/         # Pre-built agent patterns (router, guardrail, RAG, etc.)
  daemon/         # 24/7 runner, scheduler, API server, file watcher
  app.py          # One-line bootstrap — wires everything together
```

---

## 2. System Architecture

### 2.1 Layer Diagram

```
 External Clients
       |
       v
+----------------------------------------------+
| DAEMON LAYER (optional plugin)               |
| +----------+ +----------+ +---------------+ |
| | HTTP/WS  | | Job      | | File & MQ     | |
| | API      | | Scheduler| | Watchers      | |
| | Server   | | (cron)   | |               | |
| +----------+ +----------+ +---------------+ |
| +------------------------------------------+ |
| | Watchdog (auto-restart, health monitor)  | |
| +------------------------------------------+ |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
| APPLICATION LAYER (AgentXApp)                |
| +----------+ +----------+ +---------------+ |
| | Orches-  | | Auth     | | Cost          | |
| | trator   | | Gateway  | | Tracker       | |
| +----------+ +----------+ +---------------+ |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
| AGENT LAYER                                  |
| +----------+ +----------+ +---------------+ |
| | Router   | | Guardrail| | RAG Agent     | |
| | Agent    | | Agent    | |               | |
| +----------+ +----------+ +---------------+ |
| | Custom agents (your domain logic)        | |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
| INFRASTRUCTURE LAYER                         |
| +------+ +-----+ +--------+ +--------+     |
| | RAG  | | LLM | | Memory | | Tools  |     |
| +------+ +-----+ +--------+ +--------+     |
| +------+ +-----+ +--------+ +--------+     |
| |Trace | |Queue| | Cache  | | MCP    |     |
| +------+ +-----+ +--------+ +--------+     |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
| DATA LAYER                                   |
| +----------+ +----------+ +---------------+ |
| | SQLite / | | Vector   | | Redis         | |
| | Postgres | | Store    | | (optional)    | |
| +----------+ +----------+ +---------------+ |
+----------------------------------------------+
```

### 2.2 Data Flow

```
User Request
  -> API Server (auth check)
  -> Injection Guard (sanitize)
  -> Content Moderator (safety check)
  -> PII Masker (mask sensitive data)
  -> Orchestrator (route to agent)
  -> Agent (process with LLM + tools)
  -> Hallucination Guard (verify grounding)
  -> PII Masker (unmask)
  -> Content Moderator (output check)
  -> Response to user
  -> Cost Tracker (log spend)
  -> Query Analytics (log quality)
  -> Self-Learner (cache pattern)
```

### 2.3 Concurrency Model

- **Event loop**: Single-threaded asyncio (Python 3.11+)
- **Task queue**: N async workers (default: 4) for background processing
- **I/O**: All database, LLM, and network calls are async
- **Thread pool**: Used for CPU-bound operations (embeddings, BM25 scoring)
- **Signal handling**: SIGINT/SIGTERM for graceful shutdown

---

## 3. Core Module

### 3.1 Message System

**File**: `agentx/core/message.py`

```python
class MessageType(str, Enum):
    TASK = "task"           # Request to perform work
    RESPONSE = "response"   # Result of work
    ERROR = "error"         # Error occurred
    EVENT = "event"         # System event notification
    HANDOFF = "handoff"     # Transfer to another agent
    BROADCAST = "broadcast" # Message to all agents

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class AgentMessage(BaseModel):
    id: str                          # Auto-generated UUID
    type: MessageType = TASK
    sender: str = ""                 # Source agent name
    receiver: str = ""               # Target agent name
    content: str = ""                # Text payload
    data: dict[str, Any] = {}        # Structured payload
    priority: Priority = NORMAL
    parent_id: str | None = None     # For threading
    created_at: datetime             # UTC timestamp
    metadata: dict[str, Any] = {}    # Session ID, user ID, etc.

    def reply(content, data=None) -> AgentMessage
    def error(error_msg) -> AgentMessage
    def handoff(target_agent, content, data=None) -> AgentMessage
```

### 3.2 Agent System

**File**: `agentx/core/agent.py`

```python
class AgentConfig(BaseModel):
    name: str                        # Unique agent identifier
    role: str = ""                   # Human-readable role
    system_prompt: str = ""          # LLM system instruction
    model: str = "claude-sonnet-4-6" # Default model
    provider: str = "anthropic"      # LLM provider
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 2
    tools_enabled: bool = True
    metadata: dict[str, Any] = {}

class AgentState(BaseModel):
    status: str = "idle"             # idle|running|waiting|error|completed
    current_task: str = ""
    messages_processed: int = 0
    errors: list[str] = []
    results: list[Any] = []

class BaseAgent(ABC):
    """Every agent inherits this. Lifecycle: idle -> running -> completed/error."""

    Properties:
      name -> str
      llm -> BaseLLMProvider          # Lazy-initialized

    Methods:
      register_tool(tool: BaseTool)
      get_tool_schemas() -> list[dict]
      execute_tool(tool_name, **kwargs) -> ToolResult
      think(prompt, context=None, system=None, use_tools=False) -> LLMResponse
      think_json(prompt, schema=None, context=None, system=None) -> dict
      process(message, context) -> AgentMessage       # ABSTRACT - your logic
      run(message, context) -> AgentMessage            # Calls process() with hooks

    Lifecycle Hooks:
      on_start(context)
      on_complete(context, result)
      on_error(context, error)

class SimpleAgent(BaseAgent):
    """Minimal agent — sends prompt to LLM, returns response."""
```

### 3.3 Orchestrator

**File**: `agentx/core/orchestrator.py`

```python
class Orchestrator:
    """Central coordinator for multi-agent routing and execution."""

    Agent Management:
      register(agent) -> self
      register_many(*agents) -> self
      get_agent(name) -> BaseAgent | None
      set_fallback(agent_name) -> self

    Routing:
      add_route(agent_name, condition=None, priority=0) -> self
      @route_to(agent_name) decorator

    Execution:
      dispatch(message, context=None) -> AgentMessage    # Route + execute
      send(content, session_id="", user_id="") -> AgentMessage  # Convenience
      run_pipeline(pipeline_name, message, context=None) -> AgentMessage
      run_parallel(agent_names, message, context=None) -> dict[str, AgentMessage]

    Sessions:
      get_session(session_id) -> AgentContext | None
      clear_session(session_id)

    Events:
      @on(event) decorator
      emit(event, **kwargs)
```

**Routing Priority**:
1. Direct routing (message.receiver is set)
2. Rule-based routing (conditions evaluated by priority, highest first)
3. Fallback agent

**Handoff Behavior**: When an agent returns `MessageType.HANDOFF`, the orchestrator automatically re-dispatches to the target agent.

### 3.4 LLM Abstraction

**File**: `agentx/core/llm.py`

```python
class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key: str = ""             # Empty = use env var
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    extra: dict[str, Any] = {}

class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}     # input_tokens, output_tokens
    model: str = ""
    raw: Any = None                # Provider-specific raw response

class BaseLLMProvider(ABC):
    generate(messages, system="", tools=None, temperature=None, max_tokens=None) -> LLMResponse
    generate_json(messages, system="", schema=None) -> dict[str, Any]

# Implementations:
class AnthropicProvider(BaseLLMProvider)   # Uses anthropic SDK
class OpenAIProvider(BaseLLMProvider)      # Uses openai SDK

# Factory:
create_llm(config=None, **kwargs) -> BaseLLMProvider
# Supported: "anthropic", "claude", "openai", "gpt"
```

### 3.5 Tool System

**File**: `agentx/core/tool.py`

```python
class ToolResult(BaseModel):
    success: bool = True
    data: Any = None
    error: str | None = None

    @classmethod ok(data=None) -> ToolResult
    @classmethod fail(error: str) -> ToolResult

class BaseTool(ABC):
    name: str
    description: str
    execute(**kwargs) -> ToolResult      # ABSTRACT
    get_schema() -> dict[str, Any]       # JSON Schema for tool calling

class FunctionTool(BaseTool):
    """Wraps any Python function as a tool."""
    __init__(fn, name="", description="")

@tool(name="", description="")           # Decorator
```

### 3.6 Context

**File**: `agentx/core/context.py`

```python
class AgentContext(BaseModel):
    session_id: str = ""
    user_id: str = ""
    conversation_history: list[dict[str, str]] = []
    shared_state: dict[str, Any] = {}    # Shared across agents in pipeline
    agent_results: dict[str, Any] = {}   # Results from each agent
    metadata: dict[str, Any] = {}

    add_message(role, content)
    set(key, value)
    get(key, default=None) -> Any
    store_result(agent_name, result)
    get_result(agent_name, default=None) -> Any
    get_last_n_messages(n) -> list[dict]
```

---

## 4. Configuration System

**File**: `agentx/config/settings.py`

### 4.1 Master Configuration

```python
class AgentXConfig(BaseModel):
    env: Environment = DEVELOPMENT    # development|staging|production
    app_name: str = "AgentX"
    version: str = "0.2.0"
    debug: bool = False

    database: DatabaseConfig          # SQLite or PostgreSQL
    llm: LLMConfig                    # Multi-layer LLM setup
    budget: LLMBudget                 # Cost limits
    governance: DataGovernance        # PII, retention, encryption
    metrics: SystemMetrics            # Latency targets, accuracy thresholds
    cache: CacheConfig                # Semantic + response caching
    self_learning: SelfLearningConfig # Pattern learning, fine-tune
    moderation: ContentModerationConfig # Content safety

    # Paths
    data_dir: str = "./data"
    memory_dir: str = "./data/memory"
    training_dir: str = "./data/training"
    logs_dir: str = "./logs"

    # Presets:
    @classmethod from_env()           # From environment variables
    @classmethod development()        # SQLite, single model, debug=True
    @classmethod production(dsn)      # PostgreSQL, cost-optimized, strict moderation
```

### 4.2 Multi-Layer LLM Configuration

```python
class LLMConfig(BaseModel):
    default: LLMLayerConfig           # Fallback for all layers
    agent: LLMLayerConfig | None      # Main conversation (quality)
    evaluation: LLMLayerConfig | None # Scoring, hallucination check (precise)
    routing: LLMLayerConfig | None    # Classification (fast, cheap)
    embedding: LLMLayerConfig | None  # Vector embeddings (specialized)
    summary: LLMLayerConfig | None    # Summarization (mid-tier)
    fallback: LLMLayerConfig | None   # Budget exceeded / primary fails

    get_layer(layer: str) -> LLMLayerConfig

    # Presets:
    @classmethod single()             # Same model everywhere
    @classmethod cost_optimized()     # Haiku for routing, Sonnet for agent
    @classmethod quality_first()      # Opus for agent+eval, Sonnet for rest
    @classmethod openai()             # GPT-4o everywhere
    @classmethod mixed()              # Claude for gen, OpenAI for embed
```

### 4.3 Cost Budget

```python
class LLMBudget(BaseModel):
    max_monthly_spend_usd: float = 100.0
    max_daily_spend_usd: float = 10.0
    max_tokens_per_request: int = 4096
    max_requests_per_minute: int = 60
    max_requests_per_user_per_day: int = 100
    prefer_cheaper_model: bool = True
    fallback_to_cache: bool = True
    warn_at_percentage: float = 80.0
```

### 4.4 Data Governance

```python
class DataGovernance(BaseModel):
    # PII
    detect_pii: bool = True
    pii_fields_to_redact: list[str] = [
        "email", "phone", "aadhaar", "pan", "ssn",
        "credit_card", "address", "date_of_birth",
        "passport", "bank_account"
    ]
    redaction_method: str = "mask"       # mask, hash, remove

    # Retention
    conversation_retention_days: int = 90
    embedding_retention_days: int = 365
    user_data_retention_days: int = 730

    # Security
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    log_data_access: bool = True
    allow_data_export: bool = True
    allow_data_deletion: bool = True     # GDPR right-to-be-forgotten
```

### 4.5 System Metrics

```python
class SystemMetrics(BaseModel):
    # Latency targets
    max_response_time_ms: int = 5000
    max_rag_retrieval_ms: int = 1000
    max_embedding_time_ms: int = 500

    # Throughput
    target_requests_per_second: int = 50
    max_concurrent_agents: int = 10
    max_concurrent_sessions: int = 100

    # Quality
    min_rag_relevance_score: float = 0.7
    min_answer_confidence: float = 0.6
    max_hallucination_tolerance: float = 0.1

    # Freshness
    knowledge_refresh_interval_hours: int = 24
    stale_data_warning_days: int = 30
    auto_refresh_enabled: bool = True

    # Cost
    cost_per_query_target_usd: float = 0.01
    track_cost_per_agent: bool = True
    track_cost_per_user: bool = True
```

### 4.6 Content Moderation Configuration

```python
class ContentModerationConfig(BaseModel):
    enabled: bool = True
    severity_threshold: str = "low"      # low|medium|high|critical
    default_action: str = "block"        # block|warn|redact|log

    # Per-category toggles
    block_profanity: bool = True
    block_sexual: bool = True
    block_abuse: bool = True
    block_violence: bool = False         # warn by default
    block_self_harm: bool = True
    block_drugs: bool = False

    # Custom rules
    custom_blocked_words: list[str] = []
    custom_blocked_patterns: list[str] = []
    whitelist_words: list[str] = []

    # Behavior
    check_input: bool = True
    check_output: bool = True
    log_violations: bool = True
    max_violations_before_ban: int = 0

    # Vulnerability scanning
    vulnerability_scanning: bool = True
    block_credential_exposure: bool = True
    block_code_injection: bool = True
    block_unsafe_urls: bool = True

    # Presets:
    @classmethod strict()      # All categories blocked
    @classmethod moderate()    # Sexual/abuse blocked, violence warned
    @classmethod disabled()    # No moderation
```

### 4.7 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTX_ENV` | `development` | Environment |
| `AGENTX_APP_NAME` | `AgentX` | Application name |
| `AGENTX_DEBUG` | `false` | Debug mode |
| `AGENTX_DB_PROVIDER` | `sqlite` | Database provider |
| `AGENTX_DB_PATH` | `""` | SQLite path |
| `AGENTX_DATABASE_URL` | `""` | PostgreSQL DSN |
| `AGENTX_LLM_PROVIDER` | `anthropic` | LLM provider |
| `AGENTX_LLM_MODEL` | `claude-sonnet-4-6` | Default model |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API key |
| `OPENAI_API_KEY` | `""` | OpenAI API key |
| `AGENTX_PORT` | `8080` | Daemon server port |
| `AGENTX_HOST` | `0.0.0.0` | Daemon server host |
| `AGENTX_API_KEY` | `""` | API authentication key |
| `AGENTX_REDIS_URL` | `""` | Redis connection URL |
| `AGENTX_LOG_LEVEL` | `INFO` | Logging level |
| `AGENTX_LOG_FILE` | `""` | Log file path |
| `AGENTX_PID_FILE` | `""` | PID file path |
| `AGENTX_SERVER` | `true` | Enable HTTP server |
| `AGENTX_SCHEDULER` | `true` | Enable scheduler |
| `AGENTX_WATCHER` | `false` | Enable file watcher |
| `AGENTX_MQ` | `false` | Enable message queue |
| `AGENTX_WATCHDOG` | `true` | Enable watchdog |

---

## 5. Database Layer

**File**: `agentx/db/provider.py`, `agentx/db/models.py`

### 5.1 Supported Providers

| Provider | Use Case | Connection |
|----------|----------|------------|
| **SQLite** | Development, testing, single-server | `aiosqlite`, file or `:memory:` |
| **PostgreSQL** | Production, multi-server | `asyncpg`, connection pooling |

### 5.2 Schema (13 Tables)

```sql
-- 1. Users
CREATE TABLE agentx_users (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    role TEXT DEFAULT 'user',
    organization_id TEXT,
    is_active INTEGER DEFAULT 1,
    custom_permissions TEXT,      -- JSON array
    denied_permissions TEXT,      -- JSON array
    metadata TEXT,                -- JSON object
    api_key TEXT,
    created_at REAL,
    updated_at REAL
);

-- 2. Sessions
CREATE TABLE agentx_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    conversation_history TEXT DEFAULT '[]',
    shared_state TEXT DEFAULT '{}',
    agent_results TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    created_at REAL,
    updated_at REAL,
    expires_at REAL
);

-- 3. Agents (registered agent configurations)
CREATE TABLE agentx_agents (
    name TEXT PRIMARY KEY,
    role TEXT,
    system_prompt TEXT,
    model TEXT DEFAULT 'claude-sonnet-4-6',
    provider TEXT DEFAULT 'anthropic',
    temperature REAL DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 4096,
    tools TEXT,                   -- JSON array
    metadata TEXT,                -- JSON object
    is_active INTEGER DEFAULT 1,
    created_at REAL,
    updated_at REAL
);

-- 4. Conversations (full history)
CREATE TABLE agentx_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_id TEXT,
    agent_name TEXT,
    role TEXT,                    -- user|assistant|system
    content TEXT,
    data TEXT DEFAULT '{}',
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    created_at REAL
);

-- 5. Memory (key-value with TTL)
CREATE TABLE agentx_memory (
    key TEXT NOT NULL,
    user_id TEXT NOT NULL,
    value TEXT,
    memory_type TEXT DEFAULT 'general',
    agent TEXT,
    importance REAL DEFAULT 0.5,
    ttl REAL,
    metadata TEXT,
    created_at REAL,
    updated_at REAL,
    PRIMARY KEY (key, user_id)
);

-- 6. Documents (RAG metadata)
CREATE TABLE agentx_documents (
    id TEXT PRIMARY KEY,
    content_hash TEXT,
    source TEXT,
    technology TEXT,
    topic TEXT,
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    metadata TEXT,
    is_active INTEGER DEFAULT 1,
    created_at REAL,
    updated_at REAL
);

-- 7. Evaluations
CREATE TABLE agentx_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_id TEXT,
    agent_name TEXT,
    query TEXT,
    response TEXT,
    score REAL DEFAULT 0.0,
    faithfulness REAL DEFAULT 0.0,
    hallucination_detected INTEGER DEFAULT 0,
    evaluation_data TEXT,
    created_at REAL
);

-- 8. Learned Rules (self-learning patterns)
CREATE TABLE agentx_learned_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT UNIQUE,
    response TEXT,
    confidence REAL DEFAULT 0.0,
    times_used INTEGER DEFAULT 0,
    times_validated INTEGER DEFAULT 0,
    source TEXT DEFAULT 'auto',
    is_active INTEGER DEFAULT 1,
    created_at REAL,
    updated_at REAL
);

-- 9. Cost Logs
CREATE TABLE agentx_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    agent_name TEXT,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    session_id TEXT,
    created_at REAL
);

-- 10. Audit Log
CREATE TABLE agentx_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    action TEXT,
    resource TEXT,
    details TEXT,
    ip_address TEXT,
    success INTEGER DEFAULT 1,
    reason TEXT,
    created_at REAL
);

-- 11. Goals
CREATE TABLE agentx_goals (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    title TEXT,
    description TEXT,
    target_data TEXT,
    progress_data TEXT,
    status TEXT DEFAULT 'active',
    streak_days INTEGER DEFAULT 0,
    target_date REAL,
    created_at REAL,
    updated_at REAL
);

-- 12. Prompt Templates
CREATE TABLE agentx_prompts (
    name TEXT NOT NULL,
    version TEXT DEFAULT '1.0',
    template TEXT,
    description TEXT,
    variables TEXT,
    model_hint TEXT,
    tags TEXT,
    performance_data TEXT,
    is_active INTEGER DEFAULT 1,
    created_at REAL,
    updated_at REAL,
    PRIMARY KEY (name, version)
);

-- 13. Schema Version
CREATE TABLE agentx_schema (
    version INTEGER PRIMARY KEY,
    applied_at REAL
);
```

### 5.3 Database API

```python
class Database:
    # Connection
    connect() -> None
    close() -> None
    is_connected: bool

    # Raw
    execute(query, params=()) -> None
    fetch_one(query, params=()) -> dict | None
    fetch_all(query, params=()) -> list[dict]

    # CRUD (all async)
    save_user(user_data) / get_user(id) / list_users()
    create_session() / get_session(id) / update_session()
    add_message() / get_conversation(session_id)
    save_memory() / get_memory() / search_memory() / delete_memory()
    save_agent() / get_agent() / list_agents()
    save_evaluation() / get_evaluations()
    track_cost() / get_cost_summary()
    audit() / get_audit_log()
    save_goal() / get_user_goals() / update_goal_progress()
    save_rule() / find_rule() / increment_rule_usage()
    save_prompt() / get_prompt()
```

---

## 6. Memory System

**File**: `agentx/memory/store.py`

### 6.1 Memory Types

| Type | Storage | Persistence | Use Case |
|------|---------|-------------|----------|
| **ShortTermMemory** | In-memory dict | Session only | Conversation context, temp state |
| **LongTermMemory** | Database | Persistent | Facts, preferences, learned patterns |

### 6.2 Memory Entry

```python
class MemoryEntry(BaseModel):
    key: str
    value: Any
    memory_type: str = "general"     # general|fact|preference|conversation|skill
    agent: str = ""                  # Which agent stored this
    user_id: str = ""
    timestamp: float
    ttl: float | None = None         # Auto-expire after N seconds
    metadata: dict[str, Any] = {}
    importance: float = 0.5          # 0.0 (trivial) to 1.0 (critical)

    is_expired() -> bool
```

### 6.3 Unified Memory API

```python
class AgentMemory:
    remember(key, value, long_term=False, **kwargs) -> None
    recall(key) -> Any
    search(query, limit=10, **filters) -> list[MemoryEntry]
    forget(key) -> None
    clear_session() -> None
```

---

## 7. RAG System

**File**: `agentx/rag/engine.py`, `agentx/rag/retrieval.py`, `agentx/rag/stores.py`

### 7.1 RAG Engine

```python
class RAGEngine:
    # Ingestion
    ingest(text, metadata=None) -> int             # Returns chunk count
    ingest_documents(documents) -> int

    # Retrieval strategies
    search(query, limit=5, filters=None) -> list[Document]           # Semantic only
    hybrid_search(query, limit=5, keyword_weight=0.3) -> list[Document]  # BM25 + semantic
    decomposed_search(query, limit=5) -> list[Document]              # Split complex queries
    search_with_rerank(query, limit=5, initial_limit=20) -> list[Document]  # Cross-encoder
    search_with_rewrite(query, limit=5, strategy="expand") -> list[Document]  # Query rewrite

    # Context building
    get_context(query, limit=5, strategy="hybrid", max_context_length=4000) -> str
```

**Strategy Options for `get_context()`**:
- `"semantic"` — Pure vector similarity
- `"hybrid"` — BM25 + semantic (default)
- `"decomposed"` — Query decomposition for complex questions
- `"rerank"` — Retrieve 20, re-rank to top 5
- `"rewrite"` — Rewrite query before searching

### 7.2 BM25 Index

```python
class BM25Index:
    """True BM25 scoring — IDF + TF with saturation and length normalization."""
    __init__(k1=1.5, b=0.75)
    add_documents(documents) -> None
    search(query, limit=10, filters=None) -> list[Document]
    clear() -> None
    size: int
```

**BM25 Formula**:
```
score(q, d) = SUM[ IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl)) ]
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

### 7.3 Cross-Encoder Re-ranker

```python
class CrossEncoderReranker:
    """Re-ranks semantic results for higher precision."""
    __init__(model="cross-encoder/ms-marco-MiniLM-L-6-v2", provider="local")
    rerank(query, documents, limit=5) -> list[Document]
```

| Provider | Backend | Latency | Quality |
|----------|---------|---------|---------|
| `local` | sentence-transformers | ~50ms | Good |
| `cohere` | Cohere Rerank API | ~200ms | Excellent |
| `voyage` | Voyage AI API | ~200ms | Excellent |
| `llm` | LLM-based scoring | ~1000ms | Very good |

### 7.4 Semantic Cache

```python
class SemanticCache:
    """Cache query results by embedding similarity."""
    __init__(embedder=None, similarity_threshold=0.92, max_size=5000, ttl_seconds=3600)
    get(query, system="") -> str | None
    set(query, response, system="", ttl=3600) -> None
    invalidate(query) -> None
    clear() -> None
    stats() -> dict     # hit_rate, size, avg_latency
```

### 7.5 Query Rewriter

```python
class QueryRewriter:
    rewrite(query, strategy="expand", num_rewrites=3) -> list[str]
```

| Strategy | Description | Example |
|----------|-------------|---------|
| `expand` | Add related terms | "Python errors" -> "Python exception handling debugging traceback" |
| `rephrase` | Rephrase for clarity | "Why broken?" -> "What causes the application to fail?" |
| `hyde` | Generate hypothetical answer | Query -> Fake document that matches |
| `stepback` | Broader question | "Fix React useEffect loop" -> "How does React useEffect work?" |
| `multi` | Combine multiple strategies | Generates from expand + rephrase + stepback |

### 7.6 Vector Stores

| Store | Type | Use Case |
|-------|------|----------|
| `QdrantVectorStore` | External service | Production (cluster mode) |
| `ChromaVectorStore` | Embedded/client | Development, small-medium |
| `PineconeVectorStore` | Managed cloud | Serverless, large scale |

### 7.7 Embedders

| Embedder | Provider | Default Model | Dimensions |
|----------|----------|---------------|------------|
| `AnthropicEmbedder` | Voyage AI | `voyage-3` | 1024 |
| `OpenAIEmbedder` | OpenAI | `text-embedding-3-small` | 1536 |
| `LocalEmbedder` | sentence-transformers | `all-MiniLM-L6-v2` | 384 |

---

## 8. Pipeline & Ingestion

**File**: `agentx/pipeline/ingestion.py`, `agentx/pipeline/knowledge.py`

### 8.1 Ingestion Pipeline

```python
class IngestionPipeline:
    """Ingest documents through PII detection, validation, cleaning, and RAG indexing."""
    __init__(rag_engine, pii_detector=None, validator=None, cleaner=None)
    add_loader(loader) -> self
    run(detect_pii=True, validate=True) -> dict[str, int]
```

### 8.2 PII Detection

```python
class PIIDetector:
    """Detects and redacts personally identifiable information."""
    __init__(fields_to_detect=None, method="mask")
    detect(text) -> list[dict]    # [{field, value, start, end}]
    redact(text) -> str           # "email@test.com" -> "[EMAIL_REDACTED]"
    has_pii(text) -> bool
```

**Supported PII Types**: email, phone, SSN, credit_card, aadhaar, PAN, IP address

### 8.3 Knowledge Freshness Manager

```python
class KnowledgeFreshnessManager:
    """Tracks knowledge source staleness and triggers auto-refresh."""
    __init__(stale_warning_days=30, auto_refresh=False, db=None)
    register_source(source: KnowledgeSource)
    get_stale_sources() -> list[KnowledgeSource]
    check_freshness(source_id) -> dict
    refresh_source(source_id, refresh_fn) -> dict    # Calls refresh_fn, updates hash
    mark_refreshed(source_id, new_hash="", doc_count=0)
    report() -> dict
```

### 8.4 Retention Enforcer (GDPR)

```python
class RetentionEnforcer:
    """Enforce data lifecycle and GDPR right-to-be-forgotten."""
    __init__(policy=None, db=None)
    enforce() -> dict            # Delete expired data per policy
    delete_user_data(user_id) -> dict   # GDPR: archive then delete ALL user data
```

### 8.5 Fine-Tune Pipeline

```python
class FineTunePipeline:
    """Collect, curate, and export training data for model fine-tuning."""
    collect(query, response, system_prompt="", quality_score=0.0)
    curate() -> dict             # Filter low-quality, deduplicate
    export(format="jsonl") -> str  # Export to JSONL/JSON/CSV file
    stats() -> dict
```

### 8.6 Runtime PII Masker

```python
class RuntimePIIMasker:
    """Mask PII before sending to LLM, unmask after response."""
    mask(text) -> str            # "john@test.com" -> "[PII_EMAIL_1]"
    unmask(text) -> str          # "[PII_EMAIL_1]" -> "john@test.com"
    mask_contexts(contexts) -> list[str]
```

### 8.7 Citation Manager

```python
class CitationManager:
    """Source reliability scoring and citation formatting."""
    register_source(source_id, name="", authority=0.5, recency=0.5, accuracy=0.5)
    get_reliability(source_id) -> float   # Weighted score
    weight_contexts(contexts, source_ids, scores=None) -> list[tuple[str, float]]
    record_feedback(source_id, helpful: bool)
    format_citations(source_ids, format="inline") -> str
```

**Reliability Formula**: `overall = authority * 0.4 + recency * 0.3 + accuracy * 0.3`

---

## 9. Security Layer

**File**: `agentx/security/rbac.py`, `agentx/security/auth.py`, `agentx/security/moderation.py`

### 9.1 RBAC (Role-Based Access Control)

```python
class Role(str, Enum):
    ADMIN = "admin"      # All permissions
    MANAGER = "manager"  # Agent + data + RAG + team analytics
    USER = "user"        # Agent run + data read + RAG search
    VIEWER = "viewer"    # Data read + RAG search only
    API = "api"          # Agent run + data read + RAG search

class Permission(str, Enum):
    # 16 granular permissions across 5 domains
    AGENT_RUN, AGENT_CREATE, AGENT_DELETE, AGENT_CONFIG
    DATA_READ, DATA_WRITE, DATA_DELETE, DATA_EXPORT
    RAG_SEARCH, RAG_INGEST, RAG_DELETE
    ADMIN_USERS, ADMIN_SETTINGS, ADMIN_BILLING, ADMIN_AUDIT
    ANALYTICS_SELF, ANALYTICS_TEAM, ANALYTICS_ALL
```

**Default Permission Matrix**:

| Permission | ADMIN | MANAGER | USER | VIEWER | API |
|------------|-------|---------|------|--------|-----|
| AGENT_RUN | Y | Y | Y | - | Y |
| AGENT_CREATE | Y | Y | - | - | - |
| AGENT_DELETE | Y | - | - | - | - |
| AGENT_CONFIG | Y | Y | - | - | - |
| DATA_READ | Y | Y | Y | Y | Y |
| DATA_WRITE | Y | Y | - | - | - |
| DATA_DELETE | Y | - | - | - | - |
| DATA_EXPORT | Y | Y | - | - | - |
| RAG_SEARCH | Y | Y | Y | Y | Y |
| RAG_INGEST | Y | Y | - | - | - |
| RAG_DELETE | Y | - | - | - | - |
| ADMIN_* | Y | - | - | - | - |
| ANALYTICS_SELF | Y | Y | Y | Y | - |
| ANALYTICS_TEAM | Y | Y | - | - | - |
| ANALYTICS_ALL | Y | - | - | - | - |

### 9.2 JWT Authentication

```python
class AuthGateway:
    __init__(secret_key="agentx-jwt-secret", access_token_ttl=3600, refresh_token_ttl=604800)

    create_token(user_id, role="user", permissions=None, namespaces=None) -> str
    create_token_pair(...) -> {"access_token": str, "refresh_token": str}
    validate_token(token) -> JWTToken | None
    refresh_token(refresh_token_str) -> str | None
    revoke_token(token_str) -> bool
    authenticate(token, check_injection=True, query="") -> AuthResult
```

**Token Payload (JWTToken)**:
```json
{
  "sub": "user-123",
  "role": "user",
  "permissions": ["agent:run", "data:read"],
  "namespaces": ["general", "project-x"],
  "exp": 1710600000,
  "iat": 1710596400,
  "token_type": "access",
  "jti": "abc123",
  "org_id": "org-1"
}
```

### 9.3 Injection Guard

```python
class InjectionGuard:
    """Multi-layer injection detection — 30+ regex patterns across 5 attack categories."""
    __init__(block_prompt_injection=True, block_sql_injection=True,
             block_xss=True, block_command_injection=True, block_path_traversal=True)
    check(input_text) -> InjectionResult
    sanitize(input_text) -> str
```

**Attack Categories**:
1. **Prompt Injection** — "ignore previous instructions", "you are now", role override attempts
2. **SQL Injection** — `OR 1=1`, `UNION SELECT`, `DROP TABLE`, `--` comments
3. **XSS** — `<script>`, `javascript:`, `onerror=`, event handlers
4. **Command Injection** — `; rm -rf`, backtick execution, pipe chains
5. **Path Traversal** — `../`, `/etc/passwd`, `C:\Windows`

### 9.4 Content Moderation

```python
class ContentModerator:
    check(text, user_id="") -> ModerationResult
    is_user_banned(user_id) -> bool
    add_words(category, words) -> None
    remove_words(category, words) -> None
    add_whitelist(words) -> None
    add_pattern(category, pattern) -> None
```

**Moderation Categories** (each independently configurable):

| Category | Default Action | Built-in Words |
|----------|---------------|----------------|
| `profanity` | BLOCK | Common profanity list |
| `sexual` | BLOCK | Sexual content terms |
| `abuse` | BLOCK | Hate speech, slurs |
| `violence` | WARN | Violent content |
| `self_harm` | BLOCK | Self-harm references |
| `drugs` | WARN | Drug references |
| `custom` | BLOCK | User-defined words |

**Actions**: `BLOCK` (reject), `WARN` (flag), `REDACT` (replace), `LOG` (allow + record)

### 9.5 Vulnerability Scanner

```python
class VulnerabilityScanner:
    scan(text) -> VulnerabilityResult
```

**Scan Categories**:
1. **Credential Exposure** — API keys, passwords, AWS secrets, tokens
2. **Code Injection** — eval(), exec(), system(), subprocess, SQL in code
3. **Unsafe URLs** — data: URIs, javascript: URLs, file:// paths
4. **Serialization** — pickle.loads(), yaml.load(), deserialization attacks
5. **Information Disclosure** — Stack traces, internal IPs, debug output
6. **Insecure Patterns** — MD5 hashing, HTTP (not HTTPS), no-verify SSL

### 9.6 Namespace Manager

```python
class NamespaceManager:
    """RBAC-scoped vector store access — users can only search authorized namespaces."""
    assign_namespaces(user_id, namespaces) -> None
    get_allowed_namespaces(user_id, role="user") -> set[str]
    register_namespace(namespace) -> None
    build_filters(user_id, role="user") -> dict
    check_access(user_id, namespace, role="user") -> bool
```

---

## 10. Evaluation & Guardrails

**File**: `agentx/evaluation/ragas.py`, `agentx/evaluation/guardrails.py`, `agentx/evaluation/metrics.py`

### 10.1 RAGAS Evaluator

```python
class RAGASEvaluator:
    evaluate(query, answer, contexts, ground_truth="") -> RAGASResult

class RAGASResult(BaseModel):
    faithfulness: float = 0.0       # Is answer grounded in contexts?
    answer_relevance: float = 0.0   # Does answer address the query?
    context_relevance: float = 0.0  # Are retrieved contexts relevant?
    context_recall: float = 0.0     # Was all needed info retrieved?
    mrr: float = 0.0                # Mean Reciprocal Rank
    ndcg: float = 0.0              # Normalized Discounted Cumulative Gain
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    overall_score: float = 0.0
```

### 10.2 Hallucination Guard (Runtime)

```python
class HallucinationGuard:
    """Wraps LLM generation with pre-check -> generate -> post-check -> confidence gate."""

    generate_grounded(query, contexts, system_prompt="", max_tokens=4096) -> GroundedResponse

class GroundingConfig(BaseModel):
    require_sources: bool = True
    min_confidence: float = 0.6
    max_hallucination_tolerance: float = 0.2
    require_citations: bool = True
    min_sources_cited: int = 1
    retry_on_low_confidence: bool = True
    max_retries: int = 2
    escalate_on_failure: bool = True
    enforce_idk: bool = True         # Force "I don't know" when uncertain

class GroundedResponse(BaseModel):
    content: str = ""
    grounded: bool = False
    confidence: float = 0.0
    sources_cited: int = 0
    citations: list[dict] = []
    claims: list[dict] = []
    warnings: list[str] = []
    retries: int = 0
    escalated: bool = False
```

**Grounding Flow**:
```
1. Pre-check: Is there enough context to answer?
   -> If not, return "I don't have enough information"

2. Generate: Send query + contexts + grounding instructions to LLM
   -> Instructions enforce citation, source attribution

3. Post-check: Verify response against contexts
   -> Count citations, extract claims, check coverage

4. Confidence gate: Is confidence >= min_confidence?
   -> If not and retry_on_low_confidence: retry (up to max_retries)
   -> If exhausted: escalate or return with warning
```

### 10.3 Query Analytics

```python
class QueryAnalytics:
    log_query(query, answer, confidence, contexts_used, latency_ms, model, ...)
    log_feedback(query_id, thumbs_up, correction, bad_chunks)
    get_report(days=7) -> dict
    get_low_confidence_queries(threshold=0.5) -> list[dict]
    get_negative_feedback() -> list[dict]
```

### 10.4 Cost Tracking

```python
class CostTracker:
    track(model, input_tokens, output_tokens, key="global") -> float  # Returns cost USD
    get_cost(key="global") -> float
    is_over_budget(budget_usd) -> bool
    report() -> dict
```

**Model Pricing** (per 1K tokens):

| Model | Input | Output |
|-------|-------|--------|
| claude-opus-4-6 | $0.015 | $0.075 |
| claude-sonnet-4-6 | $0.003 | $0.015 |
| claude-haiku-4-5-20251001 | $0.0008 | $0.004 |
| gpt-4o | $0.005 | $0.015 |
| gpt-4o-mini | $0.00015 | $0.0006 |

---

## 11. Scaling & Operations

**File**: `agentx/scaling/tracing.py`, `agentx/scaling/optimizer.py`

### 11.1 Distributed Tracing (OpenTelemetry)

```python
class Tracer:
    """Track request flow across agents, RAG, LLM calls."""
    __init__(service="agentx", max_traces=10000)

    @asynccontextmanager
    span(operation, parent=None, attributes=None)   # Usage: async with tracer.span("llm_call") as s:

    get_trace(trace_id) -> list[Span]
    get_recent_spans(limit=50, operation="") -> list[Span]
    setup_otlp(endpoint="http://localhost:4318")     # Export to Jaeger/Zipkin
    report() -> dict                                  # Per-operation latency stats (avg, p50, p95, p99)
```

### 11.2 Latency Budget

```python
class LatencyBudget:
    """Enforce per-request time budgets."""
    __init__(total_ms=5000)
    allocate("retrieval", 1000)
    allocate("llm", 3000)
    remaining("retrieval") -> float
    is_expired("retrieval") -> bool
```

### 11.3 Circuit Breaker

```python
class CircuitBreaker:
    """Prevent cascading failures on LLM/vector store outages."""
    __init__(failure_threshold=5, recovery_timeout=30.0)

    # State machine: CLOSED -> (5 failures) -> OPEN -> (30s) -> HALF_OPEN -> (success) -> CLOSED
    state: CircuitState
    allow_request() -> bool
    record_success() -> None
    record_failure() -> None
```

### 11.4 Task Queue

```python
class TaskQueue:
    """Async priority queue with N workers for background processing."""
    __init__(max_workers=4, max_queue_size=10000)

    @handler("task_name") decorator
    submit(task_name, payload=None, priority=NORMAL, max_retries=3) -> task_id
    start() -> None
    stop() -> None
    get_result(task_id, timeout=30.0) -> Any
```

### 11.5 Model Router

```python
class ModelRouter:
    """Select optimal model based on task requirements and budget."""
    select_model(
        task_complexity="medium",     # low|medium|high
        max_cost_per_call=0.05,
        max_latency_ms=5000,
        min_quality=0.7,
        prefer="cost"                 # cost|quality|speed
    ) -> ModelConfig | None
```

### 11.6 Self-Learner

```python
class SelfLearner:
    """Reduce LLM dependency by caching high-confidence patterns."""
    check(query) -> str | None       # Return cached answer if confidence > threshold
    learn(query, response, score=0.0, validated=False)
    export_training_data(format="jsonl") -> str
```

### 11.7 Health Checks

```python
class HealthCheck:
    register(name, check_fn)         # Register a health check
    check(name) -> dict              # Run one check
    check_all() -> dict              # {"healthy": bool, "checks": {...}}
    readiness() -> bool              # Quick ready/not-ready
```

---

## 12. Prompt Engineering

**File**: `agentx/prompts/manager.py`

### 12.1 Prompt Templates

```python
class PromptTemplate(BaseModel):
    name: str
    version: str = "1.0"
    template: str                    # Uses {variable} syntax
    description: str = ""
    variables: list[str] = []
    model_hint: str = ""
    max_tokens_hint: int = 0
    temperature_hint: float | None = None
    tags: list[str] = []

    render(**kwargs) -> str
```

### 12.2 Context Manager (Token Fitting)

```python
class ContextManager:
    """Fit system prompt + RAG context + conversation into the context window."""
    __init__(max_context_tokens=100000, tokenizer="")

    estimate_tokens(text) -> int     # Uses tiktoken (cl100k_base) or fallback
    fit_context(
        system_prompt,
        rag_context="",
        conversation=None,
        max_response_tokens=4096,
        priorities=None              # {"system": 3, "rag": 2, "conversation": 1}
    ) -> dict                        # Fitted components with token counts

    build_messages(system_prompt, user_query, rag_context="", conversation=None)
      -> (system: str, messages: list[dict])
```

**Fitting Algorithm**:
1. Calculate available tokens: `max_context - max_response_tokens`
2. Sort components by priority (system > RAG > conversation)
3. Allocate tokens proportionally, highest priority gets full allocation first
4. Truncate lower-priority components to fit

### 12.3 Response Cache

```python
class ResponseCache:
    __init__(max_size=1000, similarity_threshold=0.95)
    get(query, system="") -> str | None
    set(query, response, system="", ttl=3600)
    stats() -> dict    # {hits, misses, hit_rate, size}
```

---

## 13. Tools & MCP Integration

**File**: `agentx/tools/mcp.py`

### 13.1 MCP (Model Context Protocol)

```python
class MCPConnection:
    """Connect to any MCP-compatible tool server."""
    __init__(name, command="", args=None, env=None, url="", transport="stdio")

    connect() -> None
    disconnect() -> None
    get_agentx_tools() -> list[MCPTool]   # Auto-discover tools
    get_resources() -> list[MCPResource]
    is_connected: bool
    tool_names: list[str]

class MCPManager:
    """Manage multiple MCP server connections."""
    add_server(connection) -> self
    add_stdio_server(name, command, args=None, env=None) -> self
    add_sse_server(name, url, headers=None) -> self
    connect_all() -> dict[str, bool]
    disconnect_all() -> None
    get_all_tools() -> list[MCPTool]
    get_tools_from(server_name) -> list[MCPTool]
    connected_servers: list[str]
    summary() -> dict
```

**Supported Transports**: `stdio`, `sse`, `streamable-http`

### 13.2 Built-in Tools

| Tool | Description |
|------|-------------|
| `DatabaseTool` | Execute SQL queries (SELECT, INSERT, UPDATE, DELETE) |
| `HTTPTool` | Make HTTP requests (GET, POST, PUT, PATCH) |
| `RAGSearchTool` | Search RAG knowledge base with configurable strategy |

---

## 14. Autonomous Daemon

**File**: `agentx/daemon/runner.py`, `agentx/daemon/scheduler.py`, `agentx/daemon/server.py`, `agentx/daemon/watcher.py`

### 14.1 Daemon Runner

```python
class AgentXDaemon:
    """24/7 self-healing autonomous agent system."""
    __init__(app=None, config=None, app_config=None)

    # Scheduling (fluent API + decorators)
    every(seconds=0, minutes=0, hours=0, name="", handler=None, run_immediately=False)
    cron(hour=-1, minute=0, day_of_week=-1, name="", handler=None)
    delayed(delay_seconds, name="", handler=None)
    on_event(event, name="", handler=None)

    # File watching
    watch(path, handler, patterns=None, recursive=True)

    # External integration
    on_webhook(source, handler)
    on_message(channel, handler)
    add_api_route(path, handler)

    # Lifecycle
    @on_start, @on_stop, @on_error decorators

    # Run
    run_forever() -> None            # Blocks until SIGINT/SIGTERM

    # Info
    app: AgentXApp
    uptime_seconds: float
    stats() -> dict
```

### 14.2 Daemon Configuration

```python
class DaemonConfig(BaseModel):
    # Server
    server_enabled: bool = True
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    server_api_key: str = ""
    cors_origins: list[str] = ["*"]

    # Scheduler
    scheduler_enabled: bool = True
    scheduler_tick_interval: float = 1.0

    # File watcher
    watcher_enabled: bool = False
    watcher_poll_interval: float = 2.0

    # Message queue
    mq_enabled: bool = False
    mq_redis_url: str = ""

    # Watchdog
    watchdog_enabled: bool = True
    watchdog_interval_seconds: float = 30.0
    auto_restart_on_failure: bool = True
    max_restart_attempts: int = 5
    restart_cooldown_seconds: float = 10.0

    # Health
    health_check_interval_seconds: float = 60.0
    unhealthy_threshold: int = 3

    # Process
    graceful_shutdown_timeout: float = 30.0
    pid_file: str = ""
    log_level: str = "INFO"
    log_file: str = ""

    @classmethod minimal()     # Just API server
    @classmethod full(port)    # Everything enabled
    @classmethod from_env()    # From environment variables
```

### 14.3 Job Scheduler

```python
class JobScheduler:
    __init__(tick_interval=1.0)

    # Register jobs
    add_interval(name, handler, seconds=0, minutes=0, hours=0, run_immediately=False) -> job_id
    add_cron(name, handler, hour=-1, minute=0, day_of_week=-1) -> job_id
    add_delayed(name, handler, delay_seconds=0) -> job_id
    add_event_triggered(name, handler, trigger_event="") -> job_id

    # Control
    start() / stop()
    pause_job(job_id) / resume_job(job_id) / remove_job(job_id)
    trigger_event(event, data=None) -> list[JobRun]

    # Info
    get_job(job_id) / list_jobs() / stats()
```

**Job Types**:

| Type | Trigger | Repeats | Example |
|------|---------|---------|---------|
| `INTERVAL` | Every N seconds | Yes | "Every 5 minutes" |
| `CRON` | At specific time | Yes | "Daily at 9:00 AM" |
| `DELAYED` | After N seconds | Once | "Run in 30 seconds" |
| `EVENT` | On event trigger | Yes | "On new_data event" |

**Job Execution**:
- Retry with configurable delay (default: 5s between retries)
- Max retries (default: 3)
- Timeout per job (default: 300s)
- Overlap policy: `skip` (default), `queue`, `allow`
- Execution history tracking (last 100 runs per job)

### 14.4 HTTP/WebSocket API Server

**REST Endpoints**:

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/api/v1/chat` | Send message to agent | Y |
| `POST` | `/api/v1/dispatch` | Dispatch to specific agent | Y |
| `POST` | `/api/v1/pipeline` | Run agent pipeline | Y |
| `POST` | `/api/v1/webhook` | Receive webhook | Y |
| `POST` | `/api/v1/webhook/{source}` | Receive named webhook | Y |
| `GET` | `/api/v1/health` | Health check | N |
| `GET` | `/api/v1/status` | Full daemon status | Y |
| `GET` | `/api/v1/jobs` | List scheduled jobs | Y |
| `POST` | `/api/v1/jobs/{id}/pause` | Pause a job | Y |
| `POST` | `/api/v1/jobs/{id}/resume` | Resume a job | Y |
| `GET` | `/api/v1/metrics` | System metrics | Y |
| `GET` | `/api/v1/agents` | List registered agents | Y |
| `WS` | `/ws` | WebSocket (real-time) | Y |

**Authentication**: Bearer token or X-API-Key header (when `server_api_key` is set)

**Request/Response Format**:
```json
// POST /api/v1/chat
{
  "message": "Hello, how are you?",
  "session_id": "sess-123",
  "user_id": "user-1"
}

// Response
{
  "status": "success",
  "message": "ok",
  "data": {
    "response": "I'm doing well! How can I help you?",
    "session_id": "sess-123",
    "data": {}
  },
  "timestamp": 1710596400.0
}
```

**WebSocket Protocol**:
```json
// Client -> Server
{"action": "chat", "message": "Hello"}
{"action": "status"}
{"action": "ping"}

// Server -> Client
{"status": "success", "data": {"response": "..."}}
{"action": "pong", "timestamp": 1710596400.0}
```

### 14.5 File System Watcher

```python
class FileWatcher:
    __init__(poll_interval=2.0)
    watch(path, handler, patterns=["*"], recursive=True, debounce_seconds=1.0)
    start() / stop()
    stats() -> dict
```

**Event Types**: `CREATED`, `MODIFIED`, `DELETED`

**Default Ignore Patterns**: `*.pyc`, `__pycache__`, `.git`, `.DS_Store`, `*.swp`, `*.tmp`

### 14.6 Message Queue Watcher

```python
class MessageQueueWatcher:
    __init__(redis_url="")
    subscribe(channel, handler)
    publish(channel, data) -> None
    start() / stop()
```

**Backends**: Redis pub/sub (if redis URL provided), in-memory queue (fallback)

### 14.7 Watchdog & Self-Healing

The watchdog runs every 30s (configurable) and checks:
1. Is the HTTP server still running?
2. Is the scheduler still running?
3. Is the file watcher still running?
4. Is the message queue watcher still running?

**Recovery Flow**:
```
Subsystem fails
  -> Watchdog detects (within 30s)
  -> Wait cooldown (10s)
  -> Restart failed subsystem
  -> Log recovery
  -> If max_restart_attempts (5) exceeded: stop retrying, require manual intervention
```

**Health Monitor** (separate from watchdog):
- Runs health checks every 60s
- If 3 consecutive failures: trigger recovery
- Auto-reconnects database on failure
- Calls `on_health_fail` hooks

---

## 15. Pre-built Agent Patterns

**File**: `agentx/agents/patterns.py`, `agentx/agents/`

### 15.1 Generic Patterns (use in any project)

| Agent | Purpose | Key Feature |
|-------|---------|-------------|
| `RouterAgent` | Route messages to specialized agents | LLM-based intent classification |
| `GuardrailAgent` | Input/output safety enforcement | Configurable safety rules |
| `SummarizationAgent` | Summarize text/conversations | Style control (bullet, paragraph) |
| `ClassifierAgent` | Multi-category classification | Confidence scores per category |
| `RAGAgent` | Knowledge-grounded responses | Auto-search RAG, cite sources |

### 15.2 Domain Agents (examples)

| Agent | Domain | Purpose |
|-------|--------|---------|
| `InterviewerAgent` | HR/Education | Generate interview questions, evaluate answers |
| `EvaluatorAgent` | HR/Education | Score responses, provide feedback |
| `LearningPathAgent` | Education | Adaptive learning recommendations |
| `GoalTrackerAgent` | Productivity | Track progress, suggest actions |
| `AnalyticsAgent` | Analytics | Generate reports from data |

---

## 16. Application Bootstrap

**File**: `agentx/app.py`

```python
class AgentXApp:
    """One-line setup that wires all components together."""
    __init__(config: AgentXConfig | None = None)

    # Lifecycle
    start() -> AgentXApp           # Initialize all 23 components
    stop() -> None                 # Graceful shutdown

    # Context manager support
    async with AgentXApp() as app: ...

    # Components (initialized on start())
    app.db                         # Database
    app.memory                     # AgentMemory
    app.rbac                       # RBACManager
    app.auth                       # AuthGateway
    app.injection_guard            # InjectionGuard
    app.namespaces                 # NamespaceManager
    app.learner                    # SelfLearner
    app.costs                      # CostTracker
    app.prompts                    # PromptManager
    app.orchestrator               # Orchestrator
    app.router                     # ModelRouter
    app.tracer                     # Tracer
    app.breaker                    # CircuitBreaker
    app.queue                      # TaskQueue
    app.health                     # HealthCheck
    app.analytics                  # QueryAnalytics
    app.evaluator                  # RAGASEvaluator
    app.hallucination_guard        # HallucinationGuard
    app.knowledge                  # KnowledgeFreshnessManager
    app.retention                  # RetentionEnforcer
    app.finetune                   # FineTunePipeline
    app.pii_masker                 # RuntimePIIMasker
    app.citations                  # CitationManager
    app.moderator                  # ContentModerator
    app.vulnerability_scanner      # VulnerabilityScanner

    # LLM access
    app.llm("agent")               # Get LLM for a specific layer
    app.llm("evaluation")
    app.llm("routing")

    # Info
    app.is_started: bool
    app.summary() -> dict          # Full configuration summary
```

---

## 17. API Reference

### 17.1 Quick Start

```python
from agentx import AgentXApp, SimpleAgent, AgentConfig

# 1. Create app
app = AgentXApp()
await app.start()

# 2. Register agents
app.orchestrator.register(SimpleAgent(config=AgentConfig(
    name="assistant",
    system_prompt="You are a helpful assistant.",
)))
app.orchestrator.set_fallback("assistant")

# 3. Send messages
result = await app.orchestrator.send("Hello!")
print(result.content)

# 4. Cleanup
await app.stop()
```

### 17.2 With Daemon (24/7)

```python
from agentx import AgentXApp, AgentXDaemon, DaemonConfig, SimpleAgent, AgentConfig

app = AgentXApp()
await app.start()
app.orchestrator.register(SimpleAgent(config=AgentConfig(name="bot")))
app.orchestrator.set_fallback("bot")

daemon = AgentXDaemon(app=app, config=DaemonConfig(server_port=8080))

@daemon.every(minutes=5, name="health_check")
async def check():
    print(await app.health.check_all())

await daemon.run_forever()
```

### 17.3 CLI

```bash
# Start daemon
python -m agentx.daemon --port 8080

# With API key
python -m agentx.daemon --port 8080 --api-key "secret"

# Full mode (all features)
python -m agentx.daemon --full

# From environment
AGENTX_PORT=9000 AGENTX_API_KEY=secret python -m agentx.daemon --env
```

---

## 18. Deployment

### 18.1 systemd Service

```ini
[Unit]
Description=AgentX Daemon
After=network.target postgresql.service

[Service]
Type=simple
User=agentx
WorkingDirectory=/opt/agentx
ExecStart=/opt/agentx/venv/bin/python -m agentx.daemon --port 8080 --env
Restart=always
RestartSec=10
Environment=AGENTX_ENV=production
Environment=AGENTX_DATABASE_URL=postgresql://user:pass@localhost:5432/agentx
Environment=ANTHROPIC_API_KEY=sk-ant-...

[Install]
WantedBy=multi-user.target
```

### 18.2 Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[all]"
EXPOSE 8080
CMD ["python", "-m", "agentx.daemon", "--port", "8080", "--env"]
```

### 18.3 Docker Compose

```yaml
version: "3.8"
services:
  agentx:
    build: .
    ports:
      - "8080:8080"
    environment:
      - AGENTX_ENV=production
      - AGENTX_DATABASE_URL=postgresql://agentx:pass@db:5432/agentx
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - AGENTX_PORT=8080
    depends_on:
      - db
      - redis
    restart: always

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agentx
      POSTGRES_USER: agentx
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

---

## 19. Dependencies

### 19.1 Core (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic` | >=2.0 | Data models, validation |
| `anthropic` | >=0.40.0 | Claude LLM provider |
| `aiosqlite` | >=0.19 | SQLite async driver |

### 19.2 Optional

| Package | Version | Purpose | Install |
|---------|---------|---------|---------|
| `openai` | >=1.0 | OpenAI/GPT provider | `pip install agentx[openai]` |
| `qdrant-client` | >=1.7 | Qdrant vector store | `pip install agentx[qdrant]` |
| `asyncpg` | >=0.29 | PostgreSQL driver | `pip install agentx[postgres]` |
| `redis` | >=5.0 | Caching, message queue | `pip install agentx[redis]` |
| `aiohttp` | >=3.9 | HTTP server (daemon) | `pip install agentx[all]` |
| `voyageai` | >=0.3 | Voyage embeddings | `pip install agentx[all]` |
| `mcp` | >=1.0 | MCP tool integration | `pip install agentx[mcp]` |
| `tiktoken` | any | Accurate tokenization | `pip install tiktoken` |
| `sentence-transformers` | any | Local embeddings/reranker | Manual install |
| `chromadb` | any | Chroma vector store | Manual install |
| `pinecone-client` | any | Pinecone vector store | Manual install |

### 19.3 Install All

```bash
pip install agentx[all]
```

---

## 20. Project Use Cases

### 20.1 Use Case Matrix

| Component | Interview Bot | Trading Bot | Content Pipeline | Customer Support | Code Review |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Orchestrator | Y | Y | Y | Y | Y |
| RAG Engine | Y | - | Y | Y | Y |
| JWT Auth | Y | Y | - | Y | Y |
| RBAC | Y | Y | - | Y | Y |
| Content Moderation | Y | - | Y | Y | - |
| PII Masking | Y | Y | - | Y | - |
| Hallucination Guard | Y | - | - | Y | - |
| Cost Tracking | Y | Y | Y | Y | Y |
| Daemon (24/7) | Y | Y | Y | Y | - |
| Scheduler | Y | Y | Y | - | - |
| File Watcher | - | - | Y | - | Y |
| Webhooks | Y | Y | Y | Y | Y |
| WebSocket | Y | Y | - | Y | - |
| Self-Learning | Y | Y | - | Y | - |
| Circuit Breaker | Y | Y | Y | Y | Y |

### 20.2 Architecture per Project Type

**Interview Bot**: Orchestrator routes to InterviewerAgent -> EvaluatorAgent -> LearningPathAgent. RAG provides question bank. Daemon runs 24/7 API. Scheduler refreshes questions daily.

**Trading Bot**: Daemon scheduler runs market scans every 5 minutes. Circuit breaker handles exchange API outages. Webhook receives price alerts. PII masker protects account data.

**Content Pipeline**: File watcher detects uploads in `/data/uploads`. Pipeline ingests (PII detect -> validate -> clean -> chunk -> embed). Task queue processes in parallel.

**Customer Support**: Daemon API server runs 24/7. RouterAgent classifies intent. RAGAgent searches knowledge base. GuardrailAgent enforces response safety. WebSocket for real-time chat.

---

## Appendix A: File Structure

```
agentx/
  __init__.py              # v0.6.0, all exports
  app.py                   # AgentXApp bootstrap (23 components)
  core/
    __init__.py
    agent.py               # BaseAgent, SimpleAgent, AgentConfig, AgentState
    context.py             # AgentContext
    message.py             # AgentMessage, MessageType, Priority
    orchestrator.py        # Orchestrator, Pipeline, Route
    tool.py                # BaseTool, FunctionTool, ToolResult, @tool
    llm.py                 # BaseLLMProvider, AnthropicProvider, OpenAIProvider
  config/
    __init__.py
    settings.py            # AgentXConfig + all sub-configs
  db/
    __init__.py
    provider.py            # Database, SQLiteProvider, PostgreSQLProvider
    models.py              # SQL schema (13 tables)
  memory/
    __init__.py
    store.py               # AgentMemory, ShortTermMemory, LongTermMemory
  rag/
    __init__.py
    engine.py              # RAGEngine, Document, TextChunker
    retrieval.py           # BM25Index, CrossEncoderReranker, SemanticCache, QueryRewriter
    stores.py              # ChromaVectorStore, PineconeVectorStore, LocalEmbedder
  pipeline/
    __init__.py
    ingestion.py           # IngestionPipeline, PIIDetector, DataValidator, DataCleaner, FileLoader
    knowledge.py           # KnowledgeFreshnessManager, RetentionEnforcer, FineTunePipeline, RuntimePIIMasker, CitationManager
  security/
    __init__.py
    rbac.py                # RBACManager, User, Role, Permission
    auth.py                # AuthGateway, InjectionGuard, NamespaceManager
    moderation.py          # ContentModerator, VulnerabilityScanner
  evaluation/
    __init__.py
    metrics.py             # ResponseEvaluator, HallucinationDetector, CostTracker
    ragas.py               # RAGASEvaluator, QueryAnalytics
    guardrails.py          # HallucinationGuard, GroundingConfig
  scaling/
    __init__.py
    optimizer.py           # ModelRouter, SelfLearner, RateLimiter, LatencyOptimizer
    tracing.py             # Tracer, LatencyBudget, CircuitBreaker, TaskQueue, HealthCheck
  prompts/
    __init__.py
    manager.py             # PromptTemplate, PromptManager, ContextManager, ResponseCache
  tools/
    __init__.py
    mcp.py                 # MCPConnection, MCPManager, MCPTool
    builtin.py             # DatabaseTool, HTTPTool, RAGSearchTool
  agents/
    __init__.py
    patterns.py            # RouterAgent, GuardrailAgent, SummarizationAgent, ClassifierAgent, RAGAgent
    interviewer.py         # InterviewerAgent
    evaluator.py           # EvaluatorAgent
    learning_path.py       # LearningPathAgent
    goal_tracker.py        # GoalTrackerAgent
    analytics.py           # AnalyticsAgent
  daemon/
    __init__.py
    __main__.py            # CLI entry point
    runner.py              # AgentXDaemon, DaemonConfig, run_daemon
    scheduler.py           # JobScheduler, JobConfig, Job, JobRun
    server.py              # AgentXServer, WebhookHandler, APIResponse
    watcher.py             # FileWatcher, MessageQueueWatcher
  utils/
    __init__.py
    logging.py             # setup_logging
    metrics.py             # metrics utilities
```

## Appendix B: Total Counts

| Category | Count |
|----------|-------|
| Python modules | 35+ |
| Classes | 80+ |
| Pydantic models | 40+ |
| Enums | 15+ |
| Database tables | 13 |
| API endpoints | 12 |
| Configuration options | 100+ |
| Pre-built agents | 10 |
| Security patterns (regex) | 50+ |
| Supported LLM providers | 2 (Anthropic, OpenAI) |
| Supported vector stores | 4 (Qdrant, Chroma, Pinecone, local) |
| Supported embedders | 3 (Anthropic/Voyage, OpenAI, local) |
| Supported databases | 2 (SQLite, PostgreSQL) |

---

*End of Technical Engineering Specification*
*AgentX Framework v0.6.0 | MIT License | https://github.com/grtechlearn/agentx*
