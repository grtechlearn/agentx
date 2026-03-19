"""
AgentX - Central Configuration & System Metrics.
Phase 1: Requirements analysis — system metrics, cost modeling, data governance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ---------------------------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    """Database configuration — SQLite (dev) or PostgreSQL (production)."""

    provider: str = "sqlite"  # sqlite, postgres
    sqlite_path: str = ""  # empty = auto (./data/agentx.db), ":memory:" for tests
    postgres_dsn: str = ""  # postgresql://user:pass@host:5432/dbname
    pool_min_size: int = 2
    pool_max_size: int = 10
    auto_migrate: bool = True  # auto-apply schema on connect

    @classmethod
    def sqlite(cls, path: str = "") -> DatabaseConfig:
        return cls(provider="sqlite", sqlite_path=path)

    @classmethod
    def postgres(cls, dsn: str) -> DatabaseConfig:
        return cls(provider="postgres", postgres_dsn=dsn)

    @classmethod
    def memory(cls) -> DatabaseConfig:
        """In-memory SQLite — great for tests."""
        return cls(provider="sqlite", sqlite_path=":memory:")


# ---------------------------------------------------------------------------
# Multi-LLM Layer Configuration
# ---------------------------------------------------------------------------

class LLMLayerConfig(BaseModel):
    """
    Configure different LLMs for different purposes.

    Each layer can use a different provider/model optimized for its task:
    - agent: Main agent conversations (quality matters)
    - evaluation: Hallucination detection, scoring (needs to be precise)
    - routing: Intent classification, routing decisions (can be fast/cheap)
    - embedding: Text embeddings for RAG (specialized model)
    - summary: Summarization tasks (mid-tier is fine)
    - fallback: When primary model fails or budget exceeded
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key: str = ""  # empty = use env var (ANTHROPIC_API_KEY, OPENAI_API_KEY)
    base_url: str = ""  # empty = default provider URL
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMConfig(BaseModel):
    """
    Multi-LLM configuration — assign different models to different layers.

    Example:
        llm = LLMConfig(
            agent=LLMLayerConfig(model="claude-sonnet-4-6"),       # Quality conversations
            evaluation=LLMLayerConfig(model="claude-opus-4-6"),     # Precise evaluation
            routing=LLMLayerConfig(model="claude-haiku-4-5-20251001"),      # Fast & cheap routing
            embedding=LLMLayerConfig(provider="openai", model="text-embedding-3-small"),
            fallback=LLMLayerConfig(model="claude-haiku-4-5-20251001"),     # Budget fallback
        )
    """

    # Layer-specific LLM configs (None = use default)
    default: LLMLayerConfig = Field(default_factory=lambda: LLMLayerConfig())
    agent: LLMLayerConfig | None = None
    evaluation: LLMLayerConfig | None = None
    routing: LLMLayerConfig | None = None
    embedding: LLMLayerConfig | None = None
    summary: LLMLayerConfig | None = None
    fallback: LLMLayerConfig | None = None

    def get_layer(self, layer: str) -> LLMLayerConfig:
        """Get config for a specific layer, falling back to default."""
        specific = getattr(self, layer, None)
        return specific if specific is not None else self.default

    @classmethod
    def single(cls, provider: str = "anthropic", model: str = "claude-sonnet-4-6",
               api_key: str = "", **kwargs: Any) -> LLMConfig:
        """Use a single model for everything."""
        return cls(default=LLMLayerConfig(
            provider=provider, model=model, api_key=api_key, **kwargs
        ))

    @classmethod
    def cost_optimized(cls) -> LLMConfig:
        """Use cheap models where possible, quality models where needed."""
        return cls(
            default=LLMLayerConfig(model="claude-sonnet-4-6"),
            routing=LLMLayerConfig(model="claude-haiku-4-5-20251001", temperature=0.1, max_tokens=256),
            evaluation=LLMLayerConfig(model="claude-sonnet-4-6", temperature=0.1),
            summary=LLMLayerConfig(model="claude-haiku-4-5-20251001", max_tokens=1024),
            fallback=LLMLayerConfig(model="claude-haiku-4-5-20251001"),
        )

    @classmethod
    def quality_first(cls) -> LLMConfig:
        """Use the best model for agent tasks, standard for everything else."""
        return cls(
            default=LLMLayerConfig(model="claude-sonnet-4-6"),
            agent=LLMLayerConfig(model="claude-opus-4-6", max_tokens=8192),
            evaluation=LLMLayerConfig(model="claude-opus-4-6", temperature=0.1),
            routing=LLMLayerConfig(model="claude-sonnet-4-6", temperature=0.1, max_tokens=256),
            fallback=LLMLayerConfig(model="claude-sonnet-4-6"),
        )

    @classmethod
    def openai(cls, model: str = "gpt-4o", api_key: str = "") -> LLMConfig:
        """Use OpenAI models across all layers."""
        return cls(
            default=LLMLayerConfig(provider="openai", model=model, api_key=api_key),
            routing=LLMLayerConfig(provider="openai", model="gpt-4o-mini", api_key=api_key,
                                   temperature=0.1, max_tokens=256),
            fallback=LLMLayerConfig(provider="openai", model="gpt-4o-mini", api_key=api_key),
        )

    @classmethod
    def mixed(cls) -> LLMConfig:
        """Mix providers — Claude for generation, OpenAI for embeddings."""
        return cls(
            default=LLMLayerConfig(provider="anthropic", model="claude-sonnet-4-6"),
            embedding=LLMLayerConfig(provider="openai", model="text-embedding-3-small"),
            routing=LLMLayerConfig(provider="anthropic", model="claude-haiku-4-5-20251001",
                                   temperature=0.1, max_tokens=256),
            fallback=LLMLayerConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
        )


class LLMBudget(BaseModel):
    """Cost management for LLM usage."""

    max_monthly_spend_usd: float = 100.0
    max_daily_spend_usd: float = 10.0
    max_tokens_per_request: int = 4096
    max_requests_per_minute: int = 60
    max_requests_per_user_per_day: int = 100
    prefer_cheaper_model: bool = True
    fallback_to_cache: bool = True
    warn_at_percentage: float = 80.0  # warn when 80% budget used


class DataGovernance(BaseModel):
    """Data governance and privacy configuration."""

    # PII handling
    detect_pii: bool = True
    pii_fields_to_redact: list[str] = Field(default_factory=lambda: [
        "email", "phone", "aadhaar", "pan", "ssn", "credit_card",
        "address", "date_of_birth", "passport", "bank_account",
    ])
    redaction_method: str = "mask"  # mask, hash, remove

    # Data retention
    retain_conversations: bool = True
    conversation_retention_days: int = 90
    retain_embeddings: bool = True
    embedding_retention_days: int = 365
    retain_user_data: bool = True
    user_data_retention_days: int = 730  # 2 years

    # Data access
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    log_data_access: bool = True
    allow_data_export: bool = True
    allow_data_deletion: bool = True  # GDPR right to be forgotten


class SystemMetrics(BaseModel):
    """System performance targets and thresholds."""

    # Latency
    max_response_time_ms: int = 5000
    max_rag_retrieval_ms: int = 1000
    max_embedding_time_ms: int = 500

    # Throughput
    target_requests_per_second: int = 50
    max_concurrent_agents: int = 10
    max_concurrent_sessions: int = 100

    # Accuracy
    min_rag_relevance_score: float = 0.7
    min_answer_confidence: float = 0.6
    max_hallucination_tolerance: float = 0.1  # 10% max

    # Data freshness
    knowledge_refresh_interval_hours: int = 24
    stale_data_warning_days: int = 30
    auto_refresh_enabled: bool = True

    # Cost
    cost_per_query_target_usd: float = 0.01
    track_cost_per_agent: bool = True
    track_cost_per_user: bool = True


class CacheConfig(BaseModel):
    """Caching configuration to reduce LLM calls."""

    enabled: bool = True
    backend: str = "redis"  # redis, memory, file
    redis_url: str = "redis://localhost:6379"
    ttl_seconds: int = 3600  # 1 hour default
    max_cache_size_mb: int = 512
    cache_embeddings: bool = True
    cache_llm_responses: bool = True
    cache_rag_results: bool = True
    similarity_threshold: float = 0.95  # cache hit if query >95% similar


class ContentModerationConfig(BaseModel):
    """
    Content moderation configuration — pluggable like LLM models.

    Presets: strict, moderate, permissive, disabled
    Categories: profanity, sexual, abuse, violence, self_harm, drugs, custom
    Actions per category: block, warn, redact, log
    """

    enabled: bool = True
    severity_threshold: str = "low"      # low, medium, high, critical
    default_action: str = "block"        # block, warn, redact, log

    # Per-category enable/disable + action override
    block_profanity: bool = True
    block_sexual: bool = True
    block_abuse: bool = True
    block_violence: bool = False         # warn by default
    block_self_harm: bool = True
    block_drugs: bool = False

    # Custom word lists
    custom_blocked_words: list[str] = Field(default_factory=list)
    custom_blocked_patterns: list[str] = Field(default_factory=list)
    whitelist_words: list[str] = Field(default_factory=list)

    # Behavior
    check_input: bool = True
    check_output: bool = True
    log_violations: bool = True
    max_violations_before_ban: int = 0   # 0 = no auto-ban

    # Vulnerability scanning
    vulnerability_scanning: bool = True
    block_credential_exposure: bool = True
    block_code_injection: bool = True
    block_unsafe_urls: bool = True

    @classmethod
    def strict(cls) -> ContentModerationConfig:
        """Block all categories."""
        return cls(
            enabled=True, block_profanity=True, block_sexual=True,
            block_abuse=True, block_violence=True, block_self_harm=True,
            block_drugs=True, vulnerability_scanning=True,
        )

    @classmethod
    def moderate(cls) -> ContentModerationConfig:
        """Block sexual/abuse, warn on profanity/violence."""
        return cls(
            enabled=True, block_profanity=True, block_sexual=True,
            block_abuse=True, block_violence=False, block_self_harm=True,
            block_drugs=False,
        )

    @classmethod
    def disabled(cls) -> ContentModerationConfig:
        """No moderation."""
        return cls(enabled=False)


class SelfLearningConfig(BaseModel):
    """Configuration for self-learning and reducing LLM dependency."""

    enabled: bool = True

    # Local model fallback
    use_local_model_for_simple_tasks: bool = False
    local_model_path: str = ""
    local_model_name: str = ""  # e.g., "llama3", "mistral"

    # Pattern learning
    learn_from_responses: bool = True
    min_confidence_to_cache: float = 0.9
    auto_create_rules: bool = True
    rules_review_required: bool = True  # human review before applying

    # Fine-tuning data collection
    collect_training_data: bool = True
    training_data_path: str = "./data/training"
    min_samples_for_finetune: int = 1000
    export_format: str = "jsonl"  # jsonl, csv, parquet


class AgentXConfig(BaseModel):
    """
    Master configuration for AgentX.

    Everything is configurable from one place:
    - Database: SQLite (dev) or PostgreSQL (production)
    - LLM: Different models for different layers (agent, eval, routing, etc.)
    - Budget: Cost limits and rate limiting
    - Security: PII detection, data governance
    - Performance: Caching, self-learning, latency targets
    """

    # Environment
    env: Environment = Environment.DEVELOPMENT
    app_name: str = "AgentX"
    version: str = "0.6.0"
    debug: bool = False

    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # LLM (multi-layer)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Legacy LLM fields (still work for backward compatibility)
    default_provider: str = ""
    default_model: str = ""
    fallback_model: str = ""
    api_key: str = ""
    api_base_url: str = ""

    # Components
    budget: LLMBudget = Field(default_factory=LLMBudget)
    governance: DataGovernance = Field(default_factory=DataGovernance)
    metrics: SystemMetrics = Field(default_factory=SystemMetrics)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    self_learning: SelfLearningConfig = Field(default_factory=SelfLearningConfig)
    moderation: ContentModerationConfig = Field(default_factory=ContentModerationConfig)

    # Paths
    data_dir: str = "./data"
    memory_dir: str = "./data/memory"
    training_dir: str = "./data/training"
    logs_dir: str = "./logs"

    @classmethod
    def from_env(cls) -> AgentXConfig:
        """Load config from environment variables."""
        import os

        # Database from env
        db_provider = os.getenv("AGENTX_DB_PROVIDER", "sqlite")
        db_config = DatabaseConfig(
            provider=db_provider,
            sqlite_path=os.getenv("AGENTX_DB_PATH", ""),
            postgres_dsn=os.getenv("AGENTX_DATABASE_URL", ""),
        )

        # LLM from env
        llm_provider = os.getenv("AGENTX_LLM_PROVIDER", "anthropic")
        llm_model = os.getenv("AGENTX_LLM_MODEL", "claude-sonnet-4-6")
        llm_config = LLMConfig.single(
            provider=llm_provider, model=llm_model,
            api_key=os.getenv("ANTHROPIC_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        )

        return cls(
            env=Environment(os.getenv("AGENTX_ENV", "development")),
            app_name=os.getenv("AGENTX_APP_NAME", "AgentX"),
            database=db_config,
            llm=llm_config,
            debug=os.getenv("AGENTX_DEBUG", "false").lower() == "true",
        )

    @classmethod
    def development(cls) -> AgentXConfig:
        """Development defaults — SQLite, single model."""
        return cls(
            env=Environment.DEVELOPMENT,
            debug=True,
            database=DatabaseConfig.sqlite(),
            llm=LLMConfig.single(),
        )

    @classmethod
    def production(cls, postgres_dsn: str = "") -> AgentXConfig:
        """Production-optimized — PostgreSQL, cost-optimized LLM layers."""
        import os
        dsn = postgres_dsn or os.getenv("AGENTX_DATABASE_URL", "")
        return cls(
            env=Environment.PRODUCTION,
            debug=False,
            database=DatabaseConfig.postgres(dsn) if dsn else DatabaseConfig.sqlite(),
            llm=LLMConfig.cost_optimized(),
            budget=LLMBudget(prefer_cheaper_model=True, fallback_to_cache=True),
            governance=DataGovernance(detect_pii=True, encrypt_at_rest=True),
            metrics=SystemMetrics(max_hallucination_tolerance=0.05),
            cache=CacheConfig(enabled=True, ttl_seconds=7200),
            moderation=ContentModerationConfig.strict(),
        )
