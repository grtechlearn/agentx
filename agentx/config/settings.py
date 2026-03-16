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
    """Master configuration for AgentX."""

    # Environment
    env: Environment = Environment.DEVELOPMENT
    app_name: str = "AgentX"
    version: str = "0.2.0"
    debug: bool = False

    # LLM
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-6"
    fallback_model: str = "claude-haiku-4-5-20251001"
    api_key: str = ""
    api_base_url: str = ""

    # Components
    budget: LLMBudget = Field(default_factory=LLMBudget)
    governance: DataGovernance = Field(default_factory=DataGovernance)
    metrics: SystemMetrics = Field(default_factory=SystemMetrics)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    self_learning: SelfLearningConfig = Field(default_factory=SelfLearningConfig)

    # Paths
    data_dir: str = "./data"
    memory_dir: str = "./data/memory"
    training_dir: str = "./data/training"
    logs_dir: str = "./logs"

    @classmethod
    def from_env(cls) -> AgentXConfig:
        """Load config from environment variables."""
        import os
        return cls(
            env=Environment(os.getenv("AGENTX_ENV", "development")),
            app_name=os.getenv("AGENTX_APP_NAME", "AgentX"),
            default_provider=os.getenv("AGENTX_LLM_PROVIDER", "anthropic"),
            default_model=os.getenv("AGENTX_LLM_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            debug=os.getenv("AGENTX_DEBUG", "false").lower() == "true",
        )

    @classmethod
    def production(cls) -> AgentXConfig:
        """Production-optimized defaults."""
        return cls(
            env=Environment.PRODUCTION,
            debug=False,
            budget=LLMBudget(prefer_cheaper_model=True, fallback_to_cache=True),
            governance=DataGovernance(detect_pii=True, encrypt_at_rest=True),
            metrics=SystemMetrics(max_hallucination_tolerance=0.05),
            cache=CacheConfig(enabled=True, ttl_seconds=7200),
        )
