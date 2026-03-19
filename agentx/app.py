"""
AgentX - Application Bootstrap.
One-line setup that wires Database, LLMs, Memory, RBAC, Security, Tracing, and all modules.

Usage:
    # Zero-config (SQLite + Claude Sonnet)
    app = AgentXApp()
    await app.start()

    # From environment variables
    app = AgentXApp(AgentXConfig.from_env())
    await app.start()

    # Production (PostgreSQL + cost-optimized LLM layers)
    app = AgentXApp(AgentXConfig.production("postgresql://user:pass@db:5432/agentx"))
    await app.start()

    # Access components
    app.db              # Database
    app.memory          # AgentMemory (write-through to DB)
    app.rbac            # RBACManager (persisted to DB)
    app.auth            # AuthGateway (JWT + injection guard)
    app.namespaces      # NamespaceManager (vector namespace scoping)
    app.learner         # SelfLearner (persisted to DB)
    app.costs           # CostTracker (persisted to DB)
    app.prompts         # PromptManager (persisted to DB)
    app.orchestrator    # Orchestrator
    app.tracer          # Distributed tracing
    app.health          # Health checks
    app.analytics       # Query analytics
    app.queue           # Task queue (horizontal scaling)
    app.llm("agent")    # Get LLM for a specific layer
"""

from __future__ import annotations

import logging
from typing import Any

from .config import AgentXConfig, DatabaseConfig, LLMConfig, LLMLayerConfig
from .core.llm import BaseLLMProvider, LLMConfig as CoreLLMConfig, create_llm
from .core.orchestrator import Orchestrator
from .db import Database, create_database
from .evaluation.metrics import CostTracker
from .evaluation.ragas import RAGASEvaluator, QueryAnalytics
from .evaluation.guardrails import HallucinationGuard, GroundingConfig
from .memory.store import AgentMemory
from .pipeline.knowledge import (
    KnowledgeFreshnessManager, RetentionEnforcer, RetentionPolicy,
    FineTunePipeline, RuntimePIIMasker, CitationManager,
)
from .prompts.manager import PromptManager
from .scaling.optimizer import SelfLearner, ModelRouter
from .scaling.tracing import Tracer, CircuitBreaker, TaskQueue, HealthCheck, LatencyBudget
from .security.rbac import RBACManager
from .security.auth import AuthGateway, InjectionGuard, NamespaceManager
from .security.moderation import ContentModerator, ModerationConfig, VulnerabilityScanner, CategoryConfig, ModerationAction, Severity

logger = logging.getLogger("agentx")


class AgentXApp:
    """
    Unified application bootstrap for AgentX.

    Wires all components together based on AgentXConfig:
    - Database (SQLite or PostgreSQL)
    - Multi-layer LLMs (different models for different purposes)
    - Memory (short-term + long-term with DB persistence)
    - RBAC (users, permissions, audit — persisted to DB)
    - Auth Gateway (JWT authentication + injection guard)
    - Namespace Manager (vector namespace scoping per user/role)
    - Self-Learning (rules persisted to DB)
    - Cost Tracking (persisted to DB)
    - Prompt Manager (templates persisted to DB)
    - Orchestrator (agent management)
    - Model Router (smart model selection)
    - Distributed Tracing (spans, latency tracking)
    - Circuit Breaker (cascading failure protection)
    - Task Queue (horizontal scaling with workers)
    - Health Checks (readiness and liveness probes)
    - Query Analytics (miss rates, feedback loops)
    - RAGAS Evaluator (retrieval quality metrics)
    - Hallucination Guard (runtime grounding enforcement)
    - Knowledge Freshness Manager (stale detection, auto-refresh)
    - Retention Enforcer (GDPR compliance, data lifecycle)
    - Fine-Tune Pipeline (training data collection → export)
    - Runtime PII Masker (mask PII before sending to LLM)
    - Citation Manager (source reliability scoring)
    """

    def __init__(self, config: AgentXConfig | None = None):
        self.config = config or AgentXConfig()
        self._started = False

        # Components — initialized on start()
        self.db: Database | None = None
        self.memory: AgentMemory | None = None
        self.rbac: RBACManager | None = None
        self.auth: AuthGateway | None = None
        self.injection_guard: InjectionGuard | None = None
        self.namespaces: NamespaceManager | None = None
        self.learner: SelfLearner | None = None
        self.costs: CostTracker | None = None
        self.prompts: PromptManager | None = None
        self.orchestrator: Orchestrator | None = None
        self.router: ModelRouter | None = None
        self.tracer: Tracer | None = None
        self.breaker: CircuitBreaker | None = None
        self.queue: TaskQueue | None = None
        self.health: HealthCheck | None = None
        self.analytics: QueryAnalytics | None = None
        self.evaluator: RAGASEvaluator | None = None
        self.hallucination_guard: HallucinationGuard | None = None
        self.knowledge: KnowledgeFreshnessManager | None = None
        self.retention: RetentionEnforcer | None = None
        self.finetune: FineTunePipeline | None = None
        self.pii_masker: RuntimePIIMasker | None = None
        self.citations: CitationManager | None = None
        self.moderator: ContentModerator | None = None
        self.vulnerability_scanner: VulnerabilityScanner | None = None

        # LLM provider cache — lazily created per layer
        self._llm_cache: dict[str, BaseLLMProvider] = {}

    async def start(self) -> AgentXApp:
        """
        Initialize and connect all components.
        Call this once at application startup.
        """
        if self._started:
            return self

        logger.info(f"Starting {self.config.app_name} v{self.config.version} "
                     f"[{self.config.env.value}]")

        # 1. Database
        db_cfg = self.config.database
        self.db = create_database(
            provider=db_cfg.provider,
            db_path=db_cfg.sqlite_path,
            dsn=db_cfg.postgres_dsn,
        )
        await self.db.connect()
        logger.info(f"Database: {db_cfg.provider} connected")

        # 2. Memory (write-through to DB)
        self.memory = AgentMemory(
            storage_path=self.config.memory_dir,
            db=self.db,
        )

        # 3. RBAC (persisted to DB)
        self.rbac = RBACManager(db=self.db)
        await self.rbac.load_users()

        # 4. Auth Gateway (JWT + injection guard)
        self.auth = AuthGateway()
        self.injection_guard = InjectionGuard()
        self.namespaces = NamespaceManager()

        # 5. Self-Learning (persisted to DB)
        self.learner = SelfLearner(
            storage_path=self.config.training_dir,
            min_confidence=self.config.self_learning.min_confidence_to_cache,
            db=self.db,
        )

        # 6. Cost Tracking (persisted to DB)
        self.costs = CostTracker(db=self.db)

        # 7. Prompt Manager (persisted to DB)
        self.prompts = PromptManager(db=self.db)
        await self.prompts.load_from_db()

        # 8. Model Router
        self.router = ModelRouter()
        self.router.setup_defaults()

        # 9. Orchestrator
        self.orchestrator = Orchestrator(name=self.config.app_name)

        # 10. Distributed Tracing
        self.tracer = Tracer(service=self.config.app_name)

        # 11. Circuit Breaker (for LLM API calls)
        self.breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        # 12. Task Queue (horizontal scaling)
        self.queue = TaskQueue(max_workers=4)

        # 13. Health Checks
        self.health = HealthCheck()
        self.health.register("database", self._check_db_health)

        # 14. Query Analytics
        self.analytics = QueryAnalytics(db=self.db)

        # 15. RAGAS Evaluator
        self.evaluator = RAGASEvaluator()

        # 16. Hallucination Guard (runtime grounding enforcement)
        self.hallucination_guard = HallucinationGuard(config=GroundingConfig(
            max_hallucination_tolerance=self.config.metrics.max_hallucination_tolerance,
            min_confidence=self.config.metrics.min_answer_confidence,
        ))

        # 17. Knowledge Freshness Manager
        self.knowledge = KnowledgeFreshnessManager(
            stale_warning_days=self.config.governance.stale_data_warning_days
            if hasattr(self.config.governance, 'stale_data_warning_days') else 30,
            db=self.db,
        )

        # 18. Data Retention Enforcer
        self.retention = RetentionEnforcer(
            policy=RetentionPolicy(
                conversations_days=self.config.governance.conversation_retention_days,
                allow_deletion=self.config.governance.allow_data_deletion,
            ),
            db=self.db,
        )

        # 19. Fine-Tune Pipeline
        self.finetune = FineTunePipeline(db=self.db)

        # 20. Runtime PII Masker
        self.pii_masker = RuntimePIIMasker(
            fields=self.config.governance.pii_fields_to_redact
            if self.config.governance.detect_pii else [],
        )

        # 21. Citation Manager
        self.citations = CitationManager()

        # 22. Content Moderation
        mod_cfg = self.config.moderation
        if mod_cfg.enabled:
            moderation_config = ModerationConfig(
                enabled=True,
                severity_threshold=Severity(mod_cfg.severity_threshold),
                default_action=ModerationAction(mod_cfg.default_action),
                categories={
                    "profanity": CategoryConfig(enabled=mod_cfg.block_profanity, action=ModerationAction.BLOCK),
                    "sexual": CategoryConfig(enabled=mod_cfg.block_sexual, action=ModerationAction.BLOCK, severity=Severity.HIGH),
                    "abuse": CategoryConfig(enabled=mod_cfg.block_abuse, action=ModerationAction.BLOCK, severity=Severity.HIGH),
                    "violence": CategoryConfig(enabled=mod_cfg.block_violence, action=ModerationAction.WARN),
                    "self_harm": CategoryConfig(enabled=mod_cfg.block_self_harm, action=ModerationAction.BLOCK, severity=Severity.CRITICAL),
                    "drugs": CategoryConfig(enabled=mod_cfg.block_drugs, action=ModerationAction.WARN),
                    "custom": CategoryConfig(enabled=bool(mod_cfg.custom_blocked_words), custom_words=mod_cfg.custom_blocked_words),
                },
                custom_blocked_words=mod_cfg.custom_blocked_words,
                custom_blocked_patterns=mod_cfg.custom_blocked_patterns,
                whitelist_words=mod_cfg.whitelist_words,
                check_input=mod_cfg.check_input,
                check_output=mod_cfg.check_output,
                log_violations=mod_cfg.log_violations,
                max_violations_before_ban=mod_cfg.max_violations_before_ban,
            )
            self.moderator = ContentModerator(moderation_config)
        else:
            self.moderator = ContentModerator(ModerationConfig(enabled=False))

        # 23. Vulnerability Scanner
        if mod_cfg.vulnerability_scanning:
            self.vulnerability_scanner = VulnerabilityScanner(
                check_code_injection=mod_cfg.block_code_injection,
                check_credentials=mod_cfg.block_credential_exposure,
                check_unsafe_urls=mod_cfg.block_unsafe_urls,
                block_on_critical=True,
            )
        else:
            self.vulnerability_scanner = VulnerabilityScanner(
                check_code_injection=False,
                check_credentials=False,
                check_unsafe_urls=False,
                check_serialization=False,
                check_info_disclosure=False,
                check_insecure_patterns=False,
            )
        logger.info(f"Content moderation: {'enabled' if mod_cfg.enabled else 'disabled'}")

        self._started = True
        logger.info(f"{self.config.app_name} started successfully")
        return self

    async def stop(self) -> None:
        """Shutdown — close DB connections, stop workers, cleanup."""
        if self.queue:
            await self.queue.stop()
        if self.db:
            await self.db.close()
        self._started = False
        logger.info(f"{self.config.app_name} stopped")

    # --- Health Check Implementation ---

    async def _check_db_health(self) -> dict[str, Any]:
        """Database health check."""
        if self.db and self.db.is_connected:
            return {"status": "healthy", "provider": self.config.database.provider}
        return {"status": "unhealthy", "error": "Database not connected"}

    # --- LLM Access ---

    def llm(self, layer: str = "default") -> BaseLLMProvider:
        """
        Get the LLM provider for a specific layer.

        Layers: default, agent, evaluation, routing, embedding, summary, fallback

        Usage:
            agent_llm = app.llm("agent")      # Best model for conversations
            eval_llm = app.llm("evaluation")   # Precise model for scoring
            cheap_llm = app.llm("routing")     # Fast/cheap for classification
        """
        if layer in self._llm_cache:
            return self._llm_cache[layer]

        layer_config = self.config.llm.get_layer(layer)
        provider = create_llm(CoreLLMConfig(
            provider=layer_config.provider,
            model=layer_config.model,
            api_key=layer_config.api_key,
            base_url=layer_config.base_url or None,
            temperature=layer_config.temperature,
            max_tokens=layer_config.max_tokens,
        ))
        self._llm_cache[layer] = provider
        return provider

    # --- Convenience ---

    def get_llm_config(self, layer: str = "default") -> LLMLayerConfig:
        """Get the LLM config for a specific layer."""
        return self.config.llm.get_layer(layer)

    @property
    def is_started(self) -> bool:
        return self._started

    def summary(self) -> dict[str, Any]:
        """Get a summary of the app configuration."""
        llm_layers = {}
        for layer in ("default", "agent", "evaluation", "routing", "embedding", "summary", "fallback"):
            cfg = getattr(self.config.llm, layer, None)
            if cfg is not None:
                llm_layers[layer] = f"{cfg.provider}:{cfg.model}"
            elif layer == "default":
                llm_layers[layer] = f"{self.config.llm.default.provider}:{self.config.llm.default.model}"

        return {
            "app": self.config.app_name,
            "version": self.config.version,
            "env": self.config.env.value,
            "started": self._started,
            "database": {
                "provider": self.config.database.provider,
                "connected": self.db.is_connected if self.db else False,
            },
            "llm_layers": llm_layers,
            "budget": {
                "monthly_limit": f"${self.config.budget.max_monthly_spend_usd}",
                "daily_limit": f"${self.config.budget.max_daily_spend_usd}",
            },
            "features": {
                "pii_detection": self.config.governance.detect_pii,
                "self_learning": self.config.self_learning.enabled,
                "caching": self.config.cache.enabled,
                "jwt_auth": True,
                "injection_guard": True,
                "distributed_tracing": True,
                "circuit_breaker": True,
                "task_queue": True,
                "health_checks": True,
                "ragas_evaluation": True,
                "query_analytics": True,
                "hallucination_guard": True,
                "knowledge_freshness": True,
                "data_retention": True,
                "finetune_pipeline": True,
                "runtime_pii_masking": self.config.governance.detect_pii,
                "citation_management": True,
                "content_moderation": self.config.moderation.enabled,
                "vulnerability_scanning": self.config.moderation.vulnerability_scanning,
            },
        }

    async def __aenter__(self) -> AgentXApp:
        return await self.start()

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()
