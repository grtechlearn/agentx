"""
AgentX - Application Bootstrap.
One-line setup that wires Database, LLMs, Memory, RBAC, and all modules from config.

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

    # Custom config
    app = AgentXApp(AgentXConfig(
        database=DatabaseConfig.postgres("postgresql://..."),
        llm=LLMConfig(
            agent=LLMLayerConfig(model="claude-opus-4-6"),
            routing=LLMLayerConfig(model="claude-haiku-4-5-20251001"),
            evaluation=LLMLayerConfig(model="claude-sonnet-4-6"),
        ),
    ))
    await app.start()

    # Access components
    app.db          # Database
    app.memory      # AgentMemory (write-through to DB)
    app.rbac        # RBACManager (persisted to DB)
    app.learner     # SelfLearner (persisted to DB)
    app.costs       # CostTracker (persisted to DB)
    app.prompts     # PromptManager (persisted to DB)
    app.orchestrator  # Orchestrator
    app.llm("agent")  # Get LLM for a specific layer
"""

from __future__ import annotations

import logging
from typing import Any

from .config import AgentXConfig, DatabaseConfig, LLMConfig, LLMLayerConfig
from .core.llm import BaseLLMProvider, LLMConfig as CoreLLMConfig, create_llm
from .core.orchestrator import Orchestrator
from .db import Database, create_database
from .evaluation.metrics import CostTracker
from .memory.store import AgentMemory
from .prompts.manager import PromptManager
from .scaling.optimizer import SelfLearner, ModelRouter
from .security.rbac import RBACManager

logger = logging.getLogger("agentx")


class AgentXApp:
    """
    Unified application bootstrap for AgentX.

    Wires all components together based on AgentXConfig:
    - Database (SQLite or PostgreSQL)
    - Multi-layer LLMs (different models for different purposes)
    - Memory (short-term + long-term with DB persistence)
    - RBAC (users, permissions, audit — persisted to DB)
    - Self-Learning (rules persisted to DB)
    - Cost Tracking (persisted to DB)
    - Prompt Manager (templates persisted to DB)
    - Orchestrator (agent management)
    - Model Router (smart model selection)
    """

    def __init__(self, config: AgentXConfig | None = None):
        self.config = config or AgentXConfig()
        self._started = False

        # Components — initialized on start()
        self.db: Database | None = None
        self.memory: AgentMemory | None = None
        self.rbac: RBACManager | None = None
        self.learner: SelfLearner | None = None
        self.costs: CostTracker | None = None
        self.prompts: PromptManager | None = None
        self.orchestrator: Orchestrator | None = None
        self.router: ModelRouter | None = None

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

        # 4. Self-Learning (persisted to DB)
        self.learner = SelfLearner(
            storage_path=self.config.training_dir,
            min_confidence=self.config.self_learning.min_confidence_to_cache,
            db=self.db,
        )

        # 5. Cost Tracking (persisted to DB)
        self.costs = CostTracker(db=self.db)

        # 6. Prompt Manager (persisted to DB)
        self.prompts = PromptManager(db=self.db)
        await self.prompts.load_from_db()

        # 7. Model Router
        self.router = ModelRouter()
        self.router.setup_defaults()

        # 8. Orchestrator
        self.orchestrator = Orchestrator(name=self.config.app_name)

        self._started = True
        logger.info(f"{self.config.app_name} started successfully")
        return self

    async def stop(self) -> None:
        """Shutdown — close DB connections, cleanup."""
        if self.db:
            await self.db.close()
        self._started = False
        logger.info(f"{self.config.app_name} stopped")

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
            },
        }

    async def __aenter__(self) -> AgentXApp:
        return await self.start()

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()
