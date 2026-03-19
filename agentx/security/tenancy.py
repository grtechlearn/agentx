"""
AgentX - Multi-Tenancy Support.

Organization-level data isolation for SaaS deployments.
Each tenant gets isolated: data, agents, namespaces, budgets, and rate limits.

Usage:
    tenant_mgr = TenantManager(db=app.db)

    # Create tenants
    await tenant_mgr.create_tenant(Tenant(
        id="org-acme",
        name="Acme Corp",
        plan="pro",
        config=TenantConfig(
            max_agents=20,
            max_users=50,
            monthly_budget_usd=500,
            allowed_models=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        ),
    ))

    # Resolve tenant from request
    tenant = await tenant_mgr.get_tenant("org-acme")

    # Enforce tenant limits
    tenant_mgr.check_limits(tenant, action="create_agent")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx.security")


class TenantPlan(BaseModel):
    """Predefined plan limits."""

    name: str = "free"
    max_users: int = 5
    max_agents: int = 3
    max_sessions: int = 100
    max_documents: int = 1000
    monthly_budget_usd: float = 10.0
    daily_requests: int = 1000
    allowed_models: list[str] = Field(default_factory=lambda: ["claude-sonnet-4-6"])
    features: list[str] = Field(default_factory=lambda: ["basic"])
    storage_mb: int = 100

    @classmethod
    def free(cls) -> TenantPlan:
        return cls(name="free", max_users=5, max_agents=3, monthly_budget_usd=10.0)

    @classmethod
    def starter(cls) -> TenantPlan:
        return cls(
            name="starter", max_users=20, max_agents=10,
            max_sessions=500, max_documents=10000,
            monthly_budget_usd=100.0, daily_requests=5000,
            allowed_models=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            features=["basic", "rag", "analytics"],
            storage_mb=1000,
        )

    @classmethod
    def pro(cls) -> TenantPlan:
        return cls(
            name="pro", max_users=100, max_agents=50,
            max_sessions=5000, max_documents=100000,
            monthly_budget_usd=1000.0, daily_requests=50000,
            allowed_models=["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4o"],
            features=["basic", "rag", "analytics", "finetune", "custom_models", "priority_support"],
            storage_mb=10000,
        )

    @classmethod
    def enterprise(cls) -> TenantPlan:
        return cls(
            name="enterprise", max_users=0, max_agents=0,  # 0 = unlimited
            max_sessions=0, max_documents=0,
            monthly_budget_usd=0, daily_requests=0,
            allowed_models=[],  # empty = all models
            features=["all"],
            storage_mb=0,
        )


class TenantConfig(BaseModel):
    """Per-tenant configuration overrides."""

    max_users: int = 0                  # 0 = use plan default
    max_agents: int = 0
    max_sessions: int = 0
    max_documents: int = 0
    monthly_budget_usd: float = 0
    daily_requests: int = 0
    allowed_models: list[str] = Field(default_factory=list)
    blocked_models: list[str] = Field(default_factory=list)
    custom_moderation: dict[str, Any] = Field(default_factory=dict)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class Tenant(BaseModel):
    """A tenant (organization) in the multi-tenant system."""

    id: str                             # Unique org identifier
    name: str = ""
    plan: str = "free"                  # free, starter, pro, enterprise
    config: TenantConfig = Field(default_factory=TenantConfig)
    is_active: bool = True
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Runtime stats (not persisted)
    current_users: int = 0
    current_agents: int = 0
    current_month_spend: float = 0.0
    today_requests: int = 0


class TenantManager:
    """
    Multi-tenant management — data isolation, limits, and billing.

    Enforces:
    - User limits per tenant
    - Agent limits per tenant
    - Budget limits per tenant
    - Model access restrictions
    - Rate limiting per tenant
    - Feature gating
    """

    PLANS = {
        "free": TenantPlan.free,
        "starter": TenantPlan.starter,
        "pro": TenantPlan.pro,
        "enterprise": TenantPlan.enterprise,
    }

    def __init__(self, db: Any = None):
        self._db = db
        self._tenants: dict[str, Tenant] = {}
        self._plans: dict[str, TenantPlan] = {}
        self._request_counts: dict[str, dict[str, int]] = {}  # tenant_id -> {date: count}

        # Initialize default plans
        for name, factory in self.PLANS.items():
            self._plans[name] = factory()

    # --- Tenant CRUD ---

    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create a new tenant."""
        self._tenants[tenant.id] = tenant
        logger.info(f"Tenant created: {tenant.id} ({tenant.name}) plan={tenant.plan}")
        return tenant

    async def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)

    async def update_tenant(self, tenant_id: str, **updates: Any) -> Tenant | None:
        """Update tenant fields."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        tenant.updated_at = time.time()
        return tenant

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Soft-delete a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.is_active = False
            return True
        return False

    async def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        """List all tenants."""
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.is_active]
        return tenants

    # --- Plan Management ---

    def get_plan(self, plan_name: str) -> TenantPlan:
        """Get plan limits for a plan name."""
        return self._plans.get(plan_name, TenantPlan.free())

    def get_tenant_plan(self, tenant: Tenant) -> TenantPlan:
        """Get the effective plan for a tenant (plan defaults + overrides)."""
        plan = self.get_plan(tenant.plan)

        # Apply tenant-specific overrides
        if tenant.config.max_users > 0:
            plan.max_users = tenant.config.max_users
        if tenant.config.max_agents > 0:
            plan.max_agents = tenant.config.max_agents
        if tenant.config.monthly_budget_usd > 0:
            plan.monthly_budget_usd = tenant.config.monthly_budget_usd
        if tenant.config.daily_requests > 0:
            plan.daily_requests = tenant.config.daily_requests
        if tenant.config.allowed_models:
            plan.allowed_models = tenant.config.allowed_models

        return plan

    # --- Limit Enforcement ---

    def check_user_limit(self, tenant: Tenant) -> bool:
        """Check if tenant can add more users."""
        plan = self.get_tenant_plan(tenant)
        if plan.max_users == 0:
            return True  # unlimited
        return tenant.current_users < plan.max_users

    def check_agent_limit(self, tenant: Tenant) -> bool:
        """Check if tenant can create more agents."""
        plan = self.get_tenant_plan(tenant)
        if plan.max_agents == 0:
            return True
        return tenant.current_agents < plan.max_agents

    def check_budget(self, tenant: Tenant) -> bool:
        """Check if tenant is within budget."""
        plan = self.get_tenant_plan(tenant)
        if plan.monthly_budget_usd == 0:
            return True
        return tenant.current_month_spend < plan.monthly_budget_usd

    def check_rate_limit(self, tenant: Tenant) -> bool:
        """Check if tenant is within daily request limit."""
        plan = self.get_tenant_plan(tenant)
        if plan.daily_requests == 0:
            return True

        today = time.strftime("%Y-%m-%d")
        counts = self._request_counts.get(tenant.id, {})
        today_count = counts.get(today, 0)
        return today_count < plan.daily_requests

    def check_model_access(self, tenant: Tenant, model: str) -> bool:
        """Check if tenant can use a specific model."""
        plan = self.get_tenant_plan(tenant)
        if not plan.allowed_models:
            return True  # empty = all models allowed

        # Check blocked models from config
        if model in tenant.config.blocked_models:
            return False

        return model in plan.allowed_models

    def check_feature(self, tenant: Tenant, feature: str) -> bool:
        """Check if a feature is available for this tenant."""
        plan = self.get_tenant_plan(tenant)
        if "all" in plan.features:
            return True
        return feature in plan.features

    def record_request(self, tenant_id: str) -> None:
        """Record a request for rate limiting."""
        today = time.strftime("%Y-%m-%d")
        if tenant_id not in self._request_counts:
            self._request_counts[tenant_id] = {}
        counts = self._request_counts[tenant_id]
        counts[today] = counts.get(today, 0) + 1

    def record_spend(self, tenant_id: str, amount_usd: float) -> None:
        """Record spending for budget tracking."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.current_month_spend += amount_usd

    # --- Data Isolation Helpers ---

    def get_db_filter(self, tenant_id: str) -> dict[str, str]:
        """Get database filter for tenant-scoped queries."""
        return {"organization_id": tenant_id}

    def get_namespace_prefix(self, tenant_id: str) -> str:
        """Get namespace prefix for vector store isolation."""
        return f"tenant:{tenant_id}:"

    def scope_collection(self, tenant_id: str, collection: str) -> str:
        """Scope a vector store collection name to a tenant."""
        return f"{tenant_id}__{collection}"

    # --- Info ---

    def tenant_summary(self, tenant: Tenant) -> dict[str, Any]:
        """Get tenant usage summary."""
        plan = self.get_tenant_plan(tenant)
        today = time.strftime("%Y-%m-%d")
        today_requests = self._request_counts.get(tenant.id, {}).get(today, 0)

        return {
            "id": tenant.id,
            "name": tenant.name,
            "plan": tenant.plan,
            "is_active": tenant.is_active,
            "usage": {
                "users": f"{tenant.current_users}/{plan.max_users or 'unlimited'}",
                "agents": f"{tenant.current_agents}/{plan.max_agents or 'unlimited'}",
                "budget": f"${tenant.current_month_spend:.2f}/${plan.monthly_budget_usd or 'unlimited'}",
                "requests_today": f"{today_requests}/{plan.daily_requests or 'unlimited'}",
            },
            "features": plan.features,
            "allowed_models": plan.allowed_models or ["all"],
        }
