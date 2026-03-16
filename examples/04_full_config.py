"""
Example 04: Full Configuration — Every configurable option in AgentX.

Shows:
- Database config (SQLite, PostgreSQL, in-memory)
- Multi-LLM layer config (different models for different purposes)
- Budget & cost limits
- Data governance & PII detection
- Self-learning
- RBAC & security
- Prompt templates
- Memory persistence
- All wired together through AgentXApp
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    BaseAgent, AgentConfig, AgentContext, AgentMessage,
    User, Role, Permission, PromptTemplate,
    setup_logging,
)
from agentx.config import (
    LLMConfig, LLMLayerConfig, LLMBudget, DataGovernance,
    SystemMetrics, CacheConfig, SelfLearningConfig, Environment,
)

setup_logging(level="INFO")


async def main():
    # ================================================================
    # FULL CONFIGURATION — everything in one place
    # ================================================================
    config = AgentXConfig(
        # --- Environment ---
        env=Environment.DEVELOPMENT,
        app_name="InterviewBot",
        debug=True,

        # --- Database ---
        # Option 1: SQLite (zero-config, dev)
        database=DatabaseConfig.memory(),  # Use :memory: for this demo

        # Option 2: SQLite with custom path
        # database=DatabaseConfig.sqlite("./interview_bot.db"),

        # Option 3: PostgreSQL (production)
        # database=DatabaseConfig.postgres("postgresql://user:pass@localhost:5432/interviewbot"),

        # Option 4: From env var
        # database=DatabaseConfig(
        #     provider=os.getenv("DB_PROVIDER", "sqlite"),
        #     postgres_dsn=os.getenv("DATABASE_URL", ""),
        # ),

        # --- Multi-LLM Layers ---
        # Different models for different purposes!
        llm=LLMConfig(
            # Default model (used when no specific layer is configured)
            default=LLMLayerConfig(
                provider="anthropic",
                model="claude-sonnet-4-6",
                temperature=0.7,
                max_tokens=4096,
            ),

            # Agent conversations — quality matters most
            agent=LLMLayerConfig(
                provider="anthropic",
                model="claude-sonnet-4-6",  # or claude-opus-4-6 for premium
                temperature=0.7,
                max_tokens=4096,
            ),

            # Evaluation & hallucination detection — needs precision
            evaluation=LLMLayerConfig(
                provider="anthropic",
                model="claude-sonnet-4-6",
                temperature=0.1,  # Low temp for consistent scoring
                max_tokens=2048,
            ),

            # Routing & classification — can be fast and cheap
            routing=LLMLayerConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",  # Cheapest, fastest
                temperature=0.1,
                max_tokens=256,  # Short responses for routing
            ),

            # Summarization — mid-tier is fine
            summary=LLMLayerConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                temperature=0.3,
                max_tokens=1024,
            ),

            # Fallback — when budget exceeded or primary fails
            fallback=LLMLayerConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                temperature=0.5,
                max_tokens=1024,
            ),

            # Embeddings — can use OpenAI's embedding model
            # embedding=LLMLayerConfig(
            #     provider="openai",
            #     model="text-embedding-3-small",
            #     api_key="sk-...",
            # ),
        ),

        # OR use presets:
        # llm=LLMConfig.single(),           # One model for everything
        # llm=LLMConfig.cost_optimized(),    # Auto cheap/expensive split
        # llm=LLMConfig.quality_first(),     # Opus for quality, Sonnet for rest
        # llm=LLMConfig.openai("gpt-4o"),    # All OpenAI
        # llm=LLMConfig.mixed(),             # Claude + OpenAI embeddings

        # --- Budget ---
        budget=LLMBudget(
            max_monthly_spend_usd=50.0,
            max_daily_spend_usd=5.0,
            max_tokens_per_request=4096,
            max_requests_per_minute=60,
            max_requests_per_user_per_day=100,
            prefer_cheaper_model=True,
            fallback_to_cache=True,
            warn_at_percentage=80.0,
        ),

        # --- Data Governance ---
        governance=DataGovernance(
            detect_pii=True,
            pii_fields_to_redact=["email", "phone", "aadhaar", "pan"],
            redaction_method="mask",
            retain_conversations=True,
            conversation_retention_days=90,
            encrypt_at_rest=True,
            allow_data_export=True,
            allow_data_deletion=True,  # GDPR compliance
        ),

        # --- System Metrics ---
        metrics=SystemMetrics(
            max_response_time_ms=5000,
            max_rag_retrieval_ms=1000,
            min_rag_relevance_score=0.7,
            min_answer_confidence=0.6,
            max_hallucination_tolerance=0.1,
            track_cost_per_agent=True,
            track_cost_per_user=True,
        ),

        # --- Caching ---
        cache=CacheConfig(
            enabled=True,
            backend="memory",  # or "redis" with redis_url
            ttl_seconds=3600,
            cache_llm_responses=True,
            cache_rag_results=True,
            similarity_threshold=0.95,
        ),

        # --- Self-Learning ---
        self_learning=SelfLearningConfig(
            enabled=True,
            learn_from_responses=True,
            min_confidence_to_cache=0.9,
            auto_create_rules=True,
            collect_training_data=True,
            export_format="jsonl",
        ),
    )

    # ================================================================
    # START THE APP — everything is wired automatically
    # ================================================================
    async with AgentXApp(config) as app:

        print("=" * 60)
        print(f"  {config.app_name} v{config.version}")
        print(f"  Environment: {config.env.value}")
        print("=" * 60)

        # --- Show LLM layer configuration ---
        print("\n📊 LLM Layer Configuration:")
        for layer in ("default", "agent", "evaluation", "routing", "summary", "fallback"):
            cfg = app.get_llm_config(layer)
            print(f"  {layer:12s} → {cfg.provider}:{cfg.model} (temp={cfg.temperature})")

        # --- Setup RBAC ---
        print("\n🔐 Setting up users...")
        await app.rbac.add_user_async(User(
            id="admin-1", name="Ramesh", email="ramesh@aimediahub.in", role=Role.ADMIN,
        ))
        await app.rbac.add_user_async(User(
            id="user-1", name="Student", email="student@example.com", role=Role.USER,
        ))

        # Check permissions
        print(f"  Admin can manage users: {app.rbac.authorize('admin-1', Permission.ADMIN_USERS)}")
        print(f"  Student can manage users: {app.rbac.authorize('user-1', Permission.ADMIN_USERS)}")
        print(f"  Student can run agents: {app.rbac.authorize('user-1', Permission.AGENT_RUN)}")

        # --- Register prompt templates ---
        print("\n📝 Registering prompt templates...")
        await app.prompts.register_async(PromptTemplate(
            name="interview_question",
            version="1.0",
            template=(
                "You are interviewing a candidate about {{technology}}.\n"
                "Difficulty: {{difficulty}}\n"
                "Ask a {{question_type}} question about {{topic}}.\n"
                "The question should test practical knowledge."
            ),
            variables=["technology", "difficulty", "question_type", "topic"],
            tags=["interview"],
        ))

        await app.prompts.register_async(PromptTemplate(
            name="evaluate_answer",
            version="1.0",
            template=(
                "Evaluate this {{technology}} answer:\n\n"
                "Question: {{question}}\n"
                "Answer: {{answer}}\n\n"
                "Score from 0-10 and explain what's correct, what's missing, "
                "and what's wrong."
            ),
            variables=["technology", "question", "answer"],
            tags=["evaluation"],
        ))

        # Render a prompt
        rendered = app.prompts.render(
            "interview_question",
            technology="React",
            difficulty="medium",
            question_type="conceptual",
            topic="hooks",
        )
        print(f"  Rendered prompt: {rendered[:100]}...")

        # --- Memory (persisted to DB) ---
        print("\n🧠 Testing memory persistence...")
        await app.memory.remember(
            "user_preferences",
            {"technology": "React", "difficulty": "medium", "language": "English"},
            long_term=True,
            user_id="user-1",
        )
        prefs = await app.memory.recall("user_preferences")
        print(f"  Recalled preferences: {prefs}")

        # --- Self-Learning ---
        print("\n🎓 Testing self-learning...")
        await app.learner.learn(
            query="what is usestate in react",
            response=(
                "useState is a React Hook that lets you add state to functional components. "
                "It returns [state, setState] pair. Example: const [count, setCount] = useState(0)"
            ),
            score=0.95,
        )
        cached = await app.learner.check("what is usestate in react")
        if cached:
            print(f"  Self-learned response found! (saves 1 LLM call)")
            print(f"  Response: {cached[:100]}...")
        print(f"  Learner stats: {app.learner.stats()}")

        # --- Cost Tracking ---
        print("\n💰 Testing cost tracking...")
        await app.costs.track_async("claude-sonnet-4-6", 1000, 500, key="user-1", agent_name="interviewer")
        await app.costs.track_async("claude-haiku-4-5-20251001", 500, 200, key="user-1", agent_name="router")
        await app.costs.track_async("claude-sonnet-4-6", 800, 400, key="admin-1", agent_name="evaluator")
        print(f"  Cost report: {app.costs.report()}")

        # Get cost from DB (full history)
        db_costs = await app.costs.get_cost_from_db(user_id="user-1")
        print(f"  User-1 DB costs: ${db_costs['total_cost']:.4f} ({db_costs['total_calls']} calls)")

        # --- Goals (persisted to DB) ---
        print("\n🎯 Testing goals...")
        goal_id = await app.db.save_goal(
            user_id="user-1",
            title="Master React Hooks",
            description="Learn useState, useEffect, useRef, useMemo, useCallback",
        )
        goals = await app.db.get_user_goals("user-1")
        print(f"  Goals: {[g['title'] for g in goals]}")

        # --- Audit Log ---
        print("\n📋 Audit log...")
        logs = await app.rbac.get_audit_log_async(user_id="admin-1", limit=5)
        print(f"  Admin audit entries: {len(logs)}")

        # --- Full Summary ---
        print("\n" + "=" * 60)
        print("📊 App Summary:")
        import json
        print(json.dumps(app.summary(), indent=2))
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
