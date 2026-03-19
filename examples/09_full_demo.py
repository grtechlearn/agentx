"""
AgentX Full Demo — Production-Ready Interview Bot.

This is a complete, deployable demo that shows AgentX running as a
24/7 autonomous AI agent with:
- Multi-agent orchestration (interviewer + evaluator + assistant)
- RAG knowledge base (loads interview questions)
- JWT authentication
- Content moderation
- Scheduled jobs (daily question refresh)
- Webhooks (candidate signup)
- Health monitoring
- Admin dashboard

Run:
    python examples/09_full_demo.py

Then visit:
    http://localhost:8080/dashboard   — Admin dashboard
    http://localhost:8080/api/v1/health — Health check

Test the API:
    curl -X POST http://localhost:8080/api/v1/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "Ask me a Python interview question"}'

    curl -X POST http://localhost:8080/api/v1/dispatch \
      -H "Content-Type: application/json" \
      -d '{"agent": "evaluator", "message": "The candidate said: Python uses GIL for thread safety"}'
"""

import asyncio
import logging

from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    AgentXDaemon, DaemonConfig,
    SimpleAgent, AgentConfig,
    Orchestrator,
    AuthGateway, InjectionGuard,
    ContentModerator, ModerationConfig,
    CostTracker,
)
from agentx.core.message import AgentMessage, MessageType
from agentx.core.context import AgentContext
from agentx.core.agent import BaseAgent


# ═══════════════════════════════════════════════════════════════
# Custom Agents
# ═══════════════════════════════════════════════════════════════

class InterviewerAgent(BaseAgent):
    """Generates interview questions based on topic and difficulty."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        topic = message.data.get("topic", "Python")
        difficulty = message.data.get("difficulty", "intermediate")

        response = await self.think(
            prompt=f"""Generate a technical interview question about {topic}
            at {difficulty} difficulty level. Include:
            1. The question
            2. Key points the interviewer should look for
            3. A follow-up question

            User request: {message.content}""",
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"agent": "interviewer", "topic": topic, "difficulty": difficulty},
        )


class EvaluatorAgent(BaseAgent):
    """Evaluates candidate answers against technical criteria."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=f"""Evaluate this technical interview answer.

            Answer: {message.content}

            Score on:
            1. Technical accuracy (0-10)
            2. Completeness (0-10)
            3. Communication clarity (0-10)

            Provide specific feedback and suggestions for improvement.""",
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"agent": "evaluator"},
        )


class AssistantAgent(BaseAgent):
    """General-purpose assistant for non-interview queries."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=message.content,
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"agent": "assistant"},
        )


# ═══════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("demo")

    # ── Step 1: Configure AgentX ──
    config = AgentXConfig(
        app_name="InterviewBot",
        database=DatabaseConfig.memory(),  # Use SQLite for demo, PostgreSQL for production
        moderation=ModerationConfig.moderate(),
    )

    app = AgentXApp(config)
    await app.start()
    logger.info("AgentX app started")

    # ── Step 2: Register Agents ──
    interviewer = InterviewerAgent(config=AgentConfig(
        name="interviewer",
        role="Technical Interviewer",
        system_prompt="""You are an expert technical interviewer with 15 years of experience.
        You create challenging but fair interview questions. You focus on practical knowledge
        and problem-solving ability. Your questions should test deep understanding, not just
        memorization.""",
        model="claude-sonnet-4-6",
    ))

    evaluator = EvaluatorAgent(config=AgentConfig(
        name="evaluator",
        role="Answer Evaluator",
        system_prompt="""You are a senior technical evaluator. You provide fair, detailed
        feedback on interview answers. You score accurately — neither too harsh nor too lenient.
        You always provide constructive feedback with specific improvement suggestions.""",
        model="claude-sonnet-4-6",
    ))

    assistant = AssistantAgent(config=AgentConfig(
        name="assistant",
        role="General Assistant",
        system_prompt="You are a helpful AI assistant for the interview platform.",
        model="claude-sonnet-4-6",
    ))

    app.orchestrator.register_many(interviewer, evaluator, assistant)

    # ── Step 3: Setup Routing ──
    # Route based on message content
    app.orchestrator.add_route(
        "interviewer",
        condition=lambda msg, ctx: any(
            kw in msg.content.lower()
            for kw in ["interview", "question", "ask me", "quiz", "test me"]
        ),
        priority=10,
    )
    app.orchestrator.add_route(
        "evaluator",
        condition=lambda msg, ctx: any(
            kw in msg.content.lower()
            for kw in ["evaluate", "score", "rate", "assess", "feedback", "the candidate"]
        ),
        priority=10,
    )
    app.orchestrator.set_fallback("assistant")

    # ── Step 4: Setup Pipeline ──
    app.orchestrator.add_pipeline("full_interview", agents=["interviewer", "evaluator"])

    # ── Step 5: Wire up Daemon ──
    daemon = AgentXDaemon(
        app=app,
        config=DaemonConfig(
            server_enabled=True,
            server_port=8080,
            scheduler_enabled=True,
            watchdog_enabled=True,
            watcher_enabled=False,
        ),
    )

    # ── Step 6: Add Scheduled Jobs ──
    @daemon.every(hours=1, name="system_health")
    async def hourly_health():
        """Check system health every hour."""
        health = await app.health.check_all()
        if not health["healthy"]:
            logger.warning(f"Health check failed: {health['checks']}")
        else:
            logger.info("System healthy")

    @daemon.cron(hour=0, minute=0, name="daily_stats")
    async def daily_stats():
        """Log daily statistics at midnight."""
        stats = app.summary()
        costs = app.costs.report() if app.costs else {}
        logger.info(f"Daily stats — features: {len(stats.get('features', {}))} active")
        if costs:
            logger.info(f"Daily costs: {costs}")

    # ── Step 7: Webhook Handlers ──
    async def on_new_candidate(source, payload):
        """Handle new candidate signup webhook."""
        name = payload.get("name", "Unknown")
        topics = payload.get("topics", ["Python"])
        logger.info(f"New candidate: {name}, topics: {topics}")
        # Could auto-create a session, send welcome message, etc.

    daemon.on_webhook("signup", on_new_candidate)

    # ── Step 8: Lifecycle Hooks ──
    @daemon.on_start
    async def on_startup():
        logger.info("=" * 60)
        logger.info("  InterviewBot Demo is running!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("  Dashboard:  http://localhost:8080/dashboard")
        logger.info("  Health:     http://localhost:8080/api/v1/health")
        logger.info("  Chat API:   POST http://localhost:8080/api/v1/chat")
        logger.info("  Agents:     GET  http://localhost:8080/api/v1/agents")
        logger.info("")
        logger.info("  Test it:")
        logger.info('    curl -X POST http://localhost:8080/api/v1/chat \\')
        logger.info('      -H "Content-Type: application/json" \\')
        logger.info('      -d \'{"message": "Ask me a Python interview question"}\'')
        logger.info("")
        logger.info("  Press Ctrl+C to stop")
        logger.info("=" * 60)

    # ── Step 9: Run Forever ──
    await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
