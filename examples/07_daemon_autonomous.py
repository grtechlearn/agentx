"""
Example: Running AgentX as a 24/7 Autonomous Agent.

The daemon is a SEPARATE PLUGIN layer — it wraps your normal AgentX app.
Without the daemon, your agents work normally (request-response).
With the daemon, they run 24/7 with scheduling, webhooks, and API access.

Usage:
    python examples/07_daemon_autonomous.py
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    AgentXDaemon, DaemonConfig,
    SimpleAgent, AgentConfig,
)


async def main():
    # ── Step 1: Normal AgentX setup (works without daemon) ──
    app = AgentXApp(AgentXConfig(database=DatabaseConfig.memory()))
    await app.start()

    # Register agents
    app.orchestrator.register(SimpleAgent(
        config=AgentConfig(
            name="assistant",
            role="General assistant",
            system_prompt="You are a helpful AI assistant.",
        ),
    ))
    app.orchestrator.set_fallback("assistant")

    # ── Step 2: Wrap with daemon for 24/7 operation ──
    daemon = AgentXDaemon(
        app=app,
        config=DaemonConfig(
            server_enabled=True,
            server_port=8080,
            scheduler_enabled=True,
            watchdog_enabled=True,
        ),
    )

    # ── Step 3: Add scheduled jobs ──

    @daemon.every(hours=1, name="health_report")
    async def hourly_report():
        """Run every hour — check system health."""
        health = await app.health.check_all()
        print(f"Health: {health}")

    @daemon.cron(hour=9, minute=0, name="morning_summary")
    async def morning_summary():
        """Run every day at 9 AM."""
        result = await app.orchestrator.send("Generate a morning summary")
        print(f"Morning summary: {result.content[:100]}")

    @daemon.on_start
    async def startup():
        print("Daemon started! API available at http://localhost:8080")
        print("Endpoints:")
        print("  POST /api/v1/chat     — Send messages to agents")
        print("  GET  /api/v1/health   — Health check")
        print("  GET  /api/v1/status   — Full status")
        print("  GET  /api/v1/jobs     — Scheduled jobs")
        print("  GET  /api/v1/metrics  — System metrics")

    # ── Step 4: Run forever (Ctrl+C to stop) ──
    await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
