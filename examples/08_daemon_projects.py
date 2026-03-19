"""
Example: Different project configurations using the same AgentX + Daemon.

Shows how the SAME framework serves completely different use cases
by configuring different agents, schedules, and triggers.

The daemon is OPTIONAL — without it, agents work as normal request-response.
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    AgentXDaemon, DaemonConfig,
    SimpleAgent, AgentConfig,
    FileEvent,
)


# ═══════════════════════════════════════════════════════════════
# Project 1: Interview Bot (24/7 with scheduled question updates)
# ═══════════════════════════════════════════════════════════════

async def interview_bot():
    app = AgentXApp(AgentXConfig(database=DatabaseConfig.memory()))
    await app.start()

    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="interviewer",
        role="Technical interviewer",
        system_prompt="You are a senior technical interviewer...",
    )))
    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="evaluator",
        role="Answer evaluator",
        system_prompt="You evaluate interview answers...",
    )))
    app.orchestrator.set_fallback("interviewer")

    daemon = AgentXDaemon(app=app, config=DaemonConfig(server_port=8080))

    # Scheduled: update question bank daily
    @daemon.cron(hour=2, minute=0, name="update_questions")
    async def update_questions():
        print("Updating question bank...")

    # Scheduled: generate weekly performance report
    @daemon.cron(hour=9, minute=0, day_of_week=0, name="weekly_report")
    async def weekly_report():
        print("Generating weekly interview report...")

    # Webhook: new candidate signed up
    async def on_new_candidate(source, payload):
        name = payload.get("name", "Unknown")
        print(f"New candidate: {name} — preparing interview...")

    daemon.on_webhook("signup", on_new_candidate)

    await daemon.run_forever()


# ═══════════════════════════════════════════════════════════════
# Project 2: Trading Bot (autonomous with market monitoring)
# ═══════════════════════════════════════════════════════════════

async def trading_bot():
    app = AgentXApp(AgentXConfig(database=DatabaseConfig.memory()))
    await app.start()

    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="market_analyzer",
        role="Market analysis",
        system_prompt="You analyze market conditions...",
    )))
    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="risk_manager",
        role="Risk assessment",
        system_prompt="You evaluate trading risk...",
    )))
    app.orchestrator.set_fallback("market_analyzer")

    daemon = AgentXDaemon(app=app, config=DaemonConfig(server_port=8081))

    # Every 5 minutes: scan market conditions
    @daemon.every(minutes=5, name="market_scan", run_immediately=True)
    async def scan_markets():
        result = await app.orchestrator.send("Analyze current market conditions")
        print(f"Market scan: {result.content[:100]}")

    # Every hour: risk assessment
    @daemon.every(hours=1, name="risk_check")
    async def risk_assessment():
        result = await app.orchestrator.send(
            "Evaluate current portfolio risk", data={"agent": "risk_manager"}
        )
        print(f"Risk: {result.content[:100]}")

    # Daily close report
    @daemon.cron(hour=16, minute=30, name="daily_close")
    async def daily_close():
        print("Generating daily close report...")

    # Webhook: price alert from exchange
    async def on_price_alert(source, payload):
        symbol = payload.get("symbol", "?")
        price = payload.get("price", 0)
        print(f"Price alert: {symbol} = ${price}")

    daemon.on_webhook("exchange", on_price_alert)

    await daemon.run_forever()


# ═══════════════════════════════════════════════════════════════
# Project 3: Content Pipeline (file watching + auto-processing)
# ═══════════════════════════════════════════════════════════════

async def content_pipeline():
    app = AgentXApp(AgentXConfig(database=DatabaseConfig.memory()))
    await app.start()

    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="transcriber",
        role="Audio/video transcription",
        system_prompt="You transcribe and summarize media content...",
    )))
    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="content_writer",
        role="Content generation",
        system_prompt="You create articles and social posts from transcripts...",
    )))

    daemon = AgentXDaemon(
        app=app,
        config=DaemonConfig(
            server_port=8082,
            watcher_enabled=True,
        ),
    )

    # Watch uploads folder for new media files
    async def on_new_media(event: FileEvent):
        print(f"New file: {event.path}")
        result = await app.orchestrator.send(
            f"Process new media file: {event.path}"
        )
        print(f"Processed: {result.content[:100]}")

    daemon.watch(
        "/data/uploads",
        handler=on_new_media,
        patterns=["*.mp4", "*.mp3", "*.wav", "*.pdf"],
    )

    # Nightly: clean up processed files
    @daemon.cron(hour=3, minute=0, name="cleanup")
    async def nightly_cleanup():
        print("Cleaning up processed files...")

    await daemon.run_forever()


# ═══════════════════════════════════════════════════════════════
# Project 4: Customer Support (24/7 API, no scheduling needed)
# ═══════════════════════════════════════════════════════════════

async def customer_support():
    app = AgentXApp(AgentXConfig(database=DatabaseConfig.memory()))
    await app.start()

    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="support",
        role="Customer support agent",
        system_prompt="You are a helpful customer support agent...",
    )))
    app.orchestrator.set_fallback("support")

    # Minimal daemon — just API server, no scheduler
    daemon = AgentXDaemon(
        app=app,
        config=DaemonConfig.minimal(),  # Just server + watchdog
    )

    await daemon.run_forever()


# ═══════════════════════════════════════════════════════════════
# Project 5: Normal Agents (NO daemon, request-response only)
# ═══════════════════════════════════════════════════════════════

async def normal_agents():
    """This shows that AgentX works fine WITHOUT the daemon."""
    app = AgentXApp(AgentXConfig(database=DatabaseConfig.memory()))
    await app.start()

    app.orchestrator.register(SimpleAgent(config=AgentConfig(
        name="helper",
        system_prompt="You are a helpful assistant.",
    )))
    app.orchestrator.set_fallback("helper")

    # Direct usage — no daemon needed
    result = await app.orchestrator.send("Hello, how are you?")
    print(f"Response: {result.content}")

    await app.stop()


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    projects = {
        "interview": interview_bot,
        "trading": trading_bot,
        "content": content_pipeline,
        "support": customer_support,
        "normal": normal_agents,
    }

    project = sys.argv[1] if len(sys.argv) > 1 else "normal"

    if project not in projects:
        print(f"Usage: python {sys.argv[0]} [{'/'.join(projects.keys())}]")
        sys.exit(1)

    print(f"Running project: {project}")
    asyncio.run(projects[project]())
