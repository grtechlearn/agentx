"""
Example 01: QuickStart — Simplest possible AgentX usage.

This creates a single agent and runs it.
Zero config — just works.
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig,
    SimpleAgent, AgentConfig, AgentMessage, AgentContext,
    DatabaseConfig,
)


async def main():
    # 1. Start AgentX (zero-config: SQLite + Claude Sonnet)
    async with AgentXApp(AgentXConfig(database=DatabaseConfig.memory())) as app:

        # 2. Create a simple agent
        agent = SimpleAgent(config=AgentConfig(
            name="assistant",
            system_prompt="You are a helpful coding assistant. Be concise.",
        ))

        # 3. Register with orchestrator
        app.orchestrator.register(agent)
        app.orchestrator.set_fallback("assistant")

        # 4. Send a message
        result = await app.orchestrator.send(
            content="What is the difference between let and const in JavaScript?",
            user_id="user-1",
        )

        print(f"Answer: {result.content}")

        # 5. Track cost automatically
        print(f"\nCost report: {app.costs.report()}")

        # 6. Check app summary
        print(f"\nApp: {app.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
