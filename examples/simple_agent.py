"""
Example: Create a simple custom agent with AgentX

Shows how easy it is to create agents - no framework magic needed.
"""

import asyncio
from agentx import (
    BaseAgent,
    AgentConfig,
    AgentContext,
    AgentMessage,
    Orchestrator,
    tool,
    setup_logging,
)

setup_logging()


# --- Create a custom tool ---
@tool(name="calculator", description="Perform basic math calculations")
async def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"
    return str(eval(expression))


# --- Create a custom agent ---
class MathTutorAgent(BaseAgent):
    """A simple math tutor agent."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=message.content,
            context=context,
            use_tools=True,
        )
        return message.reply(content=response.content)


async def main():
    # Create agent with custom config
    tutor = MathTutorAgent(
        config=AgentConfig(
            name="math_tutor",
            role="Math Tutor",
            system_prompt="You are a friendly math tutor. Use the calculator tool to verify calculations.",
        ),
        tools=[calculator],
    )

    # Use directly
    result = await tutor.run(
        message=AgentMessage(content="What is 15% of 2499?"),
        context=AgentContext(session_id="test"),
    )
    print(f"Answer: {result.content}")

    # Or use with orchestrator
    orchestrator = Orchestrator()
    orchestrator.register(tutor)
    orchestrator.set_fallback("math_tutor")

    result = await orchestrator.send("Calculate compound interest on 10000 at 8% for 3 years")
    print(f"\nAnswer: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
