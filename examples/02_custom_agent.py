"""
Example 02: Custom Agent with Tools — Build your own agent.

Shows:
- Custom agent class
- Custom tools with @tool decorator
- Agent with tools + LLM
- Using AgentXApp for DB persistence
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    BaseAgent, AgentConfig, AgentContext, AgentMessage,
    tool,
)


# --- Custom Tools ---

@tool(name="lookup_price", description="Look up the current price of a product")
async def lookup_price(product: str) -> str:
    """Simulated price lookup."""
    prices = {
        "iphone 16": "₹79,900",
        "macbook air m3": "₹1,14,900",
        "pixel 9": "₹79,999",
        "galaxy s24": "₹74,999",
    }
    key = product.strip().lower()
    for name, price in prices.items():
        if key in name or name in key:
            return f"{name.title()}: {price}"
    return f"Product '{product}' not found in catalog."


@tool(name="compare_specs", description="Compare specs between two products")
async def compare_specs(product_a: str, product_b: str) -> str:
    """Simulated spec comparison."""
    return f"""
Comparison: {product_a} vs {product_b}
- Display: Both have OLED displays
- Processor: {product_a} uses latest chip, {product_b} uses competitive chip
- Camera: Both have excellent camera systems
- Price: Use lookup_price tool for current pricing
"""


# --- Custom Agent ---

class ShoppingAssistantAgent(BaseAgent):
    """
    A shopping assistant that helps users compare products.
    Uses tools to look up prices and compare specs.
    """

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        # Use the LLM with tools
        response = await self.think(
            prompt=message.content,
            context=context,
            use_tools=True,
        )
        return message.reply(content=response.content, data={"usage": response.usage})


async def main():
    # Start with in-memory DB for this example
    async with AgentXApp(AgentXConfig(database=DatabaseConfig.memory())) as app:

        # Create the shopping assistant with tools
        assistant = ShoppingAssistantAgent(
            config=AgentConfig(
                name="shopping_assistant",
                role="Shopping Advisor",
                system_prompt=(
                    "You are a helpful shopping assistant for electronics in India. "
                    "Use the lookup_price tool to check prices and compare_specs to "
                    "compare products. Always show prices in INR (₹). Be concise."
                ),
            ),
            tools=[lookup_price, compare_specs],
        )

        # Register and set as default
        app.orchestrator.register(assistant)
        app.orchestrator.set_fallback("shopping_assistant")

        # Interact
        queries = [
            "What's the price of iPhone 16?",
            "Compare iPhone 16 vs Pixel 9",
        ]

        for query in queries:
            print(f"\nUser: {query}")
            result = await app.orchestrator.send(content=query, user_id="shopper-1")
            print(f"Assistant: {result.content}")

        # Show cost tracking
        print(f"\n--- Cost Report ---")
        print(app.costs.report())


if __name__ == "__main__":
    asyncio.run(main())
