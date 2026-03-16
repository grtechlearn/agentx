"""
Example 03: Multi-Agent Pipeline — Chain agents together.

Shows:
- Multiple specialized agents
- Pipeline execution (sequential agent chain)
- Parallel execution
- Routing rules
- Different LLMs for different agents
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    BaseAgent, AgentConfig, AgentContext, AgentMessage,
)
from agentx.config import LLMConfig, LLMLayerConfig


# --- Specialized Agents ---

class ContentWriterAgent(BaseAgent):
    """Writes content based on a topic."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=f"Write a short, engaging paragraph about: {message.content}",
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"stage": "content_written", "usage": response.usage},
        )


class EditorAgent(BaseAgent):
    """Reviews and improves content."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=(
                f"Review and improve this content. Fix any errors, improve clarity, "
                f"and make it more engaging:\n\n{message.content}"
            ),
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"stage": "edited", "usage": response.usage},
        )


class SEOOptimizerAgent(BaseAgent):
    """Adds SEO optimization to content."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=(
                f"Add SEO-friendly title, meta description, and keywords to this content. "
                f"Return in a structured format:\n\n{message.content}"
            ),
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"stage": "seo_optimized", "usage": response.usage},
        )


class TranslatorAgent(BaseAgent):
    """Translates content to Hindi."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        target_lang = message.data.get("target_language", "Hindi")
        response = await self.think(
            prompt=f"Translate the following to {target_lang}:\n\n{message.content}",
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"stage": "translated", "language": target_lang, "usage": response.usage},
        )


async def main():
    # Configure with cost-optimized LLM layers
    config = AgentXConfig(
        app_name="ContentFactory",
        database=DatabaseConfig.memory(),
        llm=LLMConfig.cost_optimized(),  # Haiku for routing, Sonnet for generation
    )

    async with AgentXApp(config) as app:

        # Create agents
        writer = ContentWriterAgent(config=AgentConfig(
            name="writer",
            system_prompt="You are a skilled content writer. Write engaging, informative content.",
        ))
        editor = EditorAgent(config=AgentConfig(
            name="editor",
            system_prompt="You are a meticulous editor. Improve content quality without changing the meaning.",
        ))
        seo = SEOOptimizerAgent(config=AgentConfig(
            name="seo",
            system_prompt="You are an SEO specialist. Optimize content for search engines.",
        ))
        translator = TranslatorAgent(config=AgentConfig(
            name="translator",
            system_prompt="You are a professional translator. Produce natural, fluent translations.",
        ))

        # Register all agents
        app.orchestrator.register_many(writer, editor, seo, translator)

        # --- Pipeline: Write → Edit → SEO ---
        app.orchestrator.add_pipeline("publish", agents=["writer", "editor", "seo"])

        print("=" * 60)
        print("  Content Factory — Multi-Agent Pipeline Demo")
        print("=" * 60)

        # Run the pipeline
        print("\n📝 Running pipeline: Writer → Editor → SEO")
        result = await app.orchestrator.run_pipeline(
            "publish",
            initial_message=AgentMessage(content="React Server Components and their benefits"),
            context=AgentContext(session_id="content-1", user_id="creator-1"),
        )
        print(f"\nFinal output:\n{result.content}")
        print(f"\nStage: {result.data.get('stage')}")

        # --- Parallel execution: translate to multiple languages ---
        print("\n" + "=" * 60)
        print("\n🌍 Running parallel: Translate to Hindi & Tamil")

        # Create separate translator instances for parallel execution
        translator_hi = TranslatorAgent(config=AgentConfig(
            name="translator_hi",
            system_prompt="You are a Hindi translator.",
        ))
        translator_ta = TranslatorAgent(config=AgentConfig(
            name="translator_ta",
            system_prompt="You are a Tamil translator.",
        ))
        app.orchestrator.register_many(translator_hi, translator_ta)

        results = await app.orchestrator.run_parallel(
            agent_names=["translator_hi", "translator_ta"],
            message=AgentMessage(
                content=result.content,
                data={"target_language": "Hindi"},
            ),
        )

        for agent_name, res in results.items():
            print(f"\n[{agent_name}]: {res.content[:200]}...")

        # --- Routing: auto-select agent ---
        print("\n" + "=" * 60)
        print("\n🔀 Routing demo")

        @app.orchestrator.route_to("writer")
        def route_write(msg, ctx):
            return "write" in msg.content.lower() or "create" in msg.content.lower()

        @app.orchestrator.route_to("translator")
        def route_translate(msg, ctx):
            return "translate" in msg.content.lower()

        app.orchestrator.set_fallback("editor")

        # These will be auto-routed
        for query in [
            "Write a paragraph about Python async/await",
            "Translate this to Hindi: Hello, how are you?",
            "Fix the grammar in this text: Me is going to store",
        ]:
            result = await app.orchestrator.send(content=query, user_id="creator-1")
            print(f"\nQuery: {query}")
            print(f"Routed to: {result.sender}")
            print(f"Response: {result.content[:150]}...")

        # Cost report
        print(f"\n{'=' * 60}")
        print(f"Cost report: {app.costs.report()}")


if __name__ == "__main__":
    asyncio.run(main())
