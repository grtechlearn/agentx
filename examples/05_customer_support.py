"""
Example 05: Customer Support Bot — Real-world multi-agent project.

A complete customer support system with:
- Greeting agent (fast, cheap model)
- FAQ agent (uses self-learning to reduce LLM calls)
- Complaint agent (quality model, logs to audit)
- Escalation agent (hands off to human)
- RBAC for support staff vs customers
- Cost tracking per user
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    BaseAgent, AgentConfig, AgentContext, AgentMessage, MessageType,
    User, Role, Permission, PromptTemplate,
)
from agentx.config import LLMConfig, LLMLayerConfig


# --- Agents ---

class GreeterAgent(BaseAgent):
    """Greets users and classifies their intent."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=(
                f"User message: {message.content}\n\n"
                "Classify the intent as one of: greeting, faq, complaint, escalation.\n"
                "Also provide a friendly response.\n"
                "Reply in format:\nINTENT: <intent>\nRESPONSE: <response>"
            ),
            context=context,
        )
        # Parse intent
        content = response.content
        intent = "faq"  # default
        for line in content.split("\n"):
            if line.strip().upper().startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().lower()
                break

        return message.reply(
            content=content,
            data={"intent": intent, "usage": response.usage},
        )


class FAQAgent(BaseAgent):
    """Handles frequently asked questions. Uses self-learning to avoid LLM calls."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=message.content,
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"type": "faq", "usage": response.usage},
        )


class ComplaintAgent(BaseAgent):
    """Handles complaints with empathy and logs them."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=(
                f"Customer complaint: {message.content}\n\n"
                "Respond with empathy. Acknowledge the issue. "
                "Provide a solution or next steps. "
                "Include a reference number (format: CMP-XXXX)."
            ),
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"type": "complaint", "resolved": False, "usage": response.usage},
        )


class EscalationAgent(BaseAgent):
    """Escalates to human support."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        return message.reply(
            content=(
                "I understand this needs personal attention. "
                "I've escalated your case to our support team. "
                "A human agent will contact you within 2 hours.\n\n"
                f"Reference: ESC-{context.session_id[:6].upper()}\n"
                "Priority: High"
            ),
            data={"type": "escalation", "priority": "high"},
        )


async def main():
    # Configure for customer support use case
    config = AgentXConfig(
        app_name="SupportBot",
        database=DatabaseConfig.memory(),
        llm=LLMConfig(
            # Greeter uses cheap/fast model (just classification)
            routing=LLMLayerConfig(model="claude-haiku-4-5-20251001", temperature=0.1, max_tokens=256),
            # FAQ & complaints use standard model
            default=LLMLayerConfig(model="claude-sonnet-4-6", temperature=0.5),
            # Fallback for budget overruns
            fallback=LLMLayerConfig(model="claude-haiku-4-5-20251001"),
        ),
    )

    async with AgentXApp(config) as app:

        # Create agents
        greeter = GreeterAgent(config=AgentConfig(
            name="greeter",
            system_prompt="You are a friendly customer support greeter. Classify intents and respond warmly.",
            model="claude-haiku-4-5-20251001",  # Use cheap model for greeting
        ))
        faq = FAQAgent(config=AgentConfig(
            name="faq",
            system_prompt=(
                "You are a customer support FAQ agent for TechStore India. "
                "Products: smartphones, laptops, accessories. "
                "Return policy: 7 days. Warranty: 1 year. "
                "Delivery: 3-5 days. COD available. EMI available on orders > ₹10,000."
            ),
        ))
        complaint = ComplaintAgent(config=AgentConfig(
            name="complaint",
            system_prompt="You are a compassionate customer support agent handling complaints.",
        ))
        escalation = EscalationAgent(config=AgentConfig(
            name="escalation",
            system_prompt="You handle escalations to human agents.",
        ))

        app.orchestrator.register_many(greeter, faq, complaint, escalation)

        # Routing rules
        @app.orchestrator.route_to("greeter")
        def route_greeting(msg, ctx):
            greetings = {"hi", "hello", "hey", "namaste", "good morning"}
            return msg.content.strip().lower().split()[0] in greetings if msg.content else False

        @app.orchestrator.route_to("complaint")
        def route_complaint(msg, ctx):
            complaint_words = {"broken", "defective", "wrong", "damaged", "refund", "terrible"}
            return any(w in msg.content.lower() for w in complaint_words)

        @app.orchestrator.route_to("escalation")
        def route_escalation(msg, ctx):
            return "speak to human" in msg.content.lower() or "manager" in msg.content.lower()

        app.orchestrator.set_fallback("faq")

        # Setup users
        await app.rbac.add_user_async(User(id="support-1", name="Priya", role=Role.MANAGER))
        await app.rbac.add_user_async(User(id="customer-1", name="Rahul", role=Role.USER))

        # Pre-train self-learner with common FAQs (no LLM calls for these!)
        faqs = [
            ("what is your return policy", "Our return policy is 7 days from delivery. Items must be unused and in original packaging. Refunds are processed within 5-7 business days."),
            ("do you offer emi", "Yes! EMI is available on all orders above ₹10,000. We support all major banks and credit cards. No-cost EMI available on select products."),
            ("how long does delivery take", "Standard delivery takes 3-5 business days. Express delivery (₹99 extra) delivers within 1-2 days. Free delivery on orders above ₹500."),
            ("do you accept cod", "Yes, Cash on Delivery (COD) is available on orders up to ₹50,000. COD charges: ₹40 per order."),
        ]
        for query, response in faqs:
            await app.learner.learn(query, response, score=0.99, validated=True)

        # --- Simulate customer conversations ---
        print("=" * 60)
        print("  SupportBot — Customer Support Demo")
        print("=" * 60)

        conversations = [
            ("Hello, I need help!", "customer-1"),
            ("What is your return policy?", "customer-1"),
            ("Do you offer EMI?", "customer-1"),
            ("My phone arrived broken and screen is cracked!", "customer-1"),
            ("I want to speak to a human manager", "customer-1"),
        ]

        for message, user_id in conversations:
            print(f"\n👤 Customer: {message}")

            # Check self-learner first (saves LLM calls!)
            cached = await app.learner.check(message.lower().strip("!?."))
            if cached:
                print(f"🤖 Bot (from cache): {cached}")
                print(f"   💡 No LLM call needed!")
                continue

            result = await app.orchestrator.send(
                content=message,
                user_id=user_id,
            )
            print(f"🤖 Bot ({result.sender}): {result.content}")

        # --- Reports ---
        print(f"\n{'=' * 60}")
        print("📊 Session Report")
        print(f"   Self-learner stats: {app.learner.stats()}")
        print(f"   Cost report: {app.costs.report()}")
        audit = await app.rbac.get_audit_log_async(limit=5)
        print(f"   Audit entries: {len(audit)}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
