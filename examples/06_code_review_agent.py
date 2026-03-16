"""
Example 06: Code Review Agent — AI-powered code review system.

Shows:
- Guardrail agent (checks code safety before review)
- Code reviewer agent (analyzes code quality)
- Suggestion agent (improves code)
- Pipeline: Guardrail → Review → Suggest
- Custom tools for code analysis
- Different LLMs for different tasks
"""

import asyncio
from agentx import (
    AgentXApp, AgentXConfig, DatabaseConfig,
    BaseAgent, AgentConfig, AgentContext, AgentMessage,
    tool,
)
from agentx.config import LLMConfig, LLMLayerConfig


# --- Tools ---

@tool(name="count_lines", description="Count lines of code, comments, and blank lines")
async def count_lines(code: str) -> str:
    lines = code.split("\n")
    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comments = sum(1 for l in lines if l.strip().startswith(("#", "//", "/*", "*")))
    code_lines = total - blank - comments
    return f"Total: {total}, Code: {code_lines}, Comments: {comments}, Blank: {blank}"


@tool(name="check_complexity", description="Estimate code complexity based on nesting and branches")
async def check_complexity(code: str) -> str:
    nesting = max(
        (len(line) - len(line.lstrip())) // 4
        for line in code.split("\n") if line.strip()
    ) if code.strip() else 0
    branches = sum(1 for keyword in ["if ", "elif ", "else:", "for ", "while ", "except ", "case "]
                   if keyword in code)
    complexity = "Low" if branches < 5 else "Medium" if branches < 10 else "High"
    return f"Max nesting: {nesting}, Branches: {branches}, Complexity: {complexity}"


# --- Agents ---

class GuardrailAgent(BaseAgent):
    """Checks code for security issues before review."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=(
                f"Check this code for security issues (SQL injection, XSS, hardcoded secrets, "
                f"unsafe eval, command injection). Report only actual issues found.\n\n"
                f"```\n{message.content}\n```"
            ),
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"stage": "guardrail", "usage": response.usage},
        )


class ReviewerAgent(BaseAgent):
    """Reviews code quality and provides feedback."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=f"Review this code for quality, best practices, and potential bugs:\n\n{message.content}",
            context=context,
            use_tools=True,
        )
        return message.reply(
            content=response.content,
            data={"stage": "reviewed", "usage": response.usage},
        )


class SuggestionAgent(BaseAgent):
    """Suggests improvements to the code."""

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=(
                f"Based on this code review, suggest specific improvements. "
                f"Show the improved code:\n\n{message.content}"
            ),
            context=context,
        )
        return message.reply(
            content=response.content,
            data={"stage": "suggestions", "usage": response.usage},
        )


async def main():
    config = AgentXConfig(
        app_name="CodeReviewBot",
        database=DatabaseConfig.memory(),
        llm=LLMConfig(
            # Use Haiku for quick guardrail checks (fast, cheap)
            routing=LLMLayerConfig(model="claude-haiku-4-5-20251001", temperature=0.1),
            # Sonnet for code review (needs quality)
            default=LLMLayerConfig(model="claude-sonnet-4-6", temperature=0.3),
        ),
    )

    async with AgentXApp(config) as app:

        # Create agents
        guardrail = GuardrailAgent(config=AgentConfig(
            name="guardrail",
            system_prompt="You are a security auditor. Check for OWASP top 10 vulnerabilities.",
            model="claude-haiku-4-5-20251001",  # Fast security check
        ))
        reviewer = ReviewerAgent(
            config=AgentConfig(
                name="reviewer",
                system_prompt=(
                    "You are a senior software engineer doing code review. "
                    "Focus on: correctness, readability, performance, error handling. "
                    "Be specific with line references."
                ),
            ),
            tools=[count_lines, check_complexity],
        )
        suggester = SuggestionAgent(config=AgentConfig(
            name="suggester",
            system_prompt="You are a code improvement expert. Provide actionable, specific suggestions.",
        ))

        app.orchestrator.register_many(guardrail, reviewer, suggester)

        # Pipeline: Guardrail → Review → Suggest
        app.orchestrator.add_pipeline("full_review", agents=["guardrail", "reviewer", "suggester"])

        # --- Sample code to review ---
        sample_code = '''
def get_user_data(user_id, db_connection):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db_connection.execute(query)
    user = result.fetchone()
    if user:
        data = {
            "name": user[1],
            "email": user[2],
            "password": user[3],
        }
        return data
    return None

def process_payment(amount, card_number):
    API_KEY = "sk_live_abc123xyz789"
    import requests
    response = requests.post(
        "https://api.payment.com/charge",
        json={"amount": amount, "card": card_number, "key": API_KEY}
    )
    return eval(response.text)
'''

        print("=" * 60)
        print("  CodeReviewBot — AI Code Review Demo")
        print("=" * 60)
        print(f"\nCode to review:\n{sample_code}")
        print("=" * 60)

        # Run the full review pipeline
        print("\n🔍 Running: Guardrail → Review → Suggest\n")
        result = await app.orchestrator.run_pipeline(
            "full_review",
            initial_message=AgentMessage(content=sample_code),
            context=AgentContext(session_id="review-1", user_id="dev-1"),
        )

        print(f"Final suggestions:\n{result.content}")
        print(f"\nStage: {result.data.get('stage')}")

        # Cost report
        print(f"\n{'=' * 60}")
        print(f"💰 Cost: {app.costs.report()}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
