"""
AgentX - Interviewer Agent.
Conducts technical interviews, asks questions, handles follow-ups.
"""

from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent, AgentConfig
from ..core.context import AgentContext
from ..core.message import AgentMessage, MessageType


INTERVIEWER_SYSTEM_PROMPT = """You are an expert technical interviewer. Your role is to conduct realistic technical interviews.

BEHAVIOR:
- Ask ONE question at a time
- Start with the difficulty level specified in the context
- Adjust difficulty based on the candidate's answers
- Ask follow-up questions when answers are vague or incomplete
- Be professional but encouraging
- If the candidate says "I don't know", acknowledge it and move on
- Track which topics have been covered

QUESTION STYLE:
- Mix conceptual and practical questions
- Include scenario-based questions ("How would you...")
- Ask about real-world trade-offs
- Include debugging scenarios when appropriate

OUTPUT FORMAT (JSON):
{
  "question": "Your question here",
  "topic": "The topic being tested",
  "difficulty": "easy|medium|hard",
  "question_number": 1,
  "is_follow_up": false,
  "hint": "Optional hint if candidate struggles"
}
"""


class InterviewerAgent(BaseAgent):
    """
    Conducts technical interviews.
    Asks questions, adapts difficulty, handles follow-ups.
    """

    def __init__(self, **kwargs: Any):
        config = AgentConfig(
            name="interviewer",
            role="Technical Interviewer",
            system_prompt=INTERVIEWER_SYSTEM_PROMPT,
            temperature=0.7,
            **kwargs,
        )
        super().__init__(config=config)

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        action = message.data.get("action", "ask_question")

        if action == "start_interview":
            return await self._start_interview(message, context)
        elif action == "next_question":
            return await self._next_question(message, context)
        elif action == "follow_up":
            return await self._follow_up(message, context)
        else:
            return await self._next_question(message, context)

    async def _start_interview(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        technology = message.data.get("technology", "React")
        difficulty = message.data.get("difficulty", "medium")
        total_questions = message.data.get("total_questions", 10)
        company = message.data.get("company", "")

        context.set("technology", technology)
        context.set("difficulty", difficulty)
        context.set("total_questions", total_questions)
        context.set("current_question", 0)
        context.set("company", company)
        context.set("topics_covered", [])

        prompt = f"""Start a {technology} technical interview.
Difficulty: {difficulty}
Total questions planned: {total_questions}
{"Target company: " + company if company else ""}

Ask the first question. Remember to return JSON format."""

        # Use RAG context if available
        rag_context = context.get("rag_context", "")
        if rag_context:
            prompt += f"\n\nUse this reference material for accurate questions:\n{rag_context}"

        response = await self.think_json(prompt=prompt, context=context)
        context.set("current_question", 1)

        topics = context.get("topics_covered", [])
        if response.get("topic"):
            topics.append(response["topic"])
            context.set("topics_covered", topics)

        return message.reply(
            content=response.get("question", ""),
            data={"interview_data": response, "status": "in_progress"},
        )

    async def _next_question(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        current = context.get("current_question", 0) + 1
        total = context.get("total_questions", 10)
        technology = context.get("technology", "React")
        topics_covered = context.get("topics_covered", [])

        if current > total:
            return message.reply(
                content="Interview complete! Let me evaluate your performance.",
                data={"status": "completed", "total_questions": total},
            )

        # Include candidate's previous answer for context
        previous_answer = message.content
        evaluation_hint = message.data.get("evaluation_summary", "")

        prompt = f"""Continue the {technology} interview.
Question number: {current}/{total}
Current difficulty: {context.get("difficulty", "medium")}
Topics already covered: {topics_covered}
Candidate's last answer: {previous_answer}
{"Evaluation of last answer: " + evaluation_hint if evaluation_hint else ""}

Ask the next question on a DIFFERENT topic. Return JSON format."""

        rag_context = context.get("rag_context", "")
        if rag_context:
            prompt += f"\n\nReference material:\n{rag_context}"

        response = await self.think_json(prompt=prompt, context=context)
        context.set("current_question", current)

        if response.get("topic"):
            topics_covered.append(response["topic"])
            context.set("topics_covered", topics_covered)

        return message.reply(
            content=response.get("question", ""),
            data={"interview_data": response, "status": "in_progress"},
        )

    async def _follow_up(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        technology = context.get("technology", "React")
        previous_answer = message.content

        prompt = f"""The candidate's answer to the {technology} question was incomplete.
Their answer: {previous_answer}

Ask a follow-up question to probe deeper. Return JSON format with is_follow_up: true."""

        response = await self.think_json(prompt=prompt, context=context)
        return message.reply(
            content=response.get("question", ""),
            data={"interview_data": response, "status": "in_progress"},
        )
