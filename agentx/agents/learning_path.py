"""
AgentX - Learning Path Agent.
Analyzes weaknesses, builds personalized study plans.
"""

from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent, AgentConfig
from ..core.context import AgentContext
from ..core.message import AgentMessage


LEARNING_PATH_SYSTEM_PROMPT = """You are an expert learning path designer for technical skills.
Your job is to analyze interview performance and create personalized study plans.

APPROACH:
- Identify knowledge gaps from evaluation results
- Prioritize topics by importance and weakness severity
- Create an ordered learning sequence (prerequisites first)
- Set realistic timelines based on the user's goal
- Include specific resources and practice suggestions

OUTPUT FORMAT (JSON):
{
  "overall_assessment": "Brief assessment of current level",
  "current_level": "beginner|intermediate|advanced",
  "readiness_score": 65,
  "strengths": ["Topic 1", "Topic 2"],
  "weaknesses": ["Topic 1", "Topic 2"],
  "learning_path": [
    {
      "order": 1,
      "topic": "Topic name",
      "priority": "high|medium|low",
      "estimated_hours": 4,
      "reason": "Why this topic is important",
      "subtopics": ["Subtopic 1", "Subtopic 2"],
      "practice_suggestions": ["Build X", "Practice Y"]
    }
  ],
  "estimated_total_hours": 40,
  "recommended_daily_hours": 2,
  "estimated_days_to_ready": 20
}
"""


class LearningPathAgent(BaseAgent):
    """
    Builds personalized learning paths based on interview performance.
    """

    def __init__(self, **kwargs: Any):
        config = AgentConfig(
            name="learning_path",
            role="Learning Path Designer",
            system_prompt=LEARNING_PATH_SYSTEM_PROMPT,
            temperature=0.3,
            **kwargs,
        )
        super().__init__(config=config)

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        action = message.data.get("action", "generate")

        if action == "generate":
            return await self._generate_path(message, context)
        elif action == "update":
            return await self._update_path(message, context)
        else:
            return await self._generate_path(message, context)

    async def _generate_path(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        technology = context.get("technology", message.data.get("technology", ""))
        evaluations = context.get("evaluations", [])
        scores = context.get("scores", [])
        goal = message.data.get("goal", "")
        deadline = message.data.get("deadline", "")

        prompt = f"""Generate a personalized learning path.

TECHNOLOGY: {technology}
{"GOAL: " + goal if goal else ""}
{"DEADLINE: " + deadline if deadline else ""}

INTERVIEW RESULTS:
Average Score: {sum(scores) / len(scores) if scores else 0:.0f}/100
Total Questions: {len(evaluations)}

DETAILED RESULTS:
"""
        for i, ev in enumerate(evaluations):
            eval_data = ev.get("evaluation", {})
            prompt += f"""
Question {i + 1}: {ev.get("question", "")}
Score: {eval_data.get("score", 0)}/100
Correct: {eval_data.get("correct_points", [])}
Missing: {eval_data.get("missing_points", [])}
Topics to study: {eval_data.get("topics_to_study", [])}
"""

        prompt += "\n\nCreate a structured learning path. Return JSON format."

        path = await self.think_json(prompt=prompt, context=context)
        context.set("learning_path", path)

        return message.reply(
            content=path.get("overall_assessment", ""),
            data={"learning_path": path},
        )

    async def _update_path(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        current_path = context.get("learning_path", {})
        new_evaluations = context.get("evaluations", [])

        prompt = f"""Update this learning path based on new interview results.

CURRENT PATH:
{current_path}

NEW RESULTS:
{new_evaluations[-3:]}

Adjust priorities, mark completed topics, add new weak areas. Return JSON format."""

        path = await self.think_json(prompt=prompt, context=context)
        context.set("learning_path", path)

        return message.reply(
            content=path.get("overall_assessment", ""),
            data={"learning_path": path},
        )
