"""
AgentX - Goal Tracker Agent.
Sets goals, tracks progress, sends reminders, manages accountability.
"""

from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent, AgentConfig
from ..core.context import AgentContext
from ..core.message import AgentMessage


GOAL_TRACKER_SYSTEM_PROMPT = """You are a goal tracking and accountability coach for technical interview preparation.

YOUR RESPONSIBILITIES:
- Help users set SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)
- Track progress toward goals
- Calculate readiness percentage
- Identify if user is on track or falling behind
- Suggest adjustments when needed
- Provide motivational but honest feedback

OUTPUT FORMAT (JSON):
{
  "goal": {
    "title": "Goal title",
    "technology": "React",
    "target_company": "Google",
    "target_date": "2026-05-15",
    "target_score": 80,
    "created_at": "2026-03-15"
  },
  "progress": {
    "current_score": 65,
    "readiness_percentage": 72,
    "interviews_completed": 12,
    "streak_days": 5,
    "total_practice_hours": 24,
    "topics_mastered": ["Hooks", "State Management"],
    "topics_remaining": ["System Design", "Performance"],
    "on_track": true,
    "days_remaining": 61
  },
  "recommendation": "What to focus on next",
  "milestone_status": "on_track|ahead|behind|at_risk",
  "next_milestone": "Complete React Performance section by April 1"
}
"""


class GoalTrackerAgent(BaseAgent):
    """
    Manages goals, tracks progress, and provides accountability.
    """

    def __init__(self, **kwargs: Any):
        config = AgentConfig(
            name="goal_tracker",
            role="Goal Tracker & Accountability Coach",
            system_prompt=GOAL_TRACKER_SYSTEM_PROMPT,
            temperature=0.3,
            **kwargs,
        )
        super().__init__(config=config)

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        action = message.data.get("action", "check_progress")

        if action == "set_goal":
            return await self._set_goal(message, context)
        elif action == "check_progress":
            return await self._check_progress(message, context)
        elif action == "update_streak":
            return await self._update_streak(message, context)
        else:
            return await self._check_progress(message, context)

    async def _set_goal(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        technology = message.data.get("technology", "")
        target_company = message.data.get("target_company", "")
        target_date = message.data.get("target_date", "")
        target_score = message.data.get("target_score", 80)

        prompt = f"""Help the user set a SMART interview preparation goal.

User's input: {message.content}
Technology: {technology}
Target company: {target_company}
Target date: {target_date}
Target score: {target_score}

Current performance data:
Scores: {context.get("scores", [])}
Topics covered: {context.get("topics_covered", [])}
Learning path: {context.get("learning_path", {})}

Create a structured goal with milestones. Return JSON format."""

        goal = await self.think_json(prompt=prompt, context=context)
        context.set("goal", goal)

        return message.reply(
            content=goal.get("recommendation", "Goal set successfully!"),
            data={"goal": goal},
        )

    async def _check_progress(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        goal = context.get("goal", {})
        scores = context.get("scores", [])
        evaluations = context.get("evaluations", [])
        learning_path = context.get("learning_path", {})

        prompt = f"""Check progress toward the user's goal.

GOAL:
{goal}

PERFORMANCE DATA:
Total interviews: {len(evaluations)}
Scores: {scores}
Average score: {sum(scores) / len(scores) if scores else 0:.0f}
Learning path progress: {learning_path}

Analyze progress, determine if on track, provide recommendations. Return JSON format."""

        progress = await self.think_json(prompt=prompt, context=context)
        context.set("latest_progress", progress)

        return message.reply(
            content=progress.get("recommendation", ""),
            data={"progress": progress},
        )

    async def _update_streak(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        current_streak = context.get("streak_days", 0) + 1
        context.set("streak_days", current_streak)

        prompt = f"""The user has practiced for {current_streak} consecutive days.

Acknowledge their streak and provide motivation.
Current streak: {current_streak} days
Goal: {context.get("goal", {})}
Latest score: {context.get("scores", [0])[-1] if context.get("scores") else "N/A"}

Return JSON with progress update."""

        result = await self.think_json(prompt=prompt, context=context)

        return message.reply(
            content=result.get("recommendation", f"Great job! {current_streak} day streak!"),
            data={"streak": current_streak, "progress": result},
        )
