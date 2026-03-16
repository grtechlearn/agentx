"""
AgentX - Analytics Agent.
Generates reports, insights, and team analytics for enterprise.
"""

from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent, AgentConfig
from ..core.context import AgentContext
from ..core.message import AgentMessage


ANALYTICS_SYSTEM_PROMPT = """You are a data analytics expert for technical interview preparation platforms.

YOUR RESPONSIBILITIES:
- Analyze individual performance trends
- Generate team-level insights (enterprise)
- Identify patterns in strengths and weaknesses
- Predict interview readiness
- Create actionable reports

OUTPUT FORMAT (JSON):
{
  "report_type": "individual|team|summary",
  "period": "weekly|monthly|all_time",
  "metrics": {
    "total_interviews": 15,
    "average_score": 72,
    "score_trend": "improving|stable|declining",
    "best_topic": "React Hooks",
    "worst_topic": "System Design",
    "improvement_rate": 12.5,
    "predicted_readiness_date": "2026-04-20"
  },
  "insights": [
    "Your React Hooks scores improved 25% over the last week",
    "System Design remains your weakest area - consider focused practice"
  ],
  "recommendations": [
    "Focus 60% of practice time on System Design",
    "Take one full mock interview this week"
  ],
  "chart_data": {
    "scores_over_time": [{"date": "2026-03-10", "score": 65}, {"date": "2026-03-15", "score": 72}],
    "topic_scores": {"Hooks": 85, "State": 72, "Performance": 55, "System Design": 40}
  }
}
"""


class AnalyticsAgent(BaseAgent):
    """
    Generates performance analytics and reports.
    Supports individual and team (enterprise) analytics.
    """

    def __init__(self, **kwargs: Any):
        config = AgentConfig(
            name="analytics",
            role="Analytics & Reporting",
            system_prompt=ANALYTICS_SYSTEM_PROMPT,
            temperature=0.2,
            **kwargs,
        )
        super().__init__(config=config)

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        action = message.data.get("action", "individual_report")

        if action == "individual_report":
            return await self._individual_report(message, context)
        elif action == "team_report":
            return await self._team_report(message, context)
        elif action == "predict_readiness":
            return await self._predict_readiness(message, context)
        else:
            return await self._individual_report(message, context)

    async def _individual_report(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        evaluations = context.get("evaluations", [])
        scores = context.get("scores", [])
        goal = context.get("goal", {})
        learning_path = context.get("learning_path", {})

        prompt = f"""Generate an individual performance report.

DATA:
Total interviews: {len(evaluations)}
All scores: {scores}
Goal: {goal}
Learning path: {learning_path}

Detailed evaluations:
"""
        for i, ev in enumerate(evaluations[-10:]):  # Last 10
            eval_data = ev.get("evaluation", {})
            prompt += f"Q{i + 1}: Topic={eval_data.get('topic', 'N/A')}, Score={eval_data.get('score', 0)}\n"

        prompt += "\nGenerate comprehensive analytics report. Return JSON format."

        report = await self.think_json(prompt=prompt, context=context)

        return message.reply(
            content=str(report.get("insights", [])),
            data={"report": report},
        )

    async def _team_report(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        team_data = message.data.get("team_data", [])

        prompt = f"""Generate a team performance report for enterprise.

TEAM DATA:
{team_data}

Analyze team-level patterns, identify who needs help, find common weak areas.
Return JSON format with report_type: "team"."""

        report = await self.think_json(prompt=prompt, context=context)

        return message.reply(
            content=str(report.get("insights", [])),
            data={"report": report},
        )

    async def _predict_readiness(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        scores = context.get("scores", [])
        goal = context.get("goal", {})

        prompt = f"""Predict when this user will be ready for their target interview.

Scores over time: {scores}
Goal: {goal}
Learning path: {context.get("learning_path", {})}

Analyze the trend and predict readiness date. Return JSON format."""

        prediction = await self.think_json(prompt=prompt, context=context)

        return message.reply(
            content=prediction.get("recommendation", ""),
            data={"prediction": prediction},
        )
