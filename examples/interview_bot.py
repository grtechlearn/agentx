"""
Example: Complete Interview Bot using AgentX

This demonstrates how to build a full interview coaching system
using the AgentX multi-agent framework.
"""

import asyncio
from agentx import (
    Orchestrator,
    AgentMessage,
    AgentContext,
    MessageType,
    setup_logging,
)
from agentx.agents import (
    InterviewerAgent,
    EvaluatorAgent,
    LearningPathAgent,
    GoalTrackerAgent,
    AnalyticsAgent,
)

# Setup logging
setup_logging(level="INFO")


async def main():
    # 1. Create the orchestrator
    orchestrator = Orchestrator(name="interview-bot")

    # 2. Create specialized agents
    interviewer = InterviewerAgent()
    evaluator = EvaluatorAgent()
    learning_path = LearningPathAgent()
    goal_tracker = GoalTrackerAgent()
    analytics = AnalyticsAgent()

    # 3. Register all agents
    orchestrator.register_many(
        interviewer,
        evaluator,
        learning_path,
        goal_tracker,
        analytics,
    )

    # 4. Define pipelines
    orchestrator.add_pipeline(
        "evaluate_answer",
        agents=["evaluator", "learning_path", "goal_tracker"],
    )

    # 5. Add routing rules
    @orchestrator.route_to("interviewer")
    def route_interview(msg, ctx):
        return msg.data.get("action") in ("start_interview", "next_question", "follow_up")

    @orchestrator.route_to("evaluator")
    def route_evaluation(msg, ctx):
        return msg.data.get("action") == "evaluate"

    @orchestrator.route_to("goal_tracker")
    def route_goals(msg, ctx):
        return msg.data.get("action") in ("set_goal", "check_progress")

    @orchestrator.route_to("analytics")
    def route_analytics(msg, ctx):
        return msg.data.get("action") in ("individual_report", "team_report")

    # Set fallback
    orchestrator.set_fallback("interviewer")

    # 6. Run an interview session
    print("=" * 60)
    print("  AgentX Interview Bot - Demo")
    print("=" * 60)

    # Start interview
    result = await orchestrator.send(
        content="Start a React interview",
        data={
            "action": "start_interview",
            "technology": "React",
            "difficulty": "medium",
            "total_questions": 3,
        },
        session_id="demo-session",
        user_id="user-1",
    )

    print(f"\nInterviewer: {result.content}")
    print(f"Data: {result.data}")

    # Simulate user answer
    result = await orchestrator.send(
        content="useEffect is a hook that runs side effects after render. It takes a callback and a dependency array.",
        data={"action": "next_question"},
        session_id="demo-session",
    )

    print(f"\nInterviewer: {result.content}")

    # Check progress
    result = await orchestrator.send(
        content="How am I doing?",
        data={"action": "check_progress"},
        session_id="demo-session",
    )

    print(f"\nGoal Tracker: {result.content}")

    print("\n" + "=" * 60)
    print("  Interview session complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
