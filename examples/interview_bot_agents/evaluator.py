"""
AgentX - Evaluator Agent.
Scores answers using RAG context, provides corrections and explanations.
"""

from __future__ import annotations

from typing import Any

from ..core.agent import BaseAgent, AgentConfig
from ..core.context import AgentContext
from ..core.message import AgentMessage


EVALUATOR_SYSTEM_PROMPT = """You are an expert technical answer evaluator. Your job is to:
1. Compare the candidate's answer against the reference material (ground truth)
2. Score the answer accurately
3. Identify what's correct, incorrect, and missing
4. Provide a clear, educational explanation of the correct answer
5. Suggest specific topics for improvement

IMPORTANT: Base your evaluation ONLY on the provided reference material. Do not use knowledge outside the context.

OUTPUT FORMAT (JSON):
{
  "score": 75,
  "max_score": 100,
  "correct_points": ["Point 1 they got right", "Point 2"],
  "incorrect_points": ["What they got wrong and why"],
  "missing_points": ["Important things they didn't mention"],
  "explanation": "Detailed explanation of the correct answer",
  "difficulty_adjustment": "easier|same|harder",
  "confidence_level": "low|medium|high",
  "topics_to_study": ["Topic 1", "Topic 2"],
  "summary": "One-line summary of performance on this question"
}
"""


class EvaluatorAgent(BaseAgent):
    """
    Evaluates interview answers using RAG-retrieved ground truth.
    Provides scores, corrections, and learning recommendations.
    """

    def __init__(self, **kwargs: Any):
        config = AgentConfig(
            name="evaluator",
            role="Answer Evaluator",
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for consistent scoring
            **kwargs,
        )
        super().__init__(config=config)

    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        question = message.data.get("question", "")
        answer = message.content
        technology = context.get("technology", "")
        rag_context = message.data.get("rag_context", context.get("rag_context", ""))

        prompt = f"""Evaluate this technical interview answer.

TECHNOLOGY: {technology}

QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

REFERENCE MATERIAL (ground truth):
{rag_context}

Evaluate the answer against the reference material. Return JSON format."""

        evaluation = await self.think_json(prompt=prompt, context=context)

        # Store evaluation in context for other agents
        evaluations = context.get("evaluations", [])
        evaluations.append({
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
        })
        context.set("evaluations", evaluations)

        # Update running score
        scores = context.get("scores", [])
        scores.append(evaluation.get("score", 0))
        context.set("scores", scores)

        return message.reply(
            content=evaluation.get("explanation", ""),
            data={"evaluation": evaluation},
        )
