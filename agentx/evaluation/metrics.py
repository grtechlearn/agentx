"""
AgentX - Evaluation Metrics.
Phase 5: Hallucination detection, response quality, accuracy measurement.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


class EvaluationResult(BaseModel):
    """Result of evaluating a response."""

    # Quality scores (0.0 to 1.0)
    relevance: float = 0.0
    faithfulness: float = 0.0  # How faithful to source material
    completeness: float = 0.0
    coherence: float = 0.0
    overall: float = 0.0

    # Hallucination
    hallucination_detected: bool = False
    hallucination_details: list[str] = Field(default_factory=list)
    hallucination_score: float = 0.0  # 0 = no hallucination, 1 = fully hallucinated

    # Context
    context_used: bool = True
    sources_cited: int = 0
    confidence: float = 0.0

    # Metadata
    evaluation_method: str = ""
    raw_scores: dict[str, Any] = Field(default_factory=dict)


class HallucinationDetector:
    """
    Detect hallucinations by comparing response against source documents.

    Strategies:
    1. Claim extraction: Extract claims from response, verify against sources
    2. Entailment check: Check if response is entailed by sources
    3. Keyword coverage: Check if response keywords exist in sources
    4. Contradiction detection: Find contradictions between response and sources
    """

    def __init__(self, llm: Any = None, strict_mode: bool = False):
        self.llm = llm
        self.strict_mode = strict_mode

    async def detect(self, response: str, sources: list[str], query: str = "") -> EvaluationResult:
        """Detect hallucinations in a response given source documents."""
        result = EvaluationResult(evaluation_method="hybrid")

        # Strategy 1: Keyword coverage (fast, no LLM needed)
        keyword_score = self._keyword_coverage(response, sources)
        result.raw_scores["keyword_coverage"] = keyword_score

        # Strategy 2: Claim verification (needs LLM)
        if self.llm:
            claim_result = await self._verify_claims(response, sources, query)
            result.faithfulness = claim_result.get("faithfulness", 0.5)
            result.hallucination_details = claim_result.get("hallucinated_claims", [])
            result.hallucination_score = 1.0 - result.faithfulness
            result.hallucination_detected = result.hallucination_score > 0.3
            result.raw_scores["claim_verification"] = claim_result
        else:
            # Without LLM, use keyword-based estimation
            result.faithfulness = keyword_score
            result.hallucination_score = 1.0 - keyword_score
            result.hallucination_detected = keyword_score < 0.5

        # Combined score
        result.relevance = keyword_score
        result.overall = (result.faithfulness + keyword_score) / 2
        result.confidence = min(keyword_score + 0.2, 1.0)

        return result

    def _keyword_coverage(self, response: str, sources: list[str]) -> float:
        """Check what percentage of response keywords appear in sources."""
        response_words = set(self._extract_meaningful_words(response))
        if not response_words:
            return 1.0

        source_text = " ".join(sources).lower()
        source_words = set(self._extract_meaningful_words(source_text))

        if not source_words:
            return 0.0

        overlap = response_words & source_words
        return len(overlap) / len(response_words)

    @staticmethod
    def _extract_meaningful_words(text: str) -> list[str]:
        """Extract meaningful words (skip common stop words)."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "but", "and", "or",
            "not", "no", "nor", "so", "if", "then", "than", "too", "very", "just",
            "about", "up", "out", "this", "that", "these", "those", "it", "its",
        }
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return [w for w in words if w not in stop_words]

    async def _verify_claims(self, response: str, sources: list[str], query: str) -> dict[str, Any]:
        """Use LLM to extract and verify claims against sources."""
        source_text = "\n---\n".join(sources[:5])  # Limit to 5 sources
        prompt = f"""Analyze this response for factual accuracy against the provided sources.

QUERY: {query}

RESPONSE TO VERIFY:
{response}

SOURCE DOCUMENTS:
{source_text}

For each claim in the response, determine if it is:
1. SUPPORTED - directly supported by sources
2. NOT_SUPPORTED - not mentioned in sources (possible hallucination)
3. CONTRADICTED - contradicts the sources

Return JSON:
{{
    "claims": [
        {{"claim": "...", "status": "SUPPORTED|NOT_SUPPORTED|CONTRADICTED", "evidence": "..."}}
    ],
    "faithfulness": 0.85,
    "hallucinated_claims": ["claim that was not supported"],
    "summary": "Brief assessment"
}}"""

        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                system="You are a factual accuracy evaluator. Be strict about verifying claims against sources.",
            )
            return result
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {"faithfulness": 0.5, "hallucinated_claims": [], "error": str(e)}


class ResponseEvaluator:
    """
    Evaluate LLM response quality across multiple dimensions.
    """

    def __init__(self, llm: Any = None):
        self.llm = llm
        self.hallucination_detector = HallucinationDetector(llm)

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: list[str] | None = None,
        expected: str = "",
    ) -> EvaluationResult:
        """Full evaluation of a response."""
        result = EvaluationResult()

        # Hallucination check if sources available
        if sources:
            hall_result = await self.hallucination_detector.detect(response, sources, query)
            result.faithfulness = hall_result.faithfulness
            result.hallucination_detected = hall_result.hallucination_detected
            result.hallucination_details = hall_result.hallucination_details
            result.hallucination_score = hall_result.hallucination_score

        # LLM-based evaluation
        if self.llm:
            quality = await self._evaluate_quality(query, response, expected)
            result.relevance = quality.get("relevance", 0.5)
            result.completeness = quality.get("completeness", 0.5)
            result.coherence = quality.get("coherence", 0.5)
            result.overall = quality.get("overall", 0.5)
            result.raw_scores["quality"] = quality

        return result

    async def _evaluate_quality(self, query: str, response: str, expected: str) -> dict[str, float]:
        """Use LLM to evaluate response quality."""
        prompt = f"""Rate this response on a scale of 0.0 to 1.0.

QUERY: {query}
RESPONSE: {response}
{"EXPECTED ANSWER: " + expected if expected else ""}

Return JSON:
{{
    "relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "coherence": 0.0-1.0,
    "overall": 0.0-1.0,
    "feedback": "brief feedback"
}}"""

        try:
            return await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                system="You are a response quality evaluator. Be objective and precise.",
            )
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {"relevance": 0.5, "completeness": 0.5, "coherence": 0.5, "overall": 0.5}


class CostTracker:
    """
    Track LLM usage costs per agent, user, and session.

    When a Database instance is provided, cost records persist to DB.
    Otherwise uses in-memory tracking (lost on restart).
    """

    # Approximate costs per 1K tokens (USD)
    MODEL_COSTS: dict[str, dict[str, float]] = {
        "claude-opus-4-6": {"input": 0.015, "output": 0.075},
        "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    def __init__(self, db: Any = None) -> None:
        self._usage: dict[str, dict[str, float]] = {}  # key -> {input_tokens, output_tokens, cost}
        self._total_cost: float = 0.0
        self._db = db  # Optional Database instance

    def track(self, model: str, input_tokens: int, output_tokens: int, key: str = "global",
              agent_name: str = "", session_id: str = "") -> float:
        """Track token usage and return cost."""
        costs = self.MODEL_COSTS.get(model, {"input": 0.003, "output": 0.015})
        cost = (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])

        if key not in self._usage:
            self._usage[key] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "requests": 0}

        self._usage[key]["input_tokens"] += input_tokens
        self._usage[key]["output_tokens"] += output_tokens
        self._usage[key]["cost"] += cost
        self._usage[key]["requests"] += 1
        self._total_cost += cost

        # Write-through to DB
        if self._db and self._db.is_connected:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._db.track_cost(
                    model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                    cost_usd=cost, user_id=key, agent_name=agent_name, session_id=session_id,
                ))
            except RuntimeError:
                pass

        return cost

    async def track_async(self, model: str, input_tokens: int, output_tokens: int,
                          key: str = "global", agent_name: str = "", session_id: str = "") -> float:
        """Async version of track — guaranteed DB persistence."""
        cost = self.track(model, input_tokens, output_tokens, key, agent_name, session_id)
        # Ensure DB write completes (track() fires and forgets, this awaits)
        if self._db and self._db.is_connected:
            await self._db.track_cost(
                model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                cost_usd=cost, user_id=key, agent_name=agent_name, session_id=session_id,
            )
        return cost

    def get_cost(self, key: str = "global") -> float:
        return self._usage.get(key, {}).get("cost", 0.0)

    async def get_cost_from_db(self, user_id: str = "", days: int = 30) -> dict:
        """Get cost summary from database (full history)."""
        if self._db and self._db.is_connected:
            return await self._db.get_cost_summary(user_id=user_id, days=days)
        return {"total_cost": self._total_cost, "total_input": 0, "total_output": 0, "total_calls": 0}

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def report(self) -> dict[str, Any]:
        return {
            "total_cost_usd": f"${self._total_cost:.4f}",
            "breakdown": {k: {**v, "cost": f"${v['cost']:.4f}"} for k, v in self._usage.items()},
        }

    def is_over_budget(self, budget_usd: float) -> bool:
        return self._total_cost >= budget_usd

    def reset(self) -> None:
        self._usage.clear()
        self._total_cost = 0.0
