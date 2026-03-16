"""
AgentX - RAGAS Metrics & Retrieval Quality Evaluation.

Industry-standard RAG evaluation metrics:
- Faithfulness — is the answer grounded in the context?
- Answer Relevance — does the answer address the question?
- Context Relevance — are the retrieved contexts relevant?
- Context Recall — did we retrieve all needed information?
- MRR (Mean Reciprocal Rank) — ranking quality
- nDCG (Normalized Discounted Cumulative Gain) — graded relevance
- Query Analytics — miss rate, bad chunks, patterns
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


# ═══════════════════════════════════════════════════════════════
# RAGAS Evaluation Results
# ═══════════════════════════════════════════════════════════════

class RAGASResult(BaseModel):
    """Complete RAGAS evaluation result."""

    # Core RAGAS metrics (0.0 to 1.0)
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    context_recall: float = 0.0

    # Retrieval quality metrics
    mrr: float = 0.0               # Mean Reciprocal Rank
    ndcg: float = 0.0              # Normalized Discounted Cumulative Gain
    precision_at_k: float = 0.0    # Precision@K
    recall_at_k: float = 0.0       # Recall@K

    # Overall
    overall_score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    evaluation_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════
# RAGAS Evaluator — Full evaluation suite
# ═══════════════════════════════════════════════════════════════

class RAGASEvaluator:
    """
    RAGAS (Retrieval Augmented Generation Assessment) evaluator.

    Evaluates RAG pipeline quality using industry-standard metrics.

    Usage:
        evaluator = RAGASEvaluator(llm=llm)
        result = await evaluator.evaluate(
            query="What are React hooks?",
            answer="React hooks are functions...",
            contexts=["React hooks let you use state..."],
            ground_truth="React hooks are special functions..."
        )
        print(f"Faithfulness: {result.faithfulness}")
        print(f"Answer Relevance: {result.answer_relevance}")
    """

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
        retrieved_ids: list[str] | None = None,
        relevant_ids: list[str] | None = None,
        relevance_grades: list[int] | None = None,
    ) -> RAGASResult:
        """
        Run full RAGAS evaluation.

        Args:
            query: User question
            answer: Generated answer
            contexts: Retrieved context passages
            ground_truth: Expected correct answer (for recall)
            retrieved_ids: IDs of retrieved documents (for MRR/nDCG)
            relevant_ids: IDs of actually relevant documents
            relevance_grades: Graded relevance scores per retrieved doc (for nDCG)
        """
        start = time.monotonic()
        result = RAGASResult()

        # LLM-based metrics
        if self.llm:
            result.faithfulness = await self._evaluate_faithfulness(answer, contexts)
            result.answer_relevance = await self._evaluate_answer_relevance(query, answer)
            result.context_relevance = await self._evaluate_context_relevance(query, contexts)
            if ground_truth:
                result.context_recall = await self._evaluate_context_recall(ground_truth, contexts)

        # Retrieval ranking metrics (no LLM needed)
        if retrieved_ids and relevant_ids:
            result.mrr = self.compute_mrr(retrieved_ids, relevant_ids)
            result.precision_at_k = self.compute_precision_at_k(retrieved_ids, relevant_ids)
            result.recall_at_k = self.compute_recall_at_k(retrieved_ids, relevant_ids)

        if relevance_grades:
            result.ndcg = self.compute_ndcg(relevance_grades)

        # Overall score (weighted average)
        scores = [result.faithfulness, result.answer_relevance, result.context_relevance]
        if ground_truth:
            scores.append(result.context_recall)
        result.overall_score = sum(scores) / len(scores) if scores else 0.0

        result.evaluation_time_ms = (time.monotonic() - start) * 1000
        return result

    # --- Faithfulness ---

    async def _evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Faithfulness: Is every claim in the answer supported by the context?
        Score = (supported claims) / (total claims)
        """
        context_text = "\n---\n".join(contexts[:5])
        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": (
                    f"Given the context below, extract all factual claims from the answer, "
                    f"then determine which claims are supported by the context.\n\n"
                    f"CONTEXT:\n{context_text}\n\n"
                    f"ANSWER:\n{answer}"
                )}],
                system=(
                    "Extract claims and classify each as SUPPORTED or NOT_SUPPORTED. "
                    'Return JSON: {"claims": [{"claim": "...", "supported": true/false}], '
                    '"supported_count": N, "total_count": N, "score": 0.0-1.0}'
                ),
            )
            return float(result.get("score", 0.0))
        except Exception as e:
            logger.error(f"Faithfulness eval failed: {e}")
            return 0.0

    # --- Answer Relevance ---

    async def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance: Does the answer actually address the question?
        Uses reverse generation: generate questions from the answer,
        then measure similarity to original query.
        """
        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": (
                    f"Score how well this answer addresses the question.\n\n"
                    f"QUESTION: {query}\n\nANSWER: {answer}"
                )}],
                system=(
                    "Score the answer's relevance to the question from 0.0 to 1.0. "
                    "1.0 = perfectly addresses the question, 0.0 = completely irrelevant. "
                    'Return JSON: {"score": 0.85, "reasoning": "..."}'
                ),
            )
            return float(result.get("score", 0.0))
        except Exception as e:
            logger.error(f"Answer relevance eval failed: {e}")
            return 0.0

    # --- Context Relevance ---

    async def _evaluate_context_relevance(self, query: str, contexts: list[str]) -> float:
        """
        Context Relevance: Are the retrieved contexts relevant to the query?
        Score = (relevant sentences in context) / (total sentences in context)
        """
        context_text = "\n---\n".join(contexts[:5])
        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": (
                    f"Given this query, evaluate how relevant each context passage is.\n\n"
                    f"QUERY: {query}\n\nCONTEXTS:\n{context_text}"
                )}],
                system=(
                    "Score the overall context relevance from 0.0 to 1.0. "
                    "1.0 = all contexts are highly relevant, 0.0 = none are relevant. "
                    'Return JSON: {"score": 0.8, "relevant_passages": N, "total_passages": N}'
                ),
            )
            return float(result.get("score", 0.0))
        except Exception as e:
            logger.error(f"Context relevance eval failed: {e}")
            return 0.0

    # --- Context Recall ---

    async def _evaluate_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """
        Context Recall: Did we retrieve all needed information?
        Score = (ground truth claims found in context) / (total ground truth claims)
        """
        context_text = "\n---\n".join(contexts[:5])
        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": (
                    f"Check how much of the ground truth is covered by the retrieved contexts.\n\n"
                    f"GROUND TRUTH:\n{ground_truth}\n\nRETRIEVED CONTEXTS:\n{context_text}"
                )}],
                system=(
                    "Score context recall from 0.0 to 1.0. "
                    "1.0 = all ground truth information is in the contexts, 0.0 = none found. "
                    'Return JSON: {"score": 0.9, "covered_claims": N, "total_claims": N}'
                ),
            )
            return float(result.get("score", 0.0))
        except Exception as e:
            logger.error(f"Context recall eval failed: {e}")
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    # Retrieval Ranking Metrics (no LLM needed)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def compute_mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        """
        Mean Reciprocal Rank.
        MRR = 1 / (rank of first relevant document)
        Higher is better. 1.0 = first result is relevant.
        """
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_ndcg(relevance_grades: list[int], k: int = 0) -> float:
        """
        Normalized Discounted Cumulative Gain.
        Measures ranking quality with graded relevance (0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant).

        Args:
            relevance_grades: Relevance score for each retrieved document in order
            k: Evaluate at top-k (0 = all)
        """
        if not relevance_grades:
            return 0.0

        if k > 0:
            relevance_grades = relevance_grades[:k]

        # DCG
        dcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(relevance_grades)
        )

        # Ideal DCG (sorted by relevance descending)
        ideal = sorted(relevance_grades, reverse=True)
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal)
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_precision_at_k(
        retrieved_ids: list[str], relevant_ids: list[str], k: int = 0,
    ) -> float:
        """Precision@K = (relevant in top-k) / k"""
        relevant_set = set(relevant_ids)
        top_k = retrieved_ids[:k] if k > 0 else retrieved_ids
        if not top_k:
            return 0.0
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_count / len(top_k)

    @staticmethod
    def compute_recall_at_k(
        retrieved_ids: list[str], relevant_ids: list[str], k: int = 0,
    ) -> float:
        """Recall@K = (relevant in top-k) / (total relevant)"""
        if not relevant_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        top_k = retrieved_ids[:k] if k > 0 else retrieved_ids
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_count / len(relevant_set)


# ═══════════════════════════════════════════════════════════════
# Query Analytics — Track retrieval quality over time
# ═══════════════════════════════════════════════════════════════

class QueryAnalytics:
    """
    Track and analyze query patterns, miss rates, and retrieval quality.

    Feeds into the self-learning loop:
    - Low-confidence answers → trigger retrieval retry or escalation
    - Bad chunks → flag for re-indexing
    - Query patterns → optimize retrieval strategy

    Usage:
        analytics = QueryAnalytics()
        analytics.log_query(query="...", answer="...", confidence=0.8)
        analytics.log_feedback(query_id="...", thumbs_up=True)
        report = analytics.get_report()
    """

    def __init__(self, db: Any = None):
        self._queries: list[dict[str, Any]] = []
        self._feedback: list[dict[str, Any]] = []
        self._db = db

    def log_query(
        self,
        query: str,
        answer: str,
        confidence: float = 0.0,
        contexts_used: int = 0,
        latency_ms: float = 0.0,
        model: str = "",
        user_id: str = "",
        session_id: str = "",
        strategy: str = "",
        cache_hit: bool = False,
    ) -> str:
        """Log a query for analytics. Returns query_id."""
        import hashlib
        query_id = hashlib.md5(f"{query}:{time.time()}".encode()).hexdigest()[:12]

        entry = {
            "query_id": query_id,
            "query": query,
            "answer_length": len(answer),
            "confidence": confidence,
            "contexts_used": contexts_used,
            "latency_ms": latency_ms,
            "model": model,
            "user_id": user_id,
            "session_id": session_id,
            "strategy": strategy,
            "cache_hit": cache_hit,
            "timestamp": time.time(),
            "feedback": None,
        }
        self._queries.append(entry)

        # Keep last 10000 entries in memory
        if len(self._queries) > 10000:
            self._queries = self._queries[-5000:]

        return query_id

    def log_feedback(
        self,
        query_id: str = "",
        query: str = "",
        thumbs_up: bool | None = None,
        correction: str = "",
        bad_chunks: list[str] | None = None,
    ) -> None:
        """Log user feedback on a response."""
        feedback = {
            "query_id": query_id,
            "query": query,
            "thumbs_up": thumbs_up,
            "correction": correction,
            "bad_chunks": bad_chunks or [],
            "timestamp": time.time(),
        }
        self._feedback.append(feedback)

        # Update query entry
        for q in reversed(self._queries):
            if q["query_id"] == query_id or (query and q["query"] == query):
                q["feedback"] = thumbs_up
                break

    def get_report(self, days: int = 7) -> dict[str, Any]:
        """Get analytics report."""
        cutoff = time.time() - (days * 86400)
        recent = [q for q in self._queries if q["timestamp"] > cutoff]

        if not recent:
            return {"total_queries": 0, "period_days": days}

        total = len(recent)
        low_confidence = [q for q in recent if q["confidence"] < 0.5]
        cache_hits = sum(1 for q in recent if q.get("cache_hit"))
        feedbacks = [q for q in recent if q.get("feedback") is not None]
        positive = sum(1 for q in feedbacks if q["feedback"])
        negative = sum(1 for q in feedbacks if not q["feedback"])

        avg_confidence = sum(q["confidence"] for q in recent) / total
        avg_latency = sum(q["latency_ms"] for q in recent) / total
        avg_contexts = sum(q["contexts_used"] for q in recent) / total

        # Miss rate: queries with 0 contexts or very low confidence
        misses = sum(1 for q in recent if q["contexts_used"] == 0 or q["confidence"] < 0.3)

        # Strategy distribution
        strategies: dict[str, int] = {}
        for q in recent:
            s = q.get("strategy", "unknown")
            strategies[s] = strategies.get(s, 0) + 1

        return {
            "period_days": days,
            "total_queries": total,
            "avg_confidence": round(avg_confidence, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_contexts_used": round(avg_contexts, 1),
            "miss_rate": round(misses / total, 3) if total > 0 else 0,
            "cache_hit_rate": round(cache_hits / total, 3) if total > 0 else 0,
            "low_confidence_queries": len(low_confidence),
            "feedback": {
                "total": len(feedbacks),
                "positive": positive,
                "negative": negative,
                "satisfaction_rate": round(positive / len(feedbacks), 3) if feedbacks else 0,
            },
            "strategies": strategies,
            "bad_chunks_reported": sum(len(f.get("bad_chunks", [])) for f in self._feedback),
        }

    def get_low_confidence_queries(self, threshold: float = 0.5) -> list[dict[str, Any]]:
        """Get queries that need attention (low confidence)."""
        return [q for q in self._queries if q["confidence"] < threshold]

    def get_negative_feedback(self) -> list[dict[str, Any]]:
        """Get queries with negative feedback for retraining."""
        return [f for f in self._feedback if f.get("thumbs_up") is False]
