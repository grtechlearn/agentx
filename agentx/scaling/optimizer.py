"""
AgentX - Scaling, Latency Optimization & Self-Learning.
Phase 6: Operational cost, scaling, self-learning, reducing LLM dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


# --- Smart Model Router ---

class ModelConfig(BaseModel):
    """Configuration for a model in the router."""

    name: str
    provider: str = "anthropic"
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015
    max_tokens: int = 4096
    latency_ms: int = 1000  # average
    quality_score: float = 0.9  # 0-1
    supports_tools: bool = True
    supports_json: bool = True


class ModelRouter:
    """
    Intelligently route requests to the best model based on:
    - Task complexity
    - Cost budget
    - Latency requirements
    - Quality requirements

    Reduces cost by using cheaper models for simple tasks.
    """

    def __init__(self) -> None:
        self.models: dict[str, ModelConfig] = {}
        self._task_history: dict[str, str] = {}  # task_type -> best_model

    def register_model(self, config: ModelConfig) -> None:
        self.models[config.name] = config

    def setup_defaults(self) -> None:
        """Register common models with their characteristics."""
        defaults = [
            ModelConfig(name="claude-opus-4-6", provider="anthropic", cost_per_1k_input=0.015, cost_per_1k_output=0.075, latency_ms=3000, quality_score=1.0),
            ModelConfig(name="claude-sonnet-4-6", provider="anthropic", cost_per_1k_input=0.003, cost_per_1k_output=0.015, latency_ms=1500, quality_score=0.9),
            ModelConfig(name="claude-haiku-4-5-20251001", provider="anthropic", cost_per_1k_input=0.0008, cost_per_1k_output=0.004, latency_ms=500, quality_score=0.75),
            ModelConfig(name="gpt-4o", provider="openai", cost_per_1k_input=0.005, cost_per_1k_output=0.015, latency_ms=2000, quality_score=0.9),
            ModelConfig(name="gpt-4o-mini", provider="openai", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, latency_ms=500, quality_score=0.7),
        ]
        for m in defaults:
            self.register_model(m)

    def select_model(
        self,
        task_complexity: str = "medium",  # simple, medium, complex
        max_cost_per_call: float = 0.05,
        max_latency_ms: int = 5000,
        min_quality: float = 0.7,
        prefer: str = "cost",  # cost, quality, speed
    ) -> ModelConfig | None:
        """Select the best model for the task."""
        candidates = []
        for model in self.models.values():
            if model.latency_ms > max_latency_ms:
                continue
            if model.quality_score < min_quality:
                continue
            # Estimate cost for typical request
            est_cost = (1000 / 1000 * model.cost_per_1k_input) + (500 / 1000 * model.cost_per_1k_output)
            if est_cost > max_cost_per_call:
                continue
            candidates.append((model, est_cost))

        if not candidates:
            # Fallback: return cheapest available
            if self.models:
                return min(self.models.values(), key=lambda m: m.cost_per_1k_input)
            return None

        # Sort by preference
        if prefer == "cost":
            candidates.sort(key=lambda x: x[1])
        elif prefer == "quality":
            candidates.sort(key=lambda x: x[0].quality_score, reverse=True)
        elif prefer == "speed":
            candidates.sort(key=lambda x: x[0].latency_ms)

        # For simple tasks, prefer cheaper models
        if task_complexity == "simple" and prefer != "quality":
            candidates.sort(key=lambda x: x[1])

        return candidates[0][0]


# --- Request Queue & Rate Limiter ---

class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    async def acquire(self) -> bool:
        """Wait until rate limit allows a request."""
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self.window_seconds]

        if len(self._timestamps) >= self.max_requests:
            wait_time = self._timestamps[0] + self.window_seconds - now
            if wait_time > 0:
                logger.info(f"Rate limited. Waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self._timestamps.append(time.time())
        return True

    @property
    def remaining(self) -> int:
        now = time.time()
        active = [t for t in self._timestamps if now - t < self.window_seconds]
        return max(0, self.max_requests - len(active))


# --- Self-Learning System ---

class LearnedRule(BaseModel):
    """A rule learned from past interactions."""

    pattern: str  # Query pattern
    response: str  # Cached/learned response
    confidence: float = 0.0
    times_used: int = 0
    times_validated: int = 0
    source: str = "auto"  # auto, manual, fine-tune
    created_at: float = Field(default_factory=time.time)


class SelfLearner:
    """
    Self-learning system that reduces LLM dependency over time.

    Strategies:
    1. Response caching with semantic matching
    2. Pattern extraction from successful responses
    3. Rule-based shortcuts for common queries
    4. Training data collection for future fine-tuning
    5. Confidence-based LLM bypass
    """

    def __init__(self, storage_path: str = "./data/learned", min_confidence: float = 0.9):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self._rules: dict[str, LearnedRule] = {}
        self._training_data: list[dict[str, Any]] = []
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        rules_file = self.storage_path / "rules.json"
        if rules_file.exists():
            data = json.loads(rules_file.read_text())
            for item in data:
                rule = LearnedRule(**item)
                self._rules[rule.pattern] = rule
        self._loaded = True

    async def check(self, query: str) -> str | None:
        """Check if we have a learned response for this query."""
        await self._ensure_loaded()
        normalized = query.strip().lower()

        # Exact match
        rule = self._rules.get(normalized)
        if rule and rule.confidence >= self.min_confidence:
            rule.times_used += 1
            logger.info(f"Self-learned response used (confidence: {rule.confidence:.2f})")
            return rule.response

        # Fuzzy match (simple word overlap)
        query_words = set(normalized.split())
        best_match = None
        best_score = 0.0
        for pattern, rule in self._rules.items():
            if rule.confidence < self.min_confidence:
                continue
            pattern_words = set(pattern.split())
            if not pattern_words:
                continue
            overlap = len(query_words & pattern_words) / max(len(query_words), len(pattern_words))
            if overlap > 0.85 and overlap > best_score:
                best_match = rule
                best_score = overlap

        if best_match:
            best_match.times_used += 1
            return best_match.response

        return None

    async def learn(self, query: str, response: str, score: float = 0.0, validated: bool = False) -> None:
        """Learn from a successful interaction."""
        await self._ensure_loaded()
        normalized = query.strip().lower()
        confidence = score if score > 0 else (0.95 if validated else 0.7)

        if normalized in self._rules:
            existing = self._rules[normalized]
            existing.confidence = max(existing.confidence, confidence)
            existing.times_validated += 1 if validated else 0
            if confidence > existing.confidence:
                existing.response = response
        else:
            self._rules[normalized] = LearnedRule(
                pattern=normalized,
                response=response,
                confidence=confidence,
                times_validated=1 if validated else 0,
            )

        # Save rules
        await self._save_rules()

        # Collect training data
        self._training_data.append({
            "query": query,
            "response": response,
            "score": score,
            "timestamp": time.time(),
        })

    async def _save_rules(self) -> None:
        rules_file = self.storage_path / "rules.json"
        data = [r.model_dump() for r in self._rules.values()]
        rules_file.write_text(json.dumps(data, indent=2))

    async def export_training_data(self, format: str = "jsonl") -> str:
        """Export collected data for fine-tuning."""
        output_path = self.storage_path / f"training_data.{format}"

        if format == "jsonl":
            with open(output_path, "w") as f:
                for item in self._training_data:
                    f.write(json.dumps(item) + "\n")
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(self._training_data, f, indent=2)

        logger.info(f"Exported {len(self._training_data)} training samples to {output_path}")
        return str(output_path)

    def stats(self) -> dict[str, Any]:
        high_conf = sum(1 for r in self._rules.values() if r.confidence >= self.min_confidence)
        return {
            "total_rules": len(self._rules),
            "high_confidence_rules": high_conf,
            "training_samples": len(self._training_data),
            "total_uses": sum(r.times_used for r in self._rules.values()),
            "llm_calls_saved": sum(r.times_used for r in self._rules.values() if r.confidence >= self.min_confidence),
        }


# --- Latency Optimizer ---

class LatencyOptimizer:
    """
    Optimize response latency through:
    1. Parallel retrieval + generation
    2. Streaming responses
    3. Prefetching
    4. Connection pooling awareness
    """

    def __init__(self) -> None:
        self._latency_history: dict[str, list[float]] = {}

    def record(self, operation: str, duration_ms: float) -> None:
        if operation not in self._latency_history:
            self._latency_history[operation] = []
        self._latency_history[operation].append(duration_ms)
        # Keep last 1000 measurements
        if len(self._latency_history[operation]) > 1000:
            self._latency_history[operation] = self._latency_history[operation][-500:]

    def avg_latency(self, operation: str) -> float:
        history = self._latency_history.get(operation, [])
        return sum(history) / len(history) if history else 0

    def p95_latency(self, operation: str) -> float:
        history = self._latency_history.get(operation, [])
        if not history:
            return 0
        sorted_h = sorted(history)
        idx = int(len(sorted_h) * 0.95)
        return sorted_h[min(idx, len(sorted_h) - 1)]

    def report(self) -> dict[str, Any]:
        return {
            op: {
                "avg_ms": f"{self.avg_latency(op):.0f}",
                "p95_ms": f"{self.p95_latency(op):.0f}",
                "samples": len(self._latency_history[op]),
            }
            for op in self._latency_history
        }

    async def timed(self, operation: str, coro: Any) -> Any:
        """Execute a coroutine and record its latency."""
        start = time.monotonic()
        try:
            result = await coro
            return result
        finally:
            duration = (time.monotonic() - start) * 1000
            self.record(operation, duration)
