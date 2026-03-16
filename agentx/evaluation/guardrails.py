"""
AgentX - Runtime Hallucination Guard & Grounding Enforcement.

LLM Limitation: Hallucination
- Pre-generation: Enforce grounding rules, require sources
- Post-generation: Confidence gating, claim verification, citation check
- Real-time: Low-confidence → retry with stricter prompt or escalate

LLM Limitation: Knowledge Cutoffs
- Source attribution & citation enforcement
- Confidence scoring per claim
- "I don't know" detection and escalation
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


class GroundingConfig(BaseModel):
    """Configuration for grounding enforcement."""

    require_sources: bool = True
    min_confidence: float = 0.6
    max_hallucination_tolerance: float = 0.2
    require_citations: bool = True
    min_sources_cited: int = 1
    retry_on_low_confidence: bool = True
    max_retries: int = 2
    escalate_on_failure: bool = True
    enforce_idk: bool = True  # Force "I don't know" when no context


class GroundedResponse(BaseModel):
    """A response with grounding metadata."""

    content: str = ""
    grounded: bool = False
    confidence: float = 0.0
    sources_cited: int = 0
    citations: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retries: int = 0
    escalated: bool = False
    generation_time_ms: float = 0.0


class HallucinationGuard:
    """
    Runtime hallucination prevention — wraps LLM generation.

    Flow:
    1. Pre-check: Is there enough context to answer?
    2. Generate: LLM response with grounding instructions
    3. Post-check: Verify claims against sources
    4. Gate: If confidence < threshold → retry or escalate

    Usage:
        guard = HallucinationGuard(llm=llm, config=GroundingConfig())

        result = await guard.generate_grounded(
            query="What is React useState?",
            contexts=["React hooks docs..."],
            system_prompt="You are a helpful assistant.",
        )

        if result.grounded:
            return result.content
        elif result.escalated:
            return "I need to escalate this to a human."
        else:
            return result.content  # with warnings
    """

    # Grounding instruction injected into every prompt
    GROUNDING_INSTRUCTION = (
        "\n\nIMPORTANT GROUNDING RULES:\n"
        "1. ONLY use information from the provided context/sources.\n"
        "2. If the context doesn't contain enough information, say "
        "\"I don't have enough information to answer this accurately.\"\n"
        "3. Cite sources using [Source N] format when making claims.\n"
        "4. Do NOT fabricate facts, statistics, dates, or URLs.\n"
        "5. Express uncertainty with phrases like \"Based on the available information...\"\n"
        "6. Distinguish between what the sources say and your interpretation.\n"
    )

    def __init__(
        self,
        llm: Any = None,
        config: GroundingConfig | None = None,
    ):
        self.llm = llm
        self.config = config or GroundingConfig()
        self._stats = {
            "total_generations": 0,
            "grounded": 0,
            "retried": 0,
            "escalated": 0,
            "idk_responses": 0,
        }

    async def generate_grounded(
        self,
        query: str,
        contexts: list[str],
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> GroundedResponse:
        """
        Generate a grounded response with hallucination prevention.
        """
        start = time.monotonic()
        self._stats["total_generations"] += 1

        result = GroundedResponse()
        retries = 0

        # Pre-check: Do we have enough context?
        if self.config.enforce_idk and not contexts:
            result.content = (
                "I don't have enough information to answer this question accurately. "
                "Could you provide more context or rephrase your question?"
            )
            result.grounded = True
            result.confidence = 1.0
            result.warnings.append("no_context_available")
            self._stats["idk_responses"] += 1
            result.generation_time_ms = (time.monotonic() - start) * 1000
            return result

        while retries <= self.config.max_retries:
            # Build grounded prompt
            grounded_system = system_prompt + self.GROUNDING_INSTRUCTION
            context_block = self._format_contexts(contexts)

            prompt = (
                f"CONTEXT:\n{context_block}\n\n"
                f"QUESTION: {query}\n\n"
                f"Answer based ONLY on the context above."
            )

            if retries > 0:
                prompt += (
                    f"\n\nNOTE: Your previous answer was flagged as potentially "
                    f"containing unsupported claims. Be MORE conservative this time. "
                    f"Only state what the sources explicitly say."
                )

            # Generate
            try:
                response = await self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    system=grounded_system,
                    max_tokens=max_tokens,
                )
                result.content = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                result.content = ""
                result.warnings.append(f"generation_error: {e}")
                break

            # Post-check: Verify grounding
            result.confidence = self._estimate_confidence(result.content, contexts)
            result.sources_cited = self._count_citations(result.content)
            result.citations = self._extract_citations(result.content)
            result.claims = self._extract_claims(result.content)

            # Check IDK response
            if self._is_idk_response(result.content):
                result.grounded = True
                result.confidence = 1.0
                self._stats["idk_responses"] += 1
                break

            # Gate: Is it grounded enough?
            is_grounded = True
            if result.confidence < self.config.min_confidence:
                is_grounded = False
                result.warnings.append(
                    f"low_confidence: {result.confidence:.2f} < {self.config.min_confidence}"
                )

            if self.config.require_citations and result.sources_cited < self.config.min_sources_cited:
                is_grounded = False
                result.warnings.append(
                    f"insufficient_citations: {result.sources_cited} < {self.config.min_sources_cited}"
                )

            if is_grounded:
                result.grounded = True
                self._stats["grounded"] += 1
                break

            # Retry or escalate
            if self.config.retry_on_low_confidence and retries < self.config.max_retries:
                retries += 1
                self._stats["retried"] += 1
                result.retries = retries
                continue

            # Exhausted retries
            if self.config.escalate_on_failure:
                result.escalated = True
                result.content = (
                    "I'm not confident in my answer based on the available sources. "
                    "This question may need to be reviewed by a human expert.\n\n"
                    f"Preliminary answer (low confidence): {result.content}"
                )
                self._stats["escalated"] += 1
            break

        result.generation_time_ms = (time.monotonic() - start) * 1000
        return result

    def _format_contexts(self, contexts: list[str]) -> str:
        """Format contexts with source numbers for citation."""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"[Source {i}]\n{ctx}")
        return "\n\n---\n\n".join(parts)

    def _estimate_confidence(self, response: str, contexts: list[str]) -> float:
        """
        Estimate grounding confidence without LLM call.
        Uses keyword overlap as a fast proxy.
        """
        if not response or not contexts:
            return 0.0

        response_words = set(self._meaningful_words(response))
        if not response_words:
            return 0.5

        source_text = " ".join(contexts)
        source_words = set(self._meaningful_words(source_text))

        if not source_words:
            return 0.0

        overlap = len(response_words & source_words) / len(response_words)

        # Bonus for citations
        citation_count = self._count_citations(response)
        citation_bonus = min(citation_count * 0.05, 0.15)

        # Penalty for uncertainty markers
        uncertainty_markers = [
            "i think", "probably", "might", "perhaps", "likely",
            "i'm not sure", "it seems", "it appears",
        ]
        uncertainty_count = sum(1 for m in uncertainty_markers if m in response.lower())
        uncertainty_penalty = min(uncertainty_count * 0.05, 0.15)

        return min(overlap + citation_bonus - uncertainty_penalty, 1.0)

    @staticmethod
    def _meaningful_words(text: str) -> list[str]:
        """Extract meaningful words (skip stop words)."""
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "but", "and", "or", "not", "no", "if", "this", "that", "it", "its",
        }
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return [w for w in words if w not in stop]

    @staticmethod
    def _count_citations(response: str) -> int:
        """Count [Source N] citations in response."""
        return len(re.findall(r"\[Source\s*\d+\]", response, re.IGNORECASE))

    @staticmethod
    def _extract_citations(response: str) -> list[dict[str, Any]]:
        """Extract citation details from response."""
        citations = []
        for match in re.finditer(r"\[Source\s*(\d+)\]", response, re.IGNORECASE):
            citations.append({
                "source_index": int(match.group(1)),
                "position": match.start(),
            })
        return citations

    @staticmethod
    def _extract_claims(response: str) -> list[dict[str, Any]]:
        """Extract factual claims from response (sentence-level)."""
        sentences = re.split(r'[.!?]+', response)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:  # Skip short fragments
                has_citation = bool(re.search(r"\[Source\s*\d+\]", sent, re.IGNORECASE))
                claims.append({
                    "claim": sent[:200],
                    "has_citation": has_citation,
                })
        return claims

    @staticmethod
    def _is_idk_response(response: str) -> bool:
        """Detect "I don't know" type responses."""
        idk_patterns = [
            r"i don'?t have enough information",
            r"i cannot (?:find|determine|answer)",
            r"the (?:context|sources?) (?:doesn'?t|don'?t|do not) (?:contain|mention|provide)",
            r"no (?:relevant )?information (?:is )?available",
            r"i'?m (?:not )?(?:sure|certain|confident)",
            r"this (?:question|topic) is (?:not|outside)",
            r"beyond (?:my|the) (?:available|provided)",
        ]
        text_lower = response.lower()
        return any(re.search(p, text_lower) for p in idk_patterns)

    def stats(self) -> dict[str, Any]:
        total = self._stats["total_generations"]
        return {
            **self._stats,
            "grounding_rate": (
                self._stats["grounded"] / total if total > 0 else 0.0
            ),
            "escalation_rate": (
                self._stats["escalated"] / total if total > 0 else 0.0
            ),
        }
