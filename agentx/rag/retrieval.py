"""
AgentX - Advanced Retrieval Components.
BM25 search, cross-encoder re-ranking, semantic cache, query rewrite.

These plug into RAGEngine for production-grade retrieval quality.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from .engine import Document, BaseEmbedder

logger = logging.getLogger("agentx")


# ═══════════════════════════════════════════════════════════════
# BM25 Search — True BM25 implementation for keyword retrieval
# ═══════════════════════════════════════════════════════════════

class BM25Index:
    """
    BM25 (Best Matching 25) keyword search index.

    Used alongside vector search for hybrid retrieval.
    BM25 captures exact keyword matches that embedding similarity can miss.

    Usage:
        bm25 = BM25Index()
        bm25.add_documents(documents)
        results = bm25.search("react hooks tutorial", limit=10)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation. Higher = more weight to term frequency.
            b: Length normalization. 0 = no normalization, 1 = full normalization.
        """
        self.k1 = k1
        self.b = b
        self._documents: list[Document] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._df: Counter = Counter()  # document frequency per term
        self._tf: list[Counter] = []   # term frequency per document
        self._n: int = 0

    def add_documents(self, documents: list[Document]) -> None:
        """Index documents for BM25 search."""
        for doc in documents:
            terms = self._tokenize(doc.content)
            tf = Counter(terms)
            self._tf.append(tf)
            self._doc_lengths.append(len(terms))
            self._documents.append(doc)

            # Update document frequency
            for term in set(terms):
                self._df[term] += 1

        self._n = len(self._documents)
        self._avg_dl = sum(self._doc_lengths) / self._n if self._n > 0 else 0

    def search(self, query: str, limit: int = 10, filters: dict[str, Any] | None = None) -> list[Document]:
        """Search using BM25 scoring."""
        if not self._documents:
            return []

        query_terms = self._tokenize(query)
        scores: list[tuple[int, float]] = []

        for idx in range(self._n):
            # Apply metadata filters
            if filters:
                doc = self._documents[idx]
                if not self._matches_filters(doc, filters):
                    continue

            score = self._score_document(idx, query_terms)
            if score > 0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:limit]:
            doc = self._documents[idx]
            result = Document(
                id=doc.id, content=doc.content,
                metadata=doc.metadata, score=score,
            )
            results.append(result)

        return results

    def _score_document(self, doc_idx: int, query_terms: list[str]) -> float:
        """Calculate BM25 score for a document given query terms."""
        score = 0.0
        dl = self._doc_lengths[doc_idx]
        tf = self._tf[doc_idx]

        for term in query_terms:
            if term not in self._df:
                continue

            # IDF component
            df = self._df[term]
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)

            # TF component with saturation and length normalization
            term_freq = tf.get(term, 0)
            tf_norm = (term_freq * (self.k1 + 1)) / (
                term_freq + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            )

            score += idf * tf_norm

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into terms (lowercase, alphanumeric)."""
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    @staticmethod
    def _matches_filters(doc: Document, filters: dict[str, Any]) -> bool:
        """Check if document matches metadata filters."""
        for key, value in filters.items():
            doc_val = doc.metadata.get(key)
            if isinstance(value, list):
                if doc_val not in value:
                    return False
            elif doc_val != value:
                return False
        return True

    def clear(self) -> None:
        """Clear the index."""
        self._documents.clear()
        self._doc_lengths.clear()
        self._tf.clear()
        self._df.clear()
        self._n = 0
        self._avg_dl = 0.0

    @property
    def size(self) -> int:
        return self._n


# ═══════════════════════════════════════════════════════════════
# Cross-Encoder Re-Ranker — Precision re-ranking with cross-encoder models
# ═══════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Cross-encoder re-ranking for precision retrieval.

    Two-stage retrieval:
    1. Fast bi-encoder (embedding) retrieves broad candidates
    2. Cross-encoder scores each (query, document) pair for precise ranking

    Supports:
    - Local models via sentence-transformers
    - API-based re-ranking (Cohere, Voyage)
    - LLM-based fallback

    Usage:
        reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked = await reranker.rerank(query, candidates, limit=5)
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        provider: str = "local",  # local, cohere, voyage, llm
        api_key: str = "",
        llm: Any = None,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model
        self.provider = provider
        self.api_key = api_key
        self.llm = llm
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        limit: int = 5,
    ) -> list[Document]:
        """Re-rank documents using cross-encoder scoring."""
        if not documents:
            return []

        if self.provider == "local":
            return await self._rerank_local(query, documents, limit)
        elif self.provider == "cohere":
            return await self._rerank_cohere(query, documents, limit)
        elif self.provider == "voyage":
            return await self._rerank_voyage(query, documents, limit)
        elif self.provider == "llm":
            return await self._rerank_llm(query, documents, limit)
        else:
            logger.warning(f"Unknown reranker provider: {self.provider}, using LLM fallback")
            return await self._rerank_llm(query, documents, limit)

    async def _rerank_local(
        self, query: str, documents: list[Document], limit: int,
    ) -> list[Document]:
        """Re-rank using local cross-encoder model (sentence-transformers)."""
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)

            pairs = [(query, doc.content) for doc in documents]
            scores = self._model.predict(pairs, batch_size=self.batch_size)

            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results = []
            for doc, score in scored_docs[:limit]:
                result = Document(
                    id=doc.id, content=doc.content,
                    metadata={**doc.metadata, "rerank_score": float(score)},
                    score=float(score),
                )
                results.append(result)
            return results

        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to LLM reranking")
            return await self._rerank_llm(query, documents, limit)

    async def _rerank_cohere(
        self, query: str, documents: list[Document], limit: int,
    ) -> list[Document]:
        """Re-rank using Cohere Rerank API."""
        try:
            import cohere
            client = cohere.AsyncClient(api_key=self.api_key)
            response = await client.rerank(
                model=self.model_name or "rerank-english-v3.0",
                query=query,
                documents=[doc.content for doc in documents],
                top_n=limit,
            )
            results = []
            for item in response.results:
                doc = documents[item.index]
                results.append(Document(
                    id=doc.id, content=doc.content,
                    metadata={**doc.metadata, "rerank_score": item.relevance_score},
                    score=item.relevance_score,
                ))
            return results
        except ImportError:
            logger.error("cohere not installed: pip install cohere")
            return documents[:limit]

    async def _rerank_voyage(
        self, query: str, documents: list[Document], limit: int,
    ) -> list[Document]:
        """Re-rank using Voyage AI Rerank API."""
        try:
            import voyageai
            client = voyageai.AsyncClient(api_key=self.api_key) if self.api_key else voyageai.AsyncClient()
            response = await client.rerank(
                model=self.model_name or "rerank-2",
                query=query,
                documents=[doc.content for doc in documents],
                top_k=limit,
            )
            results = []
            for item in response.results:
                doc = documents[item.index]
                results.append(Document(
                    id=doc.id, content=doc.content,
                    metadata={**doc.metadata, "rerank_score": item.relevance_score},
                    score=item.relevance_score,
                ))
            return results
        except ImportError:
            logger.error("voyageai not installed: pip install voyageai")
            return documents[:limit]

    async def _rerank_llm(
        self, query: str, documents: list[Document], limit: int,
    ) -> list[Document]:
        """Fallback: Re-rank using LLM scoring."""
        if not self.llm:
            logger.warning("No LLM available for re-ranking, returning original order")
            return documents[:limit]

        doc_summaries = "\n".join(
            f"[{i}] {doc.content[:300]}" for i, doc in enumerate(documents)
        )
        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": (
                    f"Query: {query}\n\nDocuments:\n{doc_summaries}\n\n"
                    "Score each document 0-10 for relevance to the query."
                )}],
                system="Return JSON: {\"scores\": [8, 3, 9, ...]} one score per document.",
                schema={"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "number"}}}},
            )
            scores = response.get("scores", [])
            scored = [(documents[i], scores[i] if i < len(scores) else 0) for i in range(len(documents))]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [Document(
                id=d.id, content=d.content,
                metadata={**d.metadata, "rerank_score": s / 10.0},
                score=s / 10.0,
            ) for d, s in scored[:limit]]
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return documents[:limit]


# ═══════════════════════════════════════════════════════════════
# Semantic Cache — Embedding-based cache for LLM responses
# ═══════════════════════════════════════════════════════════════

class SemanticCache:
    """
    True semantic cache using embedding similarity.

    Skip LLM on hit — if a semantically similar query was asked before,
    return the cached response without calling the LLM.

    LLM cost reduction: cache hits avoid API calls entirely.

    Usage:
        cache = SemanticCache(embedder=embedder, threshold=0.92)
        cached = await cache.get("What are React hooks?")
        if cached:
            return cached  # No LLM call!
        else:
            response = await llm.generate(...)
            await cache.set("What are React hooks?", response)
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        similarity_threshold: float = 0.92,
        max_size: int = 5000,
        ttl_seconds: int = 3600,
    ):
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: list[dict[str, Any]] = []  # [{embedding, query, response, created, hits}]
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    async def get(self, query: str, system: str = "") -> str | None:
        """Look up cache by semantic similarity."""
        if not self._embedder or not self._cache:
            self._stats["misses"] += 1
            return None

        query_emb = await self._embedder.embed(query)
        now = time.time()
        best_score = 0.0
        best_entry = None

        for entry in self._cache:
            # Check TTL
            if now - entry["created"] > self._ttl:
                continue
            # Check system prompt match (if specified)
            if system and entry.get("system", "") != system:
                continue

            similarity = self._cosine_similarity(query_emb, entry["embedding"])
            if similarity > best_score:
                best_score = similarity
                best_entry = entry

        if best_entry and best_score >= self._threshold:
            best_entry["hits"] += 1
            best_entry["last_hit"] = now
            self._stats["hits"] += 1
            logger.debug(f"Semantic cache hit (score={best_score:.3f}): {query[:50]}...")
            return best_entry["response"]

        self._stats["misses"] += 1
        return None

    async def set(self, query: str, response: str, system: str = "") -> None:
        """Cache a query-response pair with its embedding."""
        if not self._embedder:
            return

        # Evict if at capacity
        if len(self._cache) >= self._max_size:
            # Remove oldest/least-hit entries
            self._cache.sort(key=lambda e: (e["hits"], e["created"]))
            removed = len(self._cache) - self._max_size + self._max_size // 10
            self._cache = self._cache[removed:]
            self._stats["evictions"] += removed

        query_emb = await self._embedder.embed(query)
        self._cache.append({
            "embedding": query_emb,
            "query": query,
            "response": response,
            "system": system,
            "created": time.time(),
            "hits": 0,
            "last_hit": 0,
        })
        self._stats["sets"] += 1

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def invalidate(self, query: str) -> None:
        """Remove a specific query from cache."""
        self._cache = [e for e in self._cache if e["query"] != query]

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    def stats(self) -> dict[str, Any]:
        hit_rate = (
            self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
            if (self._stats["hits"] + self._stats["misses"]) > 0 else 0.0
        )
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": f"{hit_rate:.1%}",
            "llm_calls_saved": self._stats["hits"],
        }


# ═══════════════════════════════════════════════════════════════
# Query Rewriter — Expand, rephrase, and optimize queries
# ═══════════════════════════════════════════════════════════════

class QueryRewriter:
    """
    Query rewriting for better retrieval.

    Strategies:
    1. Expansion — add related terms to broaden recall
    2. Rephrasing — rephrase for better embedding match
    3. HyDE (Hypothetical Document Embeddings) — generate ideal answer, search with that
    4. Step-back — ask a broader question for context

    Usage:
        rewriter = QueryRewriter(llm=llm)
        rewritten = await rewriter.rewrite("how to use useState")
        # Returns: ["how to use useState in React", "React useState hook tutorial", ...]
    """

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def rewrite(
        self,
        query: str,
        strategy: str = "expand",  # expand, rephrase, hyde, stepback, multi
        num_rewrites: int = 3,
    ) -> list[str]:
        """Rewrite a query using the specified strategy."""
        if not self.llm:
            return [query]  # No LLM, return original

        strategies = {
            "expand": self._expand,
            "rephrase": self._rephrase,
            "hyde": self._hyde,
            "stepback": self._stepback,
            "multi": self._multi_strategy,
        }
        fn = strategies.get(strategy, self._expand)
        try:
            result = await fn(query, num_rewrites)
            # Always include original query
            if query not in result:
                result.insert(0, query)
            return result
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return [query]

    async def _expand(self, query: str, n: int) -> list[str]:
        """Add related terms to broaden recall."""
        result = await self.llm.generate_json(
            messages=[{"role": "user", "content": (
                f"Expand this search query by adding related terms and synonyms. "
                f"Generate {n} expanded versions.\n\nQuery: {query}"
            )}],
            system='Return JSON: {"queries": ["expanded query 1", "expanded query 2"]}',
            schema={"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}}},
        )
        return result.get("queries", [query])

    async def _rephrase(self, query: str, n: int) -> list[str]:
        """Rephrase for better embedding match."""
        result = await self.llm.generate_json(
            messages=[{"role": "user", "content": (
                f"Rephrase this query {n} different ways to improve search results. "
                f"Keep the same meaning but use different wording.\n\nQuery: {query}"
            )}],
            system='Return JSON: {"queries": ["rephrased 1", "rephrased 2"]}',
            schema={"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}}},
        )
        return result.get("queries", [query])

    async def _hyde(self, query: str, n: int) -> list[str]:
        """
        HyDE: Generate a hypothetical ideal answer, then search with it.
        The ideal answer's embedding is often closer to relevant documents.
        """
        result = await self.llm.generate_json(
            messages=[{"role": "user", "content": (
                f"Write a short, factual paragraph that would be the ideal answer to this question. "
                f"This will be used for semantic search.\n\nQuestion: {query}"
            )}],
            system='Return JSON: {"answer": "hypothetical ideal answer paragraph"}',
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
        hyde_doc = result.get("answer", query)
        return [query, hyde_doc]

    async def _stepback(self, query: str, n: int) -> list[str]:
        """Step-back: ask a broader question for more context."""
        result = await self.llm.generate_json(
            messages=[{"role": "user", "content": (
                f"Given this specific question, generate a broader 'step-back' question "
                f"that would help understand the underlying concept.\n\nSpecific: {query}"
            )}],
            system='Return JSON: {"stepback": "broader question", "original": "original query"}',
            schema={"type": "object", "properties": {"stepback": {"type": "string"}, "original": {"type": "string"}}},
        )
        stepback_q = result.get("stepback", query)
        return [query, stepback_q]

    async def _multi_strategy(self, query: str, n: int) -> list[str]:
        """Combine multiple strategies for maximum recall."""
        results = set()
        for strategy in [self._expand, self._rephrase]:
            queries = await strategy(query, max(1, n // 2))
            results.update(queries)
        results.add(query)
        return list(results)[:n + 1]
