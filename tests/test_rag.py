"""
Tests for agentx.rag module.
Covers: BM25Index, SemanticCache, QueryRewriter, Document.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentx.rag.engine import Document, BaseEmbedder
from agentx.rag.retrieval import BM25Index, SemanticCache, QueryRewriter


# ─────────────────────────────────────────────
# Document model
# ─────────────────────────────────────────────

class TestDocument:
    def test_creation(self):
        doc = Document(content="Hello world")
        assert doc.content == "Hello world"
        assert doc.id != ""  # auto-generated hash
        assert doc.score == 0.0
        assert doc.metadata == {}
        assert doc.embedding == []

    def test_custom_id(self):
        doc = Document(id="custom-id", content="text")
        assert doc.id == "custom-id"

    def test_auto_id_is_deterministic(self):
        d1 = Document(content="same content")
        d2 = Document(content="same content")
        assert d1.id == d2.id

    def test_different_content_different_id(self):
        d1 = Document(content="content a")
        d2 = Document(content="content b")
        assert d1.id != d2.id

    def test_score_and_metadata(self):
        doc = Document(content="x", score=0.95, metadata={"source": "test"})
        assert doc.score == 0.95
        assert doc.metadata["source"] == "test"


# ─────────────────────────────────────────────
# BM25Index
# ─────────────────────────────────────────────

class TestBM25Index:
    def setup_method(self):
        self.bm25 = BM25Index()
        self.docs = [
            Document(content="Python is a popular programming language"),
            Document(content="JavaScript is used for web development"),
            Document(content="Python frameworks like Django and Flask are great"),
            Document(content="Machine learning uses Python extensively"),
        ]

    def test_add_documents(self):
        self.bm25.add_documents(self.docs)
        assert self.bm25.size == 4

    def test_search_basic(self):
        self.bm25.add_documents(self.docs)
        results = self.bm25.search("Python programming")
        assert len(results) > 0
        # Python-related docs should score higher
        assert "python" in results[0].content.lower()

    def test_search_limit(self):
        self.bm25.add_documents(self.docs)
        results = self.bm25.search("Python", limit=2)
        assert len(results) <= 2

    def test_search_no_match(self):
        self.bm25.add_documents(self.docs)
        results = self.bm25.search("quantum physics entanglement")
        assert len(results) == 0

    def test_search_empty_index(self):
        results = self.bm25.search("anything")
        assert results == []

    def test_clear(self):
        self.bm25.add_documents(self.docs)
        assert self.bm25.size == 4
        self.bm25.clear()
        assert self.bm25.size == 0

    def test_size_property(self):
        assert self.bm25.size == 0
        self.bm25.add_documents(self.docs[:2])
        assert self.bm25.size == 2

    def test_search_with_score(self):
        self.bm25.add_documents(self.docs)
        results = self.bm25.search("Python")
        for r in results:
            assert r.score > 0

    def test_search_with_filters(self):
        docs = [
            Document(content="Python tutorial", metadata={"category": "tutorial"}),
            Document(content="Python reference", metadata={"category": "reference"}),
        ]
        self.bm25.add_documents(docs)
        results = self.bm25.search("Python", filters={"category": "tutorial"})
        assert len(results) == 1
        assert results[0].metadata["category"] == "tutorial"


# ─────────────────────────────────────────────
# Mock Embedder for SemanticCache tests
# ─────────────────────────────────────────────

class MockEmbedder(BaseEmbedder):
    """Returns a simple embedding based on text hash for testing."""

    async def embed(self, text: str) -> list[float]:
        # Return a deterministic vector; same text -> same vector
        import hashlib
        h = hashlib.md5(text.lower().strip().encode()).hexdigest()
        return [int(c, 16) / 15.0 for c in h[:8]]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# ─────────────────────────────────────────────
# SemanticCache
# ─────────────────────────────────────────────

class TestSemanticCache:
    @pytest.mark.asyncio
    async def test_miss_on_empty(self):
        cache = SemanticCache(embedder=MockEmbedder())
        result = await cache.get("anything")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_exact(self):
        cache = SemanticCache(embedder=MockEmbedder(), similarity_threshold=0.99)
        await cache.set("What are React hooks?", "React hooks are...")
        result = await cache.get("What are React hooks?")
        assert result == "React hooks are..."

    @pytest.mark.asyncio
    async def test_miss_without_embedder(self):
        cache = SemanticCache(embedder=None)
        result = await cache.get("anything")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_without_embedder_noop(self):
        cache = SemanticCache(embedder=None)
        await cache.set("q", "r")
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    async def test_invalidate(self):
        cache = SemanticCache(embedder=MockEmbedder(), similarity_threshold=0.99)
        await cache.set("query1", "response1")
        cache.invalidate("query1")
        result = await cache.get("query1")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = SemanticCache(embedder=MockEmbedder())
        await cache.set("q1", "r1")
        await cache.set("q2", "r2")
        cache.clear()
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    async def test_stats(self):
        cache = SemanticCache(embedder=MockEmbedder(), similarity_threshold=0.99)
        await cache.set("q", "r")
        await cache.get("q")        # hit
        await cache.get("other")    # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        cache = SemanticCache(embedder=MockEmbedder(), similarity_threshold=0.99, ttl_seconds=0)
        await cache.set("q", "r")
        # TTL is 0, so entry is immediately expired
        import time
        time.sleep(0.01)
        result = await cache.get("q")
        assert result is None


# ─────────────────────────────────────────────
# QueryRewriter
# ─────────────────────────────────────────────

class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_no_llm_returns_original(self):
        rewriter = QueryRewriter(llm=None)
        result = await rewriter.rewrite("how to use useState")
        assert result == ["how to use useState"]

    @pytest.mark.asyncio
    async def test_expand_strategy(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value={
            "queries": ["how to use useState in React", "React useState tutorial"]
        })
        rewriter = QueryRewriter(llm=mock_llm)
        result = await rewriter.rewrite("how to use useState", strategy="expand")
        assert "how to use useState" in result
        assert len(result) >= 2
        mock_llm.generate_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_rephrase_strategy(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value={
            "queries": ["what is the usage of useState", "useState explanation"]
        })
        rewriter = QueryRewriter(llm=mock_llm)
        result = await rewriter.rewrite("how to use useState", strategy="rephrase")
        assert "how to use useState" in result
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_hyde_strategy(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value={
            "answer": "useState is a React hook that lets you add state to functional components..."
        })
        rewriter = QueryRewriter(llm=mock_llm)
        result = await rewriter.rewrite("how to use useState", strategy="hyde")
        assert "how to use useState" in result
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_stepback_strategy(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value={
            "stepback": "What are React hooks and how do they manage state?",
            "original": "how to use useState",
        })
        rewriter = QueryRewriter(llm=mock_llm)
        result = await rewriter.rewrite("how to use useState", strategy="stepback")
        assert "how to use useState" in result

    @pytest.mark.asyncio
    async def test_error_returns_original(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(side_effect=RuntimeError("LLM error"))
        rewriter = QueryRewriter(llm=mock_llm)
        result = await rewriter.rewrite("query", strategy="expand")
        assert result == ["query"]

    @pytest.mark.asyncio
    async def test_unknown_strategy_falls_back_to_expand(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value={"queries": ["expanded"]})
        rewriter = QueryRewriter(llm=mock_llm)
        result = await rewriter.rewrite("query", strategy="nonexistent")
        assert "query" in result
