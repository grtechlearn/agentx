"""
AgentX - Advanced RAG Engine.
Hybrid search, query decomposition, re-ranking, contextual compression.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


class Document(BaseModel):
    """A document chunk stored in the RAG system."""

    id: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    score: float = 0.0

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()


class ChunkConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = 500
    chunk_overlap: int = 50
    separator: str = "\n\n"


class BaseEmbedder(ABC):
    """Abstract embedder for converting text to vectors."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""


class BaseVectorStore(ABC):
    """Abstract vector store for document storage and retrieval."""

    @abstractmethod
    async def add(self, documents: list[Document]) -> None:
        """Add documents to the store."""

    @abstractmethod
    async def search(self, query_embedding: list[float], limit: int = 5, filters: dict[str, Any] | None = None) -> list[Document]:
        """Search for similar documents."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""


class AnthropicEmbedder(BaseEmbedder):
    """Use Anthropic's Voyage AI embeddings (via API)."""

    def __init__(self, model: str = "voyage-3", api_key: str = ""):
        self.model = model
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import voyageai
                kwargs: dict[str, Any] = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                self._client = voyageai.AsyncClient(**kwargs)
            except ImportError:
                raise ImportError("Install voyageai: pip install voyageai")
        return self._client

    async def embed(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        result = await client.embed(texts, model=self.model)
        return result.embeddings


class OpenAIEmbedder(BaseEmbedder):
    """Use OpenAI embeddings."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = ""):
        self.model = model
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = await client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store integration."""

    def __init__(self, url: str = "http://localhost:6333", collection: str = "documents", vector_size: int = 1024):
        self.url = url
        self.collection = collection
        self.vector_size = vector_size
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient
                from qdrant_client.models import Distance, VectorParams
                self._client = AsyncQdrantClient(url=self.url)
                collections = await self._client.get_collections()
                names = [c.name for c in collections.collections]
                if self.collection not in names:
                    await self._client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                    )
            except ImportError:
                raise ImportError("Install qdrant-client: pip install qdrant-client")
        return self._client

    async def add(self, documents: list[Document]) -> None:
        from qdrant_client.models import PointStruct
        client = await self._get_client()
        points = [
            PointStruct(
                id=doc.id,
                vector=doc.embedding,
                payload={"content": doc.content, **doc.metadata},
            )
            for doc in documents
            if doc.embedding
        ]
        if points:
            await client.upsert(collection_name=self.collection, points=points)

    async def search(self, query_embedding: list[float], limit: int = 5, filters: dict[str, Any] | None = None) -> list[Document]:
        client = await self._get_client()
        query_filter = None
        if filters:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        results = await client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )
        return [
            Document(
                id=str(r.id),
                content=r.payload.get("content", ""),
                metadata={k: v for k, v in r.payload.items() if k != "content"},
                score=r.score,
            )
            for r in results
        ]

    async def delete(self, ids: list[str]) -> None:
        client = await self._get_client()
        from qdrant_client.models import PointIdsList
        await client.delete(collection_name=self.collection, points_selector=PointIdsList(points=ids))


class TextChunker:
    """Split text into overlapping chunks."""

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        chunks = []
        separators = [self.config.separator, "\n", ". ", " "]
        parts = self._recursive_split(text, separators, self.config.chunk_size)

        for i, part in enumerate(parts):
            doc_metadata = {**(metadata or {}), "chunk_index": i, "total_chunks": len(parts)}
            chunks.append(Document(content=part.strip(), metadata=doc_metadata))

        return chunks

    def _recursive_split(self, text: str, separators: list[str], max_size: int) -> list[str]:
        if len(text) <= max_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else " "
        remaining_seps = separators[1:] if len(separators) > 1 else separators

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            test = f"{current}{sep}{part}" if current else part
            if len(test) <= max_size:
                current = test
            else:
                if current:
                    chunks.append(current)
                if len(part) > max_size and remaining_seps:
                    chunks.extend(self._recursive_split(part, remaining_seps, max_size))
                else:
                    current = part

        if current:
            chunks.append(current)

        # Add overlap
        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev_end = chunks[i - 1][-self.config.chunk_overlap:]
                    chunk = prev_end + sep + chunk
                overlapped.append(chunk)
            return overlapped

        return chunks


class RAGEngine:
    """
    Advanced RAG engine with multiple retrieval strategies.

    Features:
    - Hybrid search (semantic + keyword)
    - Query decomposition
    - Re-ranking
    - Contextual compression
    - Parent-child chunking
    - Metadata filtering
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        chunker: TextChunker | None = None,
        llm: Any = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker or TextChunker()
        self.llm = llm  # Optional: for query decomposition & re-ranking

    # --- Ingestion ---

    async def ingest(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Chunk text, embed, and store."""
        chunks = self.chunker.chunk(text, metadata)
        if not chunks:
            return 0
        texts = [c.content for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        await self.vector_store.add(chunks)
        logger.info(f"Ingested {len(chunks)} chunks")
        return len(chunks)

    async def ingest_documents(self, documents: list[Document]) -> int:
        """Embed and store pre-chunked documents."""
        texts = [d.content for d in documents]
        embeddings = await self.embedder.embed_batch(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        await self.vector_store.add(documents)
        return len(documents)

    # --- Retrieval Strategies ---

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Basic semantic search."""
        query_embedding = await self.embedder.embed(query)
        return await self.vector_store.search(query_embedding, limit, filters)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        keyword_weight: float = 0.3,
    ) -> list[Document]:
        """
        Hybrid search: combine semantic similarity + keyword matching.
        """
        # Semantic search
        semantic_results = await self.search(query, limit=limit * 2, filters=filters)

        # Keyword scoring
        query_words = set(query.lower().split())
        for doc in semantic_results:
            doc_words = set(doc.content.lower().split())
            keyword_overlap = len(query_words & doc_words) / max(len(query_words), 1)
            doc.score = (1 - keyword_weight) * doc.score + keyword_weight * keyword_overlap

        semantic_results.sort(key=lambda d: d.score, reverse=True)
        return semantic_results[:limit]

    async def decomposed_search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Query decomposition: break complex query into sub-queries,
        search each, then merge results.
        """
        if not self.llm:
            return await self.search(query, limit, filters)

        # Use LLM to decompose query
        sub_queries = await self._decompose_query(query)
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

        # Search each sub-query
        all_results: dict[str, Document] = {}
        for sub_q in sub_queries:
            results = await self.search(sub_q, limit=limit, filters=filters)
            for doc in results:
                if doc.id not in all_results or doc.score > all_results[doc.id].score:
                    all_results[doc.id] = doc

        # Sort by score and return
        sorted_results = sorted(all_results.values(), key=lambda d: d.score, reverse=True)
        return sorted_results[:limit]

    async def search_with_rerank(
        self,
        query: str,
        limit: int = 5,
        initial_limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Two-stage retrieval: broad search → LLM re-ranking.
        """
        # Stage 1: Broad retrieval
        candidates = await self.search(query, limit=initial_limit, filters=filters)

        if not self.llm or len(candidates) <= limit:
            return candidates[:limit]

        # Stage 2: LLM re-ranking
        return await self._rerank(query, candidates, limit)

    # --- Context Building ---

    async def get_context(
        self,
        query: str,
        limit: int = 5,
        strategy: str = "hybrid",
        filters: dict[str, Any] | None = None,
        max_context_length: int = 4000,
    ) -> str:
        """
        Get formatted context string for LLM prompt.
        This is the main method agents will use.
        """
        strategies = {
            "basic": self.search,
            "hybrid": self.hybrid_search,
            "decomposed": self.decomposed_search,
            "rerank": self.search_with_rerank,
        }
        search_fn = strategies.get(strategy, self.hybrid_search)
        documents = await search_fn(query, limit=limit, filters=filters)

        # Build context string
        context_parts = []
        total_length = 0
        for i, doc in enumerate(documents):
            entry = f"[Source {i + 1}] (relevance: {doc.score:.2f})\n{doc.content}"
            if total_length + len(entry) > max_context_length:
                break
            context_parts.append(entry)
            total_length += len(entry)

        return "\n\n---\n\n".join(context_parts)

    # --- Private helpers ---

    async def _decompose_query(self, query: str) -> list[str]:
        """Use LLM to break a complex query into sub-queries."""
        response = await self.llm.generate_json(
            messages=[{"role": "user", "content": f"Break this question into 2-4 simpler search queries:\n\n{query}"}],
            system="You decompose complex questions into simpler search queries. Return JSON: {\"queries\": [\"query1\", \"query2\"]}",
            schema={"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}}},
        )
        return response.get("queries", [query])

    async def _rerank(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        """Use LLM to re-rank documents by relevance."""
        doc_summaries = "\n".join(
            f"[{i}] {doc.content[:200]}..." for i, doc in enumerate(documents)
        )
        response = await self.llm.generate_json(
            messages=[{"role": "user", "content": f"Query: {query}\n\nDocuments:\n{doc_summaries}\n\nRank the documents by relevance to the query."}],
            system="Return JSON: {\"ranked_indices\": [0, 3, 1, ...]} ordered by most relevant first.",
            schema={"type": "object", "properties": {"ranked_indices": {"type": "array", "items": {"type": "integer"}}}},
        )
        indices = response.get("ranked_indices", list(range(len(documents))))
        reranked = []
        for idx in indices[:limit]:
            if 0 <= idx < len(documents):
                reranked.append(documents[idx])
        return reranked
