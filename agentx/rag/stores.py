"""
AgentX - Additional Vector Stores & Embedding Models.

Vector Stores:
- ChromaVectorStore — Chroma DB (local or server)
- PineconeVectorStore — Pinecone (managed cloud)

Embedding Models:
- LocalEmbedder — sentence-transformers (local, no API fees)
"""

from __future__ import annotations

import logging
from typing import Any

from .engine import BaseEmbedder, BaseVectorStore, Document

logger = logging.getLogger("agentx")


# ═══════════════════════════════════════════════════════════════
# Chroma Vector Store
# ═══════════════════════════════════════════════════════════════

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store integration.

    Supports:
    - In-memory mode (testing/dev)
    - Persistent local storage
    - Client-server mode

    Usage:
        # In-memory
        store = ChromaVectorStore(collection="docs")

        # Persistent local
        store = ChromaVectorStore(collection="docs", persist_directory="./chroma_db")

        # Client-server
        store = ChromaVectorStore(collection="docs", host="localhost", port=8000)
    """

    def __init__(
        self,
        collection: str = "documents",
        persist_directory: str = "",
        host: str = "",
        port: int = 8000,
        distance_fn: str = "cosine",  # cosine, l2, ip
    ):
        self.collection_name = collection
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.distance_fn = distance_fn
        self._client: Any = None
        self._collection: Any = None

    async def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings

            if self.host:
                # Client-server mode
                self._client = chromadb.HttpClient(host=self.host, port=self.port)
            elif self.persist_directory:
                # Persistent local
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                # In-memory
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_fn},
            )
            return self._collection

        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")

    async def add(self, documents: list[Document]) -> None:
        collection = await self._get_collection()
        if not documents:
            return

        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        contents = [doc.content for doc in documents]
        metadatas = [
            {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
             for k, v in doc.metadata.items()}
            for doc in documents
        ]

        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": contents,
            "metadatas": metadatas,
        }
        if embeddings and len(embeddings) == len(documents):
            kwargs["embeddings"] = embeddings

        collection.upsert(**kwargs)

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        collection = await self._get_collection()

        where = None
        if filters:
            # Chroma uses {"key": {"$eq": value}} or {"$and": [...]}
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append({key: {"$in": value}})
                else:
                    conditions.append({key: {"$eq": value}})
            if len(conditions) == 1:
                where = conditions[0]
            elif conditions:
                where = {"$and": conditions}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i] if results.get("documents") else ""
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                # Convert distance to similarity score (cosine: 1 - distance)
                score = 1.0 - distance if self.distance_fn == "cosine" else 1.0 / (1.0 + distance)
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                ))

        return documents

    async def delete(self, ids: list[str]) -> None:
        collection = await self._get_collection()
        collection.delete(ids=ids)

    async def count(self) -> int:
        collection = await self._get_collection()
        return collection.count()


# ═══════════════════════════════════════════════════════════════
# Pinecone Vector Store
# ═══════════════════════════════════════════════════════════════

class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone managed vector store integration.

    Supports:
    - Serverless indexes
    - Namespace scoping (for RBAC)
    - Metadata filtering
    - Batch upsert

    Usage:
        store = PineconeVectorStore(
            api_key="pc-...",
            index_name="my-index",
            namespace="public",
        )
    """

    def __init__(
        self,
        api_key: str = "",
        index_name: str = "agentx",
        namespace: str = "",
        environment: str = "",
        dimension: int = 1024,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.environment = environment
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self._index: Any = None

    async def _get_index(self) -> Any:
        if self._index is not None:
            return self._index

        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=self.api_key)

            # Create index if it doesn't exist
            existing = [idx.name for idx in pc.list_indexes()]
            if self.index_name not in existing:
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )

            self._index = pc.Index(self.index_name)
            return self._index

        except ImportError:
            raise ImportError("Install pinecone: pip install pinecone")

    async def add(self, documents: list[Document]) -> None:
        index = await self._get_index()
        if not documents:
            return

        # Batch upsert (Pinecone recommends batches of 100)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectors = [
                {
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": {
                        "content": doc.content[:40960],  # Pinecone metadata limit
                        **{k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                           for k, v in doc.metadata.items()},
                    },
                }
                for doc in batch
                if doc.embedding
            ]
            if vectors:
                index.upsert(vectors=vectors, namespace=self.namespace)

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        index = await self._get_index()

        # Build Pinecone filter
        pinecone_filter = None
        if filters:
            conditions = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions[key] = {"$in": value}
                else:
                    conditions[key] = {"$eq": value}
            pinecone_filter = conditions

        results = index.query(
            vector=query_embedding,
            top_k=limit,
            namespace=self.namespace,
            filter=pinecone_filter,
            include_metadata=True,
        )

        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            content = metadata.pop("content", "")
            documents.append(Document(
                id=match["id"],
                content=content,
                metadata=metadata,
                score=match.get("score", 0.0),
            ))

        return documents

    async def delete(self, ids: list[str]) -> None:
        index = await self._get_index()
        index.delete(ids=ids, namespace=self.namespace)

    async def stats(self) -> dict[str, Any]:
        """Get index statistics."""
        index = await self._get_index()
        return index.describe_index_stats()


# ═══════════════════════════════════════════════════════════════
# Local Embedder — sentence-transformers (no API fees!)
# ═══════════════════════════════════════════════════════════════

class LocalEmbedder(BaseEmbedder):
    """
    Local embedding model using sentence-transformers.

    No API fee for vectors — run embeddings locally.
    Cost endgame: local embeddings + semantic cache + fine-tuned LLM = near-zero API spend.

    Popular models:
    - all-MiniLM-L6-v2 (384 dim, fast, good quality)
    - all-mpnet-base-v2 (768 dim, best quality)
    - e5-small-v2 (384 dim, Microsoft, good for retrieval)
    - bge-small-en-v1.5 (384 dim, BAAI, top retrieval quality)

    Usage:
        embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
        embedding = await embedder.embed("Hello world")
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",  # cpu, cuda, mps
        batch_size: int = 64,
        normalize: bool = True,
    ):
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded local embedding model: {self.model_name} on {self.device}")
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                )
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text locally."""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts locally (no API call!)."""
        model = self._get_model()

        # sentence-transformers is synchronous, run in thread pool
        import asyncio
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ),
        )

        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
