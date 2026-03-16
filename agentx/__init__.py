"""
AgentX - Enterprise Multi-Agent System Framework
Build production-ready, secure, cost-optimized AI agent systems.

Features:
- Multi-Agent Orchestration (routing, pipelines, parallel execution)
- Advanced RAG (hybrid BM25+semantic search, cross-encoder re-ranking, query rewrite)
- Data Pipeline (ingestion, PII detection, validation)
- Security & RBAC (JWT auth, injection guard, role-based access, audit logging)
- Evaluation (RAGAS metrics, hallucination detection, MRR, nDCG)
- Prompt Engineering (templates, context management, semantic caching)
- Self-Learning (reduce LLM dependency over time)
- Cost Management (model routing, budgets, tracking)
- Scaling & Latency (circuit breaker, task queue, latency budgets)
- Distributed Tracing (OpenTelemetry compatible)
- Vector Stores (Qdrant, Chroma, Pinecone + local embeddings)
- MCP Support (connect to any MCP tool server)
"""

__version__ = "0.3.0"

# --- Core ---
from .core import (
    BaseAgent, SimpleAgent, AgentConfig, AgentState,
    AgentContext, AgentMessage, MessageType, Priority,
    Orchestrator, Pipeline,
    BaseTool, FunctionTool, ToolResult, tool,
    LLMConfig, create_llm,
)

# --- Memory ---
from .memory import AgentMemory, ShortTermMemory, LongTermMemory, MemoryEntry

# --- RAG ---
from .rag import (
    RAGEngine, Document, TextChunker, ChunkConfig,
    BaseEmbedder, BaseVectorStore, AnthropicEmbedder, OpenAIEmbedder, QdrantVectorStore,
    BM25Index, CrossEncoderReranker, SemanticCache, QueryRewriter,
    ChromaVectorStore, PineconeVectorStore, LocalEmbedder,
)

# --- Config ---
from .config import AgentXConfig, LLMBudget, DataGovernance, SystemMetrics, CacheConfig, DatabaseConfig
from .config import LLMConfig as LLMLayerSetup, LLMLayerConfig

# --- App Bootstrap ---
from .app import AgentXApp

# --- Pipeline ---
from .pipeline import IngestionPipeline, PIIDetector, DataValidator, DataCleaner, FileLoader

# --- Security ---
from .security import (
    RBACManager, User, Role, Permission,
    AuthGateway, AuthResult, InjectionGuard, InjectionResult, NamespaceManager, JWTToken,
)

# --- Evaluation ---
from .evaluation import (
    ResponseEvaluator, HallucinationDetector, CostTracker, EvaluationResult,
    RAGASEvaluator, RAGASResult, QueryAnalytics,
)

# --- Prompts ---
from .prompts import PromptTemplate, PromptManager, ContextManager, ResponseCache

# --- Scaling ---
from .scaling import (
    ModelRouter, ModelConfig, RateLimiter, SelfLearner, LatencyOptimizer,
    Tracer, Span, SpanStatus, LatencyBudget,
    CircuitBreaker, CircuitState,
    TaskQueue, Task, TaskPriority,
    HealthCheck,
)

# --- MCP ---
from .tools.mcp import MCPConnection, MCPManager, MCPTool

# --- Agent Patterns ---
from .agents import RouterAgent, GuardrailAgent, SummarizationAgent, ClassifierAgent, RAGAgent

# --- Database ---
from .db import Database, create_database

# --- Utils ---
from .utils import setup_logging, metrics

__all__ = [
    # Core
    "BaseAgent", "SimpleAgent", "AgentConfig", "AgentState",
    "AgentContext", "AgentMessage", "MessageType", "Priority",
    "Orchestrator", "Pipeline",
    "BaseTool", "FunctionTool", "ToolResult", "tool",
    "LLMConfig", "create_llm",
    # Memory
    "AgentMemory", "ShortTermMemory", "LongTermMemory", "MemoryEntry",
    # RAG
    "RAGEngine", "Document", "TextChunker", "ChunkConfig",
    "BaseEmbedder", "BaseVectorStore", "AnthropicEmbedder", "OpenAIEmbedder", "QdrantVectorStore",
    "BM25Index", "CrossEncoderReranker", "SemanticCache", "QueryRewriter",
    "ChromaVectorStore", "PineconeVectorStore", "LocalEmbedder",
    # Config
    "AgentXConfig", "LLMBudget", "DataGovernance", "SystemMetrics", "CacheConfig",
    "DatabaseConfig", "LLMLayerSetup", "LLMLayerConfig",
    # App
    "AgentXApp",
    # Pipeline
    "IngestionPipeline", "PIIDetector", "DataValidator", "DataCleaner", "FileLoader",
    # Security
    "RBACManager", "User", "Role", "Permission",
    "AuthGateway", "AuthResult", "InjectionGuard", "InjectionResult", "NamespaceManager", "JWTToken",
    # Evaluation
    "ResponseEvaluator", "HallucinationDetector", "CostTracker", "EvaluationResult",
    "RAGASEvaluator", "RAGASResult", "QueryAnalytics",
    # Prompts
    "PromptTemplate", "PromptManager", "ContextManager", "ResponseCache",
    # Scaling
    "ModelRouter", "ModelConfig", "RateLimiter", "SelfLearner", "LatencyOptimizer",
    "Tracer", "Span", "SpanStatus", "LatencyBudget",
    "CircuitBreaker", "CircuitState",
    "TaskQueue", "Task", "TaskPriority",
    "HealthCheck",
    # MCP
    "MCPConnection", "MCPManager", "MCPTool",
    # Agent Patterns
    "RouterAgent", "GuardrailAgent", "SummarizationAgent", "ClassifierAgent", "RAGAgent",
    # Database
    "Database", "create_database",
    # Utils
    "setup_logging", "metrics",
]
