"""
AgentX - Enterprise Multi-Agent System Framework
Build production-ready, secure, cost-optimized AI agent systems.

Features:
- Multi-Agent Orchestration (routing, pipelines, parallel execution)
- Advanced RAG (hybrid BM25+semantic search, cross-encoder re-ranking, query rewrite)
- Data Pipeline (ingestion, PII detection, validation, knowledge freshness)
- Security & RBAC (JWT auth, injection guard, role-based access, audit logging)
- Evaluation (RAGAS metrics, hallucination guard, MRR, nDCG, grounding enforcement)
- Prompt Engineering (templates, accurate tokenization, context management)
- Self-Learning (reduce LLM dependency, fine-tune pipeline, training data export)
- Cost Management (model routing, budgets, semantic cache, tracking)
- Scaling & Latency (circuit breaker, task queue, latency budgets)
- Distributed Tracing (OpenTelemetry compatible)
- Vector Stores (Qdrant, Chroma, Pinecone + local embeddings)
- Data Governance (retention enforcement, GDPR deletion, PII masking at query time)
- Citation Management (source reliability scoring, formatted citations)
- MCP Support (connect to any MCP tool server)
"""

__version__ = "0.4.0"

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
from .pipeline import (
    IngestionPipeline, PIIDetector, DataValidator, DataCleaner, FileLoader,
    KnowledgeFreshnessManager, KnowledgeSource,
    RetentionEnforcer, RetentionPolicy,
    FineTunePipeline, FineTuneConfig, FineTuneSample,
    RuntimePIIMasker,
    CitationManager, SourceReliability,
)

# --- Security ---
from .security import (
    RBACManager, User, Role, Permission,
    AuthGateway, AuthResult, InjectionGuard, InjectionResult, NamespaceManager, JWTToken,
)

# --- Evaluation ---
from .evaluation import (
    ResponseEvaluator, HallucinationDetector, CostTracker, EvaluationResult,
    RAGASEvaluator, RAGASResult, QueryAnalytics,
    HallucinationGuard, GroundingConfig, GroundedResponse,
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
    "KnowledgeFreshnessManager", "KnowledgeSource",
    "RetentionEnforcer", "RetentionPolicy",
    "FineTunePipeline", "FineTuneConfig", "FineTuneSample",
    "RuntimePIIMasker",
    "CitationManager", "SourceReliability",
    # Security
    "RBACManager", "User", "Role", "Permission",
    "AuthGateway", "AuthResult", "InjectionGuard", "InjectionResult", "NamespaceManager", "JWTToken",
    # Evaluation
    "ResponseEvaluator", "HallucinationDetector", "CostTracker", "EvaluationResult",
    "RAGASEvaluator", "RAGASResult", "QueryAnalytics",
    "HallucinationGuard", "GroundingConfig", "GroundedResponse",
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
