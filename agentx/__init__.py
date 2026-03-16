"""
AgentX - Enterprise Multi-Agent System Framework
Build production-ready, secure, cost-optimized AI agent systems.

Features:
- Multi-Agent Orchestration (routing, pipelines, parallel execution)
- Advanced RAG (hybrid search, re-ranking, query decomposition)
- Data Pipeline (ingestion, PII detection, validation)
- Security & RBAC (role-based access, audit logging)
- Evaluation (hallucination detection, response quality)
- Prompt Engineering (templates, context management, caching)
- Self-Learning (reduce LLM dependency over time)
- Cost Management (model routing, budgets, tracking)
- Scaling & Latency (rate limiting, optimization)
"""

__version__ = "0.2.0"

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
from .rag import RAGEngine, Document, TextChunker, ChunkConfig

# --- Config ---
from .config import AgentXConfig, LLMBudget, DataGovernance, SystemMetrics, CacheConfig

# --- Pipeline ---
from .pipeline import IngestionPipeline, PIIDetector, DataValidator, DataCleaner, FileLoader

# --- Security ---
from .security import RBACManager, User, Role, Permission

# --- Evaluation ---
from .evaluation import ResponseEvaluator, HallucinationDetector, CostTracker, EvaluationResult

# --- Prompts ---
from .prompts import PromptTemplate, PromptManager, ContextManager, ResponseCache

# --- Scaling ---
from .scaling import ModelRouter, ModelConfig, RateLimiter, SelfLearner, LatencyOptimizer

# --- Agent Patterns ---
from .agents import RouterAgent, GuardrailAgent, SummarizationAgent, ClassifierAgent, RAGAgent

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
    # Config
    "AgentXConfig", "LLMBudget", "DataGovernance", "SystemMetrics", "CacheConfig",
    # Pipeline
    "IngestionPipeline", "PIIDetector", "DataValidator", "DataCleaner", "FileLoader",
    # Security
    "RBACManager", "User", "Role", "Permission",
    # Evaluation
    "ResponseEvaluator", "HallucinationDetector", "CostTracker", "EvaluationResult",
    # Prompts
    "PromptTemplate", "PromptManager", "ContextManager", "ResponseCache",
    # Scaling
    "ModelRouter", "ModelConfig", "RateLimiter", "SelfLearner", "LatencyOptimizer",
    # Agent Patterns
    "RouterAgent", "GuardrailAgent", "SummarizationAgent", "ClassifierAgent", "RAGAgent",
    # Utils
    "setup_logging", "metrics",
]
