"""
AgentX - Custom Multi-Agent System Framework
Build production-ready multi-agent AI systems without framework overhead.
"""

__version__ = "0.1.0"

from .core import (
    BaseAgent,
    SimpleAgent,
    AgentConfig,
    AgentState,
    AgentContext,
    AgentMessage,
    MessageType,
    Priority,
    Orchestrator,
    Pipeline,
    BaseTool,
    FunctionTool,
    ToolResult,
    tool,
    LLMConfig,
    create_llm,
)
from .memory import AgentMemory, ShortTermMemory, LongTermMemory, MemoryEntry
from .rag import RAGEngine, Document, TextChunker, ChunkConfig
from .agents import InterviewerAgent, EvaluatorAgent, LearningPathAgent, GoalTrackerAgent, AnalyticsAgent
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
    # Agents
    "InterviewerAgent", "EvaluatorAgent", "LearningPathAgent",
    "GoalTrackerAgent", "AnalyticsAgent",
    # Utils
    "setup_logging", "metrics",
]
