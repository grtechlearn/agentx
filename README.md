# AgentX - Enterprise Multi-Agent System Framework

A lightweight, production-ready, enterprise-grade multi-agent AI framework. Build powerful AI agent systems with Advanced RAG, RBAC, self-learning, hallucination detection, and cost management — without the overhead of CrewAI or LangGraph.

## Why AgentX?

| Feature | AgentX | CrewAI | LangGraph |
|---------|--------|--------|-----------|
| Dependencies | ~5 packages | ~50+ packages | ~30+ packages |
| RAM usage | ~100MB | ~500MB | ~300MB |
| LLM calls per task | 1 (you control) | 3-6 (hidden) | 1-3 |
| Hallucination detection | Built-in | No | No |
| RBAC & Security | Built-in | No | No |
| Self-learning | Built-in | No | No |
| Cost management | Built-in | No | No |
| PII detection | Built-in | No | No |
| Vendor lock-in | None | Yes | Some |

## Features

### Core
- **Multi-Agent Orchestration** — Routing, pipelines, parallel execution, handoffs
- **Advanced RAG Engine** — Hybrid search, query decomposition, re-ranking
- **Memory System** — Short-term (session) + Long-term (persistent)
- **Tool System** — Database, HTTP, Redis, RAG, or build your own
- **LLM Agnostic** — Claude, OpenAI, or any provider

### Enterprise (v0.2.0)
- **Data Pipeline** — Ingestion, cleaning, PII detection, validation
- **Security & RBAC** — Role-based access, permissions, audit logging, API keys
- **Hallucination Detection** — Claim verification, faithfulness scoring
- **Prompt Engineering** — Templates, versioning, context window management
- **Self-Learning** — Reduce LLM dependency over time, training data collection
- **Cost Management** — Budget tracking, model routing, response caching
- **Scaling** — Rate limiting, latency optimization, smart model selection
- **Guardrails** — Input/output safety checks, content filtering

## Installation

```bash
pip install -e .             # Core only
pip install -e ".[all]"      # All integrations
pip install -e ".[qdrant,postgres,redis]"  # Pick what you need
```

## Quick Start

### 1. Simple Agent

```python
import asyncio
from agentx import SimpleAgent, AgentConfig, AgentMessage, AgentContext

async def main():
    agent = SimpleAgent(config=AgentConfig(
        name="helper",
        system_prompt="You are a helpful coding assistant.",
    ))
    result = await agent.run(
        message=AgentMessage(content="Explain React useEffect"),
        context=AgentContext(session_id="test"),
    )
    print(result.content)

asyncio.run(main())
```

### 2. Multi-Agent Orchestrator

```python
from agentx import Orchestrator, RAGAgent, GuardrailAgent, SimpleAgent

orchestrator = Orchestrator()
orchestrator.register_many(
    GuardrailAgent(),           # Safety checks
    RAGAgent(rag_engine=rag),   # Knowledge-powered answers
    SimpleAgent(config=...),    # General assistant
)

# Pipeline: guardrail → rag → respond
orchestrator.add_pipeline("safe_answer", agents=["guardrail", "rag_agent"])

result = await orchestrator.send("What is useEffect?")
```

### 3. Advanced RAG with Data Pipeline

```python
from agentx import RAGEngine, IngestionPipeline, PIIDetector, FileLoader
from agentx.rag import QdrantVectorStore, OpenAIEmbedder

# Setup RAG
rag = RAGEngine(
    embedder=OpenAIEmbedder(),
    vector_store=QdrantVectorStore(url="http://localhost:6333"),
)

# Ingest data with PII detection
pipeline = IngestionPipeline(rag_engine=rag)
pipeline.add_loader(FileLoader("./docs"))
stats = await pipeline.run(detect_pii=True)
# → {"loaded": 50, "pii_redacted": 3, "ingested": 245}

# Search with strategies
context = await rag.get_context("React hooks", strategy="hybrid")
context = await rag.get_context("System design", strategy="rerank")
```

### 4. Security & RBAC

```python
from agentx import RBACManager, User, Role, Permission

rbac = RBACManager()
rbac.add_user(User(id="user-1", name="Ramesh", role=Role.ADMIN))
rbac.add_user(User(id="user-2", name="Student", role=Role.USER))

# Check permissions
rbac.require("user-1", Permission.RAG_INGEST)    # ✅ Admin can ingest
rbac.require("user-2", Permission.ADMIN_USERS)    # ❌ PermissionError

# Rate limiting
rbac.check_rate_limit("user-2", max_requests=60)

# Audit log
log = rbac.get_audit_log(user_id="user-2")
```

### 5. Hallucination Detection

```python
from agentx import HallucinationDetector, ResponseEvaluator

detector = HallucinationDetector(llm=my_llm)
result = await detector.detect(
    response="useEffect runs before DOM paint",   # Wrong!
    sources=["useEffect runs AFTER the DOM has been painted..."],
    query="When does useEffect run?"
)
# result.hallucination_detected = True
# result.faithfulness = 0.3
# result.hallucination_details = ["Claim contradicts sources"]
```

### 6. Cost Management & Self-Learning

```python
from agentx import ModelRouter, SelfLearner, CostTracker

# Smart model routing (cheap model for simple tasks)
router = ModelRouter()
router.setup_defaults()
model = router.select_model(
    task_complexity="simple",
    prefer="cost",              # Use cheapest model that works
)
# → "claude-haiku" for simple, "claude-sonnet" for complex

# Self-learning (reduce LLM calls over time)
learner = SelfLearner()
cached = await learner.check("What is useState?")
if cached:
    return cached  # No LLM call needed!
else:
    response = await llm.generate(...)
    await learner.learn("What is useState?", response, score=0.95)

# Cost tracking
tracker = CostTracker()
tracker.track("claude-sonnet-4-6", input_tokens=500, output_tokens=200, key="user-1")
print(tracker.report())
```

### 7. Prompt Engineering

```python
from agentx import PromptTemplate, PromptManager, ContextManager

# Template management
pm = PromptManager()
pm.register(PromptTemplate(
    name="evaluator",
    version="1.0",
    template="Evaluate this {{technology}} answer:\n{{answer}}\n\nContext:\n{{context}}",
    variables=["technology", "answer", "context"],
))
prompt = pm.render("evaluator", technology="React", answer="...", context="...")

# Context window management
cm = ContextManager(max_context_tokens=100000)
fitted = cm.fit_context(
    system_prompt="You are an expert...",
    rag_context=long_context,
    conversation=messages,
)
# Automatically truncates to fit token limit
```

## Architecture

```
agentx/
├── core/                    # Agent framework foundation
│   ├── agent.py             # BaseAgent, SimpleAgent, lifecycle hooks
│   ├── orchestrator.py      # Routing, pipelines, parallel execution
│   ├── message.py           # Inter-agent messaging & handoffs
│   ├── context.py           # Shared execution context
│   ├── llm.py               # LLM providers (Claude, OpenAI)
│   └── tool.py              # Tool system & @tool decorator
│
├── config/                  # Phase 1: Configuration & Governance
│   └── settings.py          # AgentXConfig, budgets, data governance, metrics
│
├── pipeline/                # Phase 2: Data Pipeline
│   └── ingestion.py         # PII detection, validation, cleaning, loaders
│
├── rag/                     # Phase 3 & 4: Embeddings & Retrieval
│   └── engine.py            # Hybrid search, re-ranking, decomposition, Qdrant
│
├── prompts/                 # Phase 5: Generation & Optimization
│   └── manager.py           # Templates, context management, response cache
│
├── evaluation/              # Phase 5: Quality & Safety
│   └── metrics.py           # Hallucination detection, cost tracking
│
├── security/                # Phase 6: Security & RBAC
│   └── rbac.py              # Roles, permissions, audit, rate limiting
│
├── scaling/                 # Phase 6: Operations
│   └── optimizer.py         # Model routing, self-learning, latency tracking
│
├── agents/                  # Reusable agent patterns
│   ├── patterns.py          # Router, Guardrail, Summarizer, Classifier, RAG
│   ├── interviewer.py       # Example: Technical interviewer
│   ├── evaluator.py         # Example: Answer evaluator
│   └── ...                  # More domain agents
│
├── memory/                  # Short-term + Long-term memory
│   └── store.py
│
├── tools/                   # Built-in tools
│   └── builtin.py           # Database, HTTP, Redis, RAG search
│
└── utils/                   # Observability
    └── logger.py            # Structured logging & metrics
```

## 6-Phase Architecture

| Phase | Module | What It Handles |
|-------|--------|----------------|
| **Phase 1** | `config/` | System metrics, cost modeling, data governance, budgets |
| **Phase 2** | `pipeline/` | Data ingestion, PII detection, cleaning, validation |
| **Phase 3** | `rag/` | Embedding models, vector store abstraction, chunking |
| **Phase 4** | `rag/` | Hybrid retrieval, re-ranking, query decomposition |
| **Phase 5** | `prompts/` + `evaluation/` | Generation, prompt templates, hallucination detection |
| **Phase 6** | `security/` + `scaling/` | RBAC, cost management, self-learning, latency optimization |

## License

MIT
