# AgentX - Custom Multi-Agent System Framework

A lightweight, production-ready multi-agent AI framework. Build powerful AI agent systems without the overhead of heavy frameworks like CrewAI or LangGraph.

## Why AgentX?

| Feature | AgentX | CrewAI | LangGraph |
|---------|--------|--------|-----------|
| Dependencies | ~5 packages | ~50+ packages | ~30+ packages |
| RAM usage | ~100MB | ~500MB | ~300MB |
| LLM calls per task | 1 (you control) | 3-6 (hidden) | 1-3 |
| Vendor lock-in | None | Yes | Some |
| Learning curve | Minutes | Hours | Days |
| Customization | Unlimited | Limited | High |

## Features

- **Multi-Agent Orchestration** - Route messages, run pipelines, execute agents in parallel
- **Advanced RAG Engine** - Hybrid search, query decomposition, re-ranking, contextual compression
- **Memory System** - Short-term (session) + Long-term (persistent) memory
- **Tool System** - Database, HTTP, Redis, RAG search, or build your own
- **LLM Agnostic** - Claude (Anthropic), OpenAI, or any provider
- **Built-in Agents** - Interviewer, Evaluator, Learning Path, Goal Tracker, Analytics
- **Observability** - Structured logging, metrics collection, performance tracking
- **Zero Magic** - You understand every line. No hidden LLM calls.

## Installation

```bash
# Core (Claude + Pydantic)
pip install -e .

# With all integrations
pip install -e ".[all]"

# Pick what you need
pip install -e ".[qdrant,postgres,redis]"
```

## Quick Start

### 1. Simple Agent

```python
import asyncio
from agentx import SimpleAgent, AgentConfig, AgentMessage, AgentContext

async def main():
    agent = SimpleAgent(
        config=AgentConfig(
            name="helper",
            system_prompt="You are a helpful coding assistant.",
        )
    )

    result = await agent.run(
        message=AgentMessage(content="Explain React useEffect in 3 lines"),
        context=AgentContext(session_id="test"),
    )
    print(result.content)

asyncio.run(main())
```

### 2. Custom Agent

```python
from agentx import BaseAgent, AgentConfig, AgentContext, AgentMessage

class CodeReviewAgent(BaseAgent):
    async def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        response = await self.think(
            prompt=f"Review this code:\n{message.content}",
            context=context,
        )
        return message.reply(content=response.content)
```

### 3. Multi-Agent with Orchestrator

```python
from agentx import Orchestrator
from agentx.agents import InterviewerAgent, EvaluatorAgent, LearningPathAgent

orchestrator = Orchestrator()
orchestrator.register_many(
    InterviewerAgent(),
    EvaluatorAgent(),
    LearningPathAgent(),
)

# Define a pipeline
orchestrator.add_pipeline("evaluate", agents=["evaluator", "learning_path"])

# Route messages
@orchestrator.route_to("interviewer")
def route_interview(msg, ctx):
    return msg.data.get("action") == "interview"

# Send a message
result = await orchestrator.send(
    content="Start React interview",
    data={"action": "interview", "technology": "React"},
)
```

### 4. RAG Integration

```python
from agentx.rag import RAGEngine, QdrantVectorStore, OpenAIEmbedder

rag = RAGEngine(
    embedder=OpenAIEmbedder(),
    vector_store=QdrantVectorStore(url="http://localhost:6333"),
)

# Ingest knowledge
await rag.ingest("React useEffect runs after DOM paint...", metadata={"technology": "React"})

# Search with different strategies
docs = await rag.hybrid_search("useEffect vs useLayoutEffect", limit=5)
context = await rag.get_context("When to use useMemo?", strategy="rerank")
```

### 5. Tools

```python
from agentx import tool

@tool(name="search_docs", description="Search technical documentation")
async def search_docs(query: str, technology: str = "") -> str:
    results = await rag.get_context(query, filters={"technology": technology})
    return results

# Attach to agent
agent = SimpleAgent(
    config=AgentConfig(name="helper", system_prompt="..."),
    tools=[search_docs],
)
```

### 6. Memory

```python
from agentx import AgentMemory

memory = AgentMemory(storage_path="./data/memory")

# Short-term (session)
await memory.remember("last_topic", "React Hooks")

# Long-term (persistent)
await memory.remember("user_level", "intermediate", long_term=True, importance=0.9)

# Recall
level = await memory.recall("user_level")  # "intermediate"

# Search
results = await memory.search("React skills", limit=5)
```

## Architecture

```
agentx/
├── core/
│   ├── agent.py          # BaseAgent, SimpleAgent
│   ├── orchestrator.py   # Routing, pipelines, parallel execution
│   ├── message.py        # Inter-agent messages
│   ├── context.py        # Shared execution context
│   ├── llm.py            # LLM providers (Claude, OpenAI)
│   └── tool.py           # Tool system
├── agents/
│   ├── interviewer.py    # Technical interview conductor
│   ├── evaluator.py      # Answer scoring with RAG
│   ├── learning_path.py  # Personalized study plans
│   ├── goal_tracker.py   # Goal setting & accountability
│   └── analytics.py      # Performance reports
├── rag/
│   └── engine.py         # Advanced RAG (hybrid, rerank, decompose)
├── memory/
│   └── store.py          # Short-term + Long-term memory
├── tools/
│   └── builtin.py        # Database, HTTP, Redis, RAG tools
└── utils/
    └── logger.py         # Logging & metrics
```

## License

MIT
