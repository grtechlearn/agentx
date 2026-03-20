# I Built an Open-Source Alternative to CrewAI -- Here's Why

**CrewAI makes 3-6 hidden LLM calls per task. I built AgentX to fix that.**

Every time you ask CrewAI to do something simple -- classify a support ticket, route a message, run a search -- it silently burns through multiple LLM calls behind your back. Role-playing preambles. Chain-of-thought wrappers. Internal delegation prompts. You asked for one answer, and your bill reflects six.

I spent months building production AI agent systems and watching my API costs spiral. Not because my agents were complex, but because the frameworks I used were. So I built AgentX -- an open-source multi-agent framework where you control every LLM call, and nothing happens behind your back.

## The Problem With Existing Frameworks

Most agent frameworks today share the same DNA: they were designed to be impressive in demos, not efficient in production.

**They're bloated.** CrewAI, LangGraph, AutoGen -- they pull in massive dependency trees, abstract everything behind layers of magic, and make it nearly impossible to understand what's actually happening when your agent runs.

**They're expensive.** Hidden LLM calls add up fast. When your framework makes 3-6 calls per task and you're processing thousands of tasks per day, you're paying 3-6x what you should be.

**They're opaque.** When something breaks -- and it always does in production -- good luck figuring out which of the five internal chain-of-thought calls went wrong. Debugging a black box is nobody's idea of a good time.

## What AgentX Does Differently

AgentX was built from scratch with one principle: **the developer controls every LLM call.**

No hidden prompts. No magic chains. No surprise API bills. Your agent calls the LLM when you tell it to, with the prompt you wrote, and returns exactly what you asked for.

Here's what that looks like in practice.

### 1. Dead Simple Agent Creation

```python
from agentx import SimpleAgent, AgentConfig

agent = SimpleAgent(
    config=AgentConfig(
        name="support",
        role="Customer support agent",
        system_prompt="You help customers with their orders.",
        model="claude-sonnet-4-6",
    )
)
```

One agent. One LLM call per message. No surprises.

### 2. Built-in Security -- Not an Afterthought

```python
from agentx import InjectionGuard, ContentModerator, ModerationConfig

# Prompt injection, SQL injection, XSS, command injection -- all caught
guard = InjectionGuard()
result = guard.check(user_input)

# Content moderation with configurable presets
moderator = ContentModerator(ModerationConfig.strict())
mod_result = moderator.check(user_message)
```

Most frameworks tell you to "add your own safety layer." AgentX ships with injection detection (5 attack types), content moderation (7 categories), and vulnerability scanning out of the box.

### 3. Real RAG With BM25 + Semantic Hybrid Search

```python
from agentx import RAGEngine, BM25Index, Document

# BM25 for keyword precision + vector search for semantic understanding
bm25 = BM25Index()
bm25.add_documents(documents)
keyword_results = bm25.search("react hooks tutorial", limit=10)
```

No need to bolt on LangChain for retrieval. AgentX includes BM25 search, cross-encoder re-ranking, semantic caching, and query rewriting -- all built in.

### 4. Multi-Agent Orchestration That Makes Sense

```python
from agentx import Orchestrator, SimpleAgent, AgentConfig, Pipeline

orch = Orchestrator(name="support-system")
orch.register(classifier_agent)
orch.register(support_agent)
orch.register(escalation_agent)

# Route based on your rules, not hidden LLM calls
orch.add_route("classifier", lambda msg, ctx: msg.data.get("needs_triage"))
orch.set_fallback("support")

# Or run a pipeline: classify -> support -> quality check
orch.add_pipeline("support_flow", ["classifier", "support", "quality"])
```

Message routing, pipelines, parallel execution, handoffs -- all deterministic. The orchestrator uses your rules, not extra LLM calls, to decide where messages go.

### 5. One-Line Application Bootstrap

```python
from agentx import AgentXApp, AgentXConfig, DatabaseConfig

async with AgentXApp(AgentXConfig(database=DatabaseConfig.memory())) as app:
    # Everything is wired: DB, memory, RBAC, auth, tracing, queues
    agent = SimpleAgent(config=AgentConfig(name="worker"), llm=app.llm("agent"))
    app.orchestrator.register(agent)
    response = await app.orchestrator.send("Hello!")
```

One call. Database, memory, RBAC, JWT auth, distributed tracing, circuit breakers, task queues, health checks, content moderation -- all initialized and connected.

## Architecture

AgentX is organized into focused modules that work independently or together:

- **Core**: Agents, orchestrator, LLM providers, tool system
- **Memory**: Short-term (in-session) + long-term (persisted to DB)
- **RAG**: BM25, vector stores (Qdrant/Chroma/Pinecone), cross-encoder re-ranking
- **Security**: JWT auth, RBAC, injection guard, content moderation, vulnerability scanner
- **Scaling**: Circuit breaker, task queue, distributed tracing, latency budgets
- **Daemon**: 24/7 autonomous operation with job scheduling, file watchers, HTTP/WebSocket API
- **Evaluation**: RAGAS metrics, hallucination detection, cost tracking

Every module is importable on its own. Use the full framework or cherry-pick what you need.

## The Numbers

- **60 modules** across core, security, RAG, scaling, and autonomous operation
- **360+ tests** covering every component
- **5 LLM providers** supported: Anthropic, OpenAI, Ollama, Groq, Gemini
- **3 vector stores**: Qdrant, Chroma, Pinecone (plus local embeddings)
- **Zero hidden LLM calls**: Every API call is explicit and intentional

## Get Started

AgentX is MIT-licensed and available today.

```bash
pip install agentx
```

**GitHub**: https://github.com/anthropics/agentx (star the repo if this resonates)

Whether you're building a chatbot, a support system, an autonomous research agent, or anything in between -- AgentX gives you the control that production systems demand without the bloat that burns your budget.

The framework you use should work for you, not against your wallet. That's what AgentX is about.
