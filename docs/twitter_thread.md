# AgentX Launch Thread

## Tweet 1 (Hook)

I built an open-source AI agent framework that makes 1 LLM call per task instead of 6.

CrewAI hides 3-6 internal LLM calls behind every interaction. I got tired of paying for framework overhead.

So I built AgentX. Here's what it does differently:

## Tweet 2 (Simple Agents)

1/ Dead simple agents. No magic, no hidden prompts.

```python
from agentx import SimpleAgent, AgentConfig

agent = SimpleAgent(config=AgentConfig(
    name="support",
    system_prompt="You help customers.",
    model="claude-sonnet-4-6",
))
```

One agent. One LLM call per message. That's it.

## Tweet 3 (Security)

2/ Built-in security that actually works.

```python
from agentx import InjectionGuard

guard = InjectionGuard()
result = guard.check(user_input)
# Catches: prompt injection, SQL injection,
# XSS, command injection, path traversal
```

Most frameworks say "add your own safety layer." AgentX ships with 5 injection types + content moderation for 7 categories.

## Tweet 4 (RAG)

3/ Real RAG without LangChain.

```python
from agentx import BM25Index, Document

bm25 = BM25Index()
bm25.add_documents(docs)
results = bm25.search("react hooks", limit=10)
```

Built-in: BM25 keyword search, cross-encoder re-ranking, semantic caching, query rewriting, HyDE.

3 vector stores: Qdrant, Chroma, Pinecone.

## Tweet 5 (Orchestration)

4/ Multi-agent orchestration without extra LLM calls.

```python
from agentx import Orchestrator

orch = Orchestrator()
orch.register(classifier)
orch.register(support)

# Deterministic routing -- YOUR rules, not LLM calls
orch.add_route("classifier",
    lambda msg, ctx: msg.data.get("needs_triage"))
orch.set_fallback("support")
```

Pipelines, parallel execution, handoffs -- all built in.

## Tweet 6 (One-Line Bootstrap)

5/ One line to wire everything together.

```python
from agentx import AgentXApp

async with AgentXApp() as app:
    # DB, memory, RBAC, JWT auth, tracing,
    # circuit breakers, task queues, moderation
    # -- all connected and ready
    response = await app.orchestrator.send("Hello!")
```

20+ components. Zero config needed for dev. Production-ready with one config change.

## Tweet 7 (Content Moderation)

6/ Configurable content moderation with presets.

```python
from agentx import ContentModerator, ModerationConfig

# Pick a preset or customize everything
mod = ContentModerator(ModerationConfig.strict())
result = mod.check(user_message)

if result.blocked:
    print(f"Blocked: {result.reason}")
```

7 categories (profanity, abuse, sexual, violence, self-harm, drugs, custom). Block, warn, redact, or log.

## Tweet 8 (Daemon Mode)

7/ 24/7 autonomous daemon mode.

```python
from agentx import JobScheduler

scheduler = JobScheduler()
scheduler.add_interval("refresh_docs",
    handler=refresh_fn, minutes=5)
scheduler.add_cron("daily_report",
    handler=report_fn, hour=9, minute=0)
```

Job scheduling + HTTP/WebSocket API server + file watchers + process watchdog. Your agents run while you sleep.

## Tweet 9 (Numbers)

The numbers:

- 60 modules
- 360+ tests
- 5 LLM providers (Anthropic, OpenAI, Ollama, Groq, Gemini)
- 3 vector stores + local embeddings
- JWT auth, RBAC, distributed tracing
- RAGAS evaluation, hallucination detection
- PII detection, GDPR compliance
- MIT licensed

All in a single `pip install`.

## Tweet 10 (CTA)

AgentX is open-source and available now.

```bash
pip install agentx
```

GitHub: https://github.com/anthropics/agentx

If you're building AI agents and tired of paying for framework overhead, give it a try.

Star the repo if this resonates. PRs welcome.
