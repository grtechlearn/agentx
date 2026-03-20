# Show HN: AgentX -- Open-source multi-agent AI framework (better than CrewAI)

I've been building production AI agent systems and got frustrated with existing frameworks making hidden LLM calls behind my back. CrewAI makes 3-6 internal LLM calls per task (role-playing preambles, chain-of-thought wrappers, delegation prompts). When you're processing thousands of tasks/day, that's 3-6x the API cost for framework overhead.

So I built AgentX -- an open-source Python framework where every LLM call is explicit. No magic, no hidden prompts, no surprise bills.

**What's different:**

- **1 LLM call per agent interaction** -- you control exactly when and how the LLM is called
- **Built-in security** -- injection guard (prompt/SQL/XSS/command injection), content moderation (7 categories), vulnerability scanner. Not an afterthought
- **Real RAG** -- BM25 keyword search + vector search, cross-encoder re-ranking, semantic caching, query rewriting. No LangChain dependency
- **Multi-agent orchestration** -- rule-based routing, pipelines, parallel execution, handoffs. Routing is deterministic, not another LLM call
- **Production infrastructure** -- JWT auth, RBAC, distributed tracing, circuit breakers, task queues, health checks
- **24/7 daemon mode** -- job scheduler (cron/interval/delayed), HTTP/WebSocket API server, file watchers, process watchdog
- **5 LLM providers** -- Anthropic, OpenAI, Ollama, Groq, Gemini. Swap with one config change
- **One-line bootstrap** -- `AgentXApp()` wires DB, memory, auth, tracing, moderation, and 20+ components together

**Technical details:**

- ~60 modules, 360+ tests
- SQLite (dev) or PostgreSQL (prod), in-memory for tests
- Multi-layer LLM config (different models for agents vs. evaluation vs. routing)
- Vector stores: Qdrant, Chroma, Pinecone + local embeddings
- RAGAS evaluation metrics, hallucination detection, cost tracking
- Data governance: PII detection, retention enforcement, GDPR deletion

MIT licensed. Python 3.11+. No massive dependency tree.

```bash
pip install agentx
```

GitHub: https://github.com/anthropics/agentx

Happy to answer questions about architecture decisions or benchmarks.
