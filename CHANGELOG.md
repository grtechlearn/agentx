# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-03-19

### Added
- **Autonomous 24/7 Daemon** — `AgentXDaemon` with job scheduler (interval, cron, delayed, event-triggered), HTTP/WebSocket API server (14 endpoints), file watcher, Redis message queue, watchdog with auto-restart, health monitoring, CLI (`python -m agentx.daemon`)
- **LLM Streaming** — `StreamChunk`, `stream()` on all providers, `think_stream()` on agents, SSE endpoint (`POST /api/v1/stream`), WebSocket streaming
- **5 LLM Providers** — Anthropic (Claude), OpenAI (GPT), Ollama (local), Groq (cloud), Gemini (Google)
- **Dynamic Model Registry** — `register_model()` and `register_openai_compatible()` for runtime model addition
- **A2A Protocol** — Google Agent-to-Agent: `AgentCard`, `A2AServer`, `A2AClient`, `register_as_tool()`, discovery at `/.well-known/agent.json`
- **Multi-Tenancy** — `TenantManager` with 4 plans (Free/Starter/Pro/Enterprise), per-tenant rate limits, budget caps, model access, namespace isolation
- **Plugin System** — `AgentXPlugin`, `PluginManager`, middleware hooks, auto-discovery via entry_points
- **Admin Dashboard** — single-file HTML at `/dashboard`, real-time status, metrics, agents, health, jobs, agent chat
- **Content Moderation** — 7 categories (profanity, sexual, abuse, violence, self_harm, drugs, custom), per-category actions, custom word lists, auto-ban
- **Vulnerability Scanning** — 6 categories: credential exposure, code injection, unsafe URLs, serialization, info disclosure, insecure patterns
- **Structured Logging** — `JSONFormatter` (ELK/Datadog), `PrettyFormatter` (dev), correlation IDs
- **Infrastructure** — CI/CD (Python 3.11/3.12/3.13), PyPI publish, Dockerfile + docker-compose, secret scanning, landing page, benchmarks
- **406 tests** across 10 test files

## [0.5.0] - 2026-03-19

### Added
- Content moderation module with configurable category-level policies
- Vulnerability scanner for credential exposure and code injection detection

## [0.4.0] - 2026-03-19

### Added
- Runtime hallucination guard with confidence gating and grounding enforcement
- Knowledge freshness manager with content hash versioning and staleness detection
- GDPR-compliant data retention enforcer with archival before deletion
- Fine-tuning data pipeline (collect, curate, export to JSONL/JSON/CSV)
- Runtime PII masker with reversible numbered placeholders
- Citation manager with source reliability scoring (authority, recency, accuracy)

## [0.3.0] - 2026-03-18

### Added
- BM25 index with true BM25 scoring (IDF + TF with saturation)
- Cross-encoder re-ranker (4 providers: local, Cohere, Voyage, LLM)
- Semantic cache with embedding cosine similarity, TTL, LRU eviction
- Query rewriter with 5 strategies (expand, rephrase, HyDE, step-back, multi)
- Distributed tracing with OpenTelemetry-compatible spans
- Latency budget manager, circuit breaker (CLOSED/OPEN/HALF_OPEN)
- Async task queue with priority scheduling, retry, dead letter queue
- Health check system with readiness and liveness probes

## [0.2.0] - 2026-03-18

### Added
- JWT authentication with HMAC-SHA256 token signing
- Multi-layer injection guard (prompt, SQL, XSS, command, path traversal)
- RBAC with 5 roles and 16 permissions, namespace-scoped vector access
- RAGAS evaluator (faithfulness, answer relevance, context relevance, recall)
- Retrieval ranking metrics (MRR, nDCG, Precision@K, Recall@K)
- Query analytics with miss rate tracking and feedback loops

## [0.1.0] - 2026-03-17

### Added
- Core agent framework: BaseAgent, SimpleAgent, Orchestrator, Pipeline
- Message system (6 types), tool system (@tool decorator), LLM abstraction
- Memory: short-term (in-memory) + long-term (DB-backed)
- RAG engine, vector stores (Qdrant, Chroma, Pinecone), embedders (Voyage, OpenAI, local)
- Ingestion pipeline with PII detection, validation, cleaning
- RBAC, cost tracking, model router, self-learner, rate limiter
- Prompt templates, context fitting (tiktoken), response cache
- MCP support (stdio, SSE, streamable-http)
- Pre-built agents: Router, Guardrail, Summarization, Classifier, RAG
- Database: SQLite + PostgreSQL, 13 tables, auto-migration
- AgentXApp one-line bootstrap
