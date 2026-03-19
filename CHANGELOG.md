# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-03-19

### Added
- Autonomous daemon mode for long-running background agent execution
- Content moderation pipeline with configurable policy rules
- Vulnerability scanning for dependency and container image analysis

## [0.5.0]

### Added
- Content moderation module with toxicity detection and policy enforcement
- Vulnerability scanner for automated security assessments

## [0.4.0]

### Added
- Hallucination guard with source-grounded verification
- Knowledge freshness detection and staleness warnings
- GDPR-compliant data retention policies with automatic expiry
- Fine-tuning pipeline for domain-specific model adaptation
- PII masking across prompts, responses, and stored context
- Citation extraction and inline source attribution

## [0.3.0]

### Added
- Advanced RAG retrieval with BM25 keyword scoring
- Cross-encoder reranking for improved retrieval precision
- Semantic cache for deduplicating similar queries
- Query rewriter for intent clarification and expansion
- Distributed tracing with OpenTelemetry integration
- Circuit breaker for graceful upstream service degradation
- Async task queue for background job processing

## [0.2.0]

### Added
- JWT-based authentication for API endpoints
- Prompt injection guard with input sanitization
- RBAC namespace scoping for multi-tenant isolation
- RAGAS evaluation metrics (faithfulness, relevance, context recall)
- Query analytics dashboard with latency and usage tracking

## [0.1.0]

### Added
- Core agent framework with tool-calling support
- Multi-agent orchestrator with routing and delegation
- Conversation memory with short-term and long-term stores
- RAG pipeline with chunking, embedding, and vector retrieval
- Role-based access control (RBAC) for agents and resources
- Cost tracking with per-model and per-request token accounting
- Model Context Protocol (MCP) support for external tool integration
