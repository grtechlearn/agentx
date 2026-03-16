"""
AgentX - Distributed Tracing, Latency Management & Queue Workers.

Operations layer:
- OpenTelemetry distributed tracing (spans, traces, context propagation)
- Latency budgets and circuit breakers
- Async task queue with workers for horizontal scaling
- Health checks and readiness probes
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Awaitable
from enum import Enum
from collections import deque

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


# ═══════════════════════════════════════════════════════════════
# Distributed Tracing — OpenTelemetry compatible
# ═══════════════════════════════════════════════════════════════

class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class Span(BaseModel):
    """A single span in a distributed trace."""

    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    operation: str = ""
    service: str = "agentx"
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class Tracer:
    """
    Distributed tracing system for AgentX.

    Tracks request flow across agents, RAG, LLM calls, and DB operations.
    Compatible with OpenTelemetry — can export to Jaeger, Zipkin, OTLP.

    Usage:
        tracer = Tracer(service="interview-bot")

        async with tracer.span("process_query") as span:
            span.attributes["user_id"] = "user-1"

            async with tracer.span("rag_retrieval", parent=span) as child:
                results = await rag.search(query)
                child.attributes["results_count"] = len(results)

            async with tracer.span("llm_generate", parent=span) as child:
                response = await llm.generate(...)
                child.attributes["model"] = "claude-sonnet"
                child.attributes["tokens"] = 500

        # Export to OpenTelemetry
        tracer.export_otlp(endpoint="http://jaeger:4318")
    """

    def __init__(self, service: str = "agentx", max_traces: int = 10000):
        self.service = service
        self._spans: list[Span] = []
        self._max_traces = max_traces
        self._active_traces: dict[str, list[Span]] = {}
        self._exporter: Any = None

    @asynccontextmanager
    async def span(
        self,
        operation: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        """Create a traced span as async context manager."""
        trace_id = parent.trace_id if parent else uuid.uuid4().hex[:32]
        s = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else "",
            operation=operation,
            service=self.service,
            start_time=time.time(),
            attributes=attributes or {},
        )

        try:
            yield s
            s.status = SpanStatus.OK
        except TimeoutError:
            s.status = SpanStatus.TIMEOUT
            s.error = "Operation timed out"
            raise
        except Exception as e:
            s.status = SpanStatus.ERROR
            s.error = str(e)
            s.events.append({"name": "exception", "attributes": {"message": str(e)}, "timestamp": time.time()})
            raise
        finally:
            s.end_time = time.time()
            s.duration_ms = (s.end_time - s.start_time) * 1000
            self._record_span(s)

    def _record_span(self, span: Span) -> None:
        """Record a completed span."""
        self._spans.append(span)
        if len(self._spans) > self._max_traces:
            self._spans = self._spans[-self._max_traces // 2:]

        # Group by trace
        if span.trace_id not in self._active_traces:
            self._active_traces[span.trace_id] = []
        self._active_traces[span.trace_id].append(span)

        # Export if exporter configured
        if self._exporter:
            try:
                self._exporter(span)
            except Exception as e:
                logger.debug(f"Span export failed: {e}")

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans in a trace."""
        return self._active_traces.get(trace_id, [])

    def get_recent_spans(self, limit: int = 50, operation: str = "") -> list[Span]:
        """Get recent spans, optionally filtered by operation."""
        spans = self._spans
        if operation:
            spans = [s for s in spans if s.operation == operation]
        return spans[-limit:]

    def setup_otlp(self, endpoint: str = "http://localhost:4318") -> None:
        """
        Setup OpenTelemetry OTLP exporter.
        Requires: pip install opentelemetry-sdk opentelemetry-exporter-otlp
        """
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider = TracerProvider()
            exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
            provider.add_span_processor(BatchSpanProcessor(exporter))
            otel_trace.set_tracer_provider(provider)

            self._otel_tracer = otel_trace.get_tracer(self.service)
            logger.info(f"OpenTelemetry OTLP configured: {endpoint}")
        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. "
                "pip install opentelemetry-sdk opentelemetry-exporter-otlp"
            )

    def set_exporter(self, exporter_fn: Callable[[Span], None]) -> None:
        """Set a custom span exporter function."""
        self._exporter = exporter_fn

    def report(self) -> dict[str, Any]:
        """Get tracing report with latency stats per operation."""
        ops: dict[str, list[float]] = {}
        errors = 0
        timeouts = 0

        for span in self._spans:
            op = span.operation
            if op not in ops:
                ops[op] = []
            ops[op].append(span.duration_ms)
            if span.status == SpanStatus.ERROR:
                errors += 1
            if span.status == SpanStatus.TIMEOUT:
                timeouts += 1

        op_stats = {}
        for op, durations in ops.items():
            sorted_d = sorted(durations)
            op_stats[op] = {
                "count": len(durations),
                "avg_ms": round(sum(durations) / len(durations), 1),
                "p50_ms": round(sorted_d[len(sorted_d) // 2], 1),
                "p95_ms": round(sorted_d[int(len(sorted_d) * 0.95)], 1) if len(sorted_d) > 1 else round(sorted_d[0], 1),
                "p99_ms": round(sorted_d[int(len(sorted_d) * 0.99)], 1) if len(sorted_d) > 1 else round(sorted_d[0], 1),
                "max_ms": round(max(durations), 1),
            }

        return {
            "total_spans": len(self._spans),
            "total_traces": len(self._active_traces),
            "errors": errors,
            "timeouts": timeouts,
            "operations": op_stats,
        }


# ═══════════════════════════════════════════════════════════════
# Latency Budget Manager — Per-request latency enforcement
# ═══════════════════════════════════════════════════════════════

class LatencyBudget:
    """
    Enforce latency budgets per request.

    Allocates time budget across operations (retrieval, LLM, etc.)
    and enforces timeouts to prevent slow requests.

    Usage:
        budget = LatencyBudget(total_ms=5000)
        budget.allocate("retrieval", 1000)
        budget.allocate("llm", 3000)
        budget.allocate("post_process", 500)

        remaining = budget.remaining("retrieval")
        if budget.is_expired("retrieval"):
            # skip or use cache
    """

    def __init__(self, total_ms: float = 5000):
        self.total_ms = total_ms
        self._start = time.monotonic()
        self._allocations: dict[str, float] = {}
        self._used: dict[str, float] = {}

    def allocate(self, operation: str, budget_ms: float) -> None:
        """Allocate time budget for an operation."""
        self._allocations[operation] = budget_ms

    def start_operation(self, operation: str) -> None:
        """Mark the start of an operation."""
        self._used[operation] = time.monotonic()

    def end_operation(self, operation: str) -> float:
        """Mark the end of an operation. Returns duration in ms."""
        start = self._used.get(operation, time.monotonic())
        duration = (time.monotonic() - start) * 1000
        self._used[operation] = duration
        return duration

    def remaining(self, operation: str = "") -> float:
        """Get remaining budget in ms."""
        elapsed = (time.monotonic() - self._start) * 1000
        if operation and operation in self._allocations:
            op_used = self._used.get(operation, 0)
            if isinstance(op_used, float) and op_used > 100:
                # op_used is a duration, not a timestamp
                return max(0, self._allocations[operation] - op_used)
            return max(0, self._allocations[operation])
        return max(0, self.total_ms - elapsed)

    def is_expired(self, operation: str = "") -> bool:
        """Check if budget is expired."""
        return self.remaining(operation) <= 0

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000


# ═══════════════════════════════════════════════════════════════
# Circuit Breaker — Protect against cascading failures
# ═══════════════════════════════════════════════════════════════

class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for external service calls (LLM APIs, vector stores, etc.)

    Prevents cascading failures by short-circuiting when errors exceed threshold.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many errors, reject immediately (use fallback)
    - HALF_OPEN: Try one request to test recovery

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        if breaker.allow_request():
            try:
                result = await api_call()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                result = fallback()
        else:
            result = fallback()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit breaker: CLOSED (recovered)")
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker: OPEN (recovery failed)")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit breaker: OPEN ({self._failure_count} failures)")

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }


# ═══════════════════════════════════════════════════════════════
# Async Task Queue — Horizontal scaling with workers
# ═══════════════════════════════════════════════════════════════

class TaskPriority(int, Enum):
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class Task(BaseModel):
    """A task in the queue."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = TaskPriority.NORMAL
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = Field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    result: Any = None
    error: str = ""
    retries: int = 0
    max_retries: int = 3


class TaskQueue:
    """
    Async task queue for horizontal scaling.

    Supports:
    - Priority-based scheduling
    - Multiple workers (concurrency control)
    - Retry with backoff
    - Task status tracking
    - Dead letter queue

    Usage:
        queue = TaskQueue(max_workers=4)

        @queue.handler("embed_document")
        async def embed(payload):
            return await embedder.embed(payload["text"])

        task_id = await queue.submit("embed_document", {"text": "hello"})
        await queue.start()  # Start processing
        result = await queue.get_result(task_id)
    """

    def __init__(self, max_workers: int = 4, max_queue_size: int = 10000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._tasks: dict[str, Task] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._dead_letter: list[Task] = []
        self._stats = {
            "submitted": 0, "completed": 0, "failed": 0,
            "retried": 0, "dead_lettered": 0,
        }

    def handler(self, name: str) -> Callable:
        """Decorator to register a task handler."""
        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable:
            self._handlers[name] = fn
            return fn
        return decorator

    def register_handler(self, name: str, fn: Callable[..., Awaitable[Any]]) -> None:
        """Register a task handler function."""
        self._handlers[name] = fn

    async def submit(
        self,
        task_name: str,
        payload: dict[str, Any] | None = None,
        priority: int = TaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> str:
        """Submit a task to the queue. Returns task_id."""
        task = Task(
            name=task_name,
            payload=payload or {},
            priority=priority,
            max_retries=max_retries,
        )
        self._tasks[task.id] = task
        # Priority queue uses (priority, timestamp, task_id) — lower = higher priority
        await self._queue.put((-priority, task.created_at, task.id))
        self._stats["submitted"] += 1
        return task.id

    async def start(self) -> None:
        """Start worker tasks."""
        if self._running:
            return
        self._running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)
        logger.info(f"Task queue started with {self.max_workers} workers")

    async def stop(self) -> None:
        """Stop workers gracefully."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()

    async def _worker_loop(self, worker_name: str) -> None:
        """Worker loop — process tasks from queue."""
        while self._running:
            try:
                priority, created, task_id = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            task = self._tasks.get(task_id)
            if not task:
                continue

            handler = self._handlers.get(task.name)
            if not handler:
                task.status = "failed"
                task.error = f"No handler for task: {task.name}"
                self._stats["failed"] += 1
                continue

            task.status = "running"
            task.started_at = time.time()

            try:
                task.result = await handler(task.payload)
                task.status = "completed"
                task.completed_at = time.time()
                self._stats["completed"] += 1
            except Exception as e:
                task.retries += 1
                if task.retries < task.max_retries:
                    # Retry with backoff
                    task.status = "pending"
                    delay = min(2 ** task.retries, 30)
                    await asyncio.sleep(delay)
                    await self._queue.put((-task.priority, time.time(), task.id))
                    self._stats["retried"] += 1
                else:
                    task.status = "failed"
                    task.error = str(e)
                    task.completed_at = time.time()
                    self._dead_letter.append(task)
                    self._stats["failed"] += 1
                    self._stats["dead_lettered"] += 1
                    logger.error(f"Task {task.id} ({task.name}) failed permanently: {e}")

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Wait for a task result."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            task = self._tasks.get(task_id)
            if task and task.status in ("completed", "failed"):
                if task.status == "failed":
                    raise RuntimeError(f"Task failed: {task.error}")
                return task.result
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Task {task_id} did not complete in {timeout}s")

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "active_workers": len(self._workers),
            "dead_letter_size": len(self._dead_letter),
        }


# ═══════════════════════════════════════════════════════════════
# Health Check — Readiness and liveness probes
# ═══════════════════════════════════════════════════════════════

class HealthCheck:
    """
    Health and readiness checking for AgentX services.

    Usage:
        health = HealthCheck()
        health.register("database", db_check)
        health.register("llm", llm_check)

        status = await health.check_all()
        # {"healthy": True, "checks": {"database": {...}, "llm": {...}}}
    """

    def __init__(self):
        self._checks: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}

    def register(self, name: str, check_fn: Callable[..., Awaitable[dict[str, Any]]]) -> None:
        """Register a health check function."""
        self._checks[name] = check_fn

    async def check(self, name: str) -> dict[str, Any]:
        """Run a single health check."""
        fn = self._checks.get(name)
        if not fn:
            return {"status": "unknown", "error": f"No check registered: {name}"}
        try:
            start = time.monotonic()
            result = await fn()
            result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
            return result
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_all(self) -> dict[str, Any]:
        """Run all health checks."""
        results = {}
        all_healthy = True

        for name in self._checks:
            result = await self.check(name)
            results[name] = result
            if result.get("status") != "healthy":
                all_healthy = False

        return {
            "healthy": all_healthy,
            "checks": results,
            "timestamp": time.time(),
        }

    async def readiness(self) -> bool:
        """Quick readiness check — are all critical services up?"""
        status = await self.check_all()
        return status["healthy"]
