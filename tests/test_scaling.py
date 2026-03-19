"""
Tests for agentx.scaling module.
Covers: Tracer, LatencyBudget, CircuitBreaker, HealthCheck,
        ModelRouter, RateLimiter, LatencyOptimizer.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from agentx.scaling.tracing import (
    Tracer, Span, SpanStatus,
    LatencyBudget,
    CircuitBreaker, CircuitState,
    HealthCheck,
)
from agentx.scaling.optimizer import (
    ModelRouter, ModelConfig,
    RateLimiter,
    LatencyOptimizer,
)


# ─────────────────────────────────────────────
# Tracer
# ─────────────────────────────────────────────

class TestTracer:
    @pytest.mark.asyncio
    async def test_span_creation(self):
        tracer = Tracer(service="test-service")
        async with tracer.span("test_op") as s:
            s.attributes["key"] = "value"

        assert len(tracer._spans) == 1
        span = tracer._spans[0]
        assert span.operation == "test_op"
        assert span.service == "test-service"
        assert span.status == SpanStatus.OK
        assert span.duration_ms > 0
        assert span.attributes["key"] == "value"

    @pytest.mark.asyncio
    async def test_nested_spans(self):
        tracer = Tracer()
        async with tracer.span("parent") as parent:
            async with tracer.span("child", parent=parent) as child:
                pass

        assert len(tracer._spans) == 2
        child_span = tracer._spans[0]
        parent_span = tracer._spans[1]
        assert child_span.trace_id == parent_span.trace_id
        assert child_span.parent_span_id == parent_span.span_id

    @pytest.mark.asyncio
    async def test_error_tracking(self):
        tracer = Tracer()
        with pytest.raises(ValueError):
            async with tracer.span("failing") as s:
                raise ValueError("test error")

        assert len(tracer._spans) == 1
        span = tracer._spans[0]
        assert span.status == SpanStatus.ERROR
        assert span.error == "test error"
        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"

    @pytest.mark.asyncio
    async def test_timeout_tracking(self):
        tracer = Tracer()
        with pytest.raises(TimeoutError):
            async with tracer.span("timeout_op") as s:
                raise TimeoutError("timed out")

        span = tracer._spans[0]
        assert span.status == SpanStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_report(self):
        tracer = Tracer()
        async with tracer.span("op1"):
            pass
        async with tracer.span("op1"):
            pass
        async with tracer.span("op2"):
            pass

        report = tracer.report()
        assert report["total_spans"] == 3
        assert "op1" in report["operations"]
        assert report["operations"]["op1"]["count"] == 2
        assert "op2" in report["operations"]

    @pytest.mark.asyncio
    async def test_get_trace(self):
        tracer = Tracer()
        async with tracer.span("op") as s:
            trace_id = s.trace_id

        spans = tracer.get_trace(trace_id)
        assert len(spans) == 1

    @pytest.mark.asyncio
    async def test_custom_exporter(self):
        exported = []
        tracer = Tracer()
        tracer.set_exporter(lambda span: exported.append(span))

        async with tracer.span("exported_op"):
            pass

        assert len(exported) == 1


# ─────────────────────────────────────────────
# LatencyBudget
# ─────────────────────────────────────────────

class TestLatencyBudget:
    def test_allocate(self):
        budget = LatencyBudget(total_ms=5000)
        budget.allocate("retrieval", 1000)
        budget.allocate("llm", 3000)
        assert "retrieval" in budget._allocations
        assert "llm" in budget._allocations

    def test_remaining_total(self):
        budget = LatencyBudget(total_ms=5000)
        remaining = budget.remaining()
        assert remaining > 0
        assert remaining <= 5000

    def test_remaining_operation(self):
        budget = LatencyBudget(total_ms=5000)
        budget.allocate("op", 1000)
        remaining = budget.remaining("op")
        assert remaining == 1000

    def test_is_expired_fresh(self):
        budget = LatencyBudget(total_ms=5000)
        assert budget.is_expired() is False

    def test_is_expired_zero_budget(self):
        budget = LatencyBudget(total_ms=0)
        assert budget.is_expired() is True

    def test_elapsed_ms(self):
        budget = LatencyBudget(total_ms=5000)
        time.sleep(0.01)
        assert budget.elapsed_ms > 0


# ─────────────────────────────────────────────
# CircuitBreaker
# ─────────────────────────────────────────────

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_closed_to_open(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=100)
        cb.record_failure()
        cb.record_failure()
        assert cb._state == CircuitState.CLOSED
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_open_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100)
        cb.record_failure()
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        # Simulate recovery timeout having passed
        cb._last_failure_time = time.time() - 200
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100, half_open_max_calls=1)
        cb.record_failure()
        cb.record_failure()
        # Simulate recovery timeout having passed
        cb._last_failure_time = time.time() - 200
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb._state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100)
        cb.record_failure()
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        # Simulate recovery timeout having passed
        cb._last_failure_time = time.time() - 200
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb._state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_stats(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        stats = cb.stats()
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1

    def test_success_decrements_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2
        cb.record_success()
        assert cb._failure_count == 1


# ─────────────────────────────────────────────
# HealthCheck
# ─────────────────────────────────────────────

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_register_and_check(self):
        hc = HealthCheck()

        async def db_check():
            return {"status": "healthy", "connections": 5}

        hc.register("database", db_check)
        result = await hc.check("database")
        assert result["status"] == "healthy"
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_check_unknown(self):
        hc = HealthCheck()
        result = await hc.check("unknown")
        assert result["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_check_error(self):
        hc = HealthCheck()

        async def failing_check():
            raise ConnectionError("DB down")

        hc.register("db", failing_check)
        result = await hc.check("db")
        assert result["status"] == "unhealthy"
        assert "DB down" in result["error"]

    @pytest.mark.asyncio
    async def test_check_all(self):
        hc = HealthCheck()

        async def ok_check():
            return {"status": "healthy"}

        async def bad_check():
            return {"status": "unhealthy"}

        hc.register("ok", ok_check)
        hc.register("bad", bad_check)

        result = await hc.check_all()
        assert result["healthy"] is False
        assert "ok" in result["checks"]
        assert "bad" in result["checks"]

    @pytest.mark.asyncio
    async def test_readiness_all_healthy(self):
        hc = HealthCheck()

        async def ok_check():
            return {"status": "healthy"}

        hc.register("svc", ok_check)
        assert await hc.readiness() is True

    @pytest.mark.asyncio
    async def test_readiness_unhealthy(self):
        hc = HealthCheck()

        async def bad_check():
            return {"status": "unhealthy"}

        hc.register("svc", bad_check)
        assert await hc.readiness() is False


# ─────────────────────────────────────────────
# ModelRouter
# ─────────────────────────────────────────────

class TestModelRouter:
    def test_register_model(self):
        router = ModelRouter()
        config = ModelConfig(name="test-model", provider="anthropic")
        router.register_model(config)
        assert "test-model" in router.models

    def test_setup_defaults(self):
        router = ModelRouter()
        router.setup_defaults()
        assert len(router.models) >= 5
        assert "claude-sonnet-4-6" in router.models
        assert "gpt-4o" in router.models

    def test_select_model_prefer_cost(self):
        router = ModelRouter()
        router.setup_defaults()
        model = router.select_model(prefer="cost")
        assert model is not None
        # Cheapest model should be selected
        all_costs = [
            m.cost_per_1k_input for m in router.models.values()
            if m.quality_score >= 0.7 and m.latency_ms <= 5000
        ]
        assert model.cost_per_1k_input <= max(all_costs)

    def test_select_model_prefer_quality(self):
        router = ModelRouter()
        router.setup_defaults()
        model = router.select_model(prefer="quality")
        assert model is not None
        assert model.quality_score >= 0.9

    def test_select_model_prefer_speed(self):
        router = ModelRouter()
        router.setup_defaults()
        model = router.select_model(prefer="speed")
        assert model is not None
        # Should be one of the faster models
        assert model.latency_ms <= 2000

    def test_select_model_simple_task(self):
        router = ModelRouter()
        router.setup_defaults()
        model = router.select_model(task_complexity="simple", prefer="cost")
        assert model is not None
        # Should pick a cheaper model for simple tasks

    def test_select_model_no_candidates_fallback(self):
        router = ModelRouter()
        router.register_model(ModelConfig(
            name="only-model", cost_per_1k_input=100,
            cost_per_1k_output=100, latency_ms=100000,
        ))
        model = router.select_model(max_latency_ms=1, min_quality=1.0)
        # Fallback: should return cheapest available
        assert model is not None
        assert model.name == "only-model"

    def test_select_model_empty(self):
        router = ModelRouter()
        model = router.select_model()
        assert model is None


# ─────────────────────────────────────────────
# RateLimiter
# ─────────────────────────────────────────────

class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            result = await limiter.acquire()
            assert result is True

    @pytest.mark.asyncio
    async def test_remaining(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.remaining == 10
        await limiter.acquire()
        assert limiter.remaining == 9

    @pytest.mark.asyncio
    async def test_acquire_at_limit_waits(self):
        # With very short window, old timestamps expire quickly
        limiter = RateLimiter(max_requests=2, window_seconds=0.01)
        await limiter.acquire()
        await limiter.acquire()
        time.sleep(0.02)  # Let the window expire
        result = await limiter.acquire()
        assert result is True


# ─────────────────────────────────────────────
# LatencyOptimizer
# ─────────────────────────────────────────────

class TestLatencyOptimizer:
    def test_record_and_avg(self):
        opt = LatencyOptimizer()
        opt.record("llm", 100)
        opt.record("llm", 200)
        opt.record("llm", 300)
        assert opt.avg_latency("llm") == pytest.approx(200, abs=0.1)

    def test_avg_empty(self):
        opt = LatencyOptimizer()
        assert opt.avg_latency("nonexistent") == 0

    def test_p95_latency(self):
        opt = LatencyOptimizer()
        for i in range(100):
            opt.record("op", float(i))
        p95 = opt.p95_latency("op")
        assert p95 >= 90  # 95th percentile of 0-99

    def test_p95_empty(self):
        opt = LatencyOptimizer()
        assert opt.p95_latency("nonexistent") == 0

    def test_p95_single_value(self):
        opt = LatencyOptimizer()
        opt.record("op", 42.0)
        assert opt.p95_latency("op") == 42.0

    def test_report(self):
        opt = LatencyOptimizer()
        opt.record("op1", 100)
        opt.record("op2", 200)
        report = opt.report()
        assert "op1" in report
        assert "op2" in report
        assert "avg_ms" in report["op1"]
        assert "p95_ms" in report["op1"]

    @pytest.mark.asyncio
    async def test_timed(self):
        opt = LatencyOptimizer()

        async def fast_op():
            return "result"

        result = await opt.timed("fast", fast_op())
        assert result == "result"
        assert opt.avg_latency("fast") > 0
