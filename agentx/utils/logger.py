"""
AgentX - Logging and observability.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


def setup_logging(level: str = "INFO", format: str = "detailed") -> logging.Logger:
    """Set up AgentX logging."""
    logger = logging.getLogger("agentx")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        if format == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "agent"):
            log_data["agent"] = record.agent
        if hasattr(record, "data"):
            log_data["data"] = record.data
        if record.exc_info and record.exc_info[1]:
            log_data["error"] = str(record.exc_info[1])
        return json.dumps(log_data)


@dataclass
class AgentMetrics:
    """Track agent performance metrics."""

    agent_name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_duration_ms: float = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / max(self.total_runs, 1)

    @property
    def success_rate(self) -> float:
        return self.successful_runs / max(self.total_runs, 1)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "total_runs": self.total_runs,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_duration_ms": f"{self.avg_duration_ms:.0f}",
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "errors": len(self.errors),
        }


class MetricsCollector:
    """Collect and report metrics for all agents."""

    def __init__(self) -> None:
        self._metrics: dict[str, AgentMetrics] = {}

    def get_or_create(self, agent_name: str) -> AgentMetrics:
        if agent_name not in self._metrics:
            self._metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        return self._metrics[agent_name]

    @asynccontextmanager
    async def track(self, agent_name: str) -> AsyncIterator[AgentMetrics]:
        metrics = self.get_or_create(agent_name)
        metrics.total_runs += 1
        start = time.monotonic()
        try:
            yield metrics
            metrics.successful_runs += 1
        except Exception as e:
            metrics.failed_runs += 1
            metrics.errors.append(str(e))
            raise
        finally:
            metrics.total_duration_ms += (time.monotonic() - start) * 1000

    def report(self) -> dict[str, Any]:
        return {name: m.to_dict() for name, m in self._metrics.items()}

    def reset(self) -> None:
        self._metrics.clear()


# Global metrics collector
metrics = MetricsCollector()
