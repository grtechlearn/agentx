"""
AgentX - Structured JSON Logging.

Production-grade logging with JSON output, correlation IDs,
and configurable formatters for different environments.

Usage:
    from agentx.utils import setup_logging

    # Development (human-readable)
    setup_logging(level="DEBUG", format="pretty")

    # Production (JSON for log aggregation)
    setup_logging(level="INFO", format="json", file="logs/agentx.log")

    # With correlation ID
    logger = get_logger("agentx.agent", correlation_id="req-123")
    logger.info("Processing request", extra={"user_id": "user-1", "latency_ms": 42})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

# Context variable for request correlation
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def set_correlation_id(correlation_id: str = "") -> str:
    """Set correlation ID for the current async context. Returns the ID."""
    cid = correlation_id or uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get("")


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production log aggregation (ELK, Datadog, etc.)."""

    def __init__(self, service: str = "agentx", include_extras: bool = True):
        super().__init__()
        self.service = service
        self.include_extras = include_extras
        self._skip_fields = {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelno", "lineno", "module", "msecs", "msg",
            "name", "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "taskName",
        }

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service,
        }

        # Add correlation ID if set
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add source location for errors
        if record.levelno >= logging.WARNING:
            log_entry["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        if self.include_extras:
            for key, value in record.__dict__.items():
                if key not in self._skip_fields and not key.startswith("_"):
                    try:
                        json.dumps(value)  # Check serializable
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable colored formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Timestamp
        ts = self.formatTime(record, "%H:%M:%S")

        # Correlation ID
        cid = get_correlation_id()
        cid_str = f" [{cid}]" if cid else ""

        # Level
        level = f"{color}{record.levelname:>8}{reset}"

        # Logger name (shortened)
        name = record.name
        if name.startswith("agentx."):
            name = name[7:]

        # Message
        msg = record.getMessage()

        # Extra fields
        extras = ""
        extra_keys = [k for k in record.__dict__ if k not in {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelno", "lineno", "module", "msecs", "msg",
            "name", "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "taskName", "message",
        } and not k.startswith("_")]
        if extra_keys:
            pairs = []
            for k in extra_keys:
                v = record.__dict__[k]
                pairs.append(f"{k}={v}")
            extras = f" | {', '.join(pairs)}"

        line = f"{ts} {level} {self.BOLD}{name}{reset}{cid_str}: {msg}{extras}"

        # Add exception
        if record.exc_info and record.exc_info[1]:
            line += f"\n{color}{self.formatException(record.exc_info)}{reset}"

        return line


def setup_logging(
    level: str = "INFO",
    format: str = "pretty",
    file: str = "",
    service: str = "agentx",
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure AgentX logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: "pretty" (human-readable) or "json" (structured)
        file: Log file path (empty = stdout only)
        service: Service name for JSON logs
        propagate: Propagate to root logger

    Returns:
        The configured agentx logger
    """
    logger = logging.getLogger("agentx")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = propagate

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter
    if format == "json":
        formatter = JSONFormatter(service=service)
    else:
        formatter = PrettyFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if file:
        os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(file)
        # Always use JSON for file output
        file_handler.setFormatter(JSONFormatter(service=service))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "agentx", **extra: Any) -> logging.LoggerAdapter:
    """
    Get a logger with extra context fields.

    Usage:
        logger = get_logger("agentx.agent", user_id="user-1")
        logger.info("Processing")  # Includes user_id in output
    """
    base_logger = logging.getLogger(name)
    return logging.LoggerAdapter(base_logger, extra)
