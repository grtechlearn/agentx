"""
AgentX - Message types for inter-agent communication.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    TASK = "task"
    RESPONSE = "response"
    ERROR = "error"
    EVENT = "event"
    HANDOFF = "handoff"
    BROADCAST = "broadcast"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessage(BaseModel):
    """Message passed between agents."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK
    sender: str = ""
    receiver: str = ""
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def reply(self, content: str, data: dict[str, Any] | None = None) -> AgentMessage:
        """Create a reply to this message."""
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.receiver,
            receiver=self.sender,
            content=content,
            data=data or {},
            parent_id=self.id,
        )

    def error(self, error_msg: str) -> AgentMessage:
        """Create an error reply to this message."""
        return AgentMessage(
            type=MessageType.ERROR,
            sender=self.receiver,
            receiver=self.sender,
            content=error_msg,
            parent_id=self.id,
        )

    def handoff(self, target_agent: str, content: str, data: dict[str, Any] | None = None) -> AgentMessage:
        """Hand off this task to another agent."""
        return AgentMessage(
            type=MessageType.HANDOFF,
            sender=self.receiver,
            receiver=target_agent,
            content=content,
            data={**self.data, **(data or {})},
            parent_id=self.id,
        )
