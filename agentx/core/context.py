"""
AgentX - Execution context passed to agents.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Shared context for an agent execution session."""

    session_id: str = ""
    user_id: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    shared_state: dict[str, Any] = Field(default_factory=dict)
    agent_results: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})

    def set(self, key: str, value: Any) -> None:
        self.shared_state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.shared_state.get(key, default)

    def store_result(self, agent_name: str, result: Any) -> None:
        self.agent_results[agent_name] = result

    def get_result(self, agent_name: str, default: Any = None) -> Any:
        return self.agent_results.get(agent_name, default)

    def get_last_n_messages(self, n: int) -> list[dict[str, str]]:
        return self.conversation_history[-n:]
