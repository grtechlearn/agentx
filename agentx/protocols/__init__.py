"""
AgentX - Protocol integrations for inter-agent communication.

Supported protocols:
- A2A (Agent-to-Agent) — Google's protocol for cross-system agent communication
"""

from .a2a import AgentCard, A2AServer, A2AClient

__all__ = ["AgentCard", "A2AServer", "A2AClient"]
