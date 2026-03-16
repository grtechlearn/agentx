"""
AgentX - Orchestrator.
Routes messages to the right agent, manages sessions and agent coordination.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable

from .agent import BaseAgent
from .context import AgentContext
from .message import AgentMessage, MessageType, Priority

logger = logging.getLogger("agentx")


class Route:
    """Defines a routing rule for the orchestrator."""

    def __init__(
        self,
        agent_name: str,
        condition: Callable[[AgentMessage, AgentContext], bool] | None = None,
        priority: int = 0,
    ):
        self.agent_name = agent_name
        self.condition = condition
        self.priority = priority

    def matches(self, message: AgentMessage, context: AgentContext) -> bool:
        if self.condition is None:
            return True
        return self.condition(message, context)


class Pipeline:
    """A sequence of agents to execute in order."""

    def __init__(self, name: str, agents: list[str]):
        self.name = name
        self.agents = agents


class Orchestrator:
    """
    Central coordinator for the multi-agent system.

    Features:
    - Agent registration and discovery
    - Message routing (rule-based or dynamic)
    - Pipeline execution (sequential agent chains)
    - Parallel agent execution
    - Session management
    - Error handling and fallbacks
    """

    def __init__(self, name: str = "orchestrator"):
        self.name = name
        self.agents: dict[str, BaseAgent] = {}
        self.routes: list[Route] = []
        self.pipelines: dict[str, Pipeline] = {}
        self.sessions: dict[str, AgentContext] = {}
        self._middleware: list[Callable[..., Any]] = []
        self._fallback_agent: str | None = None
        self._event_handlers: dict[str, list[Callable[..., Any]]] = {}

    # --- Agent Management ---

    def register(self, agent: BaseAgent) -> Orchestrator:
        """Register an agent."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
        return self

    def register_many(self, *agents: BaseAgent) -> Orchestrator:
        """Register multiple agents at once."""
        for agent in agents:
            self.register(agent)
        return self

    def get_agent(self, name: str) -> BaseAgent | None:
        return self.agents.get(name)

    def set_fallback(self, agent_name: str) -> Orchestrator:
        """Set the fallback agent when no route matches."""
        self._fallback_agent = agent_name
        return self

    # --- Routing ---

    def add_route(
        self,
        agent_name: str,
        condition: Callable[[AgentMessage, AgentContext], bool] | None = None,
        priority: int = 0,
    ) -> Orchestrator:
        """Add a routing rule."""
        self.routes.append(Route(agent_name, condition, priority))
        self.routes.sort(key=lambda r: r.priority, reverse=True)
        return self

    def route_to(self, agent_name: str) -> Callable[..., Any]:
        """Decorator to define a route condition."""
        def decorator(fn: Callable[[AgentMessage, AgentContext], bool]) -> Callable[..., Any]:
            self.add_route(agent_name, fn)
            return fn
        return decorator

    def _resolve_agent(self, message: AgentMessage, context: AgentContext) -> str | None:
        """Find the right agent for a message."""
        # Direct routing: message specifies receiver
        if message.receiver and message.receiver in self.agents:
            return message.receiver

        # Rule-based routing
        for route in self.routes:
            if route.matches(message, context):
                return route.agent_name

        # Fallback
        return self._fallback_agent

    # --- Pipeline ---

    def add_pipeline(self, name: str, agents: list[str]) -> Orchestrator:
        """Define a pipeline (sequential chain of agents)."""
        self.pipelines[name] = Pipeline(name, agents)
        return self

    async def run_pipeline(
        self,
        pipeline_name: str,
        initial_message: AgentMessage,
        context: AgentContext | None = None,
    ) -> AgentMessage:
        """Execute a pipeline of agents sequentially."""
        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")

        ctx = context or self._get_or_create_session(initial_message)
        message = initial_message

        for agent_name in pipeline.agents:
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found in pipeline '{pipeline_name}'")

            logger.info(f"Pipeline '{pipeline_name}' → agent '{agent_name}'")
            message = await agent.run(message, ctx)

            if message.type == MessageType.ERROR:
                logger.error(f"Pipeline '{pipeline_name}' failed at agent '{agent_name}'")
                break

            # Feed output as input to next agent
            if message.type == MessageType.RESPONSE:
                message = AgentMessage(
                    type=MessageType.TASK,
                    sender=agent_name,
                    content=message.content,
                    data=message.data,
                    parent_id=message.id,
                )

        return message

    # --- Parallel Execution ---

    async def run_parallel(
        self,
        agent_names: list[str],
        message: AgentMessage,
        context: AgentContext | None = None,
    ) -> dict[str, AgentMessage]:
        """Run multiple agents in parallel on the same message."""
        ctx = context or self._get_or_create_session(message)
        tasks = {}
        for name in agent_names:
            agent = self.agents.get(name)
            if agent:
                tasks[name] = agent.run(
                    AgentMessage(
                        type=message.type,
                        sender=message.sender,
                        receiver=name,
                        content=message.content,
                        data=message.data,
                    ),
                    ctx,
                )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            name: (
                result if isinstance(result, AgentMessage)
                else AgentMessage(type=MessageType.ERROR, content=str(result))
            )
            for name, result in zip(tasks.keys(), results)
        }

    # --- Session Management ---

    def _get_or_create_session(self, message: AgentMessage) -> AgentContext:
        session_id = message.metadata.get("session_id", str(uuid.uuid4()))
        if session_id not in self.sessions:
            self.sessions[session_id] = AgentContext(
                session_id=session_id,
                user_id=message.metadata.get("user_id", ""),
            )
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> AgentContext | None:
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # --- Events ---

    def on(self, event: str) -> Callable[..., Any]:
        """Register an event handler."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if event not in self._event_handlers:
                self._event_handlers[event] = []
            self._event_handlers[event].append(fn)
            return fn
        return decorator

    async def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                result = handler(**kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler error for '{event}': {e}")

    # --- Main Dispatch ---

    async def dispatch(
        self,
        message: AgentMessage,
        context: AgentContext | None = None,
    ) -> AgentMessage:
        """Route and dispatch a message to the appropriate agent."""
        ctx = context or self._get_or_create_session(message)

        await self.emit("before_dispatch", message=message, context=ctx)

        agent_name = self._resolve_agent(message, ctx)
        if not agent_name:
            return message.error("No agent found to handle this message")

        agent = self.agents.get(agent_name)
        if not agent:
            return message.error(f"Agent '{agent_name}' not registered")

        logger.info(f"Dispatching to agent '{agent_name}': {message.content[:80]}")
        message.receiver = agent_name

        result = await agent.run(message, ctx)

        # Handle handoff: agent wants to pass to another agent
        if result.type == MessageType.HANDOFF:
            logger.info(f"Handoff from '{agent_name}' to '{result.receiver}'")
            return await self.dispatch(result, ctx)

        await self.emit("after_dispatch", message=message, result=result, context=ctx)

        return result

    async def send(
        self,
        content: str,
        session_id: str = "",
        user_id: str = "",
        data: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """Convenience method: send a text message and get a response."""
        message = AgentMessage(
            type=MessageType.TASK,
            sender="user",
            content=content,
            data=data or {},
            metadata={"session_id": session_id or str(uuid.uuid4()), "user_id": user_id},
        )
        return await self.dispatch(message)
