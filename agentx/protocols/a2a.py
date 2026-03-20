"""
AgentX - A2A (Agent-to-Agent) Protocol Support.

Google's A2A protocol enables agents from different systems to discover
and communicate with each other. This module provides:

- AgentCard: Pydantic model describing an agent's capabilities
- A2AServer: Expose AgentX agents as A2A-compatible endpoints
- A2AClient: Connect to external A2A-compatible agents

A2A Routes (registered on AgentXServer):
    GET  /.well-known/agent.json     — Agent card discovery (all agents)
    POST /a2a/tasks                  — Create/send task
    POST /a2a/tasks/stream           — Streaming task (SSE)
    GET  /a2a/tasks/{id}             — Get task status

Reference: https://google.github.io/A2A/
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import time
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx.a2a")


# ---------------------------------------------------------------------------
# A2A Data Models
# ---------------------------------------------------------------------------


class AgentCard(BaseModel):
    """
    JSON document describing an agent's capabilities.
    Served at /.well-known/agent.json for discovery.
    """

    name: str
    description: str
    url: str  # Agent's A2A endpoint URL
    version: str = "1.0"
    capabilities: list[str] = Field(default_factory=list)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=lambda: ["text"])
    output_modes: list[str] = Field(default_factory=lambda: ["text"])
    authentication: dict[str, Any] = Field(default_factory=dict)


class A2ATask(BaseModel):
    """Internal representation of an A2A task."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"  # pending, running, completed, failed, canceled
    message: str = ""
    skill: str = ""
    result: str = ""
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# A2AServer — Expose AgentX agents via A2A protocol
# ---------------------------------------------------------------------------


class A2AServer:
    """
    Expose AgentX agents as A2A-compatible endpoints.

    Usage:
        from agentx import AgentXApp
        from agentx.protocols.a2a import A2AServer

        app = AgentXApp()
        a2a = A2AServer(app, base_url="http://localhost:8080")

        # Register routes on the daemon server
        a2a.register_routes(server)

        # External agents can now:
        # GET  http://localhost:8080/.well-known/agent.json
        # POST http://localhost:8080/a2a/tasks
    """

    def __init__(self, app: Any, base_url: str = "http://localhost:8080"):
        self._app = app
        self._base_url = base_url.rstrip("/")
        self._tasks: dict[str, A2ATask] = {}

    def get_agent_card(self, agent_name: str) -> AgentCard:
        """Build an AgentCard for a registered AgentX agent."""
        if not (self._app and self._app.orchestrator):
            raise ValueError("App or orchestrator not initialized")

        agent = self._app.orchestrator.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        # Build skills list from the agent's tools
        skills = []
        for tool_name, t in agent.tools.items():
            schema = t.get_schema()
            skills.append({
                "id": tool_name,
                "name": tool_name,
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {}),
            })

        # Add the agent's own processing as a skill
        skills.insert(0, {
            "id": f"{agent_name}_chat",
            "name": f"{agent_name} chat",
            "description": agent.config.role or f"Chat with {agent_name}",
        })

        capabilities = ["tasks"]
        if hasattr(agent, "think_stream"):
            capabilities.append("streaming")

        return AgentCard(
            name=agent.name,
            description=agent.config.role or agent.config.system_prompt[:200] or f"AgentX agent: {agent.name}",
            url=f"{self._base_url}/a2a",
            version="1.0",
            capabilities=capabilities,
            skills=skills,
            input_modes=["text"],
            output_modes=["text"],
        )

    def get_all_cards(self) -> list[AgentCard]:
        """Build AgentCards for all registered agents."""
        if not (self._app and self._app.orchestrator):
            return []

        cards = []
        for name in self._app.orchestrator.agents:
            try:
                cards.append(self.get_agent_card(name))
            except Exception as e:
                logger.warning(f"Could not build AgentCard for '{name}': {e}")
        return cards

    async def handle_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """
        Handle an incoming A2A task request.
        Dispatches the task to the appropriate AgentX agent and returns the result.
        """
        if not (self._app and self._app.orchestrator):
            return self._error_response("App not initialized", task_id=task_data.get("id"))

        # Parse the task
        task_id = task_data.get("id", str(uuid.uuid4()))
        message_data = task_data.get("message", {})
        skill = task_data.get("skill", "")

        # Extract text content from message parts
        text_content = self._extract_text(message_data)
        if not text_content:
            return self._error_response("No text content in task message", task_id=task_id)

        # Track the task
        task = A2ATask(
            id=task_id,
            status="running",
            message=text_content,
            skill=skill,
        )
        self._tasks[task_id] = task

        try:
            # Determine target agent from skill or use default routing
            agent_name = self._resolve_agent(skill)

            from ..core.message import AgentMessage, MessageType

            msg = AgentMessage(
                type=MessageType.TASK,
                sender="a2a",
                receiver=agent_name or "",
                content=text_content,
                metadata={"a2a_task_id": task_id, "skill": skill},
            )

            if agent_name:
                result = await self._app.orchestrator.dispatch(msg)
            else:
                result = await self._app.orchestrator.send(content=text_content)

            task.status = "completed"
            task.result = result.content
            task.updated_at = time.time()

            return self._task_response(task)

        except Exception as e:
            logger.error(f"A2A task '{task_id}' failed: {e}")
            task.status = "failed"
            task.result = str(e)
            task.updated_at = time.time()
            return self._task_response(task)

    async def handle_task_stream(self, task_data: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """
        Handle a streaming A2A task request.
        Yields SSE-compatible events as the agent processes the task.
        """
        if not (self._app and self._app.orchestrator):
            yield self._error_response("App not initialized", task_id=task_data.get("id"))
            return

        task_id = task_data.get("id", str(uuid.uuid4()))
        message_data = task_data.get("message", {})
        skill = task_data.get("skill", "")
        text_content = self._extract_text(message_data)

        if not text_content:
            yield self._error_response("No text content in task message", task_id=task_id)
            return

        task = A2ATask(
            id=task_id,
            status="running",
            message=text_content,
            skill=skill,
        )
        self._tasks[task_id] = task

        # Send initial status event
        yield {
            "jsonrpc": "2.0",
            "result": {
                "id": task_id,
                "status": {"state": "working", "message": "Processing..."},
            },
        }

        try:
            agent_name = self._resolve_agent(skill)
            orchestrator = self._app.orchestrator

            # Resolve the target agent for streaming
            if agent_name and agent_name in orchestrator.agents:
                agent = orchestrator.agents[agent_name]
            elif orchestrator._fallback_agent and orchestrator._fallback_agent in orchestrator.agents:
                agent = orchestrator.agents[orchestrator._fallback_agent]
            else:
                agents = list(orchestrator.agents.values())
                if not agents:
                    yield self._error_response("No agents available", task_id=task_id)
                    return
                agent = agents[0]

            accumulated = ""
            async for chunk in agent.think_stream(prompt=text_content):
                if chunk.content:
                    accumulated = chunk.accumulated
                    yield {
                        "jsonrpc": "2.0",
                        "result": {
                            "id": task_id,
                            "status": {"state": "working"},
                            "artifact": {
                                "parts": [{"type": "text", "text": chunk.content}],
                                "index": 0,
                                "append": True,
                            },
                        },
                    }

            task.status = "completed"
            task.result = accumulated
            task.updated_at = time.time()

            yield {
                "jsonrpc": "2.0",
                "result": {
                    "id": task_id,
                    "status": {"state": "completed"},
                    "artifact": {
                        "parts": [{"type": "text", "text": accumulated}],
                        "index": 0,
                    },
                },
            }

        except Exception as e:
            logger.error(f"A2A streaming task '{task_id}' failed: {e}")
            task.status = "failed"
            task.result = str(e)
            task.updated_at = time.time()
            yield self._error_response(str(e), task_id=task_id)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get the status of a previously submitted task."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        return self._task_response(task)

    def register_routes(self, server: Any) -> None:
        """
        Register A2A routes on an AgentXServer instance.
        Supports both aiohttp-based and custom route registration.
        """
        # Use the server's add_route for custom routes that need special handling
        # We need to register directly on the aiohttp app if available
        server._a2a_server = self

        # Store handlers that will be registered when the server starts
        if not hasattr(server, "_a2a_routes_pending"):
            server._a2a_routes_pending = True

            # Monkey-patch the _start_aiohttp to add our routes
            original_start = server._start_aiohttp

            async def patched_start() -> None:
                # Call original to set up the app
                await original_start()
                # Routes are already added below via add_route
                # This patch is just to ensure ordering

            # Instead of patching, use the custom route system for POST endpoints
            # and add GET routes via add_route pattern

        # Register POST handlers via the server's custom route system
        async def handle_task_post(payload: dict) -> dict:
            return await self.handle_task(payload)

        server.add_route("/a2a/tasks", handle_task_post)

        # Store reference so aiohttp routes can be added during start
        self._server_ref = server
        self._register_aiohttp_routes(server)

    def _register_aiohttp_routes(self, server: Any) -> None:
        """Register A2A-specific routes that need direct aiohttp access."""
        original_start = server._start_aiohttp

        a2a_server = self

        async def patched_start_aiohttp() -> None:
            """Wrap the original start to inject A2A routes."""
            from aiohttp import web

            # Save original method's internals — we need to replicate route setup
            # Instead, we patch the app after creation
            original_app_init = web.Application.__init__

            captured_app = []

            orig_add_get = None

            async def start_with_a2a():
                await original_start()

            await start_with_a2a()

            # After the server starts, the aiohttp app is set up via runner
            # We need to add routes before the runner starts, so we use a
            # different approach: override _start_aiohttp entirely

        # Better approach: patch to add routes during app setup
        async def start_aiohttp_with_a2a() -> None:
            from aiohttp import web

            app = web.Application(client_max_size=server.max_request_size)

            # --- Middleware (copied from original) ---
            @web.middleware
            async def auth_middleware(request: web.Request, handler: Any) -> web.Response:
                if request.path in (
                    "/api/v1/health", "/health", "/", "/dashboard",
                    "/.well-known/agent.json",
                ):
                    return await handler(request)
                if server.api_key:
                    auth = request.headers.get("Authorization", "")
                    if auth != f"Bearer {server.api_key}" and request.headers.get("X-API-Key") != server.api_key:
                        return web.json_response(
                            {"status": "error", "message": "Unauthorized", "code": 401},
                            status=401,
                        )
                server._request_count += 1
                return await handler(request)

            @web.middleware
            async def cors_middleware(request: web.Request, handler: Any) -> web.Response:
                if request.method == "OPTIONS":
                    response = web.Response()
                else:
                    try:
                        response = await handler(request)
                    except web.HTTPException as e:
                        response = web.json_response(
                            {"status": "error", "message": str(e), "code": e.status},
                            status=e.status,
                        )
                origin = request.headers.get("Origin", "*")
                if "*" in server.cors_origins or origin in server.cors_origins:
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
                return response

            app.middlewares.extend([auth_middleware, cors_middleware])

            # --- Original routes ---
            app.router.add_get("/", server._handle_dashboard)
            app.router.add_get("/dashboard", server._handle_dashboard)
            app.router.add_post("/api/v1/chat", server._handle_chat)
            app.router.add_post("/api/v1/dispatch", server._handle_dispatch)
            app.router.add_post("/api/v1/pipeline", server._handle_pipeline)
            app.router.add_post("/api/v1/webhook", server._handle_webhook)
            app.router.add_post("/api/v1/webhook/{source}", server._handle_webhook)
            app.router.add_get("/api/v1/health", server._handle_health)
            app.router.add_get("/api/v1/status", server._handle_status)
            app.router.add_get("/api/v1/jobs", server._handle_list_jobs)
            app.router.add_post("/api/v1/jobs/{job_id}/pause", server._handle_pause_job)
            app.router.add_post("/api/v1/jobs/{job_id}/resume", server._handle_resume_job)
            app.router.add_get("/api/v1/metrics", server._handle_metrics)
            app.router.add_get("/api/v1/agents", server._handle_list_agents)
            app.router.add_post("/api/v1/stream", server._handle_stream_sse)

            # --- A2A routes ---
            app.router.add_get("/.well-known/agent.json", a2a_server._aiohttp_agent_card)
            app.router.add_post("/a2a/tasks", a2a_server._aiohttp_handle_task)
            app.router.add_post("/a2a/tasks/stream", a2a_server._aiohttp_handle_task_stream)
            app.router.add_get("/a2a/tasks/{task_id}", a2a_server._aiohttp_get_task)

            # --- Custom routes ---
            for path, handler in server._custom_routes.items():
                if path != "/a2a/tasks":  # Skip our already-registered route
                    app.router.add_post(path, server._wrap_custom_handler(handler))

            # WebSocket
            app.router.add_get("/ws", server._handle_websocket)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, server.host, server.port)
            await site.start()
            server._server = runner
            server._running = True
            logger.info(f"AgentX API server started on http://{server.host}:{server.port} (A2A enabled)")

        server._start_aiohttp = start_aiohttp_with_a2a

    # --- aiohttp route handlers ---

    async def _aiohttp_agent_card(self, request: Any) -> Any:
        """GET /.well-known/agent.json — Return agent cards for discovery."""
        from aiohttp import web

        try:
            cards = self.get_all_cards()
            if len(cards) == 1:
                return web.json_response(cards[0].model_dump())
            return web.json_response([c.model_dump() for c in cards])
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _aiohttp_handle_task(self, request: Any) -> Any:
        """POST /a2a/tasks — Handle an A2A task."""
        from aiohttp import web

        try:
            payload = await request.json()
            result = await self.handle_task(payload)
            status_code = 200 if result.get("result", {}).get("status", {}).get("state") != "failed" else 500
            return web.json_response(result, status=status_code)
        except Exception as e:
            return web.json_response(
                self._error_response(str(e)),
                status=500,
            )

    async def _aiohttp_handle_task_stream(self, request: Any) -> Any:
        """POST /a2a/tasks/stream — Handle a streaming A2A task via SSE."""
        from aiohttp import web

        try:
            payload = await request.json()

            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
            )
            await response.prepare(request)

            async for event in self.handle_task_stream(payload):
                data = json.dumps(event)
                await response.write(f"data: {data}\n\n".encode())

            await response.write(b"data: [DONE]\n\n")
            return response

        except Exception as e:
            return web.json_response(
                self._error_response(str(e)),
                status=500,
            )

    async def _aiohttp_get_task(self, request: Any) -> Any:
        """GET /a2a/tasks/{task_id} — Get task status."""
        from aiohttp import web

        task_id = request.match_info.get("task_id", "")
        result = self.get_task(task_id)
        if result is None:
            return web.json_response(
                self._error_response(f"Task '{task_id}' not found", task_id=task_id),
                status=404,
            )
        return web.json_response(result)

    # --- Helper methods ---

    def _resolve_agent(self, skill: str) -> str | None:
        """Resolve an agent name from a skill identifier."""
        if not skill or not (self._app and self._app.orchestrator):
            return None

        # Try direct agent name match
        if skill in self._app.orchestrator.agents:
            return skill

        # Try matching skill_chat pattern
        if skill.endswith("_chat"):
            agent_name = skill[:-5]
            if agent_name in self._app.orchestrator.agents:
                return agent_name

        # Try matching a tool name to find which agent has it
        for name, agent in self._app.orchestrator.agents.items():
            if skill in agent.tools:
                return name

        return None

    @staticmethod
    def _extract_text(message_data: dict[str, Any] | str) -> str:
        """Extract text content from an A2A message structure."""
        if isinstance(message_data, str):
            return message_data

        # A2A message format: {"role": "user", "parts": [{"type": "text", "text": "..."}]}
        parts = message_data.get("parts", [])
        texts = []
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" or "text" in part:
                    texts.append(part.get("text", ""))

        if texts:
            return "\n".join(texts)

        # Fallback: try content or text fields directly
        return message_data.get("content", message_data.get("text", ""))

    @staticmethod
    def _task_response(task: A2ATask) -> dict[str, Any]:
        """Build an A2A-compliant task response."""
        state_map = {
            "pending": "submitted",
            "running": "working",
            "completed": "completed",
            "failed": "failed",
            "canceled": "canceled",
        }

        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "result": {
                "id": task.id,
                "status": {
                    "state": state_map.get(task.status, task.status),
                },
            },
        }

        if task.result:
            response["result"]["artifact"] = {
                "parts": [{"type": "text", "text": task.result}],
                "index": 0,
            }

        if task.metadata:
            response["result"]["metadata"] = task.metadata

        return response

    @staticmethod
    def _error_response(message: str, task_id: str | None = None) -> dict[str, Any]:
        """Build an A2A-compliant error response."""
        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": message,
            },
        }
        if task_id:
            response["result"] = {
                "id": task_id,
                "status": {"state": "failed", "message": message},
            }
        return response


# ---------------------------------------------------------------------------
# A2AClient — Connect to external A2A agents
# ---------------------------------------------------------------------------


class A2AClient:
    """
    Connect to external A2A-compatible agents.

    Usage:
        client = A2AClient()

        # Discover an external agent
        card = await client.discover("http://other-agent:8080")

        # Send a task
        result = await client.send_task(
            "http://other-agent:8080",
            message="Analyze this data",
            skill="data_analysis",
        )

        # Register as an AgentX tool
        tool = client.register_as_tool(card)
        agent.register_tool(tool)
    """

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def discover(self, url: str) -> AgentCard:
        """
        Discover an A2A agent by fetching its agent card.
        Fetches from {url}/.well-known/agent.json
        """
        import aiohttp

        base_url = url.rstrip("/")
        card_url = f"{base_url}/.well-known/agent.json"

        async with aiohttp.ClientSession() as session:
            async with session.get(card_url, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # Handle both single card and list of cards
                if isinstance(data, list):
                    if not data:
                        raise ValueError(f"No agent cards found at {card_url}")
                    data = data[0]

                return AgentCard(**data)

    async def send_task(
        self,
        url: str,
        message: str,
        skill: str = "",
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send a task to an external A2A agent and wait for the result.
        """
        import aiohttp

        base_url = url.rstrip("/")
        task_url = f"{base_url}/a2a/tasks"

        payload: dict[str, Any] = {
            "id": task_id or str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
            },
        }

        if skill:
            payload["skill"] = skill
        if metadata:
            payload["metadata"] = metadata

        async with aiohttp.ClientSession() as session:
            async with session.post(
                task_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def send_task_stream(
        self,
        url: str,
        message: str,
        skill: str = "",
        task_id: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Send a streaming task to an external A2A agent.
        Yields SSE events as they arrive.
        """
        import aiohttp

        base_url = url.rstrip("/")
        stream_url = f"{base_url}/a2a/tasks/stream"

        payload: dict[str, Any] = {
            "id": task_id or str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
            },
        }

        if skill:
            payload["skill"] = skill

        async with aiohttp.ClientSession() as session:
            async with session.post(
                stream_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                resp.raise_for_status()
                buffer = ""
                async for chunk in resp.content:
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        for line in event_str.split("\n"):
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    return
                                try:
                                    yield json.loads(data_str)
                                except json.JSONDecodeError:
                                    pass

    async def get_task_status(self, url: str, task_id: str) -> dict[str, Any]:
        """Get the status of a previously submitted task."""
        import aiohttp

        base_url = url.rstrip("/")
        status_url = f"{base_url}/a2a/tasks/{task_id}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                status_url,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    def register_as_tool(self, agent_card: AgentCard) -> Any:
        """
        Convert an external A2A agent into an AgentX tool.
        The tool can be registered on any AgentX agent.
        """
        from ..core.tool import BaseTool, ToolResult

        client = self

        class A2ARemoteTool(BaseTool):
            """Tool that delegates to an external A2A agent."""

            name = f"a2a_{agent_card.name}"
            description = agent_card.description

            def __init__(self, card: AgentCard):
                self._card = card
                self.name = f"a2a_{card.name}"
                self.description = card.description

            async def execute(self, message: str = "", skill: str = "", **kwargs: Any) -> ToolResult:
                try:
                    result = await client.send_task(
                        url=self._card.url,
                        message=message,
                        skill=skill,
                    )

                    # Extract response text from A2A result
                    artifact = result.get("result", {}).get("artifact", {})
                    parts = artifact.get("parts", [])
                    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
                    response_text = "\n".join(texts) if texts else str(result)

                    state = result.get("result", {}).get("status", {}).get("state", "")
                    if state == "failed":
                        return ToolResult.fail(response_text or "A2A task failed")

                    return ToolResult.ok(response_text)
                except Exception as e:
                    return ToolResult.fail(f"A2A call to '{self._card.name}' failed: {e}")

            def get_schema(self) -> dict[str, Any]:
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message/task to send to the A2A agent",
                            },
                            "skill": {
                                "type": "string",
                                "description": f"Specific skill to invoke. Available: {', '.join(s.get('id', '') for s in agent_card.skills)}",
                            },
                        },
                        "required": ["message"],
                    },
                }

        return A2ARemoteTool(agent_card)
