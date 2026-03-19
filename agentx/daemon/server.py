"""
AgentX - HTTP/WebSocket API Server.

Lightweight async server for agent communication, webhooks, and admin API.
Built on aiohttp (optional) with fallback to raw asyncio TCP server.

Endpoints:
    POST /api/v1/chat         — Send message to agent
    POST /api/v1/dispatch     — Dispatch to specific agent
    POST /api/v1/pipeline     — Run agent pipeline
    POST /api/v1/webhook      — Receive external webhook
    GET  /api/v1/health       — Health check
    GET  /api/v1/status       — Daemon status
    GET  /api/v1/jobs         — List scheduled jobs
    POST /api/v1/jobs/{id}/pause   — Pause a job
    POST /api/v1/jobs/{id}/resume  — Resume a job
    GET  /api/v1/metrics      — System metrics
    WS   /ws                  — WebSocket for real-time agent communication

Usage:
    server = AgentXServer(app=agentx_app, host="0.0.0.0", port=8080)
    await server.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Awaitable

logger = logging.getLogger("agentx.daemon")


class APIResponse:
    """Standard API response."""

    @staticmethod
    def success(data: Any = None, message: str = "ok") -> dict[str, Any]:
        return {"status": "success", "message": message, "data": data, "timestamp": time.time()}

    @staticmethod
    def error(message: str, code: int = 400) -> dict[str, Any]:
        return {"status": "error", "message": message, "code": code, "timestamp": time.time()}


class WebhookHandler:
    """Registry for webhook handlers."""

    def __init__(self):
        self._handlers: dict[str, list[Callable[..., Awaitable[Any]]]] = {}
        self._global_handlers: list[Callable[..., Awaitable[Any]]] = []

    def register(
        self,
        source: str = "",
        handler: Callable[..., Awaitable[Any]] | None = None,
    ) -> Callable:
        """Register a webhook handler. Can be used as decorator."""
        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable:
            if source:
                if source not in self._handlers:
                    self._handlers[source] = []
                self._handlers[source].append(fn)
            else:
                self._global_handlers.append(fn)
            return fn

        if handler:
            decorator(handler)
            return handler
        return decorator

    async def process(self, source: str, payload: dict[str, Any]) -> list[Any]:
        """Process a webhook by calling all matching handlers."""
        results = []
        handlers = self._handlers.get(source, []) + self._global_handlers
        for handler in handlers:
            try:
                result = await handler(source, payload)
                results.append(result)
            except Exception as e:
                logger.error(f"Webhook handler error for '{source}': {e}")
                results.append({"error": str(e)})
        return results


class AgentXServer:
    """
    HTTP/WebSocket server for AgentX autonomous operation.

    Provides REST API and WebSocket endpoints for:
    - Agent chat/dispatch
    - Webhook ingestion
    - Health monitoring
    - Job management
    - Real-time communication
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        api_key: str = "",
        cors_origins: list[str] | None = None,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.cors_origins = cors_origins or ["*"]
        self.max_request_size = max_request_size
        self.webhooks = WebhookHandler()

        self._app_ref: Any = None  # AgentXApp reference (set by daemon)
        self._daemon_ref: Any = None  # AgentXDaemon reference
        self._server: Any = None
        self._running = False
        self._request_count = 0
        self._error_count = 0
        self._start_time = 0.0

        # Custom route handlers
        self._custom_routes: dict[str, Callable[..., Awaitable[dict]]] = {}

    def set_app(self, app: Any) -> None:
        """Set the AgentXApp reference."""
        self._app_ref = app

    def set_daemon(self, daemon: Any) -> None:
        """Set the AgentXDaemon reference."""
        self._daemon_ref = daemon

    def add_route(self, path: str, handler: Callable[..., Awaitable[dict]]) -> None:
        """Add a custom API route."""
        self._custom_routes[path] = handler

    async def start(self) -> None:
        """Start the HTTP server."""
        if self._running:
            return

        self._start_time = time.time()

        try:
            # Try aiohttp first (full-featured)
            await self._start_aiohttp()
        except ImportError:
            # Fallback to raw asyncio TCP server
            await self._start_raw()

    async def _start_aiohttp(self) -> None:
        """Start with aiohttp (recommended)."""
        from aiohttp import web

        app = web.Application(client_max_size=self.max_request_size)

        # Middleware
        @web.middleware
        async def auth_middleware(request: web.Request, handler: Any) -> web.Response:
            # Skip auth for health check and dashboard
            if request.path in ("/api/v1/health", "/health", "/", "/dashboard"):
                return await handler(request)
            # Check API key if configured
            if self.api_key:
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {self.api_key}" and request.headers.get("X-API-Key") != self.api_key:
                    return web.json_response(
                        APIResponse.error("Unauthorized", 401), status=401
                    )
            self._request_count += 1
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
                        APIResponse.error(str(e), e.status), status=e.status
                    )
            origin = request.headers.get("Origin", "*")
            if "*" in self.cors_origins or origin in self.cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
            return response

        app.middlewares.extend([auth_middleware, cors_middleware])

        # Dashboard routes
        app.router.add_get("/", self._handle_dashboard)
        app.router.add_get("/dashboard", self._handle_dashboard)

        # Routes
        app.router.add_post("/api/v1/chat", self._handle_chat)
        app.router.add_post("/api/v1/dispatch", self._handle_dispatch)
        app.router.add_post("/api/v1/pipeline", self._handle_pipeline)
        app.router.add_post("/api/v1/webhook", self._handle_webhook)
        app.router.add_post("/api/v1/webhook/{source}", self._handle_webhook)
        app.router.add_get("/api/v1/health", self._handle_health)
        app.router.add_get("/api/v1/status", self._handle_status)
        app.router.add_get("/api/v1/jobs", self._handle_list_jobs)
        app.router.add_post("/api/v1/jobs/{job_id}/pause", self._handle_pause_job)
        app.router.add_post("/api/v1/jobs/{job_id}/resume", self._handle_resume_job)
        app.router.add_get("/api/v1/metrics", self._handle_metrics)
        app.router.add_get("/api/v1/agents", self._handle_list_agents)
        app.router.add_post("/api/v1/stream", self._handle_stream_sse)

        # Custom routes
        for path, handler in self._custom_routes.items():
            app.router.add_post(path, self._wrap_custom_handler(handler))

        # WebSocket
        app.router.add_get("/ws", self._handle_websocket)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._server = runner
        self._running = True
        logger.info(f"AgentX API server started on http://{self.host}:{self.port}")

    async def _start_raw(self) -> None:
        """Fallback: minimal HTTP server using raw asyncio."""
        async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                data = await asyncio.wait_for(reader.read(self.max_request_size), timeout=30)
                request_str = data.decode("utf-8", errors="replace")

                # Parse minimal HTTP
                lines = request_str.split("\r\n")
                if not lines:
                    writer.close()
                    return

                method, path, *_ = lines[0].split(" ", 2)
                body = request_str.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in request_str else ""

                # Route
                response_data = await self._raw_route(method, path, body)

                # Serve dashboard as HTML
                if isinstance(response_data, dict) and response_data.get("__dashboard__"):
                    from ..dashboard import get_dashboard_path
                    try:
                        with open(get_dashboard_path(), "r", encoding="utf-8") as f:
                            html = f.read()
                        http_response = (
                            "HTTP/1.1 200 OK\r\n"
                            "Content-Type: text/html; charset=utf-8\r\n"
                            f"Content-Length: {len(html.encode('utf-8'))}\r\n"
                            "\r\n"
                            f"{html}"
                        )
                    except FileNotFoundError:
                        http_response = (
                            "HTTP/1.1 404 Not Found\r\n"
                            "Content-Type: text/plain\r\n"
                            "Content-Length: 19\r\n"
                            "\r\n"
                            "Dashboard not found"
                        )
                else:
                    response_json = json.dumps(response_data)
                    http_response = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(response_json)}\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "\r\n"
                        f"{response_json}"
                    )
                writer.write(http_response.encode())
                await writer.drain()
            except Exception as e:
                logger.error(f"Raw server error: {e}")
            finally:
                writer.close()
                await writer.wait_closed()

        self._server = await asyncio.start_server(
            handle_connection, self.host, self.port
        )
        self._running = True
        logger.info(
            f"AgentX raw API server started on http://{self.host}:{self.port} "
            "(install aiohttp for full features)"
        )

    async def _raw_route(self, method: str, path: str, body: str) -> dict[str, Any]:
        """Route requests for the raw server."""
        self._request_count += 1
        try:
            payload = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            payload = {}

        if path in ("/", "/dashboard"):
            return {"__dashboard__": True}
        elif path == "/api/v1/health" or path == "/health":
            return await self._health_data()
        elif path == "/api/v1/status":
            return await self._status_data()
        elif path == "/api/v1/chat" and method == "POST":
            return await self._chat_logic(payload)
        elif path == "/api/v1/webhook" and method == "POST":
            return await self._webhook_logic("default", payload)
        else:
            return APIResponse.error(f"Not found: {path}", 404)

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._server:
            if hasattr(self._server, 'cleanup'):
                await self._server.cleanup()
            elif hasattr(self._server, 'close'):
                self._server.close()
                await self._server.wait_closed()
        logger.info("API server stopped")

    # --- aiohttp Route Handlers ---

    async def _handle_dashboard(self, request: Any) -> Any:
        """Serve the admin dashboard HTML."""
        from aiohttp import web
        from ..dashboard import get_dashboard_path

        dashboard_path = get_dashboard_path()
        try:
            with open(dashboard_path, "r", encoding="utf-8") as f:
                html = f.read()
            return web.Response(text=html, content_type="text/html")
        except FileNotFoundError:
            return web.Response(text="Dashboard not found", status=404)

    async def _handle_chat(self, request: Any) -> Any:
        from aiohttp import web
        try:
            payload = await request.json()
            result = await self._chat_logic(payload)
            return web.json_response(result)
        except Exception as e:
            self._error_count += 1
            return web.json_response(APIResponse.error(str(e)), status=500)

    async def _handle_dispatch(self, request: Any) -> Any:
        from aiohttp import web
        try:
            payload = await request.json()
            agent_name = payload.get("agent", "")
            message = payload.get("message", "")
            session_id = payload.get("session_id", "")

            if not agent_name or not message:
                return web.json_response(
                    APIResponse.error("'agent' and 'message' required"), status=400
                )

            if self._app_ref and self._app_ref.orchestrator:
                from ..core.message import AgentMessage, MessageType
                msg = AgentMessage(
                    type=MessageType.TASK,
                    sender="api",
                    receiver=agent_name,
                    content=message,
                    metadata={"session_id": session_id},
                )
                result = await self._app_ref.orchestrator.dispatch(msg)
                return web.json_response(APIResponse.success({
                    "response": result.content,
                    "agent": agent_name,
                    "data": result.data,
                }))

            return web.json_response(APIResponse.error("App not initialized"), status=503)
        except Exception as e:
            self._error_count += 1
            return web.json_response(APIResponse.error(str(e)), status=500)

    async def _handle_pipeline(self, request: Any) -> Any:
        from aiohttp import web
        try:
            payload = await request.json()
            pipeline_name = payload.get("pipeline", "")
            message = payload.get("message", "")

            if not pipeline_name or not message:
                return web.json_response(
                    APIResponse.error("'pipeline' and 'message' required"), status=400
                )

            if self._app_ref and self._app_ref.orchestrator:
                from ..core.message import AgentMessage, MessageType
                msg = AgentMessage(type=MessageType.TASK, sender="api", content=message)
                result = await self._app_ref.orchestrator.run_pipeline(pipeline_name, msg)
                return web.json_response(APIResponse.success({
                    "response": result.content,
                    "pipeline": pipeline_name,
                }))

            return web.json_response(APIResponse.error("App not initialized"), status=503)
        except Exception as e:
            self._error_count += 1
            return web.json_response(APIResponse.error(str(e)), status=500)

    async def _handle_webhook(self, request: Any) -> Any:
        from aiohttp import web
        try:
            source = request.match_info.get("source", "default")
            payload = await request.json()
            result = await self._webhook_logic(source, payload)
            return web.json_response(result)
        except Exception as e:
            self._error_count += 1
            return web.json_response(APIResponse.error(str(e)), status=500)

    async def _handle_health(self, request: Any) -> Any:
        from aiohttp import web
        data = await self._health_data()
        status = 200 if data.get("data", {}).get("healthy") else 503
        return web.json_response(data, status=status)

    async def _handle_status(self, request: Any) -> Any:
        from aiohttp import web
        data = await self._status_data()
        return web.json_response(data)

    async def _handle_list_jobs(self, request: Any) -> Any:
        from aiohttp import web
        if self._daemon_ref and self._daemon_ref.scheduler:
            jobs = self._daemon_ref.scheduler.list_jobs()
            return web.json_response(APIResponse.success(jobs))
        return web.json_response(APIResponse.error("Scheduler not available"), status=503)

    async def _handle_pause_job(self, request: Any) -> Any:
        from aiohttp import web
        job_id = request.match_info.get("job_id", "")
        if self._daemon_ref and self._daemon_ref.scheduler:
            if self._daemon_ref.scheduler.pause_job(job_id):
                return web.json_response(APIResponse.success(message=f"Job {job_id} paused"))
            return web.json_response(APIResponse.error(f"Job {job_id} not found"), status=404)
        return web.json_response(APIResponse.error("Scheduler not available"), status=503)

    async def _handle_resume_job(self, request: Any) -> Any:
        from aiohttp import web
        job_id = request.match_info.get("job_id", "")
        if self._daemon_ref and self._daemon_ref.scheduler:
            if self._daemon_ref.scheduler.resume_job(job_id):
                return web.json_response(APIResponse.success(message=f"Job {job_id} resumed"))
            return web.json_response(APIResponse.error(f"Job {job_id} not found"), status=404)
        return web.json_response(APIResponse.error("Scheduler not available"), status=503)

    async def _handle_metrics(self, request: Any) -> Any:
        from aiohttp import web
        metrics = {
            "server": {
                "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
                "total_requests": self._request_count,
                "total_errors": self._error_count,
                "error_rate": self._error_count / max(1, self._request_count),
            },
        }
        if self._daemon_ref:
            metrics["daemon"] = self._daemon_ref.stats()
        if self._app_ref:
            metrics["app"] = self._app_ref.summary()
        return web.json_response(APIResponse.success(metrics))

    async def _handle_list_agents(self, request: Any) -> Any:
        from aiohttp import web
        if self._app_ref and self._app_ref.orchestrator:
            agents = []
            for name, agent in self._app_ref.orchestrator.agents.items():
                agents.append({
                    "name": name,
                    "status": agent.state.status,
                    "messages_processed": agent.state.messages_processed,
                    "role": agent.config.role,
                })
            return web.json_response(APIResponse.success(agents))
        return web.json_response(APIResponse.error("App not initialized"), status=503)

    async def _handle_stream_sse(self, request: Any) -> Any:
        """SSE (Server-Sent Events) streaming endpoint for REST clients."""
        from aiohttp import web
        try:
            payload = await request.json()
            message = payload.get("message", "")
            agent_name = payload.get("agent", "")

            if not message:
                return web.json_response(APIResponse.error("'message' required"), status=400)

            if not (self._app_ref and self._app_ref.orchestrator):
                return web.json_response(APIResponse.error("App not initialized"), status=503)

            orchestrator = self._app_ref.orchestrator

            # Resolve agent
            if agent_name and agent_name in orchestrator.agents:
                agent = orchestrator.agents[agent_name]
            elif orchestrator._fallback_agent and orchestrator._fallback_agent in orchestrator.agents:
                agent = orchestrator.agents[orchestrator._fallback_agent]
            else:
                agents = list(orchestrator.agents.values())
                if not agents:
                    return web.json_response(APIResponse.error("No agents"), status=503)
                agent = agents[0]

            # SSE response
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

            async for chunk in agent.think_stream(prompt=message):
                if chunk.content:
                    data = json.dumps({"content": chunk.content, "accumulated": chunk.accumulated})
                    await response.write(f"data: {data}\n\n".encode())
                if chunk.done:
                    data = json.dumps({
                        "content": chunk.accumulated, "done": True,
                        "usage": chunk.usage, "model": chunk.model,
                    })
                    await response.write(f"data: {data}\n\n".encode())

            await response.write(b"data: [DONE]\n\n")
            return response

        except Exception as e:
            self._error_count += 1
            return web.json_response(APIResponse.error(str(e)), status=500)

    async def _handle_websocket(self, request: Any) -> Any:
        from aiohttp import web
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        logger.info("WebSocket client connected")

        try:
            async for msg in ws:
                if msg.type == 1:  # TEXT
                    try:
                        data = json.loads(msg.data)
                        action = data.get("action", "chat")

                        if action == "chat":
                            result = await self._chat_logic(data)
                            await ws.send_json(result)

                        elif action == "stream":
                            # Streaming chat — sends token-by-token
                            await self._handle_stream_chat(ws, data)

                        elif action == "status":
                            status = await self._status_data()
                            await ws.send_json(status)
                        elif action == "ping":
                            await ws.send_json({"action": "pong", "timestamp": time.time()})
                        else:
                            await ws.send_json(APIResponse.error(f"Unknown action: {action}"))
                    except json.JSONDecodeError:
                        await ws.send_json(APIResponse.error("Invalid JSON"))
                elif msg.type == 8:  # CLOSE
                    break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            logger.info("WebSocket client disconnected")

        return ws

    async def _handle_stream_chat(self, ws: Any, data: dict[str, Any]) -> None:
        """Handle streaming chat over WebSocket — sends tokens as they arrive."""
        message = data.get("message", "")
        session_id = data.get("session_id", "")
        user_id = data.get("user_id", "")
        agent_name = data.get("agent", "")

        if not message:
            await ws.send_json({"type": "error", "message": "'message' required"})
            return

        if not (self._app_ref and self._app_ref.orchestrator):
            await ws.send_json({"type": "error", "message": "App not initialized"})
            return

        try:
            # Resolve the target agent
            orchestrator = self._app_ref.orchestrator
            if agent_name and agent_name in orchestrator.agents:
                agent = orchestrator.agents[agent_name]
            elif orchestrator._fallback_agent and orchestrator._fallback_agent in orchestrator.agents:
                agent = orchestrator.agents[orchestrator._fallback_agent]
            else:
                agents = list(orchestrator.agents.values())
                if not agents:
                    await ws.send_json({"type": "error", "message": "No agents registered"})
                    return
                agent = agents[0]

            # Send start event
            await ws.send_json({
                "type": "stream_start",
                "agent": agent.name,
                "session_id": session_id,
            })

            # Stream tokens
            async for chunk in agent.think_stream(prompt=message):
                if chunk.content:
                    await ws.send_json({
                        "type": "stream_token",
                        "content": chunk.content,
                        "accumulated": chunk.accumulated,
                    })
                if chunk.done:
                    await ws.send_json({
                        "type": "stream_end",
                        "content": chunk.accumulated,
                        "usage": chunk.usage,
                        "model": chunk.model,
                    })

        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            await ws.send_json({"type": "error", "message": str(e)})

    def _wrap_custom_handler(self, handler: Callable) -> Callable:
        """Wrap a custom handler for aiohttp."""
        async def wrapped(request: Any) -> Any:
            from aiohttp import web
            try:
                payload = await request.json() if request.can_read_body else {}
                result = await handler(payload)
                return web.json_response(APIResponse.success(result))
            except Exception as e:
                return web.json_response(APIResponse.error(str(e)), status=500)
        return wrapped

    # --- Logic (shared between aiohttp and raw) ---

    async def _chat_logic(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process a chat message."""
        message = payload.get("message", "")
        session_id = payload.get("session_id", "")
        user_id = payload.get("user_id", "")

        if not message:
            return APIResponse.error("'message' required")

        if self._app_ref and self._app_ref.orchestrator:
            result = await self._app_ref.orchestrator.send(
                content=message,
                session_id=session_id,
                user_id=user_id,
            )
            return APIResponse.success({
                "response": result.content,
                "session_id": session_id,
                "data": result.data,
            })

        return APIResponse.error("App not initialized")

    async def _webhook_logic(self, source: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Process a webhook."""
        results = await self.webhooks.process(source, payload)

        # Also trigger scheduler events
        if self._daemon_ref and self._daemon_ref.scheduler:
            event_runs = await self._daemon_ref.scheduler.trigger_event(
                f"webhook:{source}", payload
            )
            results.extend([{"job": r.job_id, "status": r.status.value} for r in event_runs])

        return APIResponse.success(results, f"Webhook '{source}' processed")

    async def _health_data(self) -> dict[str, Any]:
        """Get health check data."""
        health = {"healthy": True, "checks": {}}

        if self._app_ref:
            if self._app_ref.health:
                health = await self._app_ref.health.check_all()
            health["checks"]["app"] = {"status": "healthy" if self._app_ref.is_started else "unhealthy"}

        if self._daemon_ref:
            health["checks"]["daemon"] = {
                "status": "healthy" if self._daemon_ref._running else "unhealthy",
                "uptime_seconds": time.time() - self._daemon_ref._start_time if self._daemon_ref._start_time else 0,
            }

        return APIResponse.success(health)

    async def _status_data(self) -> dict[str, Any]:
        """Get full daemon status."""
        status: dict[str, Any] = {
            "server": {
                "host": self.host,
                "port": self.port,
                "running": self._running,
                "requests": self._request_count,
                "errors": self._error_count,
                "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            },
        }

        if self._daemon_ref:
            status["daemon"] = self._daemon_ref.stats()
        if self._app_ref:
            status["app"] = self._app_ref.summary()

        return APIResponse.success(status)

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "requests": self._request_count,
            "errors": self._error_count,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
        }
