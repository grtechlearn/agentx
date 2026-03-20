"""
Tests for agentx.protocols.a2a module.
Covers: AgentCard, A2ATask, A2AServer, A2AClient.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agentx.protocols.a2a import AgentCard, A2ATask, A2AServer, A2AClient
from agentx.core.message import AgentMessage, MessageType
from agentx.core.agent import AgentConfig, AgentState, BaseAgent
from agentx.core.tool import BaseTool, ToolResult


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


class DummyAgent(BaseAgent):
    """A minimal agent for testing."""

    async def process(self, message: AgentMessage, context=None) -> AgentMessage:
        return message.reply(content=f"Echo: {message.content}")


class DummyTool(BaseTool):
    name = "search"
    description = "Search the web"

    async def execute(self, query: str = "", **kwargs) -> ToolResult:
        return ToolResult.ok(f"Results for: {query}")


def make_mock_app(agents=None):
    """Create a mock AgentXApp with orchestrator and agents."""
    app = MagicMock()
    orchestrator = MagicMock()

    if agents is None:
        agent = DummyAgent(config=AgentConfig(name="assistant", role="General assistant"))
        agent.register_tool(DummyTool())
        agents = {"assistant": agent}

    orchestrator.agents = agents
    orchestrator.get_agent = lambda name: agents.get(name)
    orchestrator._fallback_agent = None
    app.orchestrator = orchestrator
    return app


# ─────────────────────────────────────────────
# AgentCard
# ─────────────────────────────────────────────


class TestAgentCard:
    def test_creation_defaults(self):
        card = AgentCard(
            name="test-agent",
            description="A test agent",
            url="http://localhost:8080/a2a",
        )
        assert card.name == "test-agent"
        assert card.description == "A test agent"
        assert card.url == "http://localhost:8080/a2a"
        assert card.version == "1.0"
        assert card.capabilities == []
        assert card.skills == []
        assert card.input_modes == ["text"]
        assert card.output_modes == ["text"]
        assert card.authentication == {}

    def test_creation_full(self):
        card = AgentCard(
            name="analyzer",
            description="Analyzes data",
            url="http://example.com/a2a",
            version="2.0",
            capabilities=["tasks", "streaming"],
            skills=[{"id": "analyze", "name": "Analyze", "description": "Analyze data"}],
            input_modes=["text", "file"],
            output_modes=["text", "data"],
            authentication={"type": "bearer"},
        )
        assert card.version == "2.0"
        assert len(card.capabilities) == 2
        assert len(card.skills) == 1
        assert "file" in card.input_modes
        assert card.authentication["type"] == "bearer"

    def test_serialization(self):
        card = AgentCard(
            name="test",
            description="desc",
            url="http://localhost:8080/a2a",
        )
        data = card.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["url"] == "http://localhost:8080/a2a"

        # Round-trip
        card2 = AgentCard(**data)
        assert card2.name == card.name
        assert card2.url == card.url


# ─────────────────────────────────────────────
# A2ATask
# ─────────────────────────────────────────────


class TestA2ATask:
    def test_creation_defaults(self):
        task = A2ATask()
        assert task.status == "pending"
        assert task.message == ""
        assert task.result == ""
        assert task.artifacts == []
        assert len(task.id) > 0

    def test_creation_with_values(self):
        task = A2ATask(
            id="task-123",
            status="running",
            message="Analyze this data",
            skill="analysis",
        )
        assert task.id == "task-123"
        assert task.status == "running"
        assert task.message == "Analyze this data"
        assert task.skill == "analysis"


# ─────────────────────────────────────────────
# A2AServer
# ─────────────────────────────────────────────


class TestA2AServer:
    def test_init(self):
        app = make_mock_app()
        server = A2AServer(app, base_url="http://localhost:9090")
        assert server._base_url == "http://localhost:9090"

    def test_init_strips_trailing_slash(self):
        app = make_mock_app()
        server = A2AServer(app, base_url="http://localhost:9090/")
        assert server._base_url == "http://localhost:9090"

    def test_get_agent_card(self):
        app = make_mock_app()
        server = A2AServer(app, base_url="http://localhost:8080")
        card = server.get_agent_card("assistant")

        assert card.name == "assistant"
        assert card.description == "General assistant"
        assert card.url == "http://localhost:8080/a2a"
        assert "tasks" in card.capabilities
        # Should have the chat skill + the search tool skill
        assert len(card.skills) == 2
        assert card.skills[0]["id"] == "assistant_chat"
        assert card.skills[1]["id"] == "search"

    def test_get_agent_card_not_found(self):
        app = make_mock_app()
        server = A2AServer(app)
        with pytest.raises(ValueError, match="not found"):
            server.get_agent_card("nonexistent")

    def test_get_agent_card_no_orchestrator(self):
        app = MagicMock()
        app.orchestrator = None
        server = A2AServer(app)
        with pytest.raises(ValueError, match="not initialized"):
            server.get_agent_card("any")

    def test_get_all_cards(self):
        agent1 = DummyAgent(config=AgentConfig(name="agent1", role="First"))
        agent2 = DummyAgent(config=AgentConfig(name="agent2", role="Second"))
        app = make_mock_app(agents={"agent1": agent1, "agent2": agent2})
        server = A2AServer(app)

        cards = server.get_all_cards()
        assert len(cards) == 2
        names = {c.name for c in cards}
        assert names == {"agent1", "agent2"}

    def test_get_all_cards_empty(self):
        app = MagicMock()
        app.orchestrator = None
        server = A2AServer(app)
        assert server.get_all_cards() == []

    def test_extract_text_from_parts(self):
        msg = {
            "role": "user",
            "parts": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        assert A2AServer._extract_text(msg) == "Hello\nWorld"

    def test_extract_text_from_string(self):
        assert A2AServer._extract_text("Just a string") == "Just a string"

    def test_extract_text_from_content_field(self):
        assert A2AServer._extract_text({"content": "fallback"}) == "fallback"

    def test_extract_text_string_parts(self):
        msg = {"parts": ["hello", "world"]}
        assert A2AServer._extract_text(msg) == "hello\nworld"

    def test_task_response_format(self):
        task = A2ATask(
            id="t-1",
            status="completed",
            result="Done!",
        )
        resp = A2AServer._task_response(task)
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["id"] == "t-1"
        assert resp["result"]["status"]["state"] == "completed"
        assert resp["result"]["artifact"]["parts"][0]["text"] == "Done!"

    def test_task_response_failed(self):
        task = A2ATask(id="t-2", status="failed", result="Something went wrong")
        resp = A2AServer._task_response(task)
        assert resp["result"]["status"]["state"] == "failed"

    def test_task_response_pending(self):
        task = A2ATask(id="t-3", status="pending")
        resp = A2AServer._task_response(task)
        assert resp["result"]["status"]["state"] == "submitted"
        assert "artifact" not in resp["result"]

    def test_error_response_format(self):
        resp = A2AServer._error_response("bad request", task_id="t-99")
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32000
        assert resp["error"]["message"] == "bad request"
        assert resp["result"]["id"] == "t-99"
        assert resp["result"]["status"]["state"] == "failed"

    def test_error_response_no_task_id(self):
        resp = A2AServer._error_response("server error")
        assert "result" not in resp
        assert resp["error"]["message"] == "server error"

    def test_resolve_agent_direct_name(self):
        app = make_mock_app()
        server = A2AServer(app)
        assert server._resolve_agent("assistant") == "assistant"

    def test_resolve_agent_chat_suffix(self):
        app = make_mock_app()
        server = A2AServer(app)
        assert server._resolve_agent("assistant_chat") == "assistant"

    def test_resolve_agent_by_tool(self):
        app = make_mock_app()
        server = A2AServer(app)
        assert server._resolve_agent("search") == "assistant"

    def test_resolve_agent_empty(self):
        app = make_mock_app()
        server = A2AServer(app)
        assert server._resolve_agent("") is None

    def test_resolve_agent_unknown(self):
        app = make_mock_app()
        server = A2AServer(app)
        assert server._resolve_agent("nonexistent_skill") is None

    def test_get_task_not_found(self):
        app = make_mock_app()
        server = A2AServer(app)
        assert server.get_task("nonexistent") is None


class TestA2AServerHandleTask:
    @pytest.mark.asyncio
    async def test_handle_task_success(self):
        app = make_mock_app()

        # Mock orchestrator.dispatch to return a response
        result_msg = AgentMessage(type=MessageType.RESPONSE, content="Task completed")
        app.orchestrator.dispatch = AsyncMock(return_value=result_msg)

        server = A2AServer(app)
        task_data = {
            "id": "task-abc",
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Do something"}],
            },
            "skill": "assistant",
        }

        result = await server.handle_task(task_data)
        assert result["result"]["id"] == "task-abc"
        assert result["result"]["status"]["state"] == "completed"
        assert "Task completed" in result["result"]["artifact"]["parts"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_task_no_text(self):
        app = make_mock_app()
        server = A2AServer(app)

        result = await server.handle_task({"id": "t-1", "message": {"parts": []}})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_task_no_app(self):
        app = MagicMock()
        app.orchestrator = None
        server = A2AServer(app)

        result = await server.handle_task({"message": "hello"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_task_fallback_routing(self):
        app = make_mock_app()
        result_msg = AgentMessage(type=MessageType.RESPONSE, content="Routed OK")
        app.orchestrator.send = AsyncMock(return_value=result_msg)

        server = A2AServer(app)
        task_data = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "No skill specified"}],
            },
        }

        result = await server.handle_task(task_data)
        assert result["result"]["status"]["state"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_task_exception(self):
        app = make_mock_app()
        app.orchestrator.dispatch = AsyncMock(side_effect=RuntimeError("LLM down"))

        server = A2AServer(app)
        task_data = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "This will fail"}],
            },
            "skill": "assistant",
        }

        result = await server.handle_task(task_data)
        assert result["result"]["status"]["state"] == "failed"

    @pytest.mark.asyncio
    async def test_handle_task_generates_id(self):
        app = make_mock_app()
        result_msg = AgentMessage(type=MessageType.RESPONSE, content="OK")
        app.orchestrator.send = AsyncMock(return_value=result_msg)

        server = A2AServer(app)
        task_data = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Hello"}],
            },
        }

        result = await server.handle_task(task_data)
        # Should have auto-generated an ID
        assert len(result["result"]["id"]) > 0

    @pytest.mark.asyncio
    async def test_get_task_after_handle(self):
        app = make_mock_app()
        result_msg = AgentMessage(type=MessageType.RESPONSE, content="Done")
        app.orchestrator.dispatch = AsyncMock(return_value=result_msg)

        server = A2AServer(app)
        task_data = {
            "id": "track-me",
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Track this"}],
            },
            "skill": "assistant",
        }

        await server.handle_task(task_data)
        status = server.get_task("track-me")
        assert status is not None
        assert status["result"]["id"] == "track-me"
        assert status["result"]["status"]["state"] == "completed"


class TestA2AServerStreaming:
    @pytest.mark.asyncio
    async def test_handle_task_stream_no_app(self):
        app = MagicMock()
        app.orchestrator = None
        server = A2AServer(app)

        events = []
        async for event in server.handle_task_stream({"message": "hello"}):
            events.append(event)
        assert len(events) == 1
        assert "error" in events[0]

    @pytest.mark.asyncio
    async def test_handle_task_stream_no_text(self):
        app = make_mock_app()
        server = A2AServer(app)

        events = []
        async for event in server.handle_task_stream({"message": {"parts": []}}):
            events.append(event)
        assert len(events) == 1
        assert "error" in events[0]


# ─────────────────────────────────────────────
# A2AClient
# ─────────────────────────────────────────────


class TestA2AClient:
    def test_init_default_timeout(self):
        client = A2AClient()
        assert client._timeout == 30.0

    def test_init_custom_timeout(self):
        client = A2AClient(timeout=60.0)
        assert client._timeout == 60.0

    @pytest.mark.asyncio
    async def test_discover(self):
        client = A2AClient()
        card_data = {
            "name": "remote-agent",
            "description": "A remote agent",
            "url": "http://remote:8080/a2a",
            "version": "1.0",
            "capabilities": ["tasks"],
            "skills": [],
            "input_modes": ["text"],
            "output_modes": ["text"],
            "authentication": {},
        }

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=card_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            card = await client.discover("http://remote:8080")

        assert card.name == "remote-agent"
        assert card.url == "http://remote:8080/a2a"

    @pytest.mark.asyncio
    async def test_discover_from_list(self):
        client = A2AClient()
        card_list = [
            {
                "name": "first-agent",
                "description": "First",
                "url": "http://remote:8080/a2a",
            },
            {
                "name": "second-agent",
                "description": "Second",
                "url": "http://remote:8080/a2a",
            },
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=card_list)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            card = await client.discover("http://remote:8080")

        # Should return the first card
        assert card.name == "first-agent"

    @pytest.mark.asyncio
    async def test_send_task(self):
        client = A2AClient()
        expected_response = {
            "jsonrpc": "2.0",
            "result": {
                "id": "task-1",
                "status": {"state": "completed"},
                "artifact": {"parts": [{"type": "text", "text": "Result"}]},
            },
        }

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.send_task(
                url="http://remote:8080",
                message="Analyze this",
                skill="analysis",
            )

        assert result["result"]["status"]["state"] == "completed"

    def test_register_as_tool(self):
        client = A2AClient()
        card = AgentCard(
            name="remote-agent",
            description="Does things remotely",
            url="http://remote:8080/a2a",
            skills=[{"id": "analyze", "name": "Analyze"}],
        )

        tool = client.register_as_tool(card)

        assert tool.name == "a2a_remote-agent"
        assert tool.description == "Does things remotely"

        schema = tool.get_schema()
        assert "message" in schema["parameters"]["properties"]
        assert "skill" in schema["parameters"]["properties"]
        assert "message" in schema["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_register_as_tool_execute_success(self):
        client = A2AClient()
        card = AgentCard(
            name="remote",
            description="Remote agent",
            url="http://remote:8080/a2a",
        )

        tool = client.register_as_tool(card)

        response_data = {
            "result": {
                "status": {"state": "completed"},
                "artifact": {"parts": [{"type": "text", "text": "Analysis done"}]},
            },
        }

        with patch.object(client, "send_task", new_callable=AsyncMock, return_value=response_data):
            result = await tool.execute(message="Do analysis")

        assert result.success is True
        assert "Analysis done" in result.data

    @pytest.mark.asyncio
    async def test_register_as_tool_execute_failure(self):
        client = A2AClient()
        card = AgentCard(
            name="remote",
            description="Remote agent",
            url="http://remote:8080/a2a",
        )

        tool = client.register_as_tool(card)

        response_data = {
            "result": {
                "status": {"state": "failed"},
                "artifact": {"parts": [{"type": "text", "text": "Timeout"}]},
            },
        }

        with patch.object(client, "send_task", new_callable=AsyncMock, return_value=response_data):
            result = await tool.execute(message="Do analysis")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_register_as_tool_execute_exception(self):
        client = A2AClient()
        card = AgentCard(
            name="remote",
            description="Remote agent",
            url="http://remote:8080/a2a",
        )

        tool = client.register_as_tool(card)

        with patch.object(client, "send_task", new_callable=AsyncMock, side_effect=ConnectionError("refused")):
            result = await tool.execute(message="Do analysis")

        assert result.success is False
        assert "refused" in result.error


# ─────────────────────────────────────────────
# A2AServer route registration
# ─────────────────────────────────────────────


class TestA2AServerRouteRegistration:
    def test_register_routes_patches_server(self):
        app = make_mock_app()
        server_mock = MagicMock()
        server_mock._custom_routes = {}
        server_mock._start_aiohttp = AsyncMock()
        server_mock.add_route = MagicMock()

        a2a = A2AServer(app)
        a2a.register_routes(server_mock)

        # Should have added the /a2a/tasks route
        server_mock.add_route.assert_called_once()
        call_args = server_mock.add_route.call_args
        assert call_args[0][0] == "/a2a/tasks"

        # Should have stored a2a_server reference
        assert server_mock._a2a_server is a2a

        # _start_aiohttp should have been replaced
        assert server_mock._start_aiohttp != AsyncMock
