"""
Tests for agentx.core module.
Covers: AgentMessage, MessageType, Priority, AgentContext, AgentConfig, AgentState,
        ToolResult, FunctionTool, @tool decorator, LLMConfig, LLMResponse, StreamChunk,
        create_llm, SimpleAgent, Orchestrator.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentx.core.message import AgentMessage, MessageType, Priority
from agentx.core.context import AgentContext
from agentx.core.tool import ToolResult, FunctionTool, tool, BaseTool
from agentx.core.llm import LLMConfig, LLMResponse, StreamChunk, create_llm
from agentx.core.agent import AgentConfig, AgentState, SimpleAgent, BaseAgent
from agentx.core.orchestrator import Orchestrator, Route


# ─────────────────────────────────────────────
# MessageType & Priority enums
# ─────────────────────────────────────────────

class TestMessageType:
    def test_all_values(self):
        assert MessageType.TASK == "task"
        assert MessageType.RESPONSE == "response"
        assert MessageType.ERROR == "error"
        assert MessageType.EVENT == "event"
        assert MessageType.HANDOFF == "handoff"
        assert MessageType.BROADCAST == "broadcast"

    def test_membership(self):
        assert len(MessageType) == 6


class TestPriority:
    def test_all_values(self):
        assert Priority.LOW == "low"
        assert Priority.NORMAL == "normal"
        assert Priority.HIGH == "high"
        assert Priority.CRITICAL == "critical"

    def test_membership(self):
        assert len(Priority) == 4


# ─────────────────────────────────────────────
# AgentMessage
# ─────────────────────────────────────────────

class TestAgentMessage:
    def test_creation_defaults(self):
        msg = AgentMessage()
        assert msg.type == MessageType.TASK
        assert msg.sender == ""
        assert msg.receiver == ""
        assert msg.content == ""
        assert msg.priority == Priority.NORMAL
        assert msg.parent_id is None
        assert msg.id  # UUID generated

    def test_creation_custom(self):
        msg = AgentMessage(
            sender="agent-a",
            receiver="agent-b",
            content="hello",
            priority=Priority.HIGH,
            type=MessageType.EVENT,
            data={"key": "val"},
        )
        assert msg.sender == "agent-a"
        assert msg.receiver == "agent-b"
        assert msg.content == "hello"
        assert msg.priority == Priority.HIGH
        assert msg.type == MessageType.EVENT
        assert msg.data == {"key": "val"}

    def test_reply(self):
        msg = AgentMessage(sender="alice", receiver="bob", content="task")
        reply = msg.reply("done", data={"result": 42})
        assert reply.type == MessageType.RESPONSE
        assert reply.sender == "bob"
        assert reply.receiver == "alice"
        assert reply.content == "done"
        assert reply.parent_id == msg.id
        assert reply.data == {"result": 42}

    def test_error(self):
        msg = AgentMessage(sender="alice", receiver="bob")
        err = msg.error("something failed")
        assert err.type == MessageType.ERROR
        assert err.sender == "bob"
        assert err.receiver == "alice"
        assert err.content == "something failed"
        assert err.parent_id == msg.id

    def test_handoff(self):
        msg = AgentMessage(
            sender="user", receiver="agent-a",
            content="task", data={"ctx": "data"},
        )
        ho = msg.handoff("agent-b", "please handle", data={"extra": 1})
        assert ho.type == MessageType.HANDOFF
        assert ho.sender == "agent-a"
        assert ho.receiver == "agent-b"
        assert ho.content == "please handle"
        assert ho.parent_id == msg.id
        assert ho.data["ctx"] == "data"
        assert ho.data["extra"] == 1

    def test_unique_ids(self):
        m1 = AgentMessage()
        m2 = AgentMessage()
        assert m1.id != m2.id


# ─────────────────────────────────────────────
# AgentContext
# ─────────────────────────────────────────────

class TestAgentContext:
    def test_defaults(self):
        ctx = AgentContext()
        assert ctx.session_id == ""
        assert ctx.user_id == ""
        assert ctx.conversation_history == []
        assert ctx.shared_state == {}
        assert ctx.agent_results == {}

    def test_add_message(self):
        ctx = AgentContext()
        ctx.add_message("user", "hello")
        ctx.add_message("assistant", "hi")
        assert len(ctx.conversation_history) == 2
        assert ctx.conversation_history[0] == {"role": "user", "content": "hello"}

    def test_set_get(self):
        ctx = AgentContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_store_result_get_result(self):
        ctx = AgentContext()
        ctx.store_result("agent-1", {"answer": 42})
        assert ctx.get_result("agent-1") == {"answer": 42}
        assert ctx.get_result("agent-2") is None
        assert ctx.get_result("agent-2", "fallback") == "fallback"

    def test_get_last_n_messages(self):
        ctx = AgentContext()
        for i in range(5):
            ctx.add_message("user", f"msg-{i}")
        last2 = ctx.get_last_n_messages(2)
        assert len(last2) == 2
        assert last2[0]["content"] == "msg-3"
        assert last2[1]["content"] == "msg-4"

    def test_get_last_n_messages_more_than_available(self):
        ctx = AgentContext()
        ctx.add_message("user", "only one")
        result = ctx.get_last_n_messages(10)
        assert len(result) == 1


# ─────────────────────────────────────────────
# AgentConfig & AgentState
# ─────────────────────────────────────────────

class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig(name="test")
        assert cfg.name == "test"
        assert cfg.role == ""
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.provider == "anthropic"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.max_retries == 2
        assert cfg.tools_enabled is True
        assert cfg.metadata == {}

    def test_custom_values(self):
        cfg = AgentConfig(
            name="custom", role="assistant",
            model="gpt-4", provider="openai",
            temperature=0.2, max_tokens=2048,
            max_retries=5, tools_enabled=False,
            metadata={"version": 1},
        )
        assert cfg.name == "custom"
        assert cfg.role == "assistant"
        assert cfg.model == "gpt-4"
        assert cfg.provider == "openai"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 2048
        assert cfg.max_retries == 5
        assert cfg.tools_enabled is False
        assert cfg.metadata == {"version": 1}


class TestAgentState:
    def test_defaults(self):
        s = AgentState()
        assert s.status == "idle"
        assert s.current_task == ""
        assert s.messages_processed == 0
        assert s.errors == []
        assert s.results == []


# ─────────────────────────────────────────────
# ToolResult
# ─────────────────────────────────────────────

class TestToolResult:
    def test_ok(self):
        r = ToolResult.ok(data={"answer": 42})
        assert r.success is True
        assert r.data == {"answer": 42}
        assert r.error is None

    def test_ok_no_data(self):
        r = ToolResult.ok()
        assert r.success is True
        assert r.data is None

    def test_fail(self):
        r = ToolResult.fail("boom")
        assert r.success is False
        assert r.error == "boom"


# ─────────────────────────────────────────────
# FunctionTool & @tool decorator
# ─────────────────────────────────────────────

class TestFunctionTool:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        def add(a: int, b: int) -> int:
            return a + b

        ft = FunctionTool(add, name="add", description="Add two numbers")
        assert ft.name == "add"
        assert ft.description == "Add two numbers"
        result = await ft.execute(a=2, b=3)
        assert result.success is True
        assert result.data == 5

    @pytest.mark.asyncio
    async def test_async_function(self):
        async def fetch(url: str) -> str:
            return f"fetched {url}"

        ft = FunctionTool(fetch, name="fetch")
        result = await ft.execute(url="http://example.com")
        assert result.success is True
        assert result.data == "fetched http://example.com"

    @pytest.mark.asyncio
    async def test_function_error(self):
        def boom():
            raise ValueError("kaboom")

        ft = FunctionTool(boom, name="boom")
        result = await ft.execute()
        assert result.success is False
        assert "kaboom" in result.error

    def test_get_schema(self):
        def greet(name: str, times: int = 1):
            """Say hello."""
            pass

        ft = FunctionTool(greet, name="greet", description="Say hello.")
        schema = ft.get_schema()
        assert schema["name"] == "greet"
        assert schema["description"] == "Say hello."
        assert "parameters" in schema


class TestToolDecorator:
    @pytest.mark.asyncio
    async def test_decorator(self):
        @tool(name="multiply", description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        assert isinstance(multiply, FunctionTool)
        assert multiply.name == "multiply"
        result = await multiply.execute(a=3, b=4)
        assert result.success is True
        assert result.data == 12

    @pytest.mark.asyncio
    async def test_decorator_defaults(self):
        @tool()
        def my_func():
            """A docstring."""
            return "ok"

        assert my_func.name == "my_func"
        assert my_func.description == "A docstring."


# ─────────────────────────────────────────────
# LLMConfig, LLMResponse, StreamChunk
# ─────────────────────────────────────────────

class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.7
        assert cfg.top_p == 1.0
        assert cfg.base_url is None
        assert cfg.api_key == ""


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.usage == {}
        assert r.model == ""
        assert r.raw is None


class TestStreamChunk:
    def test_defaults(self):
        c = StreamChunk()
        assert c.content == ""
        assert c.done is False
        assert c.accumulated == ""

    def test_done_flag(self):
        c = StreamChunk(done=True, accumulated="full text")
        assert c.done is True
        assert c.accumulated == "full text"


# ─────────────────────────────────────────────
# create_llm factory
# ─────────────────────────────────────────────

class TestCreateLLM:
    def test_anthropic(self):
        llm = create_llm(LLMConfig(provider="anthropic"))
        from agentx.core.llm import AnthropicProvider
        assert isinstance(llm, AnthropicProvider)

    def test_claude_alias(self):
        llm = create_llm(LLMConfig(provider="claude"))
        from agentx.core.llm import AnthropicProvider
        assert isinstance(llm, AnthropicProvider)

    def test_openai(self):
        llm = create_llm(LLMConfig(provider="openai"))
        from agentx.core.llm import OpenAIProvider
        assert isinstance(llm, OpenAIProvider)

    def test_gpt_alias(self):
        llm = create_llm(LLMConfig(provider="gpt"))
        from agentx.core.llm import OpenAIProvider
        assert isinstance(llm, OpenAIProvider)

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm(LLMConfig(provider="unknown"))

    def test_kwargs_shortcut(self):
        llm = create_llm(provider="anthropic", model="claude-haiku-4-5-20251001")
        assert llm.config.model == "claude-haiku-4-5-20251001"


# ─────────────────────────────────────────────
# SimpleAgent
# ─────────────────────────────────────────────

class TestSimpleAgent:
    @pytest.mark.asyncio
    async def test_process_message(self):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=LLMResponse(
            content="I am fine",
            usage={"input_tokens": 10, "output_tokens": 5},
        ))

        agent = SimpleAgent(
            config=AgentConfig(name="test-agent", system_prompt="You are helpful."),
            llm=mock_llm,
        )
        assert agent.name == "test-agent"

        msg = AgentMessage(sender="user", receiver="test-agent", content="How are you?")
        ctx = AgentContext(session_id="s1")
        result = await agent.run(msg, ctx)

        assert result.type == MessageType.RESPONSE
        assert result.content == "I am fine"
        assert agent.state.status == "completed"
        assert agent.state.messages_processed == 1
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_tool(self):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=LLMResponse(content="result"))

        @tool(name="calc")
        def calc(expr: str) -> str:
            return "42"

        agent = SimpleAgent(
            config=AgentConfig(name="tool-agent"),
            llm=mock_llm,
            tools=[calc],
        )
        assert "calc" in agent.tools

    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        agent = SimpleAgent(config=AgentConfig(name="err-agent"), llm=mock_llm)
        msg = AgentMessage(sender="user", content="hello")
        ctx = AgentContext()
        result = await agent.run(msg, ctx)

        assert result.type == MessageType.ERROR
        assert agent.state.status == "error"
        assert len(agent.state.errors) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        agent = SimpleAgent(config=AgentConfig(name="a"), llm=AsyncMock())
        result = await agent.execute_tool("nonexistent")
        assert result.success is False
        assert "not found" in result.error


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

def _make_agent(name: str, response_content: str = "ok") -> SimpleAgent:
    """Helper to create a SimpleAgent with a mocked LLM."""
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=LLMResponse(content=response_content))
    return SimpleAgent(config=AgentConfig(name=name), llm=mock_llm)


class TestOrchestrator:
    def test_register(self):
        orch = Orchestrator()
        agent = _make_agent("a")
        orch.register(agent)
        assert "a" in orch.agents
        assert orch.get_agent("a") is agent

    def test_register_many(self):
        orch = Orchestrator()
        a1 = _make_agent("a1")
        a2 = _make_agent("a2")
        orch.register_many(a1, a2)
        assert "a1" in orch.agents
        assert "a2" in orch.agents

    def test_set_fallback(self):
        orch = Orchestrator()
        orch.set_fallback("fallback-agent")
        assert orch._fallback_agent == "fallback-agent"

    def test_add_route(self):
        orch = Orchestrator()
        orch.add_route("agent-a", condition=lambda m, c: "help" in m.content, priority=10)
        assert len(orch.routes) == 1
        assert orch.routes[0].agent_name == "agent-a"

    @pytest.mark.asyncio
    async def test_dispatch_direct_receiver(self):
        orch = Orchestrator()
        agent = _make_agent("target", "handled")
        orch.register(agent)

        msg = AgentMessage(sender="user", receiver="target", content="hello")
        result = await orch.dispatch(msg)
        assert result.type == MessageType.RESPONSE
        assert result.content == "handled"

    @pytest.mark.asyncio
    async def test_dispatch_route(self):
        orch = Orchestrator()
        agent = _make_agent("helper", "helped")
        orch.register(agent)
        orch.add_route("helper", condition=lambda m, c: "help" in m.content)

        msg = AgentMessage(sender="user", content="I need help")
        result = await orch.dispatch(msg)
        assert result.content == "helped"

    @pytest.mark.asyncio
    async def test_dispatch_fallback(self):
        orch = Orchestrator()
        agent = _make_agent("fallback", "fallback response")
        orch.register(agent)
        orch.set_fallback("fallback")

        msg = AgentMessage(sender="user", content="random")
        result = await orch.dispatch(msg)
        assert result.content == "fallback response"

    @pytest.mark.asyncio
    async def test_dispatch_no_agent(self):
        orch = Orchestrator()
        msg = AgentMessage(sender="user", content="hello")
        result = await orch.dispatch(msg)
        assert result.type == MessageType.ERROR

    @pytest.mark.asyncio
    async def test_run_pipeline(self):
        orch = Orchestrator()
        a1 = _make_agent("step1", "step1 output")
        a2 = _make_agent("step2", "step2 output")
        orch.register_many(a1, a2)
        orch.add_pipeline("my_pipe", ["step1", "step2"])

        msg = AgentMessage(sender="user", content="start")
        result = await orch.run_pipeline("my_pipe", msg)
        assert result.content == "step2 output"

    @pytest.mark.asyncio
    async def test_run_pipeline_not_found(self):
        orch = Orchestrator()
        msg = AgentMessage(sender="user", content="start")
        with pytest.raises(ValueError, match="not found"):
            await orch.run_pipeline("nonexistent", msg)

    @pytest.mark.asyncio
    async def test_run_parallel(self):
        orch = Orchestrator()
        a1 = _make_agent("fast", "fast result")
        a2 = _make_agent("slow", "slow result")
        orch.register_many(a1, a2)

        msg = AgentMessage(sender="user", content="go")
        results = await orch.run_parallel(["fast", "slow"], msg)
        assert "fast" in results
        assert "slow" in results
        assert results["fast"].content == "fast result"
        assert results["slow"].content == "slow result"

    def test_sessions(self):
        orch = Orchestrator()
        msg = AgentMessage(
            sender="user", content="hi",
            metadata={"session_id": "sess-1", "user_id": "u1"},
        )
        ctx = orch._get_or_create_session(msg)
        assert ctx.session_id == "sess-1"
        assert ctx.user_id == "u1"
        # Same session_id returns same context
        ctx2 = orch._get_or_create_session(msg)
        assert ctx2 is ctx

    def test_clear_session(self):
        orch = Orchestrator()
        msg = AgentMessage(metadata={"session_id": "s1"})
        orch._get_or_create_session(msg)
        assert "s1" in orch.sessions
        orch.clear_session("s1")
        assert "s1" not in orch.sessions

    @pytest.mark.asyncio
    async def test_events(self):
        orch = Orchestrator()
        events_received = []

        @orch.on("test_event")
        def handler(**kwargs):
            events_received.append(kwargs)

        await orch.emit("test_event", key="value")
        assert len(events_received) == 1
        assert events_received[0]["key"] == "value"

    @pytest.mark.asyncio
    async def test_events_async_handler(self):
        orch = Orchestrator()
        events_received = []

        @orch.on("async_event")
        async def handler(**kwargs):
            events_received.append(kwargs)

        await orch.emit("async_event", data=123)
        assert len(events_received) == 1

    @pytest.mark.asyncio
    async def test_send_convenience(self):
        orch = Orchestrator()
        agent = _make_agent("default", "default reply")
        orch.register(agent)
        orch.set_fallback("default")

        result = await orch.send("hello", session_id="s1", user_id="u1")
        assert result.content == "default reply"
