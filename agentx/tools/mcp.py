"""
AgentX - MCP (Model Context Protocol) Integration.
Connect to any MCP server and use its tools, resources, and prompts.

MCP is Anthropic's open protocol for connecting AI to external tools/data.
This module lets AgentX agents use ANY MCP server as a tool source.

Supports:
- stdio transport (local MCP servers)
- SSE transport (remote MCP servers)
- Streamable HTTP transport
- Auto-discovery of tools from MCP servers
- Converting MCP tools to AgentX tools
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..core.tool import BaseTool, ToolResult

logger = logging.getLogger("agentx")


class MCPTool(BaseTool):
    """
    An AgentX tool backed by an MCP server tool.
    Automatically created when connecting to an MCP server.
    """

    def __init__(self, mcp_tool: Any, session: Any):
        self._mcp_tool = mcp_tool
        self._session = session
        self.name = mcp_tool.name
        self.description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        self._input_schema = getattr(mcp_tool, "inputSchema", {}) or {}

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool via the MCP session."""
        try:
            from mcp.types import CallToolResult

            result = await self._session.call_tool(self._mcp_tool.name, arguments=kwargs)

            # Extract content from MCP result
            if hasattr(result, "content") and result.content:
                texts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                    elif hasattr(block, "data"):
                        texts.append(str(block.data))
                    else:
                        texts.append(str(block))
                output = "\n".join(texts)
            else:
                output = str(result)

            is_error = getattr(result, "isError", False)
            if is_error:
                return ToolResult.fail(output)
            return ToolResult.ok(output)
        except Exception as e:
            logger.error(f"MCP tool '{self.name}' execution failed: {e}")
            return ToolResult.fail(str(e))

    def get_schema(self) -> dict[str, Any]:
        """Return the MCP tool's input schema in AgentX format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._input_schema if self._input_schema else {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


class MCPResource:
    """Represents an MCP resource (read-only data source)."""

    def __init__(self, resource: Any, session: Any):
        self._resource = resource
        self._session = session
        self.uri = str(resource.uri) if hasattr(resource, "uri") else str(resource)
        self.name = getattr(resource, "name", self.uri)
        self.description = getattr(resource, "description", "")
        self.mime_type = getattr(resource, "mimeType", "text/plain")

    async def read(self) -> str:
        """Read the resource content."""
        try:
            result = await self._session.read_resource(self.uri)
            if hasattr(result, "contents") and result.contents:
                texts = []
                for content in result.contents:
                    if hasattr(content, "text"):
                        texts.append(content.text)
                    elif hasattr(content, "blob"):
                        texts.append(f"[Binary data: {len(content.blob)} bytes]")
                return "\n".join(texts)
            return str(result)
        except Exception as e:
            logger.error(f"MCP resource read failed for '{self.uri}': {e}")
            return f"Error reading resource: {e}"


class MCPConnection:
    """
    Connection to a single MCP server.

    Usage:
        # Connect to a local stdio MCP server
        conn = MCPConnection(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        await conn.connect()
        tools = conn.get_agentx_tools()

        # Connect to a remote SSE MCP server
        conn = MCPConnection(
            name="remote-db",
            url="http://localhost:8080/sse",
            transport="sse",
        )
        await conn.connect()
    """

    def __init__(
        self,
        name: str,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str = "",
        transport: str = "stdio",  # stdio, sse, streamable_http
        headers: dict[str, str] | None = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.url = url
        self.transport = transport
        self.headers = headers or {}
        self._session: Any = None
        self._client: Any = None
        self._cleanup: Any = None
        self._tools: list[MCPTool] = []
        self._resources: list[MCPResource] = []
        self._connected = False

    async def connect(self) -> None:
        """Connect to the MCP server and discover tools/resources."""
        try:
            from mcp import ClientSession
        except ImportError:
            raise ImportError(
                "MCP SDK not installed. Install it with: pip install mcp\n"
                "Or add to AgentX: pip install agentx[mcp]"
            )

        if self.transport == "stdio":
            await self._connect_stdio()
        elif self.transport == "sse":
            await self._connect_sse()
        elif self.transport == "streamable_http":
            await self._connect_streamable_http()
        else:
            raise ValueError(f"Unknown transport: {self.transport}. Use: stdio, sse, streamable_http")

        # Discover tools
        try:
            tools_result = await self._session.list_tools()
            self._tools = [MCPTool(t, self._session) for t in tools_result.tools]
            logger.info(f"MCP '{self.name}': discovered {len(self._tools)} tools")
        except Exception as e:
            logger.warning(f"MCP '{self.name}': could not list tools: {e}")

        # Discover resources
        try:
            resources_result = await self._session.list_resources()
            self._resources = [MCPResource(r, self._session) for r in resources_result.resources]
            logger.info(f"MCP '{self.name}': discovered {len(self._resources)} resources")
        except Exception as e:
            logger.debug(f"MCP '{self.name}': no resources available: {e}")

        self._connected = True

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport (local subprocess)."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        # Create the stdio client connection
        self._client_cm = stdio_client(server_params)
        streams = await self._client_cm.__aenter__()
        read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def _connect_sse(self) -> None:
        """Connect via SSE transport (remote server)."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._client_cm = sse_client(self.url, headers=self.headers)
        streams = await self._client_cm.__aenter__()
        read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def _connect_streamable_http(self) -> None:
        """Connect via Streamable HTTP transport."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._client_cm = streamablehttp_client(self.url, headers=self.headers)
        streams = await self._client_cm.__aenter__()
        read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._session:
            await self._session.__aexit__(None, None, None)
        if hasattr(self, "_client_cm") and self._client_cm:
            await self._client_cm.__aexit__(None, None, None)
        self._connected = False
        logger.info(f"MCP '{self.name}': disconnected")

    def get_agentx_tools(self) -> list[MCPTool]:
        """Get all MCP tools as AgentX-compatible tools."""
        return self._tools

    def get_resources(self) -> list[MCPResource]:
        """Get all MCP resources."""
        return self._resources

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]


class MCPManager:
    """
    Manage multiple MCP server connections.

    Usage:
        manager = MCPManager()

        # Add MCP servers
        manager.add_server(MCPConnection(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ))
        manager.add_server(MCPConnection(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "..."},
        ))

        # Connect all
        await manager.connect_all()

        # Get all tools from all servers
        tools = manager.get_all_tools()

        # Attach to an agent
        agent = SimpleAgent(config=..., tools=tools)
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}

    def add_server(self, connection: MCPConnection) -> MCPManager:
        """Add an MCP server connection."""
        self._connections[connection.name] = connection
        return self

    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPManager:
        """Shortcut to add a stdio MCP server."""
        return self.add_server(MCPConnection(
            name=name, command=command, args=args or [], env=env, transport="stdio",
        ))

    def add_sse_server(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> MCPManager:
        """Shortcut to add an SSE MCP server."""
        return self.add_server(MCPConnection(
            name=name, url=url, headers=headers, transport="sse",
        ))

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all registered MCP servers."""
        results = {}
        tasks = []
        for name, conn in self._connections.items():
            tasks.append((name, conn.connect()))

        for name, task in tasks:
            try:
                await task
                results[name] = True
                logger.info(f"MCP connected: {name} ({len(conn.get_agentx_tools())} tools)")
            except Exception as e:
                results[name] = False
                logger.error(f"MCP connection failed: {name}: {e}")

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for conn in self._connections.values():
            try:
                await conn.disconnect()
            except Exception as e:
                logger.error(f"MCP disconnect error for '{conn.name}': {e}")

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all connected MCP servers."""
        tools = []
        for conn in self._connections.values():
            if conn.is_connected:
                tools.extend(conn.get_agentx_tools())
        return tools

    def get_tools_from(self, server_name: str) -> list[MCPTool]:
        """Get tools from a specific MCP server."""
        conn = self._connections.get(server_name)
        if conn and conn.is_connected:
            return conn.get_agentx_tools()
        return []

    def get_all_resources(self) -> list[MCPResource]:
        """Get all resources from all connected servers."""
        resources = []
        for conn in self._connections.values():
            if conn.is_connected:
                resources.extend(conn.get_resources())
        return resources

    def get_connection(self, name: str) -> MCPConnection | None:
        return self._connections.get(name)

    @property
    def connected_servers(self) -> list[str]:
        return [name for name, conn in self._connections.items() if conn.is_connected]

    def summary(self) -> dict[str, Any]:
        return {
            "total_servers": len(self._connections),
            "connected": len(self.connected_servers),
            "total_tools": len(self.get_all_tools()),
            "total_resources": len(self.get_all_resources()),
            "servers": {
                name: {
                    "connected": conn.is_connected,
                    "transport": conn.transport,
                    "tools": conn.tool_names,
                }
                for name, conn in self._connections.items()
            },
        }
