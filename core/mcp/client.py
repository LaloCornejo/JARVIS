"""MCP client for connecting to MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.types import CallToolResult, TextContent

log = logging.getLogger("jarvis.mcp.client")


class MCPClient:
    """Client for connecting to an MCP server."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._tools: list[dict[str, Any]] = []

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
            )

            stdio_transport = await self._exit_stack.enter_async_context(
                asyncio.create_subprocess_exec(
                    server_params.command,
                    *server_params.args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**dict(subprocess.os.environ), **(server_params.env or {})},
                )
            )

            stdio = await self._exit_stack.enter_async_context(stdio_transport)

            read_stream = stdio.stdout
            write_stream = stdio.stdin

            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await self.session.initialize()

            # Discover available tools
            tools_response = await self.session.list_tools()
            self._tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_response.tools
            ]

            log.info(f"Connected to MCP server '{self.name}' with {len(self._tools)} tools")
            return True

        except Exception as e:
            log.error(f"Failed to connect to MCP server '{self.name}': {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        await self._exit_stack.aclose()
        self.session = None
        self._tools = []
        log.info(f"Disconnected from MCP server '{self.name}'")

    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self.session is not None

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get list of available tools."""
        return self._tools

    def get_tool(self, name: str) -> dict[str, Any] | None:
        """Get a specific tool by name."""
        for tool in self._tools:
            if tool["name"] == name:
                return tool
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call a tool on the MCP server."""
        if not self.session:
            raise RuntimeError(f"Not connected to MCP server '{self.name}'")

        try:
            result = await self.session.call_tool(name, arguments)
            return result
        except Exception as e:
            log.error(f"Error calling tool '{name}' on MCP server '{self.name}': {e}")
            raise

    def __repr__(self) -> str:
        return f"MCPClient(name='{self.name}', connected={self.is_connected}, tools={len(self._tools)})"
