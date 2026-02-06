"""MCP manager for handling multiple MCP servers."""

from __future__ import annotations

import logging
from typing import Any

from core.mcp.client import MCPClient
from core.config import Config

log = logging.getLogger("jarvis.mcp.manager")


class MCPManager:
    """Manages multiple MCP server connections."""

    # Default MCP server configurations
    DEFAULT_SERVERS: dict[str, dict[str, Any]] = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/"],
            "env": None,
        },
        "fetch": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-fetch"],
            "env": None,
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": ""},
        },
        "sequentialthinking": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequentialthinking"],
            "env": None,
        },
        "memory": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "env": None,
        },
        "sqlite": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sqlite", ":memory:"],
            "env": None,
        },
        "redis": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-redis"],
            "env": {"REDIS_URL": "redis://localhost:6379"},
        },
        "playwright": {
            "command": "npx",
            "args": ["-y", "@anthropic-ai/mcp-playwright"],
            "env": None,
        },
    }

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.clients: dict[str, MCPClient] = {}
        self._config_key = "mcp.servers"

    def _get_server_configs(self) -> dict[str, dict[str, Any]]:
        """Get server configs from settings or use defaults."""
        user_configs = self.config.get(self._config_key, {})

        # Merge with defaults, user config takes precedence
        configs = self.DEFAULT_SERVERS.copy()
        for name, config in user_configs.items():
            if name in configs:
                configs[name].update(config)
            else:
                configs[name] = config

        return configs

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all configured MCP servers."""
        configs = self._get_server_configs()
        results = {}

        for name, config in configs.items():
            if not config.get("enabled", True):
                log.info(f"Skipping disabled MCP server: {name}")
                results[name] = False
                continue

            client = MCPClient(
                name=name,
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
            )

            success = await client.connect()
            if success:
                self.clients[name] = client
                results[name] = True
            else:
                results[name] = False

        return results

    async def connect_server(self, name: str) -> bool:
        """Connect to a specific MCP server."""
        if name in self.clients:
            log.warning(f"MCP server '{name}' is already connected")
            return True

        configs = self._get_server_configs()
        if name not in configs:
            log.error(f"Unknown MCP server: {name}")
            return False

        config = configs[name]
        client = MCPClient(
            name=name,
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
        )

        success = await client.connect()
        if success:
            self.clients[name] = client

        return success

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name, client in list(self.clients.items()):
            await client.disconnect()
        self.clients.clear()

    async def disconnect_server(self, name: str) -> None:
        """Disconnect from a specific MCP server."""
        if name in self.clients:
            await self.clients[name].disconnect()
            del self.clients[name]

    def get_client(self, name: str) -> MCPClient | None:
        """Get a specific MCP client by name."""
        return self.clients.get(name)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools from all connected servers."""
        tools = []
        for server_name, client in self.clients.items():
            for tool in client.tools:
                tools.append(
                    {
                        **tool,
                        "server": server_name,
                        "mcp_tool": True,
                    }
                )
        return tools

    def get_tool(self, name: str) -> dict[str, Any] | None:
        """Get a specific tool by name across all servers."""
        for server_name, client in self.clients.items():
            tool = client.get_tool(name)
            if tool:
                return {
                    **tool,
                    "server": server_name,
                    "mcp_tool": True,
                }
        return None

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on a specific server."""
        client = self.get_client(server_name)
        if not client:
            raise ValueError(f"MCP server '{server_name}' is not connected")

        return await client.call_tool(tool_name, arguments)

    @property
    def connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return list(self.clients.keys())

    def __repr__(self) -> str:
        return f"MCPManager(servers={self.connected_servers})"


# Global manager instance
_mcp_manager: MCPManager | None = None


def get_mcp_manager(config: Config | None = None) -> MCPManager:
    """Get or create the global MCP manager."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager(config)
    return _mcp_manager
