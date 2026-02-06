"""Bridge to integrate MCP tools with JARVIS ToolRegistry."""

from __future__ import annotations

import logging
from typing import Any

from tools.base import BaseTool, ToolResult
from core.mcp.manager import get_mcp_manager

log = logging.getLogger("jarvis.mcp.bridge")


class MCPToolBridge(BaseTool):
    """Bridge wrapper that makes MCP tools compatible with JARVIS ToolRegistry."""

    def __init__(
        self,
        name: str,
        description: str,
        server_name: str,
        input_schema: dict[str, Any],
    ):
        super().__init__()
        self._name = name
        self._description = description
        self._server_name = server_name
        self._input_schema = input_schema

    @property
    def name(self) -> str:
        return f"mcp_{self._server_name}_{self._name}"

    @property
    def description(self) -> str:
        return f"[{self._server_name}] {self._description}"

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON schema for LLM function calling."""
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

        # Map MCP input schema to OpenAI function schema
        if self._input_schema:
            if "properties" in self._input_schema:
                schema["parameters"]["properties"] = self._input_schema["properties"]
            if "required" in self._input_schema:
                schema["parameters"]["required"] = self._input_schema["required"]

        return schema

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool."""
        manager = get_mcp_manager()

        try:
            result = await manager.call_tool(
                server_name=self._server_name,
                tool_name=self._name,
                arguments=kwargs,
            )

            # Extract content from MCP result
            content_parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    content_parts.append(content.text)
                elif hasattr(content, "data"):
                    content_parts.append(str(content.data))

            output = "\n".join(content_parts) if content_parts else ""

            return ToolResult(
                success=True,
                data=output,
                error=None,
            )

        except Exception as e:
            log.error(f"Error executing MCP tool '{self.name}': {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
            )


class MCPBridgeManager:
    """Manages the bridge between MCP servers and JARVIS ToolRegistry."""

    def __init__(self):
        self.bridged_tools: list[MCPToolBridge] = []

    async def discover_and_bridge(self) -> list[MCPToolBridge]:
        """Discover all MCP tools and create bridge wrappers."""
        manager = get_mcp_manager()
        self.bridged_tools = []

        for server_name, client in manager.clients.items():
            for tool_info in client.tools:
                bridge = MCPToolBridge(
                    name=tool_info["name"],
                    description=tool_info["description"],
                    server_name=server_name,
                    input_schema=tool_info.get("input_schema", {}),
                )
                self.bridged_tools.append(bridge)
                log.debug(f"Bridged MCP tool: {bridge.name}")

        log.info(f"Created {len(self.bridged_tools)} MCP tool bridges")
        return self.bridged_tools

    def get_bridged_tools(self) -> list[MCPToolBridge]:
        """Get all bridged MCP tools."""
        return self.bridged_tools

    def register_with_registry(self, registry: Any) -> None:
        """Register all bridged tools with a ToolRegistry."""
        for tool in self.bridged_tools:
            registry.register(tool)
            log.debug(f"Registered MCP tool in registry: {tool.name}")

        log.info(f"Registered {len(self.bridged_tools)} MCP tools with registry")


# Global bridge manager instance
_bridge_manager: MCPBridgeManager | None = None


def get_bridge_manager() -> MCPBridgeManager:
    """Get or create the global bridge manager."""
    global _bridge_manager
    if _bridge_manager is None:
        _bridge_manager = MCPBridgeManager()
    return _bridge_manager


async def initialize_mcp_tools(registry: Any) -> dict[str, Any]:
    """
    Initialize MCP integration:
    1. Connect to all configured MCP servers
    2. Discover tools
    3. Bridge them to JARVIS ToolRegistry

    Returns connection results by server name.
    """
    log.info("Initializing MCP integration...")

    # Connect to all servers
    manager = get_mcp_manager()
    results = await manager.connect_all()

    # Count successful connections
    connected = sum(1 for success in results.values() if success)
    log.info(f"Connected to {connected}/{len(results)} MCP servers")

    # Bridge tools
    bridge = get_bridge_manager()
    await bridge.discover_and_bridge()
    bridge.register_with_registry(registry)

    return results


async def shutdown_mcp() -> None:
    """Shutdown all MCP connections."""
    manager = get_mcp_manager()
    await manager.disconnect_all()
    log.info("MCP connections shut down")
