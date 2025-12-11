from __future__ import annotations

import os
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult


class DiscordClient:
    API_URL = "https://discord.com/api/v10"

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("DISCORD_TOKEN", "")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> dict[str, Any] | list | None:
        if not self.token:
            return None

        client = await self._get_client()
        headers = {"Authorization": f"Bot {self.token}"}

        response = await client.request(
            method, f"{self.API_URL}{endpoint}", headers=headers, **kwargs
        )

        if response.status_code in (200, 201, 204):
            if response.content:
                return response.json()
            return {}
        return None

    async def get_guilds(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/users/@me/guilds")
        if isinstance(result, list):
            return result
        return []

    async def get_channels(self, guild_id: str) -> list[dict[str, Any]]:
        result = await self._request("GET", f"/guilds/{guild_id}/channels")
        if isinstance(result, list):
            return result
        return []

    async def get_messages(self, channel_id: str, limit: int = 50) -> list[dict[str, Any]]:
        result = await self._request(
            "GET", f"/channels/{channel_id}/messages", params={"limit": limit}
        )
        if isinstance(result, list):
            return result
        return []

    async def send_message(
        self, channel_id: str, content: str, embed: dict | None = None
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"content": content}
        if embed:
            payload["embeds"] = [embed]
        result = await self._request("POST", f"/channels/{channel_id}/messages", json=payload)
        if isinstance(result, dict):
            return result
        return None

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        result = await self._request("GET", f"/users/{user_id}")
        if isinstance(result, dict):
            return result
        return None

    async def create_dm(self, user_id: str) -> dict[str, Any] | None:
        result = await self._request("POST", "/users/@me/channels", json={"recipient_id": user_id})
        if isinstance(result, dict):
            return result
        return None

    async def send_dm(self, user_id: str, content: str) -> dict[str, Any] | None:
        dm = await self.create_dm(user_id)
        if dm:
            return await self.send_message(dm["id"], content)
        return None


_discord_client: DiscordClient | None = None


def get_discord_client() -> DiscordClient:
    global _discord_client
    if _discord_client is None:
        _discord_client = DiscordClient()
    return _discord_client


class DiscordSendMessageTool(BaseTool):
    name = "discord_send"
    description = "Send a message to a Discord channel"
    parameters = {
        "type": "object",
        "properties": {
            "channel_id": {"type": "string", "description": "Discord channel ID"},
            "content": {"type": "string", "description": "Message content"},
        },
        "required": ["channel_id", "content"],
    }

    async def execute(self, channel_id: str, content: str) -> ToolResult:
        client = get_discord_client()
        if not client.token:
            return ToolResult(success=False, data=None, error="Discord token not configured")

        try:
            result = await client.send_message(channel_id, content)
            if result:
                return ToolResult(
                    success=True,
                    data={"message_id": result.get("id"), "sent": True},
                )
            return ToolResult(success=False, data=None, error="Failed to send")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DiscordReadMessagesTool(BaseTool):
    name = "discord_read"
    description = "Read recent messages from a Discord channel"
    parameters = {
        "type": "object",
        "properties": {
            "channel_id": {"type": "string", "description": "Discord channel ID"},
            "limit": {"type": "integer", "description": "Number of messages (max 50)"},
        },
        "required": ["channel_id"],
    }

    async def execute(self, channel_id: str, limit: int = 20) -> ToolResult:
        client = get_discord_client()
        if not client.token:
            return ToolResult(success=False, data=None, error="Discord token not configured")

        try:
            messages = await client.get_messages(channel_id, min(limit, 50))
            formatted = [
                {
                    "id": m.get("id"),
                    "author": m.get("author", {}).get("username", "Unknown"),
                    "content": m.get("content", ""),
                    "timestamp": m.get("timestamp", ""),
                }
                for m in messages
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DiscordListChannelsTool(BaseTool):
    name = "discord_list_channels"
    description = "List channels in a Discord server"
    parameters = {
        "type": "object",
        "properties": {
            "guild_id": {"type": "string", "description": "Discord server (guild) ID"},
        },
        "required": ["guild_id"],
    }

    async def execute(self, guild_id: str) -> ToolResult:
        client = get_discord_client()
        if not client.token:
            return ToolResult(success=False, data=None, error="Discord token not configured")

        try:
            channels = await client.get_channels(guild_id)
            text_channels = [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "type": c.get("type"),
                }
                for c in channels
                if c.get("type") == 0
            ]
            return ToolResult(success=True, data=text_channels)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DiscordListServersTool(BaseTool):
    name = "discord_list_servers"
    description = "List Discord servers the bot is in"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_discord_client()
        if not client.token:
            return ToolResult(success=False, data=None, error="Discord token not configured")

        try:
            guilds = await client.get_guilds()
            formatted = [{"id": g.get("id"), "name": g.get("name")} for g in guilds]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DiscordSendDMTool(BaseTool):
    name = "discord_dm"
    description = "Send a direct message to a Discord user"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "Discord user ID"},
            "content": {"type": "string", "description": "Message content"},
        },
        "required": ["user_id", "content"],
    }

    async def execute(self, user_id: str, content: str) -> ToolResult:
        client = get_discord_client()
        if not client.token:
            return ToolResult(success=False, data=None, error="Discord token not configured")

        try:
            result = await client.send_dm(user_id, content)
            if result:
                return ToolResult(success=True, data={"sent": True})
            return ToolResult(success=False, data=None, error="Failed to send DM")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
