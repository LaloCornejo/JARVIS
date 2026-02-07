"""Telegram Bot integration for JARVIS.

This module provides tools for interacting with Telegram Bot API:
- Send and receive messages
- Send photos and documents
- Get chat information
- Reply to and manage messages

Setup:
1. Create a bot via @BotFather on Telegram
2. Get your bot token
3. Set TELEGRAM_BOT_TOKEN environment variable
4. Start chatting with your bot
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult

log = logging.getLogger(__name__)


class TelegramClient:
    """Client for Telegram Bot API."""

    API_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_url(self, method: str) -> str:
        return f"{self.API_URL.format(token=self.token)}/{method}"

    async def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any] | None:
        if not self.token:
            return None

        client = await self._get_client()
        url = self._get_url(endpoint)

        try:
            if method == "GET":
                response = await client.get(url, params=kwargs.get("params"))
            elif method == "POST":
                data = kwargs.get("json") or kwargs.get("data")
                files = kwargs.get("files")
                response = await client.post(url, json=data, data=kwargs.get("data"), files=files)
            else:
                return None

            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    return result.get("result")
                log.warning(f"Telegram API error: {result.get('description')}")
                return None
            else:
                log.warning(f"Telegram HTTP error: {response.status_code}")
                return None
        except Exception as e:
            log.error(f"Telegram request error: {e}")
            return None

    async def get_me(self) -> dict[str, Any] | None:
        """Get information about the bot."""
        return await self._request("GET", "getMe")

    async def send_message(
        self,
        chat_id: str | int,
        text: str,
        parse_mode: str = "HTML",
        reply_to_message_id: int | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any] | None:
        """Send a text message to a chat."""
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        return await self._request("POST", "sendMessage", json=payload)

    async def get_updates(
        self,
        offset: int | None = None,
        limit: int = 100,
        timeout: int = 0,
    ) -> list[dict[str, Any]]:
        """Get updates (messages) from Telegram."""
        params = {"limit": limit, "timeout": timeout}
        if offset is not None:
            params["offset"] = offset

        result = await self._request("GET", "getUpdates", params=params)
        if isinstance(result, list):
            return result
        return []

    async def get_chat(self, chat_id: str | int) -> dict[str, Any] | None:
        """Get information about a chat."""
        return await self._request("POST", "getChat", json={"chat_id": chat_id})

    async def send_photo(
        self,
        chat_id: str | int,
        photo_path: str,
        caption: str | None = None,
        parse_mode: str = "HTML",
    ) -> dict[str, Any] | None:
        """Send a photo to a chat."""
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": (os.path.basename(photo_path), f, "image/jpeg")}
                data = {"chat_id": str(chat_id), "parse_mode": parse_mode}
                if caption:
                    data["caption"] = caption

                client = await self._get_client()
                url = self._get_url("sendPhoto")
                response = await client.post(url, data=data, files=files)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        return result.get("result")
                log.warning(f"Failed to send photo: {response.status_code}")
                return None
        except FileNotFoundError:
            log.error(f"Photo file not found: {photo_path}")
            return None
        except Exception as e:
            log.error(f"Error sending photo: {e}")
            return None

    async def send_document(
        self,
        chat_id: str | int,
        document_path: str | None = None,
        document_data: bytes | None = None,
        filename: str = "document.pdf",
        caption: str | None = None,
    ) -> dict[str, Any] | None:
        """Send a document/file to a chat."""
        try:
            # Handle both file path and file data cases
            if document_data is not None:
                # Send from memory
                import io

                file_obj = io.BytesIO(document_data)
                file_obj.seek(0)
                files = {"document": (filename, file_obj)}
            elif document_path is not None:
                # Send from file path
                file_obj = open(document_path, "rb")
                files = {"document": (os.path.basename(document_path), file_obj)}
            else:
                log.error("Either document_path or document_data must be provided")
                return None

            try:
                data = {"chat_id": str(chat_id)}
                if caption:
                    data["caption"] = caption

                client = await self._get_client()
                url = self._get_url("sendDocument")
                response = await client.post(url, data=data, files=files)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        return result.get("result")
                log.warning(f"Failed to send document: {response.status_code}")
                return None
            finally:
                # Close the file object if it was opened from a path
                if document_path is not None and hasattr(file_obj, "close"):
                    file_obj.close()
        except FileNotFoundError:
            log.error(f"Document file not found: {document_path}")
            return None
        except Exception as e:
            log.error(f"Error sending document: {e}")
            return None

    async def send_voice(
        self,
        chat_id: str | int,
        voice_data: bytes | None = None,
        voice_path: str | None = None,
        filename: str = "voice.ogg",
        caption: str | None = None,
        duration: int | None = None,
    ) -> dict[str, Any] | None:
        """Send a voice message to a chat."""
        try:
            # Handle both file path and file data cases
            if voice_data is not None:
                # Send from memory
                import io

                file_obj = io.BytesIO(voice_data)
                file_obj.seek(0)
                files = {"voice": (filename, file_obj)}
            elif voice_path is not None:
                # Send from file path
                file_obj = open(voice_path, "rb")
                files = {"voice": (os.path.basename(voice_path), file_obj)}
            else:
                log.error("Either voice_path or voice_data must be provided")
                return None

            try:
                data = {"chat_id": str(chat_id)}
                if caption:
                    data["caption"] = caption
                if duration:
                    data["duration"] = duration

                client = await self._get_client()
                url = self._get_url("sendVoice")
                response = await client.post(url, data=data, files=files)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        return result.get("result")
                log.warning(f"Failed to send voice: {response.status_code}")
                return None
            finally:
                # Close the file object if it was opened from a path
                if voice_path is not None and hasattr(file_obj, "close"):
                    file_obj.close()
        except FileNotFoundError:
            log.error(f"Voice file not found: {voice_path}")
            return None
        except Exception as e:
            log.error(f"Error sending voice: {e}")
            return None

            try:
                data = {"chat_id": str(chat_id)}
                if caption:
                    data["caption"] = caption
                if duration:
                    data["duration"] = duration

                client = await self._get_client()
                url = self._get_url("sendVoice")
                response = await client.post(url, data=data, files=files)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        return result.get("result")
                log.warning(f"Failed to send voice: {response.status_code}")
                return None
            finally:
                # Close the file object if it was opened from a path
                if voice_path is not None and file_obj:
                    file_obj.close()
        except FileNotFoundError:
            log.error(f"Voice file not found: {voice_path}")
            return None
        except Exception as e:
            log.error(f"Error sending voice: {e}")
            return None

            try:
                data = {"chat_id": str(chat_id)}
                if caption:
                    data["caption"] = caption

                client = await self._get_client()
                url = self._get_url("sendDocument")
                response = await client.post(url, data=data, files=files)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        return result.get("result")
                log.warning(f"Failed to send document: {response.status_code}")
                return None
            finally:
                # Close the file object if it was opened from a path
                if document_path is not None and file_obj:
                    file_obj.close()
        except FileNotFoundError:
            log.error(f"Document file not found: {document_path}")
            return None
        except Exception as e:
            log.error(f"Error sending document: {e}")
            return None

    async def edit_message_text(
        self,
        chat_id: str | int,
        message_id: int,
        text: str,
        parse_mode: str = "HTML",
    ) -> dict[str, Any] | None:
        """Edit a message text."""
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        return await self._request("POST", "editMessageText", json=payload)

    async def delete_message(self, chat_id: str | int, message_id: int) -> bool:
        """Delete a message."""
        result = await self._request(
            "POST", "deleteMessage", json={"chat_id": chat_id, "message_id": message_id}
        )
        return result is not None

    async def pin_message(
        self,
        chat_id: str | int,
        message_id: int,
        disable_notification: bool = False,
    ) -> dict[str, Any] | None:
        """Pin a message in a chat."""
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "disable_notification": disable_notification,
        }
        return await self._request("POST", "pinChatMessage", json=payload)

    async def get_chat_member_count(self, chat_id: str | int) -> int | None:
        """Get the number of members in a chat."""
        result = await self._request("POST", "getChatMemberCount", json={"chat_id": chat_id})
        if isinstance(result, int):
            return result
        return None


_telegram_client: TelegramClient | None = None


def get_telegram_client() -> TelegramClient:
    global _telegram_client
    if _telegram_client is None:
        _telegram_client = TelegramClient()
    return _telegram_client


class TelegramSendMessageTool(BaseTool):
    """Send a text message to a Telegram chat."""

    name = "telegram_send_message"
    description = "Send a text message to a Telegram chat or user"
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID (can be a number or username with @)",
            },
            "text": {
                "type": "string",
                "description": "Message text to send (supports HTML formatting)",
            },
            "parse_mode": {
                "type": "string",
                "description": "Message format: HTML, Markdown, or MarkdownV2",
                "enum": ["HTML", "Markdown", "MarkdownV2"],
                "default": "HTML",
            },
            "reply_to_message_id": {
                "type": "integer",
                "description": "Message ID to reply to (optional)",
            },
        },
        "required": ["chat_id", "text"],
    }

    async def execute(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "HTML",
        reply_to_message_id: int | None = None,
    ) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            result = await client.send_message(chat_id, text, parse_mode, reply_to_message_id)
            if result:
                return ToolResult(
                    success=True,
                    data={
                        "message_id": result.get("message_id"),
                        "chat_id": result.get("chat", {}).get("id"),
                        "sent": True,
                        "timestamp": result.get("date"),
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to send message")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramReceiveMessagesTool(BaseTool):
    """Receive recent messages from Telegram bot."""

    name = "telegram_receive_messages"
    description = "Get recent messages sent to your Telegram bot"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages to retrieve (1-100)",
                "default": 20,
            },
            "offset": {
                "type": "integer",
                "description": "Offset for pagination (optional)",
            },
        },
        "required": [],
    }

    async def execute(self, limit: int = 20, offset: int | None = None) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            updates = await client.get_updates(offset=offset, limit=min(limit, 100))
            messages = []

            for update in updates:
                message = update.get("message") or update.get("edited_message")
                if message:
                    chat = message.get("chat", {})
                    from_user = message.get("from", {})

                    formatted = {
                        "update_id": update.get("update_id"),
                        "message_id": message.get("message_id"),
                        "chat_id": chat.get("id"),
                        "chat_type": chat.get("type"),
                        "chat_title": chat.get("title") or chat.get("username"),
                        "from_user": {
                            "id": from_user.get("id"),
                            "username": from_user.get("username"),
                            "first_name": from_user.get("first_name"),
                        },
                        "text": message.get("text", ""),
                        "timestamp": datetime.fromtimestamp(message.get("date", 0)).isoformat(),
                    }
                    messages.append(formatted)

            return ToolResult(
                success=True,
                data={
                    "messages": messages,
                    "count": len(messages),
                    "next_offset": updates[-1]["update_id"] + 1 if updates else offset,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramGetChatInfoTool(BaseTool):
    """Get information about a Telegram chat."""

    name = "telegram_get_chat_info"
    description = "Get detailed information about a Telegram chat or group"
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID or username",
            },
        },
        "required": ["chat_id"],
    }

    async def execute(self, chat_id: str) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            chat_info = await client.get_chat(chat_id)
            if chat_info:
                return ToolResult(
                    success=True,
                    data={
                        "id": chat_info.get("id"),
                        "type": chat_info.get("type"),
                        "title": chat_info.get("title"),
                        "username": chat_info.get("username"),
                        "first_name": chat_info.get("first_name"),
                        "last_name": chat_info.get("last_name"),
                        "description": chat_info.get("description"),
                        "invite_link": chat_info.get("invite_link"),
                        "member_count": chat_info.get("member_count"),
                    },
                )
            return ToolResult(success=False, data=None, error="Chat not found")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramSendPhotoTool(BaseTool):
    """Send a photo to a Telegram chat."""

    name = "telegram_send_photo"
    description = "Send a photo/image file to a Telegram chat"
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID",
            },
            "photo_path": {
                "type": "string",
                "description": "Path to the photo file to send",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption for the photo",
            },
        },
        "required": ["chat_id", "photo_path"],
    }

    async def execute(
        self, chat_id: str, photo_path: str, caption: str | None = None
    ) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            result = await client.send_photo(chat_id, photo_path, caption)
            if result:
                return ToolResult(
                    success=True,
                    data={
                        "message_id": result.get("message_id"),
                        "photo_sent": True,
                        "chat_id": result.get("chat", {}).get("id"),
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to send photo")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramSendDocumentTool(BaseTool):
    """Send a document/file to a Telegram chat."""

    name = "telegram_send_document"
    description = "Send a document or file to a Telegram chat"
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID",
            },
            "document_path": {
                "type": "string",
                "description": "Path to the document file to send",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption for the document",
            },
        },
        "required": ["chat_id", "document_path"],
    }

    async def execute(
        self, chat_id: str, document_path: str, caption: str | None = None
    ) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            result = await client.send_document(chat_id, document_path, caption)
            if result:
                return ToolResult(
                    success=True,
                    data={
                        "message_id": result.get("message_id"),
                        "document_sent": True,
                        "chat_id": result.get("chat", {}).get("id"),
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to send document")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramEditMessageTool(BaseTool):
    """Edit a previously sent message."""

    name = "telegram_edit_message"
    description = "Edit the text of a message you previously sent"
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID where the message was sent",
            },
            "message_id": {
                "type": "integer",
                "description": "ID of the message to edit",
            },
            "text": {
                "type": "string",
                "description": "New text content",
            },
            "parse_mode": {
                "type": "string",
                "description": "Message format",
                "enum": ["HTML", "Markdown", "MarkdownV2"],
                "default": "HTML",
            },
        },
        "required": ["chat_id", "message_id", "text"],
    }

    async def execute(
        self, chat_id: str, message_id: int, text: str, parse_mode: str = "HTML"
    ) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            result = await client.edit_message_text(chat_id, message_id, text, parse_mode)
            if result:
                return ToolResult(
                    success=True,
                    data={
                        "message_id": result.get("message_id"),
                        "edited": True,
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to edit message")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramDeleteMessageTool(BaseTool):
    """Delete a message from a chat."""

    name = "telegram_delete_message"
    description = "Delete a message from a Telegram chat"
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID",
            },
            "message_id": {
                "type": "integer",
                "description": "ID of the message to delete",
            },
        },
        "required": ["chat_id", "message_id"],
    }

    async def execute(self, chat_id: str, message_id: int) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            deleted = await client.delete_message(chat_id, message_id)
            if deleted:
                return ToolResult(
                    success=True,
                    data={"deleted": True, "message_id": message_id},
                )
            return ToolResult(success=False, data=None, error="Failed to delete message")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramPinMessageTool(BaseTool):
    """Pin a message in a chat."""

    name = "telegram_pin_message"
    description = "Pin an important message in a Telegram chat or group"
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID",
            },
            "message_id": {
                "type": "integer",
                "description": "ID of the message to pin",
            },
            "disable_notification": {
                "type": "boolean",
                "description": "Send notification to all members",
                "default": False,
            },
        },
        "required": ["chat_id", "message_id"],
    }

    async def execute(
        self, chat_id: str, message_id: int, disable_notification: bool = False
    ) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            result = await client.pin_message(chat_id, message_id, disable_notification)
            if result:
                return ToolResult(
                    success=True,
                    data={"pinned": True, "message_id": message_id},
                )
            return ToolResult(success=False, data=None, error="Failed to pin message")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class TelegramGetBotInfoTool(BaseTool):
    """Get information about your Telegram bot."""

    name = "telegram_get_bot_info"
    description = "Get information about your Telegram bot (name, username, etc.)"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        client = get_telegram_client()
        if not client.token:
            return ToolResult(
                success=False,
                data=None,
                error="Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
            )

        try:
            bot_info = await client.get_me()
            if bot_info:
                return ToolResult(
                    success=True,
                    data={
                        "id": bot_info.get("id"),
                        "first_name": bot_info.get("first_name"),
                        "username": bot_info.get("username"),
                        "can_join_groups": bot_info.get("can_join_groups"),
                        "can_read_all_group_messages": bot_info.get("can_read_all_group_messages"),
                        "supports_inline_queries": bot_info.get("supports_inline_queries"),
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to get bot info")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
