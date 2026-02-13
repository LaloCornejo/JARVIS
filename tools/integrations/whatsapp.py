"""WhatsApp tools using Baileys service."""

from __future__ import annotations

import logging
from typing import Any

from core.whatsapp_bailey_client import whatsapp_bailey_client
from tools.base import BaseTool, ToolResult

log = logging.getLogger(__name__)


class WhatsAppSendMessageTool(BaseTool):
    name = "whatsapp_send_message"
    description = "Send a text message to a WhatsApp number via the Baileys service"
    parameters = {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Recipient phone number in international format (e.g., '1234567890' without + or spaces)",
            },
            "message": {
                "type": "string",
                "description": "Message text to send",
            },
        },
        "required": ["phone_number", "message"],
    }

    async def execute(self, phone_number: str, message: str) -> ToolResult:
        result = await whatsapp_bailey_client.send_message(phone_number, message)
        if result.get("success"):
            return ToolResult(
                success=True,
                data={
                    "sent": True,
                    "recipient": phone_number,
                    "message_id": result.get("message_id"),
                },
            )
        return ToolResult(
            success=False,
            data=None,
            error=result.get("error", "Failed to send message"),
        )


class WhatsAppSendMediaTool(BaseTool):
    name = "whatsapp_send_media"
    description = "Send media (image, document, audio, video) to a WhatsApp number"
    parameters = {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Recipient phone number in international format",
            },
            "media_type": {
                "type": "string",
                "description": "Type of media to send",
                "enum": ["image", "document", "audio", "video"],
            },
            "media_url": {
                "type": "string",
                "description": "Public URL of the media file",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption for the media",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename for documents",
            },
        },
        "required": ["phone_number", "media_type", "media_url"],
    }

    async def execute(
        self,
        phone_number: str,
        media_type: str,
        media_url: str,
        caption: str | None = None,
        filename: str | None = None,
    ) -> ToolResult:
        result = await whatsapp_bailey_client.send_media(
            to=phone_number,
            media_type=media_type,
            url=media_url,
            caption=caption,
            filename=filename,
        )
        if result.get("success"):
            return ToolResult(
                success=True,
                data={
                    "sent": True,
                    "recipient": phone_number,
                    "media_type": media_type,
                },
            )
        return ToolResult(
            success=False,
            data=None,
            error=result.get("error", "Failed to send media"),
        )


class WhatsAppCheckStatusTool(BaseTool):
    name = "whatsapp_check_status"
    description = "Check if WhatsApp Baileys service is running and connected"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        result = await whatsapp_bailey_client.check_status()
        return ToolResult(
            success=True,
            data={
                "connected": result.get("connected", False),
                "state": result.get("state", "unknown"),
                "user": result.get("user"),
            },
        )


class WhatsAppManualResponseTool(BaseTool):
    name = "whatsapp_manual_response"
    description = "Manually respond to a WhatsApp message (for non-owner chats that were logged but not auto-responded)"
    parameters = {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Recipient phone number in international format (e.g., '1234567890' without + or spaces)",
            },
            "message": {
                "type": "string",
                "description": "Message text to send",
            },
        },
        "required": ["phone_number", "message"],
    }

    async def execute(self, phone_number: str, message: str) -> ToolResult:
        result = await whatsapp_bailey_client.respond_manually(phone_number, message)
        if result.get("success"):
            return ToolResult(
                success=True,
                data={
                    "sent": True,
                    "recipient": phone_number,
                    "message_id": result.get("message_id"),
                },
            )
        return ToolResult(
            success=False,
            data=None,
            error=result.get("error", "Failed to send message"),
        )


class WhatsAppGetPendingMessagesTool(BaseTool):
    name = "whatsapp_get_pending_messages"
    description = "Get list of pending WhatsApp messages from non-owner chats that were logged but not responded to"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages to return (default: 50)",
                "default": 50,
            },
        },
        "required": [],
    }

    async def execute(self, limit: int = 50) -> ToolResult:
        messages = whatsapp_bailey_client.get_pending_messages(limit)
        return ToolResult(
            success=True,
            data={
                "count": len(messages),
                "messages": messages,
            },
        )
