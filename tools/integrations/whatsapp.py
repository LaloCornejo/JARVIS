"""WhatsApp Business API integration for JARVIS.

This module provides tools for interacting with WhatsApp Business API:
- Send and receive messages
- Send media (photos, documents, audio)
- Get conversation history
- Manage templates

Setup:
1. Create a Meta Developer account at https://developers.facebook.com
2. Create a WhatsApp Business app
3. Get your Phone Number ID and Access Token
4. Set WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_ACCESS_TOKEN environment variables
5. Verify your phone number for testing

Note: This uses the WhatsApp Cloud API (Business API).
For personal use, consider using WhatsApp Web libraries like:
- whatsapp-web.js (Node.js - requires separate service)
- pywhatkit (unofficial, limited)
- yowsup (unofficial, complex)

This implementation uses the official Cloud API which requires a Meta business account.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult

log = logging.getLogger(__name__)


class WhatsAppClient:
    """Client for WhatsApp Business Cloud API."""

    API_URL = "https://graph.facebook.com/v18.0"

    def __init__(
        self,
        access_token: str | None = None,
        phone_number_id: str | None = None,
    ):
        self.access_token = access_token or os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
        self.phone_number_id = phone_number_id or os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if not self.access_token or not self.phone_number_id:
            log.error("WhatsApp credentials not configured")
            return None

        client = await self._get_client()
        url = f"{self.API_URL}{endpoint}"
        headers = self._get_headers()

        try:
            if method == "GET":
                response = await client.get(url, headers=headers, params=kwargs.get("params"))
            elif method == "POST":
                data = kwargs.get("json")
                response = await client.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                return None

            if response.status_code in (200, 201):
                return response.json()
            else:
                log.warning(f"WhatsApp API error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            log.error(f"WhatsApp request error: {e}")
            return None

    async def send_text_message(
        self,
        to: str,
        text: str,
        preview_url: bool = False,
    ) -> dict[str, Any] | None:
        """Send a text message to a WhatsApp number.

        Args:
            to: Recipient phone number in international format (e.g., "1234567890")
            text: Message text (max 4096 characters)
            preview_url: Whether to show URL previews
        """
        endpoint = f"/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": text, "preview_url": preview_url},
        }

        return await self._request("POST", endpoint, json=payload)

    async def send_template_message(
        self,
        to: str,
        template_name: str,
        language_code: str = "en",
        components: list[dict] | None = None,
    ) -> dict[str, Any] | None:
        """Send a template message (requires pre-approved templates).

        Args:
            to: Recipient phone number
            template_name: Name of the approved template
            language_code: Language code (e.g., "en", "es")
            components: Template components/parameters
        """
        endpoint = f"/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
            },
        }
        if components:
            payload["template"]["components"] = components

        return await self._request("POST", endpoint, json=payload)

    async def send_media_message(
        self,
        to: str,
        media_type: str,  # "image", "document", "audio", "video", "sticker"
        media_id: str | None = None,
        media_url: str | None = None,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any] | None:
        """Send a media message (image, document, audio, video, or sticker).

        Args:
            to: Recipient phone number
            media_type: Type of media (image, document, audio, video, sticker)
            media_id: Media ID from previously uploaded media (either this or media_url required)
            media_url: URL of the media file (either this or media_id required)
            caption: Optional caption for the media
            filename: Optional filename for documents
        """
        if not media_id and not media_url:
            log.error("Either media_id or media_url must be provided")
            return None

        endpoint = f"/{self.phone_number_id}/messages"
        media_obj: dict[str, Any] = {}

        if media_id:
            media_obj["id"] = media_id
        else:
            media_obj["link"] = media_url

        if caption and media_type in ("image", "document", "video"):
            media_obj["caption"] = caption

        if filename and media_type == "document":
            media_obj["filename"] = filename

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": media_type,
            media_type: media_obj,
        }

        return await self._request("POST", endpoint, json=payload)

    async def upload_media(self, file_path: str) -> dict[str, Any] | None:
        """Upload media to WhatsApp servers.

        Args:
            file_path: Path to the local file to upload

        Returns:
            Media ID that can be used in send_media_message
        """
        if not os.path.exists(file_path):
            log.error(f"File not found: {file_path}")
            return None

        endpoint = f"/{self.phone_number_id}/media"
        url = f"{self.API_URL}{endpoint}"

        try:
            client = await self._get_client()
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"

            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, mime_type)}
                data = {"messaging_product": "whatsapp"}
                headers = {"Authorization": f"Bearer {self.access_token}"}

                response = await client.post(url, headers=headers, data=data, files=files)

            if response.status_code in (200, 201):
                return response.json()
            else:
                log.warning(f"Failed to upload media: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            log.error(f"Error uploading media: {e}")
            return None

    async def get_media_url(self, media_id: str) -> str | None:
        """Get the URL for a media file.

        Args:
            media_id: The media ID

        Returns:
            URL to download the media
        """
        endpoint = f"/{media_id}"
        result = await self._request("GET", endpoint)
        if result:
            return result.get("url")
        return None

    async def download_media(self, media_id: str, download_path: str) -> bool:
        """Download media from WhatsApp servers.

        Args:
            media_id: The media ID
            download_path: Path to save the downloaded file

        Returns:
            True if successful
        """
        media_url = await self.get_media_url(media_id)
        if not media_url:
            return False

        try:
            client = await self._get_client()
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = await client.get(media_url, headers=headers)

            if response.status_code == 200:
                os.makedirs(os.path.dirname(download_path) or ".", exist_ok=True)
                with open(download_path, "wb") as f:
                    f.write(response.content)
                return True
            return False
        except Exception as e:
            log.error(f"Error downloading media: {e}")
            return False

    async def mark_message_as_read(self, message_id: str) -> bool:
        """Mark a message as read.

        Args:
            message_id: The message ID to mark as read

        Returns:
            True if successful
        """
        endpoint = f"/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }

        result = await self._request("POST", endpoint, json=payload)
        return result is not None

    async def get_business_profile(self) -> dict[str, Any] | None:
        """Get the WhatsApp business profile information."""
        endpoint = f"/{self.phone_number_id}/whatsapp_business_profile"
        result = await self._request(
            "GET",
            endpoint,
            params={"fields": "about,description,email,websites,profile_picture_url"},
        )
        return result

    async def get_templates(self) -> list[dict[str, Any]]:
        """Get list of message templates."""
        # Requires Business Account ID, not Phone Number ID
        business_id = os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID", "")
        if not business_id:
            log.error("WHATSAPP_BUSINESS_ACCOUNT_ID not configured")
            return []

        endpoint = f"/{business_id}/message_templates"
        result = await self._request("GET", endpoint)
        if isinstance(result, dict):
            return result.get("data", [])
        return []

    async def get_phone_numbers(self) -> list[dict[str, Any]]:
        """Get list of registered phone numbers."""
        business_id = os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID", "")
        if not business_id:
            log.error("WHATSAPP_BUSINESS_ACCOUNT_ID not configured")
            return []

        endpoint = f"/{business_id}/phone_numbers"
        result = await self._request("GET", endpoint)
        if isinstance(result, dict):
            return result.get("data", [])
        return []


_whatsapp_client: WhatsAppClient | None = None


def get_whatsapp_client() -> WhatsAppClient:
    global _whatsapp_client
    if _whatsapp_client is None:
        _whatsapp_client = WhatsAppClient()
    return _whatsapp_client


class WhatsAppSendMessageTool(BaseTool):
    """Send a text message to a WhatsApp number."""

    name = "whatsapp_send_message"
    description = "Send a text message to a WhatsApp phone number"
    parameters = {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Recipient phone number in international format (e.g., '1234567890' for US, without + or spaces)",
            },
            "message": {
                "type": "string",
                "description": "Message text to send (max 4096 characters)",
            },
            "preview_url": {
                "type": "boolean",
                "description": "Whether to show URL previews in the message",
                "default": False,
            },
        },
        "required": ["phone_number", "message"],
    }

    async def execute(
        self,
        phone_number: str,
        message: str,
        preview_url: bool = False,
    ) -> ToolResult:
        client = get_whatsapp_client()
        if not client.access_token or not client.phone_number_id:
            return ToolResult(
                success=False,
                data=None,
                error="WhatsApp credentials not configured. Set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID environment variables.",
            )

        # Clean phone number (remove +, spaces, dashes)
        cleaned_number = phone_number.replace("+", "").replace(" ", "").replace("-", "")

        try:
            result = await client.send_text_message(cleaned_number, message, preview_url)
            if result:
                return ToolResult(
                    success=True,
                    data={
                        "message_id": result.get("messages", [{}])[0].get("id"),
                        "sent": True,
                        "recipient": cleaned_number,
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to send message")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WhatsAppSendTemplateTool(BaseTool):
    """Send a pre-approved template message."""

    name = "whatsapp_send_template"
    description = "Send a pre-approved WhatsApp template message (required for initial business conversations)"
    parameters = {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Recipient phone number in international format",
            },
            "template_name": {
                "type": "string",
                "description": "Name of the pre-approved template",
            },
            "language_code": {
                "type": "string",
                "description": "Language code (e.g., 'en', 'es', 'fr')",
                "default": "en",
            },
            "components": {
                "type": "array",
                "description": "Template components/parameters (optional)",
            },
        },
        "required": ["phone_number", "template_name"],
    }

    async def execute(
        self,
        phone_number: str,
        template_name: str,
        language_code: str = "en",
        components: list[dict] | None = None,
    ) -> ToolResult:
        client = get_whatsapp_client()
        if not client.access_token or not client.phone_number_id:
            return ToolResult(
                success=False,
                data=None,
                error="WhatsApp credentials not configured.",
            )

        cleaned_number = phone_number.replace("+", "").replace(" ", "").replace("-", "")

        try:
            result = await client.send_template_message(
                cleaned_number, template_name, language_code, components
            )
            if result:
                return ToolResult(
                    success=True,
                    data={
                        "message_id": result.get("messages", [{}])[0].get("id"),
                        "sent": True,
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to send template")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WhatsAppSendMediaTool(BaseTool):
    """Send media (image, document, audio, video) to WhatsApp."""

    name = "whatsapp_send_media"
    description = "Send media files (image, document, audio, video) to a WhatsApp number"
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
                "enum": ["image", "document", "audio", "video", "sticker"],
            },
            "media_path": {
                "type": "string",
                "description": "Local path to the media file (will be uploaded)",
            },
            "media_url": {
                "type": "string",
                "description": "URL of the media file (alternative to media_path)",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption for image, document, or video",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename for documents",
            },
        },
        "required": ["phone_number", "media_type"],
    }

    async def execute(
        self,
        phone_number: str,
        media_type: str,
        media_path: str | None = None,
        media_url: str | None = None,
        caption: str | None = None,
        filename: str | None = None,
    ) -> ToolResult:
        client = get_whatsapp_client()
        if not client.access_token or not client.phone_number_id:
            return ToolResult(
                success=False,
                data=None,
                error="WhatsApp credentials not configured.",
            )

        cleaned_number = phone_number.replace("+", "").replace(" ", "").replace("-", "")

        try:
            media_id = None
            if media_path:
                # Upload media first
                upload_result = await client.upload_media(media_path)
                if not upload_result:
                    return ToolResult(success=False, data=None, error="Failed to upload media")
                media_id = upload_result.get("id")
                # Use media URL from upload
                media_url = None

            result = await client.send_media_message(
                to=cleaned_number,
                media_type=media_type,
                media_id=media_id,
                media_url=media_url,
                caption=caption,
                filename=filename or (os.path.basename(media_path) if media_path else None),
            )

            if result:
                return ToolResult(
                    success=True,
                    data={
                        "message_id": result.get("messages", [{}])[0].get("id"),
                        "media_sent": True,
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to send media")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WhatsAppGetTemplatesTool(BaseTool):
    """Get list of pre-approved message templates."""

    name = "whatsapp_get_templates"
    description = "Get list of pre-approved WhatsApp message templates"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_whatsapp_client()
        if not client.access_token:
            return ToolResult(
                success=False,
                data=None,
                error="WhatsApp credentials not configured.",
            )

        try:
            templates = await client.get_templates()
            formatted = [
                {
                    "name": t.get("name"),
                    "status": t.get("status"),
                    "category": t.get("category"),
                    "language": t.get("language"),
                }
                for t in templates
            ]
            return ToolResult(success=True, data={"templates": formatted, "count": len(formatted)})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WhatsAppGetBusinessProfileTool(BaseTool):
    """Get WhatsApp business profile information."""

    name = "whatsapp_get_profile"
    description = "Get WhatsApp business profile information"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_whatsapp_client()
        if not client.access_token or not client.phone_number_id:
            return ToolResult(
                success=False,
                data=None,
                error="WhatsApp credentials not configured.",
            )

        try:
            profile = await client.get_business_profile()
            if profile:
                data = profile.get("data", [{}])[0]
                return ToolResult(
                    success=True,
                    data={
                        "about": data.get("about"),
                        "description": data.get("description"),
                        "email": data.get("email"),
                        "websites": data.get("websites", []),
                        "profile_picture_url": data.get("profile_picture_url"),
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to get profile")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WhatsAppDownloadMediaTool(BaseTool):
    """Download media from WhatsApp."""

    name = "whatsapp_download_media"
    description = "Download media file from WhatsApp using media ID"
    parameters = {
        "type": "object",
        "properties": {
            "media_id": {
                "type": "string",
                "description": "WhatsApp media ID",
            },
            "download_path": {
                "type": "string",
                "description": "Path to save the downloaded file",
            },
        },
        "required": ["media_id", "download_path"],
    }

    async def execute(self, media_id: str, download_path: str) -> ToolResult:
        client = get_whatsapp_client()
        if not client.access_token:
            return ToolResult(
                success=False,
                data=None,
                error="WhatsApp credentials not configured.",
            )

        try:
            success = await client.download_media(media_id, download_path)
            if success:
                return ToolResult(
                    success=True,
                    data={"downloaded": True, "path": download_path},
                )
            return ToolResult(success=False, data=None, error="Failed to download media")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
