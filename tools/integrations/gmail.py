from __future__ import annotations

import base64
import json
import os
from base64 import urlsafe_b64decode, urlsafe_b64encode
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult


class GmailClient:
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    API_URL = "https://gmail.googleapis.com/gmail/v1"
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
    ]

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str = "http://localhost:8888/callback",
        token_path: str = "data/gmail_token.json",
    ):
        self.client_id = client_id or os.environ.get("GOOGLE_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("GOOGLE_CLIENT_SECRET", "")
        self.redirect_uri = redirect_uri
        self.token_path = Path(token_path)
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._load_token()

    def _load_token(self) -> None:
        if self.token_path.exists():
            try:
                data = json.loads(self.token_path.read_text())
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
            except Exception:
                pass

    def _save_token(self) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(
            json.dumps(
                {
                    "access_token": self.access_token,
                    "refresh_token": self.refresh_token,
                }
            )
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_auth_url(self) -> str:
        from urllib.parse import urlencode

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def authenticate(self, code: str) -> bool:
        client = await self._get_client()
        response = await client.post(
            self.TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri,
            },
        )
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data.get("refresh_token")
            self._save_token()
            return True
        return False

    async def refresh_access_token(self) -> bool:
        if not self.refresh_token:
            return False

        client = await self._get_client()
        response = await client.post(
            self.TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            },
        )
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self._save_token()
            return True
        return False

    async def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any] | None:
        if not self.access_token:
            return None

        client = await self._get_client()
        headers = {"Authorization": f"Bearer {self.access_token}"}

        response = await client.request(
            method, f"{self.API_URL}{endpoint}", headers=headers, **kwargs
        )

        if response.status_code == 401:
            if await self.refresh_access_token():
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = await client.request(
                    method, f"{self.API_URL}{endpoint}", headers=headers, **kwargs
                )

        if response.status_code in (200, 201):
            return response.json() if response.content else {}
        elif response.status_code == 204:
            return {}
        return None

    async def list_messages(
        self,
        query: str = "",
        max_results: int = 10,
        label_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"maxResults": max_results}
        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = ",".join(label_ids)

        result = await self._request("GET", "/users/me/messages", params=params)
        if not result:
            return []

        messages = []
        for msg in result.get("messages", [])[:max_results]:
            full_msg = await self.get_message(msg["id"])
            if full_msg:
                messages.append(full_msg)
        return messages

    async def get_message(self, message_id: str) -> dict[str, Any] | None:
        result = await self._request(
            "GET",
            f"/users/me/messages/{message_id}",
            params={"format": "full"},
        )
        if not result:
            return None

        headers = {}
        for header in result.get("payload", {}).get("headers", []):
            name = header.get("name", "").lower()
            if name in ("from", "to", "subject", "date"):
                headers[name] = header.get("value", "")

        body = ""
        payload = result.get("payload", {})
        if "body" in payload and payload["body"].get("data"):
            body = urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
        elif "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                    body = urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                    break

        return {
            "id": result.get("id"),
            "thread_id": result.get("threadId"),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "snippet": result.get("snippet", ""),
            "body": body[:2000],
            "labels": result.get("labelIds", []),
        }

    async def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
    ) -> dict[str, Any] | None:
        message_lines = [
            f"To: {to}",
            f"Subject: {subject}",
            "MIME-Version: 1.0",
            "Content-Type: text/plain; charset=utf-8",
        ]
        if cc:
            message_lines.insert(1, f"Cc: {cc}")
        if bcc:
            message_lines.insert(1, f"Bcc: {bcc}")
        message_lines.append("")
        message_lines.append(body)

        raw_message = "\r\n".join(message_lines)
        encoded = urlsafe_b64encode(raw_message.encode("utf-8")).decode("ascii")

        return await self._request(
            "POST",
            "/users/me/messages/send",
            json={"raw": encoded},
        )

    async def mark_as_read(self, message_id: str) -> bool:
        result = await self._request(
            "POST",
            f"/users/me/messages/{message_id}/modify",
            json={"removeLabelIds": ["UNREAD"]},
        )
        return result is not None

    async def mark_as_unread(self, message_id: str) -> bool:
        result = await self._request(
            "POST",
            f"/users/me/messages/{message_id}/modify",
            json={"addLabelIds": ["UNREAD"]},
        )
        return result is not None

    async def archive_message(self, message_id: str) -> bool:
        result = await self._request(
            "POST",
            f"/users/me/messages/{message_id}/modify",
            json={"removeLabelIds": ["INBOX"]},
        )
        return result is not None

    async def star_message(self, message_id: str, unstar: bool = False) -> bool:
        if unstar:
            result = await self._request(
                "POST",
                f"/users/me/messages/{message_id}/modify",
                json={"removeLabelIds": ["STARRED"]},
            )
        else:
            result = await self._request(
                "POST",
                f"/users/me/messages/{message_id}/modify",
                json={"addLabelIds": ["STARRED"]},
            )
        return result is not None

    async def trash_message(self, message_id: str) -> bool:
        result = await self._request("POST", f"/users/me/messages/{message_id}/trash")
        return result is not None


_gmail_client: GmailClient | None = None


def get_gmail_client() -> GmailClient:
    global _gmail_client
    if _gmail_client is None:
        _gmail_client = GmailClient()
    return _gmail_client


class GmailListTool(BaseTool):
    name = "gmail_list"
    description = "List emails from Gmail inbox"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (Gmail search syntax)",
            },
            "limit": {"type": "integer", "description": "Max emails to return"},
            "unread_only": {"type": "boolean", "description": "Only unread emails"},
        },
        "required": [],
    }

    async def execute(
        self, query: str = "", limit: int = 10, unread_only: bool = False
    ) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            if unread_only and "is:unread" not in query:
                query = f"is:unread {query}".strip()
            messages = await client.list_messages(query=query, max_results=limit)
            formatted = [
                {
                    "id": m["id"],
                    "from": m["from"],
                    "subject": m["subject"],
                    "date": m["date"],
                    "snippet": m["snippet"],
                }
                for m in messages
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailReadTool(BaseTool):
    name = "gmail_read"
    description = "Read a specific email by ID"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Email message ID"},
        },
        "required": ["message_id"],
    }

    async def execute(self, message_id: str) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            message = await client.get_message(message_id)
            if message:
                return ToolResult(success=True, data=message)
            return ToolResult(success=False, data=None, error="Message not found")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailSendTool(BaseTool):
    name = "gmail_send"
    description = "Send an email via Gmail"
    parameters = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
            "cc": {"type": "string", "description": "CC recipients"},
        },
        "required": ["to", "subject", "body"],
    }

    async def execute(self, to: str, subject: str, body: str, cc: str = "") -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            result = await client.send_message(to=to, subject=subject, body=body, cc=cc)
            if result:
                return ToolResult(
                    success=True,
                    data={"message_id": result.get("id"), "status": "sent"},
                )
            return ToolResult(success=False, data=None, error="Failed to send")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailMarkReadTool(BaseTool):
    name = "gmail_mark_read"
    description = "Mark an email as read"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Email message ID"},
        },
        "required": ["message_id"],
    }

    async def execute(self, message_id: str) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            success = await client.mark_as_read(message_id)
            if success:
                return ToolResult(success=True, data="Marked as read")
            return ToolResult(success=False, data=None, error="Failed")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailMarkUnreadTool(BaseTool):
    name = "gmail_mark_unread"
    description = "Mark an email as unread"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Email message ID"},
        },
        "required": ["message_id"],
    }

    async def execute(self, message_id: str) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            result = await client._request(
                "POST",
                f"/users/me/messages/{message_id}/modify",
                json={"addLabelIds": ["UNREAD"]},
            )
            if result is not None:
                return ToolResult(success=True, data="Marked as unread")
            return ToolResult(success=False, data=None, error="Failed")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailDeleteTool(BaseTool):
    name = "gmail_delete"
    description = "Move an email to trash"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Email message ID"},
        },
        "required": ["message_id"],
    }

    async def execute(self, message_id: str) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            success = await client.trash_message(message_id)
            if success:
                return ToolResult(success=True, data="Moved to trash")
            return ToolResult(success=False, data=None, error="Failed")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailArchiveTool(BaseTool):
    name = "gmail_archive"
    description = "Archive an email (remove from inbox)"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Email message ID"},
        },
        "required": ["message_id"],
    }

    async def execute(self, message_id: str) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            result = await client._request(
                "POST",
                f"/users/me/messages/{message_id}/modify",
                json={"removeLabelIds": ["INBOX"]},
            )
            if result is not None:
                return ToolResult(success=True, data="Archived")
            return ToolResult(success=False, data=None, error="Failed")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailStarTool(BaseTool):
    name = "gmail_star"
    description = "Star an email"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Email message ID"},
            "unstar": {
                "type": "boolean",
                "description": "Remove star instead of adding",
                "default": False,
            },
        },
        "required": ["message_id"],
    }

    async def execute(self, message_id: str, unstar: bool = False) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            if unstar:
                result = await client._request(
                    "POST",
                    f"/users/me/messages/{message_id}/modify",
                    json={"removeLabelIds": ["STARRED"]},
                )
                msg = "Unstarred"
            else:
                result = await client._request(
                    "POST",
                    f"/users/me/messages/{message_id}/modify",
                    json={"addLabelIds": ["STARRED"]},
                )
                msg = "Starred"
            if result is not None:
                return ToolResult(success=True, data=msg)
            return ToolResult(success=False, data=None, error="Failed")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GmailReplyTool(BaseTool):
    name = "gmail_reply"
    description = "Reply to an email"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Email message ID to reply to"},
            "body": {"type": "string", "description": "Reply body text"},
        },
        "required": ["message_id", "body"],
    }

    async def execute(self, message_id: str, body: str) -> ToolResult:
        client = get_gmail_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Gmail not authenticated")

        try:
            # Get original message for thread_id and headers
            original = await client.get_message(message_id)
            if not original:
                return ToolResult(success=False, data=None, error="Original message not found")

            thread_id = original.get("thread_id")
            to = original.get("from", "")
            subject = original.get("subject", "")
            if not subject.startswith("Re:"):
                subject = f"Re: {subject}"

            # Build reply message
            msg = MIMEText(body)
            msg["To"] = to
            msg["Subject"] = subject
            msg["In-Reply-To"] = message_id
            msg["References"] = message_id

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

            result = await client._request(
                "POST",
                "/users/me/messages/send",
                json={"raw": raw, "threadId": thread_id},
            )
            if result:
                return ToolResult(
                    success=True,
                    data={"message_id": result.get("id"), "status": "sent"},
                )
            return ToolResult(success=False, data=None, error="Failed to send reply")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
