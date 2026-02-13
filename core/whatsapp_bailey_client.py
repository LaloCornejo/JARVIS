"""Python client for WhatsApp Baileys Node.js service."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import httpx

if TYPE_CHECKING:
    from jarvis.server import JarvisServer

log = logging.getLogger(__name__)

BAILEYS_SERVICE_URL = os.environ.get("BAILEYS_SERVICE_URL", "http://localhost:3001")
WHATSAPP_OWNER_NUMBER = os.environ.get("WHATSAPP_OWNER_NUMBER", "")


@dataclass
class WhatsAppSession:
    user_id: str
    user_name: str
    phone_number: str = ""
    messages: list[dict] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    max_messages: int = 100

    def add_message(self, role: str, content: str, **kwargs) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }
        )
        self.last_activity = datetime.now()
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_context(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "phone_number": self.phone_number,
            "message_count": len(self.messages),
            "last_activity": self.last_activity.isoformat(),
        }


class WhatsAppBaileyClient:
    def __init__(
        self,
        jarvis_server: Optional["JarvisServer"] = None,
        service_url: str | None = None,
        allowed_contacts: Optional[List[str]] = None,
        owner_number: str | None = None,
    ):
        self.jarvis = jarvis_server
        self.service_url = service_url or BAILEYS_SERVICE_URL
        self._client: httpx.AsyncClient | None = None

        if allowed_contacts is not None:
            self.allowed_contacts = set(allowed_contacts)
        else:
            env_contacts = os.environ.get("WHATSAPP_ALLOWED_CONTACTS", "")
            self.allowed_contacts = (
                set(c.strip() for c in env_contacts.split(",") if c.strip()) or None
            )

        self.owner_number = (
            (owner_number or WHATSAPP_OWNER_NUMBER or "")
            .replace("+", "")
            .replace("@s.whatsapp.net", "")
        )
        self._running = False
        self._sessions: Dict[str, WhatsAppSession] = {}
        self._webhook_server = None
        self._all_messages: List[dict] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def check_status(self) -> dict:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.service_url}/status")
            return response.json()
        except Exception as e:
            return {"connected": False, "error": str(e)}

    async def get_qr_code(self) -> dict:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.service_url}/qr")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def send_message(self, to: str, message: str) -> dict:
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.service_url}/send", json={"to": to, "message": message}
            )
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_media(
        self,
        to: str,
        media_type: str,
        url: str,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict:
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.service_url}/send-media",
                json={
                    "to": to,
                    "type": media_type,
                    "url": url,
                    "caption": caption,
                    "filename": filename,
                },
            )
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def disconnect(self) -> dict:
        try:
            client = await self._get_client()
            response = await client.post(f"{self.service_url}/disconnect")
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def handle_incoming_webhook(self, data: dict) -> None:
        if data.get("type") != "message":
            return

        sender = data.get("from", "")
        sender_name = data.get("sender_name", "Unknown")
        content = data.get("content", "")
        clean_sender = sender.replace("@s.whatsapp.net", "").replace("@g.us", "")

        self._all_messages.append(
            {
                "sender": sender,
                "sender_name": sender_name,
                "clean_sender": clean_sender,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "processed": False,
            }
        )

        is_owner = bool(self.owner_number) and clean_sender == self.owner_number

        if is_owner:
            log.info(f"[WhatsApp] [OWNER] Message from {sender_name}: {content[:50]}...")
            await self._process_and_respond(sender, sender_name, content)
        else:
            log.info(
                f"[WhatsApp] [MONITOR] Message from {sender_name} (not auto-responded): {content[:100]}..."
            )
            session = self._get_or_create_session(sender, sender_name)
            session.add_message("user", content)
            self._all_messages[-1]["processed"] = True

    async def _process_and_respond(self, sender: str, sender_name: str, content: str) -> None:
        if not self.jarvis:
            log.error("[WhatsApp] No JarvisServer configured")
            return

        session = self._get_or_create_session(sender, sender_name)
        session.add_message("user", content)

        try:
            full_response = ""

            async def capture_broadcast(message: dict) -> None:
                nonlocal full_response
                if message.get("type") == "message_complete":
                    full_response = message.get("full_response", "")

            await self.jarvis.process_message(content, broadcast_func=capture_broadcast)

            if full_response:
                session.add_message("assistant", full_response)
                await self.send_long_message(sender, full_response)

        except Exception as e:
            log.error(f"[WhatsApp] Error processing message: {e}")
            await self.send_message(sender, f"Error: {str(e)[:200]}")

    async def send_long_message(self, to: str, text: str) -> None:
        max_length = 3000

        if len(text) <= max_length:
            await self.send_message(to, text)
            return

        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            split_points = [
                text.rfind("\n", 0, max_length),
                text.rfind(". ", 0, max_length),
                text.rfind(" ", 0, max_length),
            ]
            split_point = (
                max(p for p in split_points if p != -1)
                if any(p != -1 for p in split_points)
                else max_length
            )

            chunks.append(text[:split_point])
            text = text[split_point:].strip()

        for i, chunk in enumerate(chunks):
            prefix = f"({i + 1}/{len(chunks)}) " if len(chunks) > 1 else ""
            await self.send_message(to, prefix + chunk)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)

    def _get_or_create_session(self, user_id: str, user_name: str) -> WhatsAppSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = WhatsAppSession(
                user_id=user_id,
                user_name=user_name,
                phone_number=user_id.replace("@s.whatsapp.net", ""),
            )
            log.info(f"[WhatsApp] New session: {user_name}")
        return self._sessions[user_id]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


whatsapp_bailey_client = WhatsAppBaileyClient()
