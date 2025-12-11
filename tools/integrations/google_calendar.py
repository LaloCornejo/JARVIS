from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult


class GoogleCalendarClient:
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    API_URL = "https://www.googleapis.com/calendar/v3"
    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str = "http://localhost:8888/callback",
        token_path: str = "data/google_calendar_token.json",
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
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        from urllib.parse import urlencode

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

    async def list_events(
        self,
        calendar_id: str = "primary",
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        if time_min is None:
            time_min = datetime.now()
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        params = {
            "timeMin": time_min.isoformat() + "Z",
            "timeMax": time_max.isoformat() + "Z",
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        result = await self._request("GET", f"/calendars/{calendar_id}/events", params=params)
        if result:
            return result.get("items", [])
        return []

    async def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: str = "",
        location: str = "",
        calendar_id: str = "primary",
    ) -> dict[str, Any] | None:
        event = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
        }
        return await self._request("POST", f"/calendars/{calendar_id}/events", json=event)

    async def delete_event(self, event_id: str, calendar_id: str = "primary") -> bool:
        result = await self._request("DELETE", f"/calendars/{calendar_id}/events/{event_id}")
        return result is not None

    async def list_calendars(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/users/me/calendarList")
        if result:
            return result.get("items", [])
        return []


_gcal_client: GoogleCalendarClient | None = None


def get_gcal_client() -> GoogleCalendarClient:
    global _gcal_client
    if _gcal_client is None:
        _gcal_client = GoogleCalendarClient()
    return _gcal_client


class CalendarListEventsTool(BaseTool):
    name = "calendar_list_events"
    description = "List upcoming calendar events"
    parameters = {
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Number of days ahead to look (default 7)",
            },
            "limit": {
                "type": "integer",
                "description": "Max events to return (default 10)",
            },
        },
        "required": [],
    }

    async def execute(self, days: int = 7, limit: int = 10) -> ToolResult:
        client = get_gcal_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Google Calendar not authenticated")

        try:
            now = datetime.now()
            events = await client.list_events(
                time_min=now,
                time_max=now + timedelta(days=days),
                max_results=limit,
            )
            formatted = []
            for event in events:
                start = event.get("start", {})
                start_time = start.get("dateTime", start.get("date", ""))
                formatted.append(
                    {
                        "id": event.get("id", ""),
                        "summary": event.get("summary", "No title"),
                        "start": start_time,
                        "location": event.get("location", ""),
                    }
                )
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CalendarCreateEventTool(BaseTool):
    name = "calendar_create_event"
    description = "Create a new calendar event"
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Event title"},
            "start": {"type": "string", "description": "Start time (ISO format)"},
            "end": {"type": "string", "description": "End time (ISO format)"},
            "description": {"type": "string", "description": "Event description"},
            "location": {"type": "string", "description": "Event location"},
        },
        "required": ["summary", "start", "end"],
    }

    async def execute(
        self,
        summary: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
    ) -> ToolResult:
        client = get_gcal_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Google Calendar not authenticated")

        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            event = await client.create_event(
                summary=summary,
                start=start_dt,
                end=end_dt,
                description=description,
                location=location,
            )
            if event:
                return ToolResult(
                    success=True,
                    data={
                        "id": event.get("id"),
                        "summary": event.get("summary"),
                        "link": event.get("htmlLink"),
                    },
                )
            return ToolResult(success=False, data=None, error="Failed to create event")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CalendarDeleteEventTool(BaseTool):
    name = "calendar_delete_event"
    description = "Delete a calendar event"
    parameters = {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "Event ID to delete"},
        },
        "required": ["event_id"],
    }

    async def execute(self, event_id: str) -> ToolResult:
        client = get_gcal_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Google Calendar not authenticated")

        try:
            success = await client.delete_event(event_id)
            if success:
                return ToolResult(success=True, data="Event deleted")
            return ToolResult(success=False, data=None, error="Failed to delete event")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
