from __future__ import annotations

import json
import os
from base64 import b64encode
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from tools.base import BaseTool, ToolResult

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SPOTIFY_TOKEN_FILE = DATA_DIR / "spotify_token.json"


class SpotifyClient:
    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_URL = "https://api.spotify.com/v1"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str = "http://localhost:8888/callback",
    ):
        self.client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET", "")
        self.redirect_uri = redirect_uri
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._load_tokens()

    def _load_tokens(self) -> None:
        if SPOTIFY_TOKEN_FILE.exists():
            try:
                data = json.loads(SPOTIFY_TOKEN_FILE.read_text())
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
            except (json.JSONDecodeError, OSError):
                pass

    def _save_tokens(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        token_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
        }
        SPOTIFY_TOKEN_FILE.write_text(json.dumps(token_data, indent=2))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_auth_url(self, scopes: list[str] | None = None) -> str:
        if scopes is None:
            scopes = [
                "user-read-playback-state",
                "user-modify-playback-state",
                "user-read-currently-playing",
                "playlist-read-private",
                "playlist-modify-public",
                "playlist-modify-private",
            ]
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes),
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def authenticate(self, code: str) -> bool:
        client = await self._get_client()
        auth_header = b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()

        response = await client.post(
            self.TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
            },
            headers={"Authorization": f"Basic {auth_header}"},
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data.get("refresh_token")
            self._save_tokens()
            return True
        return False

    async def refresh(self) -> bool:
        if not self.refresh_token:
            return False

        client = await self._get_client()
        auth_header = b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()

        response = await client.post(
            self.TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
            headers={"Authorization": f"Basic {auth_header}"},
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            if data.get("refresh_token"):
                self.refresh_token = data["refresh_token"]
            self._save_tokens()
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
            if await self.refresh():
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = await client.request(
                    method, f"{self.API_URL}{endpoint}", headers=headers, **kwargs
                )

        if response.status_code in (200, 201):
            return response.json() if response.content else {}
        elif response.status_code == 204:
            return {}
        return None

    async def get_current_playback(self) -> dict[str, Any] | None:
        return await self._request("GET", "/me/player")

    async def play(self, uri: str | None = None, context_uri: str | None = None) -> bool:
        data = {}
        if uri:
            data["uris"] = [uri]
        elif context_uri:
            data["context_uri"] = context_uri
        result = await self._request("PUT", "/me/player/play", json=data if data else None)
        return result is not None

    async def pause(self) -> bool:
        result = await self._request("PUT", "/me/player/pause")
        return result is not None

    async def next_track(self) -> bool:
        result = await self._request("POST", "/me/player/next")
        return result is not None

    async def previous_track(self) -> bool:
        result = await self._request("POST", "/me/player/previous")
        return result is not None

    async def set_volume(self, volume: int) -> bool:
        volume = max(0, min(100, volume))
        result = await self._request("PUT", f"/me/player/volume?volume_percent={volume}")
        return result is not None

    async def search(
        self, query: str, types: list[str] | None = None, limit: int = 10
    ) -> dict[str, Any] | None:
        if types is None:
            types = ["track", "artist", "album"]
        params = {"q": query, "type": ",".join(types), "limit": limit}
        return await self._request("GET", "/search", params=params)

    async def get_playlists(self, limit: int = 20) -> list[dict[str, Any]]:
        result = await self._request("GET", f"/me/playlists?limit={limit}")
        if result:
            return result.get("items", [])
        return []


_spotify_client: SpotifyClient | None = None


def get_spotify_client() -> SpotifyClient:
    global _spotify_client
    if _spotify_client is None:
        _spotify_client = SpotifyClient()
    return _spotify_client


class SpotifyPlayTool(BaseTool):
    name = "spotify_play"
    description = "Play music on Spotify. Can resume playback or play a specific track/playlist."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional search query to play a specific track or playlist",
            },
        },
        "required": [],
    }

    async def execute(self, query: str | None = None) -> ToolResult:
        client = get_spotify_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Spotify not authenticated")

        try:
            if query:
                results = await client.search(query, types=["track"], limit=1)
                if results and results.get("tracks", {}).get("items"):
                    track = results["tracks"]["items"][0]
                    uri = track["uri"]
                    await client.play(uri=uri)
                    return ToolResult(
                        success=True,
                        data=f"Playing: {track['name']} by {track['artists'][0]['name']}",
                    )
                return ToolResult(success=False, data=None, error="No tracks found")
            else:
                await client.play()
                return ToolResult(success=True, data="Resumed playback")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SpotifyPauseTool(BaseTool):
    name = "spotify_pause"
    description = "Pause Spotify playback"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_spotify_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Spotify not authenticated")

        try:
            await client.pause()
            return ToolResult(success=True, data="Paused playback")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SpotifyNextTool(BaseTool):
    name = "spotify_next"
    description = "Skip to the next track on Spotify"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_spotify_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Spotify not authenticated")

        try:
            await client.next_track()
            return ToolResult(success=True, data="Skipped to next track")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SpotifyPreviousTool(BaseTool):
    name = "spotify_previous"
    description = "Go back to the previous track on Spotify"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_spotify_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Spotify not authenticated")

        try:
            await client.previous_track()
            return ToolResult(success=True, data="Went to previous track")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SpotifyVolumeTool(BaseTool):
    name = "spotify_volume"
    description = "Set Spotify volume (0-100)"
    parameters = {
        "type": "object",
        "properties": {
            "volume": {
                "type": "integer",
                "description": "Volume level from 0 to 100",
            },
        },
        "required": ["volume"],
    }

    async def execute(self, volume: int) -> ToolResult:
        client = get_spotify_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Spotify not authenticated")

        try:
            await client.set_volume(volume)
            return ToolResult(success=True, data=f"Volume set to {volume}%")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SpotifyCurrentTool(BaseTool):
    name = "spotify_current"
    description = "Get the currently playing track on Spotify"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_spotify_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Spotify not authenticated")

        try:
            playback = await client.get_current_playback()
            if playback and playback.get("item"):
                track = playback["item"]
                artist = track["artists"][0]["name"] if track.get("artists") else "Unknown"
                return ToolResult(
                    success=True,
                    data={
                        "track": track["name"],
                        "artist": artist,
                        "album": track.get("album", {}).get("name", "Unknown"),
                        "is_playing": playback.get("is_playing", False),
                        "progress_ms": playback.get("progress_ms", 0),
                        "duration_ms": track.get("duration_ms", 0),
                    },
                )
            return ToolResult(success=True, data="Nothing is currently playing")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SpotifySearchTool(BaseTool):
    name = "spotify_search"
    description = "Search for tracks, artists, or albums on Spotify"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "type": {
                "type": "string",
                "description": "Type of search: track, artist, album, or playlist",
                "enum": ["track", "artist", "album", "playlist"],
            },
            "limit": {
                "type": "integer",
                "description": "Number of results (default 5)",
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, type: str = "track", limit: int = 5) -> ToolResult:
        client = get_spotify_client()
        if not client.access_token:
            return ToolResult(success=False, data=None, error="Spotify not authenticated")

        try:
            results = await client.search(query, types=[type], limit=limit)
            if not results:
                return ToolResult(success=False, data=None, error="Search failed")

            key = f"{type}s"
            items = results.get(key, {}).get("items", [])

            formatted = []
            for item in items:
                if type == "track":
                    formatted.append(
                        {
                            "name": item["name"],
                            "artist": item["artists"][0]["name"] if item.get("artists") else "",
                            "uri": item["uri"],
                        }
                    )
                elif type == "artist":
                    formatted.append(
                        {
                            "name": item["name"],
                            "genres": item.get("genres", []),
                            "uri": item["uri"],
                        }
                    )
                elif type == "album":
                    formatted.append(
                        {
                            "name": item["name"],
                            "artist": item["artists"][0]["name"] if item.get("artists") else "",
                            "uri": item["uri"],
                        }
                    )
                elif type == "playlist":
                    formatted.append(
                        {
                            "name": item["name"],
                            "owner": item.get("owner", {}).get("display_name", ""),
                            "uri": item["uri"],
                        }
                    )

            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
