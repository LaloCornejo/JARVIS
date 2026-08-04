from __future__ import annotations

import os
import re
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult


class YouTubeClient:
    API_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "video",
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            return []

        client = await self._get_client()
        params = {
            "key": self.api_key,
            "q": query,
            "part": "snippet",
            "maxResults": max_results,
            "type": search_type,
        }
        response = await client.get(f"{self.API_URL}/search", params=params)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("items", []):
                video_id = item.get("id", {}).get("videoId", "")
                snippet = item.get("snippet", {})
                results.append(
                    {
                        "id": video_id,
                        "title": snippet.get("title", ""),
                        "channel": snippet.get("channelTitle", ""),
                        "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
                    }
                )
            return results
        else:
            # Provide better error information
            error_text = response.text
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_message = error_data["error"].get("message", error_text)
                    raise Exception(f"YouTube API error: {error_message}")
            except Exception:
                raise Exception(f"YouTube API error ({response.status_code}): {error_text}")
        return []

    async def get_video_info(self, video_id: str) -> dict[str, Any] | None:
        if not self.api_key:
            return None

        client = await self._get_client()
        params = {
            "key": self.api_key,
            "id": video_id,
            "part": "snippet,contentDetails,statistics",
        }
        response = await client.get(f"{self.API_URL}/videos", params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            if items:
                item = items[0]
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                return {
                    "id": video_id,
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "description": snippet.get("description", "")[:500],
                    "views": stats.get("viewCount", "0"),
                    "likes": stats.get("likeCount", "0"),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }
        return None

    async def get_playlist(self, playlist_id: str, max_results: int = 10) -> list[dict[str, Any]]:
        if not self.api_key:
            return []

        client = await self._get_client()
        params = {
            "key": self.api_key,
            "playlistId": playlist_id,
            "part": "snippet",
            "maxResults": max_results,
        }
        response = await client.get(f"{self.API_URL}/playlistItems", params=params)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                video_id = snippet.get("resourceId", {}).get("videoId", "")
                results.append(
                    {
                        "id": video_id,
                        "title": snippet.get("title", ""),
                        "position": snippet.get("position", 0),
                        "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
                    }
                )
            return results
        return []


_youtube_client: YouTubeClient | None = None


def get_youtube_client() -> YouTubeClient:
    global _youtube_client
    if _youtube_client is None:
        _youtube_client = YouTubeClient()
    return _youtube_client


class YouTubeSearchTool(BaseTool):
    name = "youtube_search"
    description = "Search for videos on YouTube"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, limit: int = 5) -> ToolResult:
        client = get_youtube_client()
        try:
            results = await client.search(query, max_results=limit)
            if results:
                return ToolResult(success=True, data=results)
            return ToolResult(success=False, data=None, error="No results found")
        except Exception as e:
            error_msg = f"YouTube search failed: {str(e)}"
            return ToolResult(success=False, data=None, error=error_msg)


class YouTubePlayTool(BaseTool):
    name = "youtube_play"
    description = "Search and open a YouTube video in the browser"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Video to search for"},
        },
        "required": ["query"],
    }

    async def execute(self, query: str) -> ToolResult:
        import webbrowser

        client = get_youtube_client()
        try:
            if re.match(r"^[a-zA-Z0-9_-]{11}$", query):
                url = f"https://www.youtube.com/watch?v={query}"
                webbrowser.open(url)
                return ToolResult(success=True, data=f"Opening: {url}")

            results = await client.search(query, max_results=1)
            if results:
                url = results[0]["url"]
                webbrowser.open(url)
                return ToolResult(
                    success=True,
                    data=f"Playing: {results[0]['title']} - {url}",
                )
            return ToolResult(success=False, data=None, error="No videos found")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class YouTubeInfoTool(BaseTool):
    name = "youtube_info"
    description = "Get information about a YouTube video"
    parameters = {
        "type": "object",
        "properties": {
            "video_id": {"type": "string", "description": "YouTube video ID"},
        },
        "required": ["video_id"],
    }

    async def execute(self, video_id: str) -> ToolResult:
        client = get_youtube_client()
        try:
            info = await client.get_video_info(video_id)
            if info:
                return ToolResult(success=True, data=info)
            return ToolResult(success=False, data=None, error="Video not found")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
