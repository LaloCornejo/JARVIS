from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from tools.base import BaseTool, ToolResult


class VLCClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        password: str = "vlc",
    ):
        self.host = host
        self.port = port
        self.password = password
        self.base_url = f"http://{host}:{port}/requests"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                auth=("", self.password),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _command(self, command: str, **params: Any) -> dict[str, Any] | None:
        client = await self._get_client()
        params["command"] = command
        try:
            response = await client.get(f"{self.base_url}/status.json", params=params)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    async def status(self) -> dict[str, Any] | None:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/status.json")
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    async def play(self, uri: str | None = None) -> bool:
        if uri:
            result = await self._command("in_play", input=uri)
        else:
            result = await self._command("pl_play")
        return result is not None

    async def pause(self) -> bool:
        result = await self._command("pl_pause")
        return result is not None

    async def stop(self) -> bool:
        result = await self._command("pl_stop")
        return result is not None

    async def next_track(self) -> bool:
        result = await self._command("pl_next")
        return result is not None

    async def previous_track(self) -> bool:
        result = await self._command("pl_previous")
        return result is not None

    async def set_volume(self, volume: int) -> bool:
        vol = max(0, min(512, int(volume * 5.12)))
        result = await self._command("volume", val=str(vol))
        return result is not None

    async def seek(self, seconds: int) -> bool:
        result = await self._command("seek", val=str(seconds))
        return result is not None

    async def fullscreen(self) -> bool:
        result = await self._command("fullscreen")
        return result is not None

    async def get_playlist(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/playlist.json")
            if response.status_code == 200:
                data = response.json()
                items = []
                for child in data.get("children", []):
                    for item in child.get("children", []):
                        items.append(
                            {
                                "id": item.get("id"),
                                "name": item.get("name", ""),
                                "duration": item.get("duration", 0),
                                "uri": item.get("uri", ""),
                            }
                        )
                return items
        except Exception:
            pass
        return []


_vlc_client: VLCClient | None = None


def get_vlc_client() -> VLCClient:
    global _vlc_client
    if _vlc_client is None:
        _vlc_client = VLCClient()
    return _vlc_client


def launch_vlc_with_http() -> bool:
    vlc_paths = []
    if sys.platform == "win32":
        vlc_paths = [
            Path("C:/Program Files/VideoLAN/VLC/vlc.exe"),
            Path("C:/Program Files (x86)/VideoLAN/VLC/vlc.exe"),
        ]
    elif sys.platform == "darwin":
        vlc_paths = [Path("/Applications/VLC.app/Contents/MacOS/VLC")]
    else:
        vlc_paths = [Path("/usr/bin/vlc")]

    for vlc_path in vlc_paths:
        if vlc_path.exists():
            try:
                subprocess.Popen(
                    [str(vlc_path), "--extraintf", "http", "--http-password", "vlc"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                pass
    return False


class VLCPlayTool(BaseTool):
    name = "vlc_play"
    description = "Play a file or URL in VLC, or resume playback"
    parameters = {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "File path or URL to play (optional, resumes if empty)",
            },
        },
        "required": [],
    }

    async def execute(self, uri: str | None = None) -> ToolResult:
        client = get_vlc_client()
        try:
            success = await client.play(uri)
            if success:
                return ToolResult(
                    success=True,
                    data=f"Playing: {uri}" if uri else "Resumed playback",
                )
            return ToolResult(success=False, data=None, error="VLC not responding")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class VLCPauseTool(BaseTool):
    name = "vlc_pause"
    description = "Pause/unpause VLC playback"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_vlc_client()
        try:
            success = await client.pause()
            if success:
                return ToolResult(success=True, data="Toggled pause")
            return ToolResult(success=False, data=None, error="VLC not responding")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class VLCStopTool(BaseTool):
    name = "vlc_stop"
    description = "Stop VLC playback"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_vlc_client()
        try:
            success = await client.stop()
            if success:
                return ToolResult(success=True, data="Stopped playback")
            return ToolResult(success=False, data=None, error="VLC not responding")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class VLCNextTool(BaseTool):
    name = "vlc_next"
    description = "Skip to next track in VLC"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_vlc_client()
        try:
            success = await client.next_track()
            if success:
                return ToolResult(success=True, data="Skipped to next")
            return ToolResult(success=False, data=None, error="VLC not responding")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class VLCPreviousTool(BaseTool):
    name = "vlc_previous"
    description = "Go to previous track in VLC"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_vlc_client()
        try:
            success = await client.previous_track()
            if success:
                return ToolResult(success=True, data="Went to previous")
            return ToolResult(success=False, data=None, error="VLC not responding")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class VLCVolumeTool(BaseTool):
    name = "vlc_volume"
    description = "Set VLC volume (0-100)"
    parameters = {
        "type": "object",
        "properties": {
            "volume": {"type": "integer", "description": "Volume level 0-100"},
        },
        "required": ["volume"],
    }

    async def execute(self, volume: int) -> ToolResult:
        client = get_vlc_client()
        try:
            success = await client.set_volume(volume)
            if success:
                return ToolResult(success=True, data=f"Volume set to {volume}%")
            return ToolResult(success=False, data=None, error="VLC not responding")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class VLCStatusTool(BaseTool):
    name = "vlc_status"
    description = "Get current VLC playback status"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        client = get_vlc_client()
        try:
            status = await client.status()
            if status:
                info = status.get("information", {}).get("category", {}).get("meta", {})
                return ToolResult(
                    success=True,
                    data={
                        "state": status.get("state", "unknown"),
                        "volume": int(status.get("volume", 0) / 5.12),
                        "time": status.get("time", 0),
                        "length": status.get("length", 0),
                        "title": info.get("title", info.get("filename", "Unknown")),
                    },
                )
            return ToolResult(success=False, data=None, error="VLC not responding")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
