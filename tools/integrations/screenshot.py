from __future__ import annotations

import asyncio
import base64
import datetime
import os
from pathlib import Path
from typing import Any

from core.llm import get_vision_client
from tools.base import BaseTool, ToolResult

try:
    from PIL import ImageGrab

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import win32api
    import win32gui

    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class ScreenshotManager:
    def __init__(self, save_dir: str | None = None):
        self.save_dir = Path(
            save_dir
            or os.environ.get(
                "SCREENSHOT_DIR",
                os.path.expanduser("~/.jarvis/screenshots"),
            )
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _capture_screen_sync(
        self,
        monitor: int = 0,
        save_path: str | None = None,
    ) -> tuple[bool, str, str | None]:
        if not HAS_PIL:
            return False, "", "PIL/Pillow not installed"

        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.save_dir / f"screenshot_{timestamp}.png")

        try:
            if monitor == 0:
                img = ImageGrab.grab(all_screens=False)
            else:
                img = ImageGrab.grab(all_screens=True)
            img.save(save_path)
            return True, save_path, None
        except Exception as e:
            return False, "", str(e)

    def _capture_all_monitors_sync(
        self,
        save_path: str | None = None,
    ) -> tuple[bool, str, str | None]:
        if not HAS_PIL:
            return False, "", "PIL/Pillow not installed"

        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.save_dir / f"screenshot_all_{timestamp}.png")

        try:
            img = ImageGrab.grab(all_screens=True)
            img.save(save_path)
            return True, save_path, None
        except Exception as e:
            return False, "", str(e)

    def _capture_window_sync(
        self,
        window_title: str | None = None,
        save_path: str | None = None,
    ) -> tuple[bool, str, str | None]:
        if not HAS_PIL:
            return False, "", "PIL/Pillow not installed"

        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.save_dir / f"window_{timestamp}.png")

        try:
            if HAS_WIN32:
                if window_title:
                    hwnd = win32gui.FindWindow(None, window_title)
                    if not hwnd:

                        def callback(h, extra):
                            if window_title.lower() in win32gui.GetWindowText(h).lower():
                                extra.append(h)
                            return True

                        hwnds = []
                        win32gui.EnumWindows(callback, hwnds)
                        hwnd = hwnds[0] if hwnds else None

                    if not hwnd:
                        return False, "", f"Window '{window_title}' not found"
                else:
                    hwnd = win32gui.GetForegroundWindow()

                rect = win32gui.GetWindowRect(hwnd)
                img = ImageGrab.grab(bbox=rect)
                img.save(save_path)
                return True, save_path, None
            else:
                img = ImageGrab.grab(all_screens=False)
                img.save(save_path)
                return True, save_path, None
        except Exception as e:
            return False, "", str(e)

    def _get_monitors_info_sync(self) -> list[dict[str, Any]]:
        monitors = []
        try:
            if HAS_WIN32:
                primary_width = win32api.GetSystemMetrics(0)
                primary_height = win32api.GetSystemMetrics(1)
                monitors.append(
                    {
                        "index": 0,
                        "name": "Primary",
                        "width": primary_width,
                        "height": primary_height,
                        "primary": True,
                    }
                )

                virtual_width = win32api.GetSystemMetrics(78)
                virtual_height = win32api.GetSystemMetrics(79)

                if virtual_width > primary_width or virtual_height > primary_height:
                    monitors.append(
                        {
                            "index": 1,
                            "name": "Virtual Desktop (all monitors)",
                            "width": virtual_width,
                            "height": virtual_height,
                            "primary": False,
                        }
                    )
            else:
                monitors.append(
                    {
                        "index": 0,
                        "name": "Primary",
                        "width": 1920,
                        "height": 1080,
                        "primary": True,
                    }
                )
        except Exception:
            pass
        return monitors

    async def capture_screen(
        self,
        monitor: int = 0,
        save_path: str | None = None,
    ) -> tuple[bool, str, str | None]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture_screen_sync, monitor, save_path)

    async def capture_all_monitors(
        self,
        save_path: str | None = None,
    ) -> tuple[bool, str, str | None]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture_all_monitors_sync, save_path)

    async def capture_window(
        self,
        window_title: str | None = None,
        save_path: str | None = None,
    ) -> tuple[bool, str, str | None]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture_window_sync, window_title, save_path)

    async def get_monitors_info(self) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_monitors_info_sync)

    def get_base64_image(self, path: str) -> str | None:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def list_screenshots(self, limit: int = 20) -> list[dict[str, Any]]:
        screenshots = []
        for f in sorted(self.save_dir.glob("*.png"), key=os.path.getmtime, reverse=True)[:limit]:
            stat = f.stat()
            screenshots.append(
                {
                    "path": str(f),
                    "name": f.name,
                    "size_bytes": stat.st_size,
                    "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )
        return screenshots


_manager: ScreenshotManager | None = None


def get_screenshot_manager() -> ScreenshotManager:
    global _manager
    if _manager is None:
        _manager = ScreenshotManager()
    return _manager


class ScreenshotCaptureTool(BaseTool):
    name = "screenshot_capture"
    description = (
        "Take a screenshot and save it to a file. Returns only the file path. "
        "Use screenshot_analyze instead if you need to describe what's on screen."
    )
    parameters = {
        "type": "object",
        "properties": {
            "monitor": {
                "type": "integer",
                "description": "Monitor index (0 for primary, 1+ for all monitors). Default: 0",
            },
        },
        "required": [],
    }

    async def execute(self, monitor: int = 0, **kwargs) -> ToolResult:
        manager = get_screenshot_manager()
        try:
            success, path, error = await manager.capture_screen(monitor=monitor)
            if not success:
                return ToolResult(success=False, data=None, error=error)

            return ToolResult(success=True, data={"path": path})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenshotAllMonitorsTool(BaseTool):
    name = "screenshot_all_monitors"
    description = "Take a screenshot of all monitors combined into one image. Returns file path."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self, **kwargs) -> ToolResult:
        manager = get_screenshot_manager()
        try:
            success, path, error = await manager.capture_all_monitors()
            if not success:
                return ToolResult(success=False, data=None, error=error)

            return ToolResult(success=True, data={"path": path})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenshotWindowTool(BaseTool):
    name = "screenshot_window"
    description = (
        "Take a screenshot of a specific window by title, or the active window. Returns file path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "window_title": {
                "type": "string",
                "description": "Partial window title to match. Captures active window if omitted.",
            },
        },
        "required": [],
    }

    async def execute(self, window_title: str | None = None, **kwargs) -> ToolResult:
        manager = get_screenshot_manager()
        try:
            success, path, error = await manager.capture_window(window_title=window_title)
            if not success:
                return ToolResult(success=False, data=None, error=error)

            return ToolResult(success=True, data={"path": path})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenshotListMonitorsTool(BaseTool):
    name = "screenshot_list_monitors"
    description = "List all available monitors with their dimensions."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        manager = get_screenshot_manager()
        try:
            monitors = await manager.get_monitors_info()
            return ToolResult(success=True, data=monitors)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenshotListTool(BaseTool):
    name = "screenshot_list"
    description = "List recent screenshots saved by JARVIS."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of screenshots to return. Default: 20",
            },
        },
        "required": [],
    }

    async def execute(self, limit: int = 20) -> ToolResult:
        manager = get_screenshot_manager()
        try:
            screenshots = manager.list_screenshots(limit=limit)
            return ToolResult(success=True, data=screenshots)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenshotAnalyzeTool(BaseTool):
    name = "screenshot_analyze"
    description = (
        "Take a screenshot and analyze it using vision AI. "
        "Returns a text description of what's on screen. "
        "Use this when the user asks about what they're watching, viewing, or seeing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to analyze or look for in the screenshot.",
            },
            "monitor": {
                "type": "integer",
                "description": "Monitor index (0 for primary). Default: 0",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        question: str = "Describe what is shown on the screen in detail.",
        monitor: int = 0,
    ) -> ToolResult:
        manager = get_screenshot_manager()
        try:
            success, path, error = await manager.capture_screen(monitor=monitor)
            if not success:
                return ToolResult(success=False, data=None, error=error)

            b64 = manager.get_base64_image(path)
            if not b64:
                return ToolResult(success=False, data=None, error="Failed to read screenshot")

            vision = get_vision_client()
            healthy = await vision.health_check()
            if not healthy:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Ollama not available. Ensure Ollama is running with qwen3-vl model.",
                )

            analysis_text = ""
            async for chunk in vision.generate(
                prompt=question,
                images=[b64],
                stream=True,
                temperature=0.3,
            ):
                analysis_text += chunk

            return ToolResult(
                success=True,
                data={
                    "analysis": analysis_text,
                    "screenshot_path": path,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
