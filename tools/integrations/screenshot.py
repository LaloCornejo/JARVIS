from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import io
import os
from pathlib import Path
from typing import Any

from core.llm import get_vision_client, get_fast_client
from tools.base import BaseTool, ToolResult

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageGrab

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
        self._analysis_cache: dict[str, dict] = {}  # Cache for screenshot analyses
        self._cache_ttl = 60  # Cache TTL in seconds

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

    def get_base64_image(self, path: str, max_size: tuple[int, int] = (1024, 1024)) -> str | None:
        try:
            if not HAS_PIL:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            
            # Open and resize image to reduce processing time
            img = Image.open(path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            
            # Apply preprocessing to improve text readability for vision model
            from PIL import ImageEnhance, ImageFilter
            # Enhance sharpness more aggressively for better text clarity
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            # Increase contrast to make text stand out better
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            # Slight brightness adjustment if needed
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
            
            # Unsharp mask for even better text clarity
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
            
            # Save to bytes with balanced quality
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=75, optimize=True)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception:
            return None

    def _get_cache_key(self, path: str, question: str) -> str:
        """Generate a cache key based on file path and question."""
        return hashlib.md5(f"{path}:{question}".encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return (datetime.datetime.now().timestamp() - timestamp) < self._cache_ttl
    
    def get_cached_analysis(self, path: str, question: str) -> dict | None:
        """Get cached analysis if available and valid."""
        cache_key = self._get_cache_key(path, question)
        if cache_key in self._analysis_cache:
            entry = self._analysis_cache[cache_key]
            if self._is_cache_valid(entry["timestamp"]):
                return entry["data"]
            else:
                # Remove expired entry
                del self._analysis_cache[cache_key]
        return None
    
    def cache_analysis(self, path: str, question: str, data: dict) -> None:
        """Cache analysis result."""
        cache_key = self._get_cache_key(path, question)
        self._analysis_cache[cache_key] = {
            "data": data,
            "timestamp": datetime.datetime.now().timestamp()
        }
    
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
        question: str = "What's in this screenshot? Brief answer in 1 sentence.",
        monitor: int = 0,
    ) -> ToolResult:
        manager = get_screenshot_manager()
        try:
            # Check cache first
            success, path, error = await manager.capture_screen(monitor=monitor)
            if not success:
                return ToolResult(success=False, data=None, error=error)
            
            # Check if we have a cached result
            cached_result = manager.get_cached_analysis(path, question)
            if cached_result:
                return ToolResult(success=True, data=cached_result)
            
            b64 = manager.get_base64_image(path, max_size=(640, 480))
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

            # Stream response progressively for better perceived performance
            analysis_chunks = []
            analysis_text = ""
            chunk_count = 0
            
            try:
                async with asyncio.timeout(40):  # 40 second timeout for larger images
                    async for chunk in vision.generate(
                        prompt=question,
                        images=[b64],
                        stream=True,
                        temperature=0.7,  # Known working temperature
                        num_predict=320,  # Increased for more detailed responses
                    ):
                        analysis_text += chunk
                        analysis_chunks.append(chunk)
                        chunk_count += 1
                        
                        # Early termination if we have enough content and it ends with punctuation
                        if chunk_count > 5 and len(analysis_text) > 100 and analysis_text.strip()[-1] in '.!?':
                            break
                            
            except asyncio.TimeoutError:
                # Even with timeout, return partial results if we have any
                if analysis_text:
                    pass  # Continue with partial results
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Vision analysis timed out after 40 seconds. The image may be complex or the model busy. Try again in a moment.",
                    )
            except Exception as e:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Vision analysis failed: {str(e)}",
                )

            # If we got no meaningful result, try a simpler approach
            if not analysis_text.strip():
                # Try a simpler, faster question
                simple_question = "What's in this image? Very brief."
                analysis_text = ""
                try:
                    async with asyncio.timeout(20):
                        async for chunk in vision.generate(
                            prompt=simple_question,
                            images=[b64],
                            stream=True,
                            temperature=0.8,  # Higher temp for faster response
                            num_predict=128,  # Shorter output
                        ):
                            analysis_text += chunk
                except:
                    pass  # If this fails too, we'll return what we have

            result_data = {
                "analysis": analysis_text.strip() if analysis_text else "Unable to analyze screenshot.",
                "screenshot_path": path,
                "chunks_received": chunk_count,
            }
            
            # Cache the result
            manager.cache_analysis(path, question, result_data)
            
            return ToolResult(success=True, data=result_data)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
