"""Screen context extraction via Windows UI Automation.

Extract rich context from active window without screenshots:
- Browser: URL + page title from address bar
- IDE: open file from editor tab
- Terminal: current directory from title
- Generic: window title + visible text elements

Layer 2 in proactivity pipeline (after app name, before OCR/vision).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.proactive.screen_context")

try:
    import uiautomation as auto

    HAS_UIA = True
except ImportError:
    HAS_UIA = False

# Browser class names (Win32 / UWP)
BROWSER_CLASSES = {
    "Chrome_WidgetWin_1": "chrome",
    "Chrome_WidgetWin_0": "chrome",
    "MozillaWindowClass": "firefox",
    "MozillaDialogClass": "firefox",
    "ApplicationFrameWindow": "edge",  # UWP edge
    "CabinetWClass": "edge",  # Win32 edge
}

# IDE class names
IDE_CLASSES = {
    "Chrome_WidgetWin_1": "vscode",  # VS Code, Cursor, Windsurf use same
    "HwndWrapper[]": "vscode",
    "INTELLIJ_WINDOW": "intellij",
    "SunAwtFrame": "intellij",
    "Qt5QWindowIcon": "pycharm",
}


@dataclass
class ScreenContext:
    """Rich context from active window."""

    app_name: str = ""
    window_title: str = ""
    url: str | None = None
    page_title: str | None = None  # Browser tab title (clean)
    open_file: str | None = None  # IDE open file
    open_folder: str | None = None  # IDE project folder
    visible_texts: list[str] = field(default_factory=list)
    control_type: str = ""  # UIA control type
    extracted_at: datetime = field(default_factory=datetime.now)

    @property
    def is_browser(self) -> bool:
        return self.url is not None

    @property
    def is_ide(self) -> bool:
        return self.open_file is not None

    def has_content(self) -> bool:
        return bool(self.app_name or self.window_title)

    def diff_key(self) -> str:
        """Unique key for change detection."""
        return f"{self.app_name}|{self.window_title}|{self.url or ''}|{self.open_file or ''}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "window_title": self.window_title,
            "url": self.url,
            "page_title": self.page_title,
            "open_file": self.open_file,
            "open_folder": self.open_folder,
            "visible_texts": self.visible_texts[:10],
            "is_browser": self.is_browser,
            "is_ide": self.is_ide,
        }


class ScreenContextExtractor:
    """Extract rich context from active window via UI Automation.

    Poll interval suggested: 10-30s. Uses uiautomation to read UI element
    tree from active window — no screenshots, no image processing.
    """

    def __init__(self, poll_interval: float = 15.0):
        self.poll_interval = poll_interval
        self._last_context: ScreenContext | None = None
        self._uia_available = HAS_UIA

        # App detection via executable path
        self._exe_cache: dict[str, str | None] = {}

    # ── public API ──────────────────────────────────────────────

    async def extract(self) -> ScreenContext:
        """Extract rich context from current foreground window."""
        if not self._uia_available:
            return ScreenContext()

        loop = asyncio.get_event_loop()
        try:
            ctx = await loop.run_in_executor(None, self._extract_sync)
            return ctx
        except Exception:
            log.warning("UI Automation extract failed", exc_info=True)
            return ScreenContext()

    async def poll_context(self) -> ScreenContext | None:
        """Poll for context. Returns ScreenContext only if changed."""
        ctx = await self.extract()
        if not ctx.has_content():
            return None

        # Change detection
        if self._last_context is None:
            self._last_context = ctx
            return ctx

        if ctx.diff_key() != self._last_context.diff_key():
            self._last_context = ctx
            return ctx

        return None

    def get_last_context(self) -> ScreenContext | None:
        return self._last_context

    def force_refresh(self) -> None:
        self._last_context = None

    # ── sync extraction (runs in threadpool) ────────────────────

    def _extract_sync(self) -> ScreenContext:
        try:
            fg = auto.GetForegroundControl()
        except Exception:
            return ScreenContext()

        if not fg or not fg.Name:
            return ScreenContext()

        window_title = fg.Name
        class_name = getattr(fg, "ClassName", "")
        control_type = getattr(fg, "ControlTypeName", "")

        ctx = ScreenContext(
            window_title=window_title,
            control_type=control_type,
        )

        # Detect app from class name
        ctx.app_name = self._detect_app(class_name, fg)

        # Rich extraction per app type
        try:
            if ctx.app_name in ("chrome", "firefox", "edge", "brave"):
                self._extract_browser(fg, ctx, class_name)
            elif ctx.app_name in ("vscode", "cursor", "windsurf"):
                self._extract_ide(fg, ctx)
            elif ctx.app_name in ("terminal", "powershell", "cmd", "wsl"):
                self._extract_terminal(fg, ctx)
            else:
                self._extract_generic(fg, ctx)
        except Exception:
            log.debug("Rich extraction failed, fallback to generic", exc_info=True)
            self._extract_generic(fg, ctx)

        return ctx

    # ── app detection ───────────────────────────────────────────

    def _detect_app(self, class_name: str, fg: auto.Control) -> str:
        """Detect app from window class and executable path."""
        class_lower = class_name.lower()

        # Browser detection
        for cls, name in BROWSER_CLASSES.items():
            if cls.lower() in class_lower or class_lower.startswith(cls.lower()):
                return name

        # IDE detection — use executable name for more precision
        exe = self._get_exe_name(fg)
        if exe:
            exe_lower = exe.lower()
            ide_map = {
                "code": "vscode",
                "cursor": "cursor",
                "windsurf": "windsurf",
                "idea": "intellij",
                "pycharm": "pycharm",
            }
            for key, val in ide_map.items():
                if key in exe_lower:
                    return val

        # Terminal detection
        if "windowsterminal" in class_lower or "cascadia" in class_lower:
            return "terminal"
        if "consolewindow" in class_lower or class_name == "ConsoleWindowClass":
            return "terminal"
        if "wsl" in class_lower:
            return "wsl"

        # Fallback: extract from window title
        title = fg.Name.lower()
        terminal_keywords = ["terminal", "powershell", "cmd", "command prompt", "wsl:"]
        for kw in terminal_keywords:
            if kw in title:
                return kw.replace(":", "")

        return "unknown"

    def _get_exe_name(self, control: auto.Control) -> str | None:
        """Get executable name from window process."""
        try:
            return control.GetWindowProcessName()
        except Exception:
            return None

    # ── browser extraction ──────────────────────────────────────

    def _extract_browser(self, fg: auto.Control, ctx: ScreenContext, class_name: str) -> None:
        """Extract URL + page title from browser address bar."""
        # Chrome/Edge/Brave: URL in address bar (EditControl or ToolBar > Edit)
        if ctx.app_name in ("chrome", "edge", "brave"):
            try:
                # Chrome stores URL in omnibox (EditControl inside address bar)
                # Search patterns: direct EditControl child or nested in ToolBar
                for depth in range(1, 5):
                    omni = fg.FindFirstControl(
                        auto.EditControl,
                        lambda c, d=depth: c.ControlType == auto.ControlType.EditControl
                        and "://" in (c.Name or ""),
                        searchDepth=depth,
                    )
                    if omni and omni.Name:
                        url = omni.Name.strip()
                        if url.startswith("http"):
                            ctx.url = url
                        break
            except Exception:
                pass

            # Fallback: try to find address bar via AutomationId
            if not ctx.url:
                try:
                    omni = fg.FindFirstControl(
                        auto.EditControl,
                        lambda c: "omnibox" in (c.AutomationId or "").lower(),
                        searchDepth=4,
                    )
                    if omni and omni.Name:
                        ctx.url = omni.Name.strip()
                except Exception:
                    pass

        # Firefox: URL in address bar (has specific class)
        elif ctx.app_name == "firefox":
            try:
                urlbar = fg.FindFirstControl(
                    auto.EditControl,
                    lambda c: "urlbar" in (c.AutomationId or "").lower(),
                    searchDepth=4,
                )
                if urlbar and urlbar.Name:
                    ctx.url = urlbar.Name.strip()
            except Exception:
                pass

        # Extract clean page title from window title (strip app name suffix)
        if ctx.window_title:
            separators = [" - Google Chrome", " - Mozilla Firefox", " - Microsoft Edge", " - Brave"]
            for sep in separators:
                if ctx.window_title.endswith(sep):
                    ctx.page_title = ctx.window_title[: -len(sep)]
                    break
            if not ctx.page_title:
                ctx.page_title = ctx.window_title

    # ── IDE extraction ──────────────────────────────────────────

    def _extract_ide(self, fg: auto.Control, ctx: ScreenContext) -> None:
        """Extract open file + project from IDE."""
        # VS Code / Cursor / Windsurf: tab bar has open file
        # Pattern: find TabItem controls in window
        try:
            # Find first tab item (open file)
            tab = fg.FindFirstControl(
                auto.TabItemControl,
                searchDepth=5,
            )
            if tab and tab.Name:
                ctx.open_file = tab.Name.strip()

                # Clean file name (remove decoration chars)
                if ctx.open_file and ctx.open_file.startswith(("● ", "✕ ")):
                    ctx.open_file = ctx.open_file[2:]
        except Exception:
            pass

        # Extract folder name from window title
        # VS Code format: "file.py — project-folder [folder] — Visual Studio Code"
        if ctx.window_title:
            parts = ctx.window_title.split(" — ")
            if len(parts) >= 2:
                ctx.open_folder = parts[1].strip()
                if ctx.open_folder.endswith(" [folder]"):
                    ctx.open_folder = ctx.open_folder[: -9]
                elif ctx.open_folder.endswith(" [Administrator]"):
                    ctx.open_folder = ctx.open_folder[: -17]

        # If no tab found, try extracting from window title directly
        if not ctx.open_file and ctx.window_title:
            parts = ctx.window_title.split(" — ")
            first = parts[0].strip()
            # If first part looks like a file (has extension)
            if "." in first and not first.startswith("http"):
                ctx.open_file = first

    # ── terminal extraction ─────────────────────────────────────

    def _extract_terminal(self, fg: auto.Control, ctx: ScreenContext) -> None:
        """Extract path info from terminal title."""
        # Terminal window title often has current path
        # Format: "user@host: /path/to/dir" or "path  - PowerShell"
        title = ctx.window_title or ""

        # Try to extract current directory from title
        path_indicators = [" ~/", " /", ":\\", " ~\\"]
        for indicator in path_indicators:
            if indicator in title:
                # Take first path-like segment
                segments = title.split(" - ")
                for seg in segments:
                    seg = seg.strip()
                    if indicator in seg or "\\" in seg or "/" in seg:
                        ctx.open_folder = seg
                        break
                break

    # ── generic extraction ──────────────────────────────────────

    def _extract_generic(self, fg: auto.Control, ctx: ScreenContext) -> None:
        """Extract visible text elements from any window."""
        try:
            texts: list[str] = []
            seen: set[str] = set()

            def collect_text(control: auto.Control, depth: int = 0) -> None:
                if depth > 4:
                    return
                try:
                    name = control.Name
                    if name and name not in seen and len(name) > 2:
                        # Filter UI noise (single chars, empty, metadata)
                        if not name.startswith(("__", "{")) and len(name) < 200:
                            seen.add(name)
                            ct = control.ControlTypeName
                            if ct not in ("TitleBarControl", "MenuBarControl", "ButtonControl"):
                                texts.append(name)
                except Exception:
                    pass
                try:
                    for child in control.GetChildren():
                        collect_text(child, depth + 1)
                except Exception:
                    pass

            collect_text(fg)
            ctx.visible_texts = texts[:15]  # Keep top 15

        except Exception:
            pass


# Global instance
_extractor: ScreenContextExtractor | None = None


def get_screen_context_extractor() -> ScreenContextExtractor:
    """Get global ScreenContextExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ScreenContextExtractor()
    return _extractor


__all__ = [
    "ScreenContext",
    "ScreenContextExtractor",
    "get_screen_context_extractor",
]
