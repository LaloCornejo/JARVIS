from __future__ import annotations

import asyncio
import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult


@dataclass
class ClipboardEntry:
    content: str
    content_type: str
    timestamp: str
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipboardEntry:
        return cls(
            content=data["content"],
            content_type=data["content_type"],
            timestamp=data["timestamp"],
            source=data.get("source", ""),
        )


@dataclass
class ClipboardHistory:
    entries: list[ClipboardEntry] = field(default_factory=list)
    max_entries: int = 100

    def add(self, entry: ClipboardEntry) -> None:
        if self.entries and self.entries[-1].content == entry.content:
            return
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def search(self, query: str) -> list[ClipboardEntry]:
        query_lower = query.lower()
        return [e for e in self.entries if query_lower in e.content.lower()]

    def get_recent(self, count: int = 10) -> list[ClipboardEntry]:
        return self.entries[-count:][::-1]

    def clear(self) -> None:
        self.entries = []


class ClipboardManager:
    def __init__(self, history_file: str | None = None):
        self.history_file = Path(
            history_file
            or os.environ.get(
                "CLIPBOARD_HISTORY_FILE",
                os.path.expanduser("~/.jarvis/clipboard_history.json"),
            )
        )
        self.history = ClipboardHistory()
        self._load_history()

    def _load_history(self) -> None:
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    self.history.entries = [
                        ClipboardEntry.from_dict(e) for e in data.get("entries", [])
                    ]
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_history(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump(
                {"entries": [e.to_dict() for e in self.history.entries]},
                f,
                indent=2,
            )

    async def _run_powershell(self, script: str) -> tuple[bool, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                "powershell",
                "-NoProfile",
                "-Command",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return True, stdout.decode().strip()
            return False, stderr.decode().strip()
        except Exception as e:
            return False, str(e)

    async def get_clipboard(self) -> tuple[str, str]:
        success, content = await self._run_powershell("Get-Clipboard")
        if success:
            return content, "text"
        return "", ""

    async def set_clipboard(self, content: str) -> bool:
        escaped = content.replace("'", "''")
        success, _ = await self._run_powershell(f"Set-Clipboard -Value '{escaped}'")
        if success:
            entry = ClipboardEntry(
                content=content,
                content_type="text",
                timestamp=datetime.datetime.now().isoformat(),
                source="jarvis",
            )
            self.history.add(entry)
            self._save_history()
        return success

    async def capture_current(self) -> ClipboardEntry | None:
        content, content_type = await self.get_clipboard()
        if content:
            entry = ClipboardEntry(
                content=content,
                content_type=content_type,
                timestamp=datetime.datetime.now().isoformat(),
            )
            self.history.add(entry)
            self._save_history()
            return entry
        return None

    def get_history(self, count: int = 10) -> list[ClipboardEntry]:
        return self.history.get_recent(count)

    def search_history(self, query: str) -> list[ClipboardEntry]:
        return self.history.search(query)

    def clear_history(self) -> None:
        self.history.clear()
        self._save_history()

    async def paste_from_history(self, index: int) -> bool:
        recent = self.history.get_recent(100)
        if 0 <= index < len(recent):
            return await self.set_clipboard(recent[index].content)
        return False


_clipboard_manager: ClipboardManager | None = None


def get_clipboard_manager() -> ClipboardManager:
    global _clipboard_manager
    if _clipboard_manager is None:
        _clipboard_manager = ClipboardManager()
    return _clipboard_manager


class ClipboardCopyTool(BaseTool):
    name = "clipboard_copy"
    description = "Copy text to the clipboard"
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to copy to clipboard",
            },
        },
        "required": ["text"],
    }

    async def execute(self, text: str) -> ToolResult:
        manager = get_clipboard_manager()
        try:
            success = await manager.set_clipboard(text)
            if success:
                return ToolResult(success=True, data="Text copied to clipboard")
            return ToolResult(success=False, data=None, error="Failed to copy to clipboard")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ClipboardPasteTool(BaseTool):
    name = "clipboard_paste"
    description = "Get the current clipboard content"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        manager = get_clipboard_manager()
        try:
            content, content_type = await manager.get_clipboard()
            if content:
                return ToolResult(success=True, data={"content": content, "type": content_type})
            return ToolResult(success=True, data={"content": "", "type": ""})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ClipboardHistoryTool(BaseTool):
    name = "clipboard_history"
    description = "Get recent clipboard history"
    parameters = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of entries to retrieve (default: 10)",
            },
        },
        "required": [],
    }

    async def execute(self, count: int = 10) -> ToolResult:
        manager = get_clipboard_manager()
        try:
            entries = manager.get_history(count)
            return ToolResult(success=True, data=[e.to_dict() for e in entries])
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ClipboardSearchTool(BaseTool):
    name = "clipboard_search"
    description = "Search clipboard history for matching entries"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str) -> ToolResult:
        manager = get_clipboard_manager()
        try:
            entries = manager.search_history(query)
            return ToolResult(success=True, data=[e.to_dict() for e in entries])
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ClipboardClearHistoryTool(BaseTool):
    name = "clipboard_clear_history"
    description = "Clear the clipboard history"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        manager = get_clipboard_manager()
        try:
            manager.clear_history()
            return ToolResult(success=True, data="Clipboard history cleared")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ClipboardPasteFromHistoryTool(BaseTool):
    name = "clipboard_paste_from_history"
    description = "Set clipboard content from a history entry by index (0 = most recent)"
    parameters = {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "Index in history (0 = most recent)",
            },
        },
        "required": ["index"],
    }

    async def execute(self, index: int) -> ToolResult:
        manager = get_clipboard_manager()
        try:
            success = await manager.paste_from_history(index)
            if success:
                return ToolResult(success=True, data=f"Clipboard set from history entry {index}")
            return ToolResult(success=False, data=None, error=f"Invalid history index: {index}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
