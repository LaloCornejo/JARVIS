from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult


class ObsidianClient:
    def __init__(self, vault_path: str | None = None):
        self.vault_path = Path(vault_path or os.environ.get("OBSIDIAN_VAULT", ""))

    def _ensure_vault(self) -> bool:
        return self.vault_path.exists() and self.vault_path.is_dir()

    def list_notes(self, folder: str = "", pattern: str = "*.md") -> list[dict[str, Any]]:
        if not self._ensure_vault():
            return []

        search_path = self.vault_path / folder if folder else self.vault_path
        notes = []
        for md_file in search_path.rglob(pattern):
            if md_file.is_file() and not md_file.name.startswith("."):
                rel_path = md_file.relative_to(self.vault_path)
                stat = md_file.stat()
                notes.append(
                    {
                        "name": md_file.stem,
                        "path": str(rel_path),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "size": stat.st_size,
                    }
                )
        return sorted(notes, key=lambda x: x["modified"], reverse=True)

    def read_note(self, path: str) -> str | None:
        if not self._ensure_vault():
            return None

        note_path = self.vault_path / path
        if not path.endswith(".md"):
            note_path = note_path.with_suffix(".md")

        if note_path.exists():
            return note_path.read_text(encoding="utf-8")
        return None

    def write_note(self, path: str, content: str) -> bool:
        if not self._ensure_vault():
            return False

        note_path = self.vault_path / path
        if not path.endswith(".md"):
            note_path = note_path.with_suffix(".md")

        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(content, encoding="utf-8")
        return True

    def append_note(self, path: str, content: str) -> bool:
        if not self._ensure_vault():
            return False

        note_path = self.vault_path / path
        if not path.endswith(".md"):
            note_path = note_path.with_suffix(".md")

        existing = ""
        if note_path.exists():
            existing = note_path.read_text(encoding="utf-8")

        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(existing + "\n" + content, encoding="utf-8")
        return True

    def search_notes(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self._ensure_vault():
            return []

        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for md_file in self.vault_path.rglob("*.md"):
            if md_file.is_file() and not md_file.name.startswith("."):
                try:
                    content = md_file.read_text(encoding="utf-8")
                    matches = list(pattern.finditer(content))
                    if matches:
                        rel_path = md_file.relative_to(self.vault_path)
                        first_match = matches[0]
                        start = max(0, first_match.start() - 50)
                        end = min(len(content), first_match.end() + 50)
                        snippet = content[start:end].strip()
                        results.append(
                            {
                                "name": md_file.stem,
                                "path": str(rel_path),
                                "matches": len(matches),
                                "snippet": snippet,
                            }
                        )
                except Exception:
                    continue

        results.sort(key=lambda x: x["matches"], reverse=True)
        return results[:limit]

    def get_daily_note(self, date: datetime | None = None) -> str:
        if date is None:
            date = datetime.now()
        return f"Daily Notes/{date.strftime('%Y-%m-%d')}.md"

    def create_daily_note(self, content: str = "", date: datetime | None = None) -> bool:
        path = self.get_daily_note(date)
        if date is None:
            date = datetime.now()

        template = f"""# {date.strftime("%Y-%m-%d")}

## Tasks
- [ ]

## Notes
{content}

## Journal

"""
        return self.write_note(path, template)

    def get_backlinks(self, note_name: str) -> list[dict[str, Any]]:
        if not self._ensure_vault():
            return []

        backlinks = []
        link_pattern = re.compile(rf"\[\[{re.escape(note_name)}(\|[^\]]+)?\]\]")

        for md_file in self.vault_path.rglob("*.md"):
            if md_file.stem == note_name:
                continue
            if md_file.is_file() and not md_file.name.startswith("."):
                try:
                    content = md_file.read_text(encoding="utf-8")
                    if link_pattern.search(content):
                        rel_path = md_file.relative_to(self.vault_path)
                        backlinks.append(
                            {
                                "name": md_file.stem,
                                "path": str(rel_path),
                            }
                        )
                except Exception:
                    continue

        return backlinks


_obsidian_client: ObsidianClient | None = None


def get_obsidian_client() -> ObsidianClient:
    global _obsidian_client
    if _obsidian_client is None:
        _obsidian_client = ObsidianClient()
    return _obsidian_client


class ObsidianListNotesTool(BaseTool):
    name = "obsidian_list_notes"
    description = "List notes in Obsidian vault"
    parameters = {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Subfolder to list (optional)"},
            "limit": {"type": "integer", "description": "Max notes to return"},
        },
        "required": [],
    }

    async def execute(self, folder: str = "", limit: int = 20) -> ToolResult:
        client = get_obsidian_client()
        if not client._ensure_vault():
            return ToolResult(
                success=False,
                data=None,
                error="Obsidian vault not configured. Set OBSIDIAN_VAULT env var.",
            )

        try:
            notes = client.list_notes(folder)[:limit]
            return ToolResult(success=True, data=notes)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ObsidianReadNoteTool(BaseTool):
    name = "obsidian_read_note"
    description = "Read content of an Obsidian note"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Note path relative to vault"},
        },
        "required": ["path"],
    }

    async def execute(self, path: str) -> ToolResult:
        client = get_obsidian_client()
        if not client._ensure_vault():
            return ToolResult(
                success=False,
                data=None,
                error="Obsidian vault not configured",
            )

        try:
            content = client.read_note(path)
            if content is not None:
                return ToolResult(success=True, data=content)
            return ToolResult(success=False, data=None, error="Note not found")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ObsidianWriteNoteTool(BaseTool):
    name = "obsidian_write_note"
    description = "Create or overwrite an Obsidian note"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Note path relative to vault"},
            "content": {"type": "string", "description": "Note content in markdown"},
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str) -> ToolResult:
        client = get_obsidian_client()
        if not client._ensure_vault():
            return ToolResult(
                success=False,
                data=None,
                error="Obsidian vault not configured",
            )

        try:
            success = client.write_note(path, content)
            if success:
                return ToolResult(success=True, data=f"Saved: {path}")
            return ToolResult(success=False, data=None, error="Failed to write note")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ObsidianAppendNoteTool(BaseTool):
    name = "obsidian_append_note"
    description = "Append content to an existing Obsidian note"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Note path relative to vault"},
            "content": {"type": "string", "description": "Content to append"},
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str) -> ToolResult:
        client = get_obsidian_client()
        if not client._ensure_vault():
            return ToolResult(
                success=False,
                data=None,
                error="Obsidian vault not configured",
            )

        try:
            success = client.append_note(path, content)
            if success:
                return ToolResult(success=True, data=f"Appended to: {path}")
            return ToolResult(success=False, data=None, error="Failed to append")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ObsidianSearchTool(BaseTool):
    name = "obsidian_search"
    description = "Search for text across Obsidian notes"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, limit: int = 10) -> ToolResult:
        client = get_obsidian_client()
        if not client._ensure_vault():
            return ToolResult(
                success=False,
                data=None,
                error="Obsidian vault not configured",
            )

        try:
            results = client.search_notes(query, limit)
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ObsidianDailyNoteTool(BaseTool):
    name = "obsidian_daily_note"
    description = "Create or append to today's daily note"
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Content to add"},
            "append": {
                "type": "boolean",
                "description": "Append to existing (true) or create new (false)",
            },
        },
        "required": [],
    }

    async def execute(self, content: str = "", append: bool = True) -> ToolResult:
        client = get_obsidian_client()
        if not client._ensure_vault():
            return ToolResult(
                success=False,
                data=None,
                error="Obsidian vault not configured",
            )

        try:
            path = client.get_daily_note()
            if append and client.read_note(path):
                success = client.append_note(path, content)
            else:
                success = client.create_daily_note(content)

            if success:
                return ToolResult(success=True, data=f"Updated daily note: {path}")
            return ToolResult(success=False, data=None, error="Failed")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
