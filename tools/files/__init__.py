from __future__ import annotations

from pathlib import Path

from tools.base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read (default: 500)",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, max_lines: int = 500) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            if not file_path.exists():
                return ToolResult(success=False, data=None, error=f"File not found: {path}")
            if not file_path.is_file():
                return ToolResult(success=False, data=None, error=f"Not a file: {path}")

            with open(file_path, encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)

            content = "".join(lines)
            return ToolResult(
                success=True,
                data={
                    "path": str(file_path),
                    "content": content,
                    "lines_read": len(lines),
                    "truncated": len(lines) == max_lines,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file (creates or overwrites)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "append": {
                "type": "boolean",
                "description": "Append to file instead of overwriting (default: false)",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, append: bool = False) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                success=True,
                data={
                    "path": str(file_path),
                    "bytes_written": len(content.encode("utf-8")),
                    "mode": "append" if append else "write",
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List files and directories in a path"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current directory)",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., '*.py')",
            },
        },
        "required": [],
    }

    async def execute(self, path: str = ".", pattern: str | None = None) -> ToolResult:
        try:
            dir_path = Path(path).expanduser().resolve()
            if not dir_path.exists():
                return ToolResult(success=False, data=None, error=f"Path not found: {path}")
            if not dir_path.is_dir():
                return ToolResult(success=False, data=None, error=f"Not a directory: {path}")

            if pattern:
                items = list(dir_path.glob(pattern))
            else:
                items = list(dir_path.iterdir())

            entries = []
            for item in sorted(items)[:100]:
                entry = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "path": str(item),
                }
                if item.is_file():
                    entry["size"] = item.stat().st_size
                entries.append(entry)

            return ToolResult(
                success=True,
                data={
                    "path": str(dir_path),
                    "count": len(entries),
                    "entries": entries,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = "Search for files by name pattern recursively"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to search in",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g., '**/*.py' for all Python files)",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default: 50)",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, pattern: str, path: str = ".", max_results: int = 50) -> ToolResult:
        try:
            dir_path = Path(path).expanduser().resolve()
            if not dir_path.exists():
                return ToolResult(success=False, data=None, error=f"Path not found: {path}")

            matches = []
            for match in dir_path.glob(pattern):
                if len(matches) >= max_results:
                    break
                matches.append(
                    {
                        "path": str(match),
                        "name": match.name,
                        "type": "directory" if match.is_dir() else "file",
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "pattern": pattern,
                    "search_path": str(dir_path),
                    "count": len(matches),
                    "matches": matches,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class FileInfoTool(BaseTool):
    name = "file_info"
    description = "Get detailed information about a file or directory"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            if not file_path.exists():
                return ToolResult(success=False, data=None, error=f"Path not found: {path}")

            stat = file_path.stat()
            info = {
                "path": str(file_path),
                "name": file_path.name,
                "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
            }

            if file_path.is_file():
                info["extension"] = file_path.suffix

            return ToolResult(success=True, data=info)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
