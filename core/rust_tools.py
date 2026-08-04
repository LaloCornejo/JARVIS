"""Python wrapper for high-performance JARVIS Rust tools"""

import asyncio
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class RustToolError(Exception):
    """Exception raised when a Rust tool fails"""

    pass


class RustToolsClient:
    """Client for calling high-performance JARVIS Rust tools"""

    def __init__(self):
        # Find the rust-tools directory relative to this file
        self.rust_tools_dir = Path(__file__).parent.parent / "rust-tools"
        self.release_dir = self.rust_tools_dir / "target" / "release"
        self.debug_dir = self.rust_tools_dir / "target" / "debug"

        # Determine executable extension based on platform
        self.exe_ext = ".exe" if platform.system() == "Windows" else ""

        # Check for tools in release first, then debug
        self.tools_available = self._check_tools_available()

    def _check_tools_available(self) -> bool:
        """Check if Rust tools are available without raising an error."""
        required_tools = [f"jarvis-file-processor{self.exe_ext}"]

        for tool in required_tools:
            if (self.release_dir / tool).exists():
                return True
            if (self.debug_dir / tool).exists():
                return True
        return False

    def _get_tool_path(self, tool_name: str) -> Path:
        """Get the path to a Rust tool binary."""
        # Try release first, then debug
        release_path = self.release_dir / f"{tool_name}{self.exe_ext}"
        debug_path = self.debug_dir / f"{tool_name}{self.exe_ext}"

        if release_path.exists():
            return release_path
        if debug_path.exists():
            return debug_path

        raise FileNotFoundError(
            f"Rust tool '{tool_name}' not found. "
            f"Please build Rust tools first with 'cargo build --release'"
        )

    def line_count(self, path: str) -> Dict[str, Any]:
        """
        Count lines in a file using the high-performance Rust tool.

        Args:
            path: Path to the file to count lines in

        Returns:
            Dictionary with line count information

        Raises:
            RustToolError: If the Rust tool fails
        """
        try:
            tool_path = self._get_tool_path("jarvis-file-processor")
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}

        cmd = [str(tool_path), "line-count", path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return json.loads(result.stdout.strip())
            else:
                error_msg = f"Rust file processor failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                raise RustToolError(error_msg)

        except subprocess.TimeoutExpired:
            raise RustToolError("Rust file processor timed out")
        except json.JSONDecodeError as e:
            raise RustToolError(f"Failed to parse Rust tool output: {str(e)}")
        except Exception as e:
            raise RustToolError(f"Failed to run Rust file processor: {str(e)}")

    def file_search(
        self, pattern: str, path: str, limit: int = 100, ignore_case: bool = False
    ) -> Dict[str, Any]:
        """
        Search for patterns in files using the high-performance Rust tool.

        Args:
            pattern: Regex pattern to search for
            path: Path to search in (file or directory)
            limit: Maximum number of results to return
            ignore_case: Whether to perform case-insensitive search

        Returns:
            Dictionary with search results

        Raises:
            RustToolError: If the Rust tool fails
        """
        try:
            tool_path = self._get_tool_path("jarvis-file-processor")
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}

        cmd = [
            str(tool_path),
            "search",
            pattern,
            path,
            "--limit",
            str(limit),
        ]
        if ignore_case:
            cmd.append("--ignore-case")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return json.loads(result.stdout.strip())
            else:
                error_msg = f"Rust file processor failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                raise RustToolError(error_msg)

        except subprocess.TimeoutExpired:
            raise RustToolError("Rust file processor timed out")
        except json.JSONDecodeError as e:
            raise RustToolError(f"Failed to parse Rust tool output: {str(e)}")
        except Exception as e:
            raise RustToolError(f"Failed to run Rust file processor: {str(e)}")

    async def async_file_search(
        self, pattern: str, path: str, limit: int = 100, ignore_case: bool = False
    ) -> Dict[str, Any]:
        """Async version of file_search that runs in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.file_search, pattern, path, limit, ignore_case)

    async def async_line_count(self, path: str) -> Dict[str, Any]:
        """Async version of line_count that runs in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.line_count, path)

    async def async_extract_data(self, path: str, pattern: str) -> Dict[str, Any]:
        """Async version of extract_data that runs in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_data, path, pattern)

    def extract_data(self, path: str, pattern: str) -> Dict[str, Any]:
        """
        Extract structured data from files using regex patterns.

        Args:
            path: Path to the file
            pattern: Regex pattern with capture groups

        Returns:
            Dictionary with extracted data

        Raises:
            RustToolError: If the Rust tool fails
        """
        try:
            tool_path = self._get_tool_path("jarvis-file-processor")
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}

        cmd = [str(tool_path), "extract", path, pattern]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return json.loads(result.stdout.strip())
            else:
                error_msg = f"Rust file processor failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                raise RustToolError(error_msg)

        except subprocess.TimeoutExpired:
            raise RustToolError("Rust file processor timed out")
        except json.JSONDecodeError as e:
            raise RustToolError(f"Failed to parse Rust tool output: {str(e)}")
        except Exception as e:
            raise RustToolError(f"Failed to run Rust file processor: {str(e)}")


# Singleton instance
_rust_tools_client: Optional[RustToolsClient] = None


def get_rust_tools_client() -> RustToolsClient:
    """Get the singleton Rust tools client."""
    global _rust_tools_client
    if _rust_tools_client is None:
        _rust_tools_client = RustToolsClient()
    return _rust_tools_client


# Convenience functions
def file_search(
    pattern: str, path: str, limit: int = 100, ignore_case: bool = False
) -> Dict[str, Any]:
    """Convenience function to search files."""
    try:
        client = get_rust_tools_client()
        return client.file_search(pattern, path, limit, ignore_case)
    except (FileNotFoundError, RustToolError) as e:
        return {"success": False, "error": str(e)}


async def async_file_search(
    pattern: str, path: str, limit: int = 100, ignore_case: bool = False
) -> Dict[str, Any]:
    """Async convenience function to search files."""
    try:
        client = get_rust_tools_client()
        return await client.async_file_search(pattern, path, limit, ignore_case)
    except (FileNotFoundError, RustToolError) as e:
        return {"success": False, "error": str(e)}


def line_count(path: str) -> Dict[str, Any]:
    """Convenience function to count lines."""
    try:
        client = get_rust_tools_client()
        return client.line_count(path)
    except (FileNotFoundError, RustToolError) as e:
        return {"success": False, "error": str(e)}


async def async_line_count(path: str) -> Dict[str, Any]:
    """Async convenience function to count lines."""
    try:
        client = get_rust_tools_client()
        return await client.async_line_count(path)
    except (FileNotFoundError, RustToolError) as e:
        return {"success": False, "error": str(e)}


def extract_data(path: str, pattern: str) -> Dict[str, Any]:
    """Convenience function to extract data."""
    try:
        client = get_rust_tools_client()
        return client.extract_data(path, pattern)
    except (FileNotFoundError, RustToolError) as e:
        return {"success": False, "error": str(e)}


async def async_extract_data(path: str, pattern: str) -> Dict[str, Any]:
    """Async convenience function to extract data."""
    try:
        client = get_rust_tools_client()
        return await client.async_extract_data(path, pattern)
    except (FileNotFoundError, RustToolError) as e:
        return {"success": False, "error": str(e)}
