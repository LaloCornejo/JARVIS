"""Python wrapper for high-performance JARVIS Rust tools"""

import json
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

        # Verify tools exist
        required_tools = ["jarvis-file-processor.exe", "jarvis-model-manager.exe"]

        for tool in required_tools:
            if not (self.release_dir / tool).exists():
                raise FileNotFoundError(f"{tool} not found. Please build Rust tools first.")

    def file_search(
        self, pattern: str, path: str, limit: int = 100, ignore_case: bool = False
    ) -> Dict[str, Any]:
        """
        Count lines in a file using the high-performance Rust tool.

        Args:
            pattern: Regex pattern to search for
            path: Path to search in (file or directory)
            limit: Maximum number of results to return

            ignore_case: Whether to perform case-insensitive search

            Returns:
            Dictionary with line count information

            Raises:
            RustToolError: If the Rust tool fails
        """
        cmd = [str(self.release_dir / "jarvis-file-processor.exe"), "line-count", path]

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
        cmd = [str(self.release_dir / "jarvis-file-processor.exe"), "extract", path, pattern]

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
    client = get_rust_tools_client()
    return client.file_search(pattern, path, limit, ignore_case)


async def async_file_search(
    pattern: str, path: str, limit: int = 100, ignore_case: bool = False
) -> Dict[str, Any]:
    """Async convenience function to search files."""
    client = get_rust_tools_client()
    return await client.async_file_search(pattern, path, limit, ignore_case)


def line_count(path: str) -> Dict[str, Any]:
    """Convenience function to count lines."""
    client = get_rust_tools_client()
    return client.line_count(path)


def extract_data(path: str, pattern: str) -> Dict[str, Any]:
    """Convenience function to extract data."""
    client = get_rust_tools_client()
    return client.extract_data(path, pattern)
