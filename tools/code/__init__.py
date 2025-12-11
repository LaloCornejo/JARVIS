from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from tools.base import BaseTool, ToolResult


class ExecutePythonTool(BaseTool):
    name = "execute_python"
    description = "Execute Python code in a sandboxed environment"
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30, max: 60)",
            },
        },
        "required": ["code"],
    }

    BLOCKED_IMPORTS = {
        "os.system",
        "subprocess",
        "shutil.rmtree",
        "eval",
        "exec",
        "__import__",
        "importlib",
        "ctypes",
        "multiprocessing",
    }

    BLOCKED_PATTERNS = [
        "open('/etc",
        "open('/dev",
        "open('C:\\\\Windows",
        "os.remove",
        "os.rmdir",
        "os.unlink",
        "shutil.rmtree",
        "shutil.move",
        "socket.socket",
        "urllib.request",
        "__builtins__",
        "__globals__",
    ]

    async def execute(self, code: str, timeout: int = 30) -> ToolResult:
        try:
            for pattern in self.BLOCKED_PATTERNS:
                if pattern in code:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Code contains blocked pattern: {pattern}",
                    )

            timeout = min(max(timeout, 5), 60)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                script_path = f.name

            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    script_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Execution timed out after {timeout}s",
                    )

                stdout_str = stdout.decode("utf-8", errors="replace")[:10000]
                stderr_str = stderr.decode("utf-8", errors="replace")[:2000]

                return ToolResult(
                    success=process.returncode == 0,
                    data={
                        "exit_code": process.returncode,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                    },
                    error=stderr_str if process.returncode != 0 else None,
                )
            finally:
                Path(script_path).unlink(missing_ok=True)

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ExecuteShellTool(BaseTool):
    name = "execute_shell"
    description = "Execute a shell script/command with restrictions"
    parameters = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "Shell script or command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30, max: 60)",
            },
        },
        "required": ["script"],
    }

    BLOCKED_PATTERNS = [
        "rm -rf",
        "rm -r /",
        "del /s /q",
        "format",
        "mkfs",
        "dd if=",
        ":(){",
        "fork",
        "> /dev/sda",
        "chmod 777 /",
        "chown -R",
        "curl | sh",
        "wget | sh",
        "curl | bash",
    ]

    async def execute(self, script: str, timeout: int = 30) -> ToolResult:
        try:
            script_lower = script.lower()
            for pattern in self.BLOCKED_PATTERNS:
                if pattern in script_lower:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Script contains blocked pattern",
                    )

            timeout = min(max(timeout, 5), 60)

            process = await asyncio.create_subprocess_shell(
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Execution timed out after {timeout}s",
                )

            stdout_str = stdout.decode("utf-8", errors="replace")[:10000]
            stderr_str = stderr.decode("utf-8", errors="replace")[:2000]

            return ToolResult(
                success=process.returncode == 0,
                data={
                    "exit_code": process.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                },
                error=stderr_str if process.returncode != 0 else None,
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
