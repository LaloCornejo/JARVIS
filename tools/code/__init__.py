from __future__ import annotations

import asyncio

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

    async def execute(self, code: str, timeout: int = 30) -> ToolResult:
        import json
        import os
        import platform
        import subprocess
        from pathlib import Path

        try:
            timeout = min(max(timeout, 5), 60)

            # Determine executable extension based on platform
            exe_ext = ".exe" if platform.system() == "Windows" else ""

            # Try release build first, then debug
            tools_dir = Path(__file__).parent.parent
            release_path = tools_dir / "target" / "release" / f"execute_python{exe_ext}"
            debug_path = tools_dir / "target" / "debug" / f"execute_python{exe_ext}"

            binary_path = None
            if release_path.exists():
                binary_path = str(release_path)
            elif debug_path.exists():
                binary_path = str(debug_path)

            if binary_path:
                # Use Rust binary if available
                result = subprocess.run(
                    [binary_path, code, str(timeout)],
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5,
                )

                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    success = data["exit_code"] == 0
                    return ToolResult(
                        success=success,
                        data={
                            "exit_code": data["exit_code"],
                            "stdout": data["stdout"],
                            "stderr": data["stderr"],
                        },
                        error=data["stderr"] if not success else None,
                    )
                else:
                    return ToolResult(success=False, data=None, error=result.stderr)
            else:
                # Fallback: Execute Python code directly using Python interpreter
                import io
                import sys
                import traceback

                # Create restricted globals for safer execution
                safe_globals = {
                    "__builtins__": {
                        "len": len,
                        "range": range,
                        "enumerate": enumerate,
                        "zip": zip,
                        "map": map,
                        "filter": filter,
                        "sum": sum,
                        "min": min,
                        "max": max,
                        "abs": abs,
                        "round": round,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "set": set,
                        "print": print,
                        "sorted": sorted,
                        "reversed": reversed,
                        "isinstance": isinstance,
                        "hasattr": hasattr,
                        "getattr": getattr,
                        "setattr": setattr,
                        "type": type,
                        "help": help,
                        "dir": dir,
                        "vars": vars,
                        "locals": locals,
                        "globals": globals,
                        "__import__": __import__,
                    }
                }
                safe_locals = {}

                # Capture stdout/stderr
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                try:
                    # Execute with timeout using signal or threading
                    import threading

                    result_container = {}

                    def execute_code():
                        try:
                            exec(code, safe_globals, safe_locals)
                            result_container["success"] = True
                        except Exception:
                            result_container["success"] = False
                            result_container["error"] = traceback.format_exc()

                    thread = threading.Thread(target=execute_code)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=timeout)

                    if thread.is_alive():
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Code execution timed out after {timeout}s",
                        )

                    if result_container.get("success"):
                        return ToolResult(
                            success=True,
                            data={
                                "exit_code": 0,
                                "stdout": stdout_capture.getvalue(),
                                "stderr": stderr_capture.getvalue(),
                            },
                        )
                    else:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=result_container.get("error", "Unknown error"),
                        )
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

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
