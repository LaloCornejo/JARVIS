from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import webbrowser
from pathlib import Path

from tools.base import BaseTool, ToolResult

APP_REGISTRY = {}


def load_discovered_apps():
    """Load discovered apps from the Rust tool's cache"""
    global APP_REGISTRY
    cache_file = Path(__file__).parent.parent / "app_registry_cache.json"
    APP_REGISTRY = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                # The apps are in the "apps" array
                discovered_apps = data.get("apps", [])
                # Convert to the format expected by APP_REGISTRY
                for app_data in discovered_apps:
                    app_name = app_data["name"]
                    # Use the exe_name as the key, but create a friendly name
                    exe_name = app_data["exe_name"]
                    key = exe_name.lower()
                    APP_REGISTRY[key] = {
                        "exe": app_data["exe_path"],
                        "aliases": [app_name.lower(), app_name.lower() + " app"]
                        + app_data.get("aliases", []),
                    }
        except Exception as e:
            print(f"Error loading discovered apps: {e}")
            APP_REGISTRY = {}


def find_app(query: str) -> tuple[str, str] | None:
    # Load discovered apps if available
    load_discovered_apps()

    query_lower = query.lower().strip()

    if query_lower in APP_REGISTRY:
        return query_lower, APP_REGISTRY[query_lower]["exe"]

    for name, info in APP_REGISTRY.items():
        if query_lower == info["exe"].lower():
            return name, info["exe"]
        for alias in info["aliases"]:
            if query_lower == alias.lower():
                return name, info["exe"]

    for name, info in APP_REGISTRY.items():
        if query_lower in name or name in query_lower:
            return name, info["exe"]
        if query_lower in info["exe"].lower():
            return name, info["exe"]
        for alias in info["aliases"]:
            if query_lower in alias.lower() or alias.lower() in query_lower:
                return name, info["exe"]

    return None


class RefreshAppRegistryTool(BaseTool):
    name = "refresh_app_registry"
    description = "Refresh the application registry by scanning for installed apps"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        try:
            # Run the Rust discovery tool
            tools_dir = Path(__file__).parent.parent
            discovery_exe = tools_dir / "target" / "release" / "discover_apps.exe"

            if not discovery_exe.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    error="Discovery tool not found. Please build it first with 'cargo build --release'",
                )

            process = subprocess.run(
                [str(discovery_exe)], cwd=tools_dir, capture_output=True, text=True, timeout=30
            )

            if process.returncode == 0:
                # Load the newly discovered apps
                load_discovered_apps()

                return ToolResult(
                    success=True,
                    data={
                        "message": "App registry refreshed successfully",
                        "apps_count": len(APP_REGISTRY),
                    },
                )
            else:
                return ToolResult(
                    success=False, data=None, error=f"Discovery tool failed: {process.stderr}"
                )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ListOpenAppsTool(BaseTool):
    name = "list_open_apps"
    description = "List currently running applications/windows"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        try:
            load_discovered_apps()
            system = platform.system().lower()
            if system == "windows":
                script = """
                Get-Process | Where-Object {$_.MainWindowTitle -ne ''} |
                Select-Object ProcessName, MainWindowTitle, Id |
                ConvertTo-Json
                """
                process = await asyncio.create_subprocess_exec(
                    "powershell",
                    "-Command",
                    script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate()
                import json

                apps = json.loads(stdout.decode("utf-8", errors="replace"))
                if isinstance(apps, dict):
                    apps = [apps]
                result = []
                for a in apps:
                    proc_name = a.get("ProcessName", "").lower()
                    friendly_name = proc_name
                    for name, info in APP_REGISTRY.items():
                        if proc_name == info["exe"].lower() or proc_name == name:
                            friendly_name = name
                            break
                    result.append(
                        {
                            "name": friendly_name,
                            "process": a.get("ProcessName"),
                            "title": a.get("MainWindowTitle"),
                            "pid": a.get("Id"),
                        }
                    )
            else:
                process = await asyncio.create_subprocess_exec(
                    "wmctrl",
                    "-l",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate()
                lines = stdout.decode("utf-8", errors="replace").strip().split("\n")
                result = []
                for line in lines:
                    parts = line.split(None, 3)
                    if len(parts) >= 4:
                        result.append({"title": parts[3], "id": parts[0]})
            return ToolResult(success=True, data={"apps": result, "count": len(result)})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class LaunchAppTool(BaseTool):
    name = "launch_app"
    description = "Launch an application by name"
    parameters = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "App name to launch (e.g., 'notepad', 'zen', 'discord')",
            },
        },
        "required": ["app_name"],
    }

    # Special app handlers for apps that need specific launch commands
    SPECIAL_APPS = {
        "windows": {
            "apple music": {"cmd": "start applemusic:", "shell": True},
            "applemusic": {"cmd": "start applemusic:", "shell": True},
            "music": {"cmd": "start applemusic:", "shell": True},  # Alias
        },
        "darwin": {
            "apple music": {"cmd": ["open", "-a", "Music"], "shell": False},
            "applemusic": {"cmd": ["open", "-a", "Music"], "shell": False},
            "music": {"cmd": ["open", "-a", "Music"], "shell": False},
        },
    }

    async def execute(self, app_name: str) -> ToolResult:
        try:
            system = platform.system().lower()
            query_lower = app_name.lower().strip()

            # Check for special app handlers first
            special_apps = self.SPECIAL_APPS.get(system, {})
            if query_lower in special_apps:
                handler = special_apps[query_lower]
                cmd = handler["cmd"]
                use_shell = handler.get("shell", False)

                if isinstance(cmd, list):
                    subprocess.Popen(
                        cmd,
                        shell=use_shell,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.Popen(
                        cmd,
                        shell=use_shell,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                return ToolResult(
                    success=True,
                    data={
                        "app": app_name,
                        "resolved": query_lower,
                        "command": cmd,
                        "status": "launched",
                    },
                )

            # Try to find app in registry
            match = find_app(app_name)
            if match:
                app_key, cmd = match
            else:
                cmd = app_name.strip()
                app_key = app_name

            if system == "windows":
                subprocess.Popen(
                    f"start {cmd}" if " " not in cmd else f'start "" "{cmd}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system == "darwin":
                subprocess.Popen(
                    ["open", "-a", cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    [cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            return ToolResult(
                success=True,
                data={"app": app_name, "resolved": app_key, "command": cmd, "status": "launched"},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class OpenUrlTool(BaseTool):
    name = "open_url"
    description = "Open a URL in the default browser (Zen)"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to open",
            },
            "browser": {
                "type": "string",
                "description": "Specific browser to use (default: zen)",
            },
        },
        "required": ["url"],
    }

    async def execute(self, url: str, browser: str = "zen") -> ToolResult:
        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            system = platform.system().lower()
            browser_lower = browser.lower()

            browser_commands = {
                "windows": {
                    "zen": ["cmd", "/c", "start", "zen", url],
                    "chrome": ["cmd", "/c", "start", "chrome", url],
                    "firefox": ["cmd", "/c", "start", "firefox", url],
                    "edge": ["cmd", "/c", "start", "msedge", url],
                },
                "linux": {
                    "zen": ["zen-browser", url],
                    "chrome": ["google-chrome", url],
                    "firefox": ["firefox", url],
                },
                "darwin": {
                    "zen": ["open", "-a", "Zen Browser", url],
                    "chrome": ["open", "-a", "Google Chrome", url],
                    "firefox": ["open", "-a", "Firefox", url],
                    "safari": ["open", "-a", "Safari", url],
                },
            }

            commands = browser_commands.get(system, {})
            cmd = commands.get(browser_lower)

            if cmd:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                webbrowser.open(url)

            return ToolResult(
                success=True,
                data={"url": url, "browser": browser, "status": "opened"},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class RunCommandTool(BaseTool):
    name = "run_command"
    description = "Run a shell command (restricted to safe commands)"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30, max: 60)",
            },
        },
        "required": ["command"],
    }

    ALLOWED_COMMANDS = {
        "ls",
        "dir",
        "pwd",
        "cd",
        "cat",
        "type",
        "head",
        "tail",
        "grep",
        "find",
        "where",
        "which",
        "echo",
        "date",
        "whoami",
        "hostname",
        "df",
        "du",
        "wc",
        "sort",
        "uniq",
        "tree",
        "git",
        "python",
        "pip",
        "node",
        "npm",
        "cargo",
    }

    BLOCKED_PATTERNS = [
        "rm -rf",
        "del /",
        "format",
        "mkfs",
        "dd if=",
        "sudo",
        "su ",
        "> /dev/",
        "| sh",
        "| bash",
        "curl | ",
        "wget | ",
        "eval(",
        "exec(",
    ]

    async def execute(self, command: str, timeout: int = 30) -> ToolResult:
        try:
            cmd_parts = command.split()
            if not cmd_parts:
                return ToolResult(success=False, data=None, error="Empty command")

            base_cmd = cmd_parts[0].lower()

            for pattern in self.BLOCKED_PATTERNS:
                if pattern in command.lower():
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Blocked: command contains dangerous pattern",
                    )

            if base_cmd not in self.ALLOWED_COMMANDS:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Command '{base_cmd}' not in allowed list",
                )

            timeout = min(max(timeout, 5), 60)

            process = await asyncio.create_subprocess_shell(
                command,
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
                    error=f"Command timed out after {timeout}s",
                )

            stdout_str = stdout.decode("utf-8", errors="replace")[:10000]
            stderr_str = stderr.decode("utf-8", errors="replace")[:2000]

            return ToolResult(
                success=process.returncode == 0,
                data={
                    "command": command,
                    "exit_code": process.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                },
                error=stderr_str if process.returncode != 0 else None,
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
