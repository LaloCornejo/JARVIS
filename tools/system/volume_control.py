from __future__ import annotations

import asyncio
from pathlib import Path

from tools.base import BaseTool, ToolResult


class SystemVolumeControlTool(BaseTool):
    """Base class for system volume control tools"""

    def __init__(self):
        super().__init__()
        self.tools_dir = Path(__file__).parent.parent
        self.system_control_exe = self.tools_dir / "target" / "release" / "system-control.exe"

    async def _run_system_control(self, command: str) -> ToolResult:
        """Run the system control executable with the given command"""
        if not self.system_control_exe.exists():
            return ToolResult(
                success=False,
                data=None,
                error="System control tool not found. Please build it first with 'cargo build --release'"
            )

        try:
            process = await asyncio.create_subprocess_exec(
                str(self.system_control_exe),
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

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


class VolumeUpTool(SystemVolumeControlTool):
    name = "volume_up"
    description = "Increase system volume by 10%"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        return await self._run_system_control("volume_up")


class VolumeDownTool(SystemVolumeControlTool):
    name = "volume_down"
    description = "Decrease system volume by 10%"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        return await self._run_system_control("volume_down")


class MuteToggleTool(SystemVolumeControlTool):
    name = "mute_toggle"
    description = "Toggle system mute state"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> ToolResult:
        return await self._run_system_control("mute")


class SetVolumeTool(SystemVolumeControlTool):
    name = "set_volume"
    description = "Set system volume to a specific level (0-100)"
    parameters = {
        "type": "object",
        "properties": {
            "level": {
                "type": "integer",
                "description": "Volume level (0-100)",
                "minimum": 0,
                "maximum": 100,
            },
        },
        "required": ["level"],
    }

    async def execute(self, level: int) -> ToolResult:
        return await self._run_system_control(f"set_volume {level}")
