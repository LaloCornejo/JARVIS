from __future__ import annotations

import json
import subprocess
from datetime import datetime

from core.config import Config
from tools.base import BaseTool, ToolResult


class GetCurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = "Get the current date and time, optionally for a specific timezone"
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone (e.g., 'America/New_York', 'UTC'). Defaults to local.",
            }
        },
        "required": [],
    }

    async def execute(self, timezone: str | None = None) -> ToolResult:
        try:
            cmd = ["get_time"]

            # Handle 'auto' or missing timezone - use config
            if timezone and timezone != "auto":
                cmd.extend(["--timezone", timezone])
            else:
                # Use timezone from config
                config = Config("config/settings.yaml")
                cmd.extend(["--timezone", config.timezone])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout.strip())
            return ToolResult(success=True, data=data)
        except subprocess.CalledProcessError as e:
            return ToolResult(success=False, data=None, error=f"Binary failed: {e.stderr}")
        except json.JSONDecodeError as e:
            return ToolResult(success=False, data=None, error=f"Invalid JSON: {e}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SetTimerTool(BaseTool):
    name = "set_timer"
    description = "Set a timer for a specified duration"
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "integer",
                "description": "Duration in seconds",
            },
            "label": {
                "type": "string",
                "description": "Optional label for the timer",
            },
        },
        "required": ["seconds"],
    }

    async def execute(self, seconds: int, label: str | None = None) -> ToolResult:
        end_time = datetime.now().timestamp() + seconds
        result = {
            "seconds": seconds,
            "label": label,
            "end_timestamp": end_time,
            "message": f"Timer set for {seconds} seconds" + (f" ({label})" if label else ""),
        }
        return ToolResult(success=True, data=result)
