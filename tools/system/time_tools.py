from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

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
            tz = ZoneInfo(timezone or "America/Mexico_City")
            now = datetime.now(tz)

            result = {
                "datetime": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "timezone": timezone or "America/Mexico_City",
            }
            return ToolResult(success=True, data=result)
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
