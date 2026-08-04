from __future__ import annotations

from datetime import datetime

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

    async def execute(self, tz: str | None = None) -> ToolResult:
        try:
            from datetime import datetime, timezone, timedelta

            now = datetime.now(timezone(timedelta(hours=-6)))  # America/Mexico_City
            time_str = now.strftime("%H:%M")
            date_str = now.strftime("%Y-%m-%d")
            return ToolResult(success=True, data=f"{date_str} {time_str}")
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
