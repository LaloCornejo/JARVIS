from __future__ import annotations

import asyncio
import datetime
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult


@dataclass
class AppUsage:
    app_name: str
    window_title: str
    start_time: str
    end_time: str
    duration_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "window_title": self.window_title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
        }


class ScreenTimeTracker:
    def __init__(self, db_path: str | None = None):
        self.db_path = Path(
            db_path
            or os.environ.get(
                "SCREENTIME_DB",
                os.path.expanduser("~/.jarvis/screentime.db"),
            )
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._current_app: str | None = None
        self._current_window: str | None = None
        self._session_start: datetime.datetime | None = None

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name TEXT NOT NULL,
                    window_title TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    date TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_date ON app_usage(date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_app_name ON app_usage(app_name)
            """)

    async def _run_powershell(self, script: str) -> tuple[bool, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                "powershell",
                "-NoProfile",
                "-Command",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return True, stdout.decode().strip()
            return False, stderr.decode().strip()
        except Exception as e:
            return False, str(e)

    async def get_active_window(self) -> tuple[str, str]:
        script = """
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}
"@
$hwnd = [Win32]::GetForegroundWindow()
$sb = New-Object System.Text.StringBuilder 256
[Win32]::GetWindowText($hwnd, $sb, 256) | Out-Null
$title = $sb.ToString()
$pid = 0
[Win32]::GetWindowThreadProcessId($hwnd, [ref]$pid) | Out-Null
$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
$appName = if ($proc) { $proc.ProcessName } else { "Unknown" }
"$appName|$title"
"""
        success, output = await self._run_powershell(script)
        if success and "|" in output:
            parts = output.split("|", 1)
            return parts[0], parts[1]
        return "Unknown", ""

    def _record_session(
        self, app_name: str, window_title: str, start: datetime.datetime, end: datetime.datetime
    ) -> None:
        duration = int((end - start).total_seconds())
        if duration < 1:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO app_usage
                (app_name, window_title, start_time, end_time, duration_seconds, date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    app_name,
                    window_title,
                    start.isoformat(),
                    end.isoformat(),
                    duration,
                    start.date().isoformat(),
                ),
            )

    async def track_once(self) -> tuple[str, str]:
        app_name, window_title = await self.get_active_window()
        now = datetime.datetime.now()

        if self._current_app and self._session_start:
            if app_name != self._current_app or window_title != self._current_window:
                self._record_session(
                    self._current_app,
                    self._current_window or "",
                    self._session_start,
                    now,
                )
                self._current_app = app_name
                self._current_window = window_title
                self._session_start = now
        else:
            self._current_app = app_name
            self._current_window = window_title
            self._session_start = now

        return app_name, window_title

    def flush_current_session(self) -> None:
        if self._current_app and self._session_start:
            now = datetime.datetime.now()
            self._record_session(
                self._current_app,
                self._current_window or "",
                self._session_start,
                now,
            )
            self._session_start = now

    def get_daily_summary(self, date: str | None = None) -> dict[str, int]:
        if date is None:
            date = datetime.date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT app_name, SUM(duration_seconds) as total
                FROM app_usage
                WHERE date = ?
                GROUP BY app_name
                ORDER BY total DESC
                """,
                (date,),
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_weekly_summary(self) -> dict[str, dict[str, int]]:
        today = datetime.date.today()
        week_start = today - datetime.timedelta(days=today.weekday())
        summary = {}
        for i in range(7):
            date = (week_start + datetime.timedelta(days=i)).isoformat()
            summary[date] = self.get_daily_summary(date)
        return summary

    def get_app_history(self, app_name: str, days: int = 7) -> list[dict[str, Any]]:
        start_date = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT date, SUM(duration_seconds) as total
                FROM app_usage
                WHERE app_name = ? AND date >= ?
                GROUP BY date
                ORDER BY date DESC
                """,
                (app_name, start_date),
            )
            return [{"date": row[0], "duration_seconds": row[1]} for row in cursor.fetchall()]

    def get_recent_activity(self, limit: int = 20) -> list[AppUsage]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT app_name, window_title, start_time, end_time, duration_seconds
                FROM app_usage
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [
                AppUsage(
                    app_name=row[0],
                    window_title=row[1],
                    start_time=row[2],
                    end_time=row[3],
                    duration_seconds=row[4],
                )
                for row in cursor.fetchall()
            ]

    def get_total_today(self) -> int:
        date = datetime.date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT SUM(duration_seconds) FROM app_usage WHERE date = ?",
                (date,),
            )
            result = cursor.fetchone()[0]
            return result or 0

    def get_category_time(
        self, categories: dict[str, list[str]], date: str | None = None
    ) -> dict[str, int]:
        daily = self.get_daily_summary(date)
        result = {cat: 0 for cat in categories}
        result["other"] = 0

        for app, duration in daily.items():
            categorized = False
            for cat, apps in categories.items():
                if app.lower() in [a.lower() for a in apps]:
                    result[cat] += duration
                    categorized = True
                    break
            if not categorized:
                result["other"] += duration
        return result


_tracker: ScreenTimeTracker | None = None


def get_screen_time_tracker() -> ScreenTimeTracker:
    global _tracker
    if _tracker is None:
        _tracker = ScreenTimeTracker()
    return _tracker


class ScreenTimeCurrentTool(BaseTool):
    name = "screentime_current"
    description = "Get the currently active window/app"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            app_name, window_title = await tracker.get_active_window()
            return ToolResult(
                success=True,
                data={"app_name": app_name, "window_title": window_title},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenTimeDailySummaryTool(BaseTool):
    name = "screentime_daily"
    description = "Get screen time summary for a specific day"
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format (default: today)",
            },
        },
        "required": [],
    }

    async def execute(self, date: str | None = None) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            summary = tracker.get_daily_summary(date)
            formatted = [
                {
                    "app": app,
                    "duration_seconds": duration,
                    "duration_formatted": f"{duration // 3600}h {(duration % 3600) // 60}m",
                }
                for app, duration in summary.items()
            ]
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenTimeWeeklySummaryTool(BaseTool):
    name = "screentime_weekly"
    description = "Get screen time summary for the current week"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            summary = tracker.get_weekly_summary()
            return ToolResult(success=True, data=summary)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenTimeAppHistoryTool(BaseTool):
    name = "screentime_app_history"
    description = "Get usage history for a specific app"
    parameters = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Name of the application",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to look back (default: 7)",
            },
        },
        "required": ["app_name"],
    }

    async def execute(self, app_name: str, days: int = 7) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            history = tracker.get_app_history(app_name, days)
            return ToolResult(success=True, data=history)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenTimeRecentTool(BaseTool):
    name = "screentime_recent"
    description = "Get recent app activity log"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of entries to retrieve (default: 20)",
            },
        },
        "required": [],
    }

    async def execute(self, limit: int = 20) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            activity = tracker.get_recent_activity(limit)
            return ToolResult(success=True, data=[a.to_dict() for a in activity])
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenTimeTotalTodayTool(BaseTool):
    name = "screentime_total_today"
    description = "Get total screen time for today"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            total = tracker.get_total_today()
            hours = total // 3600
            minutes = (total % 3600) // 60
            return ToolResult(
                success=True,
                data={
                    "total_seconds": total,
                    "formatted": f"{hours}h {minutes}m",
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenTimeTrackOnceTool(BaseTool):
    name = "screentime_track"
    description = "Record the current active window (call periodically to track usage)"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            app_name, window_title = await tracker.track_once()
            return ToolResult(
                success=True,
                data={"app_name": app_name, "window_title": window_title},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ScreenTimeCategoryTool(BaseTool):
    name = "screentime_by_category"
    description = "Get screen time grouped by custom categories"
    parameters = {
        "type": "object",
        "properties": {
            "categories": {
                "type": "object",
                "description": "Category definitions as {category_name: [app1, app2, ...]}",
            },
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format (default: today)",
            },
        },
        "required": ["categories"],
    }

    async def execute(
        self, categories: dict[str, list[str]], date: str | None = None
    ) -> ToolResult:
        tracker = get_screen_time_tracker()
        try:
            result = tracker.get_category_time(categories, date)
            formatted = {
                cat: {
                    "duration_seconds": duration,
                    "formatted": f"{duration // 3600}h {(duration % 3600) // 60}m",
                }
                for cat, duration in result.items()
            }
            return ToolResult(success=True, data=formatted)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
