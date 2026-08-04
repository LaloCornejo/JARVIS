from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class TaskFrequency(Enum):
    ONCE = "once"
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class ScheduledTask:
    id: str
    name: str
    callback_name: str
    frequency: TaskFrequency
    next_run: datetime
    enabled: bool = True
    last_run: datetime | None = None
    data: dict[str, Any] | None = None
    days_of_week: list[int] | None = None
    hour: int | None = None
    minute: int | None = None


class TaskScheduler:
    def __init__(self, db_path: str | Path = "data/scheduler.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._callbacks: dict[str, Callable[..., Any]] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    callback_name TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    next_run TIMESTAMP NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    last_run TIMESTAMP,
                    data TEXT,
                    days_of_week TEXT,
                    hour INTEGER,
                    minute INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON scheduled_tasks(next_run);
                CREATE INDEX IF NOT EXISTS idx_tasks_enabled ON scheduled_tasks(enabled);
            """)

    def register_callback(self, name: str, callback: Callable[..., Any]) -> None:
        self._callbacks[name] = callback

    def schedule_task(
        self,
        task_id: str,
        name: str,
        callback_name: str,
        frequency: TaskFrequency,
        data: dict[str, Any] | None = None,
        days_of_week: list[int] | None = None,
        hour: int | None = None,
        minute: int | None = None,
        run_at: datetime | None = None,
    ) -> ScheduledTask:
        if callback_name not in self._callbacks:
            raise ValueError(f"Callback '{callback_name}' not registered")

        next_run = run_at or self._calculate_next_run(frequency, days_of_week, hour, minute)
        task = ScheduledTask(
            id=task_id,
            name=name,
            callback_name=callback_name,
            frequency=frequency,
            next_run=next_run,
            data=data,
            days_of_week=days_of_week,
            hour=hour,
            minute=minute,
        )
        self._save_task(task)
        return task

    def _calculate_next_run(
        self,
        frequency: TaskFrequency,
        days_of_week: list[int] | None,
        hour: int | None,
        minute: int | None,
    ) -> datetime:
        now = datetime.now()
        minute = minute or 0

        if frequency == TaskFrequency.MINUTELY:
            return now + timedelta(minutes=1)
        elif frequency == TaskFrequency.HOURLY:
            next_run = now.replace(minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
            return next_run
        elif frequency == TaskFrequency.DAILY:
            hour = hour or 8
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        elif frequency == TaskFrequency.WEEKLY:
            hour = hour or 8
            days = days_of_week or [0]
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            while next_run.weekday() not in days or next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        else:
            return now

    def _save_task(self, task: ScheduledTask) -> None:
        import json

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scheduled_tasks
                (id, name, callback_name, frequency, next_run, enabled,
                 last_run, data, days_of_week, hour, minute)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.name,
                    task.callback_name,
                    task.frequency.value,
                    task.next_run.isoformat(),
                    1 if task.enabled else 0,
                    task.last_run.isoformat() if task.last_run else None,
                    json.dumps(task.data) if task.data else None,
                    json.dumps(task.days_of_week) if task.days_of_week else None,
                    task.hour,
                    task.minute,
                ),
            )

    def get_due_tasks(self) -> list[ScheduledTask]:
        now = datetime.now()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_tasks WHERE enabled = 1 AND next_run <= ?",
                (now.isoformat(),),
            ).fetchall()
            return [self._row_to_task(row) for row in rows]

    def _row_to_task(self, row: sqlite3.Row) -> ScheduledTask:
        import json

        return ScheduledTask(
            id=row["id"],
            name=row["name"],
            callback_name=row["callback_name"],
            frequency=TaskFrequency(row["frequency"]),
            next_run=datetime.fromisoformat(row["next_run"]),
            enabled=bool(row["enabled"]),
            last_run=datetime.fromisoformat(row["last_run"]) if row["last_run"] else None,
            data=json.loads(row["data"]) if row["data"] else None,
            days_of_week=json.loads(row["days_of_week"]) if row["days_of_week"] else None,
            hour=row["hour"],
            minute=row["minute"],
        )

    async def execute_task(self, task: ScheduledTask) -> Any:
        callback = self._callbacks.get(task.callback_name)
        if not callback:
            return None

        try:
            if asyncio.iscoroutinefunction(callback):
                result = await callback(task.data)
            else:
                result = callback(task.data)

            task.last_run = datetime.now()
            if task.frequency != TaskFrequency.ONCE:
                task.next_run = self._calculate_next_run(
                    task.frequency, task.days_of_week, task.hour, task.minute
                )
            else:
                task.enabled = False
            self._save_task(task)
            return result
        except Exception as e:
            return {"error": str(e)}

    async def run(self) -> None:
        self._running = True
        while self._running:
            due_tasks = self.get_due_tasks()
            for task in due_tasks:
                await self.execute_task(task)
            await asyncio.sleep(30)

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self.run())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    def disable_task(self, task_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("UPDATE scheduled_tasks SET enabled = 0 WHERE id = ?", (task_id,))
            return cursor.rowcount > 0

    def enable_task(self, task_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("UPDATE scheduled_tasks SET enabled = 1 WHERE id = ?", (task_id,))
            return cursor.rowcount > 0

    def delete_task(self, task_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
            return cursor.rowcount > 0

    def list_tasks(self) -> list[ScheduledTask]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM scheduled_tasks ORDER BY next_run ASC").fetchall()
            return [self._row_to_task(row) for row in rows]
