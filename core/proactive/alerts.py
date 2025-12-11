from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Alert:
    id: str
    title: str
    message: str
    priority: AlertPriority
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    delivered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.name,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "data": self.data,
            "acknowledged": self.acknowledged,
            "delivered": self.delivered,
        }


class AlertManager:
    def __init__(self, db_path: str | Path = "data/alerts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._handlers: list[Callable[[Alert], None]] = []
        self._async_handlers: list[Callable[[Alert], Any]] = []
        self._quiet_hours: tuple[int, int] | None = None
        self._min_priority_during_quiet: AlertPriority = AlertPriority.URGENT
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    delivered INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
                CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);
                CREATE INDEX IF NOT EXISTS idx_alerts_source ON alerts(source);
            """)

    def set_quiet_hours(
        self, start_hour: int, end_hour: int, min_priority: AlertPriority = AlertPriority.URGENT
    ) -> None:
        self._quiet_hours = (start_hour, end_hour)
        self._min_priority_during_quiet = min_priority

    def _is_quiet_time(self) -> bool:
        if not self._quiet_hours:
            return False
        hour = datetime.now().hour
        start, end = self._quiet_hours
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        self._handlers.append(handler)

    def register_async_handler(self, handler: Callable[[Alert], Any]) -> None:
        self._async_handlers.append(handler)

    def create_alert(
        self,
        alert_id: str,
        title: str,
        message: str,
        priority: AlertPriority,
        source: str,
        data: dict[str, Any] | None = None,
    ) -> Alert:
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            priority=priority,
            source=source,
            data=data or {},
        )
        self._save_alert(alert)
        return alert

    def _save_alert(self, alert: Alert) -> None:
        import json

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alerts
                (id, title, message, priority, source, created_at, data, acknowledged, delivered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.id,
                    alert.title,
                    alert.message,
                    alert.priority.value,
                    alert.source,
                    alert.created_at.isoformat(),
                    json.dumps(alert.data),
                    1 if alert.acknowledged else 0,
                    1 if alert.delivered else 0,
                ),
            )

    def deliver_alert(self, alert: Alert) -> bool:
        if self._is_quiet_time() and alert.priority.value < self._min_priority_during_quiet.value:
            return False

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass

        alert.delivered = True
        self._save_alert(alert)
        return True

    async def deliver_alert_async(self, alert: Alert) -> bool:
        if self._is_quiet_time() and alert.priority.value < self._min_priority_during_quiet.value:
            return False

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass

        for handler in self._async_handlers:
            try:
                result = handler(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

        alert.delivered = True
        self._save_alert(alert)
        return True

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
            return cursor.rowcount > 0

    def get_pending_alerts(self, limit: int = 50) -> list[Alert]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM alerts
                WHERE acknowledged = 0
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._row_to_alert(row) for row in rows]

    def get_alerts_by_source(self, source: str, limit: int = 20) -> list[Alert]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE source = ? ORDER BY created_at DESC LIMIT ?",
                (source, limit),
            ).fetchall()
            return [self._row_to_alert(row) for row in rows]

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        import json

        return Alert(
            id=row["id"],
            title=row["title"],
            message=row["message"],
            priority=AlertPriority(row["priority"]),
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            data=json.loads(row["data"]) if row["data"] else {},
            acknowledged=bool(row["acknowledged"]),
            delivered=bool(row["delivered"]),
        )

    def cleanup_old_alerts(self, days: int = 30) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                DELETE FROM alerts
                WHERE acknowledged = 1
                AND created_at < datetime('now', ? || ' days')
                """,
                (-days,),
            )
            return cursor.rowcount
