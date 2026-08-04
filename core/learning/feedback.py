from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(Enum):
    COMMAND = "command"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    CORRECTION = "correction"
    FEEDBACK = "feedback"


class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"


@dataclass
class LearningEvent:
    id: int | None
    event_type: EventType
    input_text: str
    output_text: str | None
    context: dict[str, Any]
    success: bool
    created_at: datetime = field(default_factory=datetime.now)
    session_id: str | None = None
    tool_name: str | None = None
    error_message: str | None = None
    feedback: FeedbackType | None = None
    correction: str | None = None


class FeedbackCollector:
    def __init__(self, db_path: str | Path = "data/learning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT,
                    context TEXT,
                    success INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    tool_name TEXT,
                    error_message TEXT,
                    feedback TEXT,
                    correction TEXT
                );

                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_input TEXT NOT NULL,
                    original_output TEXT NOT NULL,
                    corrected_output TEXT NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    applied INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS feedback_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    positive_count INTEGER DEFAULT 0,
                    negative_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_events_type ON learning_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_session ON learning_events(session_id);
                CREATE INDEX IF NOT EXISTS idx_events_success ON learning_events(success);
                CREATE INDEX IF NOT EXISTS idx_events_created ON learning_events(created_at);
                CREATE INDEX IF NOT EXISTS idx_corrections_applied ON corrections(applied);
            """)

    def log_event(
        self,
        event_type: EventType,
        input_text: str,
        output_text: str | None = None,
        context: dict[str, Any] | None = None,
        success: bool = True,
        session_id: str | None = None,
        tool_name: str | None = None,
        error_message: str | None = None,
    ) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO learning_events
                (event_type, input_text, output_text, context, success,
                 session_id, tool_name, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type.value,
                    input_text,
                    output_text,
                    json.dumps(context) if context else None,
                    1 if success else 0,
                    session_id,
                    tool_name,
                    error_message,
                ),
            )
            return cursor.lastrowid

    def log_correction(
        self,
        original_input: str,
        original_output: str,
        corrected_output: str,
        reason: str | None = None,
    ) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO corrections
                (original_input, original_output, corrected_output, reason)
                VALUES (?, ?, ?, ?)
                """,
                (original_input, original_output, corrected_output, reason),
            )
            return cursor.lastrowid

    def add_feedback(
        self,
        event_id: int,
        feedback: FeedbackType,
        correction: str | None = None,
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE learning_events SET feedback = ?, correction = ? WHERE id = ?",
                (feedback.value, correction, event_id),
            )

    def get_recent_errors(self, limit: int = 20) -> list[LearningEvent]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM learning_events
                WHERE success = 0 OR event_type = 'error'
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._row_to_event(row) for row in rows]

    def get_corrections(self, applied: bool | None = None, limit: int = 50) -> list[dict]:
        with self._get_conn() as conn:
            if applied is None:
                rows = conn.execute(
                    "SELECT * FROM corrections ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM corrections WHERE applied = ? ORDER BY created_at DESC LIMIT ?",
                    (1 if applied else 0, limit),
                ).fetchall()
            return [dict(row) for row in rows]

    def mark_correction_applied(self, correction_id: int) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE corrections SET applied = 1 WHERE id = ?", (correction_id,)
            )
            return cursor.rowcount > 0

    def get_success_rate(self, event_type: EventType | None = None, days: int = 7) -> float:
        with self._get_conn() as conn:
            if event_type:
                row = conn.execute(
                    """
                    SELECT COUNT(*) as total, SUM(success) as successes
                    FROM learning_events
                    WHERE event_type = ?
                    AND created_at > datetime('now', ? || ' days')
                    """,
                    (event_type.value, -days),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT COUNT(*) as total, SUM(success) as successes
                    FROM learning_events
                    WHERE created_at > datetime('now', ? || ' days')
                    """,
                    (-days,),
                ).fetchone()

            if row and row["total"] > 0:
                return (row["successes"] or 0) / row["total"]
            return 1.0

    def get_feedback_summary(self) -> dict[str, dict[str, int]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT event_type, feedback, COUNT(*) as count
                FROM learning_events
                WHERE feedback IS NOT NULL
                GROUP BY event_type, feedback
                """
            ).fetchall()

            summary: dict[str, dict[str, int]] = {}
            for row in rows:
                event_type = row["event_type"]
                if event_type not in summary:
                    summary[event_type] = {"positive": 0, "negative": 0, "correction": 0}
                summary[event_type][row["feedback"]] = row["count"]
            return summary

    def _row_to_event(self, row: sqlite3.Row) -> LearningEvent:
        return LearningEvent(
            id=row["id"],
            event_type=EventType(row["event_type"]),
            input_text=row["input_text"],
            output_text=row["output_text"],
            context=json.loads(row["context"]) if row["context"] else {},
            success=bool(row["success"]),
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else datetime.now(),
            session_id=row["session_id"],
            tool_name=row["tool_name"],
            error_message=row["error_message"],
            feedback=FeedbackType(row["feedback"]) if row["feedback"] else None,
            correction=row["correction"],
        )

    def export_for_training(self, output_path: str | Path) -> int:
        output_path = Path(output_path)
        corrections = self.get_corrections(applied=False)

        training_data = []
        for correction in corrections:
            training_data.append(
                {
                    "input": correction["original_input"],
                    "rejected": correction["original_output"],
                    "chosen": correction["corrected_output"],
                    "reason": correction.get("reason"),
                }
            )

        with open(output_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        return len(training_data)
