from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class PatternAnalyzer:
    def __init__(self, db_path: str | Path = "data/learning.db"):
        self.db_path = Path(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_common_errors(self, days: int = 30, min_count: int = 2) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT error_message, tool_name, COUNT(*) as count,
                    MAX(created_at) as last_occurrence
                FROM learning_events
                WHERE success = 0
                AND error_message IS NOT NULL
                AND created_at > datetime('now', ? || ' days')
                GROUP BY error_message, tool_name
                HAVING COUNT(*) >= ?
                ORDER BY count DESC
                LIMIT 20
                """,
                (-days, min_count),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_frequent_commands(self, days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT input_text, COUNT(*) as count, AVG(success) as success_rate
                FROM learning_events
                WHERE event_type = 'command'
                AND created_at > datetime('now', ? || ' days')
                GROUP BY input_text
                ORDER BY count DESC
                LIMIT ?
                """,
                (-days, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_tool_usage_stats(self, days: int = 30) -> dict[str, dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT tool_name, COUNT(*) as total_calls,
                    SUM(success) as successful_calls, AVG(success) as success_rate
                FROM learning_events
                WHERE event_type = 'tool_call'
                AND tool_name IS NOT NULL
                AND created_at > datetime('now', ? || ' days')
                GROUP BY tool_name
                ORDER BY total_calls DESC
                """,
                (-days,),
            ).fetchall()
            return {row["tool_name"]: dict(row) for row in rows}

    def get_hourly_activity(self, days: int = 7) -> dict[int, int]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT CAST(strftime('%H', created_at) AS INTEGER) as hour,
                    COUNT(*) as count
                FROM learning_events
                WHERE created_at > datetime('now', ? || ' days')
                GROUP BY hour
                ORDER BY hour
                """,
                (-days,),
            ).fetchall()
            return {row["hour"]: row["count"] for row in rows}

    def get_daily_trends(self, days: int = 30) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT DATE(created_at) as date, COUNT(*) as total_events,
                    SUM(success) as successful_events,
                    SUM(CASE WHEN feedback = 'positive' THEN 1 ELSE 0 END) as positive_feedback,
                    SUM(CASE WHEN feedback = 'negative' THEN 1 ELSE 0 END) as negative_feedback
                FROM learning_events
                WHERE created_at > datetime('now', ? || ' days')
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                """,
                (-days,),
            ).fetchall()
            return [dict(row) for row in rows]

    def find_similar_failures(self, error_message: str, limit: int = 10) -> list[dict[str, Any]]:
        words = set(error_message.lower().split())
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, input_text, output_text, error_message, tool_name, created_at
                FROM learning_events
                WHERE success = 0 AND error_message IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100
                """
            ).fetchall()

            results = []
            for row in rows:
                row_words = set((row["error_message"] or "").lower().split())
                similarity = len(words & row_words) / max(len(words | row_words), 1)
                if similarity > 0.3:
                    result = dict(row)
                    result["similarity"] = similarity
                    results.append(result)

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]

    def get_improvement_suggestions(self) -> list[dict[str, Any]]:
        suggestions = []

        common_errors = self.get_common_errors(days=14, min_count=3)
        for error in common_errors:
            err_msg = error["error_message"][:50]
            suggestions.append(
                {
                    "type": "recurring_error",
                    "priority": "high" if error["count"] > 5 else "medium",
                    "message": f"Error '{err_msg}...' occurred {error['count']} times",
                    "tool": error["tool_name"],
                    "suggestion": "Review error handling for this tool",
                }
            )

        tool_stats = self.get_tool_usage_stats(days=14)
        for tool_name, stats in tool_stats.items():
            if stats["success_rate"] < 0.8 and stats["total_calls"] > 5:
                rate = stats["success_rate"]
                suggestions.append(
                    {
                        "type": "low_success_rate",
                        "priority": "high",
                        "message": f"Tool '{tool_name}' has {rate:.0%} success rate",
                        "tool": tool_name,
                        "suggestion": "Review tool implementation or add better error handling",
                    }
                )

        return suggestions

    def generate_report(self, days: int = 30) -> dict[str, Any]:
        return {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "common_errors": self.get_common_errors(days),
            "frequent_commands": self.get_frequent_commands(days),
            "tool_usage": self.get_tool_usage_stats(days),
            "hourly_activity": self.get_hourly_activity(min(days, 7)),
            "daily_trends": self.get_daily_trends(days),
            "improvement_suggestions": self.get_improvement_suggestions(),
        }
