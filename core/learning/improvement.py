from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .feedback import EventType, FeedbackCollector, FeedbackType, LearningEvent
from .patterns import PatternAnalyzer


class SelfImprovement:
    def __init__(self, db_path: str | Path = "data/learning.db"):
        self.feedback = FeedbackCollector(db_path)
        self.patterns = PatternAnalyzer(db_path)
        self._current_session: str | None = None
        self._last_event_id: int | None = None

    def start_session(self, session_id: str | None = None) -> str:
        self._current_session = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._current_session

    def log_command(
        self,
        user_input: str,
        response: str | None = None,
        context: dict[str, Any] | None = None,
        success: bool = True,
    ) -> int:
        self._last_event_id = self.feedback.log_event(
            event_type=EventType.COMMAND,
            input_text=user_input,
            output_text=response,
            context=context,
            success=success,
            session_id=self._current_session,
        )
        return self._last_event_id

    def log_response(
        self,
        prompt: str,
        response: str,
        model: str | None = None,
        success: bool = True,
    ) -> int:
        context = {"model": model} if model else None
        self._last_event_id = self.feedback.log_event(
            event_type=EventType.RESPONSE,
            input_text=prompt,
            output_text=response,
            context=context,
            success=success,
            session_id=self._current_session,
        )
        return self._last_event_id

    def log_tool_call(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        result: Any = None,
        success: bool = True,
        error: str | None = None,
    ) -> int:
        import json

        self._last_event_id = self.feedback.log_event(
            event_type=EventType.TOOL_CALL,
            input_text=json.dumps(input_params),
            output_text=json.dumps(result) if result else None,
            success=success,
            session_id=self._current_session,
            tool_name=tool_name,
            error_message=error,
        )
        return self._last_event_id

    def log_error(
        self,
        context: str,
        error_message: str,
        tool_name: str | None = None,
    ) -> int:
        self._last_event_id = self.feedback.log_event(
            event_type=EventType.ERROR,
            input_text=context,
            output_text=None,
            success=False,
            session_id=self._current_session,
            tool_name=tool_name,
            error_message=error_message,
        )
        return self._last_event_id

    def record_positive_feedback(self, event_id: int | None = None) -> None:
        event_id = event_id or self._last_event_id
        if event_id:
            self.feedback.add_feedback(event_id, FeedbackType.POSITIVE)

    def record_negative_feedback(self, event_id: int | None = None) -> None:
        event_id = event_id or self._last_event_id
        if event_id:
            self.feedback.add_feedback(event_id, FeedbackType.NEGATIVE)

    def record_correction(
        self,
        original_input: str,
        original_output: str,
        corrected_output: str,
        reason: str | None = None,
    ) -> int:
        correction_id = self.feedback.log_correction(
            original_input, original_output, corrected_output, reason
        )
        if self._last_event_id:
            self.feedback.add_feedback(
                self._last_event_id, FeedbackType.CORRECTION, corrected_output
            )
        return correction_id

    def get_success_rate(self, days: int = 7) -> dict[str, float]:
        return {
            "overall": self.feedback.get_success_rate(days=days),
            "commands": self.feedback.get_success_rate(EventType.COMMAND, days),
            "tool_calls": self.feedback.get_success_rate(EventType.TOOL_CALL, days),
            "responses": self.feedback.get_success_rate(EventType.RESPONSE, days),
        }

    def get_recent_errors(self, limit: int = 10) -> list[LearningEvent]:
        return self.feedback.get_recent_errors(limit)

    def get_improvement_report(self, days: int = 30) -> dict[str, Any]:
        report = self.patterns.generate_report(days)
        report["success_rates"] = self.get_success_rate(days)
        report["feedback_summary"] = self.feedback.get_feedback_summary()
        report["pending_corrections"] = len(self.feedback.get_corrections(applied=False))
        return report

    def export_training_data(self, output_path: str | Path) -> int:
        return self.feedback.export_for_training(output_path)

    def apply_learning(self) -> dict[str, Any]:
        corrections = self.feedback.get_corrections(applied=False)
        applied = []

        for correction in corrections:
            self.feedback.mark_correction_applied(correction["id"])
            applied.append(correction["id"])

        return {
            "applied_corrections": len(applied),
            "corrections": applied,
        }

    def get_tool_recommendations(self) -> list[dict[str, Any]]:
        tool_stats = self.patterns.get_tool_usage_stats(days=14)
        recommendations = []

        for tool_name, stats in tool_stats.items():
            if stats["success_rate"] < 0.7:
                recommendations.append(
                    {
                        "tool": tool_name,
                        "action": "review",
                        "reason": f"Low success rate ({stats['success_rate']:.0%})",
                        "calls": stats["total_calls"],
                    }
                )
            elif stats["total_calls"] > 50 and stats["success_rate"] > 0.95:
                recommendations.append(
                    {
                        "tool": tool_name,
                        "action": "optimize",
                        "reason": "High usage, could benefit from optimization",
                        "calls": stats["total_calls"],
                    }
                )

        return recommendations
