from __future__ import annotations

import tempfile
from pathlib import Path

from core.learning.feedback import EventType, FeedbackCollector, FeedbackType
from core.learning.improvement import SelfImprovement
from core.learning.patterns import PatternAnalyzer


class TestFeedbackCollector:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "learning.db"
        self.collector = FeedbackCollector(self.db_path)

    def test_log_event(self) -> None:
        event_id = self.collector.log_event(
            event_type=EventType.COMMAND,
            input_text="test command",
            output_text="test response",
            success=True,
        )
        assert event_id > 0

    def test_log_event_with_context(self) -> None:
        event_id = self.collector.log_event(
            event_type=EventType.TOOL_CALL,
            input_text="tool input",
            output_text="tool output",
            context={"tool": "search", "params": {"query": "test"}},
            success=True,
            tool_name="search",
        )
        assert event_id > 0

    def test_log_error_event(self) -> None:
        self.collector.log_event(
            event_type=EventType.ERROR,
            input_text="failed operation",
            success=False,
            error_message="Connection timeout",
        )

        errors = self.collector.get_recent_errors(limit=5)
        assert len(errors) == 1
        assert errors[0].error_message == "Connection timeout"

    def test_log_correction(self) -> None:
        correction_id = self.collector.log_correction(
            original_input="What's the weather?",
            original_output="I don't know.",
            corrected_output="The weather in SF is 65°F.",
            reason="Should check weather API",
        )
        assert correction_id > 0

    def test_get_corrections(self) -> None:
        self.collector.log_correction(
            original_input="input 1",
            original_output="wrong 1",
            corrected_output="correct 1",
        )
        self.collector.log_correction(
            original_input="input 2",
            original_output="wrong 2",
            corrected_output="correct 2",
        )

        corrections = self.collector.get_corrections(applied=False)
        assert len(corrections) == 2

    def test_mark_correction_applied(self) -> None:
        correction_id = self.collector.log_correction(
            original_input="input",
            original_output="wrong",
            corrected_output="correct",
        )

        assert self.collector.mark_correction_applied(correction_id)

        applied = self.collector.get_corrections(applied=True)
        assert len(applied) == 1

    def test_add_feedback(self) -> None:
        event_id = self.collector.log_event(
            event_type=EventType.RESPONSE,
            input_text="question",
            output_text="answer",
            success=True,
        )

        self.collector.add_feedback(event_id, FeedbackType.POSITIVE)

        summary = self.collector.get_feedback_summary()
        assert "response" in summary
        assert summary["response"]["positive"] == 1

    def test_success_rate(self) -> None:
        self.collector.log_event(
            event_type=EventType.COMMAND,
            input_text="cmd1",
            success=True,
        )
        self.collector.log_event(
            event_type=EventType.COMMAND,
            input_text="cmd2",
            success=True,
        )
        self.collector.log_event(
            event_type=EventType.COMMAND,
            input_text="cmd3",
            success=False,
        )

        rate = self.collector.get_success_rate(EventType.COMMAND, days=7)
        assert 0.65 < rate < 0.68

    def test_export_for_training(self) -> None:
        self.collector.log_correction(
            original_input="q1",
            original_output="wrong1",
            corrected_output="right1",
        )
        self.collector.log_correction(
            original_input="q2",
            original_output="wrong2",
            corrected_output="right2",
        )

        output_path = Path(self.temp_dir) / "training.jsonl"
        count = self.collector.export_for_training(output_path)

        assert count == 2
        assert output_path.exists()


class TestPatternAnalyzer:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "learning.db"
        FeedbackCollector(self.db_path)
        self.analyzer = PatternAnalyzer(self.db_path)

    def test_get_common_errors(self) -> None:
        errors = self.analyzer.get_common_errors(days=7)
        assert isinstance(errors, list)

    def test_get_tool_usage_stats(self) -> None:
        stats = self.analyzer.get_tool_usage_stats(days=7)
        assert isinstance(stats, dict)

    def test_generate_report(self) -> None:
        report = self.analyzer.generate_report(days=7)
        assert "common_errors" in report
        assert "tool_usage" in report
        assert "generated_at" in report


class TestSelfImprovement:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "learning.db"
        self.improvement = SelfImprovement(self.db_path)

    def test_start_session(self) -> None:
        session_id = self.improvement.start_session()
        assert session_id is not None
        assert self.improvement._current_session == session_id

    def test_start_session_custom_id(self) -> None:
        session_id = self.improvement.start_session("custom_session")
        assert session_id == "custom_session"

    def test_log_command(self) -> None:
        self.improvement.start_session()
        event_id = self.improvement.log_command(
            user_input="test command",
            response="test response",
            success=True,
        )
        assert event_id > 0

    def test_log_response(self) -> None:
        self.improvement.start_session()
        event_id = self.improvement.log_response(
            prompt="What is 2+2?",
            response="4",
            model="qwen3",
            success=True,
        )
        assert event_id > 0

    def test_log_tool_call(self) -> None:
        self.improvement.start_session()
        event_id = self.improvement.log_tool_call(
            tool_name="web_search",
            input_params={"query": "test"},
            result={"results": []},
            success=True,
        )
        assert event_id > 0

    def test_log_error(self) -> None:
        self.improvement.start_session()
        event_id = self.improvement.log_error(
            context="API call",
            error_message="Timeout",
            tool_name="api_tool",
        )
        assert event_id > 0

    def test_record_feedback(self) -> None:
        self.improvement.start_session()
        self.improvement.log_command("test", "response")

        self.improvement.record_positive_feedback()

        summary = self.improvement.feedback.get_feedback_summary()
        assert "command" in summary

    def test_record_correction(self) -> None:
        self.improvement.start_session()
        self.improvement.log_command("what's the time", "I don't know")

        correction_id = self.improvement.record_correction(
            original_input="what's the time",
            original_output="I don't know",
            corrected_output="It's 3:00 PM",
            reason="Should check system time",
        )
        assert correction_id > 0

    def test_get_success_rate(self) -> None:
        self.improvement.start_session()
        self.improvement.log_command("cmd1", "resp1", success=True)
        self.improvement.log_command("cmd2", "resp2", success=False)

        rates = self.improvement.get_success_rate(days=7)
        assert "overall" in rates
        assert "commands" in rates
        assert 0 <= rates["overall"] <= 1

    def test_get_improvement_report(self) -> None:
        self.improvement.start_session()
        self.improvement.log_command("test", "response")

        report = self.improvement.get_improvement_report(days=7)
        assert "success_rates" in report
        assert "feedback_summary" in report
        assert "pending_corrections" in report

    def test_apply_learning(self) -> None:
        self.improvement.record_correction(
            original_input="q1",
            original_output="wrong",
            corrected_output="right",
        )

        result = self.improvement.apply_learning()
        assert result["applied_corrections"] == 1

    def test_export_training_data(self) -> None:
        self.improvement.record_correction(
            original_input="q",
            original_output="wrong",
            corrected_output="right",
        )

        output_path = Path(self.temp_dir) / "export.jsonl"
        count = self.improvement.export_training_data(output_path)
        assert count == 1
