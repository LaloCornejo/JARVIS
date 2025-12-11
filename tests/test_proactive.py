from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

from core.proactive.alerts import Alert, AlertManager, AlertPriority
from core.proactive.monitor import ProactiveMonitor
from core.proactive.scheduler import TaskScheduler


class TestAlertManager:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "alerts.db"
        self.manager = AlertManager(self.db_path)

    def test_create_alert(self) -> None:
        alert = self.manager.create_alert(
            alert_id="test_1",
            title="Test Alert",
            message="This is a test",
            priority=AlertPriority.MEDIUM,
            source="test",
        )
        assert alert.id == "test_1"
        assert alert.title == "Test Alert"
        assert alert.priority == AlertPriority.MEDIUM

    def test_get_pending_alerts(self) -> None:
        self.manager.create_alert(
            alert_id="test_1",
            title="Alert 1",
            message="Message 1",
            priority=AlertPriority.LOW,
            source="test",
        )
        self.manager.create_alert(
            alert_id="test_2",
            title="Alert 2",
            message="Message 2",
            priority=AlertPriority.HIGH,
            source="test",
        )

        pending = self.manager.get_pending_alerts()
        assert len(pending) == 2
        assert pending[0].priority == AlertPriority.HIGH

    def test_acknowledge_alert(self) -> None:
        self.manager.create_alert(
            alert_id="test_ack",
            title="Ack Test",
            message="Test",
            priority=AlertPriority.MEDIUM,
            source="test",
        )

        assert self.manager.acknowledge_alert("test_ack")
        pending = self.manager.get_pending_alerts()
        assert len(pending) == 0

    def test_quiet_hours(self) -> None:
        self.manager.set_quiet_hours(22, 7, AlertPriority.URGENT)
        assert self.manager._quiet_hours == (22, 7)
        assert self.manager._min_priority_during_quiet == AlertPriority.URGENT

    def test_handler_registration(self) -> None:
        alerts_received = []

        def handler(alert: Alert) -> None:
            alerts_received.append(alert)

        self.manager.register_handler(handler)
        alert = self.manager.create_alert(
            alert_id="handler_test",
            title="Handler Test",
            message="Test",
            priority=AlertPriority.MEDIUM,
            source="test",
        )
        self.manager.deliver_alert(alert)

        assert len(alerts_received) == 1
        assert alerts_received[0].id == "handler_test"

    def test_alert_to_dict(self) -> None:
        alert = self.manager.create_alert(
            alert_id="dict_test",
            title="Dict Test",
            message="Test message",
            priority=AlertPriority.HIGH,
            source="test",
            data={"key": "value"},
        )

        d = alert.to_dict()
        assert d["id"] == "dict_test"
        assert d["priority"] == "HIGH"
        assert d["data"] == {"key": "value"}


class TestTaskScheduler:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "scheduler.db"
        self.scheduler = TaskScheduler(self.db_path)

    def test_register_callback(self) -> None:
        async def test_callback(data: dict | None = None) -> str:
            return "executed"

        self.scheduler.register_callback("test", test_callback)
        assert "test" in self.scheduler._callbacks

    def test_schedule_task(self) -> None:
        async def dummy(data: dict | None = None) -> None:
            pass

        self.scheduler.register_callback("dummy", dummy)

        from core.proactive.scheduler import TaskFrequency

        task = self.scheduler.schedule_task(
            task_id="task_1",
            name="Test Task",
            callback_name="dummy",
            frequency=TaskFrequency.HOURLY,
        )

        assert task.id == "task_1"
        tasks = self.scheduler.list_tasks()
        assert len(tasks) == 1

    def test_delete_task(self) -> None:
        async def dummy(data: dict | None = None) -> None:
            pass

        self.scheduler.register_callback("dummy", dummy)

        from core.proactive.scheduler import TaskFrequency

        self.scheduler.schedule_task(
            task_id="delete_test",
            name="Delete Test",
            callback_name="dummy",
            frequency=TaskFrequency.DAILY,
        )

        assert self.scheduler.delete_task("delete_test")
        tasks = self.scheduler.list_tasks()
        assert len(tasks) == 0

    def test_disable_enable_task(self) -> None:
        async def dummy(data: dict | None = None) -> None:
            pass

        self.scheduler.register_callback("dummy", dummy)

        from core.proactive.scheduler import TaskFrequency

        self.scheduler.schedule_task(
            task_id="toggle_test",
            name="Toggle Test",
            callback_name="dummy",
            frequency=TaskFrequency.HOURLY,
        )

        assert self.scheduler.disable_task("toggle_test")
        assert self.scheduler.enable_task("toggle_test")


class TestProactiveMonitor:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "alerts.db"
        self.alerts = AlertManager(self.db_path)
        self.monitor = ProactiveMonitor(alert_manager=self.alerts)

    def test_extract_deadline_from_text(self) -> None:
        result = self.monitor.extract_deadline_from_text("Project due on January 15th")
        assert result is not None
        assert "January 15th" in result["match"]

    def test_extract_deadline_iso_format(self) -> None:
        result = self.monitor.extract_deadline_from_text("Deadline is 2024-12-25")
        assert result is not None
        assert "2024-12-25" in result["match"]

    def test_extract_no_deadline(self) -> None:
        result = self.monitor.extract_deadline_from_text("Just a regular sentence")
        assert result is None

    def test_get_pending_alerts(self) -> None:
        self.alerts.create_alert(
            alert_id="monitor_test",
            title="Monitor Test",
            message="Test",
            priority=AlertPriority.MEDIUM,
            source="test",
        )

        pending = self.monitor.get_pending_alerts()
        assert len(pending) == 1

    def test_acknowledge_alert(self) -> None:
        self.alerts.create_alert(
            alert_id="ack_test",
            title="Ack Test",
            message="Test",
            priority=AlertPriority.LOW,
            source="test",
        )

        assert self.monitor.acknowledge_alert("ack_test")

    def test_register_source(self) -> None:
        def check_something() -> dict:
            return {"status": "ok"}

        self.monitor.register_source("custom_source", check_something)
        assert "custom_source" in self.monitor._sources

    def test_format_timedelta(self) -> None:
        td = timedelta(hours=2, minutes=30)
        result = self.monitor._format_timedelta(td)
        assert "2h" in result
        assert "30m" in result

        td_minutes = timedelta(minutes=45)
        result = self.monitor._format_timedelta(td_minutes)
        assert "45 minutes" in result
