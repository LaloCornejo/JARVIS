from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

from .alerts import Alert, AlertManager, AlertPriority
from .scheduler import TaskFrequency, TaskScheduler


class ProactiveMonitor:
    def __init__(
        self,
        alert_manager: AlertManager | None = None,
        scheduler: TaskScheduler | None = None,
    ):
        self.alerts = alert_manager or AlertManager()
        self.scheduler = scheduler or TaskScheduler()
        self._sources: dict[str, Callable[[], Any]] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._setup_default_tasks()

    def _setup_default_tasks(self) -> None:
        self.scheduler.register_callback("check_calendar", self._check_calendar)
        self.scheduler.register_callback("check_deadlines", self._check_deadlines)
        self.scheduler.register_callback("morning_briefing", self._morning_briefing)
        self.scheduler.register_callback("check_emails", self._check_emails)

    def register_source(self, name: str, check_fn: Callable[[], Any]) -> None:
        self._sources[name] = check_fn

    async def _check_calendar(self, data: dict | None = None) -> list[Alert]:
        try:
            from tools.integrations.google_calendar import GoogleCalendarTool

            calendar = GoogleCalendarTool()
            events = await calendar.get_upcoming_events(hours=1)

            alerts = []
            for event in events:
                start = event.get("start", {})
                start_time = start.get("dateTime") or start.get("date")
                if start_time:
                    alert = self.alerts.create_alert(
                        alert_id=f"calendar_{event.get('id', uuid.uuid4().hex)}",
                        title=f"Upcoming: {event.get('summary', 'Event')}",
                        message=f"Starting at {start_time}",
                        priority=AlertPriority.MEDIUM,
                        source="calendar",
                        data=event,
                    )
                    alerts.append(alert)
                    await self.alerts.deliver_alert_async(alert)
            return alerts
        except Exception:
            return []

    async def _check_emails(self, data: dict | None = None) -> list[Alert]:
        try:
            from tools.integrations.gmail import GmailTool

            gmail = GmailTool()
            emails = await gmail.get_unread(limit=5, priority_only=True)

            alerts = []
            for email in emails:
                alert = self.alerts.create_alert(
                    alert_id=f"email_{email.get('id', uuid.uuid4().hex)}",
                    title=f"Email: {email.get('subject', 'No subject')}",
                    message=f"From: {email.get('from', 'Unknown')}",
                    priority=AlertPriority.MEDIUM,
                    source="email",
                    data=email,
                )
                alerts.append(alert)
                await self.alerts.deliver_alert_async(alert)
            return alerts
        except Exception:
            return []

    async def _check_deadlines(self, data: dict | None = None) -> list[Alert]:
        from core.memory import ConversationMemory

        memory = ConversationMemory()
        deadlines = memory.get_facts_by_category("deadline")

        alerts = []
        now = datetime.now()
        for key, value in deadlines.items():
            try:
                if isinstance(value, dict):
                    deadline_str = value.get("date")
                    description = value.get("description", key)
                else:
                    deadline_str = value
                    description = key

                deadline = datetime.fromisoformat(deadline_str)
                time_until = deadline - now

                if timedelta(0) < time_until < timedelta(hours=24):
                    priority = (
                        AlertPriority.HIGH
                        if time_until < timedelta(hours=2)
                        else AlertPriority.MEDIUM
                    )
                    alert = self.alerts.create_alert(
                        alert_id=f"deadline_{key}_{deadline.date().isoformat()}",
                        title=f"Deadline approaching: {description}",
                        message=f"Due in {self._format_timedelta(time_until)}",
                        priority=priority,
                        source="deadline",
                        data={"key": key, "deadline": deadline_str},
                    )
                    alerts.append(alert)
                    await self.alerts.deliver_alert_async(alert)
            except (ValueError, TypeError):
                continue
        return alerts

    async def _morning_briefing(self, data: dict | None = None) -> Alert:
        briefing_parts = []

        try:
            from tools.integrations.google_calendar import GoogleCalendarTool

            calendar = GoogleCalendarTool()
            events = await calendar.get_today_events()
            if events:
                briefing_parts.append(f"You have {len(events)} events today")
        except Exception:
            pass

        try:
            from tools.integrations.gmail import GmailTool

            gmail = GmailTool()
            unread = await gmail.get_unread_count()
            if unread > 0:
                briefing_parts.append(f"{unread} unread emails")
        except Exception:
            pass

        from core.memory import ConversationMemory

        memory = ConversationMemory()
        deadlines = memory.get_facts_by_category("deadline")
        today_deadlines = []
        now = datetime.now()
        for key, value in deadlines.items():
            try:
                if isinstance(value, dict):
                    deadline_str = value.get("date")
                else:
                    deadline_str = value
                deadline = datetime.fromisoformat(deadline_str)
                if deadline.date() == now.date():
                    today_deadlines.append(key)
            except (ValueError, TypeError):
                continue
        if today_deadlines:
            briefing_parts.append(f"{len(today_deadlines)} deadlines today")

        message = ". ".join(briefing_parts) if briefing_parts else "No significant items for today"

        alert = self.alerts.create_alert(
            alert_id=f"briefing_{now.date().isoformat()}",
            title="Good morning! Here's your briefing",
            message=message,
            priority=AlertPriority.LOW,
            source="briefing",
        )
        await self.alerts.deliver_alert_async(alert)
        return alert

    def _format_timedelta(self, td: timedelta) -> str:
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes} minutes"

    def extract_deadline_from_text(self, text: str) -> dict | None:
        patterns = [
            r"(\d{4}-\d{2}-\d{2})",
            r"due\s+(?:on\s+)?(\w+\s+\d+(?:st|nd|rd|th)?(?:\s+\d{4})?)",
            r"deadline\s+(?:is\s+)?(\w+\s+\d+(?:st|nd|rd|th)?(?:\s+\d{4})?)",
            r"by\s+(\w+\s+\d+(?:st|nd|rd|th)?(?:\s+\d{4})?)",
            r"due\s+in\s+(\d+)\s+(days?|hours?|weeks?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {"match": match.group(1), "full_text": text}
        return None

    def setup_standard_monitors(
        self,
        calendar_interval_minutes: int = 15,
        email_interval_minutes: int = 30,
        deadline_interval_minutes: int = 60,
        morning_briefing_hour: int = 8,
    ) -> None:
        try:
            self.scheduler.schedule_task(
                task_id="calendar_check",
                name="Calendar Check",
                callback_name="check_calendar",
                frequency=TaskFrequency.MINUTELY
                if calendar_interval_minutes < 60
                else TaskFrequency.HOURLY,
                minute=calendar_interval_minutes % 60,
            )
        except ValueError:
            pass

        try:
            self.scheduler.schedule_task(
                task_id="email_check",
                name="Email Check",
                callback_name="check_emails",
                frequency=TaskFrequency.MINUTELY
                if email_interval_minutes < 60
                else TaskFrequency.HOURLY,
                minute=email_interval_minutes % 60,
            )
        except ValueError:
            pass

        try:
            self.scheduler.schedule_task(
                task_id="deadline_check",
                name="Deadline Check",
                callback_name="check_deadlines",
                frequency=TaskFrequency.HOURLY,
            )
        except ValueError:
            pass

        try:
            self.scheduler.schedule_task(
                task_id="morning_briefing",
                name="Morning Briefing",
                callback_name="morning_briefing",
                frequency=TaskFrequency.DAILY,
                hour=morning_briefing_hour,
                days_of_week=[0, 1, 2, 3, 4],
            )
        except ValueError:
            pass

    async def run(self) -> None:
        self._running = True
        self.scheduler.start()
        while self._running:
            await asyncio.sleep(1)

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self.run())

    def stop(self) -> None:
        self._running = False
        self.scheduler.stop()
        if self._task:
            self._task.cancel()

    def add_deadline(
        self,
        key: str,
        deadline: datetime | str,
        description: str | None = None,
    ) -> None:
        from core.memory import ConversationMemory

        memory = ConversationMemory()

        if isinstance(deadline, str):
            deadline_str = deadline
        else:
            deadline_str = deadline.isoformat()

        memory.store_fact(
            key,
            {"date": deadline_str, "description": description or key},
            category="deadline",
        )

    def remove_deadline(self, key: str) -> bool:
        from core.memory import ConversationMemory

        memory = ConversationMemory()
        return memory.delete_fact(key)

    def get_pending_alerts(self) -> list[Alert]:
        return self.alerts.get_pending_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        return self.alerts.acknowledge_alert(alert_id)
