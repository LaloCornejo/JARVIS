from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .alerts import Alert, AlertManager, AlertPriority
from .scheduler import TaskFrequency, TaskScheduler
from .screen_context import (
    ScreenContext,
    ScreenContextExtractor,
)

log = logging.getLogger("jarvis.proactive.monitor")

# ── aproactive.log setup ────────────────────────────────────────

_PROACTIVE_LOG_PATH = Path("data/aproactive.log")
_PROACTIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_log_file = logging.getLogger("jarvis.proactive.file")
_log_file.setLevel(logging.DEBUG)
_log_file.propagate = False  # Don't double-emit to root logger

_fh = logging.FileHandler(str(_PROACTIVE_LOG_PATH), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh.setFormatter(_fmt)
_log_file.addHandler(_fh)


def _log_watch(level: str, msg: str, *args: Any) -> None:
    """Write a line to aproactive.log at the given level."""
    getattr(_log_file, level.lower(), _log_file.info)(msg, *args)


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
        self._screen_task: asyncio.Task | None = None
        self._screen_extractor: ScreenContextExtractor | None = None
        self._screen_interval: float = 15.0
        self._screen_enabled = False
        self._last_screen_context: ScreenContext | None = None
        self._screen_change_callback: Callable[[ScreenContext], Any] | None = None
        self._setup_default_tasks()

    @property
    def screen_context(self) -> ScreenContext | None:
        return self._last_screen_context

    def setup_screen_monitoring(
        self,
        poll_interval: float = 15.0,
        change_callback: Callable[[ScreenContext], Any] | None = None,
    ) -> None:
        self._screen_extractor = ScreenContextExtractor(poll_interval=poll_interval)
        self._screen_interval = poll_interval
        self._screen_enabled = True
        self._screen_change_callback = change_callback
        _log_watch(
            "INFO",
            "Screen monitoring enabled | interval=%ss | callback=%s",
            poll_interval,
            "yes" if change_callback else "no",
        )
        log.info(
            "Screen monitoring enabled (interval=%ss, callback=%s)",
            poll_interval,
            "yes" if change_callback else "no",
        )

    async def _screen_context_loop(self) -> None:
        if not self._screen_extractor:
            return
        log.info("Screen context polling loop started (interval=%ss)", self._screen_interval)
        while self._running:
            try:
                changed = await self._screen_extractor.poll_context()
                if changed is not None:
                    self._last_screen_context = changed
                    _log_watch(
                        "INFO",
                        "Screen changed | app=%s | title=%s | url=%s | file=%s",
                        changed.app_name,
                        changed.window_title[:60] if changed.window_title else "",
                        changed.url or "",
                        changed.open_file or "",
                    )

                    if self._screen_change_callback:
                        try:
                            await self._screen_change_callback(changed)
                        except Exception:
                            log.warning("Screen change callback failed", exc_info=True)

                    alert = self.alerts.create_alert(
                        alert_id=f"screen_ctx_{datetime.now().strftime('%H%M%S')}",
                        title=f"App: {changed.app_name}",
                        message=(
                            changed.url or changed.open_file or changed.window_title
                        )[:120],
                        priority=AlertPriority.LOW,
                        source="screen_context",
                        data=changed.to_dict(),
                    )
                    await self.alerts.deliver_alert_async(alert)
            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("Screen context poll error", exc_info=True)

            await asyncio.sleep(self._screen_interval)

    def _setup_default_tasks(self) -> None:
        self.scheduler.register_callback("check_calendar", self._check_calendar)
        self.scheduler.register_callback("check_deadlines", self._check_deadlines)
        self.scheduler.register_callback("morning_briefing", self._morning_briefing)
        self.scheduler.register_callback("check_emails", self._check_emails)

    def _log_watching_status(self) -> None:
        _log_watch("INFO", "═" * 60)
        _log_watch("INFO", "PROACTIVE MONITOR STATUS")
        _log_watch("INFO", "═" * 60)

        _log_watch("INFO", "[Screen Monitoring]")
        if self._screen_enabled and self._screen_extractor:
            _log_watch("INFO", "  Enabled: yes")
            _log_watch("INFO", "  Poll interval: %ss", self._screen_interval)
            _log_watch("INFO", "  Callback: %s", "yes" if self._screen_change_callback else "no")
        else:
            _log_watch("INFO", "  Enabled: no")

        _log_watch("INFO", "[Scheduled Tasks]")
        tasks = self.scheduler.list_tasks()
        if tasks:
            for t in tasks:
                dw = f" days={t.days_of_week}" if t.days_of_week else ""
                ht = f" at {t.hour:02d}:{t.minute:02d}" if t.hour is not None else ""
                _log_watch(
                    "INFO",
                    "  %s | freq=%s | next=%s%s%s | enabled=%s",
                    t.name, t.frequency.value, t.next_run.strftime("%Y-%m-%d %H:%M"), dw, ht,
                    t.enabled,
                )
        else:
            _log_watch("INFO", "  (none)")

        _log_watch("INFO", "[Custom Sources]")
        if self._sources:
            for name in self._sources:
                _log_watch("INFO", "  %s", name)
        else:
            _log_watch("INFO", "  (none)")

        _log_watch("INFO", "[Alert Manager]")
        pending = len(self.alerts.get_pending_alerts())
        _log_watch("INFO", "  Pending alerts: %s", pending)

        _log_watch("INFO", "═" * 60)

    def register_source(self, name: str, check_fn: Callable[[], Any]) -> None:
        self._sources[name] = check_fn
        _log_watch("INFO", "Custom source registered: %s", name)

    async def _check_calendar(self, data: dict | None = None) -> list[Alert]:
        try:
            from tools.integrations.google_calendar import get_gcal_client

            calendar = get_gcal_client()
            if not calendar.access_token:
                _log_watch("DEBUG", "Calendar check skipped (not authenticated)")
                return []

            now = datetime.now()
            events = await calendar.list_events(
                time_min=now,
                time_max=now + timedelta(hours=1),
                max_results=10,
            )

            _log_watch("INFO", "Calendar check: %d upcoming events", len(events))

            alerts = []
            for event in events:
                start = event.get("start", {})
                start_time = start.get("dateTime") or start.get("date")
                if start_time:
                    summary = event.get("summary", "Event")
                    _log_watch("INFO", "  Event: %s @ %s", summary, start_time)
                    alert = self.alerts.create_alert(
                        alert_id=f"calendar_{event.get('id', uuid.uuid4().hex)}",
                        title=f"Upcoming: {summary}",
                        message=f"Starting at {start_time}",
                        priority=AlertPriority.MEDIUM,
                        source="calendar",
                        data=event,
                    )
                    alerts.append(alert)
                    await self.alerts.deliver_alert_async(alert)
            return alerts
        except Exception:
            _log_watch("WARNING", "Calendar check failed")
            return []

    async def _check_emails(self, data: dict | None = None) -> list[Alert]:
        try:
            from tools.integrations.gmail import get_gmail_client

            gmail = get_gmail_client()
            if not gmail.access_token:
                _log_watch("DEBUG", "Email check skipped (not authenticated)")
                return []

            messages = await gmail.list_messages(query="is:unread", max_results=5)
            _log_watch("INFO", "Email check: %d unread messages", len(messages))

            def _extract_header(msg: dict, name: str) -> str:
                headers = msg.get("payload", {}).get("headers", [])
                for h in headers:
                    if h.get("name", "").lower() == name.lower():
                        return h.get("value", "")
                return ""

            alerts = []
            for msg in messages:
                subject = _extract_header(msg, "subject") or "No subject"
                sender = _extract_header(msg, "from") or "Unknown"
                _log_watch("INFO", "  Email from %s: %s", sender, subject)
                alert = self.alerts.create_alert(
                    alert_id=f"email_{msg.get('id', uuid.uuid4().hex)}",
                    title=f"Email: {subject}",
                    message=f"From: {sender}",
                    priority=AlertPriority.MEDIUM,
                    source="email",
                    data=msg,
                )
                alerts.append(alert)
                await self.alerts.deliver_alert_async(alert)
            return alerts
        except Exception:
            _log_watch("WARNING", "Email check failed")
            return []

    async def _check_deadlines(self, data: dict | None = None) -> list[Alert]:
        from core.memory import ConversationMemory

        memory = ConversationMemory()
        deadlines = memory.get_facts_by_category("deadline")

        _log_watch("INFO", "Deadline check: %d stored deadlines", len(deadlines))

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

                if not isinstance(deadline_str, str):
                    continue
                deadline = datetime.fromisoformat(deadline_str)
                time_until = deadline - now

                if timedelta(0) < time_until < timedelta(hours=24):
                    priority = (
                        AlertPriority.HIGH
                        if time_until < timedelta(hours=2)
                        else AlertPriority.MEDIUM
                    )
                    _log_watch(
                        "INFO",
                        "  Deadline: %s | due in %s | priority=%s",
                        description,
                        self._format_timedelta(time_until),
                        priority.name,
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
                else:
                    _log_watch("DEBUG", "  Deadline: %s | due %s (outside 24h window)", description, deadline.date())
            except (ValueError, TypeError):
                continue
        return alerts

    async def _morning_briefing(self, data: dict | None = None) -> Alert:
        briefing_parts = []
        now = datetime.now()

        try:
            from tools.integrations.google_calendar import get_gcal_client

            calendar = get_gcal_client()
            if calendar.access_token:
                today_end = now.replace(hour=23, minute=59, second=59)
                events = await calendar.list_events(
                    time_min=now,
                    time_max=today_end,
                    max_results=20,
                )
                if events:
                    briefing_parts.append(f"You have {len(events)} events today")
        except Exception as e:
            _log_watch("WARNING", "Morning briefing calendar fetch failed: %s", e)

        try:
            from tools.integrations.gmail import get_gmail_client

            gmail = get_gmail_client()
            if gmail.access_token:
                unread = await gmail.list_messages(query="is:unread", max_results=1)
                if unread:
                    briefing_parts.append("Unread emails waiting")
        except Exception as e:
            _log_watch("WARNING", "Morning briefing email fetch failed: %s", e)

        from core.memory import ConversationMemory

        memory = ConversationMemory()
        deadlines = memory.get_facts_by_category("deadline")
        today_deadlines = []
        for key, value in deadlines.items():
            try:
                if isinstance(value, dict):
                    deadline_str = value.get("date")
                else:
                    deadline_str = value
                if not isinstance(deadline_str, str):
                    continue
                deadline = datetime.fromisoformat(deadline_str)
                if deadline.date() == now.date():
                    today_deadlines.append(key)
            except (ValueError, TypeError):
                continue
        if today_deadlines:
            briefing_parts.append(f"{len(today_deadlines)} deadlines today")

        message = ". ".join(briefing_parts) if briefing_parts else "No significant items for today"
        _log_watch("INFO", "Morning briefing: %s", message)

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

        self._log_watching_status()

        if self._screen_enabled:
            _log_watch("INFO", "Screen context loop started")
            self._screen_task = asyncio.create_task(self._screen_context_loop())

        _log_watch("INFO", "Proactive monitor running")

        while self._running:
            await asyncio.sleep(1)

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self.run())

    def stop(self) -> None:
        self._running = False
        self.scheduler.stop()
        if self._screen_task:
            self._screen_task.cancel()
            self._screen_task = None
        if self._task:
            self._task.cancel()
        _log_watch("INFO", "Proactive monitor stopped")

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
