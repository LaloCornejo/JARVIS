"""
Workflow Trigger System for JARVIS.

Provides time-based, event-based, and condition-based triggers for workflows.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of workflow triggers"""

    TIME = "time"
    EVENT = "event"
    CONDITION = "condition"
    COMPOSITE = "composite"


class TriggerStatus(Enum):
    """Trigger activation status"""

    ACTIVE = "active"
    PAUSED = "paused"
    TRIGGERED = "triggered"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class TriggerContext:
    """Context passed to trigger callbacks"""

    trigger_id: str
    trigger_type: TriggerType
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    event_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    workflow_id: Optional[str] = None


class Trigger(ABC):
    """Base class for all triggers"""

    def __init__(
        self,
        trigger_id: str,
        name: str,
        trigger_type: TriggerType,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.trigger_id = trigger_id
        self.name = name
        self.trigger_type = trigger_type
        self.workflow_id = workflow_id
        self.metadata = metadata or {}
        self.status = TriggerStatus.ACTIVE
        self.created_at = datetime.now()
        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0
        self.callbacks: List[Callable[[TriggerContext], None]] = []
        self._error_count = 0
        self._max_errors = 5

    @abstractmethod
    async def check(self) -> bool:
        """Check if trigger condition is met"""
        pass

    @abstractmethod
    async def start_monitoring(self):
        """Start monitoring for trigger conditions"""
        pass

    @abstractmethod
    async def stop_monitoring(self):
        """Stop monitoring"""
        pass

    def add_callback(self, callback: Callable[[TriggerContext], None]) -> None:
        """Add a callback to be called when trigger fires"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[TriggerContext], None]) -> None:
        """Remove a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def fire(self, event_data: Optional[Dict[str, Any]] = None) -> None:
        """Fire the trigger and call all callbacks"""
        if self.status == TriggerStatus.DISABLED:
            return

        self.last_triggered = datetime.now()
        self.trigger_count += 1
        self.status = TriggerStatus.TRIGGERED

        context = TriggerContext(
            trigger_id=self.trigger_id,
            trigger_type=self.trigger_type,
            trigger_data=self.metadata,
            event_data=event_data or {},
            workflow_id=self.workflow_id,
        )

        log.info(f"Trigger '{self.name}' fired for workflow '{self.workflow_id}'")

        # Call all callbacks
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context)
                else:
                    callback(context)
            except Exception as e:
                log.error(f"Error in trigger callback: {e}")
                self._error_count += 1

                if self._error_count >= self._max_errors:
                    self.status = TriggerStatus.ERROR
                    log.error(f"Trigger '{self.name}' disabled due to too many errors")
                    break

        # Reset status after firing
        if self.status != TriggerStatus.ERROR:
            self.status = TriggerStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert trigger to dictionary"""
        return {
            "trigger_id": self.trigger_id,
            "name": self.name,
            "type": self.trigger_type.value,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
            "metadata": self.metadata,
        }

    def pause(self) -> None:
        """Pause the trigger"""
        self.status = TriggerStatus.PAUSED
        log.info(f"Trigger '{self.name}' paused")

    def resume(self) -> None:
        """Resume the trigger"""
        if self.status == TriggerStatus.PAUSED:
            self.status = TriggerStatus.ACTIVE
            log.info(f"Trigger '{self.name}' resumed")

    def disable(self) -> None:
        """Disable the trigger"""
        self.status = TriggerStatus.DISABLED
        log.info(f"Trigger '{self.name}' disabled")


class TimeTrigger(Trigger):
    """
    Time-based trigger.

    Supports:
    - One-time triggers at specific datetime
    - Recurring triggers (interval-based)
    - Cron-like scheduling
    """

    def __init__(
        self,
        trigger_id: str,
        name: str,
        workflow_id: str,
        trigger_at: Optional[datetime] = None,
        interval: Optional[timedelta] = None,
        cron_expression: Optional[str] = None,
        timezone: str = "local",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(trigger_id, name, TriggerType.TIME, workflow_id, metadata)
        self.trigger_at = trigger_at
        self.interval = interval
        self.cron_expression = cron_expression
        self.timezone = timezone
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def check(self) -> bool:
        """Check if time condition is met"""
        now = datetime.now()

        if self.trigger_at and now >= self.trigger_at:
            # One-time trigger
            if self.trigger_count == 0:
                return True
            return False

        if self.interval and self.last_triggered:
            if now - self.last_triggered >= self.interval:
                return True

        if self.cron_expression:
            return self._check_cron(now)

        return False

    async def start_monitoring(self):
        """Start monitoring time conditions"""
        if self._monitoring_task and not self._monitoring_task.done():
            return

        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        log.info(f"Time trigger '{self.name}' started monitoring")

    async def stop_monitoring(self):
        """Stop monitoring"""
        if self._monitoring_task:
            self._stop_event.set()
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            log.info(f"Time trigger '{self.name}' stopped monitoring")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                if await self.check():
                    await self.fire()

                    # For one-time triggers, disable after firing
                    if self.trigger_at and not self.interval and not self.cron_expression:
                        self.disable()
                        break

                # Wait before next check
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Error in time trigger monitor: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    def _check_cron(self, now: datetime) -> bool:
        """Check if current time matches cron expression"""
        if not self.cron_expression:
            return False

        try:
            # Simple cron parsing (minute hour day month day_of_week)
            parts = self.cron_expression.split()
            if len(parts) != 5:
                return False

            minute, hour, day, month, day_of_week = parts

            # Check each field
            if not self._match_cron_field(minute, now.minute, 0, 59):
                return False
            if not self._match_cron_field(hour, now.hour, 0, 23):
                return False
            if not self._match_cron_field(day, now.day, 1, 31):
                return False
            if not self._match_cron_field(month, now.month, 1, 12):
                return False
            if not self._match_cron_field(day_of_week, now.weekday(), 0, 6):
                return False

            # Check if we already triggered this minute
            if self.last_triggered:
                time_since_last = now - self.last_triggered
                if time_since_last < timedelta(minutes=1):
                    return False

            return True

        except Exception as e:
            log.error(f"Error parsing cron expression: {e}")
            return False

    def _match_cron_field(self, pattern: str, value: int, min_val: int, max_val: int) -> bool:
        """Check if value matches cron pattern"""
        if pattern == "*":
            return True

        # Handle ranges (e.g., "1-5")
        if "-" in pattern:
            start, end = pattern.split("-")
            return int(start) <= value <= int(end)

        # Handle lists (e.g., "1,3,5")
        if "," in pattern:
            return str(value) in pattern.split(",")

        # Handle step values (e.g., "*/5")
        if "/" in pattern:
            base, step = pattern.split("/")
            if base == "*":
                return value % int(step) == 0

        # Direct match
        return str(value) == pattern

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base = super().to_dict()
        base.update(
            {
                "trigger_at": self.trigger_at.isoformat() if self.trigger_at else None,
                "interval_seconds": self.interval.total_seconds() if self.interval else None,
                "cron_expression": self.cron_expression,
                "timezone": self.timezone,
            }
        )
        return base


class EventTrigger(Trigger):
    """
    Event-based trigger.

    Fires when specific events occur in the system.
    """

    def __init__(
        self,
        trigger_id: str,
        name: str,
        workflow_id: str,
        event_type: str,
        event_filter: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(trigger_id, name, TriggerType.EVENT, workflow_id, metadata)
        self.event_type = event_type
        self.event_filter = event_filter or {}
        self._event_bus = None  # Will be set by trigger manager

    async def check(self) -> bool:
        """Event triggers don't check - they fire when events are received"""
        return False

    async def start_monitoring(self):
        """Subscribe to event bus"""
        # This would subscribe to the event bus
        log.info(f"Event trigger '{self.name}' subscribed to '{self.event_type}' events")

    async def stop_monitoring(self):
        """Unsubscribe from event bus"""
        log.info(f"Event trigger '{self.name}' unsubscribed")

    def matches_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Check if an event matches this trigger"""
        if event_type != self.event_type:
            return False

        # Check filter conditions
        for key, value in self.event_filter.items():
            if key not in event_data:
                return False

            event_value = event_data[key]

            # Handle different filter types
            if isinstance(value, dict):
                # Complex filter (e.g., {"$gt": 10})
                if not self._apply_complex_filter(event_value, value):
                    return False
            elif isinstance(value, list):
                # List of acceptable values
                if event_value not in value:
                    return False
            elif isinstance(value, str) and ("*" in value or "?" in value):
                # Glob pattern matching
                if not fnmatch.fnmatch(str(event_value), value):
                    return False
            else:
                # Direct comparison
                if event_value != value:
                    return False

        return True

    def _apply_complex_filter(self, value: Any, filter_spec: Dict[str, Any]) -> bool:
        """Apply complex filter operators"""
        for operator, operand in filter_spec.items():
            if operator == "$gt":
                if not (isinstance(value, (int, float)) and value > operand):
                    return False
            elif operator == "$gte":
                if not (isinstance(value, (int, float)) and value >= operand):
                    return False
            elif operator == "$lt":
                if not (isinstance(value, (int, float)) and value < operand):
                    return False
            elif operator == "$lte":
                if not (isinstance(value, (int, float)) and value <= operand):
                    return False
            elif operator == "$eq":
                if value != operand:
                    return False
            elif operator == "$ne":
                if value == operand:
                    return False
            elif operator == "$in":
                if value not in operand:
                    return False
            elif operator == "$nin":
                if value in operand:
                    return False
            elif operator == "$regex":
                if not re.search(operand, str(value)):
                    return False

        return True

    async def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle incoming event"""
        if self.matches_event(event_type, event_data):
            await self.fire(event_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base = super().to_dict()
        base.update(
            {
                "event_type": self.event_type,
                "event_filter": self.event_filter,
            }
        )
        return base


class ConditionTrigger(Trigger):
    """
    Condition-based trigger.

    Fires when a specific condition becomes true.
    """

    def __init__(
        self,
        trigger_id: str,
        name: str,
        workflow_id: str,
        condition_type: str,
        condition_config: Dict[str, Any],
        check_interval: timedelta = timedelta(seconds=30),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(trigger_id, name, TriggerType.CONDITION, workflow_id, metadata)
        self.condition_type = condition_type
        self.condition_config = condition_config
        self.check_interval = check_interval
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._was_true = False  # Track state for edge detection

    async def check(self) -> bool:
        """Check if condition is met"""
        try:
            is_true = await self._evaluate_condition()

            # Edge detection - only fire on transition from false to true
            if is_true and not self._was_true:
                self._was_true = True
                return True

            if not is_true:
                self._was_true = False

            return False

        except Exception as e:
            log.error(f"Error evaluating condition: {e}")
            return False

    async def _evaluate_condition(self) -> bool:
        """Evaluate the condition"""
        if self.condition_type == "file_exists":
            path = self.condition_config.get("path", "")
            return Path(path).exists()

        elif self.condition_type == "file_size":
            path = self.condition_config.get("path", "")
            min_size = self.condition_config.get("min_size", 0)
            if Path(path).exists():
                return Path(path).stat().st_size >= min_size
            return False

        elif self.condition_type == "system_load":
            import psutil

            max_cpu = self.condition_config.get("max_cpu", 90)
            max_memory = self.condition_config.get("max_memory", 90)

            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            return cpu_percent < max_cpu and memory.percent < max_memory

        elif self.condition_type == "process_running":
            import psutil

            process_name = self.condition_config.get("process_name", "")
            for proc in psutil.process_iter(["name"]):
                if proc.info["name"] == process_name:
                    return True
            return False

        elif self.condition_type == "http_status":
            import httpx

            url = self.condition_config.get("url", "")
            expected_status = self.condition_config.get("expected_status", 200)

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=5.0)
                    return response.status_code == expected_status
            except Exception:
                return False

        elif self.condition_type == "custom":
            # Custom condition with user-provided function
            custom_check = self.condition_config.get("check_function")
            if custom_check and callable(custom_check):
                result = custom_check()
                if asyncio.iscoroutinefunction(custom_check):
                    result = await result
                return bool(result)

        return False

    async def start_monitoring(self):
        """Start monitoring condition"""
        if self._monitoring_task and not self._monitoring_task.done():
            return

        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        log.info(f"Condition trigger '{self.name}' started monitoring")

    async def stop_monitoring(self):
        """Stop monitoring"""
        if self._monitoring_task:
            self._stop_event.set()
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            log.info(f"Condition trigger '{self.name}' stopped monitoring")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                if await self.check():
                    await self.fire()

                # Wait for next check
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.check_interval.total_seconds()
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Error in condition trigger monitor: {e}")
                await asyncio.sleep(5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base = super().to_dict()
        base.update(
            {
                "condition_type": self.condition_type,
                "condition_config": self.condition_config,
                "check_interval_seconds": self.check_interval.total_seconds(),
            }
        )
        return base


class CompositeTrigger(Trigger):
    """
    Composite trigger that combines multiple triggers.

    Supports AND, OR, and NOT logic.
    """

    def __init__(
        self,
        trigger_id: str,
        name: str,
        workflow_id: str,
        operator: str,  # "and", "or", "not"
        triggers: List[Trigger],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(trigger_id, name, TriggerType.COMPOSITE, workflow_id, metadata)
        self.operator = operator
        self.triggers = triggers
        self._trigger_states: Dict[str, bool] = {}

        # Add callback to each sub-trigger
        for trigger in triggers:
            trigger.add_callback(self._on_sub_trigger)

    async def check(self) -> bool:
        """Evaluate composite condition"""
        if not self.triggers:
            return False

        if self.operator == "and":
            return all(self._trigger_states.get(t.trigger_id, False) for t in self.triggers)

        elif self.operator == "or":
            return any(self._trigger_states.get(t.trigger_id, False) for t in self.triggers)

        elif self.operator == "not":
            if len(self.triggers) == 1:
                return not self._trigger_states.get(self.triggers[0].trigger_id, False)
            return False

        return False

    async def _on_sub_trigger(self, context: TriggerContext) -> None:
        """Handle sub-trigger firing"""
        self._trigger_states[context.trigger_id] = True

        # Check if composite condition is met
        if await self.check():
            await self.fire(context.event_data)

        # Reset state after checking
        if self.operator == "and":
            # For AND, all must fire together, so don't reset
            pass
        else:
            # For OR/NOT, reset after firing
            self._trigger_states[context.trigger_id] = False

    async def start_monitoring(self):
        """Start monitoring all sub-triggers"""
        for trigger in self.triggers:
            await trigger.start_monitoring()
        log.info(f"Composite trigger '{self.name}' started monitoring")

    async def stop_monitoring(self):
        """Stop monitoring all sub-triggers"""
        for trigger in self.triggers:
            await trigger.stop_monitoring()
        log.info(f"Composite trigger '{self.name}' stopped monitoring")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base = super().to_dict()
        base.update(
            {
                "operator": self.operator,
                "triggers": [t.to_dict() for t in self.triggers],
            }
        )
        return base


class TriggerManager:
    """
    Manages all workflow triggers.

    Provides centralized trigger registration, monitoring, and event distribution.
    """

    def __init__(self):
        self.triggers: Dict[str, Trigger] = {}
        self.event_triggers: Dict[str, List[EventTrigger]] = {}
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self):
        """Start all trigger monitoring"""
        async with self._lock:
            self._running = True
            for trigger in self.triggers.values():
                if trigger.status == TriggerStatus.ACTIVE:
                    await trigger.start_monitoring()
        log.info("Trigger manager started")

    async def stop(self):
        """Stop all trigger monitoring"""
        async with self._lock:
            self._running = False
            for trigger in self.triggers.values():
                await trigger.stop_monitoring()
        log.info("Trigger manager stopped")

    async def register_trigger(self, trigger: Trigger) -> None:
        """Register a trigger"""
        async with self._lock:
            self.triggers[trigger.trigger_id] = trigger

            # Index event triggers by type
            if isinstance(trigger, EventTrigger):
                event_type = trigger.event_type
                if event_type not in self.event_triggers:
                    self.event_triggers[event_type] = []
                self.event_triggers[event_type].append(trigger)

            # Start monitoring if already running
            if self._running and trigger.status == TriggerStatus.ACTIVE:
                await trigger.start_monitoring()

        log.info(f"Registered trigger: {trigger.name}")

    async def unregister_trigger(self, trigger_id: str) -> bool:
        """Unregister a trigger"""
        async with self._lock:
            if trigger_id not in self.triggers:
                return False

            trigger = self.triggers[trigger_id]
            await trigger.stop_monitoring()

            # Remove from event triggers index
            if isinstance(trigger, EventTrigger):
                event_type = trigger.event_type
                if event_type in self.event_triggers:
                    if trigger in self.event_triggers[event_type]:
                        self.event_triggers[event_type].remove(trigger)

            del self.triggers[trigger_id]

        log.info(f"Unregistered trigger: {trigger_id}")
        return True

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish an event to matching triggers"""
        if event_type not in self.event_triggers:
            return

        for trigger in self.event_triggers[event_type]:
            try:
                await trigger.handle_event(event_type, event_data)
            except Exception as e:
                log.error(f"Error handling event in trigger {trigger.name}: {e}")

    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """Get a trigger by ID"""
        return self.triggers.get(trigger_id)

    def get_triggers_for_workflow(self, workflow_id: str) -> List[Trigger]:
        """Get all triggers for a workflow"""
        return [t for t in self.triggers.values() if t.workflow_id == workflow_id]

    def list_triggers(
        self, trigger_type: Optional[TriggerType] = None, status: Optional[TriggerStatus] = None
    ) -> List[Dict[str, Any]]:
        """List triggers with optional filtering"""
        results = []

        for trigger in self.triggers.values():
            if trigger_type and trigger.trigger_type != trigger_type:
                continue
            if status and trigger.status != status:
                continue
            results.append(trigger.to_dict())

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get trigger statistics"""
        total = len(self.triggers)
        by_type = {}
        by_status = {}

        for trigger in self.triggers.values():
            by_type[trigger.trigger_type.value] = by_type.get(trigger.trigger_type.value, 0) + 1
            by_status[trigger.status.value] = by_status.get(trigger.status.value, 0) + 1

        return {
            "total_triggers": total,
            "by_type": by_type,
            "by_status": by_status,
            "total_fires": sum(t.trigger_count for t in self.triggers.values()),
        }

    async def check_all(self) -> List[str]:
        """Check all triggers and return list of triggered IDs"""
        triggered = []
        for trigger in self.triggers.values():
            try:
                if await trigger.check():
                    triggered.append(trigger.trigger_id)
                    await trigger.fire()
            except Exception as e:
                log.error(f"Error checking trigger {trigger.name}: {e}")
        return triggered


# Global trigger manager instance
_trigger_manager: Optional[TriggerManager] = None


def get_trigger_manager() -> TriggerManager:
    """Get the global trigger manager instance"""
    global _trigger_manager
    if _trigger_manager is None:
        _trigger_manager = TriggerManager()
    return _trigger_manager


__all__ = [
    "Trigger",
    "TriggerType",
    "TriggerStatus",
    "TriggerContext",
    "TimeTrigger",
    "EventTrigger",
    "ConditionTrigger",
    "CompositeTrigger",
    "TriggerManager",
    "get_trigger_manager",
]
