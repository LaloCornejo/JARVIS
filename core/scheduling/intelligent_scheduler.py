"""
Intelligent Smart Scheduler for JARVIS.

This module provides advanced scheduling capabilities including:
- Predictive scheduling based on user patterns
- Intelligent task prioritization and conflict resolution
- Context-aware scheduling decisions
- Resource optimization and load balancing
- Automated meeting and event management
- Smart reminder timing and notification management
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TaskType(Enum):
    """Types of schedulable tasks"""

    REMINDER = "reminder"
    MEETING = "meeting"
    DEADLINE = "deadline"
    MAINTENANCE = "maintenance"
    AUTOMATION = "automation"
    NOTIFICATION = "notification"


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""

    id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    scheduled_time: datetime
    duration_minutes: int = 0
    recurrence: Optional[str] = None  # "daily", "weekly", "monthly", or cron expression
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "task_type": self.task_type.value,
            "priority": self.priority.value,
            "scheduled_time": self.scheduled_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "recurrence": self.recurrence,
            "dependencies": self.dependencies,
            "context": self.context,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "execution_count": self.execution_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledTask":
        """Create task from dictionary"""
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            task_type=TaskType(data["task_type"]),
            priority=TaskPriority(data["priority"]),
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            duration_minutes=data.get("duration_minutes", 0),
            recurrence=data.get("recurrence"),
            dependencies=data.get("dependencies", []),
            context=data.get("context", {}),
            is_active=data.get("is_active", True),
            execution_count=data.get("execution_count", 0),
        )

        if data.get("created_at"):
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_executed"):
            task.last_executed = datetime.fromisoformat(data["last_executed"])

        return task


@dataclass
class UserPattern:
    """Represents learned user behavior patterns"""

    pattern_type: str
    description: str
    confidence: float
    data_points: List[Dict[str, Any]]
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "data_points": self.data_points,
            "last_updated": self.last_updated.isoformat(),
        }


class PredictiveEngine:
    """Engine for predicting optimal scheduling times"""

    def __init__(self):
        self.user_patterns: Dict[str, UserPattern] = {}
        self.historical_data: List[Dict[str, Any]] = []
        self.max_history_days = 90

    def record_user_activity(
        self, activity_type: str, timestamp: datetime, context: Dict[str, Any]
    ):
        """Record user activity for pattern learning"""
        activity = {
            "type": activity_type,
            "timestamp": timestamp,
            "context": context,
            "day_of_week": timestamp.weekday(),
            "hour": timestamp.hour,
            "minute": timestamp.minute,
        }

        self.historical_data.append(activity)

        # Keep only recent data
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        self.historical_data = [
            h for h in self.historical_data if datetime.fromisoformat(h["timestamp"]) > cutoff_date
        ]

        # Update patterns
        self._update_patterns(activity_type)

    def predict_optimal_time(
        self, task_type: str, preferred_time: Optional[datetime] = None
    ) -> datetime:
        """Predict the optimal time for a task based on user patterns"""
        if task_type not in self.user_patterns:
            # No pattern data, return preferred time or default
            return preferred_time or datetime.now()

        pattern = self.user_patterns[task_type]

        if pattern.confidence < 0.3:
            # Low confidence, stick with preferred time
            return preferred_time or datetime.now()

        # Find optimal time based on pattern
        optimal_hour, optimal_minute = self._find_optimal_time_from_pattern(pattern)

        # Create datetime for optimal time today
        now = datetime.now()
        optimal_time = now.replace(
            hour=optimal_hour, minute=optimal_minute, second=0, microsecond=0
        )

        # If optimal time has passed today, schedule for tomorrow
        if optimal_time < now:
            optimal_time += timedelta(days=1)

        return optimal_time

    def _find_optimal_time_from_pattern(self, pattern: UserPattern) -> Tuple[int, int]:
        """Find optimal hour and minute from pattern data"""
        # Simple analysis: find most common hour and minute
        hours = [dp.get("hour", 9) for dp in pattern.data_points]
        minutes = [dp.get("minute", 0) for dp in pattern.data_points]

        if hours:
            optimal_hour = max(set(hours), key=hours.count)
        else:
            optimal_hour = 9  # Default to 9 AM

        if minutes:
            # Round minutes to nearest 15-minute interval
            optimal_minute = round(max(set(minutes), key=minutes.count) / 15) * 15
            optimal_minute = min(optimal_minute, 45)  # Cap at 45 minutes
        else:
            optimal_minute = 0

        return optimal_hour, optimal_minute

    def _update_patterns(self, activity_type: str):
        """Update patterns based on new activity data"""
        recent_activities = [h for h in self.historical_data if h["type"] == activity_type][
            -50:
        ]  # Last 50 activities of this type

        if len(recent_activities) < 5:
            return  # Not enough data

        # Calculate confidence based on data consistency
        hours = [a["hour"] for a in recent_activities]
        hour_consistency = len(set(hours)) / len(hours)  # Lower is more consistent

        confidence = min(1.0, len(recent_activities) / 20.0) * (1.0 - hour_consistency)

        pattern = UserPattern(
            pattern_type=activity_type,
            description=f"User pattern for {activity_type}",
            confidence=confidence,
            data_points=recent_activities[-20:],  # Keep last 20 data points
        )

        self.user_patterns[activity_type] = pattern


class ConflictResolver:
    """Handles scheduling conflicts and optimization"""

    def __init__(self):
        self.resolution_strategies = {
            "postpone": self._postpone_task,
            "reschedule": self._reschedule_task,
            "prioritize": self._prioritize_task,
            "split": self._split_task,
            "merge": self._merge_tasks,
        }

    def resolve_conflict(
        self, tasks: List[ScheduledTask], strategy: str = "auto"
    ) -> List[ScheduledTask]:
        """Resolve scheduling conflicts between tasks"""
        if strategy == "auto":
            strategy = self._choose_optimal_strategy(tasks)

        if strategy in self.resolution_strategies:
            return self.resolution_strategies[strategy](tasks)
        else:
            log.warning(f"Unknown conflict resolution strategy: {strategy}")
            return tasks

    def _choose_optimal_strategy(self, tasks: List[ScheduledTask]) -> str:
        """Choose the optimal conflict resolution strategy"""
        if len(tasks) == 2:
            # Compare priorities
            high_priority = max(tasks, key=lambda t: t.priority.value)
            low_priority = min(tasks, key=lambda t: t.priority.value)

            if high_priority.priority.value >= TaskPriority.HIGH.value:
                return "prioritize"  # Keep high priority, postpone low priority
            else:
                return "reschedule"  # Try to reschedule both

        elif len(tasks) > 2:
            return "postpone"  # Postpone lower priority tasks

        return "reschedule"

    def _postpone_task(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Postpone lower priority tasks"""
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)

        for i in range(1, len(sorted_tasks)):
            task = sorted_tasks[i]
            # Postpone by 1 hour
            task.scheduled_time += timedelta(hours=1)

        return sorted_tasks

    def _reschedule_task(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Reschedule tasks to avoid conflicts"""
        # Simple rescheduling: space them out by 30 minutes
        base_time = min(t.scheduled_time for t in tasks)

        for i, task in enumerate(tasks):
            task.scheduled_time = base_time + timedelta(minutes=i * 30)

        return tasks

    def _prioritize_task(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Keep highest priority task, reschedule others"""
        highest_priority = max(tasks, key=lambda t: t.priority.value)
        others = [t for t in tasks if t != highest_priority]

        # Reschedule lower priority tasks
        for task in others:
            task.scheduled_time += timedelta(hours=1)

        return [highest_priority] + others

    def _split_task(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Split conflicting tasks into smaller time slots"""
        # For now, just reschedule them
        return self._reschedule_task(tasks)

    def _merge_tasks(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Merge related tasks (not implemented yet)"""
        return tasks


class IntelligentScheduler:
    """Main intelligent scheduling system"""

    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.predictive_engine = PredictiveEngine()
        self.conflict_resolver = ConflictResolver()
        self.active_tasks: Set[str] = set()
        self.check_interval = 60  # Check every minute
        self.is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the scheduler"""
        if not self.is_running:
            self.is_running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            log.info("Intelligent scheduler initialized")

    async def shutdown(self):
        """Shutdown the scheduler"""
        self.is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        log.info("Intelligent scheduler shutdown")

    async def schedule_task(
        self,
        title: str,
        description: str,
        task_type: TaskType,
        scheduled_time: Optional[datetime] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        duration_minutes: int = 0,
        recurrence: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Schedule a new task with intelligent timing"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Use predictive engine to find optimal time if not specified
        if scheduled_time is None:
            scheduled_time = self.predictive_engine.predict_optimal_time(
                task_type.value,
                datetime.now() + timedelta(minutes=5),  # Default to 5 minutes from now
            )

        # Check for conflicts and resolve them
        conflicts = await self._find_conflicts(scheduled_time, duration_minutes)
        if conflicts:
            resolved_tasks = self.conflict_resolver.resolve_conflict(
                [conflicts[0]] + [self.tasks[task_id]]
            )
            if len(resolved_tasks) > 1:
                # Update the scheduled time based on conflict resolution
                scheduled_time = resolved_tasks[-1].scheduled_time

        task = ScheduledTask(
            id=task_id,
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            scheduled_time=scheduled_time,
            duration_minutes=duration_minutes,
            recurrence=recurrence,
            context=context or {},
        )

        self.tasks[task_id] = task
        log.info(f"Scheduled task '{title}' for {scheduled_time}")
        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.is_active = False
            self.active_tasks.discard(task_id)
            log.info(f"Cancelled task '{task.title}'")
            return True
        return False

    async def get_upcoming_tasks(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get list of upcoming tasks within specified hours"""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = []
        for task in self.tasks.values():
            if task.is_active and task.scheduled_time >= now and task.scheduled_time <= cutoff:
                upcoming.append(task.to_dict())

        # Sort by scheduled time
        upcoming.sort(key=lambda t: t["scheduled_time"])
        return upcoming

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific task"""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None

    async def update_task_priority(self, task_id: str, new_priority: TaskPriority) -> bool:
        """Update the priority of a scheduled task"""
        if task_id in self.tasks:
            self.tasks[task_id].priority = new_priority
            log.info(
                f"Updated priority of task '{self.tasks[task_id].title}' to {new_priority.value}"
            )
            return True
        return False

    async def reschedule_task(self, task_id: str, new_time: datetime) -> bool:
        """Reschedule a task to a new time"""
        if task_id in self.tasks:
            old_time = self.tasks[task_id].scheduled_time
            self.tasks[task_id].scheduled_time = new_time
            log.info(
                f"Rescheduled task '{self.tasks[task_id].title}' from {old_time} to {new_time}"
            )
            return True
        return False

    async def _scheduler_loop(self):
        """Main scheduler loop that checks for due tasks"""
        while self.is_running:
            try:
                await self._check_due_tasks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)

    async def _check_due_tasks(self):
        """Check for tasks that are due and execute them"""
        now = datetime.now()

        due_tasks = []
        for task_id, task in self.tasks.items():
            if task.is_active and task.scheduled_time <= now and task_id not in self.active_tasks:
                due_tasks.append(task_id)

        for task_id in due_tasks:
            await self._execute_task(task_id)

    async def _execute_task(self, task_id: str):
        """Execute a scheduled task"""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        self.active_tasks.add(task_id)

        try:
            log.info(f"Executing scheduled task: {task.title}")

            # Execute based on task type
            result = await self._perform_task_execution(task)

            # Update task metadata
            task.last_executed = datetime.now()
            task.execution_count += 1

            # Handle recurrence
            if task.recurrence:
                await self._schedule_recurring_task(task)

            log.info(f"Successfully executed task: {task.title}")

        except Exception as e:
            log.error(f"Failed to execute task '{task.title}': {e}")
        finally:
            self.active_tasks.discard(task_id)

    async def _perform_task_execution(self, task: ScheduledTask) -> Any:
        """Perform the actual task execution"""
        # This would integrate with the broader JARVIS system
        # For now, we'll simulate execution based on task type

        if task.task_type == TaskType.REMINDER:
            # Send reminder notification
            message = f"Reminder: {task.title}"
            if task.description:
                message += f"\n{task.description}"
            # In real implementation, send to notification system
            log.info(f"Reminder sent: {message}")
            return {"type": "reminder", "message": message}

        elif task.task_type == TaskType.MEETING:
            # Meeting reminder
            message = f"Meeting: {task.title}"
            if task.description:
                message += f"\n{task.description}"
            log.info(f"Meeting reminder: {message}")
            return {"type": "meeting", "message": message}

        elif task.task_type == TaskType.DEADLINE:
            # Deadline reminder
            message = f"Deadline approaching: {task.title}"
            if task.description:
                message += f"\n{task.description}"
            log.info(f"Deadline reminder: {message}")
            return {"type": "deadline", "message": message}

        elif task.task_type == TaskType.MAINTENANCE:
            # System maintenance task
            log.info(f"Running maintenance task: {task.title}")
            # In real implementation, run maintenance scripts
            return {"type": "maintenance", "task": task.title}

        elif task.task_type == TaskType.AUTOMATION:
            # Automation workflow
            log.info(f"Running automation: {task.title}")
            # In real implementation, trigger workflow
            return {"type": "automation", "task": task.title}

        elif task.task_type == TaskType.NOTIFICATION:
            # General notification
            log.info(f"Sending notification: {task.title}")
            return {"type": "notification", "message": task.title}

        else:
            log.warning(f"Unknown task type: {task.task_type}")
            return {"type": "unknown", "task": task.title}

    async def _schedule_recurring_task(self, task: ScheduledTask):
        """Schedule the next occurrence of a recurring task"""
        if not task.recurrence:
            return

        next_time = task.scheduled_time

        if task.recurrence == "daily":
            next_time += timedelta(days=1)
        elif task.recurrence == "weekly":
            next_time += timedelta(weeks=1)
        elif task.recurrence == "monthly":
            # Approximate month as 30 days
            next_time += timedelta(days=30)
        else:
            # For cron expressions, we'd need a cron parser
            # For now, skip complex recurrence
            log.warning(f"Complex recurrence not supported: {task.recurrence}")
            return

        # Create new task instance
        new_task = ScheduledTask(
            id=f"{task.id}_{task.execution_count}",
            title=task.title,
            description=task.description,
            task_type=task.task_type,
            priority=task.priority,
            scheduled_time=next_time,
            duration_minutes=task.duration_minutes,
            recurrence=task.recurrence,
            dependencies=task.dependencies.copy(),
            context=task.context.copy(),
        )

        self.tasks[new_task.id] = new_task
        log.info(f"Scheduled recurring task '{task.title}' for {next_time}")

    async def _find_conflicts(
        self, scheduled_time: datetime, duration_minutes: int
    ) -> List[ScheduledTask]:
        """Find tasks that conflict with the given time slot"""
        end_time = scheduled_time + timedelta(minutes=duration_minutes)

        conflicts = []
        for task in self.tasks.values():
            if not task.is_active:
                continue

            task_end = task.scheduled_time + timedelta(minutes=task.duration_minutes)

            # Check for time overlap
            if scheduled_time < task_end and end_time > task.scheduled_time:
                conflicts.append(task)

        return conflicts

    def record_user_activity(self, activity_type: str, context: Optional[Dict[str, Any]] = None):
        """Record user activity for pattern learning"""
        self.predictive_engine.record_user_activity(activity_type, datetime.now(), context or {})

    async def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics and performance metrics"""
        total_tasks = len(self.tasks)
        active_tasks = len([t for t in self.tasks.values() if t.is_active])
        upcoming_tasks = len(await self.get_upcoming_tasks(24))

        # Calculate execution success rate
        executed_tasks = [t for t in self.tasks.values() if t.last_executed is not None]
        success_rate = len(executed_tasks) / total_tasks if total_tasks > 0 else 0

        return {
            "total_tasks": total_tasks,
            "active_tasks": active_tasks,
            "upcoming_tasks_24h": upcoming_tasks,
            "execution_success_rate": success_rate,
            "active_executions": len(self.active_tasks),
            "learned_patterns": len(self.predictive_engine.user_patterns),
        }


# Global scheduler instance
intelligent_scheduler = IntelligentScheduler()


async def get_intelligent_scheduler() -> IntelligentScheduler:
    """Get the global intelligent scheduler instance"""
    await intelligent_scheduler.initialize()
    return intelligent_scheduler


__all__ = [
    "TaskPriority",
    "TaskType",
    "ScheduledTask",
    "UserPattern",
    "PredictiveEngine",
    "ConflictResolver",
    "IntelligentScheduler",
    "intelligent_scheduler",
    "get_intelligent_scheduler",
]
