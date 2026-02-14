"""
Smart Suggestion Engine for JARVIS.

Provides contextual, proactive suggestions based on:
- Current context
- User patterns
- Time and location
- Recent activity
- System state
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.prediction.pattern_analyzer import get_pattern_analyzer

log = logging.getLogger(__name__)


class SuggestionPriority(Enum):
    """Priority levels for suggestions"""

    LOW = "low"  # Background suggestions
    MEDIUM = "medium"  # Contextually relevant
    HIGH = "high"  # Time-sensitive or important
    URGENT = "urgent"  # Immediate attention needed


@dataclass
class ContextualSuggestion:
    """A contextual suggestion for the user"""

    id: str
    title: str
    description: str
    action: str
    action_type: str  # tool, workflow, info, reminder
    priority: SuggestionPriority
    confidence: float
    context_triggers: List[str]
    expires_at: Optional[datetime] = None
    dismissed: bool = False
    acted_upon: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_active(self) -> bool:
        return not self.dismissed and not self.acted_upon and not self.is_expired


class SuggestionContext:
    """Current context for generating suggestions"""

    def __init__(
        self,
        user_id: Optional[str] = None,
        location: Optional[str] = None,
        time_of_day: Optional[str] = None,
        day_of_week: Optional[int] = None,
        recent_activity: Optional[List[str]] = None,
        current_app: Optional[str] = None,
        system_load: Optional[float] = None,
        upcoming_events: Optional[List[Dict]] = None,
        user_preferences: Optional[Dict] = None,
    ):
        self.user_id = user_id
        self.location = location
        self.time_of_day = time_of_day or self._get_time_of_day()
        self.day_of_week = day_of_week or datetime.now().weekday()
        self.recent_activity = recent_activity or []
        self.current_app = current_app
        self.system_load = system_load
        self.upcoming_events = upcoming_events or []
        self.user_preferences = user_preferences or {}

    def _get_time_of_day(self) -> str:
        """Categorize current time"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "day_of_week": self.day_of_week,
            "recent_activity": self.recent_activity,
            "current_app": self.current_app,
            "system_load": self.system_load,
            "upcoming_events": self.upcoming_events,
        }


class SmartSuggestionEngine:
    """
    Generates intelligent, contextual suggestions for users.

    Features:
    - Context-aware suggestions
    - Proactive recommendations
    - Smart reminders
    - Time-sensitive alerts
    - Learning from user feedback
    """

    def __init__(self, storage_path: str = "data/suggestions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.suggestions: Dict[str, ContextualSuggestion] = {}
        self.dismissed_suggestions: Set[str] = set()
        self.user_feedback: Dict[str, Dict[str, Any]] = {}

        self._initialized = False
        self._lock = asyncio.Lock()

        # Suggestion templates
        self._templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict]:
        """Load suggestion templates"""
        return {
            "morning_routine": {
                "title": "Morning Routine",
                "description": "Start your day with weather, calendar, and news",
                "action": "execute_workflow",
                "action_params": {"workflow": "morning_routine"},
                "context_triggers": ["time_of_day=morning"],
                "priority": SuggestionPriority.MEDIUM,
            },
            "coffee_break": {
                "title": "Take a Break",
                "description": "You've been working for 2 hours. Time for a break?",
                "action": "suggest_break",
                "action_params": {"type": "coffee"},
                "context_triggers": ["work_duration>2h"],
                "priority": SuggestionPriority.LOW,
            },
            "focus_mode": {
                "title": "Enable Focus Mode",
                "description": "You have a meeting in 15 minutes. Enable focus mode?",
                "action": "enable_focus",
                "action_params": {},
                "context_triggers": ["upcoming_meeting<15min"],
                "priority": SuggestionPriority.HIGH,
            },
            "system_maintenance": {
                "title": "System Maintenance",
                "description": "System hasn't been maintained in 7 days. Run maintenance?",
                "action": "execute_workflow",
                "action_params": {"workflow": "system_maintenance"},
                "context_triggers": ["days_since_maintenance>7"],
                "priority": SuggestionPriority.MEDIUM,
            },
            "backup_reminder": {
                "title": "Backup Your Data",
                "description": "It's been a while since your last backup",
                "action": "execute_workflow",
                "action_params": {"workflow": "data_backup"},
                "context_triggers": ["days_since_backup>3"],
                "priority": SuggestionPriority.HIGH,
            },
            "low_battery": {
                "title": "Low Battery Warning",
                "description": "Battery at {battery_level}%. Enable power saving?",
                "action": "enable_power_saving",
                "action_params": {},
                "context_triggers": ["battery<20"],
                "priority": SuggestionPriority.URGENT,
            },
            "unread_messages": {
                "title": "Unread Messages",
                "description": "You have {count} unread messages",
                "action": "show_messages",
                "action_params": {},
                "context_triggers": ["unread_messages>5"],
                "priority": SuggestionPriority.MEDIUM,
            },
            "pattern_detected": {
                "title": "Pattern Detected",
                "description": "You usually {action} around this time",
                "action": "execute_action",
                "action_params": {},
                "context_triggers": ["pattern_match"],
                "priority": SuggestionPriority.LOW,
            },
            "file_cleanup": {
                "title": "Clean Up Files",
                "description": "Downloads folder has {file_count} files. Clean up?",
                "action": "cleanup_files",
                "action_params": {"path": "~/Downloads"},
                "context_triggers": ["downloads_size>1GB"],
                "priority": SuggestionPriority.LOW,
            },
        }

    async def initialize(self):
        """Initialize the suggestion engine"""
        if self._initialized:
            return

        await self._load_data()
        self._initialized = True
        log.info(f"Suggestion engine initialized with {len(self.suggestions)} active suggestions")

    async def _load_data(self):
        """Load suggestion data"""
        data_file = self.storage_path / "suggestions.json"

        if data_file.exists():
            try:
                with open(data_file, "r") as f:
                    data = json.load(f)

                for sug_data in data.get("suggestions", []):
                    suggestion = ContextualSuggestion(
                        id=sug_data["id"],
                        title=sug_data["title"],
                        description=sug_data["description"],
                        action=sug_data["action"],
                        action_type=sug_data["action_type"],
                        priority=SuggestionPriority(sug_data["priority"]),
                        confidence=sug_data["confidence"],
                        context_triggers=sug_data["context_triggers"],
                        expires_at=datetime.fromisoformat(sug_data["expires_at"])
                        if sug_data.get("expires_at")
                        else None,
                        dismissed=sug_data.get("dismissed", False),
                        acted_upon=sug_data.get("acted_upon", False),
                        created_at=datetime.fromisoformat(sug_data["created_at"]),
                        metadata=sug_data.get("metadata", {}),
                    )
                    self.suggestions[suggestion.id] = suggestion

            except Exception as e:
                log.error(f"Error loading suggestions: {e}")

    async def _save_data(self):
        """Save suggestion data"""
        data_file = self.storage_path / "suggestions.json"

        with open(data_file, "w") as f:
            json.dump(
                {
                    "suggestions": [
                        {
                            "id": s.id,
                            "title": s.title,
                            "description": s.description,
                            "action": s.action,
                            "action_type": s.action_type,
                            "priority": s.priority.value,
                            "confidence": s.confidence,
                            "context_triggers": s.context_triggers,
                            "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                            "dismissed": s.dismissed,
                            "acted_upon": s.acted_upon,
                            "created_at": s.created_at.isoformat(),
                            "metadata": s.metadata,
                        }
                        for s in self.suggestions.values()
                    ]
                },
                f,
                indent=2,
            )

    async def generate_suggestions(
        self,
        context: SuggestionContext,
        max_suggestions: int = 5,
    ) -> List[ContextualSuggestion]:
        """Generate contextual suggestions"""
        new_suggestions = []

        # Check pattern-based suggestions
        pattern_analyzer = await get_pattern_analyzer()
        pattern_matches = await pattern_analyzer.predict_next_actions(
            context.recent_activity,
            context.to_dict(),
            top_k=3,
        )

        for match in pattern_matches:
            if match.match_confidence > 0.6:
                suggestion = await self._create_suggestion_from_pattern(match, context)
                if suggestion:
                    new_suggestions.append(suggestion)

        # Check template-based suggestions
        for template_id, template in self._templates.items():
            if await self._should_trigger(template, context):
                suggestion = await self._create_suggestion_from_template(
                    template_id, template, context
                )
                if suggestion:
                    new_suggestions.append(suggestion)

        # Check time-based suggestions
        time_suggestions = await self._generate_time_based_suggestions(context)
        new_suggestions.extend(time_suggestions)

        # Check system state suggestions
        system_suggestions = await self._generate_system_suggestions(context)
        new_suggestions.extend(system_suggestions)

        # Score and filter suggestions
        scored = []
        for suggestion in new_suggestions:
            # Skip if already dismissed recently
            if suggestion.id in self.dismissed_suggestions:
                continue

            # Skip if already exists
            if suggestion.id in self.suggestions:
                continue

            score = self._calculate_suggestion_score(suggestion, context)
            scored.append((score, suggestion))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top suggestions
        top_suggestions = [s for _, s in scored[:max_suggestions]]

        # Store suggestions
        for suggestion in top_suggestions:
            self.suggestions[suggestion.id] = suggestion

        await self._save_data()

        return top_suggestions

    async def _create_suggestion_from_pattern(
        self,
        pattern_match,
        context: SuggestionContext,
    ) -> Optional[ContextualSuggestion]:
        """Create a suggestion from a pattern match"""
        pattern = pattern_match.pattern

        suggestion_id = f"sug_pattern_{pattern.id}_{datetime.now().strftime('%H%M%S')}"

        return ContextualSuggestion(
            id=suggestion_id,
            title="Continue Your Routine",
            description=pattern_match.recommendation or f"You often {pattern.action}",
            action=pattern.action,
            action_type="pattern",
            priority=SuggestionPriority.MEDIUM,
            confidence=pattern_match.match_confidence,
            context_triggers=["pattern_match"],
            expires_at=datetime.now() + timedelta(minutes=30),
            metadata={"pattern_id": pattern.id},
        )

    async def _should_trigger(self, template: Dict, context: SuggestionContext) -> bool:
        """Check if a template should trigger based on context"""
        triggers = template.get("context_triggers", [])

        for trigger in triggers:
            if trigger == "time_of_day=morning" and context.time_of_day == "morning":
                return True
            elif trigger == "time_of_day=evening" and context.time_of_day == "evening":
                return True
            elif trigger.startswith("battery<"):
                try:
                    threshold = int(trigger.split("<")[1].replace("%", ""))
                    # Would need actual battery level
                    return False
                except Exception:
                    pass
            elif trigger.startswith("work_duration>"):
                # Would need work duration tracking
                return False

        return False

    async def _create_suggestion_from_template(
        self,
        template_id: str,
        template: Dict,
        context: SuggestionContext,
    ) -> ContextualSuggestion:
        """Create a suggestion from a template"""
        suggestion_id = f"sug_{template_id}_{datetime.now().strftime('%H%M%S')}"

        description = template["description"]
        # Fill in context variables
        if "{battery_level}" in description:
            description = description.replace("{battery_level}", "20")
        if "{count}" in description:
            description = description.replace("{count}", "5")
        if "{file_count}" in description:
            description = description.replace("{file_count}", "50")

        return ContextualSuggestion(
            id=suggestion_id,
            title=template["title"],
            description=description,
            action=template["action"],
            action_type="template",
            priority=template["priority"],
            confidence=0.7,
            context_triggers=template["context_triggers"],
            expires_at=datetime.now() + timedelta(hours=1),
            metadata={
                "template_id": template_id,
                "action_params": template.get("action_params", {}),
            },
        )

    async def _generate_time_based_suggestions(
        self,
        context: SuggestionContext,
    ) -> List[ContextualSuggestion]:
        """Generate time-based suggestions"""
        suggestions = []

        # Check for upcoming events
        for event in context.upcoming_events:
            start_time = event.get("start_time")
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)

            if start_time:
                time_until = start_time - datetime.now()
                if timedelta(minutes=10) < time_until <= timedelta(minutes=30):
                    suggestion = ContextualSuggestion(
                        id=f"sug_event_{event.get('id', 'unknown')}",
                        title=f"Upcoming: {event.get('title', 'Event')}",
                        description=f"Starting in {int(time_until.total_seconds() / 60)} minutes",
                        action="show_event_details",
                        action_type="info",
                        priority=SuggestionPriority.HIGH,
                        confidence=0.9,
                        context_triggers=["upcoming_event"],
                        expires_at=start_time,
                        metadata={"event_id": event.get("id")},
                    )
                    suggestions.append(suggestion)

        return suggestions

    async def _generate_system_suggestions(
        self,
        context: SuggestionContext,
    ) -> List[ContextualSuggestion]:
        """Generate system-state-based suggestions"""
        suggestions = []

        # High CPU usage
        if context.system_load and context.system_load > 80:
            suggestions.append(
                ContextualSuggestion(
                    id=f"sug_high_cpu_{datetime.now().strftime('%H%M%S')}",
                    title="High CPU Usage",
                    description=f"System load is at {context.system_load:.0f}%. Check running processes?",
                    action="show_processes",
                    action_type="system",
                    priority=SuggestionPriority.MEDIUM,
                    confidence=0.8,
                    context_triggers=["high_cpu"],
                    metadata={"cpu_percent": context.system_load},
                )
            )

        return suggestions

    def _calculate_suggestion_score(
        self,
        suggestion: ContextualSuggestion,
        context: SuggestionContext,
    ) -> float:
        """Calculate relevance score for a suggestion"""
        score = suggestion.confidence

        # Boost by priority
        priority_boost = {
            SuggestionPriority.URGENT: 1.0,
            SuggestionPriority.HIGH: 0.5,
            SuggestionPriority.MEDIUM: 0.2,
            SuggestionPriority.LOW: 0.0,
        }
        score += priority_boost.get(suggestion.priority, 0)

        # Penalize if user dismissed similar recently
        if suggestion.id in self.dismissed_suggestions:
            score -= 0.5

        return score

    async def get_active_suggestions(
        self,
        min_priority: Optional[SuggestionPriority] = None,
    ) -> List[ContextualSuggestion]:
        """Get all active suggestions"""
        active = []

        for suggestion in self.suggestions.values():
            if not suggestion.is_active:
                continue

            if min_priority:
                priority_order = [
                    SuggestionPriority.LOW,
                    SuggestionPriority.MEDIUM,
                    SuggestionPriority.HIGH,
                    SuggestionPriority.URGENT,
                ]
                if priority_order.index(suggestion.priority) < priority_order.index(min_priority):
                    continue

            active.append(suggestion)

        # Sort by priority and confidence
        priority_order = {
            SuggestionPriority.URGENT: 0,
            SuggestionPriority.HIGH: 1,
            SuggestionPriority.MEDIUM: 2,
            SuggestionPriority.LOW: 3,
        }

        active.sort(key=lambda s: (priority_order.get(s.priority, 4), -s.confidence))

        return active

    async def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Dismiss a suggestion"""
        async with self._lock:
            suggestion = self.suggestions.get(suggestion_id)
            if not suggestion:
                return False

            suggestion.dismissed = True
            self.dismissed_suggestions.add(suggestion_id)

            # Keep only recent dismissed suggestions
            if len(self.dismissed_suggestions) > 100:
                self.dismissed_suggestions = set(list(self.dismissed_suggestions)[-50:])

            await self._save_data()
            return True

    async def mark_acted_upon(self, suggestion_id: str) -> bool:
        """Mark a suggestion as acted upon"""
        async with self._lock:
            suggestion = self.suggestions.get(suggestion_id)
            if not suggestion:
                return False

            suggestion.acted_upon = True

            # Record feedback
            self.user_feedback[suggestion_id] = {
                "acted_upon": True,
                "timestamp": datetime.now().isoformat(),
            }

            await self._save_data()
            return True

    async def record_feedback(
        self,
        suggestion_id: str,
        helpful: bool,
        feedback: Optional[str] = None,
    ) -> None:
        """Record user feedback on a suggestion"""
        self.user_feedback[suggestion_id] = {
            "helpful": helpful,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        }

        # Learn from feedback
        suggestion = self.suggestions.get(suggestion_id)
        if suggestion and not helpful:
            # Reduce confidence of similar suggestions
            for other in self.suggestions.values():
                if other.action == suggestion.action:
                    other.confidence *= 0.9

    async def cleanup_expired(self) -> int:
        """Remove expired and acted upon suggestions"""
        async with self._lock:
            to_remove = []

            for sid, suggestion in self.suggestions.items():
                if suggestion.is_expired or suggestion.acted_upon:
                    # Keep for a day after expiration for analytics
                    if suggestion.expires_at:
                        if datetime.now() - suggestion.expires_at > timedelta(days=1):
                            to_remove.append(sid)

            for sid in to_remove:
                del self.suggestions[sid]

            await self._save_data()
            return len(to_remove)

    async def get_stats(self) -> Dict[str, Any]:
        """Get suggestion engine statistics"""
        active = sum(1 for s in self.suggestions.values() if s.is_active)
        by_priority = {}

        for s in self.suggestions.values():
            p = s.priority.value
            by_priority[p] = by_priority.get(p, 0) + 1

        acted_upon = sum(1 for s in self.suggestions.values() if s.acted_upon)
        dismissed = sum(1 for s in self.suggestions.values() if s.dismissed)

        total = len(self.suggestions)
        conversion_rate = acted_upon / total if total > 0 else 0

        return {
            "total_suggestions": total,
            "active_suggestions": active,
            "by_priority": by_priority,
            "acted_upon": acted_upon,
            "dismissed": dismissed,
            "conversion_rate": conversion_rate,
            "user_feedback_count": len(self.user_feedback),
        }


# Global instance
_suggestion_engine: Optional[SmartSuggestionEngine] = None


async def get_suggestion_engine() -> SmartSuggestionEngine:
    """Get the global suggestion engine instance"""
    global _suggestion_engine
    if _suggestion_engine is None:
        _suggestion_engine = SmartSuggestionEngine()
        await _suggestion_engine.initialize()
    return _suggestion_engine


__all__ = [
    "SmartSuggestionEngine",
    "ContextualSuggestion",
    "SuggestionPriority",
    "SuggestionContext",
    "get_suggestion_engine",
]
