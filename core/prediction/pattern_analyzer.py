"""
Pattern Analyzer for JARVIS.

Analyzes user behavior patterns and predicts future actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of behavioral patterns"""

    TEMPORAL = "temporal"  # Time-based patterns
    SEQUENTIAL = "sequential"  # Action sequences
    FREQUENCY = "frequency"  # How often actions occur
    CONTEXTUAL = "contextual"  # Context-dependent patterns
    WEEKLY = "weekly"  # Day-of-week patterns
    LOCATION = "location"  # Location-based patterns


@dataclass
class PatternOccurrence:
    """A single occurrence of a pattern"""

    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[timedelta] = None
    success: bool = True


@dataclass
class UserPattern:
    """A detected user behavior pattern"""

    id: str
    name: str
    pattern_type: PatternType
    action: str
    confidence: float
    occurrences: List[PatternOccurrence]
    first_observed: datetime
    last_observed: datetime
    frequency_per_day: float = 0.0
    typical_time: Optional[time] = None  # type: ignore
    typical_day: Optional[int] = None  # 0=Monday, 6=Sunday
    related_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_occurrences(self) -> int:
        return len(self.occurrences)

    @property
    def success_rate(self) -> float:
        if not self.occurrences:
            return 0.0
        successful = sum(1 for o in self.occurrences if o.success)
        return successful / len(self.occurrences)


@dataclass
class PatternMatch:
    """A matched pattern with prediction"""

    pattern: UserPattern
    match_confidence: float
    predicted_time: Optional[datetime] = None
    context_match_score: float = 0.0
    recommendation: Optional[str] = None


class PatternAnalyzer:
    """
    Analyzes user behavior patterns for predictive capabilities.

    Features:
    - Temporal pattern detection (daily routines)
    - Sequential pattern detection (action chains)
    - Frequency analysis
    - Context-aware predictions
    - Confidence scoring
    """

    def __init__(self, storage_path: str = "data/patterns"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.patterns: Dict[str, UserPattern] = {}
        self.action_history: List[Dict[str, Any]] = []
        self._initialized = False
        self._lock = asyncio.Lock()

        # Minimum occurrences to consider a pattern
        self.min_occurrences = 3

        # Time windows for pattern detection
        self.temporal_window = timedelta(minutes=30)

    async def initialize(self):
        """Initialize the pattern analyzer"""
        if self._initialized:
            return

        await self._load_patterns()
        self._initialized = True
        log.info(f"Pattern analyzer initialized with {len(self.patterns)} patterns")

    async def _load_patterns(self):
        """Load patterns from storage"""
        patterns_file = self.storage_path / "patterns.json"

        if patterns_file.exists():
            try:
                with open(patterns_file, "r") as f:
                    data = json.load(f)

                for pattern_data in data.get("patterns", []):
                    pattern = self._pattern_from_dict(pattern_data)
                    self.patterns[pattern.id] = pattern

            except Exception as e:
                log.error(f"Error loading patterns: {e}")

    async def _save_patterns(self):
        """Save patterns to storage"""
        patterns_file = self.storage_path / "patterns.json"

        with open(patterns_file, "w") as f:
            json.dump(
                {"patterns": [self._pattern_to_dict(p) for p in self.patterns.values()]},
                f,
                indent=2,
            )

    def _pattern_to_dict(self, pattern: UserPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            "id": pattern.id,
            "name": pattern.name,
            "pattern_type": pattern.pattern_type.value,
            "action": pattern.action,
            "confidence": pattern.confidence,
            "occurrences": [
                {
                    "timestamp": o.timestamp.isoformat(),
                    "context": o.context,
                    "duration": o.duration.total_seconds() if o.duration else None,
                    "success": o.success,
                }
                for o in pattern.occurrences
            ],
            "first_observed": pattern.first_observed.isoformat(),
            "last_observed": pattern.last_observed.isoformat(),
            "frequency_per_day": pattern.frequency_per_day,
            "typical_time": pattern.typical_time.isoformat() if pattern.typical_time else None,
            "typical_day": pattern.typical_day,
            "related_patterns": pattern.related_patterns,
            "metadata": pattern.metadata,
        }

    def _pattern_from_dict(self, data: Dict[str, Any]) -> UserPattern:
        """Create pattern from dictionary"""

        return UserPattern(
            id=data["id"],
            name=data["name"],
            pattern_type=PatternType(data["pattern_type"]),
            action=data["action"],
            confidence=data["confidence"],
            occurrences=[
                PatternOccurrence(
                    timestamp=datetime.fromisoformat(o["timestamp"]),
                    context=o.get("context", {}),
                    duration=timedelta(seconds=o["duration"]) if o.get("duration") else None,
                    success=o.get("success", True),
                )
                for o in data.get("occurrences", [])
            ],
            first_observed=datetime.fromisoformat(data["first_observed"]),
            last_observed=datetime.fromisoformat(data["last_observed"]),
            frequency_per_day=data.get("frequency_per_day", 0.0),
            typical_day=data.get("typical_day"),
            related_patterns=data.get("related_patterns", []),
            metadata=data.get("metadata", {}),
        )

    async def record_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        duration: Optional[timedelta] = None,
        success: bool = True,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a user action for pattern analysis"""
        async with self._lock:
            occurrence = PatternOccurrence(
                timestamp=timestamp or datetime.now(),
                context=context or {},
                duration=duration,
                success=success,
            )

            self.action_history.append(
                {
                    "action": action,
                    "timestamp": occurrence.timestamp,
                    "context": occurrence.context,
                }
            )

            # Update or create patterns
            await self._update_patterns(action, occurrence)

            # Keep history manageable
            if len(self.action_history) > 10000:
                self.action_history = self.action_history[-5000:]

    async def _update_patterns(self, action: str, occurrence: PatternOccurrence):
        """Update patterns based on new occurrence"""
        # Find existing pattern
        pattern_id = None
        for pid, pattern in self.patterns.items():
            if pattern.action == action:
                pattern_id = pid
                break

        if pattern_id:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern.occurrences.append(occurrence)
            pattern.last_observed = occurrence.timestamp

            # Recalculate statistics
            await self._recalculate_pattern_stats(pattern)
        else:
            # Create new pattern
            pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            pattern = UserPattern(
                id=pattern_id,
                name=f"Pattern: {action}",
                pattern_type=PatternType.TEMPORAL,  # Default, will be refined
                action=action,
                confidence=0.0,
                occurrences=[occurrence],
                first_observed=occurrence.timestamp,
                last_observed=occurrence.timestamp,
            )

            self.patterns[pattern_id] = pattern

        await self._save_patterns()

    async def _recalculate_pattern_stats(self, pattern: UserPattern):
        """Recalculate pattern statistics"""
        if len(pattern.occurrences) < self.min_occurrences:
            pattern.confidence = len(pattern.occurrences) / self.min_occurrences * 0.5
            return

        # Calculate frequency
        time_span = pattern.last_observed - pattern.first_observed
        days = max(time_span.days, 1)
        pattern.frequency_per_day = len(pattern.occurrences) / days

        # Calculate typical time
        times = []
        for occ in pattern.occurrences:
            times.append(occ.timestamp.hour + occ.timestamp.minute / 60)

        if times:
            avg_hour = np.mean(times)
            hour = int(avg_hour)
            minute = int((avg_hour - hour) * 60)
            from datetime import time

            pattern.typical_time = time(hour, minute)

        # Calculate typical day
        days_of_week = [occ.timestamp.weekday() for occ in pattern.occurrences]
        if days_of_week:
            # Find most common day
            from collections import Counter

            day_counts = Counter(days_of_week)
            pattern.typical_day = day_counts.most_common(1)[0][0]

        # Calculate confidence based on consistency
        if pattern.typical_time and pattern.frequency_per_day > 0.5:
            pattern.confidence = min(0.9, 0.5 + pattern.frequency_per_day * 0.1)

            # Refine pattern type
            if pattern.typical_day is not None:
                pattern.pattern_type = PatternType.WEEKLY
            else:
                pattern.pattern_type = PatternType.TEMPORAL

    async def analyze_sequential_patterns(self) -> List[UserPattern]:
        """Analyze for sequential patterns (action chains)"""
        sequences = []

        # Sort history by time
        sorted_history = sorted(self.action_history, key=lambda x: x["timestamp"])

        # Find common 2-action and 3-action sequences
        action_pairs = defaultdict(int)
        action_triples = defaultdict(int)

        for i in range(len(sorted_history) - 1):
            pair = (sorted_history[i]["action"], sorted_history[i + 1]["action"])
            action_pairs[pair] += 1

        for i in range(len(sorted_history) - 2):
            triple = (
                sorted_history[i]["action"],
                sorted_history[i + 1]["action"],
                sorted_history[i + 2]["action"],
            )
            action_triples[triple] += 1

        # Create patterns for frequent sequences
        for pair, count in action_pairs.items():
            if count >= self.min_occurrences:
                pattern_id = f"seq_{pair[0]}_{pair[1]}"

                if pattern_id not in self.patterns:
                    # Find occurrences
                    occurrences = []
                    for i in range(len(sorted_history) - 1):
                        if (
                            sorted_history[i]["action"] == pair[0]
                            and sorted_history[i + 1]["action"] == pair[1]
                        ):
                            occurrences.append(
                                PatternOccurrence(
                                    timestamp=sorted_history[i]["timestamp"],
                                    context={"sequence": pair},
                                )
                            )

                    pattern = UserPattern(
                        id=pattern_id,
                        name=f"Sequence: {pair[0]} → {pair[1]}",
                        pattern_type=PatternType.SEQUENTIAL,
                        action=f"{pair[0]} -> {pair[1]}",
                        confidence=min(0.9, count / 10),
                        occurrences=occurrences,
                        first_observed=occurrences[0].timestamp if occurrences else datetime.now(),
                        last_observed=occurrences[-1].timestamp if occurrences else datetime.now(),
                        frequency_per_day=count
                        / max((datetime.now() - occurrences[0].timestamp).days, 1)
                        if occurrences
                        else 0,
                    )

                    sequences.append(pattern)
                    self.patterns[pattern_id] = pattern

        await self._save_patterns()
        return sequences

    async def predict_next_actions(
        self,
        recent_actions: List[str],
        current_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[PatternMatch]:
        """Predict next likely actions based on recent activity"""
        matches = []

        # Look for sequential patterns matching recent actions
        if recent_actions:
            last_action = recent_actions[-1]

            for pattern in self.patterns.values():
                if pattern.pattern_type == PatternType.SEQUENTIAL:
                    parts = pattern.action.split(" -> ")
                    if len(parts) == 2 and parts[0] == last_action:
                        match = PatternMatch(
                            pattern=pattern,
                            match_confidence=pattern.confidence * 0.8,
                            recommendation=f"After {last_action}, you often {parts[1]}",
                        )
                        matches.append(match)

        # Look for temporal patterns
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()

        for pattern in self.patterns.values():
            if pattern.pattern_type in (PatternType.TEMPORAL, PatternType.WEEKLY):
                # Check if pattern is due
                if pattern.typical_time:
                    pattern_hour = pattern.typical_time.hour
                    pattern_minute = pattern.typical_time.minute
                    current_hour = current_time.hour
                    current_minute = current_time.minute

                    # Check if within window
                    time_diff = abs(
                        (current_hour * 60 + current_minute) - (pattern_hour * 60 + pattern_minute)
                    )

                    if time_diff <= 30:  # Within 30 minutes
                        # Check day match for weekly patterns
                        if (
                            pattern.pattern_type == PatternType.WEEKLY
                            and pattern.typical_day is not None
                        ):
                            if pattern.typical_day != current_day:
                                continue

                        match = PatternMatch(
                            pattern=pattern,
                            match_confidence=pattern.confidence * (1 - time_diff / 60),
                            predicted_time=now,
                            recommendation=f"It's around the time you usually {pattern.action}",
                        )
                        matches.append(match)

        # Sort by confidence
        matches.sort(key=lambda m: m.match_confidence, reverse=True)

        return matches[:top_k]

    async def get_daily_routine(self, day_of_week: Optional[int] = None) -> List[PatternMatch]:
        """Get typical daily routine patterns"""
        if day_of_week is None:
            day_of_week = datetime.now().weekday()

        routine = []

        for pattern in self.patterns.values():
            if pattern.pattern_type == PatternType.WEEKLY and pattern.typical_day == day_of_week:
                if pattern.typical_time and pattern.confidence > 0.5:
                    match = PatternMatch(
                        pattern=pattern,
                        match_confidence=pattern.confidence,
                        predicted_time=datetime.combine(
                            datetime.now().date(),
                            pattern.typical_time,
                        ),
                    )
                    routine.append(match)

        # Sort by time
        routine.sort(key=lambda m: m.predicted_time or datetime.min)

        return routine

    async def get_contextual_suggestions(
        self,
        context: Dict[str, Any],
        top_k: int = 3,
    ) -> List[str]:
        """Get suggestions based on current context"""
        suggestions = []

        # Analyze context for patterns
        location = context.get("location")
        time_of_day = context.get("time_of_day")
        day_of_week = context.get("day_of_week")

        for pattern in self.patterns.values():
            score = 0.0

            # Check location match
            if location:
                for occ in pattern.occurrences[-10:]:  # Recent occurrences
                    if occ.context.get("location") == location:
                        score += 0.3

            # Check time match
            if time_of_day and pattern.typical_time:
                # Simplified time comparison
                score += 0.2

            # Check day match
            if day_of_week is not None and pattern.typical_day is not None:
                if pattern.typical_day == day_of_week:
                    score += 0.2

            if score > 0.3:
                suggestions.append((score, pattern.action))

        suggestions.sort(reverse=True)
        return [action for _, action in suggestions[:top_k]]

    async def get_stats(self) -> Dict[str, Any]:
        """Get pattern analysis statistics"""
        by_type = defaultdict(int)
        for pattern in self.patterns.values():
            by_type[pattern.pattern_type.value] += 1

        high_confidence = sum(1 for p in self.patterns.values() if p.confidence > 0.7)

        return {
            "total_patterns": len(self.patterns),
            "patterns_by_type": dict(by_type),
            "high_confidence_patterns": high_confidence,
            "total_recorded_actions": len(self.action_history),
            "avg_confidence": sum(p.confidence for p in self.patterns.values()) / len(self.patterns)
            if self.patterns
            else 0,
        }


# Global instance
_pattern_analyzer: Optional[PatternAnalyzer] = None


async def get_pattern_analyzer() -> PatternAnalyzer:
    """Get the global pattern analyzer instance"""
    global _pattern_analyzer
    if _pattern_analyzer is None:
        _pattern_analyzer = PatternAnalyzer()
        await _pattern_analyzer.initialize()
    return _pattern_analyzer


__all__ = [
    "PatternAnalyzer",
    "UserPattern",
    "PatternType",
    "PatternMatch",
    "get_pattern_analyzer",
]
