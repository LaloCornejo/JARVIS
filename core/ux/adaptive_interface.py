"""
Adaptive User Experience System for JARVIS.

This module provides intelligent UI/UX adaptation including:
- Contextual interface personalization based on user behavior
- Gesture recognition and multi-modal interaction
- Emotion-aware responses and interface adaptation
- User preference learning and adaptive interfaces
- Real-time UX optimization and accessibility features
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

log = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Different interaction modes"""

    TEXT = "text"
    VOICE = "voice"
    GESTURE = "gesture"
    MULTI_MODAL = "multi_modal"
    ACCESSIBILITY = "accessibility"


class UserPreference(Enum):
    """User preference categories"""

    THEME = "theme"
    LAYOUT = "layout"
    INTERACTION_MODE = "interaction_mode"
    NOTIFICATION_STYLE = "notification_style"
    ACCESSIBILITY = "accessibility"
    LANGUAGE = "language"
    TIME_FORMAT = "time_format"


@dataclass
class UserProfile:
    """User profile with learned preferences and behavior patterns"""

    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    accessibility_needs: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    session_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "behavior_patterns": self.behavior_patterns,
            "accessibility_needs": self.accessibility_needs,
            "interaction_history": self.interaction_history[-100:],  # Keep last 100 interactions
            "last_updated": self.last_updated.isoformat(),
            "session_count": self.session_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create profile from dictionary"""
        profile = cls(
            user_id=data["user_id"],
            preferences=data.get("preferences", {}),
            behavior_patterns=data.get("behavior_patterns", {}),
            accessibility_needs=data.get("accessibility_needs", {}),
            interaction_history=data.get("interaction_history", []),
            session_count=data.get("session_count", 0),
        )

        if data.get("last_updated"):
            profile.last_updated = datetime.fromisoformat(data["last_updated"])

        return profile


@dataclass
class GesturePattern:
    """Represents a learned gesture pattern"""

    gesture_type: str
    description: str
    confidence: float
    actions: List[str]
    context_triggers: List[str]
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0


class GestureRecognizer:
    """Advanced gesture recognition system"""

    def __init__(self):
        self.gesture_patterns: Dict[str, GesturePattern] = {}
        self.active_gestures: Set[str] = set()
        self.gesture_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 50

    def add_gesture_pattern(self, pattern: GesturePattern):
        """Add a gesture pattern to the recognizer"""
        self.gesture_patterns[pattern.gesture_type] = pattern
        log.info(f"Added gesture pattern: {pattern.gesture_type}")

    async def process_gesture_input(self, gesture_data: Dict[str, Any]) -> Optional[str]:
        """Process gesture input and return recognized gesture"""
        # Add to buffer
        self.gesture_buffer.append({"data": gesture_data, "timestamp": datetime.now()})

        # Keep buffer size limited
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer = self.gesture_buffer[-self.buffer_size :]

        # Analyze gesture sequence
        recognized_gesture = await self._analyze_gesture_sequence()

        if recognized_gesture:
            # Update usage statistics
            if recognized_gesture in self.gesture_patterns:
                pattern = self.gesture_patterns[recognized_gesture]
                pattern.usage_count += 1
                pattern.last_used = datetime.now()

        return recognized_gesture

    async def _analyze_gesture_sequence(self) -> Optional[str]:
        """Analyze the gesture buffer to recognize patterns"""
        if len(self.gesture_buffer) < 3:
            return None

        # Simple gesture recognition based on patterns
        # In a real implementation, this would use ML models for gesture recognition

        recent_gestures = self.gesture_buffer[-10:]  # Last 10 gesture points

        # Check for common gesture patterns
        if self._detect_swipe_gesture(recent_gestures):
            return "swipe"

        if self._detect_tap_gesture(recent_gestures):
            return "tap"

        if self._detect_circle_gesture(recent_gestures):
            return "circle"

        if self._detect_scroll_gesture(recent_gestures):
            return "scroll"

        return None

    def _detect_swipe_gesture(self, gestures: List[Dict[str, Any]]) -> bool:
        """Detect swipe gesture"""
        if len(gestures) < 5:
            return False

        # Check for horizontal or vertical movement pattern
        start_x = gestures[0]["data"].get("x", 0)
        end_x = gestures[-1]["data"].get("x", 0)
        start_y = gestures[0]["data"].get("y", 0)
        end_y = gestures[-1]["data"].get("y", 0)

        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)

        # Significant movement in one direction
        return (dx > 50 and dy < 20) or (dy > 50 and dx < 20)

    def _detect_tap_gesture(self, gestures: List[Dict[str, Any]]) -> bool:
        """Detect tap gesture"""
        if len(gestures) < 3:
            return False

        # Check for quick touch and release
        start_time = gestures[0]["timestamp"]
        end_time = gestures[-1]["timestamp"]

        duration = (end_time - start_time).total_seconds()
        movement = self._calculate_total_movement(gestures)

        return duration < 0.3 and movement < 10  # Quick, minimal movement

    def _detect_circle_gesture(self, gestures: List[Dict[str, Any]]) -> bool:
        """Detect circular gesture"""
        if len(gestures) < 8:
            return False

        # Simplified circle detection - check for rotational movement
        # In practice, this would use more sophisticated geometry
        positions = [(g["data"].get("x", 0), g["data"].get("y", 0)) for g in gestures]

        # Check if points form a roughly circular pattern
        center_x = sum(x for x, y in positions) / len(positions)
        center_y = sum(y for x, y in positions) / len(positions)

        # Check if points are roughly equidistant from center
        distances = [((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in positions]
        avg_distance = sum(distances) / len(distances)
        variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)

        return variance < 100  # Low variance indicates circular pattern

    def _detect_scroll_gesture(self, gestures: List[Dict[str, Any]]) -> bool:
        """Detect scroll gesture"""
        if len(gestures) < 5:
            return False

        # Check for continuous directional movement
        directions = []
        for i in range(1, len(gestures)):
            prev_x, prev_y = (
                gestures[i - 1]["data"].get("x", 0),
                gestures[i - 1]["data"].get("y", 0),
            )
            curr_x, curr_y = gestures[i]["data"].get("x", 0), gestures[i]["data"].get("y", 0)

            dx = curr_x - prev_x
            dy = curr_y - prev_y

            if abs(dx) > abs(dy):
                directions.append("horizontal")
            elif abs(dy) > abs(dx):
                directions.append("vertical")

        # Check for consistent direction
        if directions and len(set(directions)) == 1:
            return True

        return False

    def _calculate_total_movement(self, gestures: List[Dict[str, Any]]) -> float:
        """Calculate total movement distance in gesture sequence"""
        if len(gestures) < 2:
            return 0

        total_distance = 0
        for i in range(1, len(gestures)):
            prev_x, prev_y = (
                gestures[i - 1]["data"].get("x", 0),
                gestures[i - 1]["data"].get("y", 0),
            )
            curr_x, curr_y = gestures[i]["data"].get("x", 0), gestures[i]["data"].get("y", 0)

            distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
            total_distance += distance

        return total_distance

    def get_gesture_stats(self) -> Dict[str, Any]:
        """Get gesture recognition statistics"""
        return {
            "total_patterns": len(self.gesture_patterns),
            "active_gestures": len(self.active_gestures),
            "buffer_size": len(self.gesture_buffer),
            "pattern_usage": {
                name: pattern.usage_count for name, pattern in self.gesture_patterns.items()
            },
        }


class ContextualUIAdapter:
    """Adapts UI based on context and user behavior"""

    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.context_history: List[Dict[str, Any]] = []
        self.adaptation_rules: Dict[str, Dict[str, Any]] = {}
        self.max_history = 1000

    def load_user_profile(self, user_id: str) -> UserProfile:
        """Load or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        return self.user_profiles[user_id]

    def save_user_profile(self, user_id: str):
        """Save user profile (would persist to database in real implementation)"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.last_updated = datetime.now()
            # In real implementation, save to database
            log.debug(f"Saved profile for user {user_id}")

    async def adapt_interface(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt interface based on user profile and context"""
        profile = self.load_user_profile(user_id)

        # Record context
        self._record_context(context)

        # Analyze context and user preferences
        adaptations = await self._analyze_context_for_adaptations(profile, context)

        # Update user profile based on interaction
        self._update_profile_from_context(profile, context)

        # Save updated profile
        self.save_user_profile(user_id)

        return adaptations

    async def _analyze_context_for_adaptations(
        self, profile: UserProfile, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze context to determine UI adaptations"""
        adaptations = {}

        # Theme adaptation based on time of day
        current_hour = datetime.now().hour
        if 6 <= current_hour < 18:
            adaptations["theme"] = profile.preferences.get("day_theme", "light")
        else:
            adaptations["theme"] = profile.preferences.get("night_theme", "dark")

        # Layout adaptation based on device type
        device_type = context.get("device_type", "desktop")
        if device_type == "mobile":
            adaptations["layout"] = "compact"
            adaptations["font_size"] = "large"
        elif device_type == "tablet":
            adaptations["layout"] = "medium"
            adaptations["font_size"] = "medium"
        else:
            adaptations["layout"] = profile.preferences.get("desktop_layout", "expanded")

        # Interaction mode adaptation
        interaction_context = context.get("interaction_context", {})
        if interaction_context.get("is_driving"):
            adaptations["interaction_mode"] = "voice_only"
            adaptations["voice_speed"] = "slow"
        elif interaction_context.get("is_busy"):
            adaptations["interaction_mode"] = "minimal_ui"
        else:
            adaptations["interaction_mode"] = profile.preferences.get(
                "preferred_interaction", "multi_modal"
            )

        # Accessibility adaptations
        if profile.accessibility_needs.get("high_contrast"):
            adaptations["theme"] = "high_contrast"
        if profile.accessibility_needs.get("large_text"):
            adaptations["font_size"] = "extra_large"
        if profile.accessibility_needs.get("reduced_motion"):
            adaptations["animations"] = "minimal"

        # Content adaptation based on user preferences
        if profile.preferences.get("concise_responses"):
            adaptations["response_style"] = "concise"
        if profile.preferences.get("technical_level"):
            adaptations["technical_level"] = profile.preferences["technical_level"]

        return adaptations

    def _record_context(self, context: Dict[str, Any]):
        """Record context for pattern analysis"""
        context_entry = {"timestamp": datetime.now(), "context": context}

        self.context_history.append(context_entry)

        # Keep history size limited
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history :]

    def _update_profile_from_context(self, profile: UserProfile, context: Dict[str, Any]):
        """Update user profile based on current context"""
        # Record interaction
        interaction = {
            "timestamp": datetime.now(),
            "context": context,
            "type": context.get("interaction_type", "unknown"),
        }

        profile.interaction_history.append(interaction)

        # Keep history limited
        if len(profile.interaction_history) > 100:
            profile.interaction_history = profile.interaction_history[-100:]

        # Update behavior patterns
        self._update_behavior_patterns(profile, context)

    def _update_behavior_patterns(self, profile: UserProfile, context: Dict[str, Any]):
        """Update behavior patterns based on context"""
        interaction_type = context.get("interaction_type", "unknown")
        device_type = context.get("device_type", "desktop")
        time_of_day = "day" if 6 <= datetime.now().hour < 18 else "night"

        # Update pattern counts
        patterns = profile.behavior_patterns

        if "interaction_types" not in patterns:
            patterns["interaction_types"] = {}
        patterns["interaction_types"][interaction_type] = (
            patterns["interaction_types"].get(interaction_type, 0) + 1
        )

        if "device_usage" not in patterns:
            patterns["device_usage"] = {}
        patterns["device_usage"][device_type] = patterns["device_usage"].get(device_type, 0) + 1

        if "time_patterns" not in patterns:
            patterns["time_patterns"] = {}
        patterns["time_patterns"][time_of_day] = patterns["time_patterns"].get(time_of_day, 0) + 1

    def set_user_preference(self, user_id: str, preference: UserPreference, value: Any):
        """Set a user preference"""
        profile = self.load_user_profile(user_id)
        profile.preferences[preference.value] = value
        self.save_user_profile(user_id)

    def get_user_preference(
        self, user_id: str, preference: UserPreference, default: Any = None
    ) -> Any:
        """Get a user preference"""
        profile = self.load_user_profile(user_id)
        return profile.preferences.get(preference.value, default)

    def update_accessibility_needs(self, user_id: str, needs: Dict[str, Any]):
        """Update accessibility needs for a user"""
        profile = self.load_user_profile(user_id)
        profile.accessibility_needs.update(needs)
        self.save_user_profile(user_id)

    def get_personalization_stats(self, user_id: str) -> Dict[str, Any]:
        """Get personalization statistics for a user"""
        profile = self.load_user_profile(user_id)

        return {
            "total_sessions": profile.session_count,
            "total_interactions": len(profile.interaction_history),
            "learned_preferences": len(profile.preferences),
            "behavior_patterns": len(profile.behavior_patterns),
            "accessibility_features": len(profile.accessibility_needs),
            "last_updated": profile.last_updated.isoformat(),
        }


class EmotionAwareInterface:
    """Emotion-aware interface adaptation"""

    def __init__(self):
        self.emotion_patterns: Dict[str, Dict[str, Any]] = {}
        self.emotion_history: List[Dict[str, Any]] = []

    async def detect_emotion_context(self, text: str, context: Dict[str, Any]) -> str:
        """Detect emotional context from text and interaction"""
        # Simple emotion detection based on keywords and patterns
        # In a real implementation, this would use ML models

        text_lower = text.lower()

        # Positive emotions
        if any(
            word in text_lower
            for word in ["great", "awesome", "excellent", "wonderful", "fantastic"]
        ):
            return "positive_excited"

        if any(word in text_lower for word in ["good", "nice", "fine", "okay", "alright"]):
            return "positive_content"

        # Negative emotions
        if any(word in text_lower for word in ["terrible", "awful", "horrible", "bad", "worst"]):
            return "negative_frustrated"

        if any(word in text_lower for word in ["annoyed", "upset", "disappointed", "worried"]):
            return "negative_concerned"

        # Neutral/questioning
        if any(word in text_lower for word in ["?", "what", "how", "why", "when", "where"]):
            return "questioning"

        if any(word in text_lower for word in ["busy", "rushed", "hurry", "quick"]):
            return "time_pressure"

        return "neutral"

    async def adapt_response_for_emotion(self, emotion: str, response: str) -> str:
        """Adapt response based on detected emotion"""
        adaptations = {
            "positive_excited": {
                "tone": "enthusiastic",
                "add_emoji": True,
                "response_style": "engaging",
            },
            "positive_content": {
                "tone": "friendly",
                "add_emoji": False,
                "response_style": "informative",
            },
            "negative_frustrated": {
                "tone": "empathetic",
                "add_emoji": False,
                "response_style": "supportive",
                "add_apology": True,
            },
            "negative_concerned": {
                "tone": "understanding",
                "add_emoji": False,
                "response_style": "helpful",
            },
            "questioning": {"tone": "helpful", "add_emoji": False, "response_style": "informative"},
            "time_pressure": {"tone": "concise", "add_emoji": False, "response_style": "brief"},
            "neutral": {"tone": "professional", "add_emoji": False, "response_style": "balanced"},
        }

        adaptation = adaptations.get(emotion, adaptations["neutral"])

        # Apply adaptations (simplified)
        adapted_response = response

        if adaptation.get("add_apology"):
            adapted_response = f"I'm sorry to hear that. {adapted_response}"

        if adaptation.get("add_emoji") and emotion == "positive_excited":
            adapted_response += " 🎉"

        return adapted_response


# Global UX system instances
gesture_recognizer = GestureRecognizer()
contextual_ui_adapter = ContextualUIAdapter()
emotion_aware_interface = EmotionAwareInterface()


async def get_gesture_recognizer() -> GestureRecognizer:
    """Get the global gesture recognizer"""
    return gesture_recognizer


async def get_contextual_ui_adapter() -> ContextualUIAdapter:
    """Get the global contextual UI adapter"""
    return contextual_ui_adapter


async def get_emotion_aware_interface() -> EmotionAwareInterface:
    """Get the global emotion-aware interface"""
    return emotion_aware_interface


# Initialize default gesture patterns
def _initialize_default_gestures():
    """Initialize common gesture patterns"""
    default_gestures = [
        GesturePattern(
            gesture_type="swipe_right",
            description="Swipe right to navigate forward",
            confidence=0.8,
            actions=["navigate_forward", "next_page"],
            context_triggers=["browsing", "reading"],
        ),
        GesturePattern(
            gesture_type="swipe_left",
            description="Swipe left to navigate back",
            confidence=0.8,
            actions=["navigate_back", "previous_page"],
            context_triggers=["browsing", "reading"],
        ),
        GesturePattern(
            gesture_type="tap",
            description="Tap to select or activate",
            confidence=0.9,
            actions=["select", "activate", "open"],
            context_triggers=["any"],
        ),
        GesturePattern(
            gesture_type="double_tap",
            description="Double tap for quick actions",
            confidence=0.7,
            actions=["favorite", "bookmark", "like"],
            context_triggers=["content_viewing"],
        ),
        GesturePattern(
            gesture_type="circle",
            description="Draw circle to refresh or reload",
            confidence=0.6,
            actions=["refresh", "reload", "update"],
            context_triggers=["any"],
        ),
        GesturePattern(
            gesture_type="scroll_up",
            description="Scroll up to see more content",
            confidence=0.8,
            actions=["scroll_up", "load_more"],
            context_triggers=["content_viewing", "lists"],
        ),
        GesturePattern(
            gesture_type="scroll_down",
            description="Scroll down through content",
            confidence=0.8,
            actions=["scroll_down", "next_item"],
            context_triggers=["content_viewing", "lists"],
        ),
    ]

    for gesture in default_gestures:
        gesture_recognizer.add_gesture_pattern(gesture)


# Initialize default gestures on module load
_initialize_default_gestures()


__all__ = [
    "InteractionMode",
    "UserPreference",
    "UserProfile",
    "GesturePattern",
    "GestureRecognizer",
    "ContextualUIAdapter",
    "EmotionAwareInterface",
    "gesture_recognizer",
    "contextual_ui_adapter",
    "emotion_aware_interface",
    "get_gesture_recognizer",
    "get_contextual_ui_adapter",
    "get_emotion_aware_interface",
]
