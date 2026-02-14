"""
Enhanced Smart Suggestion Engine for JARVIS.

Provides more sophisticated, contextual, proactive suggestions based on:
- Advanced context awareness
- ML-based pattern recognition
- Real-time system state monitoring
- User preference learning
- Environmental context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from core.prediction.anomaly_detector import get_anomaly_detector
from core.prediction.pattern_analyzer import get_pattern_analyzer
from core.prediction.suggestion_engine import (
    ContextualSuggestion,
    SmartSuggestionEngine,
    SuggestionContext,
    SuggestionPriority,
)

log = logging.getLogger(__name__)


class EnhancedSuggestionPriority(Enum):
    """Enhanced priority levels for suggestions"""

    LOW = "low"  # Background suggestions
    MEDIUM = "medium"  # Contextually relevant
    HIGH = "high"  # Time-sensitive or important
    URGENT = "urgent"  # Immediate attention needed
    CRITICAL = "critical"  # Must-address immediately


class ContextTriggerType(Enum):
    """Types of context triggers"""

    TIME_BASED = "time_based"
    LOCATION_BASED = "location_based"
    ACTIVITY_BASED = "activity_based"
    SYSTEM_STATE = "system_state"
    PATTERN_BASED = "pattern_based"
    ANOMALY_BASED = "anomaly_based"
    ENVIRONMENTAL = "environmental"
    SOCIAL_CONTEXT = "social_context"
    USER_PREFERENCE = "user_preference"
    PREDICTIVE = "predictive"


@dataclass
class EnhancedContextTrigger:
    """Enhanced context trigger with ML features"""

    id: str
    trigger_type: ContextTriggerType
    condition: str
    confidence: float
    weight: float = 1.0
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    ml_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedSuggestionContext(SuggestionContext):
    """Enhanced context with additional environmental and social data"""

    environmental_data: Optional[Dict[str, Any]] = None
    social_context: Optional[Dict[str, Any]] = None
    emotional_state: Optional[str] = None
    stress_level: Optional[float] = None
    productivity_score: Optional[float] = None
    focus_level: Optional[float] = None

    def to_enhanced_dict(self) -> Dict[str, Any]:
        """Convert to enhanced dictionary with all context data"""
        base_dict = self.to_dict()
        enhanced_dict = base_dict.copy()
        enhanced_dict.update(
            {
                "environmental_data": self.environmental_data or {},
                "social_context": self.social_context or {},
                "emotional_state": self.emotional_state,
                "stress_level": self.stress_level,
                "productivity_score": self.productivity_score,
                "focus_level": self.focus_level,
            }
        )
        return enhanced_dict


class EnhancedSmartSuggestionEngine(SmartSuggestionEngine):
    """
    Enhanced Smart Suggestion Engine with advanced context awareness.

    Features:
    - Advanced context-aware triggers
    - ML-based suggestion ranking
    - Real-time context adaptation
    - Environmental context integration
    - Social context awareness
    - Emotional state consideration
    - Stress and productivity-aware suggestions
    - Dynamic trigger weighting
    """

    def __init__(self, storage_path: str = "data/suggestions"):
        super().__init__(storage_path)
        self.ml_models_initialized = False
        self.context_classifier: Optional[RandomForestClassifier] = None
        self.context_vectorizer: Optional[TfidfVectorizer] = None
        self.context_triggers: Dict[str, EnhancedContextTrigger] = {}
        self.trigger_history: List[Dict[str, Any]] = []
        self.contextual_weights: Dict[str, float] = {}

    async def initialize_ml_models(self):
        """Initialize ML models for enhanced suggestion engine"""
        if self.ml_models_initialized:
            return

        try:
            # Initialize context classifier
            self.context_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )

            # Initialize context vectorizer
            self.context_vectorizer = TfidfVectorizer(
                max_features=200, stop_words="english", ngram_range=(1, 2)
            )

            # Initialize contextual weights
            self.contextual_weights = {
                "time_of_day": 1.0,
                "day_of_week": 0.8,
                "location": 1.2,
                "recent_activity": 1.5,
                "system_load": 0.9,
                "upcoming_events": 1.3,
                "environmental_data": 1.1,
                "social_context": 1.0,
                "emotional_state": 1.4,
                "stress_level": 1.2,
                "productivity_score": 1.1,
                "focus_level": 1.3,
            }

            self.ml_models_initialized = True
            log.info("Enhanced ML models initialized")

        except Exception as e:
            log.error(f"Error initializing enhanced ML models: {e}")
            self.ml_models_initialized = True  # Continue with basic functionality

    def _load_templates(self) -> Dict[str, Dict]:
        """Load enhanced suggestion templates"""
        templates = super()._load_templates()

        # Add enhanced templates
        enhanced_templates = {
            "stress_management": {
                "title": "Stress Management",
                "description": "Your stress level appears elevated. Consider taking a break or doing breathing exercises.",
                "action": "suggest_stress_relief",
                "action_params": {"type": "breathing"},
                "context_triggers": ["stress_level>0.7"],
                "priority": EnhancedSuggestionPriority.HIGH,
            },
            "focus_booster": {
                "title": "Boost Focus",
                "description": "Your focus level is low. Try the Pomodoro technique or eliminate distractions.",
                "action": "enable_focus_mode",
                "action_params": {},
                "context_triggers": ["focus_level<0.4"],
                "priority": EnhancedSuggestionPriority.MEDIUM,
            },
            "productivity_check": {
                "title": "Productivity Check-In",
                "description": "Your productivity has been declining. Review your goals and priorities.",
                "action": "show_productivity_dashboard",
                "action_params": {},
                "context_triggers": ["productivity_trend=declining"],
                "priority": EnhancedSuggestionPriority.MEDIUM,
            },
            "social_engagement": {
                "title": "Social Connection",
                "description": "You haven't connected with friends/family recently. Reach out to someone!",
                "action": "suggest_social_connection",
                "action_params": {},
                "context_triggers": ["days_since_social>3"],
                "priority": EnhancedSuggestionPriority.LOW,
            },
            "environment_optimization": {
                "title": "Environment Optimization",
                "description": "Your environment could be improved for better {factor}.",
                "action": "suggest_environment_changes",
                "action_params": {},
                "context_triggers": ["environment_factor_suboptimal"],
                "priority": EnhancedSuggestionPriority.MEDIUM,
            },
            "learning_opportunity": {
                "title": "Learning Opportunity",
                "description": "Based on your interests, you might enjoy exploring {topic}.",
                "action": "suggest_learning_resource",
                "action_params": {},
                "context_triggers": ["interest_match"],
                "priority": EnhancedSuggestionPriority.LOW,
            },
            "health_reminder": {
                "title": "Health Check",
                "description": "It's been a while since your last health check. Consider {activity}.",
                "action": "suggest_health_activity",
                "action_params": {},
                "context_triggers": ["time_since_health_check"],
                "priority": EnhancedSuggestionPriority.MEDIUM,
            },
            "creative_stimulation": {
                "title": "Creative Stimulation",
                "description": "Engage in a creative activity to boost your mood and cognitive function.",
                "action": "suggest_creative_activity",
                "action_params": {},
                "context_triggers": ["routine_break_needed"],
                "priority": EnhancedSuggestionPriority.LOW,
            },
        }

        templates.update(enhanced_templates)
        return templates

    async def generate_enhanced_suggestions(
        self,
        context: EnhancedSuggestionContext,
        max_suggestions: int = 5,
    ) -> List[ContextualSuggestion]:
        """Generate enhanced contextual suggestions using advanced context awareness"""
        new_suggestions = []

        # Initialize ML models if needed
        if not self.ml_models_initialized:
            await self.initialize_ml_models()

        # Check pattern-based suggestions
        pattern_analyzer = await get_pattern_analyzer()
        pattern_matches = await pattern_analyzer.predict_next_actions_enhanced(
            context.recent_activity,
            context.to_enhanced_dict(),
            top_k=3,
        )

        for match in pattern_matches:
            if match.match_confidence > 0.5:
                suggestion = await self._create_enhanced_suggestion_from_pattern(match, context)
                if suggestion:
                    new_suggestions.append(suggestion)

        # Check template-based suggestions with enhanced triggers
        for template_id, template in self._templates.items():
            if await self._should_trigger_enhanced(template, context):
                suggestion = await self._create_enhanced_suggestion_from_template(
                    template_id, template, context
                )
                if suggestion:
                    new_suggestions.append(suggestion)

        # Check time-based suggestions
        time_suggestions = await self._generate_enhanced_time_based_suggestions(context)
        new_suggestions.extend(time_suggestions)

        # Check system state suggestions
        system_suggestions = await self._generate_enhanced_system_suggestions(context)
        new_suggestions.extend(system_suggestions)

        # Check environmental suggestions
        env_suggestions = await self._generate_environmental_suggestions(context)
        new_suggestions.extend(env_suggestions)

        # Check emotional/social suggestions
        emotional_suggestions = await self._generate_emotional_suggestions(context)
        new_suggestions.extend(emotional_suggestions)

        # Check predictive suggestions
        predictive_suggestions = await self._generate_predictive_suggestions(context)
        new_suggestions.extend(predictive_suggestions)

        # Score and filter suggestions using enhanced scoring
        scored = []
        for suggestion in new_suggestions:
            # Skip if already dismissed recently
            if suggestion.id in self.dismissed_suggestions:
                continue

            # Skip if already exists
            if suggestion.id in self.suggestions:
                continue

            score = await self._calculate_enhanced_suggestion_score(suggestion, context)
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

    async def _create_enhanced_suggestion_from_pattern(
        self,
        pattern_match,
        context: EnhancedSuggestionContext,
    ) -> Optional[ContextualSuggestion]:
        """Create an enhanced suggestion from a pattern match"""
        pattern = pattern_match.pattern

        suggestion_id = f"sug_pattern_{pattern.id}_{datetime.now().strftime('%H%M%S')}"

        # Enhance description based on context
        description = pattern_match.recommendation or f"You often {pattern.action}"
        if context.emotional_state:
            description += f" (especially when feeling {context.emotional_state})"

        return ContextualSuggestion(
            id=suggestion_id,
            title="Continue Your Routine",
            description=description,
            action=pattern.action,
            action_type="pattern",
            priority=SuggestionPriority.MEDIUM,
            confidence=pattern_match.match_confidence,
            context_triggers=["pattern_match"],
            expires_at=datetime.now() + timedelta(minutes=30),
            metadata={
                "pattern_id": pattern.id,
                "contextual_enhancement": {
                    "emotional_state": context.emotional_state,
                    "stress_level": context.stress_level,
                    "focus_level": context.focus_level,
                },
            },
        )

    async def _should_trigger_enhanced(
        self, template: Dict, context: EnhancedSuggestionContext
    ) -> bool:
        """Enhanced check if a template should trigger based on context"""
        triggers = template.get("context_triggers", [])

        for trigger in triggers:
            # Handle basic time-based triggers
            if trigger == "time_of_day=morning" and context.time_of_day == "morning":
                return True
            elif trigger == "time_of_day=evening" and context.time_of_day == "evening":
                return True

            # Handle stress level triggers
            elif trigger.startswith("stress_level>"):
                try:
                    threshold = float(trigger.split(">")[1])
                    if context.stress_level and context.stress_level > threshold:
                        return True
                except Exception:
                    pass

            # Handle focus level triggers
            elif trigger.startswith("focus_level<"):
                try:
                    threshold = float(trigger.split("<")[1])
                    if context.focus_level and context.focus_level < threshold:
                        return True
                except Exception:
                    pass

            # Handle productivity trend triggers
            elif trigger == "productivity_trend=declining":
                # This would require tracking productivity over time
                # For now, we'll simulate with a simple check
                if context.productivity_score and context.productivity_score < 0.5:
                    return True

            # Handle social context triggers
            elif trigger.startswith("days_since_social>"):
                try:
                    threshold = int(trigger.split(">")[1])
                    # Simulate social interaction tracking
                    # In a real implementation, this would check actual social interactions
                    return False  # Placeholder
                except Exception:
                    pass

            # Handle environmental triggers
            elif trigger == "environment_factor_suboptimal":
                if context.environmental_data:
                    # Check for suboptimal environmental factors
                    temp = context.environmental_data.get("temperature", 22)
                    light = context.environmental_data.get("light_level", 500)
                    noise = context.environmental_data.get("noise_level", 40)

                    if temp < 18 or temp > 25 or light < 300 or noise > 60:
                        return True

            # Handle interest match triggers
            elif trigger == "interest_match":
                # This would match current context with user interests
                # For now, we'll simulate with a simple check
                return False  # Placeholder

            # Handle health check triggers
            elif trigger == "time_since_health_check":
                # This would check when last health activity occurred
                # For now, we'll simulate with a simple check
                return False  # Placeholder

            # Handle routine break triggers
            elif trigger == "routine_break_needed":
                # Check if user has been in routine for extended period
                if len(context.recent_activity) > 10:
                    # Check if activities are repetitive
                    unique_activities = len(set(context.recent_activity))
                    if unique_activities / len(context.recent_activity) < 0.3:
                        return True

        return False

    async def _create_enhanced_suggestion_from_template(
        self,
        template_id: str,
        template: Dict,
        context: EnhancedSuggestionContext,
    ) -> ContextualSuggestion:
        """Create an enhanced suggestion from a template"""
        suggestion_id = f"sug_{template_id}_{datetime.now().strftime('%H%M%S')}"

        description = template["description"]
        # Fill in enhanced context variables
        if "{battery_level}" in description:
            description = description.replace("{battery_level}", "20")
        if "{count}" in description:
            description = description.replace("{count}", "5")
        if "{file_count}" in description:
            description = description.replace("{file_count}", "50")
        if "{factor}" in description:
            # Determine suboptimal factor
            factor = "lighting"
            if context.environmental_data:
                temp = context.environmental_data.get("temperature", 22)
                light = context.environmental_data.get("light_level", 500)
                noise = context.environmental_data.get("noise_level", 40)

                if temp < 18 or temp > 25:
                    factor = "temperature"
                elif light < 300:
                    factor = "lighting"
                elif noise > 60:
                    factor = "noise control"
            description = description.replace("{factor}", factor)
        if "{topic}" in description:
            # Suggest a learning topic based on recent activity
            topic = "machine learning"  # Placeholder
            if context.recent_activity:
                # Simple topic inference
                activity_text = " ".join(context.recent_activity)
                if "code" in activity_text or "program" in activity_text:
                    topic = "advanced programming techniques"
                elif "design" in activity_text:
                    topic = "UI/UX principles"
                elif "write" in activity_text:
                    topic = "creative writing"
            description = description.replace("{topic}", topic)
        if "{activity}" in description:
            # Suggest a health activity
            activity = "take a walk"  # Placeholder
            if context.stress_level and context.stress_level > 0.7:
                activity = "do some stretching or meditation"
            elif context.focus_level and context.focus_level < 0.4:
                activity = "take a refreshing walk outside"
            description = description.replace("{activity}", activity)

        # Determine enhanced priority based on context
        priority = template["priority"]
        if isinstance(priority, EnhancedSuggestionPriority):
            # Map enhanced priority to standard priority
            priority_map = {
                EnhancedSuggestionPriority.CRITICAL: SuggestionPriority.URGENT,
                EnhancedSuggestionPriority.URGENT: SuggestionPriority.URGENT,
                EnhancedSuggestionPriority.HIGH: SuggestionPriority.HIGH,
                EnhancedSuggestionPriority.MEDIUM: SuggestionPriority.MEDIUM,
                EnhancedSuggestionPriority.LOW: SuggestionPriority.LOW,
            }
            priority = priority_map.get(priority, SuggestionPriority.MEDIUM)

        # Adjust confidence based on context match
        base_confidence = 0.7
        confidence_boost = 0.0

        # Boost confidence based on contextual factors
        if context.stress_level and context.stress_level > 0.7:
            confidence_boost += 0.1
        if context.focus_level and context.focus_level < 0.4:
            confidence_boost += 0.1
        if context.productivity_score and context.productivity_score < 0.5:
            confidence_boost += 0.1

        confidence = min(0.95, base_confidence + confidence_boost)

        return ContextualSuggestion(
            id=suggestion_id,
            title=template["title"],
            description=description,
            action=template["action"],
            action_type="template",
            priority=priority,
            confidence=confidence,
            context_triggers=template["context_triggers"],
            expires_at=datetime.now() + timedelta(hours=1),
            metadata={
                "template_id": template_id,
                "action_params": template.get("action_params", {}),
                "contextual_enhancement": {
                    "emotional_state": context.emotional_state,
                    "stress_level": context.stress_level,
                    "focus_level": context.focus_level,
                    "productivity_score": context.productivity_score,
                },
            },
        )

    async def _generate_enhanced_time_based_suggestions(
        self,
        context: EnhancedSuggestionContext,
    ) -> List[ContextualSuggestion]:
        """Generate enhanced time-based suggestions"""
        suggestions = await super()._generate_time_based_suggestions(context)

        # Add enhanced time-based suggestions
        enhanced_suggestions = []

        # Evening wind-down suggestions
        if context.time_of_day == "evening":
            if context.stress_level and context.stress_level > 0.6:
                suggestion = ContextualSuggestion(
                    id=f"sug_evening_winddown_{datetime.now().strftime('%H%M%S')}",
                    title="Evening Wind-Down",
                    description="You seem stressed. Try a relaxing evening routine to prepare for sleep.",
                    action="start_evening_routine",
                    action_type="workflow",
                    priority=SuggestionPriority.HIGH,
                    confidence=0.8,
                    context_triggers=["time_of_day=evening", "high_stress"],
                    expires_at=datetime.now() + timedelta(hours=2),
                    metadata={"stress_level": context.stress_level},
                )
                enhanced_suggestions.append(suggestion)

        # Morning energy optimization
        elif context.time_of_day == "morning":
            if context.focus_level and context.focus_level > 0.7:
                suggestion = ContextualSuggestion(
                    id=f"sug_morning_energy_{datetime.now().strftime('%H%M%S')}",
                    title="Harness Morning Energy",
                    description="You're in a high-focus state. Tackle your most challenging tasks now.",
                    action="suggest_priority_tasks",
                    action_type="info",
                    priority=SuggestionPriority.HIGH,
                    confidence=0.85,
                    context_triggers=["time_of_day=morning", "high_focus"],
                    expires_at=datetime.now() + timedelta(hours=3),
                    metadata={"focus_level": context.focus_level},
                )
                enhanced_suggestions.append(suggestion)

        suggestions.extend(enhanced_suggestions)
        return suggestions

    async def _generate_enhanced_system_suggestions(
        self,
        context: EnhancedSuggestionContext,
    ) -> List[ContextualSuggestion]:
        """Generate enhanced system-state-based suggestions"""
        suggestions = await super()._generate_system_suggestions(context)

        # Add enhanced system suggestions
        enhanced_suggestions = []

        # Resource optimization based on user state
        if context.system_load and context.system_load > 70:
            if context.stress_level and context.stress_level > 0.7:
                suggestion = ContextualSuggestion(
                    id=f"sug_system_optimization_{datetime.now().strftime('%H%M%S')}",
                    title="System Optimization",
                    description="High system load combined with your stress level. Close unnecessary applications.",
                    action="optimize_system_resources",
                    action_type="system",
                    priority=SuggestionPriority.HIGH,
                    confidence=0.8,
                    context_triggers=["high_cpu", "high_stress"],
                    expires_at=datetime.now() + timedelta(hours=1),
                    metadata={
                        "cpu_percent": context.system_load,
                        "stress_level": context.stress_level,
                    },
                )
                enhanced_suggestions.append(suggestion)

        # Low battery with productivity considerations
        # This would require actual battery level data
        # For now, we'll simulate with a generic check
        if context.productivity_score and context.productivity_score < 0.3:
            suggestion = ContextualSuggestion(
                id=f"sug_productivity_boost_{datetime.now().strftime('%H%M%S')}",
                title="Productivity Boost",
                description="Your productivity is low. Try organizing your workspace or changing your environment.",
                action="suggest_productivity_boost",
                action_type="info",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.7,
                context_triggers=["low_productivity"],
                expires_at=datetime.now() + timedelta(hours=2),
                metadata={"productivity_score": context.productivity_score},
            )
            enhanced_suggestions.append(suggestion)

        suggestions.extend(enhanced_suggestions)
        return suggestions

    async def _generate_environmental_suggestions(
        self,
        context: EnhancedSuggestionContext,
    ) -> List[ContextualSuggestion]:
        """Generate suggestions based on environmental context"""
        suggestions = []

        if not context.environmental_data:
            return suggestions

        temp = context.environmental_data.get("temperature", 22)
        light = context.environmental_data.get("light_level", 500)
        noise = context.environmental_data.get("noise_level", 40)
        air_quality = context.environmental_data.get("air_quality", 80)

        # Temperature suggestions
        if temp < 18:
            suggestion = ContextualSuggestion(
                id=f"sug_temp_low_{datetime.now().strftime('%H%M%S')}",
                title="Temperature Alert",
                description="Room temperature is low. Consider adjusting heating for comfort.",
                action="adjust_temperature",
                action_type="environment",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.8,
                context_triggers=["temperature_low"],
                expires_at=datetime.now() + timedelta(hours=1),
                metadata={"current_temp": temp, "optimal_range": "18-25°C"},
            )
            suggestions.append(suggestion)
        elif temp > 25:
            suggestion = ContextualSuggestion(
                id=f"sug_temp_high_{datetime.now().strftime('%H%M%S')}",
                title="Temperature Alert",
                description="Room temperature is high. Consider cooling down for better focus.",
                action="adjust_temperature",
                action_type="environment",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.8,
                context_triggers=["temperature_high"],
                expires_at=datetime.now() + timedelta(hours=1),
                metadata={"current_temp": temp, "optimal_range": "18-25°C"},
            )
            suggestions.append(suggestion)

        # Lighting suggestions
        if light < 300:
            suggestion = ContextualSuggestion(
                id=f"sug_light_low_{datetime.now().strftime('%H%M%S')}",
                title="Lighting Improvement",
                description="Lighting is dim. Increase brightness to reduce eye strain.",
                action="increase_lighting",
                action_type="environment",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.75,
                context_triggers=["lighting_low"],
                expires_at=datetime.now() + timedelta(hours=2),
                metadata={"current_light": light, "optimal_level": ">300 lux"},
            )
            suggestions.append(suggestion)

        # Noise suggestions
        if noise > 60:
            if context.focus_level and context.focus_level < 0.5:
                suggestion = ContextualSuggestion(
                    id=f"sug_noise_high_{datetime.now().strftime('%H%M%S')}",
                    title="Noise Reduction",
                    description="High noise levels are affecting your focus. Consider noise cancellation.",
                    action="activate_noise_cancellation",
                    action_type="environment",
                    priority=SuggestionPriority.HIGH,
                    confidence=0.85,
                    context_triggers=["noise_high", "low_focus"],
                    expires_at=datetime.now() + timedelta(hours=1),
                    metadata={
                        "current_noise": noise,
                        "optimal_level": "<60 dB",
                        "focus_level": context.focus_level,
                    },
                )
                suggestions.append(suggestion)

        # Air quality suggestions
        if air_quality < 50:
            suggestion = ContextualSuggestion(
                id=f"sug_air_poor_{datetime.now().strftime('%H%M%S')}",
                title="Air Quality Alert",
                description="Air quality is poor. Consider ventilation or air purification.",
                action="improve_air_quality",
                action_type="environment",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.8,
                context_triggers=["air_quality_poor"],
                expires_at=datetime.now() + timedelta(hours=3),
                metadata={"current_aqi": air_quality, "optimal_level": ">50"},
            )
            suggestions.append(suggestion)

        return suggestions

    async def _generate_emotional_suggestions(
        self,
        context: EnhancedSuggestionContext,
    ) -> List[ContextualSuggestion]:
        """Generate suggestions based on emotional and social context"""
        suggestions = []

        # Emotional state suggestions
        if context.emotional_state:
            if context.emotional_state in ["stressed", "anxious", "overwhelmed"]:
                suggestion = ContextualSuggestion(
                    id=f"sug_emotional_support_{datetime.now().strftime('%H%M%S')}",
                    title="Emotional Support",
                    description=f"You seem {context.emotional_state}. Try some relaxation techniques.",
                    action="suggest_relaxation_techniques",
                    action_type="wellbeing",
                    priority=SuggestionPriority.HIGH,
                    confidence=0.9,
                    context_triggers=["negative_emotional_state"],
                    expires_at=datetime.now() + timedelta(hours=1),
                    metadata={
                        "emotional_state": context.emotional_state,
                        "stress_level": context.stress_level,
                    },
                )
                suggestions.append(suggestion)
            elif context.emotional_state in ["bored", "restless"]:
                if context.productivity_score and context.productivity_score < 0.4:
                    suggestion = ContextualSuggestion(
                        id=f"sug_motivation_boost_{datetime.now().strftime('%H%M%S')}",
                        title="Motivation Boost",
                        description="You seem restless with low productivity. Try a quick energizing activity.",
                        action="suggest_energizing_activity",
                        action_type="wellbeing",
                        priority=SuggestionPriority.MEDIUM,
                        confidence=0.75,
                        context_triggers=["restless", "low_productivity"],
                        expires_at=datetime.now() + timedelta(hours=1),
                        metadata={
                            "emotional_state": context.emotional_state,
                            "productivity_score": context.productivity_score,
                        },
                    )
                    suggestions.append(suggestion)

        # Social context suggestions
        if context.social_context:
            social_density = context.social_context.get("people_nearby", 0)
            social_need = context.social_context.get("social_need", 0.5)

            if social_density == 0 and social_need > 0.7:
                suggestion = ContextualSuggestion(
                    id=f"sug_social_connection_{datetime.now().strftime('%H%M%S')}",
                    title="Social Connection",
                    description="You might benefit from social interaction. Reach out to someone.",
                    action="suggest_social_connection",
                    action_type="social",
                    priority=SuggestionPriority.MEDIUM,
                    confidence=0.7,
                    context_triggers=["social_isolation", "high_social_need"],
                    expires_at=datetime.now() + timedelta(hours=6),
                    metadata={"people_nearby": social_density, "social_need": social_need},
                )
                suggestions.append(suggestion)

        return suggestions

    async def _generate_predictive_suggestions(
        self,
        context: EnhancedSuggestionContext,
    ) -> List[ContextualSuggestion]:
        """Generate predictive suggestions based on learned patterns"""
        suggestions = []

        # Get pattern analyzer
        pattern_analyzer = await get_pattern_analyzer()

        # Get behavioral insights
        try:
            insights = await pattern_analyzer.get_stats()

            # Suggest based on pattern analysis
            if insights.get("high_confidence_patterns", 0) > 5:
                suggestion = ContextualSuggestion(
                    id=f"sug_pattern_leverage_{datetime.now().strftime('%H%M%S')}",
                    title="Leverage Your Patterns",
                    description="You have strong behavioral patterns. Leverage them for better efficiency.",
                    action="show_behavioral_patterns",
                    action_type="info",
                    priority=SuggestionPriority.MEDIUM,
                    confidence=0.8,
                    context_triggers=["strong_patterns"],
                    expires_at=datetime.now() + timedelta(hours=12),
                    metadata={"pattern_count": insights.get("high_confidence_patterns", 0)},
                )
                suggestions.append(suggestion)

        except Exception as e:
            log.debug(f"Could not get pattern insights: {e}")

        # Get anomaly detector
        anomaly_detector = await get_anomaly_detector()

        # Check for system anomalies that might affect suggestions
        try:
            active_anomalies = await anomaly_detector.get_active_anomalies()

            if active_anomalies:
                high_severity_anomalies = [
                    a for a in active_anomalies if a.severity.value in ["high", "critical"]
                ]

                if high_severity_anomalies:
                    suggestion = ContextualSuggestion(
                        id=f"sug_system_issues_{datetime.now().strftime('%H%M%S')}",
                        title="System Issues Detected",
                        description=f"{len(high_severity_anomalies)} critical system issues need attention.",
                        action="show_system_anomalies",
                        action_type="system",
                        priority=SuggestionPriority.URGENT,
                        confidence=0.95,
                        context_triggers=["system_anomalies"],
                        expires_at=datetime.now() + timedelta(hours=1),
                        metadata={
                            "anomaly_count": len(high_severity_anomalies),
                            "anomalies": [a.title for a in high_severity_anomalies[:3]],
                        },
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            log.debug(f"Could not check system anomalies: {e}")

        return suggestions

    async def _calculate_enhanced_suggestion_score(
        self,
        suggestion: ContextualSuggestion,
        context: EnhancedSuggestionContext,
    ) -> float:
        """Calculate enhanced relevance score for a suggestion using ML features"""
        # Start with base score
        score = suggestion.confidence

        # Boost by priority
        priority_boost = {
            SuggestionPriority.URGENT: 1.0,
            SuggestionPriority.HIGH: 0.5,
            SuggestionPriority.MEDIUM: 0.2,
            SuggestionPriority.LOW: 0.0,
        }
        score += priority_boost.get(suggestion.priority, 0)

        # Contextual relevance boosting
        if self.ml_models_initialized:
            contextual_boost = await self._calculate_contextual_relevance(suggestion, context)
            score += contextual_boost

        # Penalize if user dismissed similar recently
        if suggestion.id in self.dismissed_suggestions:
            score -= 0.5

        # Boost for pattern-based suggestions
        if "pattern" in suggestion.action_type:
            score += 0.1

        # Boost for environmental/contextual match
        contextual_triggers = suggestion.context_triggers
        if any("environment" in trigger for trigger in contextual_triggers):
            score += 0.15
        if any("emotional" in trigger for trigger in contextual_triggers):
            score += 0.1
        if any("social" in trigger for trigger in contextual_triggers):
            score += 0.1

        # Cap score at 1.0
        return min(1.0, score)

    async def _calculate_contextual_relevance(
        self,
        suggestion: ContextualSuggestion,
        context: EnhancedSuggestionContext,
    ) -> float:
        """Calculate contextual relevance using ML models"""
        # This is a simplified implementation
        # In a real system, this would use trained ML models

        relevance_score = 0.0

        # Extract context features
        context_features = []

        # Time features
        if context.time_of_day:
            context_features.append(f"time_{context.time_of_day}")
        if context.day_of_week is not None:
            context_features.append(f"day_{context.day_of_week}")

        # Activity features
        if context.recent_activity:
            context_features.extend(context.recent_activity[-5:])  # Last 5 activities

        # System features
        if context.system_load:
            context_features.append(f"system_load_{int(context.system_load // 10) * 10}")

        # Emotional features
        if context.emotional_state:
            context_features.append(f"emotion_{context.emotional_state}")
        if context.stress_level:
            context_features.append(f"stress_{int(context.stress_level * 10) // 2 * 2}")
        if context.focus_level:
            context_features.append(f"focus_{int(context.focus_level * 10) // 2 * 2}")

        # Environmental features
        if context.environmental_data:
            temp = context.environmental_data.get("temperature", 22)
            context_features.append(f"temp_{int(temp // 5) * 5}")

            light = context.environmental_data.get("light_level", 500)
            context_features.append(f"light_{int(light // 100) * 100}")

            noise = context.environmental_data.get("noise_level", 40)
            context_features.append(f"noise_{int(noise // 10) * 10}")

        # Convert to text for similarity calculation
        context_text = " ".join(context_features)
        suggestion_text = (
            f"{suggestion.title} {suggestion.description} {' '.join(suggestion.context_triggers)}"
        )

        # Simple keyword matching (would be replaced with ML in real implementation)
        common_words = set(context_text.lower().split()) & set(suggestion_text.lower().split())
        relevance_score = len(common_words) * 0.05

        return min(0.5, relevance_score)  # Cap at 0.5

    async def learn_from_feedback(self):
        """Learn from user feedback to improve suggestions"""
        # This would implement reinforcement learning from user interactions
        # For now, we'll just log that this functionality exists
        log.info("Enhanced suggestion engine learning from feedback")

        # In a real implementation, this would:
        # 1. Analyze which suggestions were accepted/rejected
        # 2. Update ML models based on feedback
        # 3. Adjust contextual weights
        # 4. Create new trigger patterns
        # 5. Optimize suggestion timing

    async def get_contextual_insights(self) -> Dict[str, Any]:
        """Get insights about contextual suggestion patterns"""
        insights = {
            "total_enhanced_suggestions": len(self.suggestions),
            "contextual_triggers_used": len(self.context_triggers),
            "environmental_suggestions": 0,
            "emotional_suggestions": 0,
            "predictive_suggestions": 0,
            "contextual_accuracy": 0.0,
        }

        try:
            # Count different types of suggestions
            env_count = 0
            emo_count = 0
            pred_count = 0

            for suggestion in self.suggestions.values():
                triggers = suggestion.context_triggers
                if any("environment" in trigger for trigger in triggers):
                    env_count += 1
                if any("emotional" in trigger or "stress" in trigger for trigger in triggers):
                    emo_count += 1
                if any("predict" in trigger or "pattern" in trigger for trigger in triggers):
                    pred_count += 1

            insights["environmental_suggestions"] = env_count
            insights["emotional_suggestions"] = emo_count
            insights["predictive_suggestions"] = pred_count

            # Calculate accuracy based on user feedback
            if self.user_feedback:
                helpful_count = sum(
                    1 for feedback in self.user_feedback.values() if feedback.get("helpful", False)
                )
                total_feedback = len(self.user_feedback)
                insights["contextual_accuracy"] = (
                    helpful_count / total_feedback if total_feedback > 0 else 0.0
                )

        except Exception as e:
            log.error(f"Error generating contextual insights: {e}")

        return insights


# Global instance
_enhanced_suggestion_engine: Optional[EnhancedSmartSuggestionEngine] = None


async def get_enhanced_suggestion_engine() -> EnhancedSmartSuggestionEngine:
    """Get the global enhanced suggestion engine instance"""
    global _enhanced_suggestion_engine
    if _enhanced_suggestion_engine is None:
        _enhanced_suggestion_engine = EnhancedSmartSuggestionEngine()
        await _enhanced_suggestion_engine.initialize()
    return _enhanced_suggestion_engine


__all__ = [
    "EnhancedSmartSuggestionEngine",
    "EnhancedSuggestionContext",
    "EnhancedContextTrigger",
    "EnhancedSuggestionPriority",
    "ContextTriggerType",
    "get_enhanced_suggestion_engine",
]
