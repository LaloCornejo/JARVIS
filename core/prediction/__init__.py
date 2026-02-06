"""
Predictive and Proactive Features for JARVIS.

Provides:
- Pattern analysis and behavioral prediction
- Smart contextual suggestions
- Anomaly detection
- Predictive task scheduling
"""

from core.prediction.anomaly_detector import (
    AnomalyDetector,
    AnomalyReport,
    AnomalyType,
    get_anomaly_detector,
)
from core.prediction.pattern_analyzer import (
    PatternAnalyzer,
    PatternMatch,
    PatternType,
    UserPattern,
    get_pattern_analyzer,
)
from core.prediction.suggestion_engine import (
    ContextualSuggestion,
    SmartSuggestionEngine,
    SuggestionPriority,
    get_suggestion_engine,
)

__all__ = [
    "PatternAnalyzer",
    "UserPattern",
    "PatternType",
    "PatternMatch",
    "get_pattern_analyzer",
    "SmartSuggestionEngine",
    "ContextualSuggestion",
    "SuggestionPriority",
    "get_suggestion_engine",
    "AnomalyDetector",
    "AnomalyReport",
    "AnomalyType",
    "get_anomaly_detector",
]
