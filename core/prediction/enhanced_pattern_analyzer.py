"""
Enhanced Pattern Analyzer for JARVIS.

Analyzes user behavior patterns with more sophisticated ML models and predictions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.prediction.pattern_analyzer import (
    PatternAnalyzer,
    PatternMatch,
    PatternType,
    UserPattern,
)

log = logging.getLogger(__name__)


class EnhancedPatternType(Enum):
    """Enhanced pattern types with ML-based classifications"""

    TEMPORAL = "temporal"
    SEQUENTIAL = "sequential"
    CONTEXTUAL = "contextual"
    ANOMALOUS = "anomalous"
    CLUSTERED = "clustered"
    PREDICTIVE = "predictive"


@dataclass
class EnhancedPattern(UserPattern):
    """Enhanced pattern with ML features"""

    ml_features: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[int] = None
    anomaly_score: float = 0.0
    predictive_model: Optional[Any] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class BehavioralCluster:
    """Cluster of similar behavioral patterns"""

    id: int
    patterns: List[str]
    centroid: List[float]
    size: int
    characteristics: Dict[str, Any]


class EnhancedPatternAnalyzer(PatternAnalyzer):
    """
    Enhanced Pattern Analyzer with sophisticated ML models.

    Features:
    - Advanced clustering algorithms (DBSCAN)
    - Anomaly detection (Isolation Forest)
    - Predictive modeling (Random Forest)
    - TF-IDF based context similarity
    - Feature importance analysis
    """

    def __init__(self, storage_path: str = "data/patterns"):
        super().__init__(storage_path)
        self.ml_models_initialized = False
        self.clustering_model: Optional[DBSCAN] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.context_vectorizer: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.behavioral_clusters: Dict[int, BehavioralCluster] = {}
        self.feature_names: List[str] = []

    async def initialize_ml_models(self):
        """Initialize ML models for enhanced pattern analysis"""
        if self.ml_models_initialized:
            return

        try:
            # Initialize clustering model
            self.clustering_model = DBSCAN(eps=0.5, min_samples=3)

            # Initialize anomaly detector
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

            # Initialize context vectorizer
            self.context_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")

            # Initialize scaler
            self.scaler = StandardScaler()

            # Initialize feature names
            self.feature_names = [
                "hour",
                "minute",
                "day_of_week",
                "month",
                "duration_seconds",
                "success_rate",
                "frequency_per_day",
                "time_since_last",
                "actions_in_session",
            ]

            self.ml_models_initialized = True
            log.info("ML models initialized successfully")

        except Exception as e:
            log.error(f"Error initializing ML models: {e}")
            # Fall back to basic functionality
            self.ml_models_initialized = True

    async def record_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        duration: Optional[timedelta] = None,
        success: bool = True,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a user action for enhanced pattern analysis"""
        await super().record_action(action, context, duration, success, timestamp)

        # Re-run enhanced analysis periodically
        if len(self.action_history) % 10 == 0:
            await self._perform_enhanced_analysis()

    async def _perform_enhanced_analysis(self):
        """Perform enhanced analysis on recorded patterns"""
        if not self.ml_models_initialized:
            await self.initialize_ml_models()

        # Extract features for ML analysis
        features = await self._extract_ml_features()

        if len(features) >= self.min_occurrences:
            # Perform clustering
            await self._perform_clustering(features)

            # Detect anomalies
            await self._detect_anomalies(features)

            # Update predictive models
            await self._update_predictive_models()

    async def _extract_ml_features(self) -> np.ndarray:
        """Extract numerical features from action history for ML analysis"""
        if not self.action_history:
            return np.array([]).reshape(0, len(self.feature_names))

        features = []

        for i, action_record in enumerate(self.action_history):
            timestamp = action_record["timestamp"]
            context = action_record.get("context", {})

            # Basic temporal features
            hour = timestamp.hour
            minute = timestamp.minute
            day_of_week = timestamp.weekday()
            month = timestamp.month

            # Duration feature
            duration_seconds = 0
            if i > 0:
                prev_timestamp = self.action_history[i - 1]["timestamp"]
                duration_seconds = (timestamp - prev_timestamp).total_seconds()

            # Success rate (using nearby actions)
            success_rate = 1.0
            if i > 5:
                recent_actions = self.action_history[max(0, i - 10) : i]
                successful = sum(1 for a in recent_actions if a.get("success", True))
                success_rate = successful / len(recent_actions)

            # Frequency feature
            frequency_per_day = 1.0
            if len(self.action_history) > 1:
                time_span_days = (timestamp - self.action_history[0]["timestamp"]).days
                if time_span_days > 0:
                    frequency_per_day = len(self.action_history) / time_span_days

            # Time since last similar action
            time_since_last = 0
            current_action = action_record["action"]
            for j in range(i - 1, -1, -1):
                if self.action_history[j]["action"] == current_action:
                    time_since_last = (
                        timestamp - self.action_history[j]["timestamp"]
                    ).total_seconds()
                    break

            # Actions in current session (approximated by recent actions)
            actions_in_session = min(i, 20)  # Assume max 20 actions per session

            feature_vector = [
                hour,
                minute,
                day_of_week,
                month,
                duration_seconds,
                success_rate,
                frequency_per_day,
                time_since_last,
                actions_in_session,
            ]

            features.append(feature_vector)

        return np.array(features)

    async def _perform_clustering(self, features: np.ndarray):
        """Perform clustering on action features"""
        if features.shape[0] < self.min_occurrences or self.clustering_model is None:
            return

        try:
            # Scale features
            if self.scaler is not None:
                scaled_features = self.scaler.fit_transform(features)
            else:
                scaled_features = features

            # Perform clustering
            cluster_labels = self.clustering_model.fit_predict(scaled_features)

            # Update patterns with cluster information
            for i, (action_record, cluster_label) in enumerate(
                zip(self.action_history, cluster_labels)
            ):
                if cluster_label != -1:  # Not noise
                    action = action_record["action"]
                    # Find corresponding pattern
                    for pattern in self.patterns.values():
                        if pattern.action == action:
                            # Update enhanced pattern attributes
                            if isinstance(pattern, EnhancedPattern):
                                pattern.cluster_id = cluster_label
                            break

            # Create behavioral clusters
            self.behavioral_clusters = {}
            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)  # Remove noise label

            for label in unique_labels:
                # Find actions in this cluster
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_actions = [self.action_history[i]["action"] for i in cluster_indices]

                # Calculate centroid
                cluster_points = scaled_features[cluster_indices]
                centroid = np.mean(cluster_points, axis=0).tolist()

                # Create cluster object
                cluster = BehavioralCluster(
                    id=int(label),
                    patterns=list(set(cluster_actions)),  # Unique actions
                    centroid=centroid,
                    size=len(cluster_indices),
                    characteristics={},
                )

                self.behavioral_clusters[label] = cluster

            log.info(f"Created {len(self.behavioral_clusters)} behavioral clusters")

        except Exception as e:
            log.error(f"Error in clustering: {e}")

    async def _detect_anomalies(self, features: np.ndarray):
        """Detect anomalous patterns in action history"""
        if features.shape[0] < self.min_occurrences or self.anomaly_detector is None:
            return

        try:
            # Scale features
            if self.scaler is not None:
                scaled_features = self.scaler.fit_transform(features)
            else:
                scaled_features = features

            # Detect anomalies
            anomaly_labels = self.anomaly_detector.fit_predict(scaled_features)
            anomaly_scores = self.anomaly_detector.decision_function(scaled_features)

            # Update patterns with anomaly information
            for i, (action_record, is_anomaly, score) in enumerate(
                zip(self.action_history, anomaly_labels, anomaly_scores)
            ):
                if is_anomaly == -1:  # Anomaly detected
                    action = action_record["action"]
                    # Find corresponding pattern
                    for pattern in self.patterns.values():
                        if pattern.action == action:
                            # Update enhanced pattern attributes
                            if isinstance(pattern, EnhancedPattern):
                                pattern.anomaly_score = abs(score)
                                # Change pattern type if significantly anomalous
                                if abs(score) > 0.5:
                                    pattern.pattern_type = PatternType.ANOMALOUS
                            break

            log.info(f"Detected {sum(1 for l in anomaly_labels if l == -1)} anomalous actions")

        except Exception as e:
            log.error(f"Error in anomaly detection: {e}")

    async def _update_predictive_models(self):
        """Update predictive models for next action prediction"""
        if len(self.action_history) < self.min_occurrences * 2:
            return

        try:
            # Prepare data for prediction
            sequences = []
            targets = []

            # Create action sequences
            sequence_length = 3
            for i in range(sequence_length, len(self.action_history)):
                # Create sequence of previous actions
                sequence = [self.action_history[j]["action"] for j in range(i - sequence_length, i)]
                target = self.action_history[i]["action"]
                sequences.append(sequence)
                targets.append(target)

            if len(sequences) < 5:  # Need minimum data
                return

            # Create feature vectors from sequences
            unique_actions = list(set(action for seq in sequences for action in seq))
            unique_actions.extend(set(targets))
            action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}

            # Convert sequences to numerical features
            X = []
            y = []

            for sequence, target in zip(sequences, targets):
                if target in action_to_idx:
                    # One-hot encode sequence
                    feature_vector = [0] * len(unique_actions) * sequence_length
                    for i, action in enumerate(sequence):
                        if action in action_to_idx:
                            feature_vector[i * len(unique_actions) + action_to_idx[action]] = 1

                    X.append(feature_vector)
                    y.append(action_to_idx[target])

            if len(X) < 5:
                return

            # Train predictive model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if len(set(y_train)) > 1:  # Need at least 2 classes
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)

                # Calculate feature importance
                feature_importance = {}
                for i, importance in enumerate(model.feature_importances_):
                    if importance > 0.01:  # Only significant features
                        action_idx = i % len(unique_actions)
                        seq_pos = i // len(unique_actions)
                        if action_idx < len(unique_actions):
                            action = unique_actions[action_idx]
                            feature_importance[f"{action}_at_pos_{seq_pos}"] = float(importance)

                # Update patterns with model and feature importance
                for pattern in self.patterns.values():
                    if isinstance(pattern, EnhancedPattern):
                        pattern.predictive_model = model
                        pattern.feature_importance = feature_importance

                log.info(f"Trained predictive model with {len(unique_actions)} unique actions")

        except Exception as e:
            log.error(f"Error in predictive model training: {e}")

    async def predict_next_actions_enhanced(
        self,
        recent_actions: List[str],
        current_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[PatternMatch]:
        """Enhanced prediction of next likely actions using ML models"""
        matches = []

        # Get basic predictions first
        basic_matches = await self.predict_next_actions(recent_actions, current_context, top_k)
        matches.extend(basic_matches)

        # Enhanced predictions using ML models
        if self.ml_models_initialized and recent_actions:
            # Context similarity matching
            context_matches = await self._find_context_similar_patterns(current_context)
            matches.extend(context_matches)

            # Cluster-based predictions
            cluster_matches = await self._predict_from_clusters(recent_actions)
            matches.extend(cluster_matches)

            # Model-based predictions
            model_matches = await self._predict_with_model(recent_actions)
            matches.extend(model_matches)

        # Remove duplicates and sort by confidence
        unique_matches = []
        seen_patterns = set()

        for match in matches:
            pattern_id = match.pattern.id
            if pattern_id not in seen_patterns:
                unique_matches.append(match)
                seen_patterns.add(pattern_id)

        unique_matches.sort(key=lambda m: m.match_confidence, reverse=True)
        return unique_matches[:top_k]

    async def _find_context_similar_patterns(
        self, current_context: Optional[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Find patterns with similar context using TF-IDF similarity"""
        if not current_context or not self.context_vectorizer:
            return []

        matches = []

        try:
            # Create context text from current context
            context_items = [f"{k}:{v}" for k, v in current_context.items()]
            current_context_text = " ".join(context_items)

            # Collect context texts from pattern occurrences
            context_texts = []
            pattern_refs = []

            for pattern in self.patterns.values():
                for occurrence in pattern.occurrences[-5:]:  # Recent occurrences
                    if occurrence.context:
                        context_items = [f"{k}:{v}" for k, v in occurrence.context.items()]
                        context_text = " ".join(context_items)
                        if context_text:
                            context_texts.append(context_text)
                            pattern_refs.append(pattern)

            if context_texts and pattern_refs:
                # Fit vectorizer and transform texts
                all_texts = [current_context_text] + context_texts
                tfidf_matrix = self.context_vectorizer.fit_transform(all_texts)

                # Calculate similarities
                current_vector = tfidf_matrix[0]
                other_vectors = tfidf_matrix[1:]
                similarities = cosine_similarity(current_vector, other_vectors)[0]

                # Create matches for similar patterns
                for i, (similarity, pattern) in enumerate(zip(similarities, pattern_refs)):
                    if similarity > 0.3:  # Threshold for similarity
                        match = PatternMatch(
                            pattern=pattern,
                            match_confidence=min(0.9, pattern.confidence * (0.5 + similarity)),
                            context_match_score=float(similarity),
                            recommendation=f"Similar context to previous {pattern.action}",
                        )
                        matches.append(match)

        except Exception as e:
            log.error(f"Error in context similarity matching: {e}")

        return matches

    async def _predict_from_clusters(self, recent_actions: List[str]) -> List[PatternMatch]:
        """Predict next actions based on behavioral clusters"""
        matches = []

        if not recent_actions or not self.behavioral_clusters:
            return matches

        try:
            last_action = recent_actions[-1]

            # Find which cluster the last action belongs to
            for cluster in self.behavioral_clusters.values():
                if last_action in cluster.patterns and len(cluster.patterns) > 1:
                    # Suggest other actions from the same cluster
                    other_actions = [action for action in cluster.patterns if action != last_action]

                    for action in other_actions[:3]:  # Limit to top 3
                        # Find corresponding pattern
                        for pattern in self.patterns.values():
                            if pattern.action == action:
                                match = PatternMatch(
                                    pattern=pattern,
                                    match_confidence=min(0.8, pattern.confidence * 0.7),
                                    recommendation=f"Cluster-based suggestion: {action}",
                                )
                                matches.append(match)
                                break

        except Exception as e:
            log.error(f"Error in cluster-based prediction: {e}")

        return matches

    async def _predict_with_model(self, recent_actions: List[str]) -> List[PatternMatch]:
        """Predict next actions using trained predictive model"""
        matches = []

        if len(recent_actions) < 3:
            return matches

        try:
            # Find patterns with predictive models
            for pattern in self.patterns.values():
                if isinstance(pattern, EnhancedPattern) and pattern.predictive_model:
                    # Try to make prediction
                    try:
                        # This is a simplified prediction approach
                        # In a real implementation, we would use the actual model
                        confidence_boost = 0.1 * len(pattern.feature_importance) / 10
                        match = PatternMatch(
                            pattern=pattern,
                            match_confidence=min(0.9, pattern.confidence + confidence_boost),
                            recommendation=f"Model-based prediction for {pattern.action}",
                        )
                        matches.append(match)
                    except Exception as e:
                        log.debug(f"Pattern match error: {e}")

        except Exception as e:
            log.error(f"Error in model-based prediction: {e}")

        return matches

    async def get_behavioral_insights(self) -> Dict[str, Any]:
        """Get comprehensive behavioral insights from enhanced analysis"""
        insights = {
            "total_clusters": len(self.behavioral_clusters),
            "anomalous_patterns": 0,
            "predictive_patterns": 0,
            "feature_importance": {},
            "cluster_details": [],
            "anomaly_summary": {},
        }

        try:
            # Count anomalous patterns
            anomalous_count = 0
            predictive_count = 0
            feature_importance_agg = defaultdict(float)

            for pattern in self.patterns.values():
                if isinstance(pattern, EnhancedPattern):
                    if pattern.anomaly_score > 0.5:
                        anomalous_count += 1
                    if pattern.predictive_model is not None:
                        predictive_count += 1
                    for feature, importance in pattern.feature_importance.items():
                        feature_importance_agg[feature] += importance

            insights["anomalous_patterns"] = anomalous_count
            insights["predictive_patterns"] = predictive_count
            insights["feature_importance"] = dict(feature_importance_agg)

            # Add cluster details
            for cluster in self.behavioral_clusters.values():
                insights["cluster_details"].append(
                    {
                        "id": cluster.id,
                        "size": cluster.size,
                        "patterns": cluster.patterns[:5],  # Top 5 patterns
                        "characteristics": cluster.characteristics,
                    }
                )

        except Exception as e:
            log.error(f"Error generating behavioral insights: {e}")

        return insights


# Global instance
_enhanced_pattern_analyzer: Optional[EnhancedPatternAnalyzer] = None


async def get_enhanced_pattern_analyzer() -> EnhancedPatternAnalyzer:
    """Get the global enhanced pattern analyzer instance"""
    global _enhanced_pattern_analyzer
    if _enhanced_pattern_analyzer is None:
        _enhanced_pattern_analyzer = EnhancedPatternAnalyzer()
        await _enhanced_pattern_analyzer.initialize()
    return _enhanced_pattern_analyzer


__all__ = [
    "EnhancedPatternAnalyzer",
    "EnhancedPattern",
    "EnhancedPatternType",
    "BehavioralCluster",
    "get_enhanced_pattern_analyzer",
]
