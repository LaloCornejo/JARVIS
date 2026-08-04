"""
Enhanced Anomaly Detection System for JARVIS.

Detects unusual behavior, system anomalies, and potential issues using advanced ML techniques.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from core.prediction.anomaly_detector import (
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    AnomalyType,
    MetricHistory,
)

log = logging.getLogger(__name__)


class EnhancedAnomalyType(Enum):
    """Enhanced types of anomalies with ML-based classifications"""

    BEHAVIORAL = "behavioral"  # Unusual user behavior
    SYSTEM = "system"  # System performance issues
    SECURITY = "security"  # Potential security issues
    TEMPORAL = "temporal"  # Time-based anomalies
    FREQUENCY = "frequency"  # Unusual frequency of actions
    PATTERN_BREAK = "pattern_break"  # Breaking established patterns
    PERFORMANCE = "performance"  # Performance degradation
    RESOURCE = "resource"  # Resource usage anomalies
    CLUSTER_OUTLIER = "cluster_outlier"  # Outlier in behavioral clusters
    SEASONAL_DEVIATION = "seasonal_deviation"  # Deviation from seasonal patterns
    CORRELATION_BREAK = "correlation_break"  # Breaking metric correlations


@dataclass
class EnhancedAnomalyReport(AnomalyReport):
    """Enhanced anomaly report with ML features"""

    ml_features: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[int] = None
    isolation_score: float = 0.0
    local_outlier_factor: float = 0.0
    prediction_probability: float = 0.0
    false_positive_probability: float = 0.0


class EnhancedMetricHistory(MetricHistory):
    """Enhanced metric history with ML-based anomaly detection"""

    def __init__(self, name: str, window_size: int = 100):
        super().__init__(name, window_size)
        self.ml_models_initialized = False
        self.isolation_forest: Optional[IsolationForest] = None
        self.local_outlier_detector: Optional[LocalOutlierFactor] = None
        self.scaler: Optional[StandardScaler] = None
        self.seasonal_patterns: Dict[str, List[float]] = {}
        self.correlations: Dict[str, float] = {}

    async def initialize_ml_models(self):
        """Initialize ML models for enhanced anomaly detection"""
        if self.ml_models_initialized:
            return

        try:
            # Initialize isolation forest for anomaly detection
            self.isolation_forest = IsolationForest(
                contamination=0.1, random_state=42, n_estimators=100
            )

            # Initialize local outlier factor
            self.local_outlier_detector = LocalOutlierFactor(
                n_neighbors=min(20, len(self.values) // 2) if self.values else 20, contamination=0.1
            )

            # Initialize scaler
            self.scaler = StandardScaler()

            self.ml_models_initialized = True
            log.info(f"ML models initialized for metric: {self.name}")

        except Exception as e:
            log.error(f"Error initializing ML models for {self.name}: {e}")
            self.ml_models_initialized = True  # Continue with basic functionality

    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new value with enhanced tracking"""
        super().add(value, timestamp)

        # Update seasonal patterns
        if timestamp:
            self._update_seasonal_patterns(value, timestamp)

        # Update correlations with related metrics (simplified)
        self._update_correlations(value)

    def _update_seasonal_patterns(self, value: float, timestamp: datetime):
        """Update seasonal patterns for the metric"""
        # Extract time-based features
        hour_key = f"hour_{timestamp.hour}"
        day_key = f"day_{timestamp.weekday()}"
        month_key = f"month_{timestamp.month}"

        # Update patterns (simplified approach)
        for key in [hour_key, day_key, month_key]:
            if key not in self.seasonal_patterns:
                self.seasonal_patterns[key] = []
            self.seasonal_patterns[key].append(value)

            # Keep only recent values
            if len(self.seasonal_patterns[key]) > 50:
                self.seasonal_patterns[key] = self.seasonal_patterns[key][-50:]

    def _update_correlations(self, value: float):
        """Update correlations with other metrics (simplified)"""
        # In a real implementation, this would track correlations with other metrics
        # For now, we'll just maintain a placeholder
        pass

    def is_anomaly_enhanced(
        self, value: float, threshold_std: float = 3.0
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Enhanced anomaly detection using ML models"""
        # Get basic statistical anomaly detection
        is_basic_anomaly, basic_deviation = self.is_anomaly(value, threshold_std)

        # Initialize ML features
        ml_features = {
            "statistical_anomaly": is_basic_anomaly,
            "statistical_deviation": basic_deviation,
            "value": value,
            "baseline_mean": self.baseline_mean or 0,
            "baseline_std": self.baseline_std or 0,
            "historical_min": min([v for _, v in self.values]) if self.values else value,
            "historical_max": max([v for _, v in self.values]) if self.values else value,
        }

        # Enhanced detection if we have enough data
        if len(self.values) >= 10 and self.ml_models_initialized:
            try:
                # Prepare data for ML models
                recent_values = np.array([v for _, v in self.values[-50:]]).reshape(-1, 1)

                if self.scaler is not None:
                    scaled_values = self.scaler.fit_transform(recent_values)
                    scaled_value = self.scaler.transform([[value]])[0][0]
                else:
                    scaled_values = recent_values
                    scaled_value = value

                # Isolation Forest detection
                if self.isolation_forest is not None:
                    self.isolation_forest.fit(scaled_values)
                    isolation_prediction = self.isolation_forest.predict([[scaled_value]])[0]
                    isolation_score = self.isolation_forest.decision_function([[scaled_value]])[0]
                    ml_features["isolation_anomaly"] = isolation_prediction == -1
                    ml_features["isolation_score"] = float(isolation_score)

                # Local Outlier Factor detection
                if self.local_outlier_detector is not None and len(scaled_values) >= 10:
                    try:
                        # LOF needs at least n_neighbors samples
                        n_neighbors = min(
                            len(scaled_values) - 1, self.local_outlier_detector.n_neighbors
                        )
                        if n_neighbors > 1:
                            lof_detector = LocalOutlierFactor(
                                n_neighbors=n_neighbors, contamination=0.1
                            )
                            lof_predictions = lof_detector.fit_predict(scaled_values)
                            lof_score = (
                                lof_detector.negative_outlier_factor_[len(scaled_values) - 1]
                                if len(scaled_values) > 0
                                else 0
                            )
                            ml_features["lof_anomaly"] = (
                                (lof_predictions[-1] == -1) if len(lof_predictions) > 0 else False
                            )
                            ml_features["lof_score"] = float(lof_score)
                    except Exception as e:
                        log.debug(f"LOF detection failed for {self.name}: {e}")

            except Exception as e:
                log.debug(f"Enhanced anomaly detection failed for {self.name}: {e}")

        # Combine all detection methods
        anomaly_indicators = [
            is_basic_anomaly,
            ml_features.get("isolation_anomaly", False),
            ml_features.get("lof_anomaly", False),
        ]

        # Weighted combination (basic gets highest weight)
        combined_score = (
            basic_deviation * 0.6
            + abs(ml_features.get("isolation_score", 0)) * 0.25
            + abs(ml_features.get("lof_score", 0)) * 0.15
        )

        # Final anomaly decision (majority voting)
        is_enhanced_anomaly = sum(anomaly_indicators) >= 2 or is_basic_anomaly

        return is_enhanced_anomaly, combined_score, ml_features


class EnhancedAnomalyDetector(AnomalyDetector):
    """
    Enhanced Anomaly Detector with sophisticated ML models.

    Features:
    - Isolation Forest for unsupervised anomaly detection
    - Local Outlier Factor for local anomaly detection
    - Seasonal pattern analysis
    - Correlation-based anomaly detection
    - Ensemble anomaly scoring
    - False positive reduction
    """

    def __init__(self, storage_path: str = "data/anomalies"):
        super().__init__(storage_path)
        self.ml_models_initialized = False
        self.correlation_detector: Optional[RandomForestClassifier] = None
        self.metric_clusters: Dict[str, KMeans] = {}
        self.false_positive_model: Optional[RandomForestClassifier] = None

    async def initialize_ml_models(self):
        """Initialize ML models for enhanced anomaly detection"""
        if self.ml_models_initialized:
            return

        try:
            # Initialize correlation-based anomaly detector
            self.correlation_detector = RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=10
            )

            # Initialize false positive detection model
            self.false_positive_model = RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=5
            )

            self.ml_models_initialized = True
            log.info("Enhanced ML models initialized")

        except Exception as e:
            log.error(f"Error initializing enhanced ML models: {e}")
            self.ml_models_initialized = True  # Continue with basic functionality

    async def check_metric_enhanced(
        self,
        metric_name: str,
        value: float,
        anomaly_type: AnomalyType,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnhancedAnomalyReport]:
        """Enhanced check a metric for anomalies using ML models"""
        async with self._lock:
            # Get or create enhanced metric history
            if metric_name not in self.metric_histories:
                self.metric_histories[metric_name] = EnhancedMetricHistory(metric_name)

            history = self.metric_histories[metric_name]

            # Initialize ML models if needed
            if isinstance(history, EnhancedMetricHistory) and not history.ml_models_initialized:
                await history.initialize_ml_models()

            # Add value
            history.add(value)

            # Check for anomaly using enhanced detection
            if isinstance(history, EnhancedMetricHistory):
                is_anomaly, deviation, ml_features = history.is_anomaly_enhanced(value)
            else:
                is_anomaly, deviation = history.is_anomaly(value)
                ml_features = {}

            if not is_anomaly:
                return None

            # Determine severity
            severity = self._deviation_to_severity(deviation)

            # Create enhanced anomaly report
            report = await self._create_enhanced_anomaly_report(
                anomaly_type=anomaly_type,
                severity=severity,
                metric_name=metric_name,
                metric_value=value,
                expected_range=(
                    (history.baseline_mean - history.baseline_std * 2)
                    if history.baseline_mean and history.baseline_std
                    else value * 0.9,
                    (history.baseline_mean + history.baseline_std * 2)
                    if history.baseline_mean and history.baseline_std
                    else value * 1.1,
                ),
                deviation_score=deviation,
                context=context or {},
                ml_features=ml_features,
            )

            self.anomalies[report.id] = report
            await self._save_data()

            log.warning(f"Enhanced anomaly detected: {report.title} (severity: {severity.value})")
            return report

    async def _create_enhanced_anomaly_report(
        self,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        metric_name: str,
        metric_value: float,
        expected_range: Tuple[float, float],
        deviation_score: float,
        context: Dict[str, Any],
        ml_features: Dict[str, Any],
    ) -> EnhancedAnomalyReport:
        """Create an enhanced anomaly report"""
        anomaly_id = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Generate title and description based on type
        title, description = self._generate_description(
            anomaly_type, metric_name, metric_value, expected_range, deviation_score
        )

        # Find related anomalies
        related = await self._find_related_anomalies(anomaly_type, metric_name)

        # Calculate false positive probability
        false_positive_prob = await self._calculate_false_positive_probability(
            metric_name, metric_value, ml_features
        )

        return EnhancedAnomalyReport(
            id=anomaly_id,
            anomaly_type=anomaly_type,
            severity=severity,
            title=title,
            description=description,
            detected_at=datetime.now(),
            metric_name=metric_name,
            metric_value=metric_value,
            expected_range=expected_range,
            deviation_score=deviation_score,
            context=context,
            related_anomalies=related,
            ml_features=ml_features,
            false_positive_probability=false_positive_prob,
        )

    async def _calculate_false_positive_probability(
        self, metric_name: str, metric_value: float, ml_features: Dict[str, Any]
    ) -> float:
        """Calculate the probability that an anomaly is a false positive"""
        # This is a simplified implementation
        # In a real system, this would be based on historical data about false positives

        # Base probability based on deviation score
        base_prob = max(0.0, min(1.0, 1.0 - (ml_features.get("statistical_deviation", 0) / 5.0)))

        # Adjust based on ML scores
        iso_score = abs(ml_features.get("isolation_score", 0))
        lof_score = abs(ml_features.get("lof_score", 0))

        # If multiple detectors agree, it's less likely to be a false positive
        agreement_count = sum(
            [
                ml_features.get("statistical_anomaly", False),
                ml_features.get("isolation_anomaly", False),
                ml_features.get("lof_anomaly", False),
            ]
        )

        agreement_factor = (
            1.0 - (agreement_count / 3.0) * 0.5
        )  # Reduce probability if multiple agree

        # Combine factors
        final_prob = base_prob * agreement_factor

        return max(0.0, min(1.0, final_prob))

    async def detect_multi_metric_anomalies(
        self, metrics: Dict[str, float]
    ) -> List[EnhancedAnomalyReport]:
        """Detect anomalies across multiple correlated metrics"""
        reports = []

        try:
            # Check individual metrics
            for metric_name, value in metrics.items():
                # Determine appropriate anomaly type based on metric name
                anomaly_type = self._infer_anomaly_type(metric_name)

                report = await self.check_metric_enhanced(
                    metric_name=metric_name,
                    value=value,
                    anomaly_type=anomaly_type,
                )

                if report:
                    reports.append(report)

            # Check for correlation breaks between metrics
            correlation_reports = await self._check_metric_correlations(metrics)
            reports.extend(correlation_reports)

        except Exception as e:
            log.error(f"Error in multi-metric anomaly detection: {e}")

        return reports

    def _infer_anomaly_type(self, metric_name: str) -> AnomalyType:
        """Infer anomaly type based on metric name"""
        metric_lower = metric_name.lower()

        if any(term in metric_lower for term in ["cpu", "memory", "disk", "ram", "storage"]):
            return AnomalyType.RESOURCE
        elif any(term in metric_lower for term in ["response", "latency", "time"]):
            return AnomalyType.PERFORMANCE
        elif any(term in metric_lower for term in ["error", "fail", "exception"]):
            return AnomalyType.SYSTEM
        elif any(term in metric_lower for term in ["user", "behavior", "action"]):
            return AnomalyType.BEHAVIORAL
        else:
            return AnomalyType.SYSTEM

    async def _check_metric_correlations(
        self, metrics: Dict[str, float]
    ) -> List[EnhancedAnomalyReport]:
        """Check for correlation breaks between metrics"""
        reports = []

        # This is a simplified implementation
        # In a real system, this would analyze historical correlations

        # For now, we'll just check if metrics that normally correlate are diverging
        metric_names = list(metrics.keys())

        # Simple pairwise correlation check (placeholder)
        if len(metric_names) >= 2:
            # Check if CPU and memory typically correlate but are now diverging
            cpu_metrics = [name for name in metric_names if "cpu" in name.lower()]
            memory_metrics = [name for name in metric_names if "memory" in name.lower()]

            if cpu_metrics and memory_metrics:
                cpu_value = metrics[cpu_metrics[0]]
                memory_value = metrics[memory_metrics[0]]

                # If one is high and the other is low, it might be anomalous
                if (cpu_value > 80 and memory_value < 20) or (cpu_value < 20 and memory_value > 80):
                    report = await self._create_enhanced_anomaly_report(
                        anomaly_type=AnomalyType.CORRELATION_BREAK,
                        severity=AnomalySeverity.MEDIUM,
                        metric_name="cpu_memory_correlation",
                        metric_value=max(cpu_value, memory_value),
                        expected_range=(20, 80),
                        deviation_score=2.5,
                        context={
                            "correlated_metrics": [cpu_metrics[0], memory_metrics[0]],
                            "cpu_value": cpu_value,
                            "memory_value": memory_value,
                        },
                        ml_features={"correlation_break": True, "cpu_memory_divergence": True},
                    )
                    reports.append(report)

        return reports

    async def get_anomaly_patterns(self) -> Dict[str, Any]:
        """Get insights about anomaly patterns and trends"""
        insights = {
            "total_anomalies": len(self.anomalies),
            "anomalies_by_type": {},
            "false_positive_rate": 0.0,
            "detection_accuracy": 0.0,
            "common_combinations": [],
            "trend_analysis": {},
        }

        try:
            # Count anomalies by type
            by_type = {}
            false_positives = 0
            acknowledged = 0

            for anomaly in self.anomalies.values():
                t = anomaly.anomaly_type.value
                by_type[t] = by_type.get(t, 0) + 1

                if getattr(anomaly, "false_positive", False):
                    false_positives += 1

                if anomaly.acknowledged:
                    acknowledged += 1

            insights["anomalies_by_type"] = by_type
            insights["false_positive_rate"] = (
                false_positives / len(self.anomalies) if self.anomalies else 0
            )
            insights["detection_accuracy"] = (
                acknowledged / len(self.anomalies) if self.anomalies else 0
            )

            # Find common combinations of anomalies
            # This would require more complex analysis in a real implementation

            # Trend analysis
            insights["trend_analysis"] = await self._analyze_anomaly_trends()

        except Exception as e:
            log.error(f"Error generating anomaly patterns: {e}")

        return insights

    async def _analyze_anomaly_trends(self) -> Dict[str, Any]:
        """Analyze trends in anomaly occurrences"""
        trends = {
            "hourly_distribution": {},
            "daily_trend": [],
            "weekly_pattern": {},
            "recent_spike": False,
        }

        try:
            # Group anomalies by hour
            hourly_dist = {}
            for anomaly in self.anomalies.values():
                hour = anomaly.detected_at.hour
                hourly_dist[hour] = hourly_dist.get(hour, 0) + 1

            trends["hourly_distribution"] = hourly_dist

            # Check for recent spike (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_count = sum(1 for a in self.anomalies.values() if a.detected_at >= recent_cutoff)
            total_count = len(self.anomalies)

            if total_count > 0:
                recent_ratio = recent_count / total_count
                trends["recent_spike"] = recent_ratio > 0.3  # More than 30% recent

        except Exception as e:
            log.error(f"Error in trend analysis: {e}")

        return trends

    async def predict_anomalies(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Predict potential future anomalies based on metric trends"""
        predictions = {"predictions": [], "risk_score": 0.0, "recommendations": []}

        try:
            # This is a simplified prediction approach
            # In a real implementation, this would use time series forecasting models

            risk_factors = []

            for metric_name, values in metrics.items():
                if len(values) < 5:
                    continue

                # Calculate trend
                recent_avg = np.mean(values[-5:])
                older_avg = np.mean(values[-10:-5]) if len(values) >= 10 else recent_avg

                trend = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0

                # Check for increasing trend that might lead to anomaly
                if trend > 0.1:  # 10% increase
                    anomaly_type = self._infer_anomaly_type(metric_name)
                    risk_factors.append(
                        {
                            "metric": metric_name,
                            "trend": trend,
                            "predicted_anomaly_type": anomaly_type.value,
                            "confidence": min(0.9, trend),
                        }
                    )

            predictions["predictions"] = risk_factors
            predictions["risk_score"] = min(1.0, len(risk_factors) * 0.2)

            # Generate recommendations
            if risk_factors:
                predictions["recommendations"] = [
                    "Monitor trending metrics closely",
                    "Consider proactive system maintenance",
                    "Review recent system changes",
                ]

        except Exception as e:
            log.error(f"Error in anomaly prediction: {e}")

        return predictions


# Global instance
_enhanced_anomaly_detector: Optional[EnhancedAnomalyDetector] = None


async def get_enhanced_anomaly_detector() -> EnhancedAnomalyDetector:
    """Get the global enhanced anomaly detector instance"""
    global _enhanced_anomaly_detector
    if _enhanced_anomaly_detector is None:
        _enhanced_anomaly_detector = EnhancedAnomalyDetector()
        await _enhanced_anomaly_detector.initialize()
    return _enhanced_anomaly_detector


__all__ = [
    "EnhancedAnomalyDetector",
    "EnhancedAnomalyReport",
    "EnhancedAnomalyType",
    "EnhancedMetricHistory",
    "get_enhanced_anomaly_detector",
]
