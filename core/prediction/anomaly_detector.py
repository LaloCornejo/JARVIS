"""
Anomaly Detection System for JARVIS.

Detects unusual behavior, system anomalies, and potential issues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""

    BEHAVIORAL = "behavioral"  # Unusual user behavior
    SYSTEM = "system"  # System performance issues
    SECURITY = "security"  # Potential security issues
    TEMPORAL = "temporal"  # Time-based anomalies
    FREQUENCY = "frequency"  # Unusual frequency of actions
    PATTERN_BREAK = "pattern_break"  # Breaking established patterns
    PERFORMANCE = "performance"  # Performance degradation
    RESOURCE = "resource"  # Resource usage anomalies


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""

    INFO = "info"  # Notable but not concerning
    LOW = "low"  # Minor anomaly
    MEDIUM = "medium"  # Moderate concern
    HIGH = "high"  # Significant concern
    CRITICAL = "critical"  # Immediate attention required


@dataclass
class AnomalyReport:
    """A detected anomaly report"""

    id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    title: str
    description: str
    detected_at: datetime
    metric_name: str
    metric_value: float
    expected_range: Tuple[float, float]
    deviation_score: float
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    false_positive: bool = False
    related_anomalies: List[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        # Anomalies are considered resolved after some time if not recurring
        age = datetime.now() - self.detected_at
        return age > timedelta(hours=24) and not self.acknowledged


class MetricHistory:
    """Tracks history of a metric for anomaly detection"""

    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.values: List[Tuple[datetime, float]] = []
        self.window_size = window_size
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new value"""
        if timestamp is None:
            timestamp = datetime.now()

        self.values.append((timestamp, value))

        # Keep only recent values
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size :]

        # Update baseline
        if len(self.values) >= 10:
            recent_values = [v for _, v in self.values[-30:]]
            self.baseline_mean = statistics.mean(recent_values)
            if len(recent_values) > 1:
                self.baseline_std = statistics.stdev(recent_values)

    def is_anomaly(self, value: float, threshold_std: float = 3.0) -> Tuple[bool, float]:
        """Check if value is anomalous"""
        if self.baseline_mean is None or self.baseline_std is None:
            return False, 0.0

        if self.baseline_std == 0:
            return value != self.baseline_mean, 0.0

        deviation = abs(value - self.baseline_mean) / self.baseline_std
        is_anomaly = deviation > threshold_std

        return is_anomaly, deviation

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about this metric"""
        if not self.values:
            return {}

        recent = [v for _, v in self.values[-30:]]
        return {
            "mean": statistics.mean(recent),
            "median": statistics.median(recent),
            "min": min(recent),
            "max": max(recent),
            "std": statistics.stdev(recent) if len(recent) > 1 else 0,
            "count": len(self.values),
        }


class AnomalyDetector:
    """
    Detects anomalies in user behavior and system state.

    Features:
    - Statistical anomaly detection
    - Pattern break detection
    - Temporal anomaly detection
    - Multi-metric correlation
    - Adaptive thresholds
    """

    def __init__(self, storage_path: str = "data/anomalies"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.anomalies: Dict[str, AnomalyReport] = {}
        self.metric_histories: Dict[str, MetricHistory] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

        # Detection thresholds
        self.thresholds = {
            AnomalyType.BEHAVIORAL: 2.5,  # Standard deviations
            AnomalyType.SYSTEM: 3.0,
            AnomalyType.SECURITY: 2.0,
            AnomalyType.TEMPORAL: 2.0,
            AnomalyType.FREQUENCY: 2.5,
            AnomalyType.PATTERN_BREAK: 2.0,
            AnomalyType.PERFORMANCE: 2.5,
            AnomalyType.RESOURCE: 3.0,
        }

        # Severity thresholds based on deviation
        self.severity_thresholds = {
            2.0: AnomalySeverity.LOW,
            3.0: AnomalySeverity.MEDIUM,
            4.0: AnomalySeverity.HIGH,
            5.0: AnomalySeverity.CRITICAL,
        }

    async def initialize(self):
        """Initialize the anomaly detector"""
        if self._initialized:
            return

        await self._load_data()
        self._initialized = True
        log.info(f"Anomaly detector initialized with {len(self.anomalies)} historical anomalies")

    async def _load_data(self):
        """Load anomaly data"""
        data_file = self.storage_path / "anomalies.json"

        if data_file.exists():
            try:
                with open(data_file, "r") as f:
                    data = json.load(f)

                for anomaly_data in data.get("anomalies", []):
                    report = AnomalyReport(
                        id=anomaly_data["id"],
                        anomaly_type=AnomalyType(anomaly_data["anomaly_type"]),
                        severity=AnomalySeverity(anomaly_data["severity"]),
                        title=anomaly_data["title"],
                        description=anomaly_data["description"],
                        detected_at=datetime.fromisoformat(anomaly_data["detected_at"]),
                        metric_name=anomaly_data["metric_name"],
                        metric_value=anomaly_data["metric_value"],
                        expected_range=tuple(anomaly_data["expected_range"]),
                        deviation_score=anomaly_data["deviation_score"],
                        context=anomaly_data.get("context", {}),
                        acknowledged=anomaly_data.get("acknowledged", False),
                        false_positive=anomaly_data.get("false_positive", False),
                        related_anomalies=anomaly_data.get("related_anomalies", []),
                    )
                    self.anomalies[report.id] = report

            except Exception as e:
                log.error(f"Error loading anomalies: {e}")

    async def _save_data(self):
        """Save anomaly data"""
        data_file = self.storage_path / "anomalies.json"

        with open(data_file, "w") as f:
            json.dump(
                {
                    "anomalies": [
                        {
                            "id": a.id,
                            "anomaly_type": a.anomaly_type.value,
                            "severity": a.severity.value,
                            "title": a.title,
                            "description": a.description,
                            "detected_at": a.detected_at.isoformat(),
                            "metric_name": a.metric_name,
                            "metric_value": a.metric_value,
                            "expected_range": list(a.expected_range),
                            "deviation_score": a.deviation_score,
                            "context": a.context,
                            "acknowledged": a.acknowledged,
                            "false_positive": a.false_positive,
                            "related_anomalies": a.related_anomalies,
                        }
                        for a in self.anomalies.values()
                    ]
                },
                f,
                indent=2,
            )

    async def check_metric(
        self,
        metric_name: str,
        value: float,
        anomaly_type: AnomalyType,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnomalyReport]:
        """Check a metric for anomalies"""
        async with self._lock:
            # Get or create metric history
            if metric_name not in self.metric_histories:
                self.metric_histories[metric_name] = MetricHistory(metric_name)

            history = self.metric_histories[metric_name]

            # Add value
            history.add(value)

            # Check for anomaly
            threshold = self.thresholds.get(anomaly_type, 3.0)
            is_anomaly, deviation = history.is_anomaly(value, threshold)

            if not is_anomaly:
                return None

            # Determine severity
            severity = self._deviation_to_severity(deviation)

            # Create anomaly report
            report = await self._create_anomaly_report(
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
            )

            self.anomalies[report.id] = report
            await self._save_data()

            log.warning(f"Anomaly detected: {report.title} (severity: {severity.value})")
            return report

    def _deviation_to_severity(self, deviation: float) -> AnomalySeverity:
        """Convert deviation score to severity"""
        for threshold, severity in sorted(self.severity_thresholds.items(), reverse=True):
            if deviation >= threshold:
                return severity
        return AnomalySeverity.INFO

    async def _create_anomaly_report(
        self,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        metric_name: str,
        metric_value: float,
        expected_range: Tuple[float, float],
        deviation_score: float,
        context: Dict[str, Any],
    ) -> AnomalyReport:
        """Create an anomaly report"""
        anomaly_id = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Generate title and description based on type
        title, description = self._generate_description(
            anomaly_type, metric_name, metric_value, expected_range, deviation_score
        )

        # Find related anomalies
        related = await self._find_related_anomalies(anomaly_type, metric_name)

        return AnomalyReport(
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
        )

    def _generate_description(
        self,
        anomaly_type: AnomalyType,
        metric_name: str,
        metric_value: float,
        expected_range: Tuple[float, float],
        deviation: float,
    ) -> Tuple[str, str]:
        """Generate human-readable description"""
        higher = metric_value > expected_range[1]
        direction = "higher" if higher else "lower"

        titles = {
            AnomalyType.BEHAVIORAL: f"Unusual behavior detected in {metric_name}",
            AnomalyType.SYSTEM: f"System metric '{metric_name}' is {direction} than normal",
            AnomalyType.SECURITY: f"Potential security anomaly: {metric_name}",
            AnomalyType.TEMPORAL: f"Timing anomaly in {metric_name}",
            AnomalyType.FREQUENCY: f"Unusual frequency of {metric_name}",
            AnomalyType.PATTERN_BREAK: f"Pattern break in {metric_name}",
            AnomalyType.PERFORMANCE: f"Performance issue: {metric_name}",
            AnomalyType.RESOURCE: f"Resource usage anomaly: {metric_name}",
        }

        title = titles.get(anomaly_type, f"Anomaly detected in {metric_name}")

        description = (
            f"The metric '{metric_name}' is {direction} than expected. "
            f"Current value: {metric_value:.2f}, "
            f"Expected range: {expected_range[0]:.2f} - {expected_range[1]:.2f}, "
            f"Deviation: {deviation:.2f} standard deviations."
        )

        return title, description

    async def _find_related_anomalies(
        self,
        anomaly_type: AnomalyType,
        metric_name: str,
    ) -> List[str]:
        """Find related recent anomalies"""
        related = []
        cutoff = datetime.now() - timedelta(hours=1)

        for anomaly_id, anomaly in self.anomalies.items():
            if anomaly.detected_at < cutoff:
                continue

            if anomaly.anomaly_type == anomaly_type:
                related.append(anomaly_id)
            elif anomaly.metric_name == metric_name:
                related.append(anomaly_id)

            if len(related) >= 5:
                break

        return related

    async def check_user_behavior(
        self,
        user_id: str,
        action: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[AnomalyReport]:
        """Check user behavior for anomalies"""
        if timestamp is None:
            timestamp = datetime.now()

        metric_name = f"user_{user_id}_{action}"

        # Check temporal anomaly (actions at unusual times)
        hour = timestamp.hour
        metric_key = f"user_{user_id}_{action}_hour"

        return await self.check_metric(
            metric_name=metric_key,
            value=float(hour),
            anomaly_type=AnomalyType.TEMPORAL,
            context={"user_id": user_id, "action": action, "hour": hour},
        )

    async def check_system_performance(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_usage: float,
    ) -> List[AnomalyReport]:
        """Check system metrics for anomalies"""
        reports = []

        # Check CPU
        report = await self.check_metric(
            metric_name="system_cpu_percent",
            value=cpu_percent,
            anomaly_type=AnomalyType.PERFORMANCE,
        )
        if report:
            reports.append(report)

        # Check memory
        report = await self.check_metric(
            metric_name="system_memory_percent",
            value=memory_percent,
            anomaly_type=AnomalyType.RESOURCE,
        )
        if report:
            reports.append(report)

        # Check disk
        report = await self.check_metric(
            metric_name="system_disk_usage",
            value=disk_usage,
            anomaly_type=AnomalyType.RESOURCE,
        )
        if report:
            reports.append(report)

        return reports

    async def check_response_time(
        self,
        operation: str,
        response_time_ms: float,
    ) -> Optional[AnomalyReport]:
        """Check response times for anomalies"""
        return await self.check_metric(
            metric_name=f"response_time_{operation}",
            value=response_time_ms,
            anomaly_type=AnomalyType.PERFORMANCE,
            context={"operation": operation},
        )

    async def check_error_rate(
        self,
        component: str,
        error_count: int,
        window_minutes: int = 5,
    ) -> Optional[AnomalyReport]:
        """Check error rates for anomalies"""
        error_rate = error_count / window_minutes  # Errors per minute

        return await self.check_metric(
            metric_name=f"error_rate_{component}",
            value=error_rate,
            anomaly_type=AnomalyType.SYSTEM,
            context={"component": component, "error_count": error_count},
        )

    async def check_system_health(self, health_data: Dict[str, bool]) -> Dict[str, Any]:
        """Check system health metrics for anomalies

        Args:
            health_data: Dictionary of component_name -> bool (online/offline)

        Returns:
            Dict with is_anomaly, severity, and details
        """
        # Count failed services
        failed = [k for k, v in health_data.items() if not v]
        total = len(health_data)

        if not failed:
            return {"is_anomaly": False, "status": "healthy"}

        # Calculate severity based on failure ratio
        failure_ratio = len(failed) / total if total > 0 else 0

        if failure_ratio >= 0.5:
            severity = AnomalySeverity.HIGH
        elif failure_ratio >= 0.25:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW

        return {
            "is_anomaly": True,
            "severity": severity.value,
            "failed_components": failed,
            "healthy_components": [k for k, v in health_data.items() if v],
            "failure_ratio": failure_ratio,
            "message": f"System health check failed for: {', '.join(failed)}",
        }

    async def acknowledge_anomaly(self, anomaly_id: str) -> bool:
        """Acknowledge an anomaly"""
        async with self._lock:
            anomaly = self.anomalies.get(anomaly_id)
            if not anomaly:
                return False

            anomaly.acknowledged = True
            await self._save_data()
            return True

    async def mark_false_positive(self, anomaly_id: str) -> bool:
        """Mark an anomaly as false positive"""
        async with self._lock:
            anomaly = self.anomalies.get(anomaly_id)
            if not anomaly:
                return False

            anomaly.false_positive = True
            anomaly.acknowledged = True

            # Adjust threshold for this metric
            metric_history = self.metric_histories.get(anomaly.metric_name)
            if metric_history:
                # Increase threshold slightly to reduce false positives
                current_threshold = self.thresholds.get(anomaly.anomaly_type, 3.0)
                self.thresholds[anomaly.anomaly_type] = min(current_threshold * 1.1, 5.0)

            await self._save_data()
            return True

    async def get_active_anomalies(
        self,
        min_severity: Optional[AnomalySeverity] = None,
        anomaly_type: Optional[AnomalyType] = None,
    ) -> List[AnomalyReport]:
        """Get active (unacknowledged) anomalies"""
        active = []

        for anomaly in self.anomalies.values():
            if anomaly.acknowledged:
                continue

            if anomaly.false_positive:
                continue

            if min_severity:
                severity_order = [
                    AnomalySeverity.INFO,
                    AnomalySeverity.LOW,
                    AnomalySeverity.MEDIUM,
                    AnomalySeverity.HIGH,
                    AnomalySeverity.CRITICAL,
                ]
                if severity_order.index(anomaly.severity) < severity_order.index(min_severity):
                    continue

            if anomaly_type and anomaly.anomaly_type != anomaly_type:
                continue

            active.append(anomaly)

        # Sort by severity and time
        severity_order = {
            AnomalySeverity.CRITICAL: 0,
            AnomalySeverity.HIGH: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 3,
            AnomalySeverity.INFO: 4,
        }

        active.sort(key=lambda a: (severity_order.get(a.severity, 5), a.detected_at), reverse=True)

        return active

    async def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent anomalies"""
        cutoff = datetime.now() - timedelta(hours=hours)

        recent = [a for a in self.anomalies.values() if a.detected_at >= cutoff]

        by_type = {}
        by_severity = {}

        for anomaly in recent:
            t = anomaly.anomaly_type.value
            by_type[t] = by_type.get(t, 0) + 1

            s = anomaly.severity.value
            by_severity[s] = by_severity.get(s, 0) + 1

        return {
            "total": len(recent),
            "by_type": by_type,
            "by_severity": by_severity,
            "acknowledged": sum(1 for a in recent if a.acknowledged),
            "false_positives": sum(1 for a in recent if a.false_positive),
            "active": sum(1 for a in recent if not a.acknowledged and not a.false_positive),
        }

    async def cleanup_old_anomalies(self, days: int = 30) -> int:
        """Remove old anomalies"""
        async with self._lock:
            cutoff = datetime.now() - timedelta(days=days)
            to_remove = []

            for aid, anomaly in self.anomalies.items():
                if anomaly.detected_at < cutoff:
                    to_remove.append(aid)

            for aid in to_remove:
                del self.anomalies[aid]

            await self._save_data()
            return len(to_remove)

    async def get_stats(self) -> Dict[str, Any]:
        """Get anomaly detector statistics"""
        return {
            "total_anomalies": len(self.anomalies),
            "active_anomalies": sum(1 for a in self.anomalies.values() if not a.acknowledged),
            "monitored_metrics": len(self.metric_histories),
            "by_type": {
                t.value: sum(1 for a in self.anomalies.values() if a.anomaly_type == t)
                for t in AnomalyType
            },
            "by_severity": {
                s.value: sum(1 for a in self.anomalies.values() if a.severity == s)
                for s in AnomalySeverity
            },
        }


# Global instance
_anomaly_detector: Optional[AnomalyDetector] = None


async def get_anomaly_detector() -> AnomalyDetector:
    """Get the global anomaly detector instance"""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
        await _anomaly_detector.initialize()
    return _anomaly_detector


__all__ = [
    "AnomalyDetector",
    "AnomalyReport",
    "AnomalyType",
    "AnomalySeverity",
    "get_anomaly_detector",
]
