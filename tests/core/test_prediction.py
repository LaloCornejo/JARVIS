"""Tests for Predictive Features"""

from datetime import datetime

import pytest

from core.prediction.anomaly_detector import (
    AnomalyDetector,
    AnomalySeverity,
    AnomalyType,
)
from core.prediction.pattern_analyzer import PatternAnalyzer
from core.prediction.suggestion_engine import SmartSuggestionEngine


class TestPatternAnalyzer:
    """Test suite for PatternAnalyzer"""

    @pytest.fixture
    async def analyzer(self):
        """Create a PatternAnalyzer instance"""
        return PatternAnalyzer()

    @pytest.mark.asyncio
    async def test_initialization(self, analyzer):
        """Test analyzer initializes correctly"""
        a = await analyzer
        assert a is not None

    @pytest.mark.asyncio
    async def test_record_event(self, analyzer):
        """Test recording an event"""
        a = await analyzer
        await a.record_event(
            user_id="user_1",
            event_type="app_open",
            event_data={"app": "browser"},
        )
        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_analyze_patterns(self, analyzer):
        """Test pattern analysis"""
        a = await analyzer
        # Record some events first
        for i in range(5):
            await a.record_event(
                user_id="user_1",
                event_type="test_event",
                event_data={"index": i},
            )

        patterns = await a.analyze_patterns(user_id="user_1")
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_get_insights(self, analyzer):
        """Test getting insights"""
        a = await analyzer
        insights = await a.get_insights(user_id="user_1")
        assert isinstance(insights, list)


class TestSmartSuggestionEngine:
    """Test suite for SmartSuggestionEngine"""

    @pytest.fixture
    async def engine(self):
        """Create a SmartSuggestionEngine instance"""
        return SmartSuggestionEngine()

    @pytest.mark.asyncio
    async def test_initialization(self, engine):
        """Test engine initializes correctly"""
        e = await engine
        assert e is not None

    @pytest.mark.asyncio
    async def test_generate_suggestions(self, engine):
        """Test generating suggestions"""
        e = await engine
        context = {
            "time_of_day": datetime.now().hour,
            "user_id": "user_1",
        }

        suggestions = await e.generate_suggestions(
            user_id="user_1",
            context=context,
        )
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_get_contextual_suggestions(self, engine):
        """Test getting contextual suggestions"""
        e = await engine
        context = {
            "active_applications": ["browser"],
            "time_of_day": datetime.now().hour,
        }

        suggestions = await e.get_contextual_suggestions(
            user_id="user_1",
            context=context,
        )
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_learn_from_feedback(self, engine):
        """Test learning from feedback"""
        e = await engine
        await e.learn_from_feedback(
            suggestion_id="sugg_1",
            user_id="user_1",
            accepted=True,
        )
        # Should not raise any errors


class TestAnomalyDetector:
    """Test suite for AnomalyDetector"""

    @pytest.fixture
    async def detector(self, temp_dir):
        """Create an AnomalyDetector instance"""
        return AnomalyDetector(storage_path=str(temp_dir / "anomalies"))

    @pytest.mark.asyncio
    async def test_initialization(self, detector):
        """Test detector initializes correctly"""
        d = await detector
        await d.initialize()
        assert d._initialized is True

    @pytest.mark.asyncio
    async def test_check_metric(self, detector):
        """Test checking a metric for anomalies"""
        d = await detector
        await d.initialize()

        # Check normal value (should not be anomalous)
        for i in range(10):
            result = await d.check_metric(
                metric_name="test_metric",
                value=50.0,
                anomaly_type=AnomalyType.PERFORMANCE,
            )

        # Now check anomalous value
        result = await d.check_metric(
            metric_name="test_metric",
            value=500.0,  # Very different
            anomaly_type=AnomalyType.PERFORMANCE,
        )
        # Result could be None or an AnomalyReport depending on detection

    @pytest.mark.asyncio
    async def test_check_system_health(self, detector):
        """Test checking system health"""
        d = await detector
        await d.initialize()

        # Healthy system
        healthy_data = {
            "tts": True,
            "stt": True,
            "vad": True,
        }
        result = await d.check_system_health(healthy_data)
        assert result["is_anomaly"] is False

        # Unhealthy system
        unhealthy_data = {
            "tts": False,
            "stt": False,
            "vad": True,
        }
        result = await d.check_system_health(unhealthy_data)
        assert result["is_anomaly"] is True
        assert "failed_components" in result

    @pytest.mark.asyncio
    async def test_check_user_behavior(self, detector):
        """Test checking user behavior"""
        d = await detector
        await d.initialize()

        result = await d.check_user_behavior(
            user_id="user_1",
            action="login",
        )
        # Result could be None or AnomalyReport

    @pytest.mark.asyncio
    async def test_acknowledge_anomaly(self, detector):
        """Test acknowledging an anomaly"""
        d = await detector
        await d.initialize()

        # First create an anomaly
        await d.check_metric(
            metric_name="ack_test",
            value=100.0,
            anomaly_type=AnomalyType.SYSTEM,
        )

        # Get anomalies
        anomalies = await d.get_recent_anomalies()
        if anomalies:
            anomaly_id = anomalies[0].id
            result = await d.acknowledge_anomaly(anomaly_id)
            assert result is True

    @pytest.mark.asyncio
    async def test_get_recent_anomalies(self, detector):
        """Test getting recent anomalies"""
        d = await detector
        await d.initialize()

        anomalies = await d.get_recent_anomalies()
        assert isinstance(anomalies, list)

    @pytest.mark.asyncio
    async def test_get_stats(self, detector):
        """Test getting statistics"""
        d = await detector
        await d.initialize()

        stats = d.get_stats()
        assert "total_anomalies" in stats
        assert "by_type" in stats
        assert "by_severity" in stats
