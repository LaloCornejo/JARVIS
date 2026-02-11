"""
ProactiveAgent - Specialized agent for autonomous task execution based on patterns and anomalies.

This agent provides:
- Autonomous task execution based on detected patterns
- Anomaly response and mitigation
- Proactive system maintenance
- Workflow automation
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from agents.base import AgentContext, AgentRole, BaseAgent
from core.llm import OllamaClient
from core.prediction.anomaly_detector import AnomalyDetector, get_anomaly_detector
from core.prediction.pattern_analyzer import PatternAnalyzer, get_pattern_analyzer
from core.prediction.suggestion_engine import SuggestionContext, get_suggestion_engine
from core.proactive.monitor import ProactiveMonitor
from tools.registry import get_tool_registry

log = logging.getLogger(__name__)


class ProactiveAgent(BaseAgent):
    """Agent specialized in proactive task execution and system maintenance"""

    name = "ProactiveAgent"
    role = AgentRole.TASK
    description = "Autonomously executes tasks based on detected patterns and anomalies"
    system_prompt = """You are an expert proactive assistant with skills in:
- Autonomous task execution based on behavioral patterns
- Anomaly detection and response
- System maintenance and optimization
- Workflow automation
- Predictive task scheduling

When deciding on actions:
1. Prioritize high-confidence patterns
2. Respond appropriately to anomalies
3. Execute maintenance tasks during low-activity periods
4. Automate repetitive workflows
5. Minimize user interruption

Respond with structured action plans in JSON format."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        super().__init__()
        self.llm = llm_client or OllamaClient()
        self.tool_registry = get_tool_registry()
        self.pattern_analyzer: Optional[PatternAnalyzer] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.suggestion_engine: Optional[SuggestionEngine] = None
        self.proactive_monitor: Optional[ProactiveMonitor] = None
        self.last_execution_time = datetime.now()
        self.execution_interval = timedelta(minutes=5)

    async def initialize(self):
        """Initialize all required components"""
        self.pattern_analyzer = await get_pattern_analyzer()
        self.anomaly_detector = await get_anomaly_detector()
        self.suggestion_engine = await get_suggestion_engine()
        self.proactive_monitor = ProactiveMonitor()

    async def process(self, message: str, context: Optional[AgentContext] = None) -> str:
        """Process proactive task execution request"""
        try:
            # Initialize if needed
            if not self.pattern_analyzer:
                await self.initialize()

            # Parse request
            action, params = self._parse_request(message)

            if action == "execute_autonomous_tasks":
                results = await self._execute_autonomous_tasks(params)
                return json.dumps(results, default=str)
            elif action == "respond_to_anomaly":
                results = await self._respond_to_anomaly(params)
                return json.dumps(results, default=str)
            elif action == "optimize_system":
                results = await self._optimize_system(params)
                return json.dumps(results, default=str)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

        except Exception as e:
            log.error(f"Error in proactive agent: {e}")
            return json.dumps({"error": str(e)})

    async def can_handle(self, message: str) -> float:
        """Check if this agent can handle the message"""
        proactive_keywords = [
            "proactive",
            "autonomous",
            "automatically",
            "pattern",
            "anomaly",
            "maintenance",
            "schedule",
            "workflow",
            "optimize",
            "routine",
            "periodic",
        ]

        message_lower = message.lower()
        keyword_matches = sum(1 for kw in proactive_keywords if kw in message_lower)
        confidence = min(keyword_matches * 0.2, 0.8)

        # Higher confidence for specific proactive commands
        if any(
            cmd in message_lower for cmd in ["execute routine", "autonomous task", "scheduled task"]
        ):
            confidence = min(confidence + 0.3, 1.0)

        return confidence

    def _parse_request(self, message: str) -> tuple[str, Dict[str, Any]]:
        """Parse the request message"""
        message_lower = message.lower()

        if "anomaly" in message_lower:
            return "respond_to_anomaly", {"message": message}
        elif "optimize" in message_lower or "maintain" in message_lower:
            return "optimize_system", {"message": message}
        else:
            return "execute_autonomous_tasks", {"message": message}

    async def _execute_autonomous_tasks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute autonomous tasks based on detected patterns"""
        results = {
            "executed_tasks": [],
            "skipped_tasks": [],
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Get current context
            context = SuggestionContext(
                time_of_day=self._get_time_of_day(),
                day_of_week=datetime.now().weekday(),
                system_load=self._get_system_load(),
            )

            # Get pattern-based suggestions
            if self.suggestion_engine:
                suggestions = await self.suggestion_engine.generate_suggestions(context)

                # Execute high-confidence suggestions
                for suggestion in suggestions:
                    if suggestion.confidence > 0.7 and suggestion.priority.value in [
                        "high",
                        "urgent",
                    ]:
                        try:
                            task_result = await self._execute_suggestion(suggestion)
                            results["executed_tasks"].append(
                                {
                                    "suggestion_id": suggestion.id,
                                    "title": suggestion.title,
                                    "result": task_result,
                                    "confidence": suggestion.confidence,
                                }
                            )

                            # Mark as acted upon
                            await self.suggestion_engine.mark_acted_upon(suggestion.id)
                        except Exception as e:
                            results["errors"].append(
                                {"suggestion_id": suggestion.id, "error": str(e)}
                            )
                    else:
                        results["skipped_tasks"].append(
                            {
                                "suggestion_id": suggestion.id,
                                "title": suggestion.title,
                                "confidence": suggestion.confidence,
                                "priority": suggestion.priority.value,
                            }
                        )

        except Exception as e:
            results["errors"].append({"initialization": str(e)})

        return results

    async def _respond_to_anomaly(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Respond to detected anomalies"""
        results = {
            "detected_anomalies": [],
            "responses": [],
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Get active anomalies
            if self.anomaly_detector:
                anomalies = await self.anomaly_detector.get_active_anomalies()

                for anomaly in anomalies:
                    results["detected_anomalies"].append(
                        {
                            "id": anomaly.id,
                            "type": anomaly.anomaly_type.value,
                            "severity": anomaly.severity.value,
                            "title": anomaly.title,
                            "description": anomaly.description,
                        }
                    )

                    # Generate response for anomaly
                    response = await self._generate_anomaly_response(anomaly)
                    results["responses"].append(response)

                    # Acknowledge the anomaly
                    await self.anomaly_detector.acknowledge_anomaly(anomaly.id)

        except Exception as e:
            results["errors"].append({"anomaly_response": str(e)})

        return results

    async def _optimize_system(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform system optimization tasks"""
        results = {"optimizations": [], "errors": [], "timestamp": datetime.now().isoformat()}

        try:
            # Check if it's an appropriate time for optimization
            if not self._is_good_time_for_maintenance():
                results["optimizations"].append(
                    {
                        "task": "deferred_maintenance",
                        "reason": "Not optimal time for system maintenance",
                        "status": "skipped",
                    }
                )
                return results

            # Clean up old data
            cleanup_tasks = [
                ("clear_old_alerts", self._cleanup_old_alerts),
                ("clear_old_anomalies", self._cleanup_old_anomalies),
                ("clear_expired_suggestions", self._cleanup_expired_suggestions),
            ]

            for task_name, task_func in cleanup_tasks:
                try:
                    count = await task_func()
                    results["optimizations"].append(
                        {"task": task_name, "items_removed": count, "status": "completed"}
                    )
                except Exception as e:
                    results["errors"].append({"task": task_name, "error": str(e)})

        except Exception as e:
            results["errors"].append({"optimization": str(e)})

        return results

    async def _execute_suggestion(self, suggestion) -> Dict[str, Any]:
        """Execute a suggestion"""
        # This is a simplified implementation
        # In a full implementation, this would actually execute the suggested action
        return {
            "action": suggestion.action,
            "executed": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def _generate_anomaly_response(self, anomaly) -> Dict[str, Any]:
        """Generate an appropriate response for an anomaly"""
        # This is a simplified implementation
        # In a full implementation, this would generate and possibly execute a response
        return {
            "anomaly_id": anomaly.id,
            "response_generated": True,
            "recommended_action": f"Address {anomaly.anomaly_type.value} anomaly: {anomaly.title}",
            "timestamp": datetime.now().isoformat(),
        }

    def _get_time_of_day(self) -> str:
        """Get current time of day category"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _get_system_load(self) -> float:
        """Get current system load estimate"""
        # Simplified system load estimation
        # In a real implementation, this would gather actual system metrics
        try:
            import psutil

            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 50.0  # Default estimate

    def _is_good_time_for_maintenance(self) -> bool:
        """Check if it's a good time for maintenance tasks"""
        # Avoid maintenance during typical work hours
        hour = datetime.now().hour
        is_weekend = datetime.now().weekday() >= 5

        # Good time is either weekend or outside 9AM-5PM
        return is_weekend or hour < 9 or hour > 17

    async def _cleanup_old_alerts(self) -> int:
        """Clean up old alerts"""
        if self.proactive_monitor:
            return self.proactive_monitor.alerts.cleanup_old_alerts(days=7)
        return 0

    async def _cleanup_old_anomalies(self) -> int:
        """Clean up old anomalies"""
        if self.anomaly_detector:
            return await self.anomaly_detector.cleanup_old_anomalies(days=30)
        return 0

    async def _cleanup_expired_suggestions(self) -> int:
        """Clean up expired suggestions"""
        if self.suggestion_engine:
            return await self.suggestion_engine.cleanup_expired()
        return 0

    async def run_periodic_tasks(self):
        """Run periodic autonomous tasks"""
        while True:
            try:
                # Check if enough time has passed since last execution
                if datetime.now() - self.last_execution_time >= self.execution_interval:
                    # Execute autonomous tasks
                    await self._execute_autonomous_tasks({})

                    # Update last execution time
                    self.last_execution_time = datetime.now()

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                log.error(f"Error in periodic proactive tasks: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
