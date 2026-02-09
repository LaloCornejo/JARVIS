"""
Advanced Agent Orchestrator for JARVIS.

Provides multi-agent collaboration, intelligent routing, and agent communication.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base import AgentContext, AgentMessage, AgentRole, BaseAgent

log = logging.getLogger(__name__)


@dataclass
class AgentCollaboration:
    """Represents a collaboration between agents"""

    id: str
    primary_agent: str
    supporting_agents: List[str]
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[AgentMessage] = field(default_factory=list)
    status: str = "active"  # active, completed, failed
    result: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class RoutingDecision:
    """Decision for routing a task to agents"""

    primary_agent: str
    confidence: float
    supporting_agents: List[str] = field(default_factory=list)
    strategy: str = "single"  # single, collaborative, sequential
    reason: str = ""


class AgentOrchestrator:
    """
    Orchestrates multiple agents for complex tasks.

    Features:
    - Intelligent task routing based on agent confidence
    - Multi-agent collaboration with result synthesis
    - Agent communication protocol
    - Performance tracking and metrics
    - Fallback strategies for failed tasks
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_roles: Dict[str, AgentRole] = {}
        self.collaborations: Dict[str, AgentCollaboration] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

    def register_agent(self, name: str, agent: BaseAgent, role: AgentRole) -> None:
        """Register an agent with the orchestrator"""
        self.agents[name] = agent
        self.agent_roles[name] = role
        self.performance_metrics[name] = {
            "success_rate": 1.0,
            "avg_response_time": 0.0,
            "tasks_completed": 0,
        }
        log.info(f"Registered agent: {name} ({role.value})")

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent"""
        if name in self.agents:
            del self.agents[name]
            del self.agent_roles[name]
            del self.performance_metrics[name]
            log.info(f"Unregistered agent: {name}")
            return True
        return False

    async def route_task(
        self, message: str, context: Optional[AgentContext] = None
    ) -> RoutingDecision:
        """
        Route a task to the most appropriate agent(s).

        Uses confidence scoring from each agent to determine the best match.
        """
        if not self.agents:
            return RoutingDecision(primary_agent="", confidence=0.0, reason="No agents available")

        # Get confidence scores from all agents
        agent_scores: List[tuple[str, float]] = []

        for name, agent in self.agents.items():
            try:
                confidence = await agent.can_handle(message)
                agent_scores.append((name, confidence))
            except Exception as e:
                log.warning(f"Error getting confidence from {name}: {e}")
                agent_scores.append((name, 0.0))

        # Sort by confidence
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        # Determine routing strategy
        primary_agent, primary_confidence = agent_scores[0]

        if primary_confidence >= 0.8:
            # High confidence - single agent
            decision = RoutingDecision(
                primary_agent=primary_agent,
                confidence=primary_confidence,
                strategy="single",
                reason=f"{primary_agent} has high confidence ({primary_confidence:.2f})",
            )
        elif primary_confidence >= 0.5:
            # Medium confidence - collaborative approach
            supporting = [name for name, conf in agent_scores[1:3] if conf > 0.3]
            decision = RoutingDecision(
                primary_agent=primary_agent,
                confidence=primary_confidence,
                supporting_agents=supporting,
                strategy="collaborative" if supporting else "single",
                reason=(
                    f"Primary: {primary_agent} ({primary_confidence:.2f}), Supporting: {supporting}"
                ),
            )
        else:
            # Low confidence - try multiple agents
            top_agents = [name for name, conf in agent_scores[:3] if conf > 0.2]
            decision = RoutingDecision(
                primary_agent=primary_agent
                if primary_agent
                else (top_agents[0] if top_agents else ""),
                confidence=primary_confidence,
                supporting_agents=top_agents[1:] if len(top_agents) > 1 else [],
                strategy="sequential" if len(top_agents) > 1 else "single",
                reason="Low confidence - trying multiple approaches",
            )

        # Record routing decision
        self.routing_history.append(
            {
                "message": message[:100],
                "decision": decision,
                "timestamp": datetime.now(),
                "all_scores": agent_scores,
            }
        )

        return decision

    async def execute(
        self, message: str, context: Optional[AgentContext] = None, collaborative: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a task using the appropriate agent(s).

        Args:
            message: The task message
            context: Optional agent context
            collaborative: Force collaborative mode

        Returns:
            Dictionary with results and metadata
        """
        # Route the task
        decision = await self.route_task(message, context)

        if decision.confidence < 0.2:
            return {
                "success": False,
                "error": "No agent confident enough to handle this task",
                "confidence": decision.confidence,
                "agents_tried": list(self.agents.keys()),
            }

        start_time = asyncio.get_event_loop().time()

        try:
            if collaborative or decision.strategy == "collaborative":
                result = await self._execute_collaborative(message, decision, context)
            elif decision.strategy == "sequential":
                result = await self._execute_sequential(message, decision, context)
            else:
                result = await self._execute_single(message, decision, context)

            # Update performance metrics
            duration = asyncio.get_event_loop().time() - start_time
            await self._update_metrics(decision.primary_agent, True, duration)

            return result

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self._update_metrics(decision.primary_agent, False, duration)
            log.error(f"Error executing task: {e}")

            return {
                "success": False,
                "error": str(e),
                "primary_agent": decision.primary_agent,
                "strategy": decision.strategy,
            }

    async def _execute_single(
        self, message: str, decision: RoutingDecision, context: Optional[AgentContext]
    ) -> Dict[str, Any]:
        """Execute with a single agent"""
        agent = self.agents.get(decision.primary_agent)

        if not agent:
            return {"success": False, "error": f"Agent {decision.primary_agent} not found"}

        result = await agent.process(message, context)

        return {
            "success": True,
            "result": result,
            "agent": decision.primary_agent,
            "strategy": "single",
            "confidence": decision.confidence,
        }

    async def _execute_collaborative(
        self, message: str, decision: RoutingDecision, context: Optional[AgentContext]
    ) -> Dict[str, Any]:
        """Execute with multiple agents collaborating"""
        collaboration_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        collaboration = AgentCollaboration(
            id=collaboration_id,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            task=message,
            context={"original_message": message},
        )

        self.collaborations[collaboration_id] = collaboration

        try:
            # Primary agent processes first
            primary_agent = self.agents[decision.primary_agent]
            primary_result = await primary_agent.process(message, context)

            collaboration.messages.append(
                AgentMessage(
                    role="assistant",
                    content=primary_result,
                    agent_id=decision.primary_agent,
                )
            )

            # Supporting agents provide additional insights
            supporting_results = []

            if decision.supporting_agents:
                tasks = []
                for agent_name in decision.supporting_agents:
                    agent = self.agents.get(agent_name)
                    if agent:
                        contextualized_message = f"""Original task: {message}

Primary agent ({decision.primary_agent}) provided:
{primary_result[:500]}

Please provide additional insights or perspectives on this task."""

                        tasks.append(
                            self._execute_agent_safe(
                                agent_name, agent, contextualized_message, context
                            )
                        )

                supporting_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Synthesize results
            synthesis = await self._synthesize_results(
                message, decision.primary_agent, primary_result, supporting_results
            )

            collaboration.result = synthesis
            collaboration.status = "completed"
            collaboration.completed_at = datetime.now()

            return {
                "success": True,
                "result": synthesis,
                "primary_result": primary_result,
                "supporting_results": supporting_results,
                "agents": [decision.primary_agent] + decision.supporting_agents,
                "strategy": "collaborative",
                "collaboration_id": collaboration_id,
            }

        except Exception as e:
            collaboration.status = "failed"
            log.error(f"Collaborative execution failed: {e}")
            raise

    async def _execute_sequential(
        self, message: str, decision: RoutingDecision, context: Optional[AgentContext]
    ) -> Dict[str, Any]:
        """Execute with agents in sequence"""
        results = []
        current_message = message

        agents_to_try = [decision.primary_agent] + decision.supporting_agents

        for agent_name in agents_to_try:
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            try:
                result = await agent.process(current_message, context)
                results.append({"agent": agent_name, "result": result, "success": True})

                # Use result to refine next query
                current_message = f"{message}\n\nPrevious analysis ({agent_name}): {result[:300]}"

            except Exception as e:
                results.append({"agent": agent_name, "error": str(e), "success": False})

        successful = [r for r in results if r.get("success")]

        if not successful:
            return {"success": False, "error": "All agents failed", "attempts": results}

        return {
            "success": True,
            "result": successful[0]["result"],
            "all_results": results,
            "strategy": "sequential",
            "agents_tried": agents_to_try,
        }

    async def _execute_agent_safe(
        self, name: str, agent: BaseAgent, message: str, context: Optional[AgentContext]
    ) -> Dict[str, Any]:
        """Execute an agent with error handling"""
        try:
            result = await asyncio.wait_for(agent.process(message, context), timeout=30.0)
            return {"agent": name, "result": result, "success": True}
        except asyncio.TimeoutError:
            return {"agent": name, "error": "Timeout", "success": False}
        except Exception as e:
            return {"agent": name, "error": str(e), "success": False}

    async def _synthesize_results(
        self,
        original_message: str,
        primary_agent: str,
        primary_result: str,
        supporting_results: List[Dict[str, Any]],
    ) -> str:
        """Synthesize results from multiple agents"""
        valid_supporting = [
            r for r in supporting_results if isinstance(r, dict) and r.get("success")
        ]

        if not valid_supporting:
            return primary_result

        synthesis_parts = [f"## Primary Analysis ({primary_agent})\n\n{primary_result}"]

        if valid_supporting:
            synthesis_parts.append("\n## Additional Perspectives\n")

            for result in valid_supporting:
                agent_name = result.get("agent", "Unknown")
                content = result.get("result", "")[:500]
                synthesis_parts.append(f"\n### {agent_name}\n{content}")

        return "\n".join(synthesis_parts)

    async def _update_metrics(self, agent_name: str, success: bool, duration: float) -> None:
        """Update performance metrics for an agent"""
        async with self._lock:
            if agent_name not in self.performance_metrics:
                self.performance_metrics[agent_name] = {
                    "success_rate": 1.0,
                    "avg_response_time": 0.0,
                    "tasks_completed": 0,
                }

            metrics = self.performance_metrics[agent_name]
            tasks = metrics["tasks_completed"]

            # Update success rate with exponential moving average
            success_val = 1.0 if success else 0.0
            metrics["success_rate"] = (metrics["success_rate"] * tasks + success_val) / (tasks + 1)

            # Update average response time
            metrics["avg_response_time"] = (metrics["avg_response_time"] * tasks + duration) / (
                tasks + 1
            )

            metrics["tasks_completed"] += 1

    def get_agent_performance(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for agents"""
        if agent_name:
            return self.performance_metrics.get(agent_name, {})

        return {
            "agents": self.performance_metrics,
            "total_tasks": sum(m["tasks_completed"] for m in self.performance_metrics.values()),
            "avg_success_rate": (
                sum(m["success_rate"] for m in self.performance_metrics.values())
                / len(self.performance_metrics)
                if self.performance_metrics
                else 0
            ),
        }

    def get_routing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent routing decisions"""
        return self.routing_history[-limit:]

    def get_active_collaborations(self) -> List[AgentCollaboration]:
        """Get active collaborations"""
        return [c for c in self.collaborations.values() if c.status == "active"]

    async def broadcast_message(
        self, message: str, source_agent: str, target_roles: Optional[List[AgentRole]] = None
    ) -> Dict[str, Any]:
        """Broadcast a message to agents with specific roles"""
        responses = {}

        for name, agent in self.agents.items():
            if name == source_agent:
                continue

            if target_roles and self.agent_roles.get(name) not in target_roles:
                continue

            try:
                context = AgentContext(
                    messages=[
                        AgentMessage(
                            role="system", content=f"Broadcast from {source_agent}: {message}"
                        )
                    ]
                )

                response = await agent.process(f"Respond to broadcast: {message}", context)
                responses[name] = response

            except Exception as e:
                log.warning(f"Agent {name} failed to respond to broadcast: {e}")
                responses[name] = f"Error: {e}"

        return {"source": source_agent, "message": message, "responses": responses}

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "name": name,
                "role": role.value,
                "performance": self.performance_metrics.get(name, {}),
            }
            for name, role in self.agent_roles.items()
        ]


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the global orchestrator (for testing)"""
    global _orchestrator
    _orchestrator = None


__all__ = [
    "AgentOrchestrator",
    "AgentCollaboration",
    "RoutingDecision",
    "get_orchestrator",
    "reset_orchestrator",
]
