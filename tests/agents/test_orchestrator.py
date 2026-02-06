"""Tests for Agent Orchestrator"""

import pytest

from agents.base import AgentRole
from agents.orchestrator.advanced import AgentOrchestrator, RoutingDecision
from agents.specialized.code_review import CodeReviewAgent
from agents.specialized.research import ResearchAgent


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator"""

    @pytest.fixture
    async def orchestrator(self):
        """Create an AgentOrchestrator instance"""
        orch = AgentOrchestrator()
        # Register test agents
        orch.register_agent("code_review", CodeReviewAgent(), AgentRole.CODE)
        orch.register_agent("research", ResearchAgent(), AgentRole.RESEARCH)
        return orch

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        orch = await orchestrator
        assert orch is not None
        assert isinstance(orch.agents, dict)

    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator):
        """Test registering agents"""
        orch = await orchestrator
        assert "code_review" in orch.agents
        assert "research" in orch.agents
        assert orch.agent_roles["code_review"] == AgentRole.CODE
        assert orch.agent_roles["research"] == AgentRole.RESEARCH

    @pytest.mark.asyncio
    async def test_unregister_agent(self, orchestrator):
        """Test unregistering agents"""
        orch = await orchestrator
        result = orch.unregister_agent("code_review")
        assert result is True
        assert "code_review" not in orch.agents

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_agent(self, orchestrator):
        """Test unregistering non-existent agent"""
        orch = await orchestrator
        result = orch.unregister_agent("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_route_request_code(self, orchestrator):
        """Test routing code review request"""
        orch = await orchestrator
        routing = await orch.route_request("Review this Python code")
        assert routing["use_agent"] is True
        assert routing["agent_type"] in ["code_review", "research"]
        assert routing["confidence"] > 0

    @pytest.mark.asyncio
    async def test_route_request_research(self, orchestrator):
        """Test routing research request"""
        orch = await orchestrator
        routing = await orch.route_request("Research quantum computing")
        assert routing["use_agent"] is True
        assert routing["agent_type"] in ["code_review", "research"]

    @pytest.mark.asyncio
    async def test_route_request_uncertain(self, orchestrator):
        """Test routing uncertain request"""
        orch = await orchestrator
        routing = await orch.route_request("Hello how are you?")
        # Low confidence should not use agent
        assert routing["confidence"] < 0.7 or not routing["use_agent"]

    @pytest.mark.asyncio
    async def test_get_agent_performance(self, orchestrator):
        """Test getting agent performance metrics"""
        orch = await orchestrator
        metrics = orch.get_agent_performance("code_review")
        assert metrics is not None
        assert "success_rate" in metrics
        assert "tasks_completed" in metrics

    @pytest.mark.asyncio
    async def test_get_all_performance_metrics(self, orchestrator):
        """Test getting all performance metrics"""
        orch = await orchestrator
        all_metrics = orch.get_all_performance_metrics()
        assert "code_review" in all_metrics
        assert "research" in all_metrics

    @pytest.mark.asyncio
    async def test_create_collaboration(self, orchestrator):
        """Test creating agent collaboration"""
        orch = await orchestrator
        collab_id = await orch.create_collaboration(
            primary_agent="code_review",
            supporting_agents=["research"],
            task="Research and review code",
        )
        assert collab_id is not None
        assert collab_id in orch.collaborations


class TestRoutingDecision:
    """Test suite for RoutingDecision"""

    def test_creation(self):
        """Test creating routing decision"""
        decision = RoutingDecision(
            primary_agent="code_review",
            confidence=0.85,
            supporting_agents=["research"],
            strategy="collaborative",
            reason="Code review with research",
        )
        assert decision.primary_agent == "code_review"
        assert decision.confidence == 0.85
        assert decision.strategy == "collaborative"
