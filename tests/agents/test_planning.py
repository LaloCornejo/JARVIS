"""Tests for Planning Agent"""

import pytest

from agents.specialized.planning import (
    Milestone,
    PlanAdjustment,
    PlanningAgent,
    ProjectPlan,
    Task,
    TaskPriority,
    TaskStatus,
)


class TestPlanningAgent:
    """Test suite for PlanningAgent"""

    @pytest.fixture
    async def agent(self):
        """Create a PlanningAgent instance"""
        return PlanningAgent()

    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initializes correctly"""
        a = await agent
        assert a.name == "PlanningAgent"
        assert a.role.value == "planning"

    @pytest.mark.asyncio
    async def test_can_handle_planning_requests(self, agent):
        """Test agent can handle planning requests"""
        a = await agent
        assert await a.can_handle("Create a project plan") is True
        assert await a.can_handle("Break down this task") is True
        assert await a.can_handle("Plan my week") is True

    @pytest.mark.asyncio
    async def test_cannot_handle_non_planning(self, agent):
        """Test agent rejects non-planning requests"""
        a = await agent
        assert await a.can_handle("Write a poem") is False
        assert await a.can_handle("What's the weather?") is False

    @pytest.mark.asyncio
    async def test_estimate_task_duration_simple(self, agent):
        """Test estimating simple task duration"""
        a = await agent
        task = Task(description="Write a simple function")
        duration = a._estimate_task_duration(task)
        assert duration > 0

    @pytest.mark.asyncio
    async def test_estimate_task_duration_complex(self, agent):
        """Test estimating complex task duration"""
        a = await agent
        task = Task(description="Build a complex system", priority=TaskPriority.HIGH, complexity=5)
        duration = a._estimate_task_duration(task)
        assert duration > 60  # Complex tasks should take more than 1 hour

    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        a = await agent
        capabilities = a.get_capabilities()
        assert "planning" in capabilities.lower()
        assert "project" in capabilities.lower()


class TestTask:
    """Test suite for Task"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        task = Task(
            id="task_1",
            description="Test task",
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            estimated_duration=120,
        )
        data = task.to_dict()
        assert data["id"] == "task_1"
        assert data["description"] == "Test task"
        assert data["priority"] == "high"
        assert data["status"] == "in_progress"
        assert data["estimated_duration"] == 120


class TestMilestone:
    """Test suite for Milestone"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from datetime import datetime

        milestone = Milestone(
            id="ms_1",
            name="Phase 1 Complete",
            description="Complete first phase",
            target_date=datetime.now(),
        )
        data = milestone.to_dict()
        assert data["name"] == "Phase 1 Complete"
        assert data["completed"] is False


class TestProjectPlan:
    """Test suite for ProjectPlan"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        plan = ProjectPlan(
            name="Test Project",
            description="A test project",
            tasks=[Task(id="t1", description="Task 1")],
            milestones=[Milestone(id="m1", name="Milestone 1")],
            total_estimated_duration=480,
        )
        data = plan.to_dict()
        assert data["name"] == "Test Project"
        assert len(data["tasks"]) == 1
        assert len(data["milestones"]) == 1
        assert data["total_estimated_duration"] == 480


class TestPlanAdjustment:
    """Test suite for PlanAdjustment"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        adjustment = PlanAdjustment(
            reason="Scope change", changes={"tasks": ["new task"]}, impact="Adds 2 days"
        )
        data = adjustment.to_dict()
        assert data["reason"] == "Scope change"
        assert data["impact"] == "Adds 2 days"
