"""Tests for Workflow Automation (Triggers)"""

from datetime import timedelta

import pytest

from core.automation.triggers import (
    CompositeTrigger,
    ConditionTrigger,
    EventTrigger,
    TimeTrigger,
    TriggerManager,
    TriggerType,
)


class TestTimeTrigger:
    """Test suite for TimeTrigger"""

    @pytest.fixture
    def trigger(self):
        """Create a TimeTrigger instance"""
        return TimeTrigger(
            trigger_id="test_time",
            name="Test Time Trigger",
            workflow_id="test_workflow",
            interval=timedelta(minutes=5),
        )

    def test_initialization(self, trigger):
        """Test trigger initializes correctly"""
        assert trigger.trigger_id == "test_time"
        assert trigger.name == "Test Time Trigger"
        assert trigger.workflow_id == "test_workflow"
        assert trigger.trigger_type == TriggerType.TIME
        assert trigger.interval == timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_to_dict(self, trigger):
        """Test conversion to dictionary"""
        data = trigger.to_dict()
        assert data["trigger_id"] == "test_time"
        assert data["name"] == "Test Time Trigger"
        assert data["type"] == "time"


class TestEventTrigger:
    """Test suite for EventTrigger"""

    @pytest.fixture
    def trigger(self):
        """Create an EventTrigger instance"""
        return EventTrigger(
            trigger_id="test_event",
            name="Test Event Trigger",
            workflow_id="test_workflow",
            event_type="user_login",
            event_filter={"user_id": "123"},
        )

    def test_initialization(self, trigger):
        """Test trigger initializes correctly"""
        assert trigger.trigger_id == "test_event"
        assert trigger.event_type == "user_login"
        assert trigger.event_filter == {"user_id": "123"}

    def test_matches_event(self, trigger):
        """Test event matching"""
        assert trigger.matches_event("user_login", {"user_id": "123"}) is True
        assert trigger.matches_event("user_login", {"user_id": "456"}) is False
        assert trigger.matches_event("user_logout", {"user_id": "123"}) is False


class TestConditionTrigger:
    """Test suite for ConditionTrigger"""

    @pytest.fixture
    def trigger(self):
        """Create a ConditionTrigger instance"""
        return ConditionTrigger(
            trigger_id="test_condition",
            name="Test Condition Trigger",
            workflow_id="test_workflow",
            condition_type="file_exists",
            condition_config={"path": "/tmp/test.txt"},
        )

    def test_initialization(self, trigger):
        """Test trigger initializes correctly"""
        assert trigger.trigger_id == "test_condition"
        assert trigger.condition_type == "file_exists"
        assert trigger.condition_config["path"] == "/tmp/test.txt"


class TestCompositeTrigger:
    """Test suite for CompositeTrigger"""

    @pytest.fixture
    def triggers(self):
        """Create sub-triggers"""
        return [
            TimeTrigger(
                trigger_id="time1",
                name="Time Trigger",
                workflow_id="wf1",
                interval=timedelta(minutes=5),
            ),
            EventTrigger(
                trigger_id="event1",
                name="Event Trigger",
                workflow_id="wf1",
                event_type="test_event",
            ),
        ]

    @pytest.fixture
    def and_trigger(self, triggers):
        """Create an AND composite trigger"""
        return CompositeTrigger(
            trigger_id="test_and",
            name="Test AND Trigger",
            workflow_id="test_workflow",
            operator="and",
            triggers=triggers,
        )

    @pytest.fixture
    def or_trigger(self, triggers):
        """Create an OR composite trigger"""
        return CompositeTrigger(
            trigger_id="test_or",
            name="Test OR Trigger",
            workflow_id="test_workflow",
            operator="or",
            triggers=triggers,
        )

    def test_initialization(self, and_trigger):
        """Test composite trigger initializes correctly"""
        assert and_trigger.trigger_id == "test_and"
        assert and_trigger.operator == "and"
        assert len(and_trigger.triggers) == 2

    @pytest.mark.asyncio
    async def test_and_operator(self, and_trigger):
        """Test AND operator requires all triggers"""
        # Initially should be False
        result = await and_trigger.check()
        assert result is False

    @pytest.mark.asyncio
    async def test_or_operator(self, or_trigger):
        """Test OR operator requires any trigger"""
        # Initially should be False
        result = await or_trigger.check()
        assert result is False


class TestTriggerManager:
    """Test suite for TriggerManager"""

    @pytest.fixture
    async def manager(self):
        """Create a TriggerManager instance"""
        return TriggerManager()

    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test manager initializes correctly"""
        m = await manager
        assert isinstance(m.triggers, dict)
        assert isinstance(m.event_triggers, dict)

    @pytest.mark.asyncio
    async def test_register_trigger(self, manager):
        """Test registering a trigger"""
        m = await manager
        trigger = TimeTrigger(
            trigger_id="reg_test",
            name="Registration Test",
            workflow_id="wf1",
            interval=timedelta(minutes=5),
        )
        await m.register_trigger(trigger)
        assert "reg_test" in m.triggers

    @pytest.mark.asyncio
    async def test_unregister_trigger(self, manager):
        """Test unregistering a trigger"""
        m = await manager
        trigger = TimeTrigger(
            trigger_id="unreg_test",
            name="Unregistration Test",
            workflow_id="wf1",
            interval=timedelta(minutes=5),
        )
        await m.register_trigger(trigger)
        result = await m.unregister_trigger("unreg_test")
        assert result is True
        assert "unreg_test" not in m.triggers

    @pytest.mark.asyncio
    async def test_get_trigger(self, manager):
        """Test getting a trigger by ID"""
        m = await manager
        trigger = TimeTrigger(
            trigger_id="get_test",
            name="Get Test",
            workflow_id="wf1",
            interval=timedelta(minutes=5),
        )
        await m.register_trigger(trigger)
        retrieved = m.get_trigger("get_test")
        assert retrieved is not None
        assert retrieved.trigger_id == "get_test"

    @pytest.mark.asyncio
    async def test_list_triggers(self, manager):
        """Test listing triggers"""
        m = await manager
        trigger1 = TimeTrigger(
            trigger_id="list1",
            name="List Test 1",
            workflow_id="wf1",
            interval=timedelta(minutes=5),
        )
        trigger2 = EventTrigger(
            trigger_id="list2",
            name="List Test 2",
            workflow_id="wf2",
            event_type="test",
        )
        await m.register_trigger(trigger1)
        await m.register_trigger(trigger2)

        triggers = m.list_triggers()
        assert len(triggers) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        """Test getting trigger statistics"""
        m = await manager
        stats = m.get_stats()
        assert "total_triggers" in stats
        assert "by_type" in stats
        assert "by_status" in stats

    @pytest.mark.asyncio
    async def test_check_all(self, manager):
        """Test checking all triggers"""
        m = await manager
        # Register a trigger that won't fire
        trigger = TimeTrigger(
            trigger_id="check_test",
            name="Check Test",
            workflow_id="wf1",
            interval=timedelta(hours=1),  # Won't fire soon
        )
        await m.register_trigger(trigger)

        triggered = await m.check_all()
        assert isinstance(triggered, list)
