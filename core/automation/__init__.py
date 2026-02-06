"""
Automation module for JARVIS.

Provides workflow automation, triggers, and templates.
"""

from core.automation.triggers import (
    CompositeTrigger,
    ConditionTrigger,
    EventTrigger,
    TimeTrigger,
    Trigger,
    TriggerContext,
    TriggerManager,
    TriggerStatus,
    TriggerType,
    get_trigger_manager,
)
from core.automation.templates import WorkflowRecipe, WorkflowTemplateLibrary, get_template_library
from core.automation.workflow_orchestrator import (
    TaskStatus,
    WorkflowExecution,
    WorkflowOrchestrator,
    WorkflowStatus,
    WorkflowTask,
    WorkflowTemplate,
    get_workflow_orchestrator,
    workflow_orchestrator,
)

__all__ = [
    # Workflow orchestrator
    "WorkflowOrchestrator",
    "WorkflowTask",
    "WorkflowExecution",
    "WorkflowTemplate",
    "WorkflowStatus",
    "TaskStatus",
    "get_workflow_orchestrator",
    "workflow_orchestrator",
    # Triggers
    "Trigger",
    "TriggerType",
    "TriggerStatus",
    "TriggerContext",
    "TimeTrigger",
    "EventTrigger",
    "ConditionTrigger",
    "CompositeTrigger",
    "TriggerManager",
    "get_trigger_manager",
    # Templates
    "WorkflowTemplateLibrary",
    "WorkflowRecipe",
    "get_template_library",
]
