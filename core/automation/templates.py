"""
Workflow Templates Library for JARVIS.

Provides pre-built workflow templates and IFTTT-style automation recipes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from core.automation.workflow_orchestrator import WorkflowTask, WorkflowTemplate

log = logging.getLogger(__name__)


@dataclass
class WorkflowRecipe:
    """
    IFTTT-style automation recipe.

    Recipes follow the "If This Then That" pattern for easy automation.
    """

    id: str
    name: str
    description: str
    trigger: Dict[str, Any]  # The "This" part
    actions: List[Dict[str, Any]]  # The "That" part
    category: str = "general"
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    execution_count: int = 0
    last_executed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "actions": self.actions,
            "category": self.category,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "execution_count": self.execution_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
        }


class WorkflowTemplateLibrary:
    """
    Library of pre-built workflow templates.

    Provides templates for common automation scenarios.
    """

    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.recipes: Dict[str, WorkflowRecipe] = {}
        self._initialize_default_templates()
        self._initialize_ifttt_recipes()

    def _initialize_default_templates(self):
        """Initialize default workflow templates"""
        templates = [
            # System Maintenance
            WorkflowTemplate(
                name="system_maintenance",
                description="Automated system maintenance and cleanup",
                category="system",
                tasks=[
                    {
                        "id": "check_disk",
                        "name": "Check Disk Space",
                        "description": "Check available disk space",
                        "action": "system_check",
                        "parameters": {"check": "disk_space"},
                        "timeout": 30,
                    },
                    {
                        "id": "cleanup_temp",
                        "name": "Clean Temporary Files",
                        "description": "Remove old temporary files",
                        "action": "file_cleanup",
                        "parameters": {"paths": ["/tmp", "~/.cache"], "max_age_days": 7},
                        "dependencies": ["check_disk"],
                        "timeout": 300,
                    },
                    {
                        "id": "update_packages",
                        "name": "Update System Packages",
                        "description": "Update system packages",
                        "action": "system_update",
                        "parameters": {"type": "packages"},
                        "dependencies": ["cleanup_temp"],
                        "timeout": 600,
                    },
                    {
                        "id": "generate_report",
                        "name": "Generate Maintenance Report",
                        "description": "Create maintenance completion report",
                        "action": "generate_report",
                        "parameters": {"type": "maintenance"},
                        "dependencies": ["update_packages"],
                        "timeout": 60,
                    },
                ],
            ),
            # Data Backup
            WorkflowTemplate(
                name="data_backup",
                description="Automated data backup workflow",
                category="data",
                tasks=[
                    {
                        "id": "verify_source",
                        "name": "Verify Source Data",
                        "description": "Check source data availability",
                        "action": "file_check",
                        "parameters": {"path": "{{source_path}}"},
                        "timeout": 30,
                    },
                    {
                        "id": "create_backup",
                        "name": "Create Backup Archive",
                        "description": "Create compressed backup archive",
                        "action": "create_archive",
                        "parameters": {
                            "source": "{{source_path}}",
                            "destination": "{{backup_path}}",
                            "compression": "gzip",
                        },
                        "dependencies": ["verify_source"],
                        "timeout": 1800,
                    },
                    {
                        "id": "verify_backup",
                        "name": "Verify Backup Integrity",
                        "description": "Verify backup archive is valid",
                        "action": "verify_archive",
                        "parameters": {"path": "{{backup_path}}"},
                        "dependencies": ["create_backup"],
                        "timeout": 300,
                    },
                    {
                        "id": "cleanup_old",
                        "name": "Cleanup Old Backups",
                        "description": "Remove backups older than retention period",
                        "action": "cleanup_backups",
                        "parameters": {"path": "{{backup_path}}", "retention_days": 30},
                        "dependencies": ["verify_backup"],
                        "timeout": 120,
                    },
                    {
                        "id": "notify_complete",
                        "name": "Send Completion Notification",
                        "description": "Notify that backup is complete",
                        "action": "send_notification",
                        "parameters": {"message": "Backup completed successfully", "level": "info"},
                        "dependencies": ["cleanup_old"],
                        "timeout": 30,
                    },
                ],
            ),
            # Content Processing Pipeline
            WorkflowTemplate(
                name="content_processing",
                description="Process and analyze content files",
                category="content",
                tasks=[
                    {
                        "id": "fetch_content",
                        "name": "Fetch Content",
                        "description": "Fetch content from source",
                        "action": "fetch_url",
                        "parameters": {"url": "{{content_url}}"},
                        "timeout": 60,
                    },
                    {
                        "id": "extract_text",
                        "name": "Extract Text",
                        "description": "Extract text from content",
                        "action": "extract_text",
                        "parameters": {"format": "plain"},
                        "dependencies": ["fetch_content"],
                        "timeout": 30,
                    },
                    {
                        "id": "analyze_sentiment",
                        "name": "Analyze Sentiment",
                        "description": "Perform sentiment analysis",
                        "action": "sentiment_analysis",
                        "parameters": {},
                        "dependencies": ["extract_text"],
                        "timeout": 60,
                    },
                    {
                        "id": "extract_entities",
                        "name": "Extract Named Entities",
                        "description": "Extract people, organizations, locations",
                        "action": "ner_extraction",
                        "parameters": {},
                        "dependencies": ["extract_text"],
                        "timeout": 60,
                    },
                    {
                        "id": "summarize",
                        "name": "Generate Summary",
                        "description": "Create content summary",
                        "action": "summarize",
                        "parameters": {"max_length": 200},
                        "dependencies": ["extract_text"],
                        "timeout": 120,
                    },
                    {
                        "id": "save_results",
                        "name": "Save Analysis Results",
                        "description": "Save all analysis results",
                        "action": "save_json",
                        "parameters": {"path": "{{output_path}}"},
                        "dependencies": ["analyze_sentiment", "extract_entities", "summarize"],
                        "timeout": 30,
                    },
                ],
            ),
            # Development Workflow
            WorkflowTemplate(
                name="development_deployment",
                description="Automated development and deployment workflow",
                category="development",
                tasks=[
                    {
                        "id": "run_tests",
                        "name": "Run Test Suite",
                        "description": "Execute all tests",
                        "action": "run_command",
                        "parameters": {"command": "pytest", "args": ["-v", "--tb=short"]},
                        "timeout": 300,
                    },
                    {
                        "id": "lint_code",
                        "name": "Lint Code",
                        "description": "Run code linting",
                        "action": "run_command",
                        "parameters": {"command": "ruff", "args": ["check", "."]},
                        "timeout": 120,
                    },
                    {
                        "id": "type_check",
                        "name": "Type Check",
                        "description": "Run type checking",
                        "action": "run_command",
                        "parameters": {"command": "pyright", "args": ["."]},
                        "timeout": 180,
                    },
                    {
                        "id": "build_package",
                        "name": "Build Package",
                        "description": "Build deployment package",
                        "action": "run_command",
                        "parameters": {"command": "python", "args": ["-m", "build"]},
                        "dependencies": ["run_tests", "lint_code", "type_check"],
                        "timeout": 120,
                    },
                    {
                        "id": "deploy",
                        "name": "Deploy Application",
                        "description": "Deploy to production",
                        "action": "deploy",
                        "parameters": {"target": "production", "strategy": "blue_green"},
                        "dependencies": ["build_package"],
                        "timeout": 600,
                    },
                ],
            ),
            # Research and Report Generation
            WorkflowTemplate(
                name="research_report",
                description="Conduct research and generate comprehensive report",
                category="research",
                tasks=[
                    {
                        "id": "web_search",
                        "name": "Search Web Sources",
                        "description": "Search for relevant information",
                        "action": "web_search",
                        "parameters": {"query": "{{research_topic}}", "max_results": 10},
                        "timeout": 60,
                    },
                    {
                        "id": "fetch_sources",
                        "name": "Fetch Source Content",
                        "description": "Fetch content from top sources",
                        "action": "batch_fetch",
                        "parameters": {"max_sources": 5},
                        "dependencies": ["web_search"],
                        "timeout": 120,
                    },
                    {
                        "id": "analyze_sources",
                        "name": "Analyze Sources",
                        "description": "Analyze and synthesize information",
                        "action": "analyze_content",
                        "parameters": {"extract_key_points": True},
                        "dependencies": ["fetch_sources"],
                        "timeout": 180,
                    },
                    {
                        "id": "generate_report",
                        "name": "Generate Report",
                        "description": "Create comprehensive research report",
                        "action": "generate_document",
                        "parameters": {
                            "template": "research_report",
                            "format": "markdown",
                        },
                        "dependencies": ["analyze_sources"],
                        "timeout": 120,
                    },
                    {
                        "id": "export_report",
                        "name": "Export Report",
                        "description": "Export report to specified format",
                        "action": "export_document",
                        "parameters": {"format": "{{output_format}}", "path": "{{output_path}}"},
                        "dependencies": ["generate_report"],
                        "timeout": 60,
                    },
                ],
            ),
            # Notification and Alerting
            WorkflowTemplate(
                name="smart_notifications",
                description="Context-aware notification workflow",
                category="notifications",
                tasks=[
                    {
                        "id": "check_context",
                        "name": "Check User Context",
                        "description": "Determine current user context",
                        "action": "get_context",
                        "parameters": {"include": ["location", "activity", "do_not_disturb"]},
                        "timeout": 30,
                    },
                    {
                        "id": "determine_urgency",
                        "name": "Determine Urgency",
                        "description": "Assess notification urgency",
                        "action": "assess_priority",
                        "parameters": {"content": "{{notification_content}}"},
                        "timeout": 10,
                    },
                    {
                        "id": "select_channel",
                        "name": "Select Notification Channel",
                        "description": "Choose appropriate notification channel",
                        "action": "select_channel",
                        "parameters": {"channels": ["desktop", "mobile", "email"]},
                        "dependencies": ["check_context", "determine_urgency"],
                        "timeout": 10,
                    },
                    {
                        "id": "send_notification",
                        "name": "Send Notification",
                        "description": "Send notification through selected channel",
                        "action": "notify",
                        "parameters": {
                            "message": "{{notification_content}}",
                            "priority": "{{priority}}",
                        },
                        "dependencies": ["select_channel"],
                        "timeout": 30,
                    },
                ],
            ),
        ]

        for template in templates:
            self.templates[template.name] = template

    def _initialize_ifttt_recipes(self):
        """Initialize IFTTT-style automation recipes"""
        recipes = [
            # Morning Routine
            WorkflowRecipe(
                id="recipe_morning_routine",
                name="Morning Routine",
                description="Start your day with weather, calendar, and news",
                category="lifestyle",
                trigger={
                    "type": "time",
                    "config": {"cron": "0 7 * * *"},  # 7 AM daily
                },
                actions=[
                    {
                        "type": "get_weather",
                        "config": {"location": "home", "format": "brief"},
                    },
                    {
                        "type": "get_calendar",
                        "config": {"period": "today", "max_items": 5},
                    },
                    {
                        "type": "get_news",
                        "config": {"category": "headlines", "count": 5},
                    },
                    {
                        "type": "speak_summary",
                        "config": {"voice": "default"},
                    },
                ],
            ),
            # Focus Mode
            WorkflowRecipe(
                id="recipe_focus_mode",
                name="Deep Focus Mode",
                description="Enable focus mode during work hours",
                category="productivity",
                trigger={
                    "type": "time",
                    "config": {"cron": "0 9 * * 1-5"},  # 9 AM weekdays
                },
                actions=[
                    {
                        "type": "set_status",
                        "config": {"status": "busy", "message": "In focus mode"},
                    },
                    {
                        "type": "mute_notifications",
                        "config": {"except": ["urgent"], "duration": "4 hours"},
                    },
                    {
                        "type": "launch_app",
                        "config": {"app": "{{focus_app}}", "workspace": "work"},
                    },
                ],
            ),
            # Low Battery Alert
            WorkflowRecipe(
                id="recipe_low_battery",
                name="Low Battery Saver",
                description="Take action when battery is low",
                category="system",
                trigger={
                    "type": "condition",
                    "config": {
                        "condition_type": "system_load",
                        "config": {"battery_below": 20},
                    },
                },
                actions=[
                    {
                        "type": "enable_power_saving",
                        "config": {"mode": "aggressive"},
                    },
                    {
                        "type": "close_unnecessary_apps",
                        "config": {"keep": ["browser", "terminal"]},
                    },
                    {
                        "type": "notify",
                        "config": {"message": "Power saving mode enabled", "level": "warning"},
                    },
                ],
            ),
            # File Sync on Change
            WorkflowRecipe(
                id="recipe_file_sync",
                name="Auto Sync Important Files",
                description="Sync files when they change",
                category="files",
                trigger={
                    "type": "event",
                    "config": {
                        "event_type": "file_modified",
                        "filter": {"path": "~/Documents/Important/*"},
                    },
                },
                actions=[
                    {
                        "type": "sync_to_cloud",
                        "config": {"destination": "{{cloud_storage}}", "immediate": True},
                    },
                    {
                        "type": "create_backup",
                        "config": {"versioned": True, "keep_versions": 10},
                    },
                ],
            ),
            # Meeting Prep
            WorkflowRecipe(
                id="recipe_meeting_prep",
                name="Meeting Preparation",
                description="Prepare for upcoming meetings",
                category="productivity",
                trigger={
                    "type": "event",
                    "config": {"event_type": "calendar_event", "filter": {"starts_in_minutes": 15}},
                },
                actions=[
                    {
                        "type": "open_meeting_notes",
                        "config": {"create_if_missing": True},
                    },
                    {
                        "type": "research_attendees",
                        "config": {"source": "linkedin"},
                    },
                    {
                        "type": "check_previous_notes",
                        "config": {"lookback_days": 30},
                    },
                    {
                        "type": "notify",
                        "config": {"message": "Meeting starting in 15 minutes", "level": "info"},
                    },
                ],
            ),
            # End of Day Summary
            WorkflowRecipe(
                id="recipe_eod_summary",
                name="End of Day Summary",
                description="Review your day and plan tomorrow",
                category="productivity",
                trigger={
                    "type": "time",
                    "config": {"cron": "0 18 * * 1-5"},  # 6 PM weekdays
                },
                actions=[
                    {
                        "type": "summarize_day",
                        "config": {"include": ["tasks", "meetings", "achievements"]},
                    },
                    {
                        "type": "check_incomplete_tasks",
                        "config": {"move_to_tomorrow": True},
                    },
                    {
                        "type": "suggest_tomorrow_priorities",
                        "config": {"count": 3},
                    },
                    {
                        "type": "generate_daily_report",
                        "config": {"save": True, "share": False},
                    },
                ],
            ),
            # Security Alert
            WorkflowRecipe(
                id="recipe_security_alert",
                name="Security Monitor",
                description="Alert on suspicious activity",
                category="security",
                trigger={
                    "type": "condition",
                    "config": {
                        "condition_type": "custom",
                        "config": {"check": "failed_login_attempts", "threshold": 3},
                    },
                },
                actions=[
                    {
                        "type": "log_security_event",
                        "config": {"severity": "high"},
                    },
                    {
                        "type": "notify",
                        "config": {
                            "message": "Multiple failed login attempts detected",
                            "level": "alert",
                        },
                    },
                    {
                        "type": "enable_additional_security",
                        "config": {"enable_2fa_prompt": True},
                    },
                ],
            ),
            # Health Reminder
            WorkflowRecipe(
                id="recipe_health_reminder",
                name="Health Break Reminder",
                description="Remind to take breaks and stay healthy",
                category="health",
                trigger={
                    "type": "time",
                    "config": {"interval": 3600},  # Every hour
                },
                actions=[
                    {
                        "type": "check_screen_time",
                        "config": {"warn_after_minutes": 60},
                    },
                    {
                        "type": "suggest_break",
                        "config": {"type": "eye_rest", "duration": "5 minutes"},
                    },
                    {
                        "type": "notify",
                        "config": {"message": "Time for a break!", "level": "info"},
                    },
                ],
            ),
            # Arrive Home
            WorkflowRecipe(
                id="recipe_arrive_home",
                name="Arrive Home Automation",
                description="Actions to perform when arriving home",
                category="home",
                trigger={
                    "type": "event",
                    "config": {
                        "event_type": "location",
                        "filter": {"location": "home", "event": "arrive"},
                    },
                },
                actions=[
                    {
                        "type": "connect_to_wifi",
                        "config": {"network": "home_network"},
                    },
                    {
                        "type": "sync_files",
                        "config": {"direction": "bidirectional"},
                    },
                    {
                        "type": "enable_comfort_mode",
                        "config": {"lights": "warm", "music": "chill"},
                    },
                    {
                        "type": "check_messages",
                        "config": {"summarize": True},
                    },
                ],
            ),
            # Code Push
            WorkflowRecipe(
                id="recipe_code_push",
                name="Smart Code Push",
                description="Validate and push code safely",
                category="development",
                trigger={
                    "type": "event",
                    "config": {"event_type": "git_push", "filter": {"branch": "main"}},
                },
                actions=[
                    {
                        "type": "run_tests",
                        "config": {"suite": "all", "fail_fast": True},
                    },
                    {
                        "type": "check_code_quality",
                        "config": {"threshold": 80},
                    },
                    {
                        "type": "push_to_remote",
                        "config": {"repository": "origin", "branch": "main"},
                    },
                    {
                        "type": "notify",
                        "config": {"message": "Code pushed successfully", "level": "success"},
                    },
                ],
            ),
        ]

        for recipe in recipes:
            self.recipes[recipe.id] = recipe

    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by name"""
        return self.templates.get(name)

    def get_recipe(self, recipe_id: str) -> Optional[WorkflowRecipe]:
        """Get a recipe by ID"""
        return self.recipes.get(recipe_id)

    def list_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available templates"""
        results = []
        for name, template in self.templates.items():
            if category and template.category != category:
                continue
            results.append(
                {
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "task_count": len(template.tasks),
                }
            )
        return results

    def list_recipes(
        self, category: Optional[str] = None, enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        """List available recipes"""
        results = []
        for recipe_id, recipe in self.recipes.items():
            if category and recipe.category != category:
                continue
            if enabled_only and not recipe.enabled:
                continue
            results.append(recipe.to_dict())
        return results

    def create_custom_recipe(
        self,
        name: str,
        description: str,
        trigger_type: str,
        trigger_config: Dict[str, Any],
        actions: List[Dict[str, Any]],
        category: str = "custom",
    ) -> WorkflowRecipe:
        """Create a custom IFTTT-style recipe"""
        recipe_id = f"custom_{name.lower().replace(' ', '_')}"

        recipe = WorkflowRecipe(
            id=recipe_id,
            name=name,
            description=description,
            trigger={"type": trigger_type, "config": trigger_config},
            actions=actions,
            category=category,
        )

        self.recipes[recipe_id] = recipe
        return recipe

    def convert_recipe_to_workflow(self, recipe: WorkflowRecipe) -> WorkflowTemplate:
        """Convert an IFTTT recipe to a workflow template"""
        tasks = []

        for i, action in enumerate(recipe.actions, 1):
            task = {
                "id": f"action_{i}",
                "name": action["type"].replace("_", " ").title(),
                "description": f"Execute {action['type']}",
                "action": action["type"],
                "parameters": action.get("config", {}),
                "dependencies": [f"action_{i - 1}"] if i > 1 else [],
            }
            tasks.append(task)

        return WorkflowTemplate(
            name=recipe.id,
            description=recipe.description,
            tasks=tasks,
            category=recipe.category,
        )

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        categories = set()
        for template in self.templates.values():
            categories.add(template.category)
        for recipe in self.recipes.values():
            categories.add(recipe.category)
        return sorted(list(categories))


# Global library instance
_library: Optional[WorkflowTemplateLibrary] = None


def get_template_library() -> WorkflowTemplateLibrary:
    """Get the global template library instance"""
    global _library
    if _library is None:
        _library = WorkflowTemplateLibrary()
    return _library


__all__ = [
    "WorkflowTemplateLibrary",
    "WorkflowRecipe",
    "get_template_library",
]
