"""
Intelligent Workflow Orchestrator for JARVIS.

This module provides sophisticated workflow automation capabilities including:
- Multi-step task orchestration with dependencies
- Dynamic workflow adaptation and optimization
- Intelligent scheduling and resource allocation
- Error recovery and rollback mechanisms
- Workflow templates and reusability
- Real-time workflow monitoring and analytics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class WorkflowTask:
    """Represents a single task in a workflow"""

    id: str
    name: str
    description: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTask":
        """Create task from dictionary representation"""
        task = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            action=data["action"],
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            timeout=data.get("timeout"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {}),
        )
        task.status = TaskStatus(data.get("status", "pending"))
        task.result = data.get("result")
        task.error = data.get("error")

        if data.get("start_time"):
            task.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            task.end_time = datetime.fromisoformat(data["end_time"])

        return task


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance"""

    id: str
    name: str
    description: str
    tasks: Dict[str, WorkflowTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowExecution":
        """Create execution from dictionary representation"""
        execution = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            tasks={
                tid: WorkflowTask.from_dict(tdata) for tid, tdata in data.get("tasks", {}).items()
            },
            priority=data.get("priority", 1),
            metadata=data.get("metadata", {}),
        )
        execution.status = WorkflowStatus(data.get("status", "pending"))
        execution.progress = data.get("progress", 0.0)

        if data.get("created_at"):
            execution.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            execution.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            execution.completed_at = datetime.fromisoformat(data["completed_at"])

        return execution


class WorkflowTemplate:
    """Reusable workflow template"""

    def __init__(
        self,
        name: str,
        description: str,
        tasks: List[Dict[str, Any]],
        category: str = "general",
        version: str = "1.0",
    ):
        self.name = name
        self.description = description
        self.tasks = tasks
        self.category = category
        self.version = version
        self.created_at = datetime.now()
        self.usage_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "tasks": self.tasks,
            "category": self.category,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "usage_count": self.usage_count,
        }

    def instantiate(self, parameters: Optional[Dict[str, Any]] = None) -> List[WorkflowTask]:
        """Create tasks from template with parameter substitution"""
        tasks = []
        param_dict = parameters or {}

        for task_data in self.tasks:
            # Substitute parameters in task data
            task_dict = self._substitute_parameters(task_data, param_dict)

            task = WorkflowTask(
                id=task_dict["id"],
                name=task_dict["name"],
                description=task_dict["description"],
                action=task_dict["action"],
                parameters=task_dict.get("parameters", {}),
                dependencies=task_dict.get("dependencies", []),
                timeout=task_dict.get("timeout"),
                max_retries=task_dict.get("max_retries", 3),
            )
            tasks.append(task)

        self.usage_count += 1
        return tasks

    def _substitute_parameters(self, data: Any, params: Dict[str, Any]) -> Any:
        """Recursively substitute parameters in data structure"""
        if isinstance(data, str):
            # Replace {{param_name}} with parameter values
            for key, value in params.items():
                data = data.replace(f"{{{{key}}}}", str(value))
            return data
        elif isinstance(data, dict):
            return {k: self._substitute_parameters(v, params) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_parameters(item, params) for item in data]
        else:
            return data


class WorkflowOrchestrator:
    """Main workflow orchestration engine"""

    def __init__(self):
        self.executions: Dict[str, WorkflowExecution] = {}
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.active_executions: Set[str] = set()
        self.task_queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []
        self.max_concurrent_tasks = 5
        self.is_running = False

    async def initialize(self):
        """Initialize the orchestrator"""
        if not self.is_running:
            self.is_running = True
            # Start worker tasks
            for i in range(self.max_concurrent_tasks):
                task = asyncio.create_task(self._task_worker(f"worker-{i}"))
                self.worker_tasks.append(task)
            log.info(f"Workflow orchestrator initialized with {self.max_concurrent_tasks} workers")

    async def shutdown(self):
        """Shutdown the orchestrator"""
        self.is_running = False

        # Cancel all active executions
        for execution_id in list(self.active_executions):
            await self.cancel_execution(execution_id)

        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        log.info("Workflow orchestrator shutdown complete")

    async def create_workflow_from_template(
        self, template_name: str, parameters: Optional[Dict[str, Any]] = None, priority: int = 1
    ) -> Optional[str]:
        """Create and start a workflow from a template"""
        if template_name not in self.templates:
            log.error(f"Template '{template_name}' not found")
            return None

        template = self.templates[template_name]
        tasks = template.instantiate(parameters)

        # Generate execution ID
        execution_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Create execution
        execution = WorkflowExecution(
            id=execution_id,
            name=f"{template.name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            description=template.description,
            tasks={task.id: task for task in tasks},
            priority=priority,
        )

        self.executions[execution_id] = execution
        await self.start_execution(execution_id)
        return execution_id

    async def create_custom_workflow(
        self, name: str, description: str, tasks: List[WorkflowTask], priority: int = 1
    ) -> str:
        """Create and start a custom workflow"""
        execution_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        execution = WorkflowExecution(
            id=execution_id,
            name=name,
            description=description,
            tasks={task.id: task for task in tasks},
            priority=priority,
        )

        self.executions[execution_id] = execution
        await self.start_execution(execution_id)
        return execution_id

    async def start_execution(self, execution_id: str) -> bool:
        """Start a workflow execution"""
        if execution_id not in self.executions:
            log.error(f"Execution '{execution_id}' not found")
            return False

        execution = self.executions[execution_id]
        if execution.status != WorkflowStatus.PENDING:
            log.warning(f"Execution '{execution_id}' is not in pending state")
            return False

        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.now()
        self.active_executions.add(execution_id)

        # Queue initial tasks (those with no dependencies)
        initial_tasks = [
            tid
            for tid, task in execution.tasks.items()
            if not task.dependencies and task.status == TaskStatus.PENDING
        ]

        for task_id in initial_tasks:
            await self.task_queue.put((execution_id, task_id))

        log.info(
            f"Started workflow execution '{execution_id}' with {len(initial_tasks)} initial tasks"
        )
        return True

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        if execution_id not in self.executions:
            return False

        execution = self.executions[execution_id]
        if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            return False

        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now()

        # Cancel all pending tasks
        for task in execution.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
                task.end_time = datetime.now()

        self.active_executions.discard(execution_id)
        log.info(f"Cancelled workflow execution '{execution_id}'")
        return True

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a workflow execution"""
        if execution_id not in self.executions:
            return None

        execution = self.executions[execution_id]
        return execution.to_dict()

    async def list_executions(
        self, status_filter: Optional[WorkflowStatus] = None
    ) -> List[Dict[str, Any]]:
        """List all workflow executions with optional status filter"""
        executions = []
        for execution in self.executions.values():
            if status_filter is None or execution.status == status_filter:
                executions.append(
                    {
                        "id": execution.id,
                        "name": execution.name,
                        "status": execution.status.value,
                        "progress": execution.progress,
                        "created_at": execution.created_at.isoformat(),
                        "priority": execution.priority,
                    }
                )
        return executions

    async def _task_worker(self, worker_name: str):
        """Worker task that processes workflow tasks"""
        log.debug(f"Started workflow worker {worker_name}")

        while self.is_running:
            try:
                # Wait for a task with timeout
                execution_id, task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._execute_task(execution_id, task_id)
            except Exception as e:
                log.error(f"Error executing task {task_id} in execution {execution_id}: {e}")
            finally:
                self.task_queue.task_done()

        log.debug(f"Stopped workflow worker {worker_name}")

    async def _execute_task(self, execution_id: str, task_id: str):
        """Execute a single workflow task"""
        if execution_id not in self.executions:
            log.error(f"Execution '{execution_id}' not found")
            return

        execution = self.executions[execution_id]
        if task_id not in execution.tasks:
            log.error(f"Task '{task_id}' not found in execution '{execution_id}'")
            return

        task = execution.tasks[task_id]

        # Check if task is still pending
        if task.status != TaskStatus.PENDING:
            return

        # Update task status
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()

        try:
            # Execute the task
            result = await self._perform_task_action(task)

            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.end_time = datetime.now()

            log.info(f"Completed task '{task_id}' in execution '{execution_id}'")

            # Check if execution is complete
            await self._check_execution_completion(execution_id)

            # Queue dependent tasks
            await self._queue_dependent_tasks(execution_id, task_id)

        except Exception as e:
            log.error(f"Task '{task_id}' failed: {e}")

            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                # Re-queue the task for retry
                await self.task_queue.put((execution_id, task_id))
                log.info(
                    f"Retrying task '{task_id}' (attempt {task.retry_count}/{task.max_retries})"
                )
            else:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.end_time = datetime.now()

                # Check if execution should fail
                await self._check_execution_failure(execution_id)

    async def _perform_task_action(self, task: WorkflowTask) -> Any:
        """Perform the actual task action"""
        # This is where you would integrate with the actual tool system
        # For now, we'll simulate task execution based on the action

        action = task.action.lower()

        if action == "delay":
            delay = task.parameters.get("seconds", 1)
            await asyncio.sleep(delay)
            return f"Delayed for {delay} seconds"

        elif action == "log_message":
            message = task.parameters.get("message", "No message")
            log.info(f"Workflow task: {message}")
            return f"Logged: {message}"

        elif action == "http_request":
            # Simulate HTTP request
            url = task.parameters.get("url", "http://example.com")
            method = task.parameters.get("method", "GET")
            # In real implementation, use httpx or similar
            await asyncio.sleep(0.1)  # Simulate network delay
            return f"HTTP {method} to {url}"

        elif action == "file_operation":
            operation = task.parameters.get("operation", "read")
            path = task.parameters.get("path", "/tmp/test")
            # In real implementation, perform actual file operations safely
            await asyncio.sleep(0.05)
            return f"File {operation} on {path}"

        elif action == "data_processing":
            data_type = task.parameters.get("type", "text")
            size = task.parameters.get("size", 1000)
            # Simulate data processing
            await asyncio.sleep(size / 100000)  # Scale processing time with data size
            return f"Processed {size} units of {data_type} data"

        else:
            # Generic task execution
            await asyncio.sleep(0.1)
            return f"Executed action: {action}"

    async def _check_execution_completion(self, execution_id: str):
        """Check if workflow execution is complete"""
        execution = self.executions[execution_id]

        # Check if all tasks are completed
        all_completed = all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.CANCELLED]
            for task in execution.tasks.values()
        )

        if all_completed:
            # Calculate final status
            has_failures = any(
                task.status == TaskStatus.FAILED for task in execution.tasks.values()
            )

            if has_failures:
                execution.status = WorkflowStatus.FAILED
            else:
                execution.status = WorkflowStatus.COMPLETED

            execution.completed_at = datetime.now()

            # Calculate progress
            total_tasks = len(execution.tasks)
            completed_tasks = sum(
                1
                for task in execution.tasks.values()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
            )
            execution.progress = completed_tasks / total_tasks if total_tasks > 0 else 1.0

            self.active_executions.discard(execution_id)
            log.info(
                f"Workflow execution '{execution_id}' completed with status: {execution.status.value}"
            )

    async def _check_execution_failure(self, execution_id: str):
        """Check if workflow execution should fail due to task failure"""
        execution = self.executions[execution_id]

        # For now, we'll fail the entire execution if any critical task fails
        # In a more sophisticated system, you might have different failure modes
        execution.status = WorkflowStatus.FAILED
        execution.completed_at = datetime.now()
        self.active_executions.discard(execution_id)
        log.error(f"Workflow execution '{execution_id}' failed due to task failure")

    async def _queue_dependent_tasks(self, execution_id: str, completed_task_id: str):
        """Queue tasks that depend on the completed task"""
        execution = self.executions[execution_id]

        for task_id, task in execution.tasks.items():
            if task.status == TaskStatus.PENDING and completed_task_id in task.dependencies:
                # Check if all dependencies are satisfied
                all_deps_satisfied = all(
                    execution.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )

                if all_deps_satisfied:
                    await self.task_queue.put((execution_id, task_id))
                    log.debug(f"Queued dependent task '{task_id}' after '{completed_task_id}'")

    def register_template(self, template: WorkflowTemplate):
        """Register a workflow template"""
        self.templates[template.name] = template
        log.info(f"Registered workflow template: {template.name}")

    def get_available_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available workflow templates"""
        templates = []
        for template in self.templates.values():
            if category is None or template.category == category:
                templates.append(
                    {
                        "name": template.name,
                        "description": template.description,
                        "category": template.category,
                        "version": template.version,
                        "usage_count": template.usage_count,
                    }
                )
        return templates

    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        total_executions = len(self.executions)
        active_executions = len(self.active_executions)
        completed_executions = sum(
            1 for e in self.executions.values() if e.status == WorkflowStatus.COMPLETED
        )
        failed_executions = sum(
            1 for e in self.executions.values() if e.status == WorkflowStatus.FAILED
        )

        # Calculate average execution time for completed workflows
        completed_times = []
        for execution in self.executions.values():
            if (
                execution.status == WorkflowStatus.COMPLETED
                and execution.started_at
                and execution.completed_at
            ):
                duration = (execution.completed_at - execution.started_at).total_seconds()
                completed_times.append(duration)

        avg_completion_time = sum(completed_times) / len(completed_times) if completed_times else 0

        return {
            "total_executions": total_executions,
            "active_executions": active_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "completion_rate": completed_executions / total_executions
            if total_executions > 0
            else 0,
            "average_completion_time_seconds": avg_completion_time,
            "queue_size": self.task_queue.qsize(),
            "available_templates": len(self.templates),
        }


# Global workflow orchestrator instance
workflow_orchestrator = WorkflowOrchestrator()


async def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Get the global workflow orchestrator instance"""
    await workflow_orchestrator.initialize()
    return workflow_orchestrator


# Initialize default workflow templates
def _initialize_default_templates():
    """Initialize default workflow templates"""
    templates = [
        WorkflowTemplate(
            name="data_processing_pipeline",
            description="Complete data processing workflow with validation and storage",
            category="data",
            tasks=[
                {
                    "id": "validate_input",
                    "name": "Validate Input Data",
                    "description": "Check input data format and quality",
                    "action": "data_processing",
                    "parameters": {"type": "validation", "size": 1000},
                    "timeout": 30,
                },
                {
                    "id": "process_data",
                    "name": "Process Data",
                    "description": "Transform and clean the data",
                    "action": "data_processing",
                    "parameters": {"type": "transformation", "size": 5000},
                    "dependencies": ["validate_input"],
                    "timeout": 120,
                },
                {
                    "id": "store_results",
                    "name": "Store Results",
                    "description": "Save processed data to storage",
                    "action": "file_operation",
                    "parameters": {"operation": "write", "path": "/data/results"},
                    "dependencies": ["process_data"],
                    "timeout": 60,
                },
            ],
        ),
        WorkflowTemplate(
            name="web_scraping_workflow",
            description="Automated web scraping with data extraction and storage",
            category="web",
            tasks=[
                {
                    "id": "discover_urls",
                    "name": "Discover Target URLs",
                    "description": "Find relevant URLs to scrape",
                    "action": "http_request",
                    "parameters": {"method": "GET", "url": "{{search_engine}}"},
                    "timeout": 30,
                },
                {
                    "id": "scrape_content",
                    "name": "Scrape Web Content",
                    "description": "Extract data from discovered URLs",
                    "action": "http_request",
                    "parameters": {"method": "GET", "url": "{{target_url}}"},
                    "dependencies": ["discover_urls"],
                    "timeout": 60,
                },
                {
                    "id": "extract_data",
                    "name": "Extract Structured Data",
                    "description": "Parse and structure scraped content",
                    "action": "data_processing",
                    "parameters": {"type": "parsing", "size": 2000},
                    "dependencies": ["scrape_content"],
                    "timeout": 45,
                },
                {
                    "id": "store_dataset",
                    "name": "Store Dataset",
                    "description": "Save extracted data to database",
                    "action": "file_operation",
                    "parameters": {"operation": "write", "path": "/data/{{dataset_name}}"},
                    "dependencies": ["extract_data"],
                    "timeout": 30,
                },
            ],
        ),
        WorkflowTemplate(
            name="system_maintenance",
            description="Automated system maintenance and optimization",
            category="system",
            tasks=[
                {
                    "id": "check_disk_space",
                    "name": "Check Disk Space",
                    "description": "Monitor disk usage and cleanup if needed",
                    "action": "file_operation",
                    "parameters": {"operation": "check_space", "path": "/"},
                    "timeout": 30,
                },
                {
                    "id": "update_packages",
                    "name": "Update System Packages",
                    "description": "Update system packages and dependencies",
                    "action": "file_operation",
                    "parameters": {"operation": "update_packages"},
                    "dependencies": ["check_disk_space"],
                    "timeout": 300,
                },
                {
                    "id": "cleanup_logs",
                    "name": "Clean Up Log Files",
                    "description": "Remove old log files and compress current ones",
                    "action": "file_operation",
                    "parameters": {"operation": "cleanup_logs", "path": "/var/log"},
                    "dependencies": ["update_packages"],
                    "timeout": 120,
                },
                {
                    "id": "system_report",
                    "name": "Generate System Report",
                    "description": "Create maintenance completion report",
                    "action": "file_operation",
                    "parameters": {"operation": "generate_report", "path": "/reports/maintenance"},
                    "dependencies": ["cleanup_logs"],
                    "timeout": 60,
                },
            ],
        ),
    ]

    # Register templates
    for template in templates:
        workflow_orchestrator.register_template(template)


# Initialize default templates on module load
_initialize_default_templates()


__all__ = [
    "WorkflowStatus",
    "TaskStatus",
    "WorkflowTask",
    "WorkflowExecution",
    "WorkflowTemplate",
    "WorkflowOrchestrator",
    "workflow_orchestrator",
    "get_workflow_orchestrator",
]
