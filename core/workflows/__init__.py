from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from tools.registry import ToolRegistry, get_tool_registry


class TriggerType(Enum):
    MANUAL = "manual"
    TIME = "time"
    SCHEDULE = "schedule"
    EVENT = "event"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None
    on_error: str = "stop"
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None


@dataclass
class WorkflowTrigger:
    type: TriggerType
    time_str: str | None = None
    days: list[str] | None = None
    event: str | None = None
    cron: str | None = None


@dataclass
class Workflow:
    name: str
    description: str = ""
    trigger: WorkflowTrigger | None = None
    steps: list[WorkflowStep] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: datetime | None = None
    run_count: int = 0


class WorkflowEngine:
    def __init__(self, tools: ToolRegistry | None = None):
        self.tools = tools or get_tool_registry()
        self.workflows: dict[str, Workflow] = {}
        self._running: set[str] = set()
        self._event_handlers: dict[str, list[str]] = {}

    def load_from_yaml(self, path: str | Path) -> Workflow:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse_workflow(data)

    def load_directory(self, dir_path: str | Path) -> int:
        dir_path = Path(dir_path)
        count = 0
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                workflow = self.load_from_yaml(yaml_file)
                self.register(workflow)
                count += 1
            except Exception:
                continue
        for yml_file in dir_path.glob("*.yml"):
            try:
                workflow = self.load_from_yaml(yml_file)
                self.register(workflow)
                count += 1
            except Exception:
                continue
        return count

    def _parse_workflow(self, data: dict) -> Workflow:
        trigger = None
        if "trigger" in data:
            t = data["trigger"]
            trigger = WorkflowTrigger(
                type=TriggerType(t.get("type", "manual")),
                time_str=t.get("time"),
                days=t.get("days"),
                event=t.get("event"),
                cron=t.get("cron"),
            )

        steps = []
        for step_data in data.get("steps", []):
            steps.append(
                WorkflowStep(
                    action=step_data["action"],
                    params=step_data.get("params", {}),
                    condition=step_data.get("condition"),
                    on_error=step_data.get("on_error", "stop"),
                )
            )

        return Workflow(
            name=data["name"],
            description=data.get("description", ""),
            trigger=trigger,
            steps=steps,
            variables=data.get("variables", {}),
            enabled=data.get("enabled", True),
        )

    def register(self, workflow: Workflow) -> None:
        self.workflows[workflow.name] = workflow
        if workflow.trigger and workflow.trigger.event:
            event = workflow.trigger.event
            if event not in self._event_handlers:
                self._event_handlers[event] = []
            self._event_handlers[event].append(workflow.name)

    def unregister(self, name: str) -> bool:
        if name in self.workflows:
            workflow = self.workflows[name]
            if workflow.trigger and workflow.trigger.event:
                event = workflow.trigger.event
                if event in self._event_handlers:
                    self._event_handlers[event].remove(name)
            del self.workflows[name]
            return True
        return False

    async def run(self, name: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self.workflows:
            return {"success": False, "error": f"Workflow '{name}' not found"}

        if name in self._running:
            return {"success": False, "error": f"Workflow '{name}' already running"}

        workflow = self.workflows[name]
        if not workflow.enabled:
            return {"success": False, "error": f"Workflow '{name}' is disabled"}

        self._running.add(name)
        context = {**workflow.variables, **(variables or {})}
        results = []

        try:
            for i, step in enumerate(workflow.steps):
                step.status = StepStatus.RUNNING

                if step.condition and not self._evaluate_condition(step.condition, context):
                    step.status = StepStatus.SKIPPED
                    results.append({"step": i, "action": step.action, "skipped": True})
                    continue

                try:
                    result = await self._execute_step(step, context)
                    step.status = StepStatus.COMPLETED
                    step.result = result
                    context[f"step_{i}_result"] = result
                    results.append({"step": i, "action": step.action, "result": result})
                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    results.append({"step": i, "action": step.action, "error": str(e)})

                    if step.on_error == "stop":
                        break
                    elif step.on_error == "skip":
                        continue

            workflow.last_run = datetime.now()
            workflow.run_count += 1

            success = all(
                s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in workflow.steps
            )
            return {"success": success, "results": results, "context": context}

        finally:
            self._running.discard(name)

    async def _execute_step(self, step: WorkflowStep, context: dict[str, Any]) -> Any:
        params = self._resolve_params(step.params, context)
        result = await self.tools.execute(step.action, **params)
        if not result.success:
            raise Exception(result.error)
        return result.data

    def _resolve_params(self, params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                resolved[key] = context.get(var_name, value)
            elif isinstance(value, str) and "{{" in value:
                resolved[key] = self._template_replace(value, context)
            else:
                resolved[key] = value
        return resolved

    def _template_replace(self, template: str, context: dict[str, Any]) -> str:
        import re

        def replacer(match):
            var = match.group(1).strip()
            return str(context.get(var, match.group(0)))

        return re.sub(r"\{\{(.+?)\}\}", replacer, template)

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        try:
            safe_context = {k: v for k, v in context.items() if isinstance(k, str)}
            return bool(eval(condition, {"__builtins__": {}}, safe_context))
        except Exception:
            return False

    async def trigger_event(self, event: str, data: dict[str, Any] | None = None) -> list[dict]:
        results = []
        if event in self._event_handlers:
            for workflow_name in self._event_handlers[event]:
                result = await self.run(workflow_name, data)
                results.append({"workflow": workflow_name, **result})
        return results

    def list_workflows(self) -> list[dict[str, Any]]:
        return [
            {
                "name": w.name,
                "description": w.description,
                "enabled": w.enabled,
                "trigger": w.trigger.type.value if w.trigger else "manual",
                "steps": len(w.steps),
                "last_run": w.last_run.isoformat() if w.last_run else None,
                "run_count": w.run_count,
            }
            for w in self.workflows.values()
        ]

    def get_workflow(self, name: str) -> Workflow | None:
        return self.workflows.get(name)

    def to_yaml(self, workflow: Workflow) -> str:
        data = {
            "name": workflow.name,
            "description": workflow.description,
            "enabled": workflow.enabled,
            "steps": [{"action": s.action, "params": s.params} for s in workflow.steps],
        }
        if workflow.trigger:
            data["trigger"] = {
                "type": workflow.trigger.type.value,
                "time": workflow.trigger.time_str,
                "days": workflow.trigger.days,
            }
        if workflow.variables:
            data["variables"] = workflow.variables
        return yaml.dump(data, default_flow_style=False)

    def save_workflow(self, workflow: Workflow, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml(workflow))
