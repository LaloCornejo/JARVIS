"""
PlanningAgent - Specialized agent for task planning and project management.

This agent provides:
- Task breakdown and decomposition
- Project planning and scheduling
- Resource allocation
- Dependency management
- Progress tracking
- Risk assessment
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agents.base import AgentContext, AgentRole, BaseAgent
from core.llm import OllamaClient

log = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a single task"""

    id: str
    title: str
    description: str
    estimated_duration: timedelta
    priority: str = "medium"  # low, medium, high, critical
    status: str = "pending"  # pending, in_progress, completed, blocked
    dependencies: List[str] = field(default_factory=list)
    subtasks: List["Task"] = field(default_factory=list)
    assigned_to: str = ""
    due_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)


@dataclass
class Milestone:
    """Represents a project milestone"""

    id: str
    title: str
    description: str
    target_date: datetime
    tasks: List[str] = field(default_factory=list)
    status: str = "pending"
    deliverables: List[str] = field(default_factory=list)


@dataclass
class ProjectPlan:
    """Complete project plan"""

    title: str
    description: str
    tasks: List[Task]
    milestones: List[Milestone]
    start_date: datetime
    target_end_date: datetime
    total_estimated_duration: timedelta
    critical_path: List[str] = field(default_factory=list)
    risks: List[Dict[str, Any]] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PlanAdjustment:
    """Suggested plan adjustment"""

    type: str  # add_task, remove_task, reorder, extend_timeline, etc.
    description: str
    reason: str
    impact: str
    tasks_affected: List[str] = field(default_factory=list)


class PlanningAgent(BaseAgent):
    """Agent specialized in task planning and project management"""

    name = "PlanningAgent"
    role = AgentRole.TASK
    description = (
        "Creates detailed plans, breaks down tasks, manages dependencies, and tracks progress"
    )
    system_prompt = """You are an expert project planner and task manager with expertise in:
- Work breakdown structures (WBS)
- Agile and waterfall methodologies
- Critical path analysis
- Resource allocation
- Risk assessment
- Dependency management
- Time estimation
- Priority management

When creating plans:
1. Break down tasks into manageable chunks
2. Identify dependencies clearly
3. Estimate time realistically
4. Consider resource constraints
5. Build in buffer time
6. Identify risks proactively
7. Create measurable milestones

Always provide specific, actionable plans with clear timelines."""

    PRIORITY_WEIGHTS = {"low": 1, "medium": 2, "high": 3, "critical": 5}

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        super().__init__()
        self.llm = llm_client or OllamaClient()
        self.active_plans: Dict[str, ProjectPlan] = {}
        self.task_counter = 0

    async def process(self, message: str, context: Optional[AgentContext] = None) -> str:
        """Process planning request"""
        # Parse the request
        request_type, params = self._parse_request(message)

        if request_type == "create_plan":
            result = await self.create_project_plan(**params)
        elif request_type == "breakdown":
            result = await self.breakdown_task(**params)
        elif request_type == "adjust":
            result = await self.adjust_plan(**params)
        elif request_type == "analyze":
            result = await self.analyze_plan(**params)
        elif request_type == "prioritize":
            result = await self.prioritize_tasks(**params)
        else:
            return json.dumps({"error": "Unknown planning request type"})

        return json.dumps(
            result, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)
        )

    async def can_handle(self, message: str) -> float:
        """Check if this agent can handle the message"""
        planning_keywords = [
            "plan",
            "schedule",
            "organize",
            "break down",
            "decompose",
            "milestone",
            "deadline",
            "timeline",
            "project",
            "task list",
            "todo",
            "prioritize",
            "dependencies",
            "workflow",
            "roadmap",
            "strategy",
            "steps",
            "process",
            "procedure",
        ]
        message_lower = message.lower()

        keyword_matches = sum(1 for kw in planning_keywords if kw in message_lower)

        # Check for specific patterns
        is_plan_request = any(
            phrase in message_lower
            for phrase in [
                "create a plan",
                "make a plan",
                "help me plan",
                "break this down",
                "organize this",
            ]
        )
        has_task_list = "list" in message_lower and any(
            word in message_lower for word in ["task", "step", "action", "item"]
        )

        confidence = min(keyword_matches * 0.1, 0.4)
        if is_plan_request:
            confidence += 0.4
        if has_task_list:
            confidence += 0.2

        return min(confidence, 1.0)

    async def create_project_plan(
        self,
        title: str,
        description: str,
        goal: str,
        constraints: Optional[Dict[str, Any]] = None,
        resources: Optional[List[str]] = None,
        deadline: Optional[datetime] = None,
        methodology: str = "hybrid",
    ) -> ProjectPlan:
        """Create a comprehensive project plan"""
        log.info(f"Creating project plan: {title}")

        constraints = constraints or {}
        resources = resources or []

        # Generate tasks using LLM
        tasks_data = await self._generate_tasks_with_llm(title, description, goal, constraints)

        # Create task objects
        tasks = []
        for task_data in tasks_data:
            task = Task(
                id=self._generate_task_id(),
                title=task_data["title"],
                description=task_data.get("description", ""),
                estimated_duration=self._parse_duration(task_data.get("duration", "2 hours")),
                priority=task_data.get("priority", "medium"),
                dependencies=task_data.get("dependencies", []),
                tags=task_data.get("tags", []),
                risks=task_data.get("risks", []),
            )
            tasks.append(task)

        # Create subtasks for complex tasks
        for task in tasks:
            if task.estimated_duration > timedelta(hours=8):
                task.subtasks = await self._create_subtasks(task)

        # Identify milestones
        milestones = await self._identify_milestones(tasks, title, deadline)

        # Calculate critical path
        critical_path = self._calculate_critical_path(tasks)

        # Assess risks
        risks = await self._assess_risks(tasks, constraints)

        # Calculate total duration
        total_duration = self._calculate_total_duration(tasks)

        # Set dates
        start_date = datetime.now()
        target_end_date = deadline or (start_date + total_duration)

        plan = ProjectPlan(
            title=title,
            description=description,
            tasks=tasks,
            milestones=milestones,
            start_date=start_date,
            target_end_date=target_end_date,
            total_estimated_duration=total_duration,
            critical_path=critical_path,
            risks=risks,
            resources={
                "available": resources,
                "allocated": self._allocate_resources(tasks, resources),
            },
        )

        # Store plan
        self.active_plans[title] = plan

        return plan

    async def breakdown_task(
        self, task_description: str, granularity: str = "medium", max_subtasks: int = 10
    ) -> List[Task]:
        """Break down a task into subtasks"""
        log.info(f"Breaking down task: {task_description[:50]}...")

        prompt = f"""Break down the following task into {max_subtasks} or fewer subtasks:

Task: {task_description}
Granularity: {granularity} (fine = small steps, medium = balanced, coarse = large chunks)

For each subtask, provide:
- Title
- Description
- Estimated duration (e.g., "30 minutes", "2 hours", "1 day")
- Priority (low/medium/high)
- Any dependencies on other subtasks

Respond in JSON format as a list of subtasks."""

        try:
            response = await self._generate_with_llm(prompt)
            subtasks_data = json.loads(response)

            subtasks = []
            for data in subtasks_data:
                task = Task(
                    id=self._generate_task_id(),
                    title=data["title"],
                    description=data.get("description", ""),
                    estimated_duration=self._parse_duration(data.get("duration", "1 hour")),
                    priority=data.get("priority", "medium"),
                    dependencies=data.get("dependencies", []),
                )
                subtasks.append(task)

            return subtasks

        except (json.JSONDecodeError, KeyError) as e:
            log.error(f"Error parsing task breakdown: {e}")
            # Fallback: create simple breakdown
            return self._simple_breakdown(task_description, max_subtasks)

    async def adjust_plan(
        self, plan_id: str, changes: Dict[str, Any], reason: str = ""
    ) -> List[PlanAdjustment]:
        """Suggest adjustments to an existing plan"""
        if plan_id not in self.active_plans:
            return [PlanAdjustment("error", "Plan not found", "", "", [])]

        plan = self.active_plans[plan_id]
        adjustments = []

        # Analyze changes
        if "new_deadline" in changes:
            new_deadline = changes["new_deadline"]
            current_end = plan.target_end_date

            if new_deadline < current_end:
                # Need to compress timeline
                adjustment = PlanAdjustment(
                    type="compress_timeline",
                    description=f"Compress timeline to meet new deadline: {new_deadline}",
                    reason=reason or "Deadline moved earlier",
                    impact="May require parallel work or scope reduction",
                    tasks_affected=[t.id for t in plan.tasks],
                )
                adjustments.append(adjustment)

        if "removed_resources" in changes:
            removed = changes["removed_resources"]
            affected_tasks = [
                t.id for t in plan.tasks if any(r in t.resources_needed for r in removed)
            ]

            if affected_tasks:
                adjustment = PlanAdjustment(
                    type="reallocate_resources",
                    description=f"Reallocate tasks affected by resource removal: {removed}",
                    reason=reason or "Resources no longer available",
                    impact="Timeline may be extended",
                    tasks_affected=affected_tasks,
                )
                adjustments.append(adjustment)

        if "scope_change" in changes:
            scope = changes["scope_change"]
            if scope == "expand":
                adjustment = PlanAdjustment(
                    type="extend_timeline",
                    description="Extend timeline to accommodate scope expansion",
                    reason=reason or "Additional requirements added",
                    impact="End date will be pushed back",
                    tasks_affected=[],
                )
                adjustments.append(adjustment)

        return adjustments

    async def analyze_plan(self, plan_id: str) -> Dict[str, Any]:
        """Analyze a plan for issues and optimizations"""
        if plan_id not in self.active_plans:
            return {"error": "Plan not found"}

        plan = self.active_plans[plan_id]

        analysis = {
            "plan_id": plan_id,
            "total_tasks": len(plan.tasks),
            "total_duration_days": plan.total_estimated_duration.days,
            "critical_path_length": len(plan.critical_path),
            "bottlenecks": self._identify_bottlenecks(plan.tasks),
            "resource_conflicts": self._find_resource_conflicts(plan),
            "schedule_risks": self._assess_schedule_risks(plan),
            "optimization_opportunities": await self._find_optimizations(plan),
            "recommendations": [],
        }

        # Generate recommendations
        recommendations = []

        if analysis["bottlenecks"]:
            recommendations.append(
                f"Address {len(analysis['bottlenecks'])} bottlenecks in critical path"
            )

        if analysis["resource_conflicts"]:
            recommendations.append(
                f"Resolve {len(analysis['resource_conflicts'])} resource conflicts"
            )

        if plan.target_end_date and plan.start_date:
            buffer_days = (
                plan.target_end_date - (plan.start_date + plan.total_estimated_duration)
            ).days
            if buffer_days < 2:
                recommendations.append("Consider adding buffer time for unexpected delays")

        analysis["recommendations"] = recommendations

        return analysis

    async def prioritize_tasks(self, tasks: List[Task], strategy: str = "weighted") -> List[Task]:
        """Prioritize tasks based on strategy"""
        if strategy == "weighted":
            # Consider priority, dependencies, and impact
            scored_tasks = []
            for task in tasks:
                score = self._calculate_priority_score(task)
                scored_tasks.append((score, task))

            scored_tasks.sort(reverse=True)
            return [task for _, task in scored_tasks]

        elif strategy == "dependencies":
            # Topological sort based on dependencies
            return self._topological_sort(tasks)

        elif strategy == "deadline":
            # Sort by due date
            return sorted(tasks, key=lambda t: t.due_date or datetime.max)

        elif strategy == "effort_value":
            # High value, low effort first
            return sorted(tasks, key=lambda t: (t.priority != "high", t.estimated_duration))

        return tasks

    def _parse_request(self, message: str) -> tuple[str, Dict[str, Any]]:
        """Parse planning request"""
        message_lower = message.lower()

        # Determine request type
        if any(phrase in message_lower for phrase in ["break down", "decompose", "subtasks"]):
            return "breakdown", self._parse_breakdown_params(message)
        elif any(word in message_lower for word in ["adjust", "modify", "change plan"]):
            return "adjust", self._parse_adjust_params(message)
        elif any(word in message_lower for word in ["analyze", "review", "assess plan"]):
            return "analyze", self._parse_analyze_params(message)
        elif "prioritize" in message_lower:
            return "prioritize", self._parse_prioritize_params(message)
        else:
            return "create_plan", self._parse_create_plan_params(message)

    def _parse_create_plan_params(self, message: str) -> Dict[str, Any]:
        """Parse project plan creation parameters"""
        params = {
            "title": "Project",
            "description": message,
            "goal": "",
            "constraints": {},
            "resources": [],
            "deadline": None,
            "methodology": "hybrid",
        }

        # Extract title
        title_match = re.search(r'["\'](.+?)["\']', message)
        if title_match:
            params["title"] = title_match.group(1)

        # Extract goal
        goal_patterns = [
            r"goal\s*(?:is|:)?\s*(.+?)(?:\.|$|\n)",
            r"objective\s*(?:is|:)?\s*(.+?)(?:\.|$|\n)",
            r"aim\s*(?:is|:)?\s*(.+?)(?:\.|$|\n)",
        ]

        for pattern in goal_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                params["goal"] = match.group(1).strip()
                break

        # Extract deadline
        deadline_patterns = [
            r"(?:by|before)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"deadline\s*(?::|is)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"in\s+(\d+)\s*(days?|weeks?|months?)",
        ]

        for pattern in deadline_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if match.lastindex == 1:
                    # Direct date
                    try:
                        params["deadline"] = datetime.strptime(match.group(1), "%m/%d/%Y")
                    except ValueError:
                        pass
                else:
                    # Relative time
                    amount = int(match.group(1))
                    unit = match.group(2).lower()
                    if "day" in unit:
                        params["deadline"] = datetime.now() + timedelta(days=amount)
                    elif "week" in unit:
                        params["deadline"] = datetime.now() + timedelta(weeks=amount)
                    elif "month" in unit:
                        params["deadline"] = datetime.now() + timedelta(days=amount * 30)
                break

        return params

    def _parse_breakdown_params(self, message: str) -> Dict[str, Any]:
        """Parse task breakdown parameters"""
        params = {"task_description": message, "granularity": "medium", "max_subtasks": 10}

        # Extract granularity
        if "fine" in message.lower() or "detailed" in message.lower():
            params["granularity"] = "fine"
        elif "coarse" in message.lower() or "rough" in message.lower():
            params["granularity"] = "coarse"

        # Extract max subtasks
        num_match = re.search(r"(\d+)\s+(?:sub)?tasks?", message, re.IGNORECASE)
        if num_match:
            params["max_subtasks"] = int(num_match.group(1))

        return params

    def _parse_adjust_params(self, message: str) -> Dict[str, Any]:
        return {"plan_id": "", "changes": {}, "reason": message}

    def _parse_analyze_params(self, message: str) -> Dict[str, Any]:
        return {"plan_id": ""}

    def _parse_prioritize_params(self, message: str) -> Dict[str, Any]:
        params = {"tasks": [], "strategy": "weighted"}

        if "dependency" in message.lower():
            params["strategy"] = "dependencies"
        elif "deadline" in message.lower():
            params["strategy"] = "deadline"
        elif "effort" in message.lower() or "value" in message.lower():
            params["strategy"] = "effort_value"

        return params

    async def _generate_tasks_with_llm(
        self, title: str, description: str, goal: str, constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate tasks using LLM"""
        prompt = f"""Create a detailed task list for the following project:

Title: {title}
Description: {description}
Goal: {goal}
Constraints: {json.dumps(constraints)}

Generate 8-15 specific tasks. For each task, provide:
- title: Clear, actionable task name
- description: Brief description of what needs to be done
- duration: Estimated time (e.g., "2 hours", "1 day", "3 days")
- priority: low, medium, high, or critical
- dependencies: List of task titles this depends on (can be empty)
- tags: Relevant tags
- risks: Potential risks or blockers

Respond in JSON format as a list of tasks."""

        try:
            response = await self._generate_with_llm(prompt)
            tasks = json.loads(response)
            if isinstance(tasks, dict) and "tasks" in tasks:
                return tasks["tasks"]
            return tasks if isinstance(tasks, list) else []
        except json.JSONDecodeError as e:
            log.error(f"Error parsing LLM response: {e}")
            return self._fallback_task_generation(title, description)

    def _fallback_task_generation(self, title: str, description: str) -> List[Dict[str, Any]]:
        """Fallback task generation if LLM fails"""
        return [
            {
                "title": f"Planning and research for {title}",
                "description": "Gather requirements and research approach",
                "duration": "4 hours",
                "priority": "high",
                "dependencies": [],
            },
            {
                "title": "Design and architecture",
                "description": "Create detailed design and architecture plan",
                "duration": "1 day",
                "priority": "high",
                "dependencies": [f"Planning and research for {title}"],
            },
            {
                "title": "Implementation",
                "description": "Execute the main implementation work",
                "duration": "3 days",
                "priority": "critical",
                "dependencies": ["Design and architecture"],
            },
            {
                "title": "Testing and validation",
                "description": "Test and validate the work",
                "duration": "1 day",
                "priority": "high",
                "dependencies": ["Implementation"],
            },
            {
                "title": "Review and documentation",
                "description": "Final review and documentation",
                "duration": "4 hours",
                "priority": "medium",
                "dependencies": ["Testing and validation"],
            },
        ]

    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        self.task_counter += 1
        return f"TASK-{self.task_counter:03d}"

    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parse duration string to timedelta"""
        duration_str = duration_str.lower().strip()

        # Patterns
        patterns = [
            (r"(\d+)\s*min(?:ute)?s?", lambda m: timedelta(minutes=int(m.group(1)))),
            (r"(\d+)\s*hour?s?", lambda m: timedelta(hours=int(m.group(1)))),
            (r"(\d+)\s*day?s?", lambda m: timedelta(days=int(m.group(1)))),
            (r"(\d+)\s*week?s?", lambda m: timedelta(weeks=int(m.group(1)))),
            (r"(\d+)\s*month?s?", lambda m: timedelta(days=int(m.group(1)) * 30)),
        ]

        for pattern, converter in patterns:
            match = re.match(pattern, duration_str)
            if match:
                return converter(match)

        return timedelta(hours=2)  # Default 2 hours

    async def _create_subtasks(self, task: Task) -> List[Task]:
        """Create subtasks for a complex task"""
        return await self.breakdown_task(task.description, granularity="fine", max_subtasks=5)

    async def _identify_milestones(
        self, tasks: List[Task], project_title: str, deadline: Optional[datetime]
    ) -> List[Milestone]:
        """Identify key milestones from tasks"""
        milestones = []

        # Group tasks by logical phases
        phases = self._identify_phases(tasks)

        for i, (phase_name, phase_tasks) in enumerate(phases.items(), 1):
            milestone = Milestone(
                id=f"MS-{i:03d}",
                title=f"{phase_name} Complete",
                description=f"Completion of {phase_name} phase",
                target_date=self._calculate_milestone_date(phase_tasks, deadline),
                tasks=[t.id for t in phase_tasks],
                deliverables=[f"{t.title} completed" for t in phase_tasks],
            )
            milestones.append(milestone)

        return milestones

    def _identify_phases(self, tasks: List[Task]) -> Dict[str, List[Task]]:
        """Group tasks into phases"""
        phases: Dict[str, List[Task]] = {}

        # Simple phase detection based on task keywords
        phase_keywords = {
            "Planning": ["plan", "research", "design", "architecture"],
            "Development": ["implement", "code", "build", "create", "develop"],
            "Testing": ["test", "validate", "verify", "check"],
            "Deployment": ["deploy", "release", "launch", "publish"],
            "Documentation": ["document", "write", "review"],
        }

        for task in tasks:
            assigned = False
            task_lower = task.title.lower()

            for phase, keywords in phase_keywords.items():
                if any(kw in task_lower for kw in keywords):
                    if phase not in phases:
                        phases[phase] = []
                    phases[phase].append(task)
                    assigned = True
                    break

            if not assigned:
                if "Other" not in phases:
                    phases["Other"] = []
                phases["Other"].append(task)

        return phases

    def _calculate_milestone_date(
        self, tasks: List[Task], project_deadline: Optional[datetime]
    ) -> datetime:
        """Calculate target date for a milestone"""
        total_duration = sum((t.estimated_duration for t in tasks), timedelta())
        return datetime.now() + total_duration

    def _calculate_critical_path(self, tasks: List[Task]) -> List[str]:
        """Calculate critical path through tasks"""
        # Simplified critical path: tasks with most dependents
        task_dependents: Dict[str, int] = {t.id: 0 for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                for t in tasks:
                    if t.title == dep or t.id == dep:
                        task_dependents[t.id] += 1

        # Sort by number of dependents (descending)
        sorted_tasks = sorted(task_dependents.items(), key=lambda x: x[1], reverse=True)

        # Return top 30% as critical path
        critical_count = max(1, len(sorted_tasks) // 3)
        return [task_id for task_id, _ in sorted_tasks[:critical_count]]

    async def _assess_risks(
        self, tasks: List[Task], constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assess project risks"""
        risks = []

        # Check for tight deadline
        if constraints.get("deadline"):
            deadline = constraints["deadline"]
            if isinstance(deadline, str):
                try:
                    deadline = datetime.fromisoformat(deadline)
                except ValueError:
                    deadline = None

            if deadline:
                total_duration = self._calculate_total_duration(tasks)
                time_available = deadline - datetime.now()

                if total_duration > time_available:
                    risks.append(
                        {
                            "type": "schedule",
                            "severity": "high",
                            "description": "Timeline may be too aggressive",
                            "mitigation": "Consider scope reduction or parallel work",
                        }
                    )

        # Check for resource constraints
        if constraints.get("max_resources"):
            max_resources = constraints["max_resources"]
            concurrent_tasks = self._estimate_concurrent_tasks(tasks)

            if concurrent_tasks > max_resources:
                risks.append(
                    {
                        "type": "resource",
                        "severity": "medium",
                        "description": (
                            f"May need {concurrent_tasks} resources but only "
                            f"{max_resources} available"
                        ),
                        "mitigation": "Adjust timeline to reduce parallel work",
                    }
                )

        # Check for dependency risks
        long_chains = self._find_long_dependency_chains(tasks)
        if len(long_chains) > 0:
            risks.append(
                {
                    "type": "dependency",
                    "severity": "medium",
                    "description": f"Found {len(long_chains)} long dependency chains",
                    "mitigation": "Consider breaking down dependent tasks",
                }
            )

        return risks

    def _calculate_total_duration(self, tasks: List[Task]) -> timedelta:
        """Calculate total project duration considering dependencies"""
        # Simplified: assume 50% can be parallelized
        sequential_duration = sum((t.estimated_duration for t in tasks), timedelta())
        return timedelta(seconds=int(sequential_duration.total_seconds() * 0.6))

    def _allocate_resources(self, tasks: List[Task], resources: List[str]) -> Dict[str, List[str]]:
        """Allocate resources to tasks"""
        allocation: Dict[str, List[str]] = {}

        if not resources:
            return allocation

        # Simple round-robin allocation
        for i, task in enumerate(tasks):
            resource = resources[i % len(resources)]
            if resource not in allocation:
                allocation[resource] = []
            allocation[resource].append(task.id)

        return allocation

    def _simple_breakdown(self, task_description: str, max_subtasks: int) -> List[Task]:
        """Simple fallback task breakdown"""
        subtasks = []

        # Generic breakdown
        steps = [
            ("Preparation", "Gather necessary information and resources"),
            ("Planning", "Create detailed approach and timeline"),
            ("Execution", "Perform the main work"),
            ("Review", "Review and validate results"),
            ("Completion", "Finalize and deliver"),
        ]

        for i, (title, desc) in enumerate(steps[:max_subtasks], 1):
            task = Task(
                id=self._generate_task_id(),
                title=f"{title} - {task_description[:30]}",
                description=desc,
                estimated_duration=timedelta(hours=2),
                priority="medium",
            )
            subtasks.append(task)

        return subtasks

    def _identify_bottlenecks(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Identify bottlenecks in the task list"""
        bottlenecks = []

        # Find tasks with many dependents
        dependent_count: Dict[str, int] = {}
        for task in tasks:
            for dep in task.dependencies:
                dependent_count[dep] = dependent_count.get(dep, 0) + 1

        for task_id, count in dependent_count.items():
            if count > 2:  # More than 2 dependents
                bottlenecks.append(
                    {
                        "task_id": task_id,
                        "dependents": count,
                        "type": "high_dependency",
                    }
                )

        # Find long-duration tasks
        for task in tasks:
            if task.estimated_duration > timedelta(days=3):
                bottlenecks.append(
                    {
                        "task_id": task.id,
                        "duration_days": task.estimated_duration.days,
                        "type": "long_duration",
                    }
                )

        return bottlenecks

    def _find_resource_conflicts(self, plan: ProjectPlan) -> List[Dict[str, Any]]:
        """Find resource conflicts"""
        conflicts = []

        # Group tasks by resource
        for resource, task_ids in plan.resources.get("allocated", {}).items():
            if len(task_ids) > 3:  # Resource overloaded
                conflicts.append(
                    {
                        "resource": resource,
                        "task_count": len(task_ids),
                        "type": "overload",
                    }
                )

        return conflicts

    def _assess_schedule_risks(self, plan: ProjectPlan) -> List[Dict[str, Any]]:
        """Assess schedule-related risks"""
        risks = []

        if plan.target_end_date and plan.start_date:
            available_time = plan.target_end_date - plan.start_date

            if plan.total_estimated_duration > available_time:
                risks.append(
                    {
                        "type": "insufficient_time",
                        "shortfall_days": (plan.total_estimated_duration - available_time).days,
                        "severity": "high",
                    }
                )

        return risks

    async def _find_optimizations(self, plan: ProjectPlan) -> List[Dict[str, Any]]:
        """Find optimization opportunities"""
        optimizations = []

        # Look for parallelization opportunities
        independent_tasks = [t for t in plan.tasks if not t.dependencies]
        if len(independent_tasks) > 3:
            optimizations.append(
                {
                    "type": "parallelization",
                    "opportunity": f"{len(independent_tasks)} tasks can be worked on in parallel",
                    "potential_time_savings": "20-30%",
                }
            )

        # Look for task merging opportunities
        similar_tasks = self._find_similar_tasks(plan.tasks)
        if similar_tasks:
            optimizations.append(
                {
                    "type": "consolidation",
                    "opportunity": f"{len(similar_tasks)} pairs of similar tasks could be merged",
                    "potential_time_savings": "10-15%",
                }
            )

        return optimizations

    def _find_similar_tasks(self, tasks: List[Task]) -> List[tuple]:
        """Find similar tasks that could be merged"""
        similar = []

        for i, task1 in enumerate(tasks):
            for task2 in tasks[i + 1 :]:
                # Check for similar titles
                words1 = set(task1.title.lower().split())
                words2 = set(task2.title.lower().split())
                overlap = words1.intersection(words2)

                if len(overlap) >= 2 and len(overlap) / max(len(words1), len(words2)) > 0.3:
                    similar.append((task1.id, task2.id))

        return similar

    def _calculate_priority_score(self, task: Task) -> float:
        """Calculate priority score for a task"""
        score = 0.0

        # Priority weight
        score += self.PRIORITY_WEIGHTS.get(task.priority, 2) * 10

        # Dependency penalty (tasks with more dependencies should come first)
        score += len(task.dependencies) * 5

        # Subtask bonus (parent tasks should be prioritized)
        if task.subtasks:
            score += len(task.subtasks) * 3

        return score

    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by dependencies (topological sort)"""
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        in_degree: Dict[str, int] = {t.id: 0 for t in tasks}
        graph: Dict[str, List[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep in task_map:
                    graph[dep].append(task.id)
                    in_degree[task.id] += 1

        # Kahn's algorithm
        queue = [t_id for t_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort queue by priority
            queue.sort(
                key=lambda t_id: self._calculate_priority_score(task_map[t_id]), reverse=True
            )

            t_id = queue.pop(0)
            result.append(task_map[t_id])

            for neighbor in graph[t_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _estimate_concurrent_tasks(self, tasks: List[Task]) -> int:
        """Estimate number of tasks that can run concurrently"""
        independent = len([t for t in tasks if not t.dependencies])
        return min(independent, len(tasks) // 2)

    def _find_long_dependency_chains(self, tasks: List[Task]) -> List[List[str]]:
        """Find long chains of dependent tasks"""
        chains = []
        task_map = {t.id: t for t in tasks}

        for task in tasks:
            chain = [task.id]
            current = task

            while current.dependencies:
                next_id = current.dependencies[0]
                if next_id in task_map:
                    chain.append(next_id)
                    current = task_map[next_id]
                else:
                    break

            if len(chain) > 3:
                chains.append(chain)

        return chains

    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate content using LLM"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response_chunks = []

            async for chunk in self.llm.chat(messages=messages, system=self.system_prompt):
                if "message" in chunk and "content" in chunk["message"]:
                    response_chunks.append(chunk["message"]["content"])

            return "".join(response_chunks)

        except Exception as e:
            log.error(f"Error generating with LLM: {e}")
            return "[]"
