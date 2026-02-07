"""
Procedural Memory System for JARVIS.

Stores learned skills, procedures, and how-to knowledge.
Enables the assistant to perform tasks and follow procedures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

log = logging.getLogger(__name__)


class SkillLevel(Enum):
    """Skill proficiency levels"""

    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ProcedureStatus(Enum):
    """Status of a procedure execution"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Step:
    """A single step in a procedure"""

    id: str
    description: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[str] = None
    error_handling: str = "abort"
    timeout_seconds: int = 60
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Procedure:
    """A learned procedure or process"""

    id: str
    name: str
    description: str
    domain: str
    steps: List[Step]
    prerequisites: List[str] = field(default_factory=list)
    expected_duration: Optional[timedelta] = None
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "action": s.action,
                    "parameters": s.parameters,
                    "validation": s.validation,
                    "error_handling": s.error_handling,
                    "timeout_seconds": s.timeout_seconds,
                    "depends_on": s.depends_on,
                }
                for s in self.steps
            ],
            "prerequisites": self.prerequisites,
            "expected_duration_seconds": self.expected_duration.total_seconds()
            if self.expected_duration
            else None,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "tags": self.tags,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Procedure":
        steps = [
            Step(
                id=s["id"],
                description=s["description"],
                action=s["action"],
                parameters=s.get("parameters", {}),
                validation=s.get("validation"),
                error_handling=s.get("error_handling", "abort"),
                timeout_seconds=s.get("timeout_seconds", 60),
                depends_on=s.get("depends_on", []),
            )
            for s in data.get("steps", [])
        ]

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            domain=data["domain"],
            steps=steps,
            prerequisites=data.get("prerequisites", []),
            expected_duration=timedelta(seconds=data["expected_duration_seconds"])
            if data.get("expected_duration_seconds")
            else None,
            success_rate=data.get("success_rate", 0.0),
            usage_count=data.get("usage_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            tags=data.get("tags", []),
            version=data.get("version", 1),
        )


@dataclass
class Skill:
    """A learned skill with proficiency tracking"""

    id: str
    name: str
    description: str
    domain: str
    level: SkillLevel
    confidence: float = 0.0
    practice_count: int = 0
    success_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_practiced: Optional[datetime] = None
    related_procedures: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "level": self.level.value,
            "confidence": self.confidence,
            "practice_count": self.practice_count,
            "success_count": self.success_count,
            "created_at": self.created_at.isoformat(),
            "last_practiced": self.last_practiced.isoformat() if self.last_practiced else None,
            "related_procedures": self.related_procedures,
            "tags": self.tags,
        }

    def update_proficiency(self, success: bool):
        """Update skill proficiency based on practice"""
        self.practice_count += 1
        if success:
            self.success_count += 1

        # Calculate confidence
        self.confidence = (
            self.success_count / self.practice_count if self.practice_count > 0 else 0.0
        )

        # Update level based on practice and success
        if self.practice_count >= 100 and self.confidence >= 0.9:
            self.level = SkillLevel.EXPERT
        elif self.practice_count >= 50 and self.confidence >= 0.8:
            self.level = SkillLevel.ADVANCED
        elif self.practice_count >= 20 and self.confidence >= 0.7:
            self.level = SkillLevel.INTERMEDIATE
        elif self.practice_count >= 5 and self.confidence >= 0.6:
            self.level = SkillLevel.BEGINNER

        self.last_practiced = datetime.now()


@dataclass
class ExecutionResult:
    """Result of executing a procedure"""

    success: bool
    procedure_id: str
    completed_steps: int
    total_steps: int
    output: str
    error: Optional[str] = None
    duration: Optional[timedelta] = None
    step_results: List[Dict[str, Any]] = field(default_factory=list)


class ProceduralMemory:
    """
    Procedural memory system for storing and executing learned procedures.

    Features:
    - Store and retrieve procedures
    - Track skill proficiency
    - Execute procedures with error handling
    - Learn from successful executions
    - Chain procedures together
    """

    def __init__(self, storage_path: str = "data/procedural_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.procedures: Dict[str, Procedure] = {}
        self.skills: Dict[str, Skill] = {}

        # Indexes
        self._procedures_by_domain: Dict[str, Set[str]] = {}
        self._procedures_by_tag: Dict[str, Set[str]] = {}
        self._skills_by_domain: Dict[str, Set[str]] = {}

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the procedural memory system"""
        if self._initialized:
            return

        await self._load_data()
        self._initialized = True
        log.info(
            f"Procedural memory initialized with {len(self.procedures)} procedures, {len(self.skills)} skills"
        )

    async def _load_data(self):
        """Load procedures and skills from storage"""
        procedures_file = self.storage_path / "procedures.json"
        skills_file = self.storage_path / "skills.json"

        if procedures_file.exists():
            try:
                with open(procedures_file, "r") as f:
                    data = json.load(f)

                for proc_data in data.get("procedures", []):
                    procedure = Procedure.from_dict(proc_data)
                    self.procedures[procedure.id] = procedure
                    self._index_procedure(procedure)

            except Exception as e:
                log.error(f"Error loading procedures: {e}")

        if skills_file.exists():
            try:
                with open(skills_file, "r") as f:
                    data = json.load(f)

                for skill_data in data.get("skills", []):
                    skill = Skill(
                        id=skill_data["id"],
                        name=skill_data["name"],
                        description=skill_data["description"],
                        domain=skill_data["domain"],
                        level=SkillLevel(skill_data["level"]),
                        confidence=skill_data.get("confidence", 0.0),
                        practice_count=skill_data.get("practice_count", 0),
                        success_count=skill_data.get("success_count", 0),
                        created_at=datetime.fromisoformat(skill_data["created_at"]),
                        last_practiced=datetime.fromisoformat(skill_data["last_practiced"])
                        if skill_data.get("last_practiced")
                        else None,
                        related_procedures=skill_data.get("related_procedures", []),
                        tags=skill_data.get("tags", []),
                    )
                    self.skills[skill.id] = skill
                    self._index_skill(skill)

            except Exception as e:
                log.error(f"Error loading skills: {e}")

    async def _save_data(self):
        """Save procedures and skills to storage"""
        procedures_file = self.storage_path / "procedures.json"
        skills_file = self.storage_path / "skills.json"

        with open(procedures_file, "w") as f:
            json.dump(
                {"procedures": [p.to_dict() for p in self.procedures.values()]},
                f,
                indent=2,
            )

        with open(skills_file, "w") as f:
            json.dump(
                {"skills": [s.to_dict() for s in self.skills.values()]},
                f,
                indent=2,
            )

    def _index_procedure(self, procedure: Procedure):
        """Index a procedure"""
        if procedure.domain not in self._procedures_by_domain:
            self._procedures_by_domain[procedure.domain] = set()
        self._procedures_by_domain[procedure.domain].add(procedure.id)

        for tag in procedure.tags:
            if tag not in self._procedures_by_tag:
                self._procedures_by_tag[tag] = set()
            self._procedures_by_tag[tag].add(procedure.id)

    def _index_skill(self, skill: Skill):
        """Index a skill"""
        if skill.domain not in self._skills_by_domain:
            self._skills_by_domain[skill.domain] = set()
        self._skills_by_domain[skill.domain].add(skill.id)

    async def learn_procedure(
        self,
        name: str,
        description: str,
        domain: str,
        steps: List[Step],
        prerequisites: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Procedure:
        """Learn a new procedure"""
        async with self._lock:
            procedure_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            procedure = Procedure(
                id=procedure_id,
                name=name,
                description=description,
                domain=domain,
                steps=steps,
                prerequisites=prerequisites or [],
                tags=tags or [],
            )

            self.procedures[procedure_id] = procedure
            self._index_procedure(procedure)

            await self._save_data()

            log.info(f"Learned procedure: {name}")
            return procedure

    async def learn_skill(
        self,
        name: str,
        description: str,
        domain: str,
        level: SkillLevel = SkillLevel.NOVICE,
        related_procedures: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Skill:
        """Learn a new skill"""
        async with self._lock:
            skill_id = f"skill_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            skill = Skill(
                id=skill_id,
                name=name,
                description=description,
                domain=domain,
                level=level,
                related_procedures=related_procedures or [],
                tags=tags or [],
            )

            self.skills[skill_id] = skill
            self._index_skill(skill)

            await self._save_data()

            log.info(f"Learned skill: {name}")
            return skill

    async def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """Get a procedure by ID"""
        return self.procedures.get(procedure_id)

    async def find_procedure(
        self,
        query: str,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Procedure]:
        """Find a procedure by query string"""
        query_lower = query.lower()

        candidates = []

        for procedure in self.procedures.values():
            if domain and procedure.domain != domain:
                continue

            if tags and not any(tag in procedure.tags for tag in tags):
                continue

            # Calculate relevance score
            score = 0.0

            if query_lower in procedure.name.lower():
                score += 10.0

            if query_lower in procedure.description.lower():
                score += 5.0

            for tag in procedure.tags:
                if query_lower in tag.lower():
                    score += 3.0

            if score > 0:
                candidates.append((score, procedure))

        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            return candidates[0][1]

        return None

    async def list_procedures(
        self,
        domain: Optional[str] = None,
        tag: Optional[str] = None,
        min_success_rate: float = 0.0,
    ) -> List[Procedure]:
        """List procedures with optional filtering"""
        results = []

        if domain and domain in self._procedures_by_domain:
            procedure_ids = self._procedures_by_domain[domain]
        elif tag and tag in self._procedures_by_tag:
            procedure_ids = self._procedures_by_tag[tag]
        else:
            procedure_ids = set(self.procedures.keys())

        for proc_id in procedure_ids:
            procedure = self.procedures.get(proc_id)
            if procedure and procedure.success_rate >= min_success_rate:
                results.append(procedure)

        # Sort by usage count and success rate
        results.sort(key=lambda p: (p.usage_count * p.success_rate), reverse=True)

        return results

    async def execute_procedure(
        self,
        procedure_id: str,
        context: Optional[Dict[str, Any]] = None,
        step_handlers: Optional[Dict[str, Callable]] = None,
    ) -> ExecutionResult:
        """Execute a procedure"""
        procedure = self.procedures.get(procedure_id)
        if not procedure:
            return ExecutionResult(
                success=False,
                procedure_id=procedure_id,
                completed_steps=0,
                total_steps=0,
                output="",
                error="Procedure not found",
            )

        start_time = datetime.now()
        completed_steps = 0
        step_results = []

        try:
            # Check prerequisites
            for prereq in procedure.prerequisites:
                if prereq not in self.procedures:
                    raise ValueError(f"Prerequisite not met: {prereq}")

            # Execute steps in order
            executed_steps = set()
            pending_steps = procedure.steps.copy()

            while pending_steps:
                # Find next executable step
                next_step = None
                for step in pending_steps:
                    # Check if dependencies are satisfied
                    if all(dep in executed_steps for dep in step.depends_on):
                        next_step = step
                        break

                if not next_step:
                    raise ValueError("Unable to resolve step dependencies")

                # Execute the step
                step_result = await self._execute_step(
                    next_step, context or {}, step_handlers or {}
                )

                step_results.append(step_result)
                executed_steps.add(next_step.id)
                pending_steps.remove(next_step)

                if step_result.get("success"):
                    completed_steps += 1
                else:
                    # Handle step failure
                    if next_step.error_handling == "abort":
                        raise ValueError(f"Step {next_step.id} failed: {step_result.get('error')}")
                    elif next_step.error_handling == "skip":
                        continue
                    elif next_step.error_handling == "retry":
                        # Retry the step
                        await asyncio.sleep(1)
                        step_result = await self._execute_step(
                            next_step, context or {}, step_handlers or {}
                        )
                        if not step_result.get("success"):
                            raise ValueError(f"Step {next_step.id} failed after retry")

            # Update procedure stats
            duration = datetime.now() - start_time
            procedure.usage_count += 1
            procedure.last_used = datetime.now()

            # Calculate new success rate
            successful_executions = int(procedure.success_rate * (procedure.usage_count - 1))
            successful_executions += 1
            procedure.success_rate = successful_executions / procedure.usage_count

            await self._save_data()

            return ExecutionResult(
                success=True,
                procedure_id=procedure_id,
                completed_steps=completed_steps,
                total_steps=len(procedure.steps),
                output=f"Procedure completed successfully in {duration}",
                duration=duration,
                step_results=step_results,
            )

        except Exception as e:
            duration = datetime.now() - start_time

            # Update failure stats
            procedure.usage_count += 1
            successful_executions = int(procedure.success_rate * (procedure.usage_count - 1))
            procedure.success_rate = successful_executions / procedure.usage_count

            await self._save_data()

            return ExecutionResult(
                success=False,
                procedure_id=procedure_id,
                completed_steps=completed_steps,
                total_steps=len(procedure.steps),
                output="",
                error=str(e),
                duration=duration,
                step_results=step_results,
            )

    async def _execute_step(
        self,
        step: Step,
        context: Dict[str, Any],
        handlers: Dict[str, Callable],
    ) -> Dict[str, Any]:
        """Execute a single step"""
        try:
            # Get the handler for this action
            handler = handlers.get(step.action)
            if not handler:
                return {
                    "success": False,
                    "error": f"No handler for action: {step.action}",
                    "step_id": step.id,
                }

            # Prepare parameters
            params = self._substitute_parameters(step.parameters, context)

            # Execute with timeout
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**params),
                    timeout=step.timeout_seconds,
                )
            else:
                result = handler(**params)

            # Validate result if validation is specified
            if step.validation and not self._validate_result(result, step.validation):
                return {
                    "success": False,
                    "error": "Validation failed",
                    "step_id": step.id,
                    "result": result,
                }

            return {
                "success": True,
                "step_id": step.id,
                "result": result,
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Step timed out after {step.timeout_seconds}s",
                "step_id": step.id,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step_id": step.id,
            }

    def _substitute_parameters(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Substitute context variables in parameters"""
        result = {}

        for key, value in parameters.items():
            if isinstance(value, str):
                # Look for {{variable}} patterns
                pattern = r"\{\{(\w+)\}\}"
                matches = re.findall(pattern, value)

                substituted = value
                for match in matches:
                    if match in context:
                        substituted = substituted.replace(f"{{{{{match}}}}}", str(context[match]))

                result[key] = substituted
            elif isinstance(value, dict):
                result[key] = self._substitute_parameters(value, context)
            elif isinstance(value, list):
                result[key] = [
                    self._substitute_parameters({"_": item}, context)["_"]
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def _validate_result(self, result: Any, validation: str) -> bool:
        """Validate step result"""
        try:
            # Simple expression evaluation
            if validation.startswith("equals:"):
                expected = validation.split(":", 1)[1].strip()
                return str(result) == expected
            elif validation.startswith("contains:"):
                expected = validation.split(":", 1)[1].strip()
                return expected in str(result)
            elif validation.startswith("not_empty"):
                return bool(result)
            elif validation.startswith("is_true"):
                return bool(result)

            return True
        except Exception:
            return False

    async def practice_skill(
        self,
        skill_id: str,
        success: bool,
    ) -> Optional[Skill]:
        """Record practice of a skill"""
        skill = self.skills.get(skill_id)
        if not skill:
            return None

        skill.update_proficiency(success)
        await self._save_data()

        return skill

    async def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID"""
        return self.skills.get(skill_id)

    async def find_skill(self, name: str) -> Optional[Skill]:
        """Find a skill by name"""
        name_lower = name.lower()

        for skill in self.skills.values():
            if name_lower in skill.name.lower():
                return skill

        return None

    async def list_skills(
        self,
        domain: Optional[str] = None,
        min_level: Optional[SkillLevel] = None,
    ) -> List[Skill]:
        """List skills with optional filtering"""
        results = []

        if domain and domain in self._skills_by_domain:
            skill_ids = self._skills_by_domain[domain]
        else:
            skill_ids = set(self.skills.keys())

        for skill_id in skill_ids:
            skill = self.skills.get(skill_id)
            if skill:
                if min_level and self._level_value(skill.level) < self._level_value(min_level):
                    continue
                results.append(skill)

        # Sort by level and confidence
        results.sort(key=lambda s: (self._level_value(s.level), s.confidence), reverse=True)

        return results

    def _level_value(self, level: SkillLevel) -> int:
        """Get numeric value for skill level"""
        values = {
            SkillLevel.NOVICE: 1,
            SkillLevel.BEGINNER: 2,
            SkillLevel.INTERMEDIATE: 3,
            SkillLevel.ADVANCED: 4,
            SkillLevel.EXPERT: 5,
        }
        return values.get(level, 0)

    async def chain_procedures(
        self,
        procedure_ids: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExecutionResult]:
        """Chain multiple procedures together"""
        results = []
        current_context = context or {}

        for proc_id in procedure_ids:
            result = await self.execute_procedure(proc_id, current_context)
            results.append(result)

            if not result.success:
                break

            # Update context with results
            current_context[f"{proc_id}_result"] = result.output

        return results

    async def optimize_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """Optimize a procedure based on execution history"""
        procedure = self.procedures.get(procedure_id)
        if not procedure or procedure.usage_count < 5:
            return None

        # Identify bottlenecks (steps that often fail or timeout)
        # This is a simplified version - in practice would analyze step_results

        # Calculate expected duration based on actual performance
        # For now, just return the procedure as-is

        return procedure

    async def get_stats(self) -> Dict[str, Any]:
        """Get procedural memory statistics"""
        return {
            "procedure_count": len(self.procedures),
            "skill_count": len(self.skills),
            "procedures_by_domain": {d: len(s) for d, s in self._procedures_by_domain.items()},
            "skills_by_domain": {d: len(s) for d, s in self._skills_by_domain.items()},
            "avg_procedure_success_rate": sum(p.success_rate for p in self.procedures.values())
            / len(self.procedures)
            if self.procedures
            else 0,
            "most_used_procedure": max(self.procedures.values(), key=lambda p: p.usage_count).name
            if self.procedures
            else None,
            "highest_skill": max(
                self.skills.values(), key=lambda s: (self._level_value(s.level), s.confidence)
            ).name
            if self.skills
            else None,
        }

    async def close(self):
        """Close the procedural memory system"""
        await self._save_data()
        log.info("Procedural memory system closed")


# Global instance
_procedural_memory: Optional[ProceduralMemory] = None


async def get_procedural_memory() -> ProceduralMemory:
    """Get the global procedural memory instance"""
    global _procedural_memory
    if _procedural_memory is None:
        _procedural_memory = ProceduralMemory()
        await _procedural_memory.initialize()
    return _procedural_memory


__all__ = [
    "ProceduralMemory",
    "Procedure",
    "Step",
    "Skill",
    "SkillLevel",
    "ExecutionResult",
    "get_procedural_memory",
]
