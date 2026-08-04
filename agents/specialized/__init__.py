"""
Specialized agents for JARVIS agent framework.

This module provides domain-specific agents for various tasks:
- ResearchAgent: Information gathering and synthesis
- CodeAgent: Code writing, review, and debugging
- TaskAgent: Task planning and execution
- MemoryAgent: Memory management and retrieval
- CodeReviewAgent: Comprehensive code review and analysis
- ResearchAgent (Enhanced): Advanced research with fact verification
- CreativeAgent: Creative writing and content generation
- PlanningAgent: Project planning and management
"""

from __future__ import annotations

from agents.specialized.code_review import CodeIssue, CodeReviewAgent, ReviewResult
from agents.specialized.creative import (
    BrainstormResult,
    CreativeAgent,
    CreativeCritique,
    CreativePiece,
)
from agents.specialized.legacy import (
    LegacyCodeAgent as _LegacyCodeAgent,
)
from agents.specialized.legacy import (
    LegacyMemoryAgent as _LegacyMemoryAgent,
)

# Legacy agents (keeping for backward compatibility)
from agents.specialized.legacy import (
    LegacyResearchAgent as _LegacyResearchAgent,
)
from agents.specialized.legacy import (
    LegacyTaskAgent as _LegacyTaskAgent,
)
from agents.specialized.planning import Milestone, PlanAdjustment, PlanningAgent, ProjectPlan, Task
from agents.specialized.research import (
    ResearchAgent,
    ResearchFinding,
    ResearchReport,
    ResearchSource,
)

# Export the enhanced versions as primary
ResearchAgent = ResearchAgent
CodeReviewAgent = CodeReviewAgent
CreativeAgent = CreativeAgent
PlanningAgent = PlanningAgent

# Legacy aliases for backward compatibility
LegacyResearchAgent = _LegacyResearchAgent
LegacyCodeAgent = _LegacyCodeAgent
LegacyTaskAgent = _LegacyTaskAgent
LegacyMemoryAgent = _LegacyMemoryAgent

__all__ = [
    # New enhanced agents
    "ResearchAgent",
    "CodeReviewAgent",
    "CreativeAgent",
    "PlanningAgent",
    # Data classes
    "ResearchSource",
    "ResearchFinding",
    "ResearchReport",
    "CodeIssue",
    "ReviewResult",
    "CreativePiece",
    "CreativeCritique",
    "BrainstormResult",
    "Task",
    "Milestone",
    "ProjectPlan",
    "PlanAdjustment",
    # Legacy agents
    "LegacyResearchAgent",
    "LegacyCodeAgent",
    "LegacyTaskAgent",
    "LegacyMemoryAgent",
]
