from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCH = "research"
    CODE = "code"
    TASK = "task"
    MEMORY = "memory"


@dataclass
class AgentMessage:
    role: str
    content: str
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    id: str
    description: str
    agent_type: AgentRole
    status: str = "pending"
    result: Any = None
    error: str | None = None
    subtasks: list[AgentTask] = field(default_factory=list)


@dataclass
class AgentContext:
    conversation_id: int | None = None
    messages: list[AgentMessage] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    rag_context: str | None = None
    tools_available: list[str] = field(default_factory=list)


class BaseAgent(ABC):
    name: str
    role: AgentRole
    description: str
    system_prompt: str

    def __init__(self):
        self.context: AgentContext = AgentContext()

    @abstractmethod
    async def process(self, message: str, context: AgentContext | None = None) -> str:
        pass

    @abstractmethod
    async def can_handle(self, message: str) -> float:
        pass

    def update_context(self, context: AgentContext) -> None:
        self.context = context
