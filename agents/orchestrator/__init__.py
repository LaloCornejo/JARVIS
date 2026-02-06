"""
Agent Orchestrator for JARVIS.

This module provides two orchestrator implementations:
1. Orchestrator (Legacy) - Simple agent routing
2. AgentOrchestrator (Enhanced) - Advanced multi-agent collaboration
"""

from __future__ import annotations

from typing import Any

from agents.base import AgentContext, AgentMessage, AgentRole, BaseAgent
from agents.specialized import CodeReviewAgent, CreativeAgent, PlanningAgent, ResearchAgent
from agents.specialized.legacy import (
    LegacyCodeAgent as CodeAgent,
    LegacyMemoryAgent as MemoryAgent,
    LegacyResearchAgent as _LegacyResearchAgent,
    LegacyTaskAgent as TaskAgent,
)
from core.llm import OllamaClient
from core.memory import ConversationMemory
from core.rag import RAGEngine
from tools.registry import ToolRegistry, get_tool_registry

from .advanced import (
    AgentCollaboration,
    AgentOrchestrator,
    RoutingDecision,
    get_orchestrator,
    reset_orchestrator,
)

# Export both implementations
__all__ = [
    # Legacy orchestrator
    "Orchestrator",
    # Enhanced orchestrator
    "AgentOrchestrator",
    "AgentCollaboration",
    "RoutingDecision",
    "get_orchestrator",
    "reset_orchestrator",
]


class Orchestrator:
    """
    Legacy orchestrator with basic agent routing.

    This is the original implementation for backward compatibility.
    For new code, consider using AgentOrchestrator for advanced features.
    """

    def __init__(
        self,
        llm: OllamaClient | None = None,
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        rag: RAGEngine | None = None,
    ):
        self.llm = llm or OllamaClient()
        self.tools = tools or get_tool_registry()
        self.memory = memory or ConversationMemory()
        self.rag = rag

        # Use legacy agents for backward compatibility
        self.agents: dict[AgentRole, BaseAgent] = {
            AgentRole.RESEARCH: _LegacyResearchAgent(self.llm, self.tools, self.rag),
            AgentRole.CODE: CodeAgent(self.llm, self.tools),
            AgentRole.TASK: TaskAgent(self.llm, self.tools),
            AgentRole.MEMORY: MemoryAgent(self.llm, self.rag),
        }

        self.context = AgentContext(tools_available=self.tools.list_tools())
        self.conversation_id: int | None = None
        self._coordinator_prompt = """You are JARVIS, an AI assistant. You coordinate agents:
- Research Agent: for web searches and information retrieval
- Code Agent: for programming tasks
- Task Agent: for complex multi-step tasks
- Memory Agent: for recalling past conversations

Analyze user requests and either:
1. Handle simple requests directly
2. Delegate to the appropriate agent for complex tasks

Be concise, helpful, and proactive."""

    async def start_conversation(self, session_id: str | None = None) -> int:
        self.conversation_id = self.memory.create_conversation(session_id)
        self.context.conversation_id = self.conversation_id
        return self.conversation_id

    async def process(self, message: str) -> str:
        if self.conversation_id is None:
            await self.start_conversation()

        self.memory.add_message(self.conversation_id, "user", message)
        self.context.messages.append(AgentMessage(role="user", content=message))

        if self.rag:
            self.context.rag_context = self.rag.get_context(message, limit=2)

        agent_scores = await self._score_agents(message)
        best_agent, score = max(agent_scores.items(), key=lambda x: x[1])

        if score > 0.5:
            response = await self._delegate_to_agent(best_agent, message)
        else:
            response = await self._handle_directly(message)

        self.memory.add_message(self.conversation_id, "assistant", response)
        self.context.messages.append(AgentMessage(role="assistant", content=response))

        return response

    async def _score_agents(self, message: str) -> dict[AgentRole, float]:
        scores = {}
        for role, agent in self.agents.items():
            scores[role] = await agent.can_handle(message)
        return scores

    async def _delegate_to_agent(self, role: AgentRole, message: str) -> str:
        agent = self.agents[role]
        agent.update_context(self.context)
        return await agent.process(message, self.context)

    async def _handle_directly(self, message: str) -> str:
        messages = self._build_messages(message)
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get LLM response
            response = await self.llm.chat_completion(
                messages=messages,
                system=self._coordinator_prompt,
                tools=self.tools.get_schemas(),
            )

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # If no tool calls in structured response, check for <tool_call> in content
            if not tool_calls:
                tool_calls = self._parse_tool_calls_from_text(content)

            if tool_calls:
                # Execute tools
                tool_results = await self._execute_tools(tool_calls)
                # Append tool results as assistant message
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Tool results:\n{tool_results}"})
            else:
                # No more tool calls, return the final response
                return content.strip()

        # If max iterations reached, return last content
        return content.strip()

    def _build_messages(self, current_message: str) -> list[dict[str, str]]:
        messages = []
        history = self.memory.get_conversation_context(self.conversation_id, max_messages=10)
        messages.extend(history[:-1] if history else [])

        if self.context.rag_context:
            current_message = f"Context:\n{self.context.rag_context}\n\n{current_message}"

        messages.append({"role": "user", "content": current_message})
        return messages

    async def _execute_tools(self, tool_calls: list[dict]) -> str:
        import json

        results = []
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            result = await self.tools.execute(name, **args)
            if result.success:
                results.append(f"{result.data}")
            else:
                results.append(f"Error: {result.error}")
        return "\n".join(results)

    def _parse_tool_calls_from_text(self, content: str) -> list[dict]:
        """Parse tool calls embedded in text"""
        import re

        tool_calls = []
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            try:
                import json

                call = json.loads(match.strip())
                if isinstance(call, dict) and "function" in call:
                    tool_calls.append(call)
            except json.JSONDecodeError:
                continue

        return tool_calls

    async def store_fact(self, key: str, value: Any, category: str | None = None) -> None:
        self.memory.store_fact(key, value, category)

    async def recall_fact(self, key: str) -> Any:
        return self.memory.get_fact(key)

    async def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        if self.rag:
            return self.rag.search(query, limit=limit)
        return self.memory.search_facts(query)

    async def close(self) -> None:
        await self.llm.close()
