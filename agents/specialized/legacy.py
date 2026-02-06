"""
Legacy specialized agents (backward compatibility).

These are the original agent implementations preserved for compatibility.
New code should use the enhanced agents from the main specialized module.
"""

from __future__ import annotations

import json
import logging

from agents.base import AgentContext, AgentRole, BaseAgent
from core.llm import OllamaClient
from core.rag import RAGEngine
from tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class LegacyResearchAgent(BaseAgent):
    """Original research agent (legacy)"""

    name = "research_agent"
    role = AgentRole.RESEARCH
    description = "Handles web searches, document analysis, and information retrieval"
    system_prompt = """You are a research assistant. Your job is to:
1. Search the web for information when needed
2. Analyze documents and extract relevant information
3. Synthesize findings into clear, concise answers

Use available tools to search and fetch information. Always cite sources."""

    def __init__(self, llm: OllamaClient, tools: ToolRegistry, rag: RAGEngine | None = None):
        super().__init__()
        self.llm = llm
        self.tools = tools
        self.rag = rag
        self._research_keywords = {
            "search",
            "find",
            "look up",
            "research",
            "what is",
            "who is",
            "when",
            "where",
            "how",
            "why",
            "explain",
            "define",
            "information",
        }

    async def can_handle(self, message: str) -> float:
        lower = message.lower()
        score = 0.0
        for keyword in self._research_keywords:
            if keyword in lower:
                score += 0.2
        return min(score, 1.0)

    async def process(self, message: str, context: AgentContext | None = None) -> str:
        rag_context = ""
        if self.rag:
            rag_context = self.rag.get_context(message, limit=3)

        messages = [{"role": "user", "content": message}]
        if rag_context:
            messages[0]["content"] = f"Context:\n{rag_context}\n\nQuestion: {message}"

        full_response = ""
        async for chunk in self.llm.chat(
            messages=messages,
            system=self.system_prompt,
            tools=self._get_tool_schemas(),
        ):
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                full_response += content
                if tool_calls := chunk["message"].get("tool_calls"):
                    tool_results = await self._execute_tools(tool_calls)
                    full_response += f"\n{tool_results}"

        return full_response

    def _get_tool_schemas(self) -> list[dict]:
        allowed = ["web_search", "fetch_url"]
        return [t for t in self.tools.get_schemas() if t["function"]["name"] in allowed]

    async def _execute_tools(self, tool_calls: list[dict]) -> str:
        results = []
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            result = await self.tools.execute(name, **args)
            results.append(f"[{name}]: {result.data}")
        return "\n".join(results)


class LegacyCodeAgent(BaseAgent):
    """Original code agent (legacy)"""

    name = "code_agent"
    role = AgentRole.CODE
    description = "Handles code writing, review, debugging, and execution"
    system_prompt = """You are a coding assistant. Your job is to:
1. Write clean, efficient code
2. Debug and fix code issues
3. Explain code concepts
4. Execute code when needed

Follow best practices and explain your reasoning."""

    def __init__(self, llm: OllamaClient, tools: ToolRegistry):
        super().__init__()
        self.llm = llm
        self.tools = tools
        self._code_keywords = {
            "code",
            "program",
            "function",
            "class",
            "debug",
            "fix",
            "error",
            "python",
            "javascript",
            "rust",
            "compile",
            "run",
            "execute",
            "script",
            "algorithm",
            "implement",
            "refactor",
        }

    async def can_handle(self, message: str) -> float:
        lower = message.lower()
        score = 0.0
        for keyword in self._code_keywords:
            if keyword in lower:
                score += 0.25
        if "```" in message:
            score += 0.3
        return min(score, 1.0)

    async def process(self, message: str, context: AgentContext | None = None) -> str:
        messages = [{"role": "user", "content": message}]
        full_response = ""

        async for chunk in self.llm.chat(
            messages=messages,
            system=self.system_prompt,
            tools=self._get_tool_schemas(),
        ):
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                full_response += content
                if tool_calls := chunk["message"].get("tool_calls"):
                    tool_results = await self._execute_tools(tool_calls)
                    full_response += f"\n\nExecution result:\n{tool_results}"

        return full_response

    def _get_tool_schemas(self) -> list[dict]:
        allowed = ["execute_python", "execute_shell", "read_file", "write_file"]
        return [t for t in self.tools.get_schemas() if t["function"]["name"] in allowed]

    async def _execute_tools(self, tool_calls: list[dict]) -> str:
        results = []
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            result = await self.tools.execute(name, **args)
            if result.success:
                results.append(str(result.data))
            else:
                results.append(f"Error: {result.error}")
        return "\n".join(results)


class LegacyTaskAgent(BaseAgent):
    """Original task agent (legacy)"""

    name = "task_agent"
    role = AgentRole.TASK
    description = "Breaks down complex tasks into subtasks and coordinates execution"
    system_prompt = """You are a task planning assistant. Your job is to:
1. Break down complex requests into manageable steps
2. Identify which tools or agents are needed for each step
3. Execute steps in order
4. Synthesize results

Think step-by-step and be thorough."""

    def __init__(self, llm: OllamaClient, tools: ToolRegistry):
        super().__init__()
        self.llm = llm
        self.tools = tools

    async def can_handle(self, message: str) -> float:
        lower = message.lower()
        complexity_indicators = [
            "and then",
            "after that",
            "first",
            "next",
            "finally",
            "step by step",
            "multiple",
            "several",
            "all",
        ]
        score = 0.1
        for indicator in complexity_indicators:
            if indicator in lower:
                score += 0.2
        if len(message.split()) > 30:
            score += 0.2
        return min(score, 1.0)

    async def process(self, message: str, context: AgentContext | None = None) -> str:
        planning_prompt = f"""Break down this request into steps:
{message}

Respond with a numbered list of steps."""

        messages = [{"role": "user", "content": planning_prompt}]
        plan = ""
        async for chunk in self.llm.chat(messages=messages, system=self.system_prompt):
            if "message" in chunk:
                plan += chunk["message"].get("content", "")

        execution_prompt = f"""Original request: {message}

Plan:
{plan}

Now execute each step and provide the final result."""

        messages = [{"role": "user", "content": execution_prompt}]
        result = ""
        async for chunk in self.llm.chat(
            messages=messages,
            system=self.system_prompt,
            tools=self.tools.get_schemas(),
        ):
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                result += content
                if tool_calls := chunk["message"].get("tool_calls"):
                    tool_results = await self._execute_tools(tool_calls)
                    result += f"\n{tool_results}"

        return result

    async def _execute_tools(self, tool_calls: list[dict]) -> str:
        results = []
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            result = await self.tools.execute(name, **args)
            results.append(f"[{name}]: {result.data if result.success else result.error}")
        return "\n".join(results)


class LegacyMemoryAgent(BaseAgent):
    """Original memory agent (legacy)"""

    name = "memory_agent"
    role = AgentRole.MEMORY
    description = "Manages conversation history and knowledge retrieval"
    system_prompt = """You are a memory assistant. Your job is to:
1. Store important facts and information
2. Retrieve relevant past conversations
3. Provide context from memory when needed

Be precise about what you remember and admit when you don't know."""

    def __init__(self, llm: OllamaClient, rag: RAGEngine | None = None):
        super().__init__()
        self.llm = llm
        self.rag = rag
        self._memory_keywords = {
            "remember",
            "recall",
            "forgot",
            "memory",
            "stored",
            "save",
            "last time",
            "previously",
            "earlier",
            "before",
        }

    async def can_handle(self, message: str) -> float:
        lower = message.lower()
        score = 0.0
        for keyword in self._memory_keywords:
            if keyword in lower:
                score += 0.3
        return min(score, 1.0)

    async def process(self, message: str, context: AgentContext | None = None) -> str:
        memory_context = ""

        if self.rag:
            memory_context = self.rag.get_context(message, limit=5)

        prompt = message
        if memory_context:
            prompt = f"Relevant memories:\n{memory_context}\n\nUser: {message}"

        messages = [{"role": "user", "content": prompt}]
        response = ""
        async for chunk in self.llm.chat(messages=messages, system=self.system_prompt):
            if "message" in chunk:
                response += chunk["message"].get("content", "")

        return response
