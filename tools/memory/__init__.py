from __future__ import annotations

from typing import Any

from core.memory.vector import get_vector_memory
from tools.base import BaseTool, ToolResult


class StoreMemoryTool(BaseTool):
    name = "store_memory"
    description = (
        "Store a fact or piece of information about the user to remember for future "
        "conversations. Use this when the user shares personal info, preferences, or details."
    )
    parameters = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact or information to remember",
            },
            "category": {
                "type": "string",
                "description": "Category for the fact",
                "enum": [
                    "personal",
                    "preference",
                    "project",
                    "technical",
                    "location",
                    "work",
                    "general",
                ],
            },
        },
        "required": ["fact"],
    }

    async def execute(self, fact: str, category: str = "general", **kwargs: Any) -> ToolResult:
        try:
            memory = get_vector_memory()
            memory_id = memory.add(
                text=fact,
                category=category,
                metadata={"source": "tool_store"},
            )
            return ToolResult(
                success=True,
                data={"id": memory_id, "message": f"Stored: {fact[:50]}..."},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class RecallMemoryTool(BaseTool):
    name = "recall_memory"
    description = (
        "Search for relevant memories about the user. Use this when you need to remember "
        "something about the user or their preferences."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memories",
            },
            "category": {
                "type": "string",
                "description": "Optionally filter by category",
                "enum": [
                    "personal",
                    "preference",
                    "project",
                    "technical",
                    "location",
                    "work",
                    "general",
                ],
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        query: str,
        category: str | None = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            memory = get_vector_memory()
            results = memory.search(query=query, limit=limit, category=category)
            if not results:
                return ToolResult(
                    success=True,
                    data={"memories": [], "message": "No relevant memories found"},
                )
            return ToolResult(
                success=True,
                data={
                    "memories": [
                        {
                            "fact": r["text"],
                            "category": r["category"],
                            "relevance": round(r["score"], 2),
                        }
                        for r in results
                    ],
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ForgetMemoryTool(BaseTool):
    name = "forget_memory"
    description = (
        "Delete a specific memory by ID or clear all memories in a category. "
        "Use when the user asks you to forget something."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Specific memory ID to delete",
            },
            "category": {
                "type": "string",
                "description": "Delete all memories in this category",
                "enum": [
                    "personal",
                    "preference",
                    "project",
                    "technical",
                    "location",
                    "work",
                    "general",
                ],
            },
        },
    }

    async def execute(
        self,
        memory_id: str | None = None,
        category: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not memory_id and not category:
            return ToolResult(
                success=False,
                data=None,
                error="Must provide either memory_id or category",
            )

        try:
            memory = get_vector_memory()
            if memory_id:
                memory.delete(memory_id)
                return ToolResult(
                    success=True,
                    data={"message": f"Deleted memory {memory_id}"},
                )
            else:
                count = memory.delete_by_category(category)
                return ToolResult(
                    success=True,
                    data={"message": f"Deleted {count} memories from {category}"},
                )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ListMemoryCategoresTool(BaseTool):
    name = "list_memory_categories"
    description = "Get count of memories stored in each category."
    parameters = {
        "type": "object",
        "properties": {},
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            memory = get_vector_memory()
            categories = [
                "personal",
                "preference",
                "project",
                "technical",
                "location",
                "work",
                "general",
            ]
            counts = {cat: memory.count(cat) for cat in categories}
            total = memory.count()
            return ToolResult(
                success=True,
                data={"categories": counts, "total": total},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
