from __future__ import annotations

from typing import Any

from core.memory.knowledge_graph import (
    EntityType,
    KnowledgeGraph,
    RelationType,
    get_knowledge_graph,
)
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


class AddEntityTool(BaseTool):
    name = "add_knowledge_entity"
    description = "Add an entity to the knowledge graph. Use when the user mentions a person, organization, location, concept, or thing they want to remember relationships about."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the entity"},
            "entity_type": {
                "type": "string",
                "description": "Type of entity",
                "enum": [e.value for e in EntityType],
            },
            "properties": {
                "type": "object",
                "description": "Additional properties as key-value pairs",
            },
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Alternative names",
            },
        },
        "required": ["name", "entity_type"],
    }

    async def execute(
        self,
        name: str,
        entity_type: str,
        properties: dict | None = None,
        aliases: list | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            kg = get_knowledge_graph()
            entity_id = await kg.add_entity(
                name=name,
                entity_type=EntityType(entity_type),
                properties=properties or {},
                aliases=aliases or [],
            )
            return ToolResult(
                success=True,
                data={"entity_id": entity_id, "name": name, "type": entity_type},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class AddRelationshipTool(BaseTool):
    name = "add_knowledge_relationship"
    description = "Add a relationship between two entities in the knowledge graph."
    parameters = {
        "type": "object",
        "properties": {
            "source_name": {"type": "string", "description": "Name of source entity"},
            "target_name": {"type": "string", "description": "Name of target entity"},
            "relation_type": {
                "type": "string",
                "description": "Type of relationship",
                "enum": [e.value for e in RelationType],
            },
            "properties": {"type": "object", "description": "Additional properties"},
            "bidirectional": {
                "type": "boolean",
                "description": "Is relationship bidirectional?",
                "default": False,
            },
        },
        "required": ["source_name", "target_name", "relation_type"],
    }

    async def execute(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        properties: dict | None = None,
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            kg = get_knowledge_graph()
            source_entities = await kg.find_entities(name=source_name)
            target_entities = await kg.find_entities(name=target_name)

            if not source_entities:
                return ToolResult(
                    success=False, data=None, error=f"Entity '{source_name}' not found"
                )
            if not target_entities:
                return ToolResult(
                    success=False, data=None, error=f"Entity '{target_name}' not found"
                )

            rel_id = await kg.add_relationship(
                source_id=source_entities[0].id,
                target_id=target_entities[0].id,
                relation_type=RelationType(relation_type),
                properties=properties or {},
                bidirectional=bidirectional,
            )
            return ToolResult(
                success=True,
                data={
                    "relationship_id": rel_id,
                    "from": source_name,
                    "to": target_name,
                    "type": relation_type,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class QueryKnowledgeGraphTool(BaseTool):
    name = "query_knowledge_graph"
    description = "Query the knowledge graph to find entities and their relationships. Use when user asks about connections between things or wants to explore related entities."
    parameters = {
        "type": "object",
        "properties": {
            "entity_name": {"type": "string", "description": "Name of entity to query"},
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth to traverse",
                "default": 2,
            },
            "relation_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by relation types",
            },
        },
        "required": ["entity_name"],
    }

    async def execute(
        self,
        entity_name: str,
        max_depth: int = 2,
        relation_types: list | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            kg = get_knowledge_graph()
            entities = await kg.find_entities(name=entity_name)

            if not entities:
                return ToolResult(
                    success=True,
                    data={
                        "entities": [],
                        "relationships": [],
                        "message": f"No entity found matching '{entity_name}'",
                    },
                )

            result = await kg.query(
                start_entity=entities[0].id,
                max_depth=max_depth,
            )

            return ToolResult(
                success=True,
                data={
                    "entities": [
                        {"name": e.name, "type": e.entity_type.value} for e in result.entities
                    ],
                    "relationships": [
                        {"from": r.source_id, "to": r.target_id, "type": r.relation_type.value}
                        for r in result.relationships
                    ],
                    "message": f"Found {len(result.entities)} entities and {len(result.relationships)} relationships",
                },
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class FindEntityPathTool(BaseTool):
    name = "find_entity_path"
    description = "Find how two entities are connected in the knowledge graph."
    parameters = {
        "type": "object",
        "properties": {
            "from_entity": {"type": "string", "description": "Starting entity name"},
            "to_entity": {"type": "string", "description": "Target entity name"},
            "max_depth": {"type": "integer", "description": "Maximum path length", "default": 5},
        },
        "required": ["from_entity", "to_entity"],
    }

    async def execute(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            kg = get_knowledge_graph()
            from_entities = await kg.find_entities(name=from_entity)
            to_entities = await kg.find_entities(name=to_entity)

            if not from_entities:
                return ToolResult(
                    success=False, data=None, error=f"Entity '{from_entity}' not found"
                )
            if not to_entities:
                return ToolResult(success=False, data=None, error=f"Entity '{to_entity}' not found")

            path = await kg.find_path(from_entities[0].id, to_entities[0].id, max_depth)

            if path:
                path_entities = []
                for entity_id in path:
                    entity = await kg.get_entity(entity_id)
                    if entity:
                        path_entities.append(entity.name)

                return ToolResult(
                    success=True,
                    data={
                        "path": path_entities,
                        "steps": len(path) - 1,
                        "message": f"Found path with {len(path) - 1} hops",
                    },
                )
            else:
                return ToolResult(
                    success=True,
                    data={
                        "path": None,
                        "message": f"No path found between '{from_entity}' and '{to_entity}'",
                    },
                )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetKnowledgeStatsTool(BaseTool):
    name = "get_knowledge_stats"
    description = "Get statistics about the knowledge graph."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            kg = get_knowledge_graph()
            stats = await kg.get_stats()
            return ToolResult(success=True, data=stats)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
