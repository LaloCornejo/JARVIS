"""Knowledge Graph System for JARVIS.

Manages entities, relationships, and automated inference for structured knowledge.
Enables complex queries and relationship-based reasoning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph"""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"
    TECHNOLOGY = "technology"
    FILE = "file"
    PROJECT = "project"
    TASK = "task"
    CUSTOM = "custom"


class RelationType(Enum):
    """Types of relationships between entities"""

    # Social/Organizational
    KNOWS = "knows"
    WORKS_FOR = "works_for"
    MEMBER_OF = "member_of"
    MANAGES = "manages"

    # Spatial
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    NEAR = "near"

    # Conceptual
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"

    # Causal/Temporal
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    PRECEDES = "precedes"
    FOLLOWS = "follows"

    # Functional
    USES = "uses"
    USED_BY = "used_by"
    DEPENDS_ON = "depends_on"
    REQUIRED_FOR = "required_for"

    # Project/Task
    CREATED = "created"
    OWNS = "owns"
    ASSIGNED_TO = "assigned_to"
    BLOCKED_BY = "blocked_by"

    CUSTOM = "custom"


@dataclass
class Entity:
    """A node in the knowledge graph"""

    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
        }


@dataclass
class Relationship:
    """An edge in the knowledge graph"""

    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    bidirectional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "bidirectional": self.bidirectional,
        }


@dataclass
class QueryResult:
    """Result of a knowledge graph query"""

    entities: List[Entity]
    relationships: List[Relationship]
    paths: List[List[str]] = field(default_factory=list)  # Entity ID paths
    inferred_facts: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "paths": self.paths,
            "inferred_facts": self.inferred_facts,
            "confidence": self.confidence,
        }


@dataclass
class InferenceRule:
    """Rule for automated inference"""

    id: str
    name: str
    description: str
    premises: List[Dict[str, Any]]  # Required relationships
    conclusion: Dict[str, Any]  # Relationship to infer
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
        }


class KnowledgeGraph:
    """Knowledge graph with entities, relationships, and inference"""

    def __init__(self, db_path: Path | str = "data/knowledge_graph.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._inference_rules: List[InferenceRule] = []
        self._load_default_rules()

    def _init_db(self) -> None:
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    properties TEXT DEFAULT '{}',
                    aliases TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT
                );

                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT DEFAULT '{}',
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bidirectional INTEGER DEFAULT 0,
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                );

                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
                CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
                CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type);
            """)

    def _load_default_rules(self) -> None:
        """Load default inference rules"""
        self._inference_rules = [
            InferenceRule(
                id="transitive_works_for",
                name="Transitive Works-For",
                description="If A works for B and B is part of C, then A works for C",
                premises=[
                    {"relation": RelationType.WORKS_FOR.value},
                    {"relation": RelationType.PART_OF.value},
                ],
                conclusion={"relation": RelationType.WORKS_FOR.value},
                confidence=0.9,
            ),
            InferenceRule(
                id="symmetric_knows",
                name="Symmetric Knows",
                description="If A knows B, then B knows A",
                premises=[{"relation": RelationType.KNOWS.value}],
                conclusion={"relation": RelationType.KNOWS.value, "bidirectional": True},
                confidence=0.95,
            ),
        ]

    async def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        properties: Dict[str, Any] | None = None,
        aliases: List[str] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> str:
        """Add an entity to the graph"""
        entity_id = f"{entity_type.value}_{name.lower().replace(' ', '_')}_{hash(name) % 10000}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO entities
                (id, name, entity_type, properties, aliases, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    entity_id,
                    name,
                    entity_type.value,
                    json.dumps(properties or {}),
                    json.dumps(aliases or []),
                    confidence,
                    source,
                ),
            )

        return entity_id

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Dict[str, Any] | None = None,
        confidence: float = 1.0,
        bidirectional: bool = False,
    ) -> str:
        """Add a relationship between entities"""
        rel_id = f"rel_{source_id}_{relation_type.value}_{target_id}_{hash(asyncio.get_event_loop().time()) % 10000}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO relationships
                (id, source_id, target_id, relation_type, properties, confidence, bidirectional)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_id,
                    source_id,
                    target_id,
                    relation_type.value,
                    json.dumps(properties or {}),
                    confidence,
                    1 if bidirectional else 0,
                ),
            )

        # If bidirectional, add reverse relationship
        if bidirectional:
            await self.add_relationship(
                target_id, source_id, relation_type, properties, confidence, False
            )

        return rel_id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()

            if row:
                return Entity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=EntityType(row["entity_type"]),
                    properties=json.loads(row["properties"]),
                    aliases=json.loads(row["aliases"]),
                    confidence=row["confidence"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    source=row["source"],
                )
        return None

    async def find_entities(
        self, name: str | None = None, entity_type: EntityType | None = None, limit: int = 20
    ) -> List[Entity]:
        """Find entities by name or type"""
        query = "SELECT * FROM entities WHERE 1=1"
        params = []

        if name:
            query += " AND (name LIKE ? OR aliases LIKE ?)"
            params.extend([f"%{name}%", f"%{name}%"])

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type.value)

        query += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

            return [
                Entity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=EntityType(row["entity_type"]),
                    properties=json.loads(row["properties"]),
                    aliases=json.loads(row["aliases"]),
                    confidence=row["confidence"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    source=row["source"],
                )
                for row in rows
            ]

    async def get_relationships(
        self,
        entity_id: str | None = None,
        relation_type: RelationType | None = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> List[Relationship]:
        """Get relationships for an entity"""
        query = "SELECT * FROM relationships WHERE 1=1"
        params = []

        if entity_id:
            if direction == "outgoing":
                query += " AND source_id = ?"
                params.append(entity_id)
            elif direction == "incoming":
                query += " AND target_id = ?"
                params.append(entity_id)
            else:  # both
                query += " AND (source_id = ? OR target_id = ?)"
                params.extend([entity_id, entity_id])

        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type.value)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

            return [
                Relationship(
                    id=row["id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relation_type=RelationType(row["relation_type"]),
                    properties=json.loads(row["properties"]),
                    confidence=row["confidence"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    bidirectional=bool(row["bidirectional"]),
                )
                for row in rows
            ]

    async def query(
        self,
        start_entity: str,
        relation_types: List[RelationType] | None = None,
        max_depth: int = 2,
        min_confidence: float = 0.5,
    ) -> QueryResult:
        """Query the knowledge graph from a starting entity"""
        visited: Set[str] = set()
        entities: Dict[str, Entity] = {}
        relationships: List[Relationship] = []
        paths: List[List[str]] = []

        async def traverse(entity_id: str, depth: int, current_path: List[str]):
            if depth > max_depth or entity_id in visited:
                return

            visited.add(entity_id)
            current_path = current_path + [entity_id]

            # Get entity
            entity = await self.get_entity(entity_id)
            if entity and entity.confidence >= min_confidence:
                entities[entity_id] = entity

            # Get relationships
            rels = await self.get_relationships(entity_id)
            for rel in rels:
                if rel.confidence >= min_confidence:
                    if relation_types is None or rel.relation_type in relation_types:
                        relationships.append(rel)

                        # Continue traversal
                        next_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                        if next_id not in visited:
                            await traverse(next_id, depth + 1, current_path)

        await traverse(start_entity, 0, [])

        # Build paths
        if start_entity in entities:
            paths = [[start_entity, e.id] for e in entities.values() if e.id != start_entity]

        return QueryResult(
            entities=list(entities.values()),
            relationships=relationships,
            paths=paths,
            confidence=1.0,
        )

    async def infer_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Infer new relationships using inference rules"""
        inferred = []

        # Get existing relationships
        relationships = await self.get_relationships(entity_id)

        for rule in self._inference_rules:
            # Check if premises match
            matching_premises = 0
            for premise in rule.premises:
                for rel in relationships:
                    if rel.relation_type.value == premise.get("relation"):
                        matching_premises += 1
                        break

            if matching_premises >= len(rule.premises):
                # Generate inference
                inferred.append(
                    {
                        "rule": rule.name,
                        "confidence": rule.confidence,
                        "conclusion": rule.conclusion,
                        "entity_id": entity_id,
                    }
                )

        return inferred

    async def find_path(
        self, start_id: str, end_id: str, max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find a path between two entities"""
        visited: Set[str] = set()

        async def bfs(
            current: str, target: str, depth: int, path: List[str]
        ) -> Optional[List[str]]:
            if depth > max_depth or current in visited:
                return None

            visited.add(current)
            path = path + [current]

            if current == target:
                return path

            # Get outgoing relationships
            relationships = await self.get_relationships(current, direction="outgoing")
            for rel in relationships:
                result = await bfs(rel.target_id, target, depth + 1, path)
                if result:
                    return result

            return None

        return await bfs(start_id, end_id, 0, [])

    async def merge_entities(self, entity_ids: List[str], new_name: str) -> Optional[str]:
        """Merge multiple entities into one"""
        if not entity_ids:
            return None

        # Get first entity to determine type
        first = await self.get_entity(entity_ids[0])
        if not first:
            return None

        # Create new merged entity
        merged_id = await self.add_entity(
            name=new_name,
            entity_type=first.entity_type,
            properties=first.properties,
            aliases=list(
                set(
                    first.aliases
                    + [e.name for e in [await self.get_entity(eid) for eid in entity_ids] if e]
                )
            ),
            confidence=sum(
                e.confidence for e in [await self.get_entity(eid) for eid in entity_ids] if e
            )
            / len(entity_ids),
        )

        # Redirect relationships
        for old_id in entity_ids:
            if old_id != merged_id:
                # Update relationships
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE relationships SET source_id = ? WHERE source_id = ?",
                        (merged_id, old_id),
                    )
                    conn.execute(
                        "UPDATE relationships SET target_id = ? WHERE target_id = ?",
                        (merged_id, old_id),
                    )
                    # Delete old entity
                    conn.execute("DELETE FROM entities WHERE id = ?", (old_id,))

        return merged_id

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        with sqlite3.connect(self.db_path) as conn:
            entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relationship_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

            entity_types = conn.execute(
                "SELECT entity_type, COUNT(*) as count FROM entities GROUP BY entity_type"
            ).fetchall()

            relation_types = conn.execute(
                "SELECT relation_type, COUNT(*) as count FROM relationships GROUP BY relation_type"
            ).fetchall()

            return {
                "entity_count": entity_count,
                "relationship_count": relationship_count,
                "entity_types": {t: c for t, c in entity_types},
                "relation_types": {t: c for t, c in relation_types},
                "inference_rules": len(self._inference_rules),
            }

    async def export_graph(self, format: str = "json") -> str:
        """Export the knowledge graph"""
        stats = await self.get_stats()

        if format == "json":
            entities = await self.find_entities(limit=stats["entity_count"])
            relationships = await self.get_relationships()

            return json.dumps(
                {
                    "entities": [e.to_dict() for e in entities],
                    "relationships": [r.to_dict() for r in relationships],
                    "stats": stats,
                },
                indent=2,
            )

        return ""

    async def close(self):
        """Close the knowledge graph connection"""
        # SQLite connections are context managers, nothing to close
        pass


# Global instance
_knowledge_graph: KnowledgeGraph | None = None


async def get_knowledge_graph(db_path: Path | str = "data/knowledge_graph.db") -> KnowledgeGraph:
    """Get or create the global knowledge graph instance"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph(db_path)
    return _knowledge_graph
