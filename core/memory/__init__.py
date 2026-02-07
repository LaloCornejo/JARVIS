"""
Memory system for JARVIS.

Provides multiple memory types:
- ConversationMemory: Chat history and facts
- VectorMemory: Semantic search and embeddings
- SemanticMemory: Enhanced memory with importance scoring
- EpisodicMemory: Temporal experiences and events
- ProceduralMemory: Skills and procedures
- KnowledgeGraph: Entity relationships and inference
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

# Original memory systems
from core.memory.episodic import (
    Episode,
    EpisodeQuery,
    EpisodeType,
    EpisodicMemory,
    TemporalSequence,
    get_episodic_memory,
)
from core.memory.extractor import FactExtractor, get_fact_extractor
from core.memory.knowledge_graph import (
    Entity,
    EntityType,
    InferenceRule,
    KnowledgeGraph,
    QueryResult,
    Relationship,
    RelationType,
    get_knowledge_graph,
)
from core.memory.procedural import (
    ExecutionResult,
    ProceduralMemory,
    Procedure,
    Skill,
    SkillLevel,
    Step,
    get_procedural_memory,
)
from core.memory.semantic_memory import (
    EnhancedMemorySystem,
    SemanticMemory,
    enhanced_memory,
    get_enhanced_memory,
)
from core.memory.vector import VectorMemory, get_vector_memory

__all__ = [
    # Original
    "ConversationMemory",
    "VectorMemory",
    "get_vector_memory",
    "FactExtractor",
    "get_fact_extractor",
    # Enhanced semantic memory
    "SemanticMemory",
    "EnhancedMemorySystem",
    "enhanced_memory",
    "get_enhanced_memory",
    # Episodic memory
    "EpisodicMemory",
    "Episode",
    "EpisodeType",
    "EpisodeQuery",
    "TemporalSequence",
    "get_episodic_memory",
    # Procedural memory
    "ProceduralMemory",
    "Procedure",
    "Step",
    "Skill",
    "SkillLevel",
    "ExecutionResult",
    "get_procedural_memory",
    # Knowledge graph
    "KnowledgeGraph",
    "Entity",
    "Relationship",
    "EntityType",
    "RelationType",
    "QueryResult",
    "InferenceRule",
    "get_knowledge_graph",
]


class ConversationMemory:
    def __init__(self, db_path: str | Path = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_facts_category
                ON facts(category);
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id);
            """)

    def create_conversation(self, session_id: str | None = None) -> int:
        session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (session_id) VALUES (?)",
                (session_id,),
            )
            return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )
            return cursor.lastrowid

    def get_messages(self, conversation_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_recent_conversations(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.session_id, c.created_at,
                       COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def store_fact(self, key: str, value: Any, category: str | None = None) -> None:
        if not isinstance(value, str):
            value = json.dumps(value)
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO facts (key, value, category, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value, category),
            )

    def get_fact(self, key: str) -> Any | None:
        with self._get_conn() as conn:
            row = conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
            if row:
                try:
                    return json.loads(row["value"])
                except json.JSONDecodeError:
                    return row["value"]
            return None

    def get_facts_by_category(self, category: str) -> dict[str, Any]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE category = ?", (category,)
            ).fetchall()
            result = {}
            for row in rows:
                try:
                    result[row["key"]] = json.loads(row["value"])
                except json.JSONDecodeError:
                    result[row["key"]] = row["value"]
            return result

    def search_facts(self, query: str) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT key, value, category, updated_at
                FROM facts
                WHERE key LIKE ? OR value LIKE ?
                ORDER BY updated_at DESC
                LIMIT 20
                """,
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_fact(self, key: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def get_conversation_context(
        self, conversation_id: int, max_messages: int = 20
    ) -> list[dict[str, str]]:
        messages = self.get_messages(conversation_id, max_messages)
        return [{"role": m["role"], "content": m["content"]} for m in messages]
