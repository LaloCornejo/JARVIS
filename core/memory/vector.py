from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer


class VectorMemory:
    TABLE_NAME = "memories"

    def __init__(
        self,
        db_path: str | Path = "data/vectors",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self._model: SentenceTransformer | None = None
        self._model_name = embedding_model
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._embedding_dim: int | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def _get_db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _get_table(self) -> lancedb.table.Table:
        if self._table is not None:
            return self._table

        db = self._get_db()
        self._get_model()

        if self.TABLE_NAME in db.table_names():
            self._table = db.open_table(self.TABLE_NAME)
        else:
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("category", pa.string()),
                    pa.field("metadata", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self._embedding_dim)),
                ]
            )
            self._table = db.create_table(self.TABLE_NAME, schema=schema)

        return self._table

    def _embed(self, text: str) -> list[float]:
        model = self._get_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def add(
        self,
        text: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        import json
        import uuid

        table = self._get_table()
        memory_id = memory_id or str(uuid.uuid4())
        vector = self._embed(text)

        data = [
            {
                "id": memory_id,
                "text": text,
                "category": category,
                "metadata": json.dumps(metadata or {}),
                "created_at": datetime.now().isoformat(),
                "vector": vector,
            }
        ]

        table.add(data)
        return memory_id

    def add_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[str]:
        import json
        import uuid

        table = self._get_table()
        texts = [item["text"] for item in items]
        vectors = self._embed_batch(texts)

        data = []
        ids = []
        for item, vector in zip(items, vectors):
            memory_id = item.get("id") or str(uuid.uuid4())
            ids.append(memory_id)
            data.append(
                {
                    "id": memory_id,
                    "text": item["text"],
                    "category": item.get("category", "general"),
                    "metadata": json.dumps(item.get("metadata", {})),
                    "created_at": datetime.now().isoformat(),
                    "vector": vector,
                }
            )

        table.add(data)
        return ids

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        import json

        table = self._get_table()
        query_vector = self._embed(query)

        search_builder = table.search(query_vector).limit(limit)

        if category:
            search_builder = search_builder.where(f"category = '{category}'")

        results = search_builder.to_list()

        return [
            {
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
                "score": 1 - r["_distance"],
            }
            for r in results
        ]

    def get_by_id(self, memory_id: str) -> dict[str, Any] | None:
        import json

        table = self._get_table()
        results = table.search().where(f"id = '{memory_id}'").limit(1).to_list()

        if not results:
            return None

        r = results[0]
        return {
            "id": r["id"],
            "text": r["text"],
            "category": r["category"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            "created_at": r["created_at"],
        }

    def get_by_category(
        self,
        category: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        import json

        table = self._get_table()
        results = table.search().where(f"category = '{category}'").limit(limit).to_list()

        return [
            {
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
            }
            for r in results
        ]

    def delete(self, memory_id: str) -> bool:
        table = self._get_table()
        table.delete(f"id = '{memory_id}'")
        return True

    def delete_by_category(self, category: str) -> int:
        table = self._get_table()
        before_count = table.count_rows()
        table.delete(f"category = '{category}'")
        after_count = table.count_rows()
        return before_count - after_count

    def count(self, category: str | None = None) -> int:
        table = self._get_table()
        if category:
            return len(table.search().where(f"category = '{category}'").limit(100000).to_list())
        return table.count_rows()

    async def add_async(
        self,
        text: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.add(text, category, metadata, memory_id)
        )

    async def search_async(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search(query, limit, category))


_instance: VectorMemory | None = None


def get_vector_memory(
    db_path: str | Path | None = None,
    embedding_model: str | None = None,
) -> VectorMemory:
    global _instance
    if _instance is None:
        _instance = VectorMemory(
            db_path=db_path or "data/vectors",
            embedding_model=embedding_model or "all-MiniLM-L6-v2",
        )
    return _instance
