from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentChunk:
    def __init__(
        self,
        content: str,
        source: str,
        chunk_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.source = source
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        hash_input = f"{self.source}:{self.chunk_index}:{self.content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()


class RAGEngine:
    def __init__(
        self,
        db_path: str | Path = "data/vectors",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._embedder: SentenceTransformer | None = None
        self._embedding_model = embedding_model
        self._embedding_dim: int | None = None

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model)
            self._embedding_dim = self._embedder.get_sentence_embedding_dimension()
        return self._embedder

    def _get_db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _get_table(self) -> lancedb.table.Table:
        if self._table is None:
            db = self._get_db()
            try:
                self._table = db.open_table("documents")
            except Exception:
                embedder = self._get_embedder()
                dim = embedder.get_sentence_embedding_dimension()
                schema = {
                    "id": str,
                    "content": str,
                    "source": str,
                    "chunk_index": int,
                    "metadata": str,
                    "created_at": str,
                    "vector": lancedb.vector(dim),
                }
                self._table = db.create_table("documents", schema=schema)
        return self._table

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        return embedder.encode(texts, convert_to_numpy=True)

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks if chunks else [text]

    def _chunk_code(self, content: str, file_ext: str) -> list[str]:
        lines = content.split("\n")
        chunks = []
        current_chunk: list[str] = []
        current_size = 0

        for line in lines:
            line_words = len(line.split())
            if current_size + line_words > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                overlap_lines = current_chunk[-5:] if len(current_chunk) > 5 else []
                current_chunk = overlap_lines
                current_size = sum(len(ln.split()) for ln in current_chunk)
            current_chunk.append(line)
            current_size += line_words

        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks if chunks else [content]

    def _chunk_markdown(self, content: str) -> list[str]:
        sections = []
        current_section: list[str] = []
        for line in content.split("\n"):
            if line.startswith("#") and current_section:
                sections.append("\n".join(current_section))
                current_section = []
            current_section.append(line)
        if current_section:
            sections.append("\n".join(current_section))

        chunks = []
        for section in sections:
            if len(section.split()) > self.chunk_size:
                chunks.extend(self._chunk_text(section))
            else:
                chunks.append(section)
        return chunks if chunks else [content]

    def add_document(
        self,
        content: str,
        source: str,
        file_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if file_type is None:
            ext = Path(source).suffix.lower()
            file_type = ext if ext else "text"

        if file_type in (".py", ".js", ".ts", ".rs", ".c", ".cpp", ".java", ".go"):
            chunks = self._chunk_code(content, file_type)
        elif file_type in (".md", ".markdown"):
            chunks = self._chunk_markdown(content)
        else:
            chunks = self._chunk_text(content)

        embeddings = self._embed(chunks)
        table = self._get_table()
        now = datetime.now().isoformat()
        metadata_str = str(metadata) if metadata else "{}"

        records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_chunk = DocumentChunk(chunk, source, i, metadata)
            records.append(
                {
                    "id": doc_chunk.id,
                    "content": chunk,
                    "source": source,
                    "chunk_index": i,
                    "metadata": metadata_str,
                    "created_at": now,
                    "vector": embedding.tolist(),
                }
            )

        table.add(records)
        return len(records)

    def add_file(self, file_path: str | Path, metadata: dict[str, Any] | None = None) -> int:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return self.add_document(
            content=content,
            source=str(file_path),
            file_type=file_path.suffix.lower(),
            metadata=metadata,
        )

    def add_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> int:
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        if extensions is None:
            extensions = [".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"]

        total = 0
        pattern = "**/*" if recursive else "*"
        for ext in extensions:
            for file_path in dir_path.glob(f"{pattern}{ext}"):
                if file_path.is_file():
                    try:
                        total += self.add_file(file_path)
                    except Exception:
                        continue
        return total

    def search(
        self,
        query: str,
        limit: int = 5,
        filter_source: str | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self._embed([query])[0]
        table = self._get_table()

        results = table.search(query_embedding.tolist()).limit(limit)
        if filter_source:
            results = results.where(f"source LIKE '%{filter_source}%'")

        rows = results.to_list()
        return [
            {
                "content": row["content"],
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "score": row.get("_distance", 0),
            }
            for row in rows
        ]

    def get_context(self, query: str, limit: int = 3, max_tokens: int = 2000) -> str:
        results = self.search(query, limit=limit)
        context_parts = []
        total_words = 0

        for result in results:
            content = result["content"]
            words = len(content.split())
            if total_words + words > max_tokens:
                break
            context_parts.append(f"[Source: {result['source']}]\n{content}")
            total_words += words

        return "\n\n---\n\n".join(context_parts)

    def delete_source(self, source: str) -> int:
        table = self._get_table()
        initial_count = table.count_rows()
        table.delete(f"source = '{source}'")
        return initial_count - table.count_rows()

    def list_sources(self) -> list[str]:
        table = self._get_table()
        df = table.to_pandas()
        if "source" in df.columns:
            return df["source"].unique().tolist()
        return []

    def clear(self) -> None:
        db = self._get_db()
        try:
            db.drop_table("documents")
        except Exception:
            pass
        self._table = None
