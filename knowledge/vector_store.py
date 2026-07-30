"""
knowledge/vector_store.py — Vector database abstraction (Step 6).

VectorStore is the interface the rest of the pipeline codes against.
LanceDBStore is the concrete, recommended implementation: embedded,
serverless, Python-native, with first-class metadata filtering — matching
the roadmap's choice.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from knowledge.schema import Chunk

VAULT_ROOT = Path(os.environ.get("FRIDAY_VAULT", "vault"))
LANCE_DIR  = VAULT_ROOT / "knowledge" / "lancedb"

TABLE_NAME = "chunks"


class VectorStore(ABC):
    """Interface every vector database backend implements."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks + their embeddings."""
        raise NotImplementedError

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        """Delete all chunks belonging to a given source. Returns count deleted."""
        raise NotImplementedError

    @abstractmethod
    def delete_by_parent(self, parent_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        project: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Return top_k results as dicts with chunk fields + a 'score' key."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def all_sources(self) -> dict[str, str]:
        """Map source -> most recent content_hash seen, for incremental indexing."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        raise NotImplementedError


class LanceDBStore(VectorStore):
    """LanceDB-backed vector store."""

    def __init__(self, db_path: Path | None = None, table_name: str = TABLE_NAME):
        self._db_path    = db_path or LANCE_DIR
        self._table_name = table_name
        self._db         = None
        self._table      = None

    # ── lazy connection ───────────────────────────────────────────────────
    def _connect(self):
        if self._db is None:
            try:
                import lancedb
            except ImportError as e:
                raise RuntimeError(
                    "lancedb is not installed. Run: pip install lancedb --break-system-packages"
                ) from e
            self._db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self._db_path))
        return self._db

    def _get_table(self, vector_dim: int | None = None):
        if self._table is not None:
            return self._table

        db = self._connect()
        if self._table_name in db.table_names():
            self._table = db.open_table(self._table_name)
            return self._table

        if vector_dim is None:
            return None  # nothing to create against yet

        import pyarrow as pa

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("domain", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("text", pa.string()),
            pa.field("heading_path", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("created_at", pa.float64()),
            pa.field("tags", pa.string()),          # comma-joined for simplicity
            pa.field("project", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("source_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ])
        self._table = db.create_table(self._table_name, schema=schema)
        return self._table

    # ── writes ────────────────────────────────────────────────────────────
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        table = self._get_table(vector_dim=len(vectors[0]))
        assert table is not None  # vector_dim was provided, so a table always exists here

        # Remove any existing rows with the same id (update semantics)
        ids = [c.id for c in chunks]
        id_list = ", ".join(f"'{i}'" for i in ids)
        try:
            table.delete(f"id IN ({id_list})")
        except Exception:
            pass  # table may be empty / filter syntax not supported yet

        rows = []
        for chunk, vector in zip(chunks, vectors):
            row = chunk.to_dict()
            row["tags"]   = ",".join(row.get("tags") or [])
            row["project"] = row.get("project") or ""
            row["vector"] = vector
            rows.append(row)

        table.add(rows)

    def delete_by_source(self, source: str) -> int:
        table = self._get_table()
        if table is None:
            return 0
        escaped = source.replace("'", "''")
        before = table.count_rows()
        table.delete(f"source = '{escaped}'")
        return before - table.count_rows()

    def delete_by_parent(self, parent_id: str) -> int:
        table = self._get_table()
        if table is None:
            return 0
        before = table.count_rows()
        table.delete(f"parent_id = '{parent_id}'")
        return before - table.count_rows()

    # ── reads ─────────────────────────────────────────────────────────────
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        project: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        table = self._get_table()
        if table is None:
            return []

        q = table.search(query_vector).limit(top_k * 4 if (project or domain or tags) else top_k)

        filters = []
        if project:
            filters.append(f"project = '{project}'")
        if domain:
            filters.append(f"domain = '{domain}'")
        if filters:
            q = q.where(" AND ".join(filters))

        results = q.to_list()

        if tags:
            wanted = set(tags)
            results = [
                r for r in results
                if wanted.intersection(set((r.get("tags") or "").split(",")))
            ]

        results = results[:top_k]
        for r in results:
            # LanceDB returns _distance; convert to a similarity-style score
            dist = r.pop("_distance", None)
            r["score"] = None if dist is None else 1.0 / (1.0 + dist)
            r["tags"] = [t for t in (r.get("tags") or "").split(",") if t]
        return results

    def count(self) -> int:
        table = self._get_table()
        return table.count_rows() if table is not None else 0

    def all_sources(self) -> dict[str, str]:
        table = self._get_table()
        if table is None:
            return {}
        df = table.to_pandas()[["source", "source_hash"]]
        result: dict[str, str] = {}
        for _, r in df.iterrows():
            result[str(r["source"])] = str(r["source_hash"])
        return result

    def health_check(self) -> dict[str, Any]:
        try:
            table = self._get_table()
            return {
                "ok": True,
                "path": str(self._db_path),
                "table": self._table_name,
                "row_count": table.count_rows() if table is not None else 0,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}


def get_default_store() -> VectorStore:
    """Factory used by the rest of the pipeline — swap here to change backends."""
    return LanceDBStore()
