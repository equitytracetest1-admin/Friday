"""
knowledge/indexer.py — Index management (Step 7).

Keeps the vector store synchronized with knowledge sources:
  * index_item()        — chunk + embed + upsert one KnowledgeItem
  * index_path()        — load a file from disk, then index_item()
  * reindex_if_changed() — skip re-embedding when content hash is unchanged
  * remove_source()     — delete a source's chunks (file removed/renamed)
  * IndexManager        — stateful wrapper tracking source -> hash for
                           incremental runs over a whole directory tree
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from knowledge.schema import KnowledgeItem, content_hash
from knowledge.chunker import chunk_item, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from knowledge.embeddings import EmbeddingProvider, get_default_provider
from knowledge.vector_store import VectorStore, get_default_store
from knowledge.ingest.loaders import load_file, detect_doc_type


class IndexManager:
    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        self.store      = store or get_default_store()
        self.embedder   = embedder or get_default_provider()
        self.chunk_size = chunk_size
        self.overlap    = overlap

    # ── core operations ───────────────────────────────────────────────────
    def index_item(self, item: KnowledgeItem, force: bool = False) -> int:
        """Chunk, embed, and upsert a single KnowledgeItem. Returns #chunks written."""
        if not force and not self._changed(item):
            return 0

        # Replace old chunks for this source entirely (simplest correct
        # approach — avoids orphaned stale chunks from a shrunk document).
        self.store.delete_by_source(item.source)

        chunks = chunk_item(item, chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            return 0

        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert(chunks, vectors)
        return len(chunks)

    def _changed(self, item: KnowledgeItem) -> bool:
        known = self.store.all_sources()
        prior_hash = known.get(item.source)
        return prior_hash != item.hash

    def index_path(
        self,
        path: str | Path,
        domain: str = "documents",
        project: str | None = None,
        tags: list[str] | None = None,
        force: bool = False,
    ) -> int:
        item = load_file(path, domain=domain, project=project, tags=tags)
        return self.index_item(item, force=force)

    def index_directory(
        self,
        root: str | Path,
        domain: str = "documents",
        project: str | None = None,
        tags: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        """Walk a directory, indexing every supported file. Skips unchanged files
        automatically unless force=True. Returns {path: chunk_count} for files
        that were (re)indexed."""
        root_path = Path(root).expanduser().resolve()
        results: dict[str, int] = {}

        for p in sorted(root_path.rglob("*")):
            if not p.is_file():
                continue
            if detect_doc_type(p) is None:
                continue
            if any(part.startswith(".") for part in p.parts):
                continue
            try:
                n = self.index_path(p, domain=domain, project=project, tags=tags, force=force)
                if n:
                    results[str(p)] = n
            except Exception as e:
                results[str(p)] = -1
                print(f"[indexer] failed to index {p}: {e}")

        return results

    def remove_source(self, source: str) -> int:
        """Delete all chunks belonging to a source (e.g. file was deleted)."""
        return self.store.delete_by_source(source)

    def prune_missing(self, root: str | Path) -> list[str]:
        """Remove index entries for files that no longer exist on disk."""
        root_path = Path(root).expanduser().resolve()
        removed = []
        for source in list(self.store.all_sources().keys()):
            sp = Path(source)
            try:
                sp.relative_to(root_path)
            except ValueError:
                continue  # not under this root, leave it alone
            if not sp.exists():
                self.remove_source(source)
                removed.append(source)
        return removed

    def stats(self) -> dict:
        return {
            "total_chunks": self.store.count(),
            "total_sources": len(self.store.all_sources()),
            "embedding_model": self.embedder.name,
        }


def get_default_indexer() -> IndexManager:
    return IndexManager()
