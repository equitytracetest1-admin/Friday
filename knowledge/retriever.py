"""
knowledge/retriever.py — Semantic retrieval (Step 8) + project-aware search (Step 13).

Query -> embed -> vector search -> (optional rerank) -> top-K results.

Project-aware behavior: if a project is given, search that project first;
only fall back to / blend in global (project=None) results if the project
alone doesn't return enough hits. This matches the roadmap's "active
project first, then global knowledge" retrieval order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from knowledge.embeddings import EmbeddingProvider, get_default_provider
from knowledge.vector_store import VectorStore, get_default_store
from knowledge.reranker import Reranker, get_default_reranker

_QUERY_CACHE_TTL = 30.0  # seconds — short-lived, just to dedupe rapid repeat queries
_TYPE_WEIGHTS = {"fact": 1.15, "project": 1.05, "document": 1.0, "conversation": 0.8}

def _infer_source_type(source: str) -> str:
    s = source.lower().replace("\\", "/")
    if "/facts/projects/" in s:
        return "project"
    if "/facts/" in s:
        return "fact"
    if "/conversations/" in s:
        return "conversation"
    return "document"

@dataclass
class RetrievalResult:
    text: str
    source: str
    domain: str
    heading_path: str
    score: float
    project: Optional[str]
    tags: list[str]
    chunk_id: str
    parent_id: str
    source_type: str = "document"


class Retriever:
    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.store    = store or get_default_store()
        self.embedder = embedder or get_default_provider()
        self.reranker = reranker or get_default_reranker()
        self._cache: dict[tuple, tuple[float, list[RetrievalResult]]] = {}

    def _cache_key(self, query: str, top_k: int, project, domain, tags) -> tuple:
        return (query, top_k, project, domain, tuple(sorted(tags or [])))

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[list[str]] = None,
        rerank: bool = False,
        fetch_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []

        key = self._cache_key(query, top_k, project, domain, tags)
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < _QUERY_CACHE_TTL:
            return cached[1]

        vector = self.embedder.embed_one(query)
        pool_k = fetch_k or (top_k * 4 if rerank else top_k)

        raw = self.store.search(vector, top_k=pool_k, project=project, domain=domain, tags=tags)

        for r in raw:
            st = _infer_source_type(r.get("source", ""))
            r["score"] = (r.get("score") or 0.0) * _TYPE_WEIGHTS.get(st, 1.0)
            
        for r in raw:
            st = _infer_source_type(r.get("source", ""))
            r["score"] = (r.get("score") or 0.0) * _TYPE_WEIGHTS.get(st, 1.0)

        raw.sort(key=lambda r: r.get("score") or 0.0, reverse=True)

        # Step 13 — project-aware fallback: if the active project doesn't
        # have enough hits on its own, top up with global knowledge.
        if project and len(raw) < top_k:
            extra = self.store.search(
                vector, top_k=pool_k, project=None, domain=domain, tags=tags,
            )
            seen_ids = {r["id"] for r in raw}
            for r in extra:
                if r["id"] not in seen_ids and (r.get("project") in (None, "")):
                    raw.append(r)
                    seen_ids.add(r["id"])

        if rerank:
            raw = self.reranker.rerank(query, raw, top_k)
        else:
            raw = raw[:top_k]

        results = [
            RetrievalResult(
                text=r.get("text", ""),
                source=r.get("source", ""),
                domain=r.get("domain", ""),
                heading_path=r.get("heading_path", ""),
                score=r.get("rerank_score", r.get("score", 0.0)) or 0.0,
                project=r.get("project") or None,
                tags=r.get("tags", []),
                chunk_id=r.get("id", ""),
                parent_id=r.get("parent_id", ""),
                source_type=_infer_source_type(r.get("source", "")),
            )
            for r in raw
        ]

        self._cache[key] = (time.time(), results)
        return results


def get_default_retriever() -> Retriever:
    return Retriever()
