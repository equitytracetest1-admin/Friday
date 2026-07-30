"""
knowledge/pipeline.py — End-to-end RAG entry point (Step 11, 12, 13).

This is the single function `retrieve_context_for_query()` that
agent/assistant.py should call on every user turn. It hides ingestion,
embedding, vector search, and context assembly behind one call so
retrieval is invisible to the user, per the roadmap's Step 12 goal.

Usage from assistant.py (sketch):

    from knowledge.pipeline import RAGPipeline

    _rag = RAGPipeline()

    def _build_system_prompt(active_project: str | None = None) -> str:
        ...
        rag_block = _rag.retrieved_block_for_prompt(user_text, project=active_project)
        if rag_block:
            base += f"\\n\\n<retrieved_knowledge>\\n{rag_block}\\n</retrieved_knowledge>"
        ...
"""

from __future__ import annotations

from typing import Optional

from knowledge.context_builder import BuiltContext, ContextBudget, build_context
from knowledge.embeddings import EmbeddingProvider, get_default_provider
from knowledge.indexer import IndexManager
from knowledge.retriever import Retriever, RetrievalResult
from knowledge.vector_store import VectorStore, get_default_store


class RAGPipeline:
    """Wires retriever + indexer together behind one simple interface."""

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
        top_k: int = 5,
    ):
        self.embedder  = embedder or get_default_provider()
        self.store     = store or get_default_store()
        self.retriever = Retriever(store=self.store, embedder=self.embedder)
        self.indexer   = IndexManager(store=self.store, embedder=self.embedder)
        self.top_k     = top_k

    # ── Step 12 — automatic retrieval, called once per user turn ───────────
    def retrieve(
        self,
        query: str,
        project: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        return self.retriever.search(
            query, top_k=top_k or self.top_k, project=project, domain=domain,
        )

    def retrieved_block_for_prompt(
        self,
        query: str,
        project: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        results = self.retrieve(query, project=project, domain=domain, top_k=top_k)
        if not results:
            return ""
        from knowledge.context_builder import _format_retrieved
        return _format_retrieved(results)

    # ── Step 10/11 — full context assembly, if a caller wants the whole
    #    prioritized/trimmed context rather than just a text block ─────────
    def build_full_context(
        self,
        system_prompt: str,
        user_memory: str,
        project_memory: str,
        conversation: list[dict[str, str]],
        current_message: str,
        project: Optional[str] = None,
        budget: Optional[ContextBudget] = None,
    ) -> BuiltContext:
        retrieved = self.retrieve(current_message, project=project)
        return build_context(
            system_prompt=system_prompt,
            user_memory=user_memory,
            project_memory=project_memory,
            retrieved=retrieved,
            conversation=conversation,
            current_message=current_message,
            budget=budget,
        )

    # ── ingestion passthroughs ──────────────────────────────────────────────
    def index_directory(self, root: str, **kwargs) -> dict[str, int]:
        return self.indexer.index_directory(root, **kwargs)

    def index_path(self, path: str, **kwargs) -> int:
        return self.indexer.index_path(path, **kwargs)

    def stats(self) -> dict:
        return self.indexer.stats()


_default_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Process-wide singleton so embedder/model + DB connection are reused
    across turns instead of reloading per call."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline
