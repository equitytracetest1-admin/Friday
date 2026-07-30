"""
knowledge/reranker.py — Optional reranking stage (Step 9).

Pluggable and off by default. A cross-encoder reranker takes the top-N
vector search hits and rescoring them against the raw query for higher
precision, at extra latency cost. Wire in `sentence-transformers`
CrossEncoder here when you want it; retriever.py works fine without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class NoOpReranker(Reranker):
    """Default: pass results through unchanged (already sorted by vector score)."""

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        return results[:top_k]


class CrossEncoderReranker(Reranker):
    """
    Optional cross-encoder reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2).
    Lazy-loaded so importing this module never requires the model to be
    downloaded. Enable by passing this instead of NoOpReranker to Retriever.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is required for CrossEncoderReranker."
                ) from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not results:
            return results
        model = self._get_model()
        pairs = [(query, r.get("text", "")) for r in results]
        scores = model.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
        return results[:top_k]


def get_default_reranker() -> Reranker:
    """Off by default — flip to CrossEncoderReranker() to enable Step 9."""
    return NoOpReranker()
