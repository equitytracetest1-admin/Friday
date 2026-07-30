"""
knowledge/embeddings.py — Embedding abstraction (Step 5).

EmbeddingProvider is the interface every embedding backend implements.
LocalEmbeddingProvider wraps sentence-transformers and runs fully offline —
no API key, no network call, matching the user's choice for Friday.

A small on-disk cache (keyed by content hash + model name) avoids
re-embedding unchanged chunks on every re-index.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from knowledge.schema import content_hash

VAULT_ROOT   = Path(os.environ.get("FRIDAY_VAULT", "vault"))
EMBED_CACHE  = VAULT_ROOT / "knowledge" / "embed_cache"

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement."""

    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input string."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class _DiskCache:
    """Simple content-hash → vector cache, one JSON file per model."""

    def __init__(self, model_name: str):
        safe_name = model_name.replace("/", "_")
        EMBED_CACHE.mkdir(parents=True, exist_ok=True)
        self._path = EMBED_CACHE / f"{safe_name}.json"
        self._data: dict[str, list[float]] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def get(self, key: str) -> list[float] | None:
        return self._data.get(key)

    def set_many(self, items: dict[str, list[float]]) -> None:
        self._data.update(items)
        self._flush()

    def _flush(self) -> None:
        try:
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
        except OSError:
            pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Offline embeddings via sentence-transformers. Default model
    (all-MiniLM-L6-v2, 384-dim) is small, fast on CPU, and good enough for
    a personal-assistant-scale knowledge base.

    The model is lazy-loaded on first use so importing this module never
    triggers a download.
    """

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, cache: bool = True):
        self.name       = model_name
        self._model     = None
        self._cache     = _DiskCache(model_name) if cache else None
        # all-MiniLM-L6-v2 is 384-dim; overridden after first real load if
        # a different model is used.
        self.dimension  = 384

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers --break-system-packages"
                ) from e
            self._model    = SentenceTransformer(self.name)
            self.dimension = self._model.get_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_embed_idx: list[int] = []
        to_embed_txt: list[str] = []
        keys = [content_hash(t) for t in texts]

        if self._cache is not None:
            for i, key in enumerate(keys):
                cached = self._cache.get(key)
                if cached is not None:
                    results[i] = cached
                else:
                    to_embed_idx.append(i)
                    to_embed_txt.append(texts[i])
        else:
            to_embed_idx = list(range(len(texts)))
            to_embed_txt = list(texts)

        if to_embed_txt:
            model = self._get_model()
            vectors = model.encode(to_embed_txt, convert_to_numpy=True, show_progress_bar=False)
            new_cache_items: dict[str, list[float]] = {}
            for pos, idx in enumerate(to_embed_idx):
                vec = vectors[pos].tolist()
                results[idx] = vec
                if self._cache is not None:
                    new_cache_items[keys[idx]] = vec
            if self._cache is not None and new_cache_items:
                self._cache.set_many(new_cache_items)

        return results  # type: ignore[return-value]


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic, dependency-free embedding provider for tests and
    environments without network/model access. NOT for production use —
    it's a hashing trick, not a semantic embedding.
    """

    def __init__(self, dimension: int = 64):
        self.name      = "fake-hash-embedding"
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib as _hashlib
        out = []
        for t in texts:
            vec = [0.0] * self.dimension
            for i, tok in enumerate(t.lower().split()):
                h = int(_hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dimension] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


def get_default_provider() -> EmbeddingProvider:
    """Factory used by the rest of the pipeline — swap here to change providers."""
    return LocalEmbeddingProvider()
