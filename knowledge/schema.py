"""
knowledge/schema.py — Core data model for Friday's knowledge base.

Every piece of ingested knowledge (a document, a conversation session, a
project note, ...) is represented as a KnowledgeItem. Once chunked, each
piece becomes one or more Chunk objects that actually get embedded and
stored in the vector database.

Keeping this schema in one place means every other module (loaders,
chunker, vector_store, retriever) speaks the same language.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ── Fixed knowledge domains (mirrors the roadmap's knowledge/ tree) ──────────
KNOWLEDGE_DOMAINS = (
    "conversations",
    "user",
    "projects",
    "documents",
    "notes",
    "manuals",
    "code",
    "web_cache",
)

# ── Supported document types (Step 2) ────────────────────────────────────────
DOCUMENT_TYPES = (
    "markdown", "text", "pdf", "docx", "html",
    "json", "csv", "python", "yaml",
)


def new_id() -> str:
    return uuid.uuid4().hex


def content_hash(text: str) -> str:
    """Stable hash used for dedup + incremental re-indexing (Step 7)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class KnowledgeItem:
    """A single ingested unit of knowledge, before chunking."""

    id: str
    source: str                      # file path, session file, URL, etc.
    domain: str                      # one of KNOWLEDGE_DOMAINS
    doc_type: str                    # one of DOCUMENT_TYPES
    text: str                        # cleaned, extracted plain text
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    project: Optional[str] = None
    author: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def new(
        cls,
        source: str,
        domain: str,
        doc_type: str,
        text: str,
        tags: list[str] | None = None,
        project: str | None = None,
        author: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "KnowledgeItem":
        return cls(
            id=new_id(),
            source=source,
            domain=domain,
            doc_type=doc_type,
            text=text,
            tags=tags or [],
            project=project,
            author=author,
            extra=extra or {},
        )


@dataclass
class Chunk:
    """A retrievable slice of a KnowledgeItem, ready for embedding."""

    id: str
    parent_id: str                   # KnowledgeItem.id
    source: str
    domain: str
    doc_type: str
    text: str
    heading_path: str = ""           # e.g. "Setup > Installation"
    chunk_index: int = 0
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    project: Optional[str] = None
    content_hash: str = ""           # hash of THIS chunk's text
    source_hash: str = ""            # hash of the PARENT item's full text —
                                      # used for incremental re-indexing so we
                                      # compare like-for-like (item vs item)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def new(
        cls,
        parent: KnowledgeItem,
        text: str,
        chunk_index: int,
        heading_path: str = "",
    ) -> "Chunk":
        return cls(
            id=new_id(),
            parent_id=parent.id,
            source=parent.source,
            domain=parent.domain,
            doc_type=parent.doc_type,
            text=text,
            heading_path=heading_path,
            chunk_index=chunk_index,
            tags=list(parent.tags),
            project=parent.project,
            source_hash=parent.hash,
        )
