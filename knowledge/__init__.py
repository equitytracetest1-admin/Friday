"""
knowledge/ — Friday's RAG subsystem (Phase 2).

Public surface most callers need:

    from knowledge.pipeline import get_pipeline
    rag = get_pipeline()
    rag.index_directory("vault/facts")
    context_block = rag.retrieved_block_for_prompt("what did we decide about X?")
"""

from knowledge.schema import KnowledgeItem, Chunk, KNOWLEDGE_DOMAINS, DOCUMENT_TYPES
from knowledge.pipeline import RAGPipeline, get_pipeline

__all__ = [
    "KnowledgeItem",
    "Chunk",
    "KNOWLEDGE_DOMAINS",
    "DOCUMENT_TYPES",
    "RAGPipeline",
    "get_pipeline",
]
