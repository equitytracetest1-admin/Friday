"""
test_rag_pipeline.py — Smoke test for knowledge/ (Phase 2 RAG).

Uses FakeEmbeddingProvider so this runs anywhere with no model download
(sentence-transformers' real model requires a huggingface.co fetch that
isn't available in every sandbox). Swap in LocalEmbeddingProvider for a
real end-to-end run once sentence-transformers has downloaded its model.

Run:
    python test_rag_pipeline.py
"""

import os
import shutil
import tempfile
from pathlib import Path

vault_dir = Path(tempfile.mkdtemp(prefix="friday_rag_test_"))
os.environ["FRIDAY_VAULT"] = str(vault_dir)

from knowledge.schema import KnowledgeItem
from knowledge.chunker import chunk_item
from knowledge.embeddings import FakeEmbeddingProvider
from knowledge.vector_store import LanceDBStore
from knowledge.indexer import IndexManager
from knowledge.retriever import Retriever
from knowledge.context_builder import build_context, ContextBudget
from knowledge.ingest.loaders import load_text
from knowledge.project_detect import detect_active_project

PASS, FAIL = 0, 0


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def main():
    global PASS, FAIL
    embedder = FakeEmbeddingProvider(dimension=32)
    store    = LanceDBStore()
    indexer  = IndexManager(store=store, embedder=embedder)
    retriever = Retriever(store=store, embedder=embedder)

    # ── Step 2/1: ingestion + knowledge item ────────────────────────────────
    doc_text = (
        "# Friday Assistant\n\n"
        "Friday is a voice assistant built with Groq and Kokoro TTS.\n\n"
        "## Memory System\n\n"
        "Friday stores facts, preferences, projects, and logs in markdown files. "
        "The RAG system adds semantic search on top of that.\n\n"
        "## Skills\n\n"
        "Friday supports run_command, read_file, create_file, edit_file, "
        "search_folder, fetch_url, and web_search skills."
    )
    item = load_text(doc_text, source="test_doc.md", domain="documents", doc_type="markdown")
    check("load_text produces a KnowledgeItem", isinstance(item, KnowledgeItem))

    # ── Step 4: chunking ─────────────────────────────────────────────────────
    chunks = chunk_item(item, chunk_size=40, overlap=10)
    check("chunking splits by heading", len(chunks) >= 2, f"got {len(chunks)} chunks")
    check("chunk carries heading_path", any(c.heading_path for c in chunks),
          str([c.heading_path for c in chunks]))

    # ── Step 5/6/7: embed + store + index ───────────────────────────────────
    n_written = indexer.index_item(item)
    check("indexer writes chunks", n_written == len(chunks), f"wrote {n_written}")
    check("store.count reflects written chunks", store.count() == len(chunks),
          f"count={store.count()}")

    # Re-indexing unchanged content should be a no-op (Step 7 incremental).
    n_second = indexer.index_item(item)
    check("unchanged re-index is skipped", n_second == 0, f"got {n_second}")

    # Changing content should force a re-write.
    item2 = load_text(doc_text + "\n\n## New Section\n\nSomething new.",
                       source="test_doc.md", domain="documents", doc_type="markdown")
    n_third = indexer.index_item(item2)
    check("changed content triggers re-index", n_third > 0, f"got {n_third}")

    # ── project-aware indexing ──────────────────────────────────────────────
    proj_item = load_text(
        "Equity Trace is a stock-analysis project using Python and pandas.",
        source="equity_trace_notes.md", domain="projects", project="Equity Trace",
    )
    indexer.index_item(proj_item)

    # ── Step 8: semantic retrieval ───────────────────────────────────────────
    results = retriever.search("what skills does Friday support", top_k=3)
    check("retrieval returns results", len(results) > 0)
    check("retrieval finds the skills chunk",
          any("run_command" in r.text for r in results),
          str([r.text[:60] for r in results]))

    # ── Step 13: project-aware filtering ────────────────────────────────────
    proj_results = retriever.search("stock analysis project", project="Equity Trace", top_k=3)
    check("project filter returns project content",
          any("Equity Trace" in r.text or (r.project == "Equity Trace") for r in proj_results),
          str([(r.project, r.text[:40]) for r in proj_results]))

    detected = detect_active_project("tell me about Equity Trace", projects=["Equity Trace", "Friday"])
    check("project detection matches mentioned project", detected == "Equity Trace", detected)

    detected_none = detect_active_project("what's the weather", projects=["Equity Trace", "Friday"])
    check("project detection falls back to None/global", detected_none is None, detected_none)

    # ── Step 10/11: context builder respects budget + priority ─────────────
    built = build_context(
        system_prompt="You are Friday.",
        user_memory="User prefers concise answers.",
        project_memory="",
        retrieved=results,
        conversation=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        current_message="what skills does Friday support",
        budget=ContextBudget(max_tokens=200, reserve_for_reply=20),
    )
    check("built context stays near budget", built.total_tokens_est <= 220,
          f"got {built.total_tokens_est}")
    check("built context includes retrieved block", len(built.retrieved_block) > 0)

    # ── health check ─────────────────────────────────────────────────────────
    health = store.health_check()
    check("health check reports ok", health.get("ok") is True, health)

    print(f"\n{PASS} passed, {FAIL} failed")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(vault_dir, ignore_errors=True)
